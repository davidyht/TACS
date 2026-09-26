"""
CIFAR-10 noisy-pool selection — FULL ResNet-18 fine-tuning variant.

Companion to run_cifar10_noisy.py (which uses cached layer3 features and
trains layer4+fc only). Here every parameter of ResNet-18 (~11.7M) is
trainable in warmup, scoring and retrain. Tests the hypothesis that TACS
shines more when the trainable subspace is genuinely large.

Methods: Random, EmbedRetrieval, EL2N, GraNd, LESS, ToV, TACS

Approximation notes:
  * GraNd uses the analytic fc-layer grad-norm surrogate (‖softmax−onehot‖₂ · ‖feats‖₂).
  * LESS dot product is computed on the fc layer only (per-sample analytic).
  * The retrain and the body of TACS / ToV warmup train all 11.7M params.
The selection-side approximations are standard and keep cost tractable; what
this script actually changes is the *retrain* (and TACS / ToV warmup) regime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# data — preload CIFAR-10 into memory as (N,3,H,W) tensors, no transforms reapplied per epoch
# --------------------------------------------------------------------------- #

def _load_cifar10(data_root: str, image_size: int):
    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)

    def _to_tensor(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))])
        return xs, ys

    print(f"[data] loading CIFAR-10 at {image_size}x{image_size} into memory ...", flush=True)
    t0 = time.time()
    train_x, train_y = _to_tensor(train)
    test_x, test_y = _to_tensor(test)
    print(f"[data] done ({time.time()-t0:.1f}s) train={tuple(train_x.shape)} test={tuple(test_x.shape)}", flush=True)
    return train_x, train_y, test_x, test_y


# --------------------------------------------------------------------------- #
# model: full ResNet-18, ImageNet-pretrained, fc replaced for 10 classes
# --------------------------------------------------------------------------- #

def _build_full_model(num_classes: int = 10) -> nn.Module:
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    for p in m.parameters():
        p.requires_grad = True
    return m.to(DEVICE)


def _train_epochs(model, loader, lr, n_epochs):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    for _ in range(n_epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()


@torch.no_grad()
def _per_sample_loss(model, x_all, y_all, batch=256):
    crit = nn.CrossEntropyLoss(reduction="none")
    out = []
    model.eval()
    for i in range(0, len(x_all), batch):
        x = x_all[i:i + batch].to(DEVICE)
        y = y_all[i:i + batch].to(DEVICE)
        out.append(crit(model(x), y).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def _eval_acc_binary(model, test_x, test_y, target_classes, batch=256):
    mask = torch.isin(test_y, torch.tensor(target_classes))
    x, y = test_x[mask], test_y[mask]
    pred = []
    model.eval()
    for i in range(0, len(x), batch):
        pred.append(model(x[i:i + batch].to(DEVICE)).argmax(1).cpu())
    pred = torch.cat(pred)
    return (pred == y).float().mean().item() * 100


@torch.no_grad()
def _embed_avgpool(model, x_all, batch=256):
    """Penultimate features (avg-pooled) — used for embedding retrieval and analytic grad surrogates."""
    body = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                         model.layer1, model.layer2, model.layer3, model.layer4,
                         model.avgpool, nn.Flatten())
    out = []
    body.eval()
    for i in range(0, len(x_all), batch):
        out.append(body(x_all[i:i + batch].to(DEVICE)).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# selection methods
# --------------------------------------------------------------------------- #

def sel_random(pool_x, pool_y, val_x, val_y, k, rng):
    return rng.choice(len(pool_x), size=k, replace=False)


def sel_embed(pool_x, pool_y, val_x, val_y, k, rng):
    """Cosine retrieval from val centroid using ImageNet-pretrained ResNet-18 features."""
    model = _build_full_model().eval()
    pool_e = F.normalize(_embed_avgpool(model, pool_x), dim=1)
    val_e = F.normalize(_embed_avgpool(model, val_x).mean(0, keepdim=True), dim=1)
    scores = (pool_e @ val_e.T).squeeze(1).numpy()
    return np.argsort(scores)[-k:]


def _warmup_full(pool_x, pool_y, sub=2000, lr=1e-4, n_epochs=2, rng=None):
    sub_idx = rng.choice(len(pool_x), size=min(sub, len(pool_x)), replace=False)
    loader = DataLoader(TensorDataset(pool_x[sub_idx], pool_y[sub_idx]), batch_size=64, shuffle=True)
    model = _build_full_model()
    _train_epochs(model, loader, lr=lr, n_epochs=n_epochs)
    return model


def sel_el2n(pool_x, pool_y, val_x, val_y, k, rng):
    model = _warmup_full(pool_x, pool_y, n_epochs=2, rng=rng).eval()
    bs = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(pool_x), bs):
            x = pool_x[i:i + bs].to(DEVICE)
            y = pool_y[i:i + bs].to(DEVICE)
            p = F.softmax(model(x), dim=1)
            yh = F.one_hot(y, num_classes=10).float()
            scores.append(torch.norm(p - yh, dim=1).cpu())
    s = torch.cat(scores).numpy()
    return np.argsort(s)[-k:]


def sel_grand(pool_x, pool_y, val_x, val_y, k, rng):
    model = _warmup_full(pool_x, pool_y, n_epochs=1, rng=rng).eval()
    feats = _embed_avgpool(model, pool_x)  # CPU
    bs = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(pool_x), bs):
            f = feats[i:i + bs].to(DEVICE)
            y = pool_y[i:i + bs].to(DEVICE)
            logits = model.fc(f)
            p = F.softmax(logits, dim=1)
            yh = F.one_hot(y, num_classes=10).float()
            err = p - yh
            scores.append((err.norm(dim=1) * f.norm(dim=1)).cpu())
    s = torch.cat(scores).numpy()
    return np.argsort(s)[-k:]


def sel_less(pool_x, pool_y, val_x, val_y, k, rng):
    """LESS with fc-only per-sample grad surrogate, averaged over 3 checkpoints."""
    sub_idx = rng.choice(len(pool_x), size=min(2000, len(pool_x)), replace=False)
    loader = DataLoader(TensorDataset(pool_x[sub_idx], pool_y[sub_idx]), batch_size=64, shuffle=True)
    model = _build_full_model()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    accum = np.zeros(len(pool_x))
    for _ in range(3):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

        # val grad on the fc layer only
        model.zero_grad()
        feats_val = _embed_avgpool(model, val_x).to(DEVICE)
        logits_v = model.fc(feats_val)
        loss_v = crit(logits_v, val_y.to(DEVICE))
        loss_v.backward()
        gv_w = model.fc.weight.grad.detach().clone()
        gv_b = model.fc.bias.grad.detach().clone()

        # per-sample fc-grad . gv
        feats_pool = _embed_avgpool(model, pool_x)
        bs = 256
        chunk_scores = []
        with torch.no_grad():
            for i in range(0, len(pool_x), bs):
                f = feats_pool[i:i + bs].to(DEVICE)
                y = pool_y[i:i + bs].to(DEVICE)
                logits = model.fc(f)
                p = F.softmax(logits, dim=1)
                yh = F.one_hot(y, num_classes=10).float()
                err = p - yh
                t = (gv_w @ f.T).T  # [B, C]
                dot_w = (err * t).sum(1)
                dot_b = (err * gv_b.unsqueeze(0)).sum(1)
                chunk_scores.append((dot_w + dot_b).cpu())
        accum += torch.cat(chunk_scores).numpy()
    s = accum / 3
    return np.argsort(s)[-k:]


def sel_tov(pool_x, pool_y, val_x, val_y, k, rng):
    base_idx = rng.choice(len(pool_x), size=min(1024, len(pool_x)), replace=False)
    base_loader = DataLoader(TensorDataset(pool_x[base_idx], pool_y[base_idx]), batch_size=64, shuffle=True)
    model = _build_full_model()
    _train_epochs(model, base_loader, lr=1e-4, n_epochs=2)

    loss_before = _per_sample_loss(model, pool_x, pool_y)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=True)
    _train_epochs(model, val_loader, lr=1e-5, n_epochs=2)
    loss_after = _per_sample_loss(model, pool_x, pool_y)

    s = loss_before - loss_after
    out_mask = np.ones(len(pool_x), dtype=bool); out_mask[base_idx] = False
    s[~out_mask] = -np.inf
    return np.argsort(s)[-k:]


def sel_tacs(pool_x, pool_y, val_x, val_y, k, rng):
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=True)
    model = _build_full_model()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    crit = nn.CrossEntropyLoss()

    model.train()
    for x, y in val_loader:
        opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    loss_1 = _per_sample_loss(model, pool_x, pool_y)

    for _ in range(3):
        model.train()
        for x, y in val_loader:
            opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    loss_T = _per_sample_loss(model, pool_x, pool_y)

    s = (loss_1 - loss_T) / np.maximum(loss_1, 1e-6)
    return np.argsort(s)[-k:]


METHODS: dict[str, Callable] = {
    "Random": sel_random,
    "EmbedRetrieval": sel_embed,
    "EL2N": sel_el2n,
    "GraNd": sel_grand,
    "LESS": sel_less,
    "ToV": sel_tov,
    "TACS": sel_tacs,
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./cifar_data")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 250, 500, 1000])
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument("--pool-size", type=int, default=10000)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--target-classes", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--retrain-epochs", type=int, default=8)
    ap.add_argument("--retrain-lr", type=float, default=1e-4)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"[device] {DEVICE}", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_x, train_y_clean, test_x, test_y = _load_cifar10(args.data_root, args.image_size)
    targets_clean_np = train_y_clean.numpy()
    target_classes = args.target_classes

    runs = []
    t0 = time.time()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        val_idx = []
        for c in target_classes:
            idx_c = np.where(targets_clean_np == c)[0]
            val_idx.extend(rng.choice(idx_c, args.val_size // len(target_classes), replace=False))
        val_idx = np.array(val_idx)
        val_x = train_x[val_idx]; val_y = train_y_clean[val_idx]

        avail = np.setdiff1d(np.arange(len(train_x)), val_idx)
        pool_idx = rng.choice(avail, args.pool_size, replace=False)
        pool_x = train_x[pool_idx]
        pool_y_clean = train_y_clean[pool_idx].clone()
        noise_mask = rng.random(args.pool_size) < args.noise_rate
        pool_y_noisy = pool_y_clean.clone()
        pool_y_noisy[noise_mask] = torch.from_numpy(rng.integers(0, 10, size=int(noise_mask.sum())))
        is_clean = ~noise_mask
        is_target = np.isin(pool_y_clean.numpy(), target_classes)

        for method_name in args.methods:
            if method_name not in METHODS:
                print(f"[warn] unknown method {method_name}"); continue
            t_m = time.time()
            print(f"\n[seed={seed}] === {method_name} ===", flush=True)
            try:
                for k in args.budgets:
                    sel_rng = np.random.default_rng(seed * 1000 + k)
                    sel = METHODS[method_name](pool_x, pool_y_noisy, val_x, val_y, k, sel_rng)
                    sel = np.asarray(sel)
                    head = _build_full_model()
                    rt_loader = DataLoader(TensorDataset(pool_x[sel], pool_y_noisy[sel]),
                                           batch_size=64, shuffle=True)
                    _train_epochs(head, rt_loader, lr=args.retrain_lr, n_epochs=args.retrain_epochs)
                    acc = _eval_acc_binary(head, test_x, test_y, target_classes)
                    runs.append({
                        "seed": seed, "method": method_name, "k": k,
                        "test_acc_binary": acc,
                        "selected_clean_fraction": float(is_clean[sel].mean()),
                        "selected_target_fraction": float(is_target[sel].mean()),
                    })
                    print(f"  k={k:>5} acc={acc:5.2f} clean={is_clean[sel].mean():.3f} "
                          f"target={is_target[sel].mean():.3f}", flush=True)
            except Exception as e:
                print(f"  [error] {method_name}: {e}", flush=True)
                runs.append({"seed": seed, "method": method_name, "error": str(e)})
            print(f"  ({time.time()-t_m:.1f}s)", flush=True)

    summary = {}
    for m in args.methods:
        per_k = {}
        for k in args.budgets:
            xs = [r for r in runs if r.get("method") == m and r.get("k") == k]
            if not xs: continue
            accs = [r["test_acc_binary"] for r in xs]
            cleans = [r["selected_clean_fraction"] for r in xs]
            tgts = [r["selected_target_fraction"] for r in xs]
            per_k[str(k)] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "clean_frac_mean": float(np.mean(cleans)),
                "target_frac_mean": float(np.mean(tgts)),
                "n_seeds": len(accs),
            }
        summary[m] = {"k": per_k}

    out = {"config": vars(args), "runs": runs, "summary": summary,
           "runtime_seconds": time.time() - t0, "scope": "full_resnet18"}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
