"""
CIFAR-10 noisy-pool selection — camera-ready CV experiment for TACS.

Story: a label-noisy pool of 10 CIFAR-10 classes; a small clean validation set on
two target classes (cats / dogs); each method picks a budget-k subset; we retrain
on the subset and evaluate binary accuracy on the target classes.

Methods compared:
  Random, EmbedRetrieval, EL2N, GraNd, LESS, ToV, TACS

Speed unlock: ResNet-18 weights up to layer3 are frozen and their features are
cached once per (seed,noise) split. All warmup / scoring / retraining only runs
the layer4 + avgpool + fc head on the cached feature tensors — ~50x faster than
running the full ResNet on raw images per epoch.

Outputs a single results.json that the companion plot.py turns into figures.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def _build_loaders(data_root: str, image_size: int = 96):
    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    return train, test


def _cache_layer3_features(train, test, cache_path: Path, batch: int = 128):
    """Run frozen ResNet-18 (up to layer3) once and cache (N, 256, 14, 14) features."""
    if cache_path.exists():
        blob = np.load(cache_path)
        return (torch.from_numpy(blob["train_feats"]),
                torch.from_numpy(blob["train_labels_clean"]),
                torch.from_numpy(blob["test_feats"]),
                torch.from_numpy(blob["test_labels"]))

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to(DEVICE).eval()
    trunk = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                          model.layer1, model.layer2, model.layer3)

    @torch.no_grad()
    def _run(ds):
        feats, labels = [], []
        for i in range(0, len(ds), batch):
            xs = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(DEVICE)
            ys = torch.tensor([ds[j][1] for j in range(i, min(i + batch, len(ds)))])
            feats.append(trunk(xs).cpu())
            labels.append(ys)
        return torch.cat(feats), torch.cat(labels)

    print(f"[cache] computing layer3 features → {cache_path}")
    tf, tl = _run(train)
    sf, sl = _run(test)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path,
                        train_feats=tf.numpy(), train_labels_clean=tl.numpy(),
                        test_feats=sf.numpy(), test_labels=sl.numpy())
    return tf, tl, sf, sl


# --------------------------------------------------------------------------- #
# model: layer4 + avgpool + fc on cached features
# --------------------------------------------------------------------------- #

def _build_head(num_classes: int = 10) -> nn.Module:
    """Layer4 + avgpool + fc, initialized from ImageNet pretrained ResNet-18."""
    src = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    head = nn.Sequential()
    head.add_module("layer4", src.layer4)
    head.add_module("avgpool", src.avgpool)
    head.add_module("flatten", nn.Flatten())
    head.add_module("fc", nn.Linear(src.fc.in_features, num_classes))
    return head.to(DEVICE)


def _train_steps(model, loader, lr, n_steps, criterion=None):
    crit = criterion or nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    it = iter(loader)
    for _ in range(n_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        crit(model(x), y).mean().backward()
        opt.step()


def _train_epochs(model, loader, lr, n_epochs, criterion=None):
    crit = criterion or nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).mean().backward()
            opt.step()


@torch.no_grad()
def _per_sample_loss(model, feats, labels, batch=512):
    crit = nn.CrossEntropyLoss(reduction="none")
    out = []
    model.eval()
    for i in range(0, len(feats), batch):
        x = feats[i:i + batch].to(DEVICE)
        y = labels[i:i + batch].to(DEVICE)
        out.append(crit(model(x), y).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def _eval_acc_binary(model, test_feats, test_labels, target_classes):
    mask = torch.isin(test_labels, torch.tensor(target_classes))
    feats, labs = test_feats[mask], test_labels[mask]
    bs = 256
    pred = []
    model.eval()
    for i in range(0, len(feats), bs):
        logits = model(feats[i:i + bs].to(DEVICE))
        pred.append(logits.argmax(1).cpu())
    pred = torch.cat(pred)
    return (pred == labs).float().mean().item() * 100


# --------------------------------------------------------------------------- #
# selection methods — all take (pool_feats, pool_labels_noisy, val_feats,
# val_labels, budget) and return np.ndarray of pool-indices
# --------------------------------------------------------------------------- #

def sel_random(pool_f, pool_y, val_f, val_y, k, rng):
    return rng.choice(len(pool_f), size=k, replace=False)


def sel_embed(pool_f, pool_y, val_f, val_y, k, rng):
    """Cosine retrieval from val centroid in the cached feature space."""
    pool_e = F.adaptive_avg_pool2d(pool_f, 1).flatten(1)
    val_e = F.adaptive_avg_pool2d(val_f, 1).flatten(1)
    pool_e = F.normalize(pool_e, dim=1)
    val_c = F.normalize(val_e.mean(0, keepdim=True), dim=1)
    scores = (pool_e @ val_c.T).squeeze(1).numpy()
    return np.argsort(scores)[-k:]


def _warmup_on_pool(pool_f, pool_y, n_steps=200, lr=1e-4, sub=2000, rng=None):
    sub_idx = rng.choice(len(pool_f), size=min(sub, len(pool_f)), replace=False)
    ds = TensorDataset(pool_f[sub_idx], pool_y[sub_idx])
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = _build_head()
    _train_steps(model, loader, lr=lr, n_steps=n_steps)
    return model


def sel_el2n(pool_f, pool_y, val_f, val_y, k, rng):
    model = _warmup_on_pool(pool_f, pool_y, n_steps=200, rng=rng)
    model.eval()
    bs = 512
    scores = []
    with torch.no_grad():
        for i in range(0, len(pool_f), bs):
            x = pool_f[i:i + bs].to(DEVICE)
            y = pool_y[i:i + bs].to(DEVICE)
            p = F.softmax(model(x), dim=1)
            yh = F.one_hot(y, num_classes=10).float()
            scores.append(torch.norm(p - yh, dim=1).cpu())
    s = torch.cat(scores).numpy()
    return np.argsort(s)[-k:]


def sel_grand(pool_f, pool_y, val_f, val_y, k, rng):
    """GraNd: ‖∇_θ L(x,y)‖₂ at a lightly-warmed model. Approximated via fc-only grad norm."""
    model = _warmup_on_pool(pool_f, pool_y, n_steps=100, rng=rng)
    model.eval()
    fc = model.fc
    bs = 256
    scores = []
    crit = nn.CrossEntropyLoss(reduction="none")
    for i in range(0, len(pool_f), bs):
        x = pool_f[i:i + bs].to(DEVICE)
        y = pool_y[i:i + bs].to(DEVICE)
        # forward up to fc
        with torch.no_grad():
            feats = F.adaptive_avg_pool2d(model.layer4(x), 1).flatten(1)
        feats.requires_grad_(False)
        logits = fc(feats)
        # grad of CE wrt fc weights ≈ (softmax - onehot) ⊗ feats; norm per-sample analytic
        with torch.no_grad():
            p = F.softmax(logits, dim=1)
            yh = F.one_hot(y, num_classes=10).float()
            err = p - yh                              # [B, C]
            # ‖err‖_F ⊗ ‖feats‖₂ upper bounds true grad-norm and is the standard analytic GraNd surrogate
            err_n = err.norm(dim=1)
            feat_n = feats.norm(dim=1)
            scores.append((err_n * feat_n).cpu())
    s = torch.cat(scores).numpy()
    return np.argsort(s)[-k:]


def _flatten_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad]).detach()


def _val_grad(model, val_f, val_y):
    model.zero_grad()
    crit = nn.CrossEntropyLoss()
    loss = crit(model(val_f.to(DEVICE)), val_y.to(DEVICE))
    loss.backward()
    return _flatten_grad(model)


def _pool_grads_dot(model, pool_f, pool_y, gv, batch=128):
    """Per-sample dot ⟨∇L_pool(x_i), gv⟩ via per-example fc-grad analytic + layer4 vjp.
    For tractability we approximate per-sample grad-dot by (full-model grad of *one batch* averaged) — but
    standard LESS-style approximation: take per-sample grad of fc only and dot with gv-fc-portion.
    """
    crit = nn.CrossEntropyLoss(reduction="none")
    # split gv into fc/layer4 chunks
    sizes = [p.numel() for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    splits = torch.split(gv, sizes)
    gv_dict = dict(zip(names, splits))
    fc_w_idx = names.index("fc.weight"); fc_b_idx = names.index("fc.bias")
    gv_fc_w = splits[fc_w_idx].view(model.fc.weight.shape)
    gv_fc_b = splits[fc_b_idx]

    scores = []
    model.eval()
    for i in range(0, len(pool_f), batch):
        x = pool_f[i:i + batch].to(DEVICE)
        y = pool_y[i:i + batch].to(DEVICE)
        with torch.no_grad():
            feats = F.adaptive_avg_pool2d(model.layer4(x), 1).flatten(1)
            logits = model.fc(feats)
            p = F.softmax(logits, dim=1)
            yh = F.one_hot(y, num_classes=10).float()
            err = p - yh                     # [B, C]
            # per-sample fc-weight grad = err_i ⊗ feats_i, fc-bias grad = err_i
            # dot with gv_fc_w (shape [C, F]) gives <gv, err⊗feats> = err_i · (gv_fc_w · feats_i)
            t = (gv_fc_w @ feats.T).T        # [B, C]
            dot_w = (err * t).sum(1)
            dot_b = (err * gv_fc_b.unsqueeze(0)).sum(1)
            scores.append((dot_w + dot_b).cpu())
    return torch.cat(scores).numpy()


def sel_less(pool_f, pool_y, val_f, val_y, k, rng):
    """LESS: average ⟨∇_θ L_train(x_i; θ_t), ∇_θ L_val(θ_t)⟩ across multiple checkpoints."""
    sub_idx = rng.choice(len(pool_f), size=min(2000, len(pool_f)), replace=False)
    ds = TensorDataset(pool_f[sub_idx], pool_y[sub_idx])
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = _build_head()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    accum = np.zeros(len(pool_f))
    n_ckpts = 0
    epochs_per_ckpt = 1
    for _ in range(3):  # 3 checkpoints
        model.train()
        for _e in range(epochs_per_ckpt):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
        gv = _val_grad(model, val_f, val_y)
        accum += _pool_grads_dot(model, pool_f, pool_y, gv)
        n_ckpts += 1
    s = accum / n_ckpts
    return np.argsort(s)[-k:]


def sel_tov(pool_f, pool_y, val_f, val_y, k, rng):
    """ToV: warm on a small base subset of pool → perturb on val with 0.1× lr → score (loss before − loss after) on pool."""
    base_idx = rng.choice(len(pool_f), size=min(1024, len(pool_f)), replace=False)
    base_loader = DataLoader(TensorDataset(pool_f[base_idx], pool_y[base_idx]), batch_size=64, shuffle=True)
    model = _build_head()
    _train_epochs(model, base_loader, lr=1e-4, n_epochs=2)

    loss_before = _per_sample_loss(model, pool_f, pool_y)
    val_loader = DataLoader(TensorDataset(val_f, val_y), batch_size=32, shuffle=True)
    _train_epochs(model, val_loader, lr=1e-5, n_epochs=2)  # 0.1× perturbation lr
    loss_after = _per_sample_loss(model, pool_f, pool_y)

    s = loss_before - loss_after
    # ToV scores only over pool \ base
    out_mask = np.ones(len(pool_f), dtype=bool); out_mask[base_idx] = False
    s[~out_mask] = -np.inf
    return np.argsort(s)[-k:]


def sel_tacs(pool_f, pool_y, val_f, val_y, k, rng):
    """TACS: warmup on val from base model → score on pool by (loss_ckpt1 − loss_ckptT)."""
    val_loader = DataLoader(TensorDataset(val_f, val_y), batch_size=32, shuffle=True)
    model = _build_head()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    crit = nn.CrossEntropyLoss()

    # one epoch → θ_1
    model.train()
    for x, y in val_loader:
        opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    loss_1 = _per_sample_loss(model, pool_f, pool_y)

    # 3 more epochs → θ_T
    for _ in range(3):
        model.train()
        for x, y in val_loader:
            opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
    loss_T = _per_sample_loss(model, pool_f, pool_y)

    s = (loss_1 - loss_T) / np.maximum(loss_1, 1e-6)  # normalized drop
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
# main loop
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./cifar_data")
    ap.add_argument("--image-size", type=int, default=96, help="Input resolution; 96 is fast and keeps useful spatial features. Cluster runs can use 224.")
    ap.add_argument("--feature-cache", default="experiments/cv_cifar10_noisy/cifar10_layer3_feats.npz")
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train_ds, test_ds = _build_loaders(args.data_root, image_size=args.image_size)
    train_feats, train_labels_clean, test_feats, test_labels = _cache_layer3_features(
        train_ds, test_ds, Path(args.feature_cache))

    targets_clean_np = train_labels_clean.numpy()
    target_classes = args.target_classes

    runs = []
    t0 = time.time()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # val set: balanced from clean target classes
        val_idx = []
        for c in target_classes:
            idx_c = np.where(targets_clean_np == c)[0]
            val_idx.extend(rng.choice(idx_c, args.val_size // len(target_classes), replace=False))
        val_idx = np.array(val_idx)
        val_f = train_feats[val_idx]
        val_y = train_labels_clean[val_idx]

        # noisy pool
        avail = np.setdiff1d(np.arange(len(train_feats)), val_idx)
        pool_idx = rng.choice(avail, args.pool_size, replace=False)
        pool_f = train_feats[pool_idx]
        pool_y_clean = train_labels_clean[pool_idx].clone()
        noise_mask = rng.random(args.pool_size) < args.noise_rate
        pool_y_noisy = pool_y_clean.clone()
        pool_y_noisy[noise_mask] = torch.from_numpy(rng.integers(0, 10, size=int(noise_mask.sum())))
        is_clean = ~noise_mask
        is_target = np.isin(pool_y_clean.numpy(), target_classes)

        for method_name in args.methods:
            if method_name not in METHODS:
                print(f"[warn] unknown method {method_name}, skipping"); continue
            t_m = time.time()
            print(f"\n[seed={seed}] === {method_name} ===")
            try:
                # selection only depends on pool, val, budget; we re-use a single per-(seed,method)
                # selection by computing it at the largest budget and slicing for smaller budgets where
                # the score function is consistent. To keep methods independent, just run per-budget.
                for k in args.budgets:
                    sel_rng = np.random.default_rng(seed * 1000 + k)
                    sel = METHODS[method_name](pool_f, pool_y_noisy, val_f, val_y, k, sel_rng)
                    sel = np.asarray(sel)
                    # retrain
                    head = _build_head()
                    rt_loader = DataLoader(TensorDataset(pool_f[sel], pool_y_noisy[sel]),
                                           batch_size=64, shuffle=True)
                    _train_epochs(head, rt_loader, lr=args.retrain_lr, n_epochs=args.retrain_epochs)
                    acc = _eval_acc_binary(head, test_feats, test_labels, target_classes)
                    runs.append({
                        "seed": seed,
                        "method": method_name,
                        "k": k,
                        "test_acc_binary": acc,
                        "selected_clean_fraction": float(is_clean[sel].mean()),
                        "selected_target_fraction": float(is_target[sel].mean()),
                    })
                    print(f"  k={k:>5} acc={acc:5.2f} clean={is_clean[sel].mean():.3f} "
                          f"target={is_target[sel].mean():.3f}")
            except Exception as e:
                print(f"  [error] {method_name}: {e}")
                runs.append({"seed": seed, "method": method_name, "error": str(e)})
            print(f"  ({time.time()-t_m:.1f}s)")

    # aggregate
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
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "clean_frac_mean": float(np.mean(cleans)),
                "target_frac_mean": float(np.mean(tgts)),
                "n_seeds": len(accs),
            }
        summary[m] = {"k": per_k}

    out = {"config": vars(args), "runs": runs, "summary": summary,
           "runtime_seconds": time.time() - t0}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.output} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
