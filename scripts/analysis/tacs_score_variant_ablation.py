#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
LOGISTIC_DIR = REPO / "experiments" / "logistic_shift"
CIFAR_DIR = REPO / "experiments" / "cv_cifar10_noisy"

VARIANTS = [
    "raw_theta0",
    "norm_theta0",
    "raw_theta1",
    "norm_theta1",
]
MAIN_VARIANT = "norm_theta1"
VARIANT_LABELS = {
    "raw_theta0": "raw, theta0->thetaT",
    "norm_theta0": "normalized, theta0->thetaT",
    "raw_theta1": "raw, theta1->thetaT",
    "norm_theta1": "normalized, theta1->thetaT",
}


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0.0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rankdata_average(a), rankdata_average(b))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), len(scores))
    return np.argsort(-np.asarray(scores), kind="mergesort")[:k]


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(topk_indices(a, k).tolist())
    ib = set(topk_indices(b, k).tolist())
    return float(len(ia & ib) / max(len(ia), 1))


def binary_auc(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    n_pos = int(positive_mask.sum())
    n_neg = int((~positive_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(scores)
    rank_sum_pos = float(ranks[positive_mask].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_variants(loss_stack: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    loss_stack = np.asarray(loss_stack, dtype=np.float64)
    if loss_stack.ndim != 2 or loss_stack.shape[0] < 3:
        raise ValueError(f"expected losses for theta0, theta1, thetaT; got shape {loss_stack.shape}")
    l0 = loss_stack[0]
    l1 = loss_stack[1]
    lt = loss_stack[-1]
    raw0 = l0 - lt
    raw1 = l1 - lt
    return {
        "raw_theta0": raw0,
        "norm_theta0": raw0 / np.maximum(l0, eps),
        "raw_theta1": raw1,
        "norm_theta1": raw1 / np.maximum(l1, eps),
    }


def summarize_scores(
    scores: dict[str, np.ndarray],
    budgets: list[int],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    main = scores[MAIN_VARIANT]
    for variant in VARIANTS:
        s = scores[variant]
        row: dict[str, Any] = {
            "spearman_vs_main": spearman(s, main),
            "pearson_vs_main": corr(s, main),
            "k": {},
        }
        for name, mask in masks.items():
            row[f"auc_{name}"] = binary_auc(s, mask)
        for k in budgets:
            idx = topk_indices(s, k)
            krow = {"overlap_vs_main": topk_overlap(s, main, k)}
            for name, mask in masks.items():
                krow[f"top_{name}_mean"] = float(np.asarray(mask)[idx].mean())
            row["k"][str(k)] = krow
        out[variant] = row
    return out


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def aggregate(section_rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_seed": section_rows, "mean": {}}
    if not section_rows:
        return out
    for variant in VARIANTS:
        vrows = [row["variants"][variant] for row in section_rows]
        agg: dict[str, Any] = {}
        scalar_keys = sorted(k for k in vrows[0].keys() if k != "k")
        for key in scalar_keys:
            agg[key] = mean_std([float(r[key]) for r in vrows])
        agg["k"] = {}
        for k in budgets:
            kk = str(k)
            kkeys = sorted(vrows[0]["k"][kk].keys())
            agg["k"][kk] = {
                key: mean_std([float(r["k"][kk][key]) for r in vrows])
                for key in kkeys
            }
        out["mean"][variant] = agg
    return out


def add_logistic_downstream(
    variant_rows: dict[str, Any],
    scores: dict[str, np.ndarray],
    problem: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    exp: Any,
    budgets: list[int],
) -> None:
    cand = problem["candidate_idx"]
    for variant in VARIANTS:
        order = topk_indices(scores[variant], max(budgets))
        for k in budgets:
            selected = cand[order[: min(k, len(order))]]
            metrics = exp.train_final_model(selected, config, problem, seed + 100_003 + k)
            variant_rows[variant]["k"][str(k)].update(
                {
                    "classification_error": float(metrics["classification_error"]),
                    "test_log_loss": float(metrics["test_log_loss"]),
                    "selected_target_fraction": float(metrics["selected_target_fraction"]),
                }
            )


def run_logistic(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(LOGISTIC_DIR))
    import run_logistic_tov_style_comparison as exp  # type: ignore

    config = exp.build_config(
        quick=True,
        val_size_override=args.logistic_val_size,
        seed_count_override=args.seeds,
        tacs_score_window="ckpt1_to_last",
        tacs_hp_search=False,
    )
    config["train_size"] = int(args.logistic_train_size)
    config["base_size"] = min(int(args.logistic_base_size), config["train_size"] // 2)
    config["proxy_size"] = min(int(args.logistic_proxy_size), config["proxy_size"])
    config["budgets"] = list(args.budgets)

    rows = []
    lrs = exp.make_linear_decay_schedule(config["tacs_val_epochs"], config["lr0"] * config["val_lr_scale"])
    for seed in config["seeds"]:
        problem = exp.make_problem(config, seed)
        params = exp.init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))
        cand = problem["candidate_idx"]
        losses = [exp.per_sample_losses(params, problem["x_train"][cand], problem["y_train"][cand])]
        for lr in lrs:
            params = exp.gd_step(params, problem["x_val"], problem["y_val"], lr)
            losses.append(exp.per_sample_losses(params, problem["x_train"][cand], problem["y_train"][cand]))
        scores = score_variants(np.stack(losses, axis=0), eps=args.logistic_eps)
        variant_rows = summarize_scores(
            scores,
            args.budgets,
            {"target": problem["source_target"][cand]},
        )
        if not args.skip_logistic_downstream:
            add_logistic_downstream(variant_rows, scores, problem, config, seed, exp, args.budgets)
        rows.append({"seed": int(seed), "variants": variant_rows})
    return aggregate(rows, args.budgets)


def run_cifar(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(CIFAR_DIR))
    import torch  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    import run_cifar10_noisy as exp  # type: ignore

    cache_path = Path(args.cifar_feature_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CIFAR feature cache not found: {cache_path}. "
            "Run run_cifar10_noisy.py once, or pass --skip-cifar."
        )
    cache = np.load(cache_path)
    train_feats = torch.from_numpy(cache["train_feats"])
    train_labels_clean = torch.from_numpy(cache["train_labels_clean"])
    targets_clean_np = train_labels_clean.numpy()
    target_classes = list(args.cifar_target_classes)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        val_idx = []
        for c in target_classes:
            idx_c = np.where(targets_clean_np == c)[0]
            val_idx.extend(rng.choice(idx_c, args.cifar_val_size // len(target_classes), replace=False))
        val_idx = np.asarray(val_idx)
        val_f = train_feats[val_idx]
        val_y = train_labels_clean[val_idx]

        avail = np.setdiff1d(np.arange(len(train_feats)), val_idx)
        pool_idx = rng.choice(avail, args.cifar_pool_size, replace=False)
        pool_f = train_feats[pool_idx]
        pool_y_clean = train_labels_clean[pool_idx].clone()
        noise_mask = rng.random(args.cifar_pool_size) < args.cifar_noise_rate
        pool_y_noisy = pool_y_clean.clone()
        pool_y_noisy[noise_mask] = torch.from_numpy(rng.integers(0, 10, size=int(noise_mask.sum())))

        val_loader = DataLoader(TensorDataset(val_f, val_y), batch_size=args.cifar_val_batch_size, shuffle=True)
        model = exp._build_head()
        opt = optim.Adam(model.parameters(), lr=args.cifar_tacs_lr)
        crit = nn.CrossEntropyLoss()
        losses = [exp._per_sample_loss(model, pool_f, pool_y_noisy, batch=args.cifar_score_batch_size)]
        for _ in range(args.cifar_tacs_epochs):
            model.train()
            for x, y in val_loader:
                opt.zero_grad()
                crit(model(x.to(exp.DEVICE)), y.to(exp.DEVICE)).backward()
                opt.step()
            losses.append(exp._per_sample_loss(model, pool_f, pool_y_noisy, batch=args.cifar_score_batch_size))

        scores = score_variants(np.stack(losses, axis=0), eps=args.cifar_eps)
        masks = {
            "clean": ~noise_mask,
            "target": np.isin(pool_y_clean.numpy(), target_classes),
        }
        variant_rows = summarize_scores(scores, args.budgets, masks)

        if args.cifar_retrain:
            test_feats = torch.from_numpy(cache["test_feats"])
            test_labels = torch.from_numpy(cache["test_labels"])
            for variant in VARIANTS:
                order = topk_indices(scores[variant], max(args.budgets))
                for k in args.budgets:
                    sel = order[: min(k, len(order))]
                    head = exp._build_head()
                    loader = DataLoader(
                        TensorDataset(pool_f[sel], pool_y_noisy[sel]),
                        batch_size=args.cifar_retrain_batch_size,
                        shuffle=True,
                    )
                    exp._train_epochs(head, loader, lr=args.cifar_retrain_lr, n_epochs=args.cifar_retrain_epochs)
                    acc = exp._eval_acc_binary(head, test_feats, test_labels, target_classes)
                    variant_rows[variant]["k"][str(k)]["binary_acc"] = float(acc)

        rows.append({"seed": int(seed), "variants": variant_rows})
    return aggregate(rows, args.budgets)


def fmt_ms(stat: dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def write_scalar_table(lines: list[str], title: str, section: dict[str, Any], keys: list[str], digits: int = 3) -> None:
    lines.extend([f"## {title}", ""])
    headers = ["variant"] + keys
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for variant in VARIANTS:
        row = [VARIANT_LABELS[variant]]
        for key in keys:
            row.append(fmt_ms(section["mean"][variant][key], digits))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_k_table(
    lines: list[str],
    title: str,
    section: dict[str, Any],
    budgets: list[int],
    metric: str,
    digits: int = 3,
) -> None:
    lines.extend([f"## {title}", ""])
    headers = ["variant"] + [f"k={k}" for k in budgets]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for variant in VARIANTS:
        row = [VARIANT_LABELS[variant]]
        for k in budgets:
            row.append(fmt_ms(section["mean"][variant]["k"][str(k)][metric], digits))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# TACS score variant ablation",
        "",
        f"Main score: `{MAIN_VARIANT}` = normalized endpoint drop from theta1 to thetaT.",
        "The tables report mean +/- std over seeds. Higher AUC, overlap, target fraction, clean fraction, and binary accuracy are better; lower logistic error/log-loss is better.",
        "",
    ]
    budgets = payload["config"]["budgets"]
    if "logistic" in payload:
        section = payload["logistic"]
        scalar_keys = ["auc_target", "spearman_vs_main", "pearson_vs_main"]
        lines.extend(["# Logistic regression", ""])
        write_scalar_table(lines, "Logistic score diagnostics", section, scalar_keys)
        write_k_table(lines, "Logistic top-k target fraction", section, budgets, "top_target_mean")
        if not payload["config"]["skip_logistic_downstream"]:
            write_k_table(lines, "Logistic downstream classification error", section, budgets, "classification_error")
            write_k_table(lines, "Logistic downstream test log-loss", section, budgets, "test_log_loss")
    if "cifar10" in payload:
        section = payload["cifar10"]
        scalar_keys = ["auc_clean", "auc_target", "spearman_vs_main", "pearson_vs_main"]
        lines.extend(["# CIFAR-10 noisy CV", ""])
        write_scalar_table(lines, "CIFAR score diagnostics", section, scalar_keys)
        write_k_table(lines, "CIFAR top-k clean fraction", section, budgets, "top_clean_mean")
        write_k_table(lines, "CIFAR top-k target-class fraction", section, budgets, "top_target_mean")
        if payload["config"]["cifar_retrain"]:
            write_k_table(lines, "CIFAR downstream binary accuracy", section, budgets, "binary_acc")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast 2x2 TACS score ablation: normalization x theta0 dropping.")
    ap.add_argument("--out-json", default=str(REPO / "analysis_outputs" / "tacs_score_variant_ablation.json"))
    ap.add_argument("--out-md", default=str(REPO / "analysis_outputs" / "tacs_score_variant_ablation.md"))
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 250, 500])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-logistic", action="store_true")
    ap.add_argument("--skip-cifar", action="store_true")

    ap.add_argument("--logistic-train-size", type=int, default=8192)
    ap.add_argument("--logistic-base-size", type=int, default=512)
    ap.add_argument("--logistic-val-size", type=int, default=256)
    ap.add_argument("--logistic-proxy-size", type=int, default=128)
    ap.add_argument("--logistic-eps", type=float, default=1e-12)
    ap.add_argument("--skip-logistic-downstream", action="store_true")

    ap.add_argument("--cifar-feature-cache", default=str(CIFAR_DIR / "cifar10_layer3_feats_96.npz"))
    ap.add_argument("--cifar-pool-size", type=int, default=2000)
    ap.add_argument("--cifar-val-size", type=int, default=50)
    ap.add_argument("--cifar-noise-rate", type=float, default=0.4)
    ap.add_argument("--cifar-target-classes", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--cifar-tacs-lr", type=float, default=5e-5)
    ap.add_argument("--cifar-tacs-epochs", type=int, default=4)
    ap.add_argument("--cifar-eps", type=float, default=1e-6)
    ap.add_argument("--cifar-val-batch-size", type=int, default=32)
    ap.add_argument("--cifar-score-batch-size", type=int, default=512)
    ap.add_argument("--cifar-retrain", action="store_true")
    ap.add_argument("--cifar-retrain-epochs", type=int, default=4)
    ap.add_argument("--cifar-retrain-lr", type=float, default=1e-4)
    ap.add_argument("--cifar-retrain-batch-size", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    payload: dict[str, Any] = {"config": vars(args)}
    if not args.skip_logistic:
        payload["logistic"] = run_logistic(args)
    if not args.skip_cifar:
        payload["cifar10"] = run_cifar(args)
    payload["runtime_seconds"] = time.time() - t0

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"runtime_seconds={payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
