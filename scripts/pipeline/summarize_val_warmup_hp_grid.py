#!/usr/bin/env python3
"""Summarize a full LR × Depth/epoch grid search for val-warmup HP selection.

Reads the directory layout produced by run_val_warmup_hp_grid_search.sh:
  <run_root>/grid/<fold_label>/lr_<lr>/depth_<dd>/<task>/probe_scores.json
  <run_root>/grid/<fold_label>/lr_<lr>/depth_<dd>/<task>/train_probe_influence_score.pt

Outputs:
  <run_root>/analysis/grid_summary.json  -- full (lr, depth, fold) table
  <run_root>/analysis/grid_summary.csv   -- flat CSV of mean AUROC per cell
  <run_root>/analysis/hp_selection.json  -- {task: {lr, lr_value, depth, auroc, ...}}
                                            (same schema as the 2-stage hp_selection.json
                                             so summarize_val_warmup_hp_tasks.py can reuse it)
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# helpers (copied from summarize_val_warmup_hp_search.py)
# ---------------------------------------------------------------------------

def _load_probe_scores(path: Path) -> torch.Tensor:
    obj = json.loads(path.read_text(encoding="utf-8"))
    target_valid = obj.get("target_valid")
    if not isinstance(target_valid, dict):
        raise ValueError(f"target_valid missing in {path}")
    scores = target_valid.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"target_valid.scores missing in {path}")
    return torch.tensor([float(x) for x in scores], dtype=torch.float32)


def _load_pool_scores(path: Path) -> torch.Tensor:
    scores = torch.load(path, map_location="cpu")
    if not torch.is_tensor(scores):
        raise ValueError(f"expected tensor in {path}, got {type(scores)}")
    scores = scores.float().flatten()
    if scores.numel() == 0:
        raise ValueError(f"empty tensor in {path}")
    return scores


def _pairwise_auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((((diff > 0).float()) + 0.5 * ((diff == 0).float())).mean().item())


def _pairwise_soft_auroc(
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> float:
    """Continuous, scale-normalized relaxation of pairwise AUROC."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    if temperature <= 0:
        raise ValueError("soft AUROC temperature must be positive")
    scale = float(neg.std(unbiased=True).item()) if neg.numel() > 1 else 0.0
    scale = max(scale, 1e-8)
    logits = (pos[:, None] - neg[None, :]) / (float(temperature) * scale)
    return float(torch.sigmoid(logits).mean().item())


def select_grid_cell(
    eligible_cells: List[Dict],
    *,
    policy: str,
    plateau_se_multiplier: float = 0.0,
    score_key: str = "mean_auroc",
) -> Tuple[Dict, Dict]:
    """Select an HP cell, optionally preferring the earliest reliable plateau."""
    if not eligible_cells:
        raise ValueError("eligible_cells must not be empty")
    if policy not in {"max_auroc", "earliest_reliable_plateau"}:
        raise ValueError(f"unknown selection policy: {policy}")
    if plateau_se_multiplier < 0:
        raise ValueError("plateau_se_multiplier must be non-negative")

    reference_best = max(
        eligible_cells,
        key=lambda c: (
            c[score_key],
            c["mean_auroc"],
            c["target_positive_share_mean"],
            c["target_min_min"],
            -c["depth"],
        ),
    )
    if policy == "max_auroc":
        return reference_best, {
            "policy": policy,
            "reference_best_lr": reference_best["lr_str"],
            "reference_best_depth": reference_best["depth"],
            "selection_score_key": score_key,
            "plateau_threshold": reference_best[score_key],
            "plateau_tolerance": 0.0,
            "plateau_cells": 1,
        }

    fold_values_key = "fold_aurocs" if score_key == "mean_auroc" else "fold_soft_aurocs"
    fold_aurocs = [float(value) for value in reference_best.get(fold_values_key, [])]
    standard_error = (
        statistics.stdev(fold_aurocs) / math.sqrt(len(fold_aurocs))
        if len(fold_aurocs) > 1
        else 0.0
    )
    comparisons = sum(
        int(fold.get("n_target", 0)) * int(fold.get("n_pool", 0))
        for fold in reference_best.get("fold_details", [])
    )
    # One win among all cross-fold positive/negative comparisons is the
    # smallest meaningful AUROC change when there are no exact score ties.
    resolution = 1.0 / comparisons if comparisons > 0 and score_key == "mean_auroc" else 0.0
    tolerance = max(float(plateau_se_multiplier) * standard_error, resolution)
    threshold = float(reference_best[score_key]) - tolerance - 1e-12
    plateau = [cell for cell in eligible_cells if float(cell[score_key]) >= threshold]
    best_cell = min(
        plateau,
        key=lambda c: (
            c["depth"],
            -c[score_key],
            -c["mean_auroc"],
            -c["target_positive_share_mean"],
            -c["target_min_min"],
            c["lr_value"],
        ),
    )
    return best_cell, {
        "policy": policy,
        "reference_best_lr": reference_best["lr_str"],
        "reference_best_depth": reference_best["depth"],
        "selection_score_key": score_key,
        "reference_best_score": reference_best[score_key],
        "reference_best_auroc": reference_best["mean_auroc"],
        "reference_best_fold_se": standard_error,
        "auroc_resolution": resolution,
        "plateau_se_multiplier": float(plateau_se_multiplier),
        "plateau_tolerance": tolerance,
        "plateau_threshold": threshold,
        "plateau_cells": len(plateau),
        "plateau_members": [
            {
                "lr": cell["lr_str"],
                "depth": cell["depth"],
                "mean_auroc": cell["mean_auroc"],
                "selection_score": cell[score_key],
            }
            for cell in sorted(plateau, key=lambda c: (c["depth"], c["lr_value"]))
        ],
    }


# ---------------------------------------------------------------------------
# directory traversal
# ---------------------------------------------------------------------------

def _parse_lr(lr_str: str) -> Optional[float]:
    """Parse lr string like '2e-5' or '5e-6' to float."""
    try:
        return float(lr_str)
    except ValueError:
        return None


def _parse_depth(depth_str: str) -> Optional[int]:
    """Parse depth string like '01', '02', '05', '10' to int."""
    try:
        return int(depth_str)
    except ValueError:
        return None


def collect_grid(run_root: Path, task: str) -> List[Dict]:
    """Walk run_root/grid/fold_*/lr_*/depth_*/{task}/ and collect score paths."""
    records = []
    grid_dir = run_root / "grid"
    if not grid_dir.is_dir():
        return records

    for fold_dir in sorted(grid_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        fold_label = fold_dir.name

        for lr_dir in sorted(fold_dir.iterdir()):
            if not lr_dir.is_dir() or not lr_dir.name.startswith("lr_"):
                continue
            lr_str = lr_dir.name[len("lr_"):]
            lr_value = _parse_lr(lr_str)
            if lr_value is None:
                continue

            for depth_dir in sorted(lr_dir.iterdir()):
                if not depth_dir.is_dir() or not depth_dir.name.startswith("depth_"):
                    continue
                depth_str = depth_dir.name[len("depth_"):]
                depth = _parse_depth(depth_str)
                if depth is None:
                    continue

                task_dir = depth_dir / task
                probe_path = task_dir / "probe_scores.json"
                pool_path = task_dir / "train_probe_influence_score.pt"
                records.append(
                    {
                        "fold": fold_label,
                        "lr_str": lr_str,
                        "lr_value": lr_value,
                        "depth": depth,
                        "probe_path": probe_path,
                        "pool_path": pool_path,
                    }
                )
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize val-warmup HP grid search (LR × Depth)."
    )
    ap.add_argument("--run-root", required=True, help="OUTPUT_ROOT for this task.")
    ap.add_argument("--task", required=True, choices=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--score-metric-name", default="norm_loss_gap")
    ap.add_argument(
        "--selection-metric",
        choices=["hard_auroc", "soft_auroc", "hard_then_soft"],
        default="hard_auroc",
        help=(
            "soft_auroc replaces hard pairwise wins with a sigmoid of the "
            "positive-negative margin normalized by the negative-score SD; "
            "hard_then_soft preserves a unique hard-AUROC winner and uses "
            "soft AUROC only to resolve an exact hard-AUROC tie."
        ),
    )
    ap.add_argument("--soft-auroc-temperature", type=float, default=1.0)
    ap.add_argument("--grid-value-label", choices=["depth", "epoch"], default="depth")
    ap.add_argument("--require-positive-holdout", type=int, default=0)
    ap.add_argument("--positive-holdout-mode", choices=["all", "mean"], default="all")
    ap.add_argument("--positive-holdout-threshold", type=float, default=0.0)
    ap.add_argument(
        "--selection-policy",
        choices=["max_auroc", "earliest_reliable_plateau"],
        default="max_auroc",
        help=(
            "earliest_reliable_plateau chooses the smallest depth within one "
            "empirical AUROC resolution unit, optionally widened by fold SE."
        ),
    )
    ap.add_argument(
        "--plateau-se-multiplier",
        type=float,
        default=0.0,
        help=(
            "Optional fold-SE widening for the earliest plateau. The default "
            "uses only one empirical AUROC resolution unit because three-fold "
            "SE bands can be excessively wide."
        ),
    )
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    records = collect_grid(run_root, args.task)
    if not records:
        raise SystemExit(
            f"No grid results found under {run_root / 'grid'}. "
            "Run run_val_warmup_hp_grid_search.sh first."
        )

    # Group by (lr_str, depth) → list of folds
    CellKey = Tuple[str, int]
    cell_folds: Dict[CellKey, List[Dict]] = {}
    all_lrs: Dict[str, float] = {}
    all_depths: set = set()

    for rec in records:
        key: CellKey = (rec["lr_str"], rec["depth"])
        cell_folds.setdefault(key, []).append(rec)
        all_lrs[rec["lr_str"]] = rec["lr_value"]
        all_depths.add(rec["depth"])

    sorted_lrs = sorted(all_lrs.keys(), key=lambda s: all_lrs[s])
    sorted_depths = sorted(all_depths)

    # Compute mean AUROC per cell
    cell_results: Dict[CellKey, Dict] = {}
    require_positive_holdout = bool(args.require_positive_holdout)

    for key, folds in cell_folds.items():
        lr_str, depth = key
        fold_aurocs = []
        fold_soft_aurocs = []
        fold_details = []
        error = ""
        fold_positive_ok = []
        fold_target_positive_shares = []
        fold_target_mins = []

        for fold in sorted(folds, key=lambda r: r["fold"]):
            p_path = fold["probe_path"]
            pool_path = fold["pool_path"]
            if not p_path.exists() or not pool_path.exists():
                error = f"missing artifacts: fold={fold['fold']}"
                break
            try:
                pos = _load_probe_scores(p_path)
                neg = _load_pool_scores(pool_path)
                auroc = _pairwise_auroc(pos, neg)
                soft_auroc = _pairwise_soft_auroc(
                    pos,
                    neg,
                    temperature=args.soft_auroc_temperature,
                )
            except Exception as exc:
                error = f"{fold['fold']}: {exc}"
                break
            target_mean = float(pos.mean().item())
            target_min = float(pos.min().item())
            target_positive_share = float((pos > args.positive_holdout_threshold).float().mean().item())
            if args.positive_holdout_mode == "all":
                positive_ok = bool(bool((pos > args.positive_holdout_threshold).all().item()))
            else:
                positive_ok = bool(target_mean > args.positive_holdout_threshold)
            fold_aurocs.append(auroc)
            fold_soft_aurocs.append(soft_auroc)
            fold_positive_ok.append(positive_ok)
            fold_target_positive_shares.append(target_positive_share)
            fold_target_mins.append(target_min)
            fold_details.append(
                {
                    "fold": fold["fold"],
                    "auroc": auroc,
                    "soft_auroc": soft_auroc,
                    "n_target": int(pos.numel()),
                    "n_pool": int(neg.numel()),
                    "target_mean": target_mean,
                    "target_min": target_min,
                    "target_positive_share": target_positive_share,
                    "positive_holdout_ok": positive_ok,
                }
            )

        mean_auroc = (
            float(sum(fold_aurocs) / len(fold_aurocs))
            if fold_aurocs and not error
            else float("nan")
        )
        positive_holdout_ok = bool(fold_positive_ok and all(fold_positive_ok) and not error)
        mean_soft_auroc = (
            float(sum(fold_soft_aurocs) / len(fold_soft_aurocs))
            if fold_soft_aurocs and not error
            else float("nan")
        )

        cell_results[key] = {
            "lr_str": lr_str,
            "lr_value": all_lrs[lr_str],
            "depth": depth,
            "n_folds": len(folds),
            "mean_auroc": mean_auroc,
            "fold_aurocs": fold_aurocs,
            "mean_soft_auroc": mean_soft_auroc,
            "fold_soft_aurocs": fold_soft_aurocs,
            "fold_details": fold_details,
            "positive_holdout_ok": positive_holdout_ok,
            "positive_holdout_mode": args.positive_holdout_mode,
            "positive_holdout_threshold": float(args.positive_holdout_threshold),
            "target_positive_share_mean": (
                float(sum(fold_target_positive_shares) / len(fold_target_positive_shares))
                if fold_target_positive_shares else float("nan")
            ),
            "target_min_min": (
                float(min(fold_target_mins)) if fold_target_mins else float("nan")
            ),
            "error": error,
        }

    # Find best cell
    selection_score_key = (
        "mean_soft_auroc"
        if args.selection_metric in {"soft_auroc", "hard_then_soft"}
        else "mean_auroc"
    )
    valid_cells = [
        v for v in cell_results.values()
        if not v["error"] and not (v[selection_score_key] != v[selection_score_key])
    ]
    if not valid_cells:
        raise SystemExit(f"No valid grid cells found under {run_root}")
    eligible_cells = valid_cells
    selection_reason = f"best_{args.selection_metric}"
    if require_positive_holdout:
        positive_cells = [v for v in valid_cells if v["positive_holdout_ok"]]
        if positive_cells:
            eligible_cells = positive_cells
            selection_reason = f"best_{args.selection_metric}_subject_to_positive_holdout"
        else:
            raise SystemExit(
                "No eligible grid cell has a positive holdout under "
                "--require-positive-holdout; refusing degenerate calibration selection"
            )

    if args.selection_metric == "hard_then_soft":
        hard_best = max(float(cell["mean_auroc"]) for cell in eligible_cells)
        hard_tie_tolerance = 1e-12
        hard_tied_cells = [
            cell
            for cell in eligible_cells
            if math.isclose(
                float(cell["mean_auroc"]),
                hard_best,
                rel_tol=0.0,
                abs_tol=hard_tie_tolerance,
            )
        ]
        best_cell, selection_diagnostics = select_grid_cell(
            hard_tied_cells,
            policy=args.selection_policy,
            plateau_se_multiplier=args.plateau_se_multiplier,
            score_key="mean_soft_auroc",
        )
        selection_diagnostics = {
            **selection_diagnostics,
            "hard_stage_max_auroc": hard_best,
            "hard_stage_tie_tolerance": hard_tie_tolerance,
            "hard_stage_tie_cells": [
                {"lr": cell["lr_str"], "depth": cell["depth"]}
                for cell in sorted(
                    hard_tied_cells,
                    key=lambda c: (c["depth"], c["lr_value"]),
                )
            ],
        }
        selection_reason = selection_reason.replace(
            "best_hard_then_soft", "best_hard_auroc_then_soft_tiebreak"
        )
    else:
        best_cell, selection_diagnostics = select_grid_cell(
            eligible_cells,
            policy=args.selection_policy,
            plateau_se_multiplier=args.plateau_se_multiplier,
            score_key=selection_score_key,
        )
    if args.selection_policy == "earliest_reliable_plateau":
        selection_reason = f"{selection_reason}_earliest_reliable_plateau"

    # ---- write grid_summary.json ----
    summary_payload = {
        "task": args.task,
        "run_root": str(run_root),
        "lr_grid": sorted_lrs,
        "depth_grid": sorted_depths,
        "score_metric_name": args.score_metric_name,
        "selection_metric": args.selection_metric,
        "soft_auroc_temperature": float(args.soft_auroc_temperature),
        "grid_value_label": args.grid_value_label,
        "require_positive_holdout": require_positive_holdout,
        "positive_holdout_mode": args.positive_holdout_mode,
        "positive_holdout_threshold": float(args.positive_holdout_threshold),
        "selection_reason": selection_reason,
        "selection_policy": args.selection_policy,
        "selection_diagnostics": selection_diagnostics,
        "best_lr": best_cell["lr_str"],
        "best_depth": best_cell["depth"],
        "best_grid_value": best_cell["depth"],
        "best_auroc": best_cell["mean_auroc"],
        "best_selection_score": best_cell[selection_score_key],
        "cells": [
            {
                "lr": v["lr_str"],
                "lr_value": v["lr_value"],
                "depth": v["depth"],
                "grid_value": v["depth"],
                "n_folds": v["n_folds"],
                "mean_auroc": v["mean_auroc"],
                "fold_aurocs": v["fold_aurocs"],
                "mean_soft_auroc": v["mean_soft_auroc"],
                "fold_soft_aurocs": v["fold_soft_aurocs"],
                "fold_details": v["fold_details"],
                "positive_holdout_ok": v["positive_holdout_ok"],
                "positive_holdout_mode": v["positive_holdout_mode"],
                "positive_holdout_threshold": v["positive_holdout_threshold"],
                "target_positive_share_mean": v["target_positive_share_mean"],
                "target_min_min": v["target_min_min"],
                "error": v["error"],
            }
            for v in sorted(
                cell_results.values(),
                key=lambda c: (all_lrs[c["lr_str"]], c["depth"]),
            )
        ],
    }
    json_path = analysis_dir / "grid_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    # ---- write grid_summary.csv ----
    csv_path = analysis_dir / "grid_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "lr",
                "lr_value",
                "depth",
                "grid_value",
                "n_folds",
                "mean_auroc",
                "mean_soft_auroc",
                "positive_holdout_ok",
                "target_positive_share_mean",
                "target_min_min",
                "error",
            ],
        )
        writer.writeheader()
        for v in sorted(
            cell_results.values(),
            key=lambda c: (all_lrs[c["lr_str"]], c["depth"]),
        ):
            writer.writerow(
                {
                    "task": args.task,
                    "lr": v["lr_str"],
                    "lr_value": v["lr_value"],
                    "depth": v["depth"],
                    "grid_value": v["depth"],
                    "n_folds": v["n_folds"],
                    "mean_auroc": v["mean_auroc"],
                    "mean_soft_auroc": v["mean_soft_auroc"],
                    "positive_holdout_ok": v["positive_holdout_ok"],
                    "target_positive_share_mean": v["target_positive_share_mean"],
                    "target_min_min": v["target_min_min"],
                    "error": v["error"],
                }
            )
    print(f"wrote {csv_path}")

    # ---- write hp_selection.json (same schema as 2-stage) ----
    # summarize_val_warmup_hp_tasks.py expects:
    #   {task: {lr, lr_value, depth, lr_auroc, depth_auroc}}
    hp_path = analysis_dir / "hp_selection.json"
    hp_payload = {
        args.task: {
            "lr": best_cell["lr_str"],
            "lr_value": best_cell["lr_value"],
            "depth": best_cell["depth"],
            "epoch_k": best_cell["depth"] if args.grid_value_label == "epoch" else None,
            "lr_auroc": best_cell["mean_auroc"],   # single joint AUROC
            "depth_auroc": best_cell["mean_auroc"],
            "grid_auroc": best_cell["mean_auroc"],
            "grid_soft_auroc": best_cell["mean_soft_auroc"],
            "grid_selection_score": best_cell[selection_score_key],
            "score_metric": args.score_metric_name,
            "selection_metric": args.selection_metric,
            "soft_auroc_temperature": float(args.soft_auroc_temperature),
            "grid_value_label": args.grid_value_label,
            "require_positive_holdout": require_positive_holdout,
            "positive_holdout_mode": args.positive_holdout_mode,
            "positive_holdout_threshold": float(args.positive_holdout_threshold),
            "positive_holdout_ok": best_cell["positive_holdout_ok"],
            "selection_reason": selection_reason,
            "selection_policy": args.selection_policy,
            "selection_diagnostics": selection_diagnostics,
            "grid_best_cell": {
                "lr": best_cell["lr_str"],
                "depth": best_cell["depth"],
            },
        }
    }
    hp_path.write_text(json.dumps(hp_payload, indent=2), encoding="utf-8")
    print(f"wrote {hp_path}")

    # ---- print grid table ----
    print(
        f"\n=== Grid AUROC table: {args.task} "
        f"(best: LR={best_cell['lr_str']}, D={best_cell['depth']}, "
        f"AUROC={best_cell['mean_auroc']:.4f}, "
        f"selection={args.selection_metric}:{best_cell[selection_score_key]:.6f}) ==="
    )
    # header
    header = f"{'LR':>10} | " + " | ".join(f"D={d:>2}" for d in sorted_depths)
    print(header)
    print("-" * len(header))
    for lr_str in sorted_lrs:
        row_vals = []
        for depth in sorted_depths:
            cell = cell_results.get((lr_str, depth))
            if cell and not cell["error"]:
                mark = "*" if (lr_str == best_cell["lr_str"] and depth == best_cell["depth"]) else " "
                row_vals.append(f"{cell['mean_auroc']:.4f}{mark}")
            else:
                row_vals.append("  ---- ")
        print(f"{lr_str:>10} | " + " | ".join(f"{v:>7}" for v in row_vals))
    print()
    print(
        f"best_lr={best_cell['lr_str']} "
        f"best_depth={best_cell['depth']} "
        f"auroc={best_cell['mean_auroc']:.6f} "
        f"selection_score={best_cell[selection_score_key]:.6f}"
    )


if __name__ == "__main__":
    main()
