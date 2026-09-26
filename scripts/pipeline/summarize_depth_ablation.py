#!/usr/bin/env python3
"""Summarize depth ablation results (warmup-seed variation).

Walks retrain output directories to collect eval metrics per
(task, depth, warmup_seed), computes mean +/- std across warmup seeds,
and reports optimal depth per task.

Usage:
    python3 scripts/pipeline/summarize_depth_ablation.py \
        --study-root /path/to/selected_data_runs/depth_ablation_l32_3b_... \
        --scratch /path/to/scratch

Outputs:
    {study_root}/analysis/summary.csv
    {study_root}/analysis/summary.json
    {study_root}/analysis/optimal_depths.json
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metric extraction (mirrors backfill_results_store.py)
# ---------------------------------------------------------------------------

METRIC_NAMES = {"tydiqa": "f1", "mmlu": "accuracy", "bbh": "em"}


def extract_metric(metrics_path: Path, task: str) -> Optional[float]:
    """Extract the primary metric from a metrics.json file."""
    if not metrics_path.is_file():
        return None
    try:
        d = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if task == "tydiqa":
        avg = d.get("average", {})
        return avg.get("f1") or avg.get("exact_match")
    elif task == "mmlu":
        return d.get("average_acc")
    elif task == "bbh":
        if "average_exact_match" in d:
            return d["average_exact_match"]
        vals = [v for k, v in d.items() if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None
    return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def parse_retrain_dir_name(name: str) -> Optional[Dict]:
    """Extract task, seed from a canonical retrain directory name.

    Canonical short names look like:
        rtr__<task>__dolly__l32_3b__lr2e-5__ep4__lora-r128a512d01__s<seed>__<tag>
    """
    parts = name.split("__")
    info = {}
    for part in parts:
        if part in ("tydiqa", "mmlu", "bbh"):
            info["task"] = part
        elif part.startswith("s") and part[1:].isdigit():
            info["seed"] = int(part[1:])
    return info if "task" in info and "seed" in info else None


def find_metrics_in_run(run_dir: Path, task: str) -> Optional[float]:
    """Find metrics.json in a retrain run.

    Preferred location is the run root (`run_dir/eval/<task>/metrics.json`) so
    metrics survive checkpoint cleanup. Older runs may only have checkpoint-local
    eval outputs, so fall back to the last checkpoint.
    """
    for metrics_path in (
        run_dir / "eval" / task / "metrics.json",
        run_dir / "eval_results" / task / "metrics.json",
    ):
        val = extract_metric(metrics_path, task)
        if val is not None:
            return val

    ckpt_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(re.search(r"\d+", p.name).group()) if re.search(r"\d+", p.name) else 0,
    )
    if not ckpt_dirs:
        return None
    for ckpt in reversed(ckpt_dirs):
        for eval_mode in ("eval", "eval_results"):
            metrics_path = ckpt / eval_mode / task / "metrics.json"
            val = extract_metric(metrics_path, task)
            if val is not None:
                return val
    return None


def discover_depth_from_run(run_dir: Path, depth_id_pattern: str) -> bool:
    """Check if any JSON config in the run references a depth_NN pattern."""
    for json_file in run_dir.rglob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if depth_id_pattern in content:
                return True
        except Exception:
            pass
    # Also check trainer_state for depth_NN
    for ckpt in run_dir.iterdir():
        if not ckpt.is_dir():
            continue
        ts = ckpt / "trainer_state.json"
        if ts.is_file():
            try:
                if depth_id_pattern in ts.read_text(encoding="utf-8"):
                    return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# Main discovery: structured from study_config.json
# ---------------------------------------------------------------------------


def discover_from_config(study_root: str, scratch: str) -> List[Dict]:
    """Use study_config.json to reconstruct expected paths and collect results.

    The study uses warmup-seed variation:
      depth_selected/seed_XX/task/depth_DD/task/top_p0.05.jsonl
    All retrain jobs use a fixed retrain_seed.
    """
    config_path = Path(study_root) / "manifests" / "study_config.json"
    if not config_path.is_file():
        print(f"WARNING: study_config.json not found at {config_path}", file=sys.stderr)
        return []

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = cfg["tasks"]
    depths = cfg["depths"]
    study_tag = cfg["study_tag"]
    top_pct = cfg["top_percentage"]

    # Handle both old (train_seeds) and new (warmup_seeds + retrain_seed) config formats
    warmup_seeds = cfg.get("warmup_seeds", cfg.get("train_seeds", []))
    retrain_seed = cfg.get("retrain_seed")

    results = []
    out_root = Path(scratch) / "out"

    # Pre-scan all matching retrain dirs once
    matching_dirs = []
    if out_root.is_dir():
        for d in out_root.iterdir():
            if d.is_dir() and study_tag in d.name:
                matching_dirs.append(d)

    for task in tasks:
        for depth in depths:
            depth_id = f"depth_{depth:02d}"

            for wseed in warmup_seeds:
                wseed_id = f"seed_{wseed:02d}"
                sel_file = (
                    Path(study_root) / "depth_selected" / wseed_id / task / depth_id / task / f"top_p{top_pct}.jsonl"
                )

                metric_val = None
                run_dir_found = None

                for candidate in matching_dirs:
                    info = parse_retrain_dir_name(candidate.name)
                    if not info or info.get("task") != task:
                        continue
                    # If retrain_seed is known, match on it
                    if retrain_seed is not None and info.get("seed") != retrain_seed:
                        continue

                    # Match depth: check if this run references the depth_NN pattern
                    # and the seed_XX pattern in its config files
                    if discover_depth_from_run(candidate, depth_id) and (
                        discover_depth_from_run(candidate, wseed_id) or
                        # Fallback: if only one warmup seed, don't require seed match
                        len(warmup_seeds) == 1
                    ):
                        metric_val = find_metrics_in_run(candidate, task)
                        run_dir_found = str(candidate)
                        break

                results.append({
                    "task": task,
                    "depth": depth,
                    "warmup_seed": wseed,
                    "metric_name": METRIC_NAMES.get(task, "score"),
                    "metric_value": metric_val,
                    "run_dir": run_dir_found,
                    "selected_file": str(sel_file) if sel_file.is_file() else None,
                })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Compute mean +/- std per (task, depth) across warmup seeds."""
    grouped = defaultdict(list)
    for r in results:
        if r["metric_value"] is not None:
            grouped[(r["task"], r["depth"])].append(r["metric_value"])

    agg_rows = []
    task_best: Dict[str, Dict] = {}

    for (task, depth), vals in sorted(grouped.items()):
        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
        row = {
            "task": task,
            "depth": depth,
            "n_warmup_seeds": len(vals),
            "metric_name": METRIC_NAMES.get(task, "score"),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "values": [round(v, 6) for v in vals],
        }
        agg_rows.append(row)

        if task not in task_best or mean_val > task_best[task]["mean"]:
            task_best[task] = {
                "depth": depth,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "n_warmup_seeds": len(vals),
            }

    return agg_rows, task_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Summarize depth ablation results")
    ap.add_argument("--study-root", required=True, help="Path to the depth ablation study root")
    ap.add_argument("--scratch", required=True, help="Path to $SCRATCH (where out/ dirs live)")
    ap.add_argument("--output-csv", default=None, help="Path to write summary CSV")
    ap.add_argument("--output-json", default=None, help="Path to write summary JSON")
    ap.add_argument("--optimal-json", default=None, help="Path to write optimal_depths JSON")
    args = ap.parse_args()

    study_root = args.study_root
    scratch = args.scratch

    analysis_dir = Path(study_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_csv or str(analysis_dir / "summary.csv")
    json_path = args.output_json or str(analysis_dir / "summary.json")
    optimal_path = args.optimal_json or str(analysis_dir / "optimal_depths.json")

    print(f"study_root: {study_root}")
    print(f"scratch:    {scratch}")

    results = discover_from_config(study_root, scratch)
    if not results:
        print("WARNING: no results discovered. Check study_config.json and retrain dirs.", file=sys.stderr)

    found = sum(1 for r in results if r["metric_value"] is not None)
    missing = sum(1 for r in results if r["metric_value"] is None)
    print(f"\nDiscovered {found} results, {missing} missing")
    print()

    for r in sorted(results, key=lambda x: (x["task"], x["depth"], x["warmup_seed"])):
        val_str = f"{r['metric_value']:.4f}" if r["metric_value"] is not None else "MISSING"
        print(f"  {r['task']:8s} depth={r['depth']:2d} wseed={r['warmup_seed']:2d} {r['metric_name']}={val_str}")

    agg_rows, task_best = aggregate(results)

    print("\n--- Aggregated (mean +/- std across warmup seeds) ---")
    for row in agg_rows:
        print(
            f"  {row['task']:8s} depth={row['depth']:2d} "
            f"{row['metric_name']}={row['mean']:.4f} +/- {row['std']:.4f} "
            f"(n={row['n_warmup_seeds']})"
        )

    print("\n--- Optimal depth per task ---")
    for task, best in sorted(task_best.items()):
        print(f"  {task:8s} -> depth={best['depth']} ({METRIC_NAMES.get(task, 'score')}={best['mean']:.4f})")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "depth", "warmup_seed", "metric_name", "metric_value", "run_dir", "selected_file"],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["task"], x["depth"], x["warmup_seed"])):
            writer.writerow(r)
    print(f"\nWrote {csv_path}")

    # Write JSON
    summary = {
        "study_root": study_root,
        "per_run": results,
        "aggregated": agg_rows,
        "optimal_depths": task_best,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    with open(optimal_path, "w", encoding="utf-8") as f:
        json.dump(task_best, f, indent=2)
    print(f"Wrote {optimal_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
