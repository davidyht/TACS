#!/usr/bin/env python3
"""Summarize rank ablation results (warmup-seed variation).

Walks retrain output directories to collect eval metrics per
(task, rank, warmup_seed), computes mean +/- std across warmup seeds,
and reports optimal rank per task.

Can import rank=1 results from a depth ablation study via --import-rank1-from.

Usage:
    python3 scripts/pipeline/summarize_rank_ablation.py \
        --study-root /path/to/selected_data_runs/rank_ablation_... \
        --scratch /path/to/scratch

    # With rank=1 import from depth ablation:
    python3 scripts/pipeline/summarize_rank_ablation.py \
        --study-root /path/to/rank_ablation_... \
        --scratch /path/to/scratch \
        --import-rank1-from /path/to/depth_ablation_...

Outputs:
    {study_root}/analysis/summary.csv
    {study_root}/analysis/summary.json
    {study_root}/analysis/optimal_ranks.json
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
# Metric extraction (mirrors summarize_depth_ablation.py)
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
    """Extract task, seed from a canonical retrain directory name."""
    parts = name.split("__")
    info = {}
    for part in parts:
        if part in ("tydiqa", "mmlu", "bbh"):
            info["task"] = part
        elif part.startswith("s") and part[1:].isdigit():
            info["seed"] = int(part[1:])
    return info if "task" in info and "seed" in info else None


def find_metrics_in_run(run_dir: Path, task: str) -> Optional[float]:
    """Find metrics.json in the last checkpoint of a retrain run."""
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


def discover_rank_from_run(run_dir: Path, pattern: str) -> bool:
    """Check if any JSON config in the run references a rank pattern."""
    for json_file in run_dir.rglob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if pattern in content:
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Import rank=1 from depth ablation
# ---------------------------------------------------------------------------


def import_rank1_from_depth(depth_study_root: str, scratch: str) -> List[Dict]:
    """Import rank=1 results from a depth ablation study.

    Reads the depth ablation's summary.json and extracts results at the
    optimal depth for each task (which corresponds to rank=1 results).
    """
    summary_path = Path(depth_study_root) / "analysis" / "summary.json"
    optimal_path = Path(depth_study_root) / "analysis" / "optimal_depths.json"

    if not summary_path.is_file():
        print(f"WARNING: depth ablation summary not found: {summary_path}", file=sys.stderr)
        return []

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    optimal = {}
    if optimal_path.is_file():
        optimal = json.loads(optimal_path.read_text(encoding="utf-8"))

    results = []
    per_run = summary.get("per_run", [])

    for r in per_run:
        task = r.get("task")
        depth = r.get("depth")
        wseed = r.get("warmup_seed")
        val = r.get("metric_value")

        if task is None or depth is None or wseed is None:
            continue

        # Only import results at the optimal depth for this task
        opt = optimal.get(task, {})
        opt_depth = opt.get("depth") if isinstance(opt, dict) else opt
        if opt_depth is not None and depth == opt_depth and val is not None:
            results.append({
                "task": task,
                "rank": 1,
                "warmup_seed": wseed,
                "metric_name": METRIC_NAMES.get(task, "score"),
                "metric_value": val,
                "run_dir": r.get("run_dir"),
                "source": "depth_ablation_import",
            })

    return results


# ---------------------------------------------------------------------------
# Main discovery: structured from study_config.json
# ---------------------------------------------------------------------------


def discover_from_config(study_root: str, scratch: str) -> List[Dict]:
    """Use study_config.json to reconstruct expected paths and collect results."""
    config_path = Path(study_root) / "manifests" / "study_config.json"
    if not config_path.is_file():
        print(f"WARNING: study_config.json not found at {config_path}", file=sys.stderr)
        return []

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = cfg["tasks"]
    rank_grid = cfg["rank_grid"]
    study_tag = cfg["study_tag"]
    warmup_seeds = cfg.get("warmup_seeds", [])
    retrain_seed = cfg.get("retrain_seed")
    import_rank1 = cfg.get("import_rank1_from")

    results = []
    out_root = Path(scratch) / "out"

    # Pre-scan matching retrain dirs
    matching_dirs = []
    if out_root.is_dir():
        for d in out_root.iterdir():
            if d.is_dir() and study_tag in d.name:
                matching_dirs.append(d)

    for task in tasks:
        for rank in rank_grid:
            rank_id = f"rank_{rank:03d}"

            # Skip rank=1 if imported
            if rank == 1 and import_rank1:
                continue

            for wseed in warmup_seeds:
                wseed_id = f"seed_{wseed:02d}"

                metric_val = None
                run_dir_found = None

                for candidate in matching_dirs:
                    info = parse_retrain_dir_name(candidate.name)
                    if not info or info.get("task") != task:
                        continue
                    if retrain_seed is not None and info.get("seed") != retrain_seed:
                        continue

                    # Match rank and seed patterns in run configs
                    if discover_rank_from_run(candidate, rank_id) and (
                        discover_rank_from_run(candidate, wseed_id) or
                        len(warmup_seeds) == 1
                    ):
                        metric_val = find_metrics_in_run(candidate, task)
                        run_dir_found = str(candidate)
                        break

                results.append({
                    "task": task,
                    "rank": rank,
                    "warmup_seed": wseed,
                    "metric_name": METRIC_NAMES.get(task, "score"),
                    "metric_value": metric_val,
                    "run_dir": run_dir_found,
                    "source": "rank_ablation",
                })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Compute mean +/- std per (task, rank) across warmup seeds."""
    grouped = defaultdict(list)
    for r in results:
        if r["metric_value"] is not None:
            grouped[(r["task"], r["rank"])].append(r["metric_value"])

    agg_rows = []
    task_best: Dict[str, Dict] = {}

    for (task, rank), vals in sorted(grouped.items()):
        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
        row = {
            "task": task,
            "rank": rank,
            "n_warmup_seeds": len(vals),
            "metric_name": METRIC_NAMES.get(task, "score"),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "values": [round(v, 6) for v in vals],
        }
        agg_rows.append(row)

        if task not in task_best or mean_val > task_best[task]["mean"]:
            task_best[task] = {
                "rank": rank,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "n_warmup_seeds": len(vals),
            }

    return agg_rows, task_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Summarize rank ablation results")
    ap.add_argument("--study-root", required=True, help="Path to the rank ablation study root")
    ap.add_argument("--scratch", required=True, help="Path to $SCRATCH (where out/ dirs live)")
    ap.add_argument("--import-rank1-from", default=None, help="Path to depth ablation study root (for rank=1 import)")
    ap.add_argument("--output-csv", default=None, help="Path to write summary CSV")
    ap.add_argument("--output-json", default=None, help="Path to write summary JSON")
    ap.add_argument("--optimal-json", default=None, help="Path to write optimal_ranks JSON")
    args = ap.parse_args()

    study_root = args.study_root
    scratch = args.scratch

    analysis_dir = Path(study_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_csv or str(analysis_dir / "summary.csv")
    json_path = args.output_json or str(analysis_dir / "summary.json")
    optimal_path = args.optimal_json or str(analysis_dir / "optimal_ranks.json")

    print(f"study_root:        {study_root}")
    print(f"scratch:           {scratch}")
    print(f"import_rank1_from: {args.import_rank1_from or '<none>'}")

    # Discover results from rank ablation runs
    results = discover_from_config(study_root, scratch)

    # Import rank=1 from depth ablation if specified
    if args.import_rank1_from:
        rank1_results = import_rank1_from_depth(args.import_rank1_from, scratch)
        print(f"\nImported {len(rank1_results)} rank=1 results from depth ablation")
        results.extend(rank1_results)

    if not results:
        print("WARNING: no results discovered.", file=sys.stderr)

    found = sum(1 for r in results if r["metric_value"] is not None)
    missing = sum(1 for r in results if r["metric_value"] is None)
    print(f"\nDiscovered {found} results, {missing} missing")
    print()

    for r in sorted(results, key=lambda x: (x["task"], x["rank"], x["warmup_seed"])):
        val_str = f"{r['metric_value']:.4f}" if r["metric_value"] is not None else "MISSING"
        src_tag = " [imported]" if r.get("source") == "depth_ablation_import" else ""
        print(f"  {r['task']:8s} rank={r['rank']:3d} wseed={r['warmup_seed']:2d} {r['metric_name']}={val_str}{src_tag}")

    agg_rows, task_best = aggregate(results)

    print("\n--- Aggregated (mean +/- std across warmup seeds) ---")
    for row in agg_rows:
        print(
            f"  {row['task']:8s} rank={row['rank']:3d} "
            f"{row['metric_name']}={row['mean']:.4f} +/- {row['std']:.4f} "
            f"(n={row['n_warmup_seeds']})"
        )

    print("\n--- Optimal rank per task ---")
    for task, best in sorted(task_best.items()):
        print(f"  {task:8s} -> rank={best['rank']} ({METRIC_NAMES.get(task, 'score')}={best['mean']:.4f})")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "rank", "warmup_seed", "metric_name", "metric_value", "run_dir", "source"],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["task"], x["rank"], x["warmup_seed"])):
            writer.writerow(r)
    print(f"\nWrote {csv_path}")

    # Write JSON
    summary = {
        "study_root": study_root,
        "import_rank1_from": args.import_rank1_from,
        "per_run": results,
        "aggregated": agg_rows,
        "optimal_ranks": task_best,
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
