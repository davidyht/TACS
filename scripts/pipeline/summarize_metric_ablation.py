#!/usr/bin/env python3
"""Summarize metric ablation results (EXP-F).

Walks retrain output directories to collect eval metrics per
(task, scoring_method, warmup_seed), computes mean +/- std across seeds,
and ranks scoring methods per task.

Usage:
    python3 scripts/pipeline/summarize_metric_ablation.py \
        --study-root /path/to/selected_data_runs/metric_ablation_... \
        --scratch /path/to/scratch

Outputs:
    {study_root}/analysis/summary.csv
    {study_root}/analysis/summary.json
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRIC_NAMES = {"tydiqa": "f1", "mmlu": "accuracy", "bbh": "em"}


def extract_metric(metrics_path: Path, task: str) -> Optional[float]:
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


def parse_retrain_dir_name(name: str) -> Optional[Dict]:
    parts = name.split("__")
    info = {}
    for part in parts:
        if part in ("tydiqa", "mmlu", "bbh"):
            info["task"] = part
        elif part.startswith("s") and part[1:].isdigit():
            info["seed"] = int(part[1:])
    return info if "task" in info and "seed" in info else None


def find_metrics_in_run(run_dir: Path, task: str) -> Optional[float]:
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


def discover_from_config(study_root: str, scratch: str) -> List[Dict]:
    config_path = Path(study_root) / "manifests" / "study_config.json"
    if not config_path.is_file():
        print(f"WARNING: study_config.json not found at {config_path}", file=sys.stderr)
        return []

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = cfg["tasks"]
    metric_grid = cfg["metric_grid"]
    study_tag = cfg["study_tag"]
    warmup_seeds = cfg.get("warmup_seeds", [])
    retrain_seed = cfg.get("retrain_seed")

    results = []
    out_root = Path(scratch) / "out"

    matching_dirs = []
    if out_root.is_dir():
        for d in out_root.iterdir():
            if d.is_dir() and study_tag in d.name:
                matching_dirs.append(d)

    for task in tasks:
        for metric in metric_grid:
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

                    # Check if this run references the metric and seed
                    found_metric = False
                    found_seed = False
                    for json_file in candidate.rglob("*.json"):
                        try:
                            content = json_file.read_text(encoding="utf-8")
                            if metric in content:
                                found_metric = True
                            if wseed_id in content:
                                found_seed = True
                            if found_metric and found_seed:
                                break
                        except Exception:
                            pass

                    if found_metric and (found_seed or len(warmup_seeds) == 1):
                        metric_val = find_metrics_in_run(candidate, task)
                        run_dir_found = str(candidate)
                        break

                results.append({
                    "task": task,
                    "scoring_method": metric,
                    "warmup_seed": wseed,
                    "metric_name": METRIC_NAMES.get(task, "score"),
                    "metric_value": metric_val,
                    "run_dir": run_dir_found,
                })

    return results


def aggregate(results: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in results:
        if r["metric_value"] is not None:
            grouped[(r["task"], r["scoring_method"])].append(r["metric_value"])

    agg_rows = []
    for (task, method), vals in sorted(grouped.items()):
        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append({
            "task": task,
            "scoring_method": method,
            "n_warmup_seeds": len(vals),
            "metric_name": METRIC_NAMES.get(task, "score"),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "values": [round(v, 6) for v in vals],
        })

    return agg_rows


def main():
    ap = argparse.ArgumentParser(description="Summarize metric ablation results")
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    study_root = args.study_root
    scratch = args.scratch

    analysis_dir = Path(study_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_csv or str(analysis_dir / "summary.csv")
    json_path = args.output_json or str(analysis_dir / "summary.json")

    print(f"study_root: {study_root}")
    print(f"scratch:    {scratch}")

    results = discover_from_config(study_root, scratch)

    found = sum(1 for r in results if r["metric_value"] is not None)
    missing = sum(1 for r in results if r["metric_value"] is None)
    print(f"\nDiscovered {found} results, {missing} missing")

    for r in sorted(results, key=lambda x: (x["task"], x["scoring_method"], x["warmup_seed"])):
        val_str = f"{r['metric_value']:.4f}" if r["metric_value"] is not None else "MISSING"
        print(f"  {r['task']:8s} {r['scoring_method']:15s} wseed={r['warmup_seed']:2d} {r['metric_name']}={val_str}")

    agg_rows = aggregate(results)

    print("\n--- Aggregated (mean +/- std across warmup seeds) ---")
    for row in agg_rows:
        print(
            f"  {row['task']:8s} {row['scoring_method']:15s} "
            f"{row['metric_name']}={row['mean']:.4f} +/- {row['std']:.4f} "
            f"(n={row['n_warmup_seeds']})"
        )

    # Rank methods per task
    print("\n--- Ranking per task ---")
    task_methods = defaultdict(list)
    for row in agg_rows:
        task_methods[row["task"]].append(row)
    for task, methods in sorted(task_methods.items()):
        methods_sorted = sorted(methods, key=lambda x: x["mean"], reverse=True)
        print(f"  {task}:")
        for i, m in enumerate(methods_sorted, 1):
            print(f"    {i}. {m['scoring_method']:15s} {m['mean']:.4f} +/- {m['std']:.4f}")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "scoring_method", "warmup_seed", "metric_name", "metric_value", "run_dir"],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["task"], x["scoring_method"], x["warmup_seed"])):
            writer.writerow(r)
    print(f"\nWrote {csv_path}")

    # Write JSON
    summary = {
        "study_root": study_root,
        "per_run": results,
        "aggregated": agg_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
