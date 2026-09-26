#!/usr/bin/env python3
"""Summarize val-set size (n_shot) ablation results (EXP-E).

Walks retrain output directories to collect eval metrics per
(n_shot, val_subset_seed, warmup_seed), computes mean +/- std,
and reports F1 vs n_shot with error bars.

Usage:
    python3 scripts/pipeline/summarize_val_size_ablation.py \
        --study-root /path/to/selected_data_runs/valsize_ablation_... \
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
        value = avg.get("f1")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if task == "mmlu":
        value = d.get("average_acc")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if task == "bbh":
        value = d.get("average_exact_match")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return None


def parse_retrain_dir_name(name: str) -> Optional[Dict]:
    info: Dict = {}
    task_match = re.search(r"(?:^|__)t-(tydiqa|mmlu|bbh)(?:__|$)", name)
    if task_match:
        info["task"] = task_match.group(1)
    seed_match = re.search(r"cfg-[^_]*s(\d+)(?:__|_|$)", name)
    if seed_match:
        info["seed"] = int(seed_match.group(1))
    return info if "task" in info and "seed" in info else None


def find_metrics_in_run(run_dir: Path, task: str) -> Optional[float]:
    # Rebuttal-era val-size runs use an a priori fixed checkpoint policy:
    # evaluate the final model saved at the run root. Check this stable path
    # before legacy checkpoint-local metrics.
    for eval_mode in ("eval", "eval_results"):
        metrics_path = run_dir / eval_mode / task / "metrics.json"
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


def discover_from_config(study_root: str, scratch: str) -> List[Dict]:
    config_path = Path(study_root) / "manifests" / "study_config.json"
    if not config_path.is_file():
        print(f"WARNING: study_config.json not found at {config_path}", file=sys.stderr)
        return []

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    task = cfg["task"]
    n_shot_grid = cfg["n_shot_grid"]
    val_subset_seeds = cfg["val_subset_seeds"]
    warmup_seeds = cfg["warmup_seeds"]
    study_tag = cfg["study_tag"]
    retrain_seed = cfg.get("retrain_seed")

    results = []
    out_root = Path(scratch) / "out"
    jobs_path = Path(study_root) / "manifests" / "jobs.tsv"

    # New runs record the exact canonical run directory in the notes column.
    # This is authoritative and avoids reverse-engineering truncated/hash-based
    # canonical names.
    manifested_runs: Dict[Tuple[int, int, int], Path] = {}
    if jobs_path.is_file():
        try:
            with jobs_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row.get("stage") not in ("retrain", "eval+cleanup"):
                        continue
                    run_dir = (row.get("notes") or "").strip()
                    if not run_dir:
                        continue
                    key = (
                        int(row["n_shot"]),
                        int(row["val_seed"]),
                        int(row["warmup_seed"]),
                    )
                    manifested_runs[key] = Path(run_dir)
        except (OSError, ValueError, KeyError) as exc:
            print(f"WARNING: unable to parse {jobs_path}: {exc}", file=sys.stderr)

    matching_dirs = []
    if out_root.is_dir():
        for d in out_root.iterdir():
            if d.is_dir() and study_tag in d.name:
                matching_dirs.append(d)

    for nshot in n_shot_grid:
        for vseed in val_subset_seeds:
            for wseed in warmup_seeds:
                combo_id = f"nshot{nshot}_vs{vseed}_ws{wseed:02d}"

                metric_val = None
                run_dir_found = None
                manifested = manifested_runs.get((nshot, vseed, wseed))
                if manifested is not None and manifested.is_dir():
                    metric_val = find_metrics_in_run(manifested, task)
                    run_dir_found = str(manifested)

                if manifested is None:
                    for candidate in matching_dirs:
                        info = parse_retrain_dir_name(candidate.name)
                        if not info or info.get("task") != task:
                            continue
                        if retrain_seed is not None and info.get("seed") != retrain_seed:
                            continue

                        # Legacy fallback: check whether run metadata references
                        # this combination.
                        found = False
                        for json_file in candidate.rglob("*.json"):
                            try:
                                content = json_file.read_text(encoding="utf-8")
                                if combo_id in content or f"nshot_{nshot}_vseed_{vseed}" in content:
                                    found = True
                                    break
                            except Exception:
                                pass

                        if found:
                            metric_val = find_metrics_in_run(candidate, task)
                            run_dir_found = str(candidate)
                            break

                results.append({
                    "n_shot": nshot,
                    "n_examples": n_examples_for_task(task, nshot),
                    "val_subset_seed": vseed,
                    "warmup_seed": wseed,
                    "metric_name": METRIC_NAMES.get(task, "score"),
                    "metric_value": metric_val,
                    "run_dir": run_dir_found,
                })

    return results


def aggregate(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Compute aggregations:
    1. Per (n_shot, val_subset_seed): mean +/- std across warmup seeds
    2. Per n_shot: mean +/- std across all seeds (val_subset + warmup)
    """
    # Per (n_shot, val_subset_seed)
    grouped_fine = defaultdict(list)
    for r in results:
        if r["metric_value"] is not None:
            grouped_fine[(r["n_shot"], r["val_subset_seed"])].append(r["metric_value"])

    fine_rows = []
    for (nshot, vseed), vals in sorted(grouped_fine.items()):
        fine_rows.append({
            "n_shot": nshot,
            "n_examples": n_examples_for_task_from_results(vals=vals, results=results, nshot=nshot),
            "val_subset_seed": vseed,
            "n_warmup_seeds": len(vals),
            "mean": round(statistics.mean(vals), 6),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 6),
            "values": [round(v, 6) for v in vals],
        })

    # Per n_shot (all seeds pooled)
    grouped_coarse = defaultdict(list)
    for r in results:
        if r["metric_value"] is not None:
            grouped_coarse[r["n_shot"]].append(r["metric_value"])

    coarse_rows = []
    for nshot, vals in sorted(grouped_coarse.items()):
        coarse_rows.append({
            "n_shot": nshot,
            "n_examples": n_examples_for_task_from_results(vals=vals, results=results, nshot=nshot),
            "n_total_runs": len(vals),
            "mean": round(statistics.mean(vals), 6),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 6),
            "values": [round(v, 6) for v in vals],
        })

    return fine_rows, coarse_rows


def n_examples_for_task(task: str, nshot: int) -> int:
    if task == "tydiqa":
        return int(nshot) * 9
    if task == "mmlu":
        return int(nshot) * 57
    if task == "bbh":
        return 81
    return int(nshot)


def n_examples_for_task_from_results(vals: List[float], results: List[Dict], nshot: int) -> int:
    del vals
    for row in results:
        if row["n_shot"] == nshot:
            return int(row["n_examples"])
    return int(nshot)


def main():
    ap = argparse.ArgumentParser(description="Summarize val-size ablation results")
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-json", default=None)
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write partial outputs and exit zero even when planned runs are missing.",
    )
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
    task = json.loads((Path(study_root) / "manifests" / "study_config.json").read_text(encoding="utf-8"))["task"]
    metric_name = METRIC_NAMES.get(task, "score")
    print(f"\nDiscovered {found} results, {missing} missing")

    for r in sorted(results, key=lambda x: (x["n_shot"], x["val_subset_seed"], x["warmup_seed"])):
        val_str = f"{r['metric_value']:.4f}" if r["metric_value"] is not None else "MISSING"
        print(
            f"  n_shot={r['n_shot']} ({r['n_examples']:2d} ex) "
            f"vseed={r['val_subset_seed']:2d} wseed={r['warmup_seed']:2d} "
            f"{metric_name}={val_str}"
        )

    fine_rows, coarse_rows = aggregate(results)

    print("\n--- Per (n_shot, val_subset_seed) ---")
    for row in fine_rows:
        print(
            f"  n_shot={row['n_shot']} ({row['n_examples']:2d} ex) vseed={row['val_subset_seed']:2d} "
            f"{metric_name}={row['mean']:.4f} +/- {row['std']:.4f} (n={row['n_warmup_seeds']})"
        )

    print("\n--- Per n_shot (all seeds pooled) ---")
    for row in coarse_rows:
        print(
            f"  n_shot={row['n_shot']} ({row['n_examples']:2d} ex) "
            f"{metric_name}={row['mean']:.4f} +/- {row['std']:.4f} (n={row['n_total_runs']})"
        )

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_shot", "n_examples", "val_subset_seed", "warmup_seed",
                         "metric_name", "metric_value", "run_dir"],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["n_shot"], x["val_subset_seed"], x["warmup_seed"])):
            writer.writerow(r)
    print(f"\nWrote {csv_path}")

    # Write JSON
    summary = {
        "study_root": study_root,
        "per_run": results,
        "per_nshot_vseed": fine_rows,
        "per_nshot": coarse_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    if missing and not args.allow_missing:
        print(
            f"ERROR: {missing} planned result(s) are missing; "
            "rerun with --allow-missing only for intentional partial inspection.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
