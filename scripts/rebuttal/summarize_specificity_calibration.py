#!/usr/bin/env python3
"""Select the TACS warmup (learning rate, depth) per task by mean held-out specificity.

Inputs are alignment JSONs from alignment_calibration_scorer.py (rows with fold, lr, depth,
spec_train). For each task the cells (lr, depth) are averaged over folds; the pick maximizes the
mean, breaking ties toward the shorter depth and then the lower learning rate. The script refuses
duplicate (fold, lr, depth) rows and cells missing a fold, and flags picks on a grid edge.

  python3 scripts/rebuttal/summarize_specificity_calibration.py \
    --task-align tydiqa=a.json,b.json --task-align mmlu=c.json --task-align bbh=d.json \
    --expected-folds 3 --out-summary summary.json --out-raw-hp raw_hp.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


class CalibrationError(RuntimeError):
    pass


def cell_table(rows, expected_folds: int):
    seen, cells = set(), {}
    for r in rows:
        if r.get("spec_train") is None:
            raise CalibrationError(f"missing spec_train in row {r}")
        key = (int(r["fold"]), str(r["lr"]), int(r["depth"]))
        if key in seen:
            raise CalibrationError(f"duplicate row for fold/lr/depth {key}")
        seen.add(key)
        cells.setdefault((str(r["lr"]), int(r["depth"])), {})[int(r["fold"])] = float(r["spec_train"])
    table = {}
    for (lr, depth), by_fold in cells.items():
        if len(by_fold) != expected_folds:
            raise CalibrationError(f"cell lr={lr} depth={depth} has {len(by_fold)} folds, expected {expected_folds}")
        table[(lr, depth)] = {"mean": statistics.fmean(by_fold.values()), "folds": dict(sorted(by_fold.items()))}
    return table


def select(table):
    if not table:
        raise CalibrationError("empty calibration table")
    return max(table, key=lambda k: (table[k]["mean"], -k[1], -float(k[0])))


def check_grid(task: str, table: dict, lrs, depths) -> None:
    """Refuse a table whose (lr, depth) cells differ from the declared grid (e.g. a failed per-LR job)."""
    want = {(str(lr), int(d)) for lr in lrs for d in depths}
    missing, extra = sorted(want - set(table)), sorted(set(table) - want)
    if missing or extra:
        raise CalibrationError(f"{task}: grid mismatch; missing cells {missing}, unexpected cells {extra}")


def summarize(task_rows: dict, expected_folds: int, expect_grid: dict | None = None):
    summary, raw = {}, {}
    for task, rows in task_rows.items():
        table = cell_table(rows, expected_folds)
        if expect_grid and task in expect_grid:
            check_grid(task, table, *expect_grid[task])
        lr, depth = select(table)
        depths = sorted({d for _, d in table})
        summary[task] = {
            "pick": {"lr": lr, "depth": depth, "mean_specificity": table[(lr, depth)]["mean"]},
            "searched_depths": depths,
            "searched_lrs": sorted({l for l, _ in table}, key=float),
            "pick_on_grid_edge": depth in (depths[0], depths[-1]),
            "table": {f"{l}|{d}": v for (l, d), v in sorted(table.items(), key=lambda kv: (float(kv[0][0]), kv[0][1]))},
        }
        raw[task] = {"lr": lr, "depth": depth, "rank": 1, "alpha": 4,
                     "mean_specificity": table[(lr, depth)]["mean"], "n_folds": expected_folds,
                     "criterion": "mean_heldout_specificity"}
    return summary, raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pick (lr, depth) per task by mean held-out specificity.")
    ap.add_argument("--task-align", action="append", required=True, help="task=path[,path...]")
    ap.add_argument("--expected-folds", type=int, default=3)
    ap.add_argument("--expect-grid", action="append", default=[],
                    help="task=lr,lr,...:depth,depth,... ; refuse any missing or unexpected (lr, depth) cell")
    ap.add_argument("--out-summary", required=True, type=Path)
    ap.add_argument("--out-raw-hp", required=True, type=Path)
    args = ap.parse_args(argv)
    task_rows = {}
    for spec in args.task_align:
        task, _, paths = spec.partition("=")
        rows = task_rows.setdefault(task, [])
        for p in paths.split(","):
            rows += json.loads(Path(p).read_text(encoding="utf-8"))["rows"]
    expect_grid = {}
    for spec in args.expect_grid:
        task, _, grid = spec.partition("=")
        lrs, _, depths = grid.partition(":")
        expect_grid[task] = (lrs.split(","), [int(d) for d in depths.split(",")])
    summary, raw = summarize(task_rows, args.expected_folds, expect_grid)
    for out in (args.out_summary, args.out_raw_hp):
        if out.exists():
            raise CalibrationError(f"refusing to overwrite {out}")
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_raw_hp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({t: {**s["pick"], "edge": s["pick_on_grid_edge"], "depths": s["searched_depths"]}
                      for t, s in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
