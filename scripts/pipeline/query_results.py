#!/usr/bin/env python3
"""
query_results.py
----------------
Read analysis/results_store.csv and print paper table values.

Examples:
    # Single cell
    python scripts/pipeline/query_results.py \
        --model llama32_3b --method normlg --source dolly --task tydiqa

    # Full paper table for one model
    python scripts/pipeline/query_results.py --table --model llama32_3b

    # Both models
    python scripts/pipeline/query_results.py --table

    # Filter to a specific run set
    python scripts/pipeline/query_results.py --table \
        --selection-tag-contains less_resubmit

    # Dump best-only CSV
    python scripts/pipeline/query_results.py --table --csv > results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


STORE_DEFAULT = Path(__file__).parent.parent.parent / "analysis" / "results_store.csv"

TASKS   = ["tydiqa", "bbh", "mmlu"]
METHODS = ["random", "less", "tov", "normlg"]
SOURCES = ["dolly", "flan_v2", "cot", "oasst1"]
MODELS  = ["llama32_3b", "qwen3_8b", "llama2_7b"]

FMT = {"tydiqa": ".2f", "bbh": ".4f", "mmlu": ".4f"}


# ── load ─────────────────────────────────────────────────────────────────────

def load_store(store_path: Path) -> list[dict]:
    if not store_path.exists():
        sys.exit(f"Store not found: {store_path}\nRun backfill_results_store.py first.")
    with store_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in ("ckpt_value", "best_value", "ckpt_id", "best_ckpt_id"):
            try:
                r[col] = float(r[col])
            except (ValueError, TypeError):
                r[col] = None
    return rows


def filter_rows(
    rows: list[dict],
    model: Optional[str] = None,
    method: Optional[str] = None,
    source: Optional[str] = None,
    task: Optional[str] = None,
    best_only: bool = True,
    selection_tag_contains: Optional[str] = None,
) -> list[dict]:
    out = rows
    if best_only:
        out = [r for r in out if r.get("is_best") == "1"]
    if model:
        out = [r for r in out if r["model"] == model]
    if method:
        out = [r for r in out if r["method"] == method]
    if source:
        out = [r for r in out if r["source"] == source]
    if task:
        out = [r for r in out if r["task"] == task]
    if selection_tag_contains:
        out = [r for r in out if selection_tag_contains in r.get("selection_tag", "")]
    return out


# ── query single cell ─────────────────────────────────────────────────────────

def get_cell(
    rows: list[dict],
    model: str,
    method: str,
    source: str,
    task: str,
    selection_tag_contains: Optional[str] = None,
) -> dict:
    matches = filter_rows(rows, model=model, method=method, source=source,
                          task=task, selection_tag_contains=selection_tag_contains)
    if not matches:
        return {}
    vals = [r["best_value"] for r in matches if r["best_value"] is not None]
    return {
        "mean_best": sum(vals) / len(vals) if vals else None,
        "max_best":  max(vals) if vals else None,
        "n_runs":    len(matches),
        "runs":      [r["run_name"][:60] for r in matches],
        "ckpts":     [r["best_ckpt_id"] for r in matches],
    }


# ── paper table ───────────────────────────────────────────────────────────────

def _cell_val(rows, model, method, source, task, tag_filter) -> Optional[float]:
    cell = get_cell(rows, model, method, source, task, tag_filter)
    return cell.get("max_best")


def _fmt(val, task) -> str:
    if val is None:
        return "—"
    f = FMT.get(task, ".4f")
    return format(val, f)


def print_table(
    rows: list[dict],
    models: list[str],
    tag_filter: Optional[str],
    as_csv: bool = False,
) -> None:
    if as_csv:
        w = csv.writer(sys.stdout)
        w.writerow(["model", "task", "source", "random", "less", "normlg"])
        for model in models:
            for task in TASKS:
                for src in SOURCES:
                    vals = [_cell_val(rows, model, m, src, task, tag_filter)
                            for m in METHODS]
                    w.writerow([model, task, src] + [_fmt(v, task) for v in vals])
        return

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")
        for task in TASKS:
            print(f"\n  {task.upper()}")
            print(f"  {'Source':<12}  {'Random':>8}  {'LESS':>8}  {'NormLG':>8}")
            print(f"  {'-'*44}")
            for src in SOURCES:
                vals = [_cell_val(rows, model, m, src, task, tag_filter)
                        for m in METHODS]
                row_str = "  {:<12}  {:>8}  {:>8}  {:>8}".format(
                    src, *[_fmt(v, task) for v in vals]
                )
                # bold the best (non-random) value
                best_nonrand = max(
                    (v for v in vals[1:] if v is not None), default=None
                )
                if best_nonrand is not None and best_nonrand == vals[2]:
                    row_str = row_str.rstrip() + "  ◀"
                print(row_str)


# ── diff table (compare two run sets) ────────────────────────────────────────

def print_diff_table(
    rows: list[dict],
    model: str,
    tag_a: str,
    tag_b: str,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    print(f"\nDiff: {label_a} vs {label_b}  (model={model})")
    print(f"  {'task':<8} {'src':<12} {'method':<8} "
          f"  {label_a:>8}  {label_b:>8}  {'Δ':>8}")
    print(f"  {'-'*55}")
    for task in TASKS:
        for src in SOURCES:
            for method in METHODS:
                va = _cell_val(rows, model, method, src, task, tag_a)
                vb = _cell_val(rows, model, method, src, task, tag_b)
                if va is None and vb is None:
                    continue
                delta = (vb - va) if (va is not None and vb is not None) else None
                sign = ("+" if delta >= 0 else "") if delta is not None else ""
                print(f"  {task:<8} {src:<12} {method:<8}  "
                      f"{_fmt(va, task):>8}  {_fmt(vb, task):>8}  "
                      f"{sign + _fmt(delta, task) if delta is not None else '—':>8}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(STORE_DEFAULT))

    # filters
    ap.add_argument("--model",   choices=MODELS)
    ap.add_argument("--method",  choices=METHODS)
    ap.add_argument("--source",  choices=SOURCES)
    ap.add_argument("--task",    choices=TASKS)
    ap.add_argument("--selection-tag-contains", default=None,
                    help="only include rows whose selection_tag contains this string")

    # output modes
    ap.add_argument("--table",     action="store_true",
                    help="print full paper table")
    ap.add_argument("--diff",      nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="compare two selection-tag subsets")
    ap.add_argument("--csv",       action="store_true",
                    help="output as CSV (with --table)")
    ap.add_argument("--best-only", action="store_true", default=True)
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="show all checkpoints, not just best")
    args = ap.parse_args()

    rows = load_store(Path(args.store))
    tag_filter = args.selection_tag_contains
    best_only = not args.all_checkpoints

    if args.diff:
        model = args.model or MODELS[0]
        tag_a, tag_b = args.diff
        print_diff_table(rows, model, tag_a, tag_b, tag_a, tag_b)
        return

    if args.table:
        models = [args.model] if args.model else MODELS
        print_table(rows, models, tag_filter, as_csv=args.csv)
        return

    # single-cell query
    if args.model and args.method and args.source and args.task:
        cell = get_cell(rows, args.model, args.method, args.source, args.task, tag_filter)
        if not cell:
            print("No matching rows found.")
            return
        task = args.task
        print(f"model={args.model}  method={args.method}  "
              f"source={args.source}  task={task}")
        print(f"  max_best = {_fmt(cell['max_best'], task)}")
        print(f"  mean_best= {_fmt(cell['mean_best'], task)}")
        print(f"  n_runs   = {cell['n_runs']}")
        for r, c in zip(cell["runs"], cell["ckpts"]):
            print(f"    {r}  ckpt={c}")
        return

    # default: print everything that matches the given filters
    filtered = filter_rows(rows, model=args.model, method=args.method,
                           source=args.source, task=args.task,
                           best_only=best_only,
                           selection_tag_contains=tag_filter)
    if not filtered:
        print("No matching rows. Try --table to see the full store.")
        return
    w = csv.DictWriter(sys.stdout, fieldnames=[
        "model", "method", "source", "task", "best_value", "best_ckpt_id",
        "run_name", "ingest_source",
    ])
    w.writeheader()
    w.writerows(filtered)


if __name__ == "__main__":
    main()
