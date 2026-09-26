#!/usr/bin/env python
# coding: utf-8
"""
Summarize relaunch weight computation outputs into a CSV and Markdown table.
"""
import argparse
import csv
import json
import os
from pathlib import Path


TASKS = {"tydiqa", "mmlu", "bbh"}


def _find_task(parts):
    for p in parts:
        if p in TASKS:
            return p
    return None


def _parse_k_eps(parts):
    k = None
    eps = None
    for p in parts:
        if p.startswith("k") and len(p) > 1:
            try:
                k = int(p[1:])
            except ValueError:
                pass
        if p.startswith("eps") and len(p) > 3:
            try:
                eps = float(p[3:])
            except ValueError:
                pass
    return k, eps


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_w4(root: Path):
    w4_map = {}
    for task in TASKS:
        cand = root / task / "w4" / f"w4_{task}.json"
        if cand.exists():
            data = _load_json(cand)
            if data:
                w4_map[task] = data
    return w4_map


def main():
    ap = argparse.ArgumentParser(description="Summarize relaunch weight comp results")
    ap.add_argument("--root", required=True, help="Root output dir (contains task/k/eps folders)")
    ap.add_argument("--out_csv", default=None, help="CSV output path")
    ap.add_argument("--out_md", default=None, help="Markdown output path")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv) if args.out_csv else root / "weight_comp_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "weight_comp_summary.md"

    w4_map = _load_w4(root)

    records = []
    for path in root.rglob("ckpt_*_relaunch_result.json"):
        data = _load_json(path)
        if not data:
            continue
        parts = path.parts
        task = _find_task(parts) or data.get("validation_task")
        k, eps = _parse_k_eps(parts)
        if k is None:
            k = data.get("k")
        if eps is None:
            eps = data.get("epsilon")

        rec = {
            "task": task,
            "k": k,
            "epsilon": eps,
            "metric": data.get("metric"),
            "denominator": data.get("denominator"),
            "avg_lr": data.get("avg_lr"),
            "grad_norm_sq": data.get("grad_norm_sq"),
            "baseline_L": data.get("baseline_L"),
            "L_prime": data.get("L_prime"),
            "L_prime_plus": data.get("L_prime_plus"),
            "L_prime_minus": data.get("L_prime_minus"),
            "output_dir": str(path.parent),
        }
        try:
            if rec["metric"] is not None and rec["avg_lr"] is not None:
                rec["metric_times_avg_lr"] = float(rec["metric"]) * float(rec["avg_lr"])
            else:
                rec["metric_times_avg_lr"] = None
        except (TypeError, ValueError):
            rec["metric_times_avg_lr"] = None

        if task in w4_map:
            w4 = w4_map[task]
            rec["w4"] = w4.get("w4")
            rec["w4_scaled_by_avg_lr"] = w4.get("w4_scaled_by_avg_lr")
            rec["w4_avg_lr"] = w4.get("avg_lr")
        records.append(rec)

    def sort_key(r):
        return (
            r.get("task") or "",
            r.get("k") if r.get("k") is not None else 999,
            r.get("epsilon") if r.get("epsilon") is not None else 999.0,
        )

    records = sorted(records, key=sort_key)

    fields = [
        "task",
        "k",
        "epsilon",
        "metric",
        "metric_times_avg_lr",
        "denominator",
        "avg_lr",
        "grad_norm_sq",
        "baseline_L",
        "L_prime",
        "L_prime_plus",
        "L_prime_minus",
        "w4",
        "w4_scaled_by_avg_lr",
        "w4_avg_lr",
        "output_dir",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(out_md, "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for r in records:
            row = []
            for k in fields:
                v = r.get(k)
                if isinstance(v, float):
                    row.append(f"{v:.6g}")
                else:
                    row.append("" if v is None else str(v))
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
