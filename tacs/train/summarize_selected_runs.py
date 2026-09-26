#!/usr/bin/env python
# coding: utf-8
"""
Summarize evaluation metrics for selected-data retraining runs.
Scans for eval/*/metrics.json under a root of model directories.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path


def _load_json(path: Path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_metric(task: str, metrics: dict):
    if not isinstance(metrics, dict):
        return None, None
    if task == "mmlu":
        return "average_acc", metrics.get("average_acc")
    if task == "bbh":
        return "average_exact_match", metrics.get("average_exact_match")
    if task == "tydiqa":
        avg = metrics.get("average")
        if isinstance(avg, dict):
            return "average.f1", avg.get("f1")
        return "average.f1", metrics.get("f1")
    # Fallback: use first numeric field if any
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v
    return None, None


def _parse_model_name(model_dir: Path):
    name = model_dir.name
    if name.startswith("less_"):
        name = name[len("less_") :]
    percentage = None
    run_tag = name
    if "_p" in name:
        base, pval = name.rsplit("_p", 1)
        run_tag = base
        try:
            percentage = float(pval)
        except ValueError:
            percentage = None
    return run_tag, percentage


def _parse_run_tag(tag: str):
    info = {}
    parts = tag.split("_")
    if parts and parts[0] == "ds":
        if len(parts) >= 3:
            info["target_task"] = parts[1]
            info["weight_label"] = parts[2]
        for p in parts:
            if re.fullmatch(r"j\\d+", p):
                info["job_id"] = p[1:]
            if re.fullmatch(r"a\\d+", p):
                info["array_id"] = p[1:]
    return info


def main():
    ap = argparse.ArgumentParser(description="Summarize selected-data evaluation runs")
    ap.add_argument("--root", required=True, help="Root dir that contains model output dirs")
    ap.add_argument("--out_csv", default=None, help="CSV output path")
    ap.add_argument("--out_md", default=None, help="Markdown output path")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out_csv) if args.out_csv else root / "selected_data_eval_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "selected_data_eval_summary.md"

    records = []
    for metrics_path in root.rglob("eval/*/metrics.json"):
        task = metrics_path.parent.name
        model_dir = metrics_path.parent.parent
        metrics = _load_json(metrics_path)
        metric_name, metric = _extract_metric(task, metrics or {})

        run_tag, percentage = _parse_model_name(model_dir)
        info = _parse_run_tag(run_tag)

        rec = {
            "run_tag": run_tag,
            "model_dir": str(model_dir),
            "task": task,
            "metric_name": metric_name,
            "metric": metric,
            "percentage": percentage,
            "target_task": info.get("target_task"),
            "weight_label": info.get("weight_label"),
            "job_id": info.get("job_id"),
            "array_id": info.get("array_id"),
        }
        records.append(rec)

    def sort_key(r):
        return (
            r.get("task") or "",
            r.get("weight_label") or "",
            r.get("percentage") if r.get("percentage") is not None else 999.0,
            r.get("run_tag") or "",
        )

    records = sorted(records, key=sort_key)

    fields = [
        "task",
        "metric_name",
        "metric",
        "percentage",
        "weight_label",
        "target_task",
        "run_tag",
        "job_id",
        "array_id",
        "model_dir",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with out_md.open("w") as f:
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
