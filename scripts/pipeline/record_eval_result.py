#!/usr/bin/env python3
"""
record_eval_result.py
---------------------
Append one (run, checkpoint, task) result row to the canonical results store.

Called at the end of retrain_eval_array.sbatch after eval completes:

    python scripts/pipeline/record_eval_result.py \
        --metrics-path "$METRICS_PATH" \
        --run-dir      "$RUN_DIR"      \
        --ckpt-dir     "$CKPT_DIR"     \
        --task         "$TARGET_TASK"  \
        --store        analysis/results_store.csv

Uses fcntl advisory locking so multiple Slurm array tasks can write safely.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── metric extraction ────────────────────────────────────────────────────────

STORE_COLUMNS = [
    "model", "method", "source", "task", "metric_name",
    "ckpt_id", "ckpt_value", "best_value", "best_ckpt_id", "is_best",
    "selection_tag", "run_name", "run_dir",
    "train_file", "metrics_path", "ingest_source", "ingest_ts", "notes",
]


def extract_metric(metrics_path: Path, task: str) -> Optional[float]:
    """Return the primary scalar metric from a metrics.json file."""
    try:
        d = json.loads(metrics_path.read_text())
    except Exception as exc:
        print(f"[warn] cannot read {metrics_path}: {exc}", file=sys.stderr)
        return None

    if task == "tydiqa":
        avg = d.get("average", {})
        value = avg.get("f1")
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    elif task == "mmlu":
        return d.get("average_acc")
    elif task == "bbh":
        if isinstance(d.get("average_exact_match"), (int, float)):
            return d["average_exact_match"]
        vals = [v for k, v in d.items() if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None
    return None


def metric_name_for(task: str) -> str:
    return {"tydiqa": "f1", "mmlu": "accuracy", "bbh": "em"}.get(task, "score")


# ── run metadata extraction ──────────────────────────────────────────────────

METHOD_HINTS = [
    ("tov_mainfix", "tov"), ("tov_topk", "tov"), ("tov_native", "tov"),
    ("tov", "tov"),
    ("normlg", "normlg"), ("norm_loss_gap", "normlg"),
    ("norm_final_drop", "normlg"),
    ("less_resubmit", "less"), ("less_baseline", "less"),
    ("less_orig4src", "less"), ("less_source_samples", "random"),
    ("less_warmup_l32_3b", "random"), ("less_warmup", "random"),
    ("randomp005", "random"), ("random_p005", "random"),
    ("sample_p0.05", "random"), ("random_selection", "random"),
]
SOURCE_HINTS = [
    ("flan_v2", "flan_v2"), ("flan", "flan_v2"),
    ("dolly", "dolly"), ("oasst1", "oasst1"), ("oasst", "oasst1"),
    ("cot", "cot"),
]


def _match_hints(text: str, hints: list) -> Optional[str]:
    for token, label in hints:
        if token in text:
            return label
    return None


def parse_run_metadata(run_dir: Path) -> dict:
    """Extract model/method/source from train.log if present, else run name."""
    meta: dict = {
        "model": "unknown", "method": "unknown", "source": "unknown",
        "selection_tag": "unknown", "train_file": "",
    }

    train_log = run_dir / "train.log"
    if train_log.exists():
        try:
            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from scripts.pipeline.retrain_naming import (
                model_key_from_path, parse_selected_dataset,
                parse_retrain_train_log,
            )
            log_meta = parse_retrain_train_log(train_log)
            model_raw = log_meta.get("model_name_or_path", "")
            meta["model"] = model_key_from_path(model_raw) if model_raw else "unknown"
            if meta["model"] == "l32_3b":
                meta["model"] = "llama32_3b"
            train_files = log_meta.get("train_files", [])
            if train_files:
                tf = train_files[0]
                meta["train_file"] = tf
                ds = parse_selected_dataset(tf)
                meta["selection_tag"] = ds.get("selection_tag", "unknown")
                # derive method/source from selection_tag + train_file
                combined = tf + " " + meta["selection_tag"]
                meta["method"] = _match_hints(combined, METHOD_HINTS) or "unknown"
                meta["source"] = _match_hints(combined, SOURCE_HINTS) or "unknown"
        except Exception as exc:
            print(f"[warn] retrain_naming import failed: {exc}", file=sys.stderr)

    # fallback: infer from run directory name
    run_name = run_dir.name
    if meta["method"] == "unknown":
        meta["method"] = _match_hints(run_name, METHOD_HINTS) or "unknown"
    if meta["source"] == "unknown":
        meta["source"] = _match_hints(run_name, SOURCE_HINTS) or "unknown"
    if meta["model"] == "unknown":
        for token, label in [("l32_3b", "llama32_3b"), ("llama32", "llama32_3b"),
                              ("llama-3.2", "llama32_3b"), ("qwen3_8b", "qwen3_8b"),
                              ("qwen3-8b", "qwen3_8b"), ("qwen", "qwen3_8b"),
                              ("llama2_7b", "llama2_7b"), ("llama-2-7b", "llama2_7b")]:
            if token in run_name.lower():
                meta["model"] = label
                break
    if meta["selection_tag"] == "unknown":
        m = re.search(r"__ds-([^_][^_]*(?:_[^_][^_]*)*)__", run_name)
        if m:
            meta["selection_tag"] = m.group(1)

    return meta


# ── store read/write ─────────────────────────────────────────────────────────

def _read_store(store_path: Path) -> list[dict]:
    if not store_path.exists():
        return []
    with store_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_store(store_path: Path, rows: list[dict]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with store_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STORE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _recompute_best(rows: list[dict]) -> list[dict]:
    """Recompute best_value/best_ckpt_id/is_best within each (run_name, task) group."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["run_name"], r["task"])].append(i)

    for idxs in groups.values():
        vals = []
        for i in idxs:
            try:
                vals.append((float(rows[i]["ckpt_value"]), i))
            except (TypeError, ValueError):
                pass
        if not vals:
            continue
        best_val, best_i = max(vals)
        best_ckpt = rows[best_i]["ckpt_id"]
        for i in idxs:
            rows[i]["best_value"] = best_val
            rows[i]["best_ckpt_id"] = best_ckpt
            rows[i]["is_best"] = "1" if i == best_i else "0"
    return rows


def record(
    metrics_path: Path,
    run_dir: Path,
    ckpt_dir: Path,
    task: str,
    store_path: Path,
    ingest_source: str = "checkpoint_json",
    notes: str = "",
) -> None:
    ckpt_value = extract_metric(metrics_path, task)
    if ckpt_value is None:
        print(f"[warn] could not extract metric from {metrics_path}", file=sys.stderr)
        return

    ckpt_id_m = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
    ckpt_id = int(ckpt_id_m.group(1)) if ckpt_id_m else -1

    run_meta = parse_run_metadata(run_dir)

    new_row = {
        "model":        run_meta["model"],
        "method":       run_meta["method"],
        "source":       run_meta["source"],
        "task":         task,
        "metric_name":  metric_name_for(task),
        "ckpt_id":      ckpt_id,
        "ckpt_value":   ckpt_value,
        "best_value":   ckpt_value,   # will be recomputed below
        "best_ckpt_id": ckpt_id,
        "is_best":      "1",
        "selection_tag": run_meta["selection_tag"],
        "run_name":     run_dir.name,
        "run_dir":      str(run_dir),
        "train_file":   run_meta["train_file"],
        "metrics_path": str(metrics_path),
        "ingest_source": ingest_source,
        "ingest_ts":    datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "notes":        notes,
    }

    store_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = store_path.with_suffix(".lock")

    with lock_path.open("a") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            rows = _read_store(store_path)
            rows.append(new_row)
            rows = _recompute_best(rows)
            _write_store(store_path, rows)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    print(f"[store] recorded {task} {run_dir.name}/checkpoint-{ckpt_id} "
          f"value={ckpt_value:.4f} → {store_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-path", required=True)
    ap.add_argument("--run-dir",      required=True)
    ap.add_argument("--ckpt-dir",     required=True)
    ap.add_argument("--task",         required=True)
    ap.add_argument("--store",        required=True)
    ap.add_argument("--notes",        default="")
    args = ap.parse_args()

    record(
        metrics_path=Path(args.metrics_path),
        run_dir=Path(args.run_dir),
        ckpt_dir=Path(args.ckpt_dir),
        task=args.task,
        store_path=Path(args.store),
        ingest_source="checkpoint_json",
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
