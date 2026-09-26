#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from tacs.data_selection.val_warmup_loss_gap import _load_cached_losses


def parse_args():
    ap = argparse.ArgumentParser(
        description="Materialize normalized final-drop selections from cached val-warmup losses and compare with norm_loss_gap selections."
    )
    ap.add_argument("--cache-source-root", required=True)
    ap.add_argument("--norm-selected-root", required=True)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--source-name", default="dolly")
    ap.add_argument("--tasks", nargs="+", default=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--prefix-ks", nargs="+", type=int, default=[4, 3, 2, 1])
    ap.add_argument("--warmup-lora-r", type=int, default=1)
    ap.add_argument("--top-percentage", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-8)
    return ap.parse_args()


def discover_lr_for_task(cache_source_root: Path, task: str, warmup_lora_r: int) -> str:
    task_root = cache_source_root / task / f"rank_{warmup_lora_r}"
    lr_dirs = sorted(p for p in task_root.glob("lr_*") if p.is_dir())
    if len(lr_dirs) != 1:
        raise RuntimeError(f"expected exactly one lr_* dir under {task_root}, found {len(lr_dirs)}")
    return lr_dirs[0].name.replace("lr_", "", 1)


def discover_source_cache(cache_source_root: Path, task: str, lr: str, source_name: str, warmup_lora_r: int) -> Path:
    task_root = cache_source_root / task / f"rank_{warmup_lora_r}" / f"lr_{lr}" / task / "score_cache" / source_name
    if not task_root.is_dir():
        raise FileNotFoundError(f"missing source cache: {task_root}")
    return task_root


def load_rows(source_file: Path) -> List[dict]:
    rows = []
    with source_file.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stable_ranks_desc(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(-scores, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(scores), dtype=order.dtype)
    return ranks


def spearman_desc(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = stable_ranks_desc(a).float()
    rb = stable_ranks_desc(b).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.norm(ra) * torch.norm(rb)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(ra, rb) / denom)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = torch.norm(a) * torch.norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def compute_scores(cache_dir: Path, k: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    losses = []
    for step in range(1, k + 1):
        step_dir = cache_dir / f"step_{step:04d}"
        if not step_dir.is_dir():
            raise FileNotFoundError(f"missing cache step dir: {step_dir}")
        losses.append(_load_cached_losses(str(step_dir)))
    losses = torch.stack(losses, dim=0)
    first = torch.clamp(losses[0], min=eps)
    last = losses[-1]
    norm_final_drop = (losses[0] - last) / first
    norm_loss_gap = (losses.sum(dim=0) - float(k) * last) / first
    return norm_final_drop, norm_loss_gap


def topk_indices(scores: torch.Tensor, top_percentage: float) -> torch.Tensor:
    k = max(1, math.ceil(len(scores) * top_percentage))
    _, idx = torch.topk(scores, k)
    return idx


def write_selected(rows: List[dict], idx: torch.Tensor, scores: torch.Tensor, output_file: Path, source_name: str):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for rank, pos in enumerate(idx.tolist(), start=1):
        row = dict(rows[pos])
        row["_influence_score"] = float(scores[pos].item())
        row["_source"] = source_name
        row["_source_index"] = int(pos)
        row["_selection_rank"] = rank
        selected.append(row)
    with output_file.open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")


def load_norm_selected(norm_selected_root: Path, task: str, k: int) -> List[dict]:
    p = next(norm_selected_root.glob(f"*__{task}__dolly__k{k}__norm_loss_gap/{task}/top_p0.05.jsonl"))
    with p.open() as f:
        return [json.loads(line) for line in f]


def summarize_overlap(a_rows: List[dict], b_rows: List[dict]) -> Dict[str, float]:
    a = {int(r["_source_index"]) for r in a_rows}
    b = {int(r["_source_index"]) for r in b_rows}
    inter = len(a & b)
    union = len(a | b)
    return {
        "intersection": inter,
        "union": union,
        "jaccard": inter / union,
        "overlap_pct": inter / max(1, len(a)),
    }


def main():
    args = parse_args()
    cache_source_root = Path(args.cache_source_root)
    norm_selected_root = Path(args.norm_selected_root)
    source_file = Path(args.source_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(source_file)
    summary = {
        "cache_source_root": str(cache_source_root),
        "norm_selected_root": str(norm_selected_root),
        "source_file": str(source_file),
        "source_name": args.source_name,
        "top_percentage": args.top_percentage,
        "eps": args.eps,
        "tasks": {},
    }

    for task in args.tasks:
        lr = discover_lr_for_task(cache_source_root, task, args.warmup_lora_r)
        cache_dir = discover_source_cache(cache_source_root, task, lr, args.source_name, args.warmup_lora_r)
        task_summary = {"lr": lr, "cache_dir": str(cache_dir), "ks": {}}
        for k in args.prefix_ks:
            norm_final_drop, norm_loss_gap = compute_scores(cache_dir, k, args.eps)
            idx = topk_indices(norm_final_drop, args.top_percentage)
            out_file = output_root / "selected" / f"{task}__{args.source_name}__k{k}__norm_final_drop" / task / "top_p0.05.jsonl"
            write_selected(rows, idx, norm_final_drop, out_file, args.source_name)

            current_rows = [dict(rows[i], _source_index=int(i), _influence_score=float(norm_final_drop[i].item())) for i in idx.tolist()]
            norm_rows = load_norm_selected(norm_selected_root, task, k)
            overlap = summarize_overlap(current_rows, norm_rows)
            task_summary["ks"][str(k)] = {
                "selected_file": str(out_file),
                "score_mean": float(norm_final_drop.mean().item()),
                "score_min": float(norm_final_drop.min().item()),
                "score_max": float(norm_final_drop.max().item()),
                "pearson_vs_norm_loss_gap": pearson(norm_final_drop, norm_loss_gap),
                "spearman_vs_norm_loss_gap": spearman_desc(norm_final_drop, norm_loss_gap),
                "selection_overlap_vs_norm_loss_gap": overlap,
            }
        summary["tasks"][task] = task_summary

    summary_path = output_root / "analysis" / "norm_final_drop_vs_norm_loss_gap.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
