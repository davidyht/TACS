#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from tacs.data_selection.get_validation_dataset import get_dataset, resolve_mmlu_n_shot
from tacs.train.model_arguments import add_padding_to_tokenizer


def _build_fold_sizes(total_n: int, fold_count: int, holdout_ratio: float) -> List[int]:
    if total_n < fold_count:
        raise ValueError(f"need at least {fold_count} examples for {fold_count}-fold CV; got {total_n}")
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError(f"holdout_ratio must be in (0,1), got {holdout_ratio}")

    target = max(1, int(round(total_n * holdout_ratio)))
    sizes: List[int] = []
    remaining = total_n
    for fold_idx in range(fold_count - 1):
        min_rest = (fold_count - fold_idx - 1)
        size = min(max(1, target), remaining - min_rest)
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    if sum(sizes) != total_n or any(size <= 0 for size in sizes):
        raise RuntimeError(f"invalid fold sizes: total_n={total_n} sizes={sizes}")
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare explicit CV split files for val-warmup HP search.")
    ap.add_argument(
        "--task",
        required=True,
        choices=["tydiqa", "mmlu", "bbh", "gsm8k", "truthfulqa"],
    )
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fold-count", type=int, default=3)
    ap.add_argument("--holdout-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--chat-format", default="tokenizer")
    ap.add_argument("--mmlu-n-shot", type=int, default=1)
    ap.add_argument(
        "--mmlu-subjects",
        nargs="+",
        help="Explicit MMLU subject filter; recorded and checked against the resulting row count.",
    )
    ap.add_argument("--use-chat-format", type=int, default=1)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    add_padding_to_tokenizer(tokenizer)
    dataset = get_dataset(
        args.task,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        use_chat_format=bool(args.use_chat_format),
        chat_format=args.chat_format,
        mmlu_n_shot=args.mmlu_n_shot,
        mmlu_subjects=args.mmlu_subjects,
    )
    total_n = len(dataset)
    resolved_mmlu_n_shot = (
        int(resolve_mmlu_n_shot(args.mmlu_n_shot)) if args.task == "mmlu" else None
    )
    if args.mmlu_subjects and args.task != "mmlu":
        raise ValueError("--mmlu-subjects is only valid with --task mmlu")
    expected_total_n = (
        len(args.mmlu_subjects) * resolved_mmlu_n_shot
        if args.task == "mmlu" and args.mmlu_subjects
        else None
    )
    if expected_total_n is not None and total_n != expected_total_n:
        raise ValueError(
            "MMLU subject filter did not produce the expected dataset size: "
            f"subjects={len(args.mmlu_subjects)} n_shot={resolved_mmlu_n_shot} "
            f"expected_total_n={expected_total_n} actual_total_n={total_n}"
        )
    if total_n < 2:
        raise ValueError(f"target validation dataset is too small: {total_n}")

    perm = list(range(total_n))
    random.Random(int(args.seed)).shuffle(perm)
    sizes = _build_fold_sizes(total_n=total_n, fold_count=int(args.fold_count), holdout_ratio=float(args.holdout_ratio))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": args.task,
        "model_name_or_path": args.model_name_or_path,
        "data_dir": args.data_dir,
        "fold_count": int(args.fold_count),
        "requested_holdout_ratio": float(args.holdout_ratio),
        "seed": int(args.seed),
        "total_n": int(total_n),
        "expected_total_n": expected_total_n,
        "mmlu_n_shot": resolved_mmlu_n_shot,
        "mmlu_subjects": list(args.mmlu_subjects) if args.mmlu_subjects else None,
        "folds": [],
    }

    start = 0
    for fold_idx, fold_size in enumerate(sizes, start=1):
        probe_indices = sorted(perm[start:start + fold_size])
        start += fold_size
        probe_set = set(probe_indices)
        warmup_indices = [idx for idx in range(total_n) if idx not in probe_set]
        split_obj = {
            "version": 1,
            "target_task": args.task,
            "split_strategy": "random",
            "warmup_ratio": float(len(warmup_indices)) / float(total_n),
            "requested_holdout_ratio": float(args.holdout_ratio),
            "split_seed": int(args.seed),
            "fold_index": int(fold_idx),
            "fold_count": int(args.fold_count),
            "mmlu_n_shot": resolved_mmlu_n_shot,
            "mmlu_subjects": list(args.mmlu_subjects) if args.mmlu_subjects else None,
            "total_n": int(total_n),
            "warmup_indices": [int(x) for x in warmup_indices],
            "probe_indices": [int(x) for x in probe_indices],
            "warmup_n": int(len(warmup_indices)),
            "probe_n": int(len(probe_indices)),
            "grouped": False,
        }
        split_path = out_dir / f"fold_{fold_idx:02d}.json"
        split_path.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
        manifest["folds"].append(
            {
                "fold_index": int(fold_idx),
                "path": str(split_path),
                "warmup_n": int(len(warmup_indices)),
                "probe_n": int(len(probe_indices)),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    for fold in manifest["folds"]:
        print(
            f"fold={fold['fold_index']} warmup_n={fold['warmup_n']} "
            f"probe_n={fold['probe_n']} path={fold['path']}"
        )


if __name__ == "__main__":
    main()
