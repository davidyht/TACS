import argparse
import json
import os
import random
from typing import List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ToV source split into a random base subset U and candidate pool X\\U."
    )
    parser.add_argument("--train_file", required=True, help="Original source JSONL file.")
    parser.add_argument("--source_name", required=True, help="Short source name, e.g. flan_v2.")
    parser.add_argument("--output_dir", required=True, help="Directory to write split artifacts.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for base subset sampling.")
    parser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Fraction of the original source used for the random base subset U.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Exact number of examples to place in the random base subset U.",
    )
    parser.add_argument(
        "--sampler",
        choices=["python_random", "warmup_np"],
        default="python_random",
        help=(
            "How to sample the warmup subset U. "
            "'warmup_np' matches tacs.train.get_training_dataset "
            "(np.random.permutation with sample_data_seed)."
        ),
    )
    return parser.parse_args()


def target_size(total: int, max_samples: int, percentage: float) -> int:
    if (max_samples is None) == (percentage is None):
        raise ValueError("Specify exactly one of --max_samples or --percentage")
    if percentage is not None:
        if not (0.0 < percentage <= 1.0):
            raise ValueError("--percentage must be in (0, 1]")
        # TOV_TOP_ROUNDING=floor also floors the base-subset size, matching the LESS warmup (int(p*N)).
        _mode = __import__("os").environ.get("TOV_TOP_ROUNDING", "round")
        if _mode not in ("round", "floor"):
            raise ValueError(f"TOV_TOP_ROUNDING must be round or floor, got {_mode}")
        _n = total * percentage
        return max(1, int(__import__("math").floor(_n + 1e-9)) if _mode == "floor" else int(round(_n)))
    if max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    return min(total, max_samples)


def load_jsonl_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.readlines()


def annotate(raw_line: str, source_name: str, source_index: int, partition: str) -> str:
    row = json.loads(raw_line)
    row["_source"] = source_name
    row["_source_index"] = source_index
    row["_tov_partition"] = partition
    return json.dumps(row, ensure_ascii=False) + "\n"


def write_jsonl(path: str, rows: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)


def sample_base_indices(total: int, base_size: int, seed: int, sampler: str) -> List[int]:
    if sampler == "warmup_np":
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            chosen = np.random.permutation(total)[:base_size].tolist()
        finally:
            np.random.set_state(state)
        return sorted(int(idx) for idx in chosen)
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), base_size))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_lines = load_jsonl_lines(args.train_file)
    total = len(raw_lines)
    if total == 0:
        raise ValueError(f"Source file is empty: {args.train_file}")

    base_size = target_size(total, args.max_samples, args.percentage)
    base_indices = sample_base_indices(total, base_size, args.seed, args.sampler)
    base_index_set = set(base_indices)

    base_rows: List[str] = []
    candidate_rows: List[str] = []
    candidate_indices: List[int] = []
    for idx, raw in enumerate(raw_lines):
        if idx in base_index_set:
            base_rows.append(annotate(raw, args.source_name, idx, "base_subset"))
        else:
            candidate_rows.append(annotate(raw, args.source_name, idx, "candidate_pool"))
            candidate_indices.append(idx)

    base_file = os.path.join(args.output_dir, "base_subset.jsonl")
    candidate_file = os.path.join(args.output_dir, "candidate_pool.jsonl")
    meta_file = os.path.join(args.output_dir, "split_meta.json")

    write_jsonl(base_file, base_rows)
    write_jsonl(candidate_file, candidate_rows)

    payload = {
        "source_name": args.source_name,
        "train_file": args.train_file,
        "seed": args.seed,
        "sampler": args.sampler,
        "total_examples": total,
        "base_subset_examples": len(base_rows),
        "candidate_pool_examples": len(candidate_rows),
        "base_subset_indices": base_indices,
        "candidate_pool_indices": candidate_indices,
        "base_subset_file": base_file,
        "candidate_pool_file": candidate_file,
    }
    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote base subset ({len(base_rows)}) -> {base_file}")
    print(f"Wrote candidate pool ({len(candidate_rows)}) -> {candidate_file}")
    print(f"Wrote metadata -> {meta_file}")


if __name__ == "__main__":
    main()
