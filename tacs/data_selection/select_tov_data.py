import argparse
import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch


KNOWN_TEXT_KEYS = [
    "text",
    "instruction",
    "input",
    "output",
    "response",
    "prompt",
    "completion",
    "question",
    "answer",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select ToV data from merged scores with optional length binning and "
            "paper-style score+random selection."
        )
    )
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--candidate_file", required=True)
    parser.add_argument("--base_subset_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--meta_file", required=True)
    parser.add_argument("--selection_mode", choices=["score_only", "score_random"], default="score_random")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--percentage", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--length_mode", choices=["words", "chars"], default="words")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--disable_length_binning", action="store_true")
    return parser.parse_args()


def _target_k(total: int, max_samples: Optional[int], percentage: Optional[float]) -> int:
    if (max_samples is None) == (percentage is None):
        raise ValueError("Specify exactly one of --max_samples or --percentage")
    if percentage is not None:
        if not (0.0 < percentage <= 1.0):
            raise ValueError("--percentage must be in (0, 1]")
        # TOV_TOP_ROUNDING=floor matches LESS/Random/TACS in the clean protocol; default keeps history.
        _mode = __import__("os").environ.get("TOV_TOP_ROUNDING", "round")
        if _mode not in ("round", "floor"):
            raise ValueError(f"TOV_TOP_ROUNDING must be round or floor, got {_mode}")
        _n = total * percentage
        return max(1, int(__import__("math").floor(_n + 1e-9)) if _mode == "floor" else int(round(_n)))
    if max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    return max_samples


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _extract_text(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        chunks: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content)
        if chunks:
            return "\n".join(chunks)

    chunks = []
    for key in KNOWN_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value)
    if chunks:
        return "\n".join(chunks)

    fallback = []
    for key, value in row.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, str) and value.strip():
            fallback.append(value)
    return "\n".join(fallback)


def _length_value(text: str, mode: str) -> int:
    if mode == "chars":
        return len(text)
    if not text:
        return 0
    return len(text.split())


def _build_quantile_edges(lengths: torch.Tensor, num_bins: int) -> List[float]:
    if num_bins <= 1 or lengths.numel() == 0:
        return []
    qs = torch.linspace(0, 1, steps=num_bins + 1, dtype=torch.float64)[1:-1]
    if qs.numel() == 0:
        return []
    edges = torch.quantile(lengths.to(dtype=torch.float64), qs).tolist()
    cleaned: List[float] = []
    last = None
    for edge in edges:
        edge = float(edge)
        if last is None or edge > last:
            cleaned.append(edge)
            last = edge
    return cleaned


def _proportional_quotas(group_sizes: Dict[int, int], target_k: int) -> Dict[int, int]:
    total = sum(group_sizes.values())
    if total <= 0:
        return {group: 0 for group in group_sizes}

    raw = {group: target_k * (size / total) for group, size in group_sizes.items()}
    quotas = {group: int(math.floor(value)) for group, value in raw.items()}
    used = sum(quotas.values())
    remain = max(0, target_k - used)

    frac_order = sorted(
        raw.keys(),
        key=lambda group: (raw[group] - quotas[group], group_sizes[group]),
        reverse=True,
    )
    i = 0
    while remain > 0 and frac_order:
        group = frac_order[i % len(frac_order)]
        if quotas[group] < group_sizes[group]:
            quotas[group] += 1
            remain -= 1
        i += 1
        if i > len(frac_order) * (target_k + 1):
            break
    return quotas


def _select_scored(
    candidates: Sequence[dict],
    scores: torch.Tensor,
    k: int,
    length_mode: str,
    num_bins: int,
    use_length_binning: bool,
) -> List[dict]:
    if k <= 0:
        return []
    if len(candidates) != scores.numel():
        raise ValueError(
            f"Candidate length {len(candidates)} != score length {scores.numel()}"
        )

    enriched: List[dict] = []
    for idx, row in enumerate(candidates):
        ex = dict(row)
        score = float(scores[idx].item())
        ex["_influence_score"] = score
        text = _extract_text(ex)
        ex["_length_value"] = _length_value(text, length_mode)
        enriched.append(ex)

    if not use_length_binning:
        ranked = sorted(enriched, key=lambda row: row["_influence_score"], reverse=True)
        return ranked[: min(k, len(ranked))]

    lengths = torch.tensor([row["_length_value"] for row in enriched], dtype=torch.float32)
    edges = _build_quantile_edges(lengths, num_bins)
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in enriched:
        row["_length_bin"] = bisect_right(edges, float(row["_length_value"]))
        grouped[row["_length_bin"]].append(row)
    quotas = _proportional_quotas({bin_id: len(rows) for bin_id, rows in grouped.items()}, k)

    selected: List[dict] = []
    for bin_id, rows in grouped.items():
        rows.sort(key=lambda row: row["_influence_score"], reverse=True)
        selected.extend(rows[: min(len(rows), quotas.get(bin_id, 0))])

    if len(selected) < k:
        seen = {(row["_source_index"], row.get("_source")) for row in selected}
        ranked_all = sorted(enriched, key=lambda row: row["_influence_score"], reverse=True)
        for row in ranked_all:
            key = (row["_source_index"], row.get("_source"))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= k:
                break
    elif len(selected) > k:
        selected = sorted(selected, key=lambda row: row["_influence_score"], reverse=True)[:k]

    return sorted(selected, key=lambda row: row["_influence_score"], reverse=True)


def _sample_base_subset(rows: Sequence[dict], k: int, seed: int) -> List[dict]:
    if k <= 0:
        return []
    k = min(k, len(rows))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(rows)), k))
    return [dict(rows[idx]) for idx in chosen]


def _write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    candidate_rows = _load_jsonl(args.candidate_file)
    base_rows = _load_jsonl(args.base_subset_file)
    scores = torch.load(args.score_file, map_location="cpu")
    if not torch.is_tensor(scores):
        scores = torch.tensor(scores)
    scores = scores.float().view(-1)

    total_examples = len(candidate_rows) + len(base_rows)
    target_k = min(total_examples, _target_k(total_examples, args.max_samples, args.percentage))

    if args.selection_mode == "score_random":
        scored_k = min(len(candidate_rows), target_k // 2)
        random_k = min(len(base_rows), target_k - scored_k)
        if scored_k + random_k < target_k:
            scored_k = min(len(candidate_rows), target_k - random_k)
    else:
        scored_k = min(len(candidate_rows), target_k)
        random_k = 0

    scored_selected = _select_scored(
        candidates=candidate_rows,
        scores=scores,
        k=scored_k,
        length_mode=args.length_mode,
        num_bins=args.num_bins,
        use_length_binning=not args.disable_length_binning,
    )
    random_selected = _sample_base_subset(base_rows, random_k, args.seed + 17)

    final_rows: List[dict] = []
    for row in scored_selected:
        out = dict(row)
        out["_selection_component"] = "score"
        final_rows.append(out)
    for row in random_selected:
        out = dict(row)
        out["_selection_component"] = "random_base"
        out["_influence_score"] = None
        final_rows.append(out)

    for rank, row in enumerate(final_rows, start=1):
        row["_selection_rank"] = rank

    _write_jsonl(args.output_file, final_rows)

    meta = {
        "selection_mode": args.selection_mode,
        "target_k": target_k,
        "selected_scored": len(scored_selected),
        "selected_random_base": len(random_selected),
        "candidate_pool_examples": len(candidate_rows),
        "base_subset_examples": len(base_rows),
        "length_mode": args.length_mode,
        "num_bins": None if args.disable_length_binning else args.num_bins,
        "score_file": args.score_file,
        "candidate_file": args.candidate_file,
        "base_subset_file": args.base_subset_file,
    }
    with open(args.meta_file, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Wrote {len(final_rows)} examples -> {args.output_file}")
    print(f"Wrote metadata -> {args.meta_file}")


if __name__ == "__main__":
    main()
