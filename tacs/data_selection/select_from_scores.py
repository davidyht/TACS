import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset


def parse_source_mapping(pairs: List[str]) -> Dict[str, str]:
    mapping = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid source mapping '{p}'. Use source=/path/to/file.jsonl")
        k, v = p.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def load_scores(scores_dir: str, sources: List[str]) -> Tuple[List[torch.Tensor], List[Tuple[str, int]]]:
    all_scores = []
    index_map: List[Tuple[str, int]] = []
    for src in sources:
        path = os.path.join(scores_dir, f"{src}_influence_score.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")
        v = torch.load(path, map_location="cpu")
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        v = v.float().view(-1)
        all_scores.append(v)
        index_map.extend([(src, i) for i in range(v.numel())])
    cat = torch.cat(all_scores)
    return cat, index_map


def select_indices(scores: torch.Tensor, index_map: List[Tuple[str, int]], top_k: int = None, top_pct: float = None):
    if (top_k is None) == (top_pct is None):
        raise ValueError("Specify exactly one of --top_k or --top_percentage")
    if top_pct is not None:
        if not (0.0 < top_pct <= 1.0):
            raise ValueError("--top_percentage must be in (0, 1]")
        # TACS_TOP_ROUNDING=floor matches LESS/Random (int(fraction * N)); default keeps historical rounding.
        _n = scores.numel() * top_pct
        _rounding = __import__("os").environ.get("TACS_TOP_ROUNDING", "round")
        if _rounding not in ("round", "floor"):
            raise ValueError(f"TACS_TOP_ROUNDING must be round or floor, got {_rounding}")
        k = max(1, int(__import__("math").floor(_n + 1e-9)) if _rounding == "floor" else int(round(_n)))
    else:
        k = max(1, int(top_k))
    top = torch.topk(scores, k)
    sel = [(index_map[idx][0], index_map[idx][1], float(scores[idx])) for idx in top.indices.tolist()]
    return sel


def write_selected_jsonl(output_file: str, selections: List[Tuple[str, int, float]], source_files: Dict[str, str]):
    # Load each source dataset once
    datasets_cache: Dict[str, List[dict]] = {}
    direct = os.environ.get("SELECT_DIRECT_JSONL", "0").lower() in {"1", "true", "yes"}
    for src, fpath in source_files.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Training file for source '{src}' not found: {fpath}")
        if direct:
            # Avoid Hugging Face Datasets' cache/materialization, which can fail
            # on quota-constrained Lustre even for a small JSONL source.
            with open(fpath, encoding="utf-8") as fh:
                datasets_cache[src] = [json.loads(line) for line in fh if line.strip()]
        else:
            ds = load_dataset("json", data_files=fpath)["train"]
            # For sample_percentage=1.0 used in gradient collection, ordering equals file order
            datasets_cache[src] = [ds[i] for i in range(len(ds))]

    with open(output_file, "w") as out:
        for rank, (src, idx, score) in enumerate(selections, 1):
            ex = datasets_cache[src][idx]
            # Optionally attach score for traceability
            ex_with_score = dict(ex)
            ex_with_score["_influence_score"] = score
            ex_with_score["_source"] = src
            ex_with_score["_source_index"] = idx
            ex_with_score["_selection_rank"] = rank
            out.write(json.dumps(ex_with_score) + "\n")


def write_metadata(meta_file: str, selections: List[Tuple[str, int, float]]):
    meta = [
        {"source": src, "index": idx, "score": score}
        for (src, idx, score) in selections
    ]
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Select final training data from influence score files and write a combined JSONL")
    ap.add_argument("--scores_dir", required=True, help="Directory containing {source}_influence_score.pt files (e.g., /.../out/selection/tydiqa)")
    ap.add_argument("--source", action="append", required=True, help="One or more 'name=/path/to/train.jsonl' pairs. Repeat per source.")
    ap.add_argument("--top_k", type=int, default=None, help="Select top K examples across all sources")
    ap.add_argument("--top_percentage", type=float, default=None, help="Select top percentage across all sources (0-1]")
    ap.add_argument("--output_file", required=True, help="Path to write the combined selected JSONL")
    ap.add_argument("--meta_file", default=None, help="Optional path to write selection metadata as JSON")
    args = ap.parse_args()

    source_map = parse_source_mapping(args.source)
    sources = list(source_map.keys())
    scores, index_map = load_scores(args.scores_dir, sources)
    selections = select_indices(scores, index_map, top_k=args.top_k, top_pct=args.top_percentage)
    write_selected_jsonl(args.output_file, selections, source_map)
    if args.meta_file:
        write_metadata(args.meta_file, selections)
    print(f"Wrote {len(selections)} examples to {args.output_file}")
    if args.meta_file:
        print(f"Wrote metadata to {args.meta_file}")


if __name__ == "__main__":
    main()
