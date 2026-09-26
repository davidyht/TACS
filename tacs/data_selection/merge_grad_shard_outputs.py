"""Concatenate sharded gradient outputs into the unsharded layout used by matching.

Each shard k of N was produced by ``get_info.py --num_shards N --shard_index k`` into
``<output_path>/shard_<k>_of_<N>/dim<d>/{all_orig,all_unormalized}.pt``. Rows are
per-example gradients normalized row by row, so concatenating shards in index order
gives the same ``<output_path>/dim<d>/all_orig.pt`` as one unsharded run.

    python -m tacs.data_selection.merge_grad_shard_outputs --output_path <dir> --num_shards 4 \
        --dims 8192 --expected_rows 100000
"""
import argparse
import json
import os

import torch


def shard_bounds(n_rows: int, shard_index: int, num_shards: int):
    """Contiguous row range [start, end) of one shard; shards tile [0, n_rows) in order."""
    return n_rows * shard_index // num_shards, n_rows * (shard_index + 1) // num_shards


def load_shard_bounds(path: str, n_rows: int, num_shards: int):
    """Row boundaries [0 = b_0 < b_1 < ... < b_N = n_rows] of contiguous shards from a JSON file
    (``make_length_balanced_shards.py``). Shard k covers rows [b_k, b_{k+1}); merging is unchanged."""
    with open(path) as fh:
        bounds = [int(b) for b in json.load(fh)["bounds"]]
    if len(bounds) != num_shards + 1:
        raise ValueError(f"{path}: {len(bounds) - 1} shards, expected {num_shards}")
    if bounds[0] != 0 or bounds[-1] != n_rows:
        raise ValueError(f"{path}: bounds cover [{bounds[0]}, {bounds[-1]}), dataset has {n_rows} rows")
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"{path}: bounds must be strictly increasing")
    return bounds


def shard_dir(output_path: str, k: int, num_shards: int) -> str:
    return os.path.join(output_path, f"shard_{k}_of_{num_shards}")


def merge(output_path: str, num_shards: int, dims, expected_rows=None) -> dict:
    written = {}
    for dim in dims:
        final_dir = os.path.join(output_path, f"dim{dim}")
        for name in ("all_orig.pt", "all_unormalized.pt"):
            parts = [os.path.join(shard_dir(output_path, k, num_shards), f"dim{dim}", name) for k in range(num_shards)]
            if name == "all_unormalized.pt" and not all(os.path.exists(p) for p in parts):
                continue
            missing = [p for p in parts if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"missing shard output: {missing[0]}")
            target = os.path.join(final_dir, name)
            if os.path.exists(target):
                raise FileExistsError(f"refusing to overwrite {target}")
            tensors = [torch.load(p, map_location="cpu") for p in parts]
            if len({t.shape[1] for t in tensors}) != 1 or tensors[0].shape[1] != dim:
                raise ValueError(f"shard widths differ from dim {dim}: {[tuple(t.shape) for t in tensors]}")
            merged = torch.cat(tensors, dim=0)
            if expected_rows is not None and merged.shape[0] != expected_rows:
                raise ValueError(f"{target}: merged {merged.shape[0]} rows, expected {expected_rows}")
            os.makedirs(final_dir, exist_ok=True)
            tmp = target + ".tmp"
            torch.save(merged, tmp)
            os.replace(tmp, target)
            written[target] = tuple(merged.shape)
            print(f"merged {num_shards} shards -> {target} {tuple(merged.shape)}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[8192])
    ap.add_argument("--expected_rows", type=int, default=None)
    args = ap.parse_args()
    merge(args.output_path, args.num_shards, args.dims, args.expected_rows)


if __name__ == "__main__":
    main()
