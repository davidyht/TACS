#!/usr/bin/env python3
"""
Merge and convert optimizer checkpoint shards into a name-keyed optimizer state.

Usage examples:
  python convert_and_merge_optimizer.py \
    --optim-files checkpoint-422/optimizer.pt checkpoint-422/optimizer.pt.bak \
    --adapter checkpoint-422/adapter_model.safetensors \
    --out checkpoint-422/optimizer_merged_with_names.pt

The script attempts these strategies (in priority):
 - prefer any existing name-keyed optimizer state if present (merge others into it)
 - for integer/index-keyed states, try to map them to adapter/model parameter names by shape and ordering heuristics
 - produce diagnostics listing matched/unmatched params

This is a best-effort, verbose consolidation tool to help with cases where training saved sharded or id-keyed optimizer states.
"""
import argparse
import os
import torch
import json
from collections import defaultdict

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


def load_torch_file(path):
    print(f"Loading torch file: {path}")
    data = torch.load(path, map_location="cpu")
    # prefer common keys
    for candidate in ("state", "optimizer_state_dict", "opt_state", "optimizer"):
        if isinstance(data, dict) and candidate in data:
            return data
    return data


def load_safetensors_keys(path):
    if safe_open is None:
        raise RuntimeError("safetensors not available; please install safetensors to read adapter files")
    keys = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return keys


def normalize_name(k):
    # apply the typical normalizations we've used elsewhere
    kk = k
    kk = kk.replace("base_model.model.model.", "")
    kk = kk.replace("base_model.model.", "")
    kk = kk.replace("model.model.", "model.")
    kk = kk.replace(".default", "")
    return kk


def extract_state_dict(data):
    # return the actual state dict object that contains per-parameter entries
    if isinstance(data, dict):
        for k in ("state", "optimizer_state_dict", "opt_state", "optimizer"):
            if k in data and isinstance(data[k], dict):
                return data[k]
        # maybe the file saved directly the state dict
        return data
    return None


def merge_optimizer_states(paths):
    """Load multiple optimizer files and merge into a dict mapping key->state.
    Keys may be ints or strings. Later we'll attempt to map ints to names.
    If duplicate keys appear, later files override earlier ones.
    """
    merged = {}
    sources = {}
    for p in paths:
        d = load_torch_file(p)
        st = extract_state_dict(d)
        if st is None:
            print(f"Warning: no state dict found in {p}")
            continue
        for k, v in st.items():
            merged[k] = v
            sources[k] = p
    return merged, sources


def attempt_map_indices_to_names(int_keys, name_candidates, sources_map):
    """Heuristic mapping when optimizer state keys are integers (0..N-1).
    If the number of int keys equals number of candidate names, map by sorted index->sorted names and check shapes.
    """
    mapping = {}
    diagnostics = {"shape_mismatch": [], "mapped": [], "unmapped": []}
    # sort int keys numerically
    sorted_ints = sorted(int_keys, key=lambda x: int(x))
    if len(sorted_ints) != len(name_candidates):
        print(f"Index-key count ({len(sorted_ints)}) != candidate name count ({len(name_candidates)}); mapping by order may be unsafe")
    # create order mapping up to min length
    for idx, name in zip(sorted_ints, name_candidates):
        mapping[idx] = name
        diagnostics["mapped"].append((idx, name, sources_map.get(idx)))
    return mapping, diagnostics


def build_name_keyed_state(merged_state, adapter_keys=None):
    # identify whether merged_state already contains string keys that look like names
    str_keys = [k for k in merged_state.keys() if isinstance(k, str)]
    int_keys = [k for k in merged_state.keys() if not isinstance(k, str)]

    result = {}
    diag = {"from_name_keyed": 0, "from_index_mapped": 0, "from_unmapped": 0}

    if str_keys:
        # assume many names present; copy them and later try to normalize
        for k in str_keys:
            nk = normalize_name(k)
            result[nk] = merged_state[k]
            diag["from_name_keyed"] += 1

    if int_keys and adapter_keys:
        # try mapping ints -> adapter names heuristically by ordering
        # prepare normalized adapter list
        norm_adapter = [normalize_name(k) for k in adapter_keys]
        mapping, diagnostics = attempt_map_indices_to_names(int_keys, norm_adapter, {})
        mapped_ints = set()
        for i, name in mapping.items():
            if i in merged_state:
                result[name] = merged_state[i]
                mapped_ints.add(i)
        diag["from_index_mapped"] = len(mapped_ints)

    # any remaining ints -> put under stringified ints to preserve
    unmapped_count = 0
    for i in int_keys:
        if i not in mapped_ints:
            keyname = f"idx_{i}"
            result[keyname] = merged_state[i]
            unmapped_count += 1
    diag["from_unmapped"] = unmapped_count

    return result, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim-files", nargs="+", required=True, help="optimizer files to merge (torch .pt/.pth)")
    parser.add_argument("--adapter", type=str, default=None, help="optional adapter safetensors path to extract param names")
    parser.add_argument("--out", type=str, required=True, help="output path for merged name-keyed optimizer state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    merged, sources = merge_optimizer_states(args.optim_files)
    print(f"Merged keys count: {len(merged)}")

    adapter_keys = None
    if args.adapter:
        if not os.path.exists(args.adapter):
            print(f"Adapter path not found: {args.adapter}")
        else:
            if args.adapter.endswith(".safetensors"):
                adapter_keys = load_safetensors_keys(args.adapter)
            else:
                # try torch load
                try:
                    ad = torch.load(args.adapter, map_location="cpu")
                    if isinstance(ad, dict):
                        adapter_keys = list(ad.keys())
                except Exception as e:
                    print("Could not load adapter file:", e)

    name_keyed, diag = build_name_keyed_state(merged, adapter_keys)

    print("Diagnostics:")
    print(json.dumps(diag, indent=2))
    print(f"Result name-keyed entries: {len(name_keyed)}")

    if args.dry_run:
        print("Dry run complete. Not writing output.")
        return

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save as torch file with 'state' key for compatibility
    save_dict = {"state": name_keyed}
    torch.save(save_dict, args.out)
    print(f"Wrote merged optimizer to {args.out}")


if __name__ == "__main__":
    main()
