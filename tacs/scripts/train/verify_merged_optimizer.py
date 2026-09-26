#!/usr/bin/env python3
"""
Verify a merged, name-keyed optimizer file against a LoRA adapter safetensors file.

Usage:
  python verify_merged_optimizer.py \
    --merged optimizer_merged_with_names.pt \
    --adapter adapter_model.safetensors

Outputs a concise summary: total adapter keys, matched-by-name count,
missing keys, and any shape mismatches between adapter tensors and
optimizer moment tensors (exp_avg/exp_avg_sq).
"""
import argparse
import os
import sys
import torch
from collections import defaultdict

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


def normalize(k):
    return k.replace("base_model.model.model.", "").replace("base_model.model.", "").replace("model.model.", "model.").replace(".default", "")


def load_merged(path):
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict) and "state" in d and isinstance(d["state"], dict):
        return d["state"]
    if isinstance(d, dict):
        return d
    raise RuntimeError("Unexpected merged optimizer format: expected dict or {'state': dict}")


def load_adapter_keys_and_tensors(path):
    if safe_open is None:
        raise RuntimeError("safetensors not installed; install via `pip install safetensors`")
    keys = []
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            tensors[k] = f.get_tensor(k)
    return keys, tensors


def find_representative_tensor_in_opt_entry(entry):
    # entry may be a dict with keys 'exp_avg','exp_avg_sq', or maybe raw tensor
    if entry is None:
        return None
    if hasattr(entry, 'shape'):
        return entry
    if isinstance(entry, dict):
        for candidate in ('exp_avg', 'exp_avg_sq', 'momentum_buffer', 0, '0'):
            if candidate in entry:
                return entry[candidate]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', required=True)
    parser.add_argument('--adapter', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.merged):
        print('Merged optimizer not found:', args.merged)
        sys.exit(2)
    if not os.path.exists(args.adapter):
        print('Adapter file not found:', args.adapter)
        sys.exit(2)

    state = load_merged(args.merged)
    adapter_keys, adapter_tensors = load_adapter_keys_and_tensors(args.adapter)

    norm_state_keys = {normalize(k): k for k in state.keys()}

    matched = []
    missing = []
    shape_mismatches = []

    for ak in adapter_keys:
        nk = normalize(ak)
        if nk not in norm_state_keys:
            missing.append((ak, nk))
            continue
        opt_key = norm_state_keys[nk]
        opt_entry = state[opt_key]
        rep = find_representative_tensor_in_opt_entry(opt_entry)
        adapter_tensor = adapter_tensors[ak]
        if rep is None:
            # can't find tensor to compare, still count as matched-by-name
            matched.append((ak, nk, 'no_tensor'))
            continue
        # compare shapes
        if tuple(rep.shape) != tuple(adapter_tensor.shape):
            shape_mismatches.append((ak, nk, adapter_tensor.shape, tuple(rep.shape)))
        else:
            matched.append((ak, nk, adapter_tensor.shape))

    print('Adapter total:', len(adapter_keys))
    print('Matched-by-name:', len(matched))
    print('Missing:', len(missing))
    if missing:
        print('Missing sample (up to 20):')
        for m in missing[:20]:
            print(' ', m)
    print('Shape mismatches:', len(shape_mismatches))
    if shape_mismatches:
        print('Shape mismatch samples (up to 20):')
        for s in shape_mismatches[:20]:
            print(' ', s)

    # report sample of state keys not referenced by adapter (extra params)
    adapter_norm_set = set(normalize(k) for k in adapter_keys)
    extra = [k for k in state.keys() if normalize(k) not in adapter_norm_set]
    print('Extra optimizer keys (not in adapter) count:', len(extra))
    if extra:
        print('Extra sample (20):')
        for k in extra[:20]:
            print(' ', k)


if __name__ == '__main__':
    main()
