#!/usr/bin/env python3
"""
Convert an optimizer state file whose 'state' dict is keyed by integers
into a file keyed by parameter names using a model checkpoint to map indices.

Usage:
  python convert_optimizer_ids_to_names.py \
    --optimizer /path/to/optimizer.pt \
    --model_path /path/to/model_checkpoint_dir \
    --output /path/to/optimizer_with_names.pt

The script attempts several heuristics to map integer keys to parameter names:
 1. If integer keys are indices and fit within the model parameter list length,
    map by index -> named parameter list.
 2. If integer keys look like Python id() values, map by matching id(p) for each
    parameter in the loaded model.
 3. If some keys can't be mapped, they will be retained under a fallback name
    `param_<key>` and a warning will be printed.

This is a best-effort utility; verify outputs before using for downstream jobs.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
import sys


def load_optimizer(opt_path: str):
    print(f"Loading optimizer state from {opt_path}")
    d = torch.load(opt_path, map_location="cpu")
    if not isinstance(d, dict):
        raise RuntimeError("Loaded optimizer file is not a dict")
    if 'state' not in d or 'param_groups' not in d:
        raise RuntimeError("Optimizer file missing expected keys ('state','param_groups')")
    return d


def load_model(checkpoint: str):
    print(f"Loading model for mapping from {checkpoint} (cpu)")
    # Load minimal model on CPU. This may still require significant memory for large models.
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cpu', low_cpu_mem_usage=True)
    except Exception:
        # Fallback to loading configs then model
        cfg = AutoConfig.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_config(cfg)
    return model


def build_param_name_list(model):
    names = []
    params = []
    for name, p in model.named_parameters():
        names.append(name)
        params.append(p)
    return names, params


def map_keys_by_index(state_keys, names):
    # state_keys are ints maybe 0..N-1
    maxk = max(state_keys)
    if maxk < len(names):
        mapping = {int(k): names[int(k)] for k in state_keys}
        print("Mapping by index successful (keys look like indices).")
        return mapping
    return None


def map_keys_by_id(state_keys, params, names):
    id_to_name = {id(p): n for p, n in zip(params, names)}
    mapping = {}
    found = 0
    for k in state_keys:
        if int(k) in id_to_name:
            mapping[int(k)] = id_to_name[int(k)]
            found += 1
    if found > 0:
        print(f"Mapped {found} keys by matching id(param).")
        return mapping
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    opt = load_optimizer(args.optimizer)
    state = opt['state']
    param_groups = opt.get('param_groups', [])
    state_keys = list(state.keys())
    # Normalize keys to ints where possible
    try:
        state_keys_int = [int(k) for k in state_keys]
    except Exception:
        state_keys_int = state_keys

    model = load_model(args.model_path)
    names, params = build_param_name_list(model)

    mapping = None
    # Try index mapping
    try:
        mapping = map_keys_by_index(state_keys_int, names)
    except Exception:
        mapping = None

    # Try id mapping
    if mapping is None:
        mapping = map_keys_by_id(state_keys_int, params, names)

    new_state = {}
    unmapped = []
    for k in state_keys_int:
        name = None
        if mapping and k in mapping:
            name = mapping[k]
        else:
            # fallback name
            name = f"param_{k}"
            unmapped.append(k)
        new_state[name] = state[k]

    # Replace params list in param_groups (attempt best-effort)
    new_param_groups = []
    for g in param_groups:
        new_g = dict(g)
        new_params = []
        for p in g.get('params', []):
            try:
                idx = int(p)
            except Exception:
                idx = p
            if mapping and idx in mapping:
                new_params.append(mapping[idx])
            else:
                new_params.append(f"param_{idx}")
        new_g['params'] = new_params
        new_param_groups.append(new_g)

    new_opt = dict(opt)
    new_opt['state'] = new_state
    new_opt['param_groups'] = new_param_groups

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.optimizer), 'optimizer_with_names.pt')

    backup_path = args.optimizer + '.bak'
    if not os.path.exists(backup_path):
        print(f"Backing up original optimizer to {backup_path}")
        torch.save(opt, backup_path)

    print(f"Saving converted optimizer to {out_path}")
    torch.save(new_opt, out_path)

    if unmapped:
        print(f"Warning: {len(unmapped)} keys could not be mapped to parameter names; assigned fallback names.\nSample unmapped keys: {unmapped[:10]}")
    else:
        print("All keys mapped to parameter names.")

    print("Done.")


if __name__ == '__main__':
    main()
