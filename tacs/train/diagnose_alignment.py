#!/usr/bin/env python
# coding: utf-8
"""
Dry-run diagnostics: compare warmup (train.py) vs relaunch (relaunch.py) settings without training.

- Loads warmup config via HF JSON (ModelArguments, DataArguments, TrainingArguments)
- Loads relaunch config via JSON (the same keys relaunch.py accepts)
- Builds tokenizers and tokenized training sets for both flows
- Prints key hparams, seeds, tokenizer properties, dataset sizes
- Prints N sample previews (lengths and short decoded snippets)

Usage examples (CPU friendly):

  python -m tacs.train.diagnose_alignment \
    --warmup-json /path/to/warmup.json \
    --relaunch-json /path/to/relaunch.json \
    --num-samples 3

Notes:
- This script does not train and avoids loading models by default.
- If you want tokenizer-only checks, keep defaults (no model load).
"""
import argparse
import json
import math
import os
import random
from typing import Any, Dict, List

from transformers import AutoTokenizer
from tacs.train.model_arguments import add_padding_to_tokenizer
from tacs.data_selection.get_training_dataset import get_training_dataset


def _decode_preview(tokenizer, token_ids: List[int], limit_tokens: int = 64, limit_chars: int = 200) -> str:
    if not token_ids:
        return ""
    piece = token_ids[:limit_tokens]
    try:
        text = tokenizer.decode(piece, skip_special_tokens=True)
    except Exception:
        text = str(piece)
    if len(text) > limit_chars:
        text = text[:limit_chars] + " …"
    return text


def _completion_ids_from_labels(input_ids, labels) -> List[int]:
    try:
        comp = [iid for iid, lab in zip(input_ids, labels) if (isinstance(lab, int) and lab >= 0)]
        return comp
    except Exception:
        return []


def _steps_per_epoch(num_examples: int, per_device_bs: int, grad_accum: int) -> int:
    eff = max(1, per_device_bs) * max(1, grad_accum)
    return max(1, math.ceil(num_examples / float(eff)))


def _print_header(title: str):
    print("\n== {} ==".format(title))


def _print_kv(label: str, value: Any):
    print(f"- {label}: {value}")


def _normalize_train_files(v) -> List[str]:
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        # allow space-separated
        return v.split()
    return []


def load_warmup_cfg(path: str) -> Dict[str, Any]:
    """Load warmup JSON as a flat dict without instantiating HF TrainingArguments.
    This avoids GPU/bf16 validation errors in CPU-only environments."""
    with open(os.path.abspath(path), "r") as f:
        cfg = json.load(f)
    return cfg


def load_relaunch_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    # minor normalization like relaunch.py does
    if "train_files" in cfg and isinstance(cfg["train_files"], (list, tuple)):
        cfg["train_files"] = " ".join(cfg["train_files"])
    return cfg


def build_train_dataset(train_files: List[str], tokenizer, max_seq_length: int, sample_percentage: float, seed: int):
    ds = get_training_dataset(train_files, tokenizer=tokenizer, max_seq_length=max_seq_length,
                              sample_percentage=sample_percentage, seed=seed)
    return ds


def preview_dataset(name: str, ds, tokenizer, num_samples: int = 3, seed: int = 0):
    _print_header(f"{name} Dataset Preview")
    try:
        total = len(ds)
    except Exception:
        total = None
    _print_kv("size", total)
    if not total:
        return
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(total), min(num_samples, total)))
    print(f"- samples (indices={idxs}):")
    for i in idxs:
        try:
            ex = ds[i]
        except Exception as e:
            print(f"  [#{i}] <failed to read: {e}>")
            continue
        input_ids = ex.get("input_ids", [])
        labels = ex.get("labels", [])
        attn = ex.get("attention_mask", [])
        valid_labels = sum(1 for x in labels if isinstance(x, int) and x >= 0) if isinstance(labels, list) else None
        _print_kv(f"  [#{i}] input_len", len(input_ids))
        _print_kv(f"  [#{i}] labels_valid", valid_labels)
        _print_kv(f"  [#{i}] attn_len", len(attn) if isinstance(attn, list) else None)
        # decoded previews
        inp_text = _decode_preview(tokenizer, input_ids)
        comp_ids = _completion_ids_from_labels(input_ids, labels) if isinstance(labels, list) else []
        comp_text = _decode_preview(tokenizer, comp_ids)
        print(f"  [#{i}] input_preview: {inp_text}")
        if comp_ids:
            print(f"  [#{i}] completion_preview: {comp_text}")


def main():
    ap = argparse.ArgumentParser(description="Print diagnostics to compare warmup vs relaunch without training")
    ap.add_argument("--warmup-json", required=True, help="Path to HF JSON for warmup (train.py)")
    ap.add_argument("--relaunch-json", required=True, help="Path to JSON for relaunch (relaunch.py)")
    ap.add_argument("--num-samples", type=int, default=3, help="Number of dataset examples to preview for each side")
    ap.add_argument("--decode", action="store_true", help="Decode and show short text previews (tokenizer required)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for choosing preview indices")
    args = ap.parse_args()

    # Load configs
    warm_cfg = load_warmup_cfg(args.warmup_json)
    rel_cfg = load_relaunch_cfg(args.relaunch_json)

    # Tokenizers (keep it light; no model load by default)
    warm_model_path = warm_cfg.get("model_name_or_path")
    tok_warm = AutoTokenizer.from_pretrained(warm_model_path)
    add_padding_to_tokenizer(tok_warm)
    tok_rel = AutoTokenizer.from_pretrained(rel_cfg.get("checkpoint_path"))
    if tok_rel.pad_token is None:
        tok_rel.add_special_tokens({"pad_token": "<pad>"})

    # Print high-level params (seeds/hparams)
    _print_header("Warmup (train.py) HParams")
    _print_kv("model_name_or_path", warm_model_path)
    _print_kv("seed(training)", warm_cfg.get("seed"))
    _print_kv("data_seed(TrainingArguments)", warm_cfg.get("data_seed"))
    _print_kv("sample_data_seed(DataArguments)", warm_cfg.get("sample_data_seed"))
    _print_kv("max_seq_length", warm_cfg.get("max_seq_length"))
    _print_kv("percentage", warm_cfg.get("percentage"))
    _print_kv("per_device_train_batch_size", warm_cfg.get("per_device_train_batch_size"))
    _print_kv("gradient_accumulation_steps", warm_cfg.get("gradient_accumulation_steps"))
    _print_kv("learning_rate", warm_cfg.get("learning_rate"))
    _print_kv("lr_scheduler_type", warm_cfg.get("lr_scheduler_type"))
    _print_kv("warmup_ratio", warm_cfg.get("warmup_ratio"))
    _print_kv("weight_decay", warm_cfg.get("weight_decay"))
    _print_kv("bf16", warm_cfg.get("bf16"))
    _print_kv("fp16", warm_cfg.get("fp16"))

    _print_header("Relaunch (relaunch.py) HParams")
    for k in [
        "checkpoint_path", "train_files", "k", "T", "seed", "val_sample_seed",
        "max_seq_length", "sample_percentage", "per_device_train_batch_size",
        "gradient_accumulation_steps", "base_lr", "lr_scheduler_type", "warmup_ratio",
        "weight_decay", "optim", "bf16", "fp16", "sample_data_seed",
    ]:
        _print_kv(k, rel_cfg.get(k))

    # Tokenizer summaries
    _print_header("Tokenizer Summary (warmup)")
    _print_kv("vocab_size", len(tok_warm))
    _print_kv("model_max_length", getattr(tok_warm, "model_max_length", None))
    _print_kv("padding_side", getattr(tok_warm, "padding_side", None))
    _print_kv("truncation_side", getattr(tok_warm, "truncation_side", None))

    _print_header("Tokenizer Summary (relaunch)")
    _print_kv("vocab_size", len(tok_rel))
    _print_kv("model_max_length", getattr(tok_rel, "model_max_length", None))
    _print_kv("padding_side", getattr(tok_rel, "padding_side", None))
    _print_kv("truncation_side", getattr(tok_rel, "truncation_side", None))

    # Training files
    warm_train_files = warm_cfg.get("train_files", []) if isinstance(warm_cfg.get("train_files", []), list) else []
    rel_train_files = _normalize_train_files(rel_cfg.get("train_files"))
    _print_header("Training Files")
    _print_kv("warmup.train_files", warm_train_files)
    _print_kv("relaunch.train_files", rel_train_files)

    # Build tokenized training datasets
    # Warmup
    if warm_train_files:
        ds_warm = build_train_dataset(
            warm_train_files,
            tok_warm,
            max_seq_length=int(warm_cfg.get("max_seq_length", 2048) or 2048),
            sample_percentage=float(warm_cfg.get("percentage", 1.0) or 1.0),
            seed=int(warm_cfg.get("sample_data_seed", 42) or 42),
        )
        steps_warm = _steps_per_epoch(
            len(ds_warm),
            int(warm_cfg.get("per_device_train_batch_size", 1) or 1),
            int(warm_cfg.get("gradient_accumulation_steps", 1) or 1),
        )
        _print_header("Warmup Derived")
        _print_kv("dataset_size", len(ds_warm))
        _print_kv("steps_per_epoch(ceil)", steps_warm)
        preview_dataset("Warmup", ds_warm, tok_warm, num_samples=args.num_samples, seed=args.seed)
    else:
        _print_header("Warmup Derived")
        _print_kv("dataset_size", "N/A (no train_files)")

    # Relaunch
    if rel_train_files:
        ds_rel = build_train_dataset(
            rel_train_files,
            tok_rel,
            max_seq_length=int(rel_cfg.get("max_seq_length", 2048) or 2048),
            sample_percentage=float(rel_cfg.get("sample_percentage", 1.0) or 1.0),
            seed=int(rel_cfg.get("sample_data_seed", 42) or 42),
        )
        steps_rel = _steps_per_epoch(
            len(ds_rel),
            int(rel_cfg.get("per_device_train_batch_size", 1) or 1),
            int(rel_cfg.get("gradient_accumulation_steps", 1) or 1),
        )
        _print_header("Relaunch Derived")
        _print_kv("dataset_size", len(ds_rel))
        _print_kv("steps_per_epoch(ceil)", steps_rel)
        # If trainer_state.json exists, try infer effective_k and remaining steps
        ckpt = rel_cfg.get("checkpoint_path")
        if ckpt and os.path.isdir(ckpt):
            ts_path = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(ts_path):
                try:
                    with open(ts_path, "r") as f:
                        ts = json.load(f)
                    eff_k = None
                    if "epoch" in ts and isinstance(ts["epoch"], (int, float)):
                        eff_k = int(round(ts["epoch"]))
                    elif "global_step" in ts and isinstance(ts["global_step"], (int, float)):
                        global_step = int(ts["global_step"])
                        eff_k = int(global_step // max(1, steps_rel))
                    _print_kv("effective_k(inferred)", eff_k)
                    T = int(rel_cfg.get("T", 0) or 0)
                    if eff_k is not None and T > 0:
                        remaining_epochs = max(0, T - eff_k)
                        _print_kv("T", T)
                        _print_kv("remaining_epochs(T-eff_k)", remaining_epochs)
                        _print_kv("remaining_steps(est)", remaining_epochs * steps_rel)
                except Exception as e:
                    _print_kv("effective_k(inferred)", f"failed: {e}")
        preview_dataset("Relaunch", ds_rel, tok_rel, num_samples=args.num_samples, seed=args.seed)
    else:
        _print_header("Relaunch Derived")
        _print_kv("dataset_size", "N/A (no train_files)")


if __name__ == "__main__":
    main()
