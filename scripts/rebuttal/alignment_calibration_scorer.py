#!/usr/bin/env python3
"""Trajectory<->holdout alignment calibration scorer (verification of the
alignment-vs-AUROC idea; see
analysis/tacs_algorithm_improvement_20260731/calibration_trajectory_alignment_brainstorm.md).

Iterates every (fold, lr, depth) cell of a val-warmup HP grid produced by
submit_val_warmup_hp_grid_search.sh and, at each cell's warmup endpoint
(step_{depth}.pt), computes gradient-direction alignment signals in LoRA
parameter space:

  g_train = grad of mean loss over the fold proxy-train examples (warmup_indices)
  g_hold  = grad of mean loss over the fold held-out target examples (probe_indices)
  g_pool  = grad of mean loss over the shared generic reference (100 Aya rows)
  theta   = current trainable LoRA params (net displacement from 0-init)

Metrics (cosine in [-1,1], non-saturating -> resolves what AUROC cannot):
  cos_train_hold = cos(g_train, g_hold)          # HEADLINE: warmup pull aligned w/ holdout?
  cos_disp_hold  = cos(theta, -g_hold)           # net displacement vs holdout descent
  cos_train_pool = cos(g_train, g_pool)          # generic-movement control
  spec_train     = cos_train_hold - cos_train_pool   # specificity (O_spec, grad form)
  cos_hold_pool  = cos(g_hold, g_pool)

Checkpoint format note: warmup checkpoints are per-step trainable LoRA state
dicts (step_N.pt, peft-named keys), NOT adapter dirs. We build the peft model
once from meta.json and load_state_dict(strict=False) per cell.
"""
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model

from tacs.data_selection.val_warmup_loss_gap import (
    _iter_trainable_lora_params,
    _prepare_candidate_dataset,
    _load_warmup_meta,
)
from tacs.data_selection.get_info import load_model as load_checkpoint_model
from tacs.data_selection.get_validation_dataset import get_dataset
from tacs.train.model_arguments import add_padding_to_tokenizer


def _build_peft(base_model: str, meta: Dict, trust_remote_code: bool):
    base = load_checkpoint_model(base_model, trust_remote_code=trust_remote_code)
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, bias="none",
        r=int(meta["lora_r"]), lora_alpha=int(meta["lora_alpha"]),
        lora_dropout=float(meta.get("lora_dropout", 0.1)),
        target_modules=list(meta["lora_target_modules"]),
        rank_pattern=dict(meta.get("lora_rank_pattern") or {}),
        alpha_pattern=dict(meta.get("lora_alpha_pattern") or {}),
    )
    return get_peft_model(base, cfg)


def _grad(model, dataset, device, max_examples: int) -> Optional[torch.Tensor]:
    if dataset is None or len(dataset) == 0:
        return None
    if max_examples and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    collator = DataCollatorForSeq2Seq(tokenizer=model._align_tok, model=model, padding="longest")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    model.zero_grad(set_to_none=True)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        (loss / len(loader)).backward()
    named = _iter_trainable_lora_params(model)
    chunks = [p.grad.detach().float().view(-1).cpu() for _, p in named if p.grad is not None]
    model.zero_grad(set_to_none=True)
    return torch.cat(chunks) if chunks else None


def _theta(model) -> torch.Tensor:
    return torch.cat([p.detach().float().view(-1).cpu() for _, p in _iter_trainable_lora_params(model)])


def _cos(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True, help=".../tasks/<task> root (has grid/ and splits/)")
    ap.add_argument("--reference-file", "--pool-file", dest="pool_file", required=True,
                    help="Fixed generic reference JSONL; the paper uses 100 Latin-script Aya rows")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", required=True, choices=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--depths", default="4 8 12 16")
    ap.add_argument("--lrs", default=None, help="Optional space-separated LR directory labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--mmlu-n-shot", type=int, default=None,
                    help="MMLU proxy rows per subject; overrides warmup meta.json (older metas omit it and would fall back to 1)")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split()]
    requested_lrs = set(args.lrs.split()) if args.lrs else None
    grid_root = os.path.join(args.study_root, "grid")
    split_dir = os.path.join(args.study_root, "splits")
    # discover one meta.json to configure the shared model
    any_meta_dir = os.path.dirname(sorted(glob.glob(os.path.join(grid_root, "fold_*/lr_*/warmup/*/warmup_ckpts/meta.json")))[0])
    meta = _load_warmup_meta(any_meta_dir)
    base_model = args.base_model or meta["model_name_or_path"]
    chat_format = meta.get("chat_format", "tokenizer")
    max_seq = int(meta.get("max_seq_length", 2048))
    mmlu_n_shot = args.mmlu_n_shot or int(meta.get("mmlu_n_shot", 1) or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    add_padding_to_tokenizer(tok)
    model = _build_peft(base_model, meta, args.trust_remote_code).to(device)
    # Specificity compares deterministic local gradient directions.  Keep LoRA
    # dropout disabled while taking those gradients; leaving the model in train
    # mode makes the selected HP depend on stochastic dropout masks rather than
    # on the declared validation-warmup split.
    model.eval()
    model._align_tok = tok

    # datasets built once; folds select different indices
    full = get_dataset(args.task, data_dir=args.data_dir, tokenizer=tok, max_length=max_seq,
                       use_chat_format=True, chat_format=chat_format, mmlu_n_shot=mmlu_n_shot)
    ds_pool = _prepare_candidate_dataset(args.pool_file, tok, max_seq, percentage=1.0, seed=42, chat_format=chat_format)

    rows: List[Dict] = []
    for split_file in sorted(glob.glob(os.path.join(split_dir, "fold_*.json"))):
        split = json.load(open(split_file))
        fold = re.search(r"fold_(\d+)", split_file).group(1)
        if max(split["warmup_indices"] + split["probe_indices"]) >= len(full):
            raise SystemExit(f"{split_file}: split indices exceed the {len(full)}-row proxy; check --mmlu-n-shot")
        warm_idx = list(split["warmup_indices"])
        hold_idx = list(split["probe_indices"])
        ds_train, ds_hold = full.select(warm_idx), full.select(hold_idx)
        for lr_dir in sorted(glob.glob(os.path.join(grid_root, f"fold_{fold}", "lr_*"))):
            lr = os.path.basename(lr_dir).replace("lr_", "")
            if requested_lrs is not None and lr not in requested_lrs:
                continue
            ck = glob.glob(os.path.join(lr_dir, "warmup", "*", "warmup_ckpts"))
            if not ck:
                continue
            ck = ck[0]
            for d in depths:
                step = os.path.join(ck, f"step_{d}.pt")
                if not os.path.exists(step):
                    continue
                sd = torch.load(step, map_location="cpu")
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if unexpected:
                    print(f"[warn] unexpected keys at {step}: {unexpected[:2]}...")
                g_tr = _grad(model, ds_train, device, args.max_examples)
                g_ho = _grad(model, ds_hold, device, args.max_examples)
                g_po = _grad(model, ds_pool, device, args.max_examples)
                th = _theta(model)
                cth = _cos(g_tr, g_ho); ctp = _cos(g_tr, g_po)
                row = {
                    "fold": int(fold), "lr": lr, "depth": d,
                    "n_train": len(ds_train), "n_hold": len(ds_hold), "n_pool": min(len(ds_pool), args.max_examples),
                    "cos_train_hold": cth, "cos_disp_hold": _cos(th, (-g_ho) if g_ho is not None else None),
                    "cos_train_pool": ctp,
                    "spec_train": (cth - ctp) if (cth is not None and ctp is not None) else None,
                    "cos_hold_pool": _cos(g_ho, g_po),
                }
                rows.append(row)
                print(json.dumps(row))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"base_model": base_model, "task": args.task, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"[done] wrote {len(rows)} cells -> {args.out}")


if __name__ == "__main__":
    main()
