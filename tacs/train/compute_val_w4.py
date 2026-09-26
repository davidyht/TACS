#!/usr/bin/env python
# coding: utf-8
"""
Compute w4 = (1 - beta) * <P_T g_val(T), g_val(T)> / ||g_val(T)||_2 for a validation task.
P_T is the Adam preconditioner from checkpoint-T optimizer state.
Also report w4 scaled by the average LR between epochs (k-1, k).
"""
import argparse
import json
import math
import os
import random
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, get_scheduler

from tacs.data_selection.collect_grad_reps import prepare_optimizer_state
from tacs.data_selection.get_validation_dataset import get_dataset, get_dataloader
from tacs.data_selection.get_info import load_model
from tacs.train.utils import estimate_full_gradient


def _find_optimizer_state(checkpoint_path: str, override: Optional[str]) -> str:
    if override:
        return override
    candidates = [
        os.path.join(checkpoint_path, "optimizer_with_names.pt"),
        os.path.join(checkpoint_path, "optimizer.pt"),
        os.path.join(checkpoint_path, "optimizer.bin"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No optimizer state found under {checkpoint_path}")


def _compute_avg_lr(checkpoint_path: str, epoch_end: int) -> Tuple[Optional[float], Optional[str]]:
    args_path = os.path.join(checkpoint_path, "training_args.bin")
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    try:
        train_args = torch.load(args_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"compute_val_w4: failed to read training_args.bin ({e}); skipping avg_lr", flush=True)
        return None, None
    try:
        with open(state_path, "r") as f:
            trainer_state = json.load(f)
    except Exception as e:
        print(f"compute_val_w4: failed to read trainer_state.json ({e}); skipping avg_lr", flush=True)
        return None, None

    max_steps = int(trainer_state.get("max_steps", 0) or 0)
    num_epochs = float(trainer_state.get("num_train_epochs", 0) or 0)
    if max_steps <= 0 or num_epochs <= 0:
        print("compute_val_w4: trainer_state.json missing max_steps/num_train_epochs; skipping avg_lr", flush=True)
        return None, None

    steps_per_epoch = int(math.ceil(max_steps / num_epochs))
    epoch_end = max(1, int(epoch_end))
    start_step = (epoch_end - 1) * steps_per_epoch
    end_step = epoch_end * steps_per_epoch

    base_lr = float(getattr(train_args, "learning_rate", 0.0))
    warmup_steps = int(getattr(train_args, "warmup_steps", 0) or 0)
    warmup_ratio = float(getattr(train_args, "warmup_ratio", 0.0) or 0.0)
    if warmup_steps == 0 and warmup_ratio > 0:
        warmup_steps = int(max_steps * warmup_ratio)
    sched_type = getattr(train_args, "lr_scheduler_type", "linear")
    sched_type = str(sched_type).replace("SchedulerType.", "").lower()

    dummy_param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.AdamW([dummy_param], lr=base_lr)
    scheduler = get_scheduler(
        sched_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    lrs = []
    for step in range(max_steps):
        scheduler.step()
        if start_step <= step < end_step:
            lrs.append(optimizer.param_groups[0]["lr"])
    if not lrs:
        return None, None
    return sum(lrs) / float(len(lrs)), "scheduler_avg"


def main():
    ap = argparse.ArgumentParser(description="Compute w4 at checkpoint-T for a validation task")
    ap.add_argument("--checkpoint_path", required=True, help="Path to checkpoint-T")
    ap.add_argument("--validation_task", required=True, choices=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--data_dir", type=str, default="..")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--adam_eps", type=float, default=1e-8)
    ap.add_argument("--optimizer_state_path", type=str, default=None)
    ap.add_argument("--val_sample_seed", type=int, default=42)
    ap.add_argument("--val_sample_size", type=int, default=None)
    ap.add_argument("--val_sample_percentage", type=float, default=None)
    ap.add_argument("--avg_lr_epoch_end", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = load_model(args.checkpoint_path)
    model.to(device)

    dataset = get_dataset(
        args.validation_task,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    try:
        if args.val_sample_size is not None or args.val_sample_percentage is not None:
            total = len(dataset)
            if args.val_sample_size is not None:
                want = int(args.val_sample_size)
            else:
                want = int(total * float(args.val_sample_percentage))
            want = max(1, min(total, want))
            rng = random.Random(args.val_sample_seed)
            indices = rng.sample(range(total), want)
            if hasattr(dataset, "select"):
                dataset = dataset.select(indices)
            else:
                print("compute_val_w4: dataset does not support select(); using full dataset", flush=True)
            print(f"compute_val_w4: sampled {want} examples from {total}", flush=True)
    except Exception as e:
        print(f"compute_val_w4: sampling failed ({e}); using full dataset", flush=True)
    val_loader = get_dataloader(dataset, tokenizer, batch_size=1)

    opt_path = _find_optimizer_state(args.checkpoint_path, args.optimizer_state_path)
    opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
    avg, avg_sq = prepare_optimizer_state(model, opt_state, device)
    precond_concat = 1.0 / torch.sqrt(avg_sq + float(args.adam_eps))
    precond_concat = precond_concat.detach().cpu()

    avg_grads, precond_norm_sq, meta = estimate_full_gradient(
        model,
        val_loader,
        device,
        verbose=True,
        transfer_interval=10,
        heartbeat_interval=10,
        precond_concat=precond_concat,
    )

    g_norm_sq = 0.0
    for g in avg_grads:
        g_norm_sq += float(g.view(-1).double().pow(2).sum().item())
    g_norm = float(g_norm_sq ** 0.5)

    w4 = (1.0 - float(args.beta)) * float(precond_norm_sq) / g_norm if g_norm else float("nan")
    epoch_end = args.avg_lr_epoch_end
    if epoch_end is None:
        try:
            with open(os.path.join(args.checkpoint_path, "trainer_state.json"), "r") as f:
                ts = json.load(f)
            epoch_end = int(round(float(ts.get("epoch", 0) or 0)))
        except Exception:
            epoch_end = None
    avg_lr, lr_source = (None, None)
    if epoch_end is not None and epoch_end > 0:
        avg_lr, lr_source = _compute_avg_lr(args.checkpoint_path, epoch_end)
    w4_scaled = w4 * float(avg_lr) if avg_lr is not None else None
    result = {
        "checkpoint": args.checkpoint_path,
        "validation_task": args.validation_task,
        "beta": float(args.beta),
        "adam_eps": float(args.adam_eps),
        "precond_quadratic_form": float(precond_norm_sq),
        "grad_norm_sq": float(g_norm_sq),
        "grad_norm": float(g_norm),
        "w4": float(w4),
        "avg_lr_epoch_end": epoch_end,
        "avg_lr": float(avg_lr) if avg_lr is not None else None,
        "avg_lr_source": lr_source,
        "w4_scaled_by_avg_lr": float(w4_scaled) if w4_scaled is not None else None,
        "grad_meta": meta,
        "optimizer_state_path": opt_path,
    }

    out_path = os.path.join(args.output_dir, f"w4_{args.validation_task}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
