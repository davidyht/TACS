"""
ToV (Train on Validation) baseline scoring.

Algorithm (from the ToV paper):
  For each ToV warmup checkpoint theta_k:
    1. Load theta_k (base checkpoint).
    2. Perturb: fine-tune a copy on target-task validation data for 1 epoch -> theta_k'.
    3. Score each candidate z: s_k(z) = loss(z | theta_k) - loss(z | theta_k').
    4. Accumulate: S(z) += s_k(z) across all checkpoints.
  Select top-p fraction by accumulated score.

Usage:
  python -m tacs.data_selection.tov_score \
    --base_model meta-llama/Llama-2-7b-hf \
    --ckpt_dir $OUTPUT_ROOT/$TARGET_TASK/warmup_ckpts \
    --target_task tydiqa \
    --data_dir ../data \
    --train_file ../data/train/processed/flan_v2/flan_v2_data_less.jsonl \
    --train_file_name flan_v2 \
    --output_path $OUTPUT_ROOT \
    --perturb_lr 1e-4 \
    --perturb_epochs 1 \
    --bf16
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from tacs.data_selection.get_validation_dataset import get_dataset
from tacs.data_selection.get_info import load_model as load_checkpoint_model
from tacs.data_selection.val_warmup_loss_gap import (
    _per_sample_loss,
    _prepare_candidate_dataset,
)
from tacs.train.model_arguments import add_padding_to_tokenizer
from tacs.train.utils import set_seed
from tacs.data_selection.score_shards import shard_batch_range

MODEL_FEATURE_COLUMNS = {
    "input_ids",
    "attention_mask",
    "labels",
    "token_type_ids",
    "position_ids",
}


def _parse_args():
    ap = argparse.ArgumentParser(description="ToV baseline scoring")
    ap.add_argument("--base_model", type=str, required=True,
                    help="HuggingFace model name or path (e.g. meta-llama/Llama-2-7b-hf).")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Directory containing ToV warmup checkpoints (checkpoint-1/, checkpoint-2/, ...).")
    ap.add_argument("--target_task", type=str, required=True,
                    help="Target task for validation data (bbh, mmlu, tydiqa).")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Root data directory containing eval/ and train/ subdirectories.")
    ap.add_argument("--train_file", type=str, required=True,
                    help="Path to candidate training pool JSONL file.")
    ap.add_argument("--train_file_name", type=str, required=True,
                    help="Short name for the training source (e.g. flan_v2).")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Output root directory. Scores go to {output_path}/{target_task}/.")
    ap.add_argument("--score_ckpt_ids", nargs="+", type=int, default=None,
                    help="Optional: only score these checkpoint IDs (e.g. 1 2 3 4).")
    ap.add_argument("--skip_score_write", action="store_true",
                    help="Skip writing final influence scores (useful for partial shard jobs).")
    ap.add_argument("--score_shard", type=str, default=None,
                    help="k/K: base and perturbed passes over shard k (0-based) of K contiguous batch ranges on "
                         "--loss_save_interval boundaries (same batches and loss-chunk files as an unsharded "
                         "pass). Needs --skip_score_write.")
    ap.add_argument("--perturb_stage", choices=["inline", "save", "load"], default="inline",
                    help="inline: perturb in this job (default). save: per checkpoint, perturb and save the "
                         "perturbed trainable state to <output>/<task>/tov_perturb_state, no candidate passes. "
                         "load: load that state instead of perturbing, so every score shard uses one "
                         "perturbed model.")

    # Perturbation hyperparams
    ap.add_argument("--perturb_lr", type=float, default=1e-4,
                    help="Learning rate for validation perturbation.")
    ap.add_argument("--perturb_epochs", type=int, default=1,
                    help="Number of epochs for validation perturbation.")
    ap.add_argument("--perturb_optim", type=str, default="adamw",
                    choices=["sgd", "adamw"],
                    help="Optimizer for perturbation step.")
    ap.add_argument("--perturb_batch_size", type=int, default=1,
                    help="Batch size for perturbation training.")

    # LoRA config for perturbation
    ap.add_argument("--perturb_lora", action="store_true", default=True,
                    help="Use LoRA for perturbation (default: True).")
    ap.add_argument("--no_perturb_lora", dest="perturb_lora", action="store_false")
    ap.add_argument("--perturb_lora_r", type=int, default=1,
                    help="LoRA rank for perturbation.")
    ap.add_argument("--perturb_lora_alpha", type=int, default=4,
                    help="LoRA alpha for perturbation.")
    ap.add_argument("--perturb_lora_dropout", type=float, default=0.1)
    ap.add_argument("--perturb_lora_target_modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Scoring
    ap.add_argument("--loss_batch_size", type=int, default=4,
                    help="Batch size for loss computation on candidates.")
    ap.add_argument("--loss_save_interval", type=int, default=160,
                    help="Save loss cache chunks every N batches.")
    ap.add_argument("--candidate_percentage", type=float, default=1.0,
                    help="Fraction of candidate pool to score.")

    # General
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--torch_dtype", type=str, default=None,
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--chat_format", type=str, default="tulu")
    ap.add_argument("--no_chat_format", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    return ap.parse_args()


def _resolve_torch_dtype(args):
    if args.torch_dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return mapping.get(args.torch_dtype, "auto")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.bfloat16


def _load_warmup_meta(ckpt_dir: str) -> Dict:
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _discover_checkpoints(
    ckpt_dir: str,
    score_ckpt_ids: Optional[List[int]] = None,
) -> List[Tuple[int, str, str]]:
    """Discover warmup checkpoints from either step_*.pt or checkpoint-* directories."""
    step_entries: List[Tuple[int, str, str]] = []
    for fname in os.listdir(ckpt_dir):
        if not (fname.startswith("step_") and fname.endswith(".pt")):
            continue
        p = os.path.join(ckpt_dir, fname)
        if not os.path.isfile(p):
            continue
        try:
            ckpt_id = int(fname.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        step_entries.append((ckpt_id, p, "state"))
    if step_entries:
        entries = sorted(step_entries, key=lambda x: x[0])
    else:
        hf_entries = []
        for fname in os.listdir(ckpt_dir):
            if not fname.startswith("checkpoint-"):
                continue
            p = os.path.join(ckpt_dir, fname)
            if not os.path.isdir(p):
                continue
            try:
                raw_id = int(fname.split("checkpoint-")[1])
            except (ValueError, IndexError):
                continue
            hf_entries.append((raw_id, p))
        hf_entries.sort(key=lambda x: x[0])
        entries = [(i, path, "hf_dir") for i, (_, path) in enumerate(hf_entries, start=1)]
    if score_ckpt_ids is not None:
        desired = set(score_ckpt_ids)
        entries = [item for item in entries if item[0] in desired]
    if not entries:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}. "
            "Expected either step_*.pt files or checkpoint-* directories."
        )
    return entries


def _load_base_model(ckpt_path: str, torch_dtype, trust_remote_code: bool):
    """Load a warmup checkpoint (PEFT adapter or full model)."""
    model = load_checkpoint_model(
        ckpt_path,
        torch_dtype=torch_dtype if torch_dtype != "auto" else torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    return model


def _build_state_checkpoint_model(
    base_model_name: str,
    tokenizer,
    warmup_meta: Dict,
    torch_dtype,
    trust_remote_code: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype if torch_dtype != "auto" else torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if warmup_meta.get("lora"):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(warmup_meta.get("lora_r", 8)),
            lora_alpha=float(warmup_meta.get("lora_alpha", 32)),
            lora_dropout=float(warmup_meta.get("lora_dropout", 0.1)),
            target_modules=warmup_meta.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        model = get_peft_model(model, lora_config)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def _load_state_checkpoint_model(
    ckpt_path: str,
    base_model_name: str,
    tokenizer,
    warmup_meta: Dict,
    torch_dtype,
    trust_remote_code: bool,
):
    model = _build_state_checkpoint_model(
        base_model_name=base_model_name,
        tokenizer=tokenizer,
        warmup_meta=warmup_meta,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a dict-like checkpoint at {ckpt_path}, got {type(state)}")
    allow_partial_state = (
        str(warmup_meta.get("warmup_checkpoint_format", "")).strip()
        == "trainable_state_dict"
    )
    strict_ok = False
    try:
        model.load_state_dict(state, strict=True)
        strict_ok = True
    except RuntimeError as exc:
        if any(k.startswith("module.") for k in state.keys()):
            stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
            try:
                model.load_state_dict(stripped, strict=True)
                state = stripped
                strict_ok = True
            except Exception:
                pass
        if not strict_ok:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or (missing and not allow_partial_state):
                hint = (
                    "Common cause: LoRA config mismatch between warmup and ToV loading.\n"
                    f"Warmup meta lora_r={warmup_meta.get('lora_r')}, "
                    f"lora_alpha={warmup_meta.get('lora_alpha')}, "
                    f"lora_target_modules={warmup_meta.get('lora_target_modules')}."
                )
                raise RuntimeError(
                    f"Failed to load checkpoint state_dict for {ckpt_path}.\n"
                    f"Missing keys (sample): {missing[:5]}\n"
                    f"Unexpected keys (sample): {unexpected[:5]}\n"
                    + hint
                ) from exc
            print(
                f"[tov] partial state_dict load for {ckpt_path}: "
                f"loaded trainable-only checkpoint "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
    return model


def _apply_perturb_lora(model, args):
    """If model already has LoRA (from LESS warmup), merge it and add fresh LoRA for perturbation."""
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.perturb_lora_r,
        lora_alpha=args.perturb_lora_alpha,
        lora_dropout=args.perturb_lora_dropout,
        target_modules=args.perturb_lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def _perturbation_path(output_dir: str, ckpt_id: int) -> str:
    return os.path.join(output_dir, "tov_perturb_state", f"ckpt_{ckpt_id:04d}.pt")


def _trainable_fingerprint(state: Dict[str, torch.Tensor]) -> float:
    return float(sum(t.detach().double().abs().sum().item() for t in state.values()))


def _save_perturbation(model, path: str, avg_loss: float, meta: Dict) -> None:
    state = {n: p.detach().to("cpu").clone() for n, p in model.named_parameters() if p.requires_grad}
    blob = {"state": state, "avg_perturb_loss": float(avg_loss), "meta": meta,
            "fingerprint": _trainable_fingerprint(state), "n_tensors": len(state)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    print(f"[tov] saved perturbation {path}: tensors={len(state)} fingerprint={blob['fingerprint']:.10e}", flush=True)


def _load_perturbation(model, path: str) -> float:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"saved perturbation not found: {path} (run --perturb_stage save first)")
    blob = torch.load(path, map_location="cpu")
    state = blob["state"]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != trainable:
        raise RuntimeError(f"saved perturbation keys do not match the perturbation LoRA: "
                           f"{len(set(state) - trainable)} extra, {len(trainable - set(state))} missing")
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys loading {path}: {unexpected[:5]}")
    loaded = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fp = _trainable_fingerprint(loaded)
    print(f"[tov] loaded perturbation {path}: tensors={len(state)} fingerprint={fp:.10e} "
          f"saved={blob['fingerprint']:.10e} avg val loss {blob['avg_perturb_loss']:.6f}", flush=True)
    if fp != blob["fingerprint"]:
        raise RuntimeError(f"perturbation fingerprint changed on load: {fp} != {blob['fingerprint']}")
    return float(blob["avg_perturb_loss"])


def _perturb_model(model, val_loader, device, args, use_amp, amp_dtype):
    """Fine-tune model on validation data for perturb_epochs epochs. Modifies model in-place."""
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters for perturbation.")

    if args.perturb_optim == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.perturb_lr)
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=args.perturb_lr)

    try:
        scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    total_loss = 0.0
    total_steps = 0
    for epoch in range(args.perturb_epochs):
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss


def _compute_losses(model, dataloader, device, use_amp, amp_dtype,
                    cache_dir=None, step_id=None, save_interval=160, batch_range=None):
    # batch_range=(start, end]: score only 1-based batches start < i <= end (one score shard).
    """Compute per-example loss for all examples in dataloader. Returns 1-D CPU tensor."""
    model.eval()
    losses = []
    use_cache = cache_dir is not None and step_id is not None
    save_interval = max(1, int(save_interval))
    max_index = -1

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        saved_files = sorted(
            [f for f in os.listdir(cache_dir) if f.startswith("losses-") and f.endswith(".pt")],
            key=lambda x: int(x.split(".")[0].split("-")[1]),
        )
        range_prev = batch_range[0] if batch_range is not None else None
        for fname in saved_files:
            idx = int(fname.split(".")[0].split("-")[1])
            if batch_range is not None:
                # Resume a shard from its own contiguous chunks only.
                if idx <= batch_range[0] or idx > batch_range[1]:
                    continue
                if idx - range_prev > save_interval:
                    break
                range_prev = idx
            elif max_index != -1 and idx - max_index > save_interval:
                break
            try:
                chunk = torch.load(os.path.join(cache_dir, fname), map_location="cpu")
            except Exception:
                break
            losses.append(chunk)
            max_index = idx

    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        pass

    pending = []
    last_done = None
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader, start=1),
            total=total_batches,
            desc=f"tov loss step={step_id}",
            ncols=100,
        ):
            if batch_range is not None:
                if batch_idx > batch_range[1]:
                    break
                if batch_idx <= batch_range[0]:
                    continue
            if use_cache and batch_idx <= max_index:
                continue
            last_done = batch_idx
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(**batch)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss_per_sample = _per_sample_loss(logits, batch["labels"])
            else:
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                loss_per_sample = _per_sample_loss(logits, batch["labels"])
            loss_cpu = loss_per_sample.detach().cpu()
            losses.append(loss_cpu)
            if use_cache:
                pending.append(loss_cpu)
                if batch_idx % save_interval == 0:
                    tmp_path = os.path.join(cache_dir, f"losses-{batch_idx}.pt.tmp")
                    final_path = os.path.join(cache_dir, f"losses-{batch_idx}.pt")
                    torch.save(torch.cat(pending, dim=0), tmp_path)
                    os.replace(tmp_path, final_path)
                    pending = []
        if use_cache and pending:
            try:
                last_idx = batch_idx
            except NameError:
                last_idx = max_index
            if batch_range is not None:
                last_idx = last_done
            tmp_path = os.path.join(cache_dir, f"losses-{last_idx}.pt.tmp")
            final_path = os.path.join(cache_dir, f"losses-{last_idx}.pt")
            torch.save(torch.cat(pending, dim=0), tmp_path)
            os.replace(tmp_path, final_path)

    return torch.cat(losses, dim=0) if losses else torch.tensor([])


def _load_cached_losses(cache_dir: str, expected_len: Optional[int] = None) -> torch.Tensor:
    """Load all cached loss chunks from a cache directory."""
    saved_files = sorted(
        [f for f in os.listdir(cache_dir) if f.startswith("losses-") and f.endswith(".pt")],
        key=lambda x: int(x.split(".")[0].split("-")[1]),
    )
    chunks = []
    for fname in saved_files:
        chunks.append(torch.load(os.path.join(cache_dir, fname), map_location="cpu"))
    if not chunks:
        raise FileNotFoundError(f"No cached losses found in {cache_dir}")
    result = torch.cat(chunks, dim=0)
    if expected_len is not None and len(result) != expected_len:
        raise RuntimeError(
            f"Cached loss length {len(result)} != expected {expected_len} in {cache_dir}"
        )
    return result


def _cache_has_expected_losses(cache_dir: str, expected_len: int) -> bool:
    if not os.path.isdir(cache_dir):
        return False
    try:
        return len(_load_cached_losses(cache_dir, expected_len=None)) == expected_len
    except Exception:
        return False


def _sanitize_model_dataset(dataset, dataset_name: str):
    """Drop metadata columns that break the HF collator/model forward path."""
    extra_columns = [c for c in dataset.column_names if c not in MODEL_FEATURE_COLUMNS]
    if extra_columns:
        print(
            f"[tov] dropping non-model columns from {dataset_name}: {extra_columns}",
            flush=True,
        )
        dataset = dataset.remove_columns(extra_columns)
        dataset.set_format(type="pt")
    return dataset


def main():
    args = _parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_torch_dtype(args)
    use_amp = args.bf16 or args.fp16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    use_chat_format = not args.no_chat_format
    data_dir = args.data_dir
    if os.path.basename(os.path.normpath(data_dir)) == "eval":
        data_dir = os.path.dirname(os.path.normpath(data_dir))

    warmup_meta = _load_warmup_meta(args.ckpt_dir)
    base_model_name = args.base_model
    if warmup_meta.get("model_name_or_path"):
        if warmup_meta["model_name_or_path"] != args.base_model:
            print(
                "[tov] warning: base_model differs from warmup meta; "
                f"using warmup meta model {warmup_meta['model_name_or_path']} "
                f"instead of requested {args.base_model}",
                flush=True,
            )
        base_model_name = warmup_meta["model_name_or_path"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=args.trust_remote_code
    )
    add_padding_to_tokenizer(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")

    # Output dir
    output_dir = os.path.join(args.output_path, args.target_task)
    os.makedirs(output_dir, exist_ok=True)

    # Discover checkpoints
    ckpt_entries = _discover_checkpoints(args.ckpt_dir, args.score_ckpt_ids)
    print(
        f"[tov] found {len(ckpt_entries)} checkpoints: "
        f"{[(cid, kind, os.path.basename(p)) for cid, p, kind in ckpt_entries]}",
        flush=True,
    )
    if any(kind == "state" for _, _, kind in ckpt_entries) and not warmup_meta:
        print(
            "[tov] warning: step_*.pt checkpoints found without warmup meta.json; "
            "loading may fail if LoRA settings are not encoded in args.base_model.",
            flush=True,
        )

    # Load validation data (for perturbation)
    val_ds = get_dataset(
        args.target_task,
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        use_chat_format=use_chat_format,
        chat_format=args.chat_format,
    )
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset for {args.target_task} is empty.")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.perturb_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    print(f"[tov] validation examples: {len(val_ds)}", flush=True)

    # Load candidate pool
    candidate_ds = _prepare_candidate_dataset(
        args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        percentage=args.candidate_percentage,
        seed=args.seed,
        chat_format=args.chat_format,
    )
    candidate_ds = _sanitize_model_dataset(candidate_ds, "candidate dataset")
    candidate_loader = DataLoader(
        candidate_ds,
        batch_size=args.loss_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    print(f"[tov] candidate pool: {len(candidate_ds)} examples from {args.train_file_name}",
          flush=True)
    shard_range = None
    if args.score_shard:
        m = re.fullmatch(r"(\d+)/(\d+)", str(args.score_shard).strip())
        if not m:
            raise ValueError(f"--score_shard must be k/K, got {args.score_shard!r}")
        if not args.skip_score_write:
            raise ValueError("--score_shard scores part of the pool; use it with --skip_score_write")
        if args.perturb_stage != "load" and os.environ.get("TOV_ALLOW_INDEPENDENT_SHARD_PERTURBATION") != "1":
            # A shard that perturbs by itself scores against its own perturbed model; the perturbation is not
            # bitwise reproducible (bench 3148801: avg val loss 0.296078 vs 0.296223). Shards must load one
            # saved perturbation (--perturb_stage load).
            raise ValueError("--score_shard needs --perturb_stage load (one saved perturbation per checkpoint)")
        shard_range = shard_batch_range(len(candidate_loader), int(m.group(1)), int(m.group(2)),
                                        args.loss_save_interval)
        if shard_range is None:
            raise ValueError(f"score shard {args.score_shard} is empty for {len(candidate_loader)} batches")
        print(f"[tov] score shard {args.score_shard}: batches ({shard_range[0]}, {shard_range[1]}] "
              f"of {len(candidate_loader)}", flush=True)

    # Accumulate scores across checkpoints
    accumulated_score = None
    n_ckpts = 0

    for ckpt_id, ckpt_path, ckpt_kind in ckpt_entries:
        print(f"\n[tov] === checkpoint {ckpt_id} ({ckpt_kind}): {ckpt_path} ===", flush=True)

        if args.perturb_stage == "save":
            if not args.perturb_lora:
                raise ValueError("--perturb_stage save needs --perturb_lora (only the LoRA state is saved)")
            perturb_path = _perturbation_path(output_dir, ckpt_id)
            if os.path.isfile(perturb_path):
                print(f"[tov] perturbation already saved: {perturb_path}", flush=True)
                continue
            if ckpt_kind == "state":
                model = _load_state_checkpoint_model(
                    ckpt_path=ckpt_path,
                    base_model_name=base_model_name,
                    tokenizer=tokenizer,
                    warmup_meta=warmup_meta,
                    torch_dtype=torch_dtype,
                    trust_remote_code=args.trust_remote_code,
                )
            else:
                model = _load_base_model(ckpt_path, torch_dtype, args.trust_remote_code)
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model = _apply_perturb_lora(model, args)
            model.to(device)
            model.print_trainable_parameters()
            avg_perturb_loss = _perturb_model(model, val_loader, device, args, use_amp, amp_dtype)
            print(f"[tov] perturbation done, avg val loss: {avg_perturb_loss:.6f}", flush=True)
            _save_perturbation(model, perturb_path, avg_perturb_loss, {
                "ckpt_id": ckpt_id, "ckpt_path": ckpt_path, "target_task": args.target_task,
                "perturb_lr": args.perturb_lr, "perturb_epochs": args.perturb_epochs,
                "perturb_optim": args.perturb_optim, "perturb_batch_size": args.perturb_batch_size,
                "perturb_lora_r": args.perturb_lora_r, "perturb_lora_alpha": args.perturb_lora_alpha,
                "perturb_lora_dropout": args.perturb_lora_dropout, "seed": args.seed,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Cache directories for this checkpoint
        score_cache_root = os.path.join(output_dir, "tov_score_cache", args.train_file_name)
        base_cache_dir = os.path.join(score_cache_root, f"ckpt_{ckpt_id:04d}_base")
        perturbed_cache_dir = os.path.join(score_cache_root, f"ckpt_{ckpt_id:04d}_perturbed")

        # --- Step 1: Compute base checkpoint losses ---
        base_cache_complete = (
            _cache_has_expected_losses(base_cache_dir, len(candidate_ds))
        )
        perturbed_cache_complete = (
            _cache_has_expected_losses(perturbed_cache_dir, len(candidate_ds))
        )

        if shard_range is None and base_cache_complete and perturbed_cache_complete:
            print(f"[tov] loading cached losses for ckpt {ckpt_id}", flush=True)
            base_losses = _load_cached_losses(base_cache_dir, expected_len=len(candidate_ds))
            perturbed_losses = _load_cached_losses(perturbed_cache_dir, expected_len=len(candidate_ds))
        else:
            # Load the warmup checkpoint
            print(f"[tov] loading checkpoint: {ckpt_path}", flush=True)
            if ckpt_kind == "state":
                model = _load_state_checkpoint_model(
                    ckpt_path=ckpt_path,
                    base_model_name=base_model_name,
                    tokenizer=tokenizer,
                    warmup_meta=warmup_meta,
                    torch_dtype=torch_dtype,
                    trust_remote_code=args.trust_remote_code,
                )
            else:
                model = _load_base_model(ckpt_path, torch_dtype, args.trust_remote_code)
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))
            model.to(device)

            # Compute losses on candidate pool with base checkpoint
            print(f"[tov] computing base losses (ckpt {ckpt_id})...", flush=True)
            base_losses = _compute_losses(
                model, candidate_loader, device, use_amp, amp_dtype,
                cache_dir=base_cache_dir, step_id=ckpt_id,
                save_interval=args.loss_save_interval, batch_range=shard_range,
            )
            print(f"[tov] base losses: n={len(base_losses)} "
                  f"mean={base_losses.mean():.6f} std={base_losses.std():.6f}", flush=True)

            # --- Step 2: Perturb with validation data ---
            print(f"[tov] perturbing with {args.perturb_epochs} epoch(s) on {args.target_task} val...",
                  flush=True)
            if args.perturb_lora:
                model = _apply_perturb_lora(model, args)
                model.to(device)
                model.print_trainable_parameters()

            if args.perturb_stage == "load":
                avg_perturb_loss = _load_perturbation(model, _perturbation_path(output_dir, ckpt_id))
            else:
                avg_perturb_loss = _perturb_model(
                    model, val_loader, device, args, use_amp, amp_dtype
                )
            print(f"[tov] perturbation done, avg val loss: {avg_perturb_loss:.6f}", flush=True)

            # --- Step 3: Compute losses on candidate pool with perturbed model ---
            print(f"[tov] computing perturbed losses (ckpt {ckpt_id})...", flush=True)
            perturbed_losses = _compute_losses(
                model, candidate_loader, device, use_amp, amp_dtype,
                cache_dir=perturbed_cache_dir, step_id=ckpt_id,
                save_interval=args.loss_save_interval, batch_range=shard_range,
            )
            print(f"[tov] perturbed losses: n={len(perturbed_losses)} "
                  f"mean={perturbed_losses.mean():.6f} std={perturbed_losses.std():.6f}", flush=True)

            # Free GPU memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # --- Step 4: Score = base_loss - perturbed_loss ---
        # Higher score = perturbed model has lower loss = validation training helped this example more
        ckpt_score = base_losses - perturbed_losses
        print(f"[tov] ckpt {ckpt_id} score: mean={ckpt_score.mean():.6f} "
              f"std={ckpt_score.std():.6f} min={ckpt_score.min():.6f} max={ckpt_score.max():.6f}",
              flush=True)

        if accumulated_score is None:
            accumulated_score = ckpt_score.clone()
        else:
            accumulated_score += ckpt_score
        n_ckpts += 1

    # --- Write final scores ---
    if accumulated_score is not None and not args.skip_score_write:
        score_path = os.path.join(output_dir, f"{args.train_file_name}_influence_score.pt")
        torch.save(accumulated_score, score_path)
        print(f"\n[tov] saved final scores to {score_path} (n={len(accumulated_score)}, "
              f"ckpts={n_ckpts})", flush=True)
        print(f"[tov] final score stats: mean={accumulated_score.mean():.6f} "
              f"std={accumulated_score.std():.6f}", flush=True)

        # Save metadata
        meta_path = os.path.join(output_dir, "tov_meta.json")
        meta = {
            "method": "tov",
            "target_task": args.target_task,
            "train_file": args.train_file,
            "train_file_name": args.train_file_name,
            "base_model": base_model_name,
            "ckpt_dir": args.ckpt_dir,
            "n_checkpoints": n_ckpts,
            "checkpoint_ids": [cid for cid, _, _ in ckpt_entries],
            "perturb_lr": args.perturb_lr,
            "perturb_epochs": args.perturb_epochs,
            "perturb_optim": args.perturb_optim,
            "perturb_lora": args.perturb_lora,
            "perturb_lora_r": args.perturb_lora_r if args.perturb_lora else None,
            "perturb_lora_alpha": args.perturb_lora_alpha if args.perturb_lora else None,
            "candidate_percentage": args.candidate_percentage,
            "n_candidates": len(accumulated_score),
            "seed": args.seed,
            "score_mean": float(accumulated_score.mean()),
            "score_std": float(accumulated_score.std()),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[tov] saved metadata to {meta_path}", flush=True)
    elif args.skip_score_write:
        print(f"[tov] skip_score_write: scores computed but not written (ckpts={n_ckpts})", flush=True)
    else:
        print("[tov] warning: no checkpoints processed, no scores written.", flush=True)


if __name__ == "__main__":
    main()
