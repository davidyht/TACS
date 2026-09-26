import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from datasets import Dataset, load_dataset

from tacs.data_selection.get_training_dataset import encode_data, _resolve_train_files
from tacs.data_selection.chat_format_utils import apply_chat_template_prompt, resolve_chat_format
from tacs.data_selection.get_validation_dataset import (
    DEFAULT_MMLU_N_SHOT,
    get_dataset,
    resolve_mmlu_n_shot,
)
from tacs.data_selection.validation_grouping import load_validation_group_assignments
from tacs.data_selection.get_test_dataset import get_tydiqa_dataset, get_tydiqa_goldp_dataset, get_bbh_test_dataset, get_mmlu_dataset as get_mmlu_test_dataset
from tacs.data_selection.get_info import load_model as load_checkpoint_model
from tacs.train.model_arguments import add_padding_to_tokenizer
from tacs.data_selection.score_shards import shard_batch_range
from tacs.train.utils import set_seed


def _parse_module_int_pattern(
    values: List[str],
    *,
    name: str,
    minimum: int = 1,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"{name}: expected MODULE=INTEGER, got {raw!r}")
        module, value_raw = raw.split("=", 1)
        module = module.strip()
        if not module or module in result:
            raise argparse.ArgumentTypeError(
                f"{name}: module names must be non-empty and unique, got {module!r}"
            )
        try:
            value = int(value_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name}: expected integer value in {raw!r}"
            ) from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"{name}: values must be >= {minimum}, got {raw!r}"
            )
        result[module] = value
    return result


def _atomic_json_dump(obj, path: str, *, indent: Optional[int] = 2) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_json_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_torch_save(obj, path: str) -> None:
    """Write a torch artifact without exposing a partially-written cache."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_torch_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_warmup_dataset(
    *,
    target_task: str,
    data_dir: str,
    tokenizer,
    max_length: int,
    use_chat_format: bool,
    chat_format: str,
    mmlu_n_shot: int,
    warmup_data_file: Optional[str],
    warmup_data_sample_size: int,
    seed: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Load canonical validation data or an explicit target-like proxy JSONL."""
    if not warmup_data_file:
        dataset = get_dataset(
            target_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            use_chat_format=use_chat_format,
            chat_format=chat_format,
            mmlu_n_shot=mmlu_n_shot,
        )
        return dataset, {"kind": "canonical_validation", "file": None, "sha256": None}

    resolved = os.path.expandvars(os.path.expanduser(str(warmup_data_file))).format(
        task=target_task
    )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"warmup proxy file not found: {resolved}")
    raw = load_dataset("json", data_files=[resolved])["train"]
    original_rows = len(raw)
    if warmup_data_sample_size > 0 and warmup_data_sample_size < len(raw):
        raw = _sample_hf_dataset(raw, warmup_data_sample_size, seed)
    dataset = encode_data(raw, tokenizer, max_length, chat_format=chat_format)
    keep = {"input_ids", "labels", "attention_mask"}
    drop = [name for name in dataset.column_names if name not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    dataset.set_format(type="pt")
    return dataset, {
        "kind": "explicit_proxy_jsonl",
        "file": os.path.abspath(resolved),
        "sha256": _sha256_file(resolved),
        "source_rows": int(original_rows),
        "sample_rows": int(len(raw)),
        "sample_size_requested": int(warmup_data_sample_size),
        "sample_seed": int(seed),
    }


def _iter_trainable_lora_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    params: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("lora_A" in name) or ("lora_B" in name):
            params.append((name, param))
    if params:
        return params
    # Fallback for custom PEFT naming: use all trainable params.
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _flatten_trainable_grads(
    named_params: List[Tuple[str, torch.nn.Parameter]],
) -> Optional[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    for _, param in named_params:
        if param.grad is None:
            continue
        chunks.append(param.grad.detach().float().view(-1).cpu())
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _split_batch_even(batch: Dict[str, torch.Tensor]) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    first_tensor = None
    for v in batch.values():
        if torch.is_tensor(v) and v.dim() > 0:
            first_tensor = v
            break
    if first_tensor is None:
        return None, None
    bs = int(first_tensor.size(0))
    if bs < 2:
        return None, None
    half = bs // 2
    if half < 1:
        return None, None
    end_b = half * 2
    batch_a: Dict[str, torch.Tensor] = {}
    batch_b: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() > 0 and int(v.size(0)) == bs:
            batch_a[k] = v[:half]
            batch_b[k] = v[half:end_b]
        else:
            batch_a[k] = v
            batch_b[k] = v
    return batch_a, batch_b


def _compute_intra_batch_grad_cosine(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    named_params: List[Tuple[str, torch.nn.Parameter]],
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler,
) -> Tuple[Optional[float], str]:
    batch_a, batch_b = _split_batch_even(batch)
    if batch_a is None or batch_b is None:
        return None, "batch_too_small"
    device_type = next((v.device.type for v in batch_a.values() if torch.is_tensor(v)), "cuda")

    model.zero_grad(set_to_none=True)
    if use_amp:
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            outputs_a = model(**batch_a)
            loss_a = outputs_a.loss if hasattr(outputs_a, "loss") else outputs_a[0]
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss_a).backward()
        else:
            loss_a.backward()
    else:
        outputs_a = model(**batch_a)
        loss_a = outputs_a.loss if hasattr(outputs_a, "loss") else outputs_a[0]
        loss_a.backward()
    grad_a = _flatten_trainable_grads(named_params)

    model.zero_grad(set_to_none=True)
    if use_amp:
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            outputs_b = model(**batch_b)
            loss_b = outputs_b.loss if hasattr(outputs_b, "loss") else outputs_b[0]
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss_b).backward()
        else:
            loss_b.backward()
    else:
        outputs_b = model(**batch_b)
        loss_b = outputs_b.loss if hasattr(outputs_b, "loss") else outputs_b[0]
        loss_b.backward()
    grad_b = _flatten_trainable_grads(named_params)
    model.zero_grad(set_to_none=True)

    if grad_a is None or grad_b is None:
        return None, "missing_grad"
    if grad_a.numel() == 0 or grad_b.numel() == 0:
        return None, "empty_grad"
    if grad_a.numel() != grad_b.numel():
        return None, f"shape_mismatch:{grad_a.numel()}!={grad_b.numel()}"

    cosine = F.cosine_similarity(grad_a, grad_b, dim=0, eps=1e-12).item()
    return float(cosine), "ok"


def _compute_rolling_k_grad_cosine(
    model: torch.nn.Module,
    first_batch: Dict[str, torch.Tensor],
    batch_iter,
    named_params: List[Tuple[str, torch.nn.Parameter]],
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler,
    k: int,
) -> Tuple[Optional[float], str]:
    if k <= 0:
        return None, "invalid_k"

    # Rolling mode: use current batch as first element, then consume next (2k-1) batches.
    def _move_batch_to_device(b: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, val in b.items():
            if torch.is_tensor(val) and key in ref and torch.is_tensor(ref[key]):
                out[key] = val.to(ref[key].device)
            else:
                out[key] = val
        return out

    group_a: List[Dict[str, torch.Tensor]] = [first_batch]
    group_b: List[Dict[str, torch.Tensor]] = []
    need = (2 * k) - 1
    fetched = 0
    while fetched < need:
        try:
            nxt = next(batch_iter)
        except StopIteration:
            return None, "insufficient_batches"
        nxt = _move_batch_to_device(nxt, first_batch)
        group_a.append(nxt) if len(group_a) < k else group_b.append(nxt)
        fetched += 1

    if len(group_a) != k or len(group_b) != k:
        return None, "insufficient_batches"

    device_type = next((v.device.type for v in first_batch.values() if torch.is_tensor(v)), "cuda")

    def _accum_group_grad(group: List[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        for b in group:
            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    outputs = model(**b)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(**b)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()
        return _flatten_trainable_grads(named_params)

    grad_a = _accum_group_grad(group_a)
    grad_b = _accum_group_grad(group_b)
    model.zero_grad(set_to_none=True)

    if grad_a is None or grad_b is None:
        return None, "missing_grad"
    if grad_a.numel() == 0 or grad_b.numel() == 0:
        return None, "empty_grad"
    if grad_a.numel() != grad_b.numel():
        return None, f"shape_mismatch:{grad_a.numel()}!={grad_b.numel()}"

    cosine = F.cosine_similarity(grad_a, grad_b, dim=0, eps=1e-12).item()
    return float(cosine), "ok"


def _extract_optimizer_state_with_names(optimizer: torch.optim.Optimizer,
                                        model: torch.nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build a parameter-name keyed optimizer state dict for robust ADAM loading later.
    """
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        st = optimizer.state.get(param, {})
        if not isinstance(st, dict):
            continue
        row: Dict[str, torch.Tensor] = {}
        if "exp_avg" in st and torch.is_tensor(st["exp_avg"]):
            row["exp_avg"] = torch.nan_to_num(
                st["exp_avg"].detach().to(torch.float32).cpu(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        if "exp_avg_sq" in st and torch.is_tensor(st["exp_avg_sq"]):
            row["exp_avg_sq"] = torch.clamp(
                torch.nan_to_num(
                    st["exp_avg_sq"].detach().to(torch.float32).cpu(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                min=0.0,
            )
        step = st.get("step")
        if step is not None:
            if torch.is_tensor(step):
                row["step"] = step.detach().cpu()
            else:
                row["step"] = torch.tensor(step)
        out[name] = row
    return out


def _validate_model_name_or_path(model_name_or_path: str) -> None:
    if model_name_or_path is None or str(model_name_or_path).strip() == "":
        raise ValueError(
            "model_name_or_path is empty. Provide --model_name_or_path or set BASE_MODEL."
        )
    candidate = os.path.expanduser(model_name_or_path)
    if os.path.isabs(candidate) or candidate.startswith("."):
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"model_name_or_path looks like a local path but was not found: {candidate}"
            )


def _resolve_attr_path(root: object, path: str) -> Optional[object]:
    cur = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _summarize_trainable_params(model: torch.nn.Module,
                                max_name_samples: int = 128) -> Dict[str, object]:
    total_params = 0
    trainable_params = 0
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        n = int(param.numel())
        total_params += n
        if param.requires_grad:
            trainable_params += n
            if len(trainable_names) < max_name_samples:
                trainable_names.append(name)
    trainable_ratio = (
        float(trainable_params) / float(total_params) if total_params > 0 else 0.0
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_ratio),
        "trainable_param_name_samples": trainable_names,
        "trainable_param_name_sample_count": int(len(trainable_names)),
    }


def _apply_freeze_to_final_n_layers(model: torch.nn.Module,
                                    final_n_layers: int,
                                    train_final_norm: bool,
                                    train_lm_head: bool) -> Dict[str, object]:
    if final_n_layers <= 0:
        return {"trainable_scope_mode": "default"}

    layer_patterns = [
        re.compile(r"(^|\.)model\.layers\.(\d+)\."),
        re.compile(r"(^|\.)transformer\.h\.(\d+)\."),
        re.compile(r"(^|\.)model\.decoder\.layers\.(\d+)\."),
        re.compile(r"(^|\.)gpt_neox\.layers\.(\d+)\."),
    ]
    named_params = list(model.named_parameters())
    best_pattern = None
    best_ids: List[int] = []
    for pat in layer_patterns:
        ids = []
        for name, _ in named_params:
            m = pat.search(name)
            if m is None:
                continue
            ids.append(int(m.group(2)))
        uniq = sorted(set(ids))
        if len(uniq) > len(best_ids):
            best_ids = uniq
            best_pattern = pat
    if best_pattern is None or len(best_ids) == 0:
        raise RuntimeError(
            "Could not locate transformer layer parameter names for final-layer freeze. "
            "Supported prefixes include model.layers.*, transformer.h.*, model.decoder.layers.*, gpt_neox.layers.*"
        )

    total_layers = int(max(best_ids)) + 1
    keep_n = min(int(final_n_layers), total_layers)
    start_idx = total_layers - keep_n

    # Freeze everything first.
    for _, param in named_params:
        param.requires_grad = False

    # Unfreeze final N transformer blocks.
    for name, param in named_params:
        m = best_pattern.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(2))
        if layer_idx >= start_idx:
            param.requires_grad = True

    # Optionally unfreeze final norm modules.
    if train_final_norm:
        norm_paths = [
            "model.norm",
            "transformer.ln_f",
            "model.final_layernorm",
            "gpt_neox.final_layer_norm",
            "base_model.model.model.norm",
            "base_model.model.transformer.ln_f",
        ]
        for path in norm_paths:
            mod = _resolve_attr_path(model, path)
            if isinstance(mod, torch.nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True

    # Optionally unfreeze lm_head (and tied output embeddings).
    if train_lm_head:
        for name, param in named_params:
            if re.search(r"(^|\.)lm_head\.", name):
                param.requires_grad = True
        if hasattr(model, "get_output_embeddings"):
            out_head = model.get_output_embeddings()
            if isinstance(out_head, torch.nn.Module):
                for p in out_head.parameters():
                    p.requires_grad = True

    summary = _summarize_trainable_params(model)
    summary.update(
        {
            "trainable_scope_mode": "freeze_to_final_n_layers",
            "final_n_layers_requested": int(final_n_layers),
            "final_n_layers_effective": int(keep_n),
            "total_transformer_layers": int(total_layers),
            "first_trainable_layer_idx": int(start_idx),
            "train_final_norm": bool(train_final_norm),
            "train_lm_head": bool(train_lm_head),
        }
    )
    return summary


def _load_warmup_meta(ckpt_dir: str) -> Dict:
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _check_lora_meta_compat(meta: Dict, args) -> None:
    if not meta:
        return
    if "lora" not in meta:
        return
    if meta.get("lora") != args.lora:
        raise ValueError(
            "LoRA mismatch between warmup checkpoint and scoring run: "
            f"warmup lora={meta.get('lora')} vs scoring lora={args.lora}. "
            "Please use the same LoRA setting as warmup."
        )
    if not args.lora:
        return
    mismatches = []
    for key in (
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_rank_pattern",
        "lora_alpha_pattern",
    ):
        if key in meta and meta.get(key) != getattr(args, key):
            mismatches.append(f"{key}: warmup={meta.get(key)} scoring={getattr(args, key)}")
    if mismatches:
        raise ValueError(
            "LoRA config mismatch between warmup checkpoint and scoring run:\n  "
            + "\n  ".join(mismatches)
            + "\nUse the same LoRA hyperparameters as warmup."
        )


def _apply_meta_to_args(meta: Dict, args) -> None:
    if not meta:
        return
    # Fields that must match model architecture / tokenization.
    field_map = {
        "model_name_or_path": "model_name_or_path",
        "lora": "lora",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "lora_target_modules",
        "lora_rank_pattern": "lora_rank_pattern",
        "lora_alpha_pattern": "lora_alpha_pattern",
        "freeze_to_final_n_layers": "freeze_to_final_n_layers",
        "train_final_norm": "train_final_norm",
        "train_lm_head": "train_lm_head",
        "max_seq_length": "max_seq_length",
        "chat_format": "chat_format",
        "use_chat_format": "use_chat_format",
        "torch_dtype": "torch_dtype",
        "bf16": "bf16",
        "fp16": "fp16",
    }
    for meta_key, arg_key in field_map.items():
        if meta_key not in meta:
            continue
        meta_val = meta.get(meta_key)
        if meta_val is None:
            continue
        if arg_key == "use_chat_format":
            # args stores inverse flag
            desired_no_chat = not bool(meta_val)
            if not args.allow_meta_override:
                if args.no_chat_format != desired_no_chat:
                    print(
                        f"[loss_gap] overriding no_chat_format={args.no_chat_format} "
                        f"with warmup meta use_chat_format={meta_val}",
                        flush=True,
                    )
                args.no_chat_format = desired_no_chat
            continue
        current = getattr(args, arg_key)
        if current != meta_val:
            if args.allow_meta_override:
                continue
            print(
                f"[loss_gap] overriding {arg_key}={current} with warmup meta {meta_val}",
                flush=True,
            )
            setattr(args, arg_key, meta_val)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Validation-warmup loss-gap data selection.")
    ap.add_argument("--model_name_or_path", required=False,
                    help="Base model name or checkpoint path. Required for warmup; optional for score when using warmup meta.")
    ap.add_argument("--data_dir", required=True,
                    help="Root data dir containing eval/ for target task.")
    ap.add_argument("--target_task_names", nargs="+", required=True,
                    help="Target tasks to warm up on (e.g., tydiqa mmlu bbh).")
    ap.add_argument("--train_files", nargs="+", required=True,
                    help="Candidate training JSONL files to score.")
    ap.add_argument("--train_file_names", nargs="+", required=True,
                    help="Names for score files (same length as train_files).")
    ap.add_argument("--output_path", required=True,
                    help="Output root for score files; scores saved under {output_path}/{task}/.")
    ap.add_argument("--mode", type=str, default="warmup",
                    choices=["warmup", "score", "warmup_and_score"],
                    help="warmup: train on val only. score: score using saved ckpts. "
                         "warmup_and_score: train+score each epoch on the fly (no ckpt saving).")
    ap.add_argument("--ckpt_dir", type=str, default=None,
                    help="Checkpoint directory; defaults to {output_path}/{task}/warmup_ckpts.")
    ap.add_argument(
        "--warmup_data_file",
        type=str,
        default=None,
        help=(
            "Optional target-like prompt/completion or messages JSONL used only for warmup. "
            "May contain {task}. Canonical task validation data is still used by score-mode probes."
        ),
    )
    ap.add_argument(
        "--warmup_data_sample_size",
        type=int,
        default=0,
        help="Deterministically sample this many rows from --warmup_data_file; 0 uses all rows.",
    )
    ap.add_argument(
        "--warmup_group_file",
        type=str,
        default=None,
        help=(
            "Optional JSON validation-group manifest. In warmup mode, select only "
            "the rows assigned to --warmup_group_label. Use {task} in the path "
            "for task-specific manifests. The manifest must match the canonical "
            "validation ordering exactly."
        ),
    )
    ap.add_argument(
        "--warmup_group_label",
        type=str,
        default=None,
        help="Group label to retain from --warmup_group_file for this trajectory.",
    )
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--warmup_steps", type=int, default=5,
                    help="Deprecated: used as warmup_epochs when warmup_epochs is unset.")
    ap.add_argument("--warmup_epochs", type=int, default=None,
                    help="Number of warmup checkpoints to save.")
    ap.add_argument("--warmup_epochs_per_ckpt", type=int, default=5,
                    help="Number of full validation epochs per checkpoint (set 1 for old behavior).")
    ap.add_argument("--warmup_checkpoint_format", type=str, default="trainable_state_dict",
                    choices=["state_dict", "hf_checkpoint", "trainable_state_dict"],
                    help="Checkpoint format for warmup steps. "
                         "state_dict writes step_<n>.pt (FULL model – use only if needed); "
                         "hf_checkpoint writes checkpoint-<n>/ directories (recommended for large LoRA runs); "
                         "trainable_state_dict writes only trainable parameters to step_<n>.pt (default, ~1000x smaller for LoRA).")
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--lr_scale", type=float, default=1.0,
                    help="Scale factor applied to learning_rate for warmup.")
    ap.add_argument("--warmup_lr", type=float, default=None,
                    help="Override warmup LR directly (takes precedence over learning_rate*lr_scale).")
    ap.add_argument("--warmup_batch_size", type=int, default=1)
    ap.add_argument(
        "--warmup_gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Number of warmup microbatches averaged into one optimizer update. "
            "Values above 1 currently require mode=warmup together with a positive "
            "per-checkpoint update cap and --warmup_continuous_batches."
        ),
    )
    ap.add_argument(
        "--warmup_max_batches_per_checkpoint",
        type=int,
        default=0,
        help=(
            "If >0, cap optimizer updates between consecutive warmup checkpoints. "
            "This is intended for fixed-update validation-size controls; 0 uses "
            "every validation batch in warmup_epochs_per_ckpt epochs."
        ),
    )
    ap.add_argument(
        "--warmup_continuous_batches",
        action="store_true",
        help=(
            "Keep one shuffled data-loader stream across checkpoint boundaries. "
            "With a per-checkpoint batch cap, this exposes each unique proxy row once "
            "before reshuffling instead of repeatedly taking prefixes of new epochs."
        ),
    )
    ap.add_argument("--warmup_optim", type=str, default="adamw",
                    choices=["sgd", "adamw"])
    ap.add_argument("--loss_batch_size", type=int, default=4)
    ap.add_argument(
        "--length_bucket_candidates",
        action="store_true",
        help=(
            "Batch candidate examples by tokenized length during scoring and "
            "restore the original source-row order before writing scores. "
            "This disables resumable loss chunks for the bucketed loader."
        ),
    )
    ap.add_argument("--loss_save_interval", type=int, default=160,
                    help="Save loss chunks every N batches during scoring.")
    ap.add_argument("--score_ckpt_ids", nargs="+", type=int, default=None,
                    help="Optional checkpoint step ids to score (e.g., 1 2 3 4).")
    ap.add_argument("--skip_score_write", action="store_true",
                    help="Skip writing final influence scores (useful for partial scoring).")
    ap.add_argument("--save_endpoint_losses", action="store_true",
                    help="Write each scored checkpoint's raw candidate-loss vector in source-row order. "
                         "Useful for deriving several endpoint-depth scores from one scoring pass.")
    ap.add_argument("--score_shard", type=str, default=None,
                    help="k/K: score only shard k (0-based) of K contiguous batch ranges on --loss_save_interval "
                         "boundaries (same batches and loss-chunk files as an unsharded pass). Needs "
                         "--skip_score_write and an unbucketed candidate loader.")
    ap.add_argument("--score_metric", choices=["loss_gap", "cov_with_val", "var", "dense_embed_sim", "final_drop", "min_loss", "avg_loss", "max_drop"], default="loss_gap",
                    help="Scoring metric. loss_gap uses sum(l(z,t)-l(z,last)); "
                         "cov_with_val uses covariance between l(z,t) and val loss over t; "
                         "var uses sample variance of l(z,t) across checkpoints; "
                         "dense_embed_sim uses cosine similarity between dense example embeddings and a pooled target-task dev embedding.")
    ap.add_argument("--score_normalize_by_first", action="store_true",
                    help="Normalize each loss by l(z, step1) before computing loss-gap scores.")
    ap.add_argument("--score_normalize_eps", type=float, default=1e-8,
                    help="Epsilon added to l(z, step1) to avoid division by zero when normalizing.")
    ap.add_argument("--probe_mmlu_test_n", type=int, default=0,
                    help="If >0, sample N examples from MMLU test and compute loss-gap scores.")
    ap.add_argument("--probe_mmlu_valid_n", type=int, default=0,
                    help="If >0, sample N examples from MMLU dev/valid and compute loss-gap scores.")
    ap.add_argument("--mmlu_n_shot", type=int, default=DEFAULT_MMLU_N_SHOT,
                    help="Number of MMLU dev examples per subject used for validation warmup/prompts.")
    ap.add_argument("--probe_target_valid_n", type=int, default=0,
                    help="If >0, sample N examples from the target task validation set and compute loss-gap scores.")
    ap.add_argument(
        "--save_probe_trajectory",
        action="store_true",
        help=(
            "In score mode, persist the per-checkpoint, per-example probe loss matrices "
            "to probe_trajectories.pt. This is an opt-in diagnostic for function-space "
            "rank analysis; it does not change influence scores."
        ),
    )
    ap.add_argument("--target_valid_split", action="store_true",
                    help="Use a disjoint split of target validation data: warmup uses one subset, probe uses the holdout subset.")
    ap.add_argument("--target_valid_warmup_ratio", type=float, default=0.5,
                    help="Warmup subset ratio for target validation split. Must be in (0,1).")
    ap.add_argument("--target_valid_split_seed", type=int, default=None,
                    help="Seed for target validation split. Defaults to --seed.")
    ap.add_argument("--target_valid_split_strategy", choices=["random", "task_group"], default="task_group",
                    help="Split strategy for target validation split. task_group uses task-aware groups when available.")
    ap.add_argument("--target_valid_split_file", type=str, default=None,
                    help="Optional split JSON path. If relative, resolved under ckpt_dir. Supports '{task}' placeholder.")
    ap.add_argument("--probe_cot_n", type=int, default=0,
                    help="If >0, sample N examples from COT train file and compute loss-gap scores.")
    ap.add_argument("--probe_cot_file", type=str, default=None,
                    help="Optional path to COT train jsonl to sample from (defaults to train_files entry named 'cot').")
    ap.add_argument("--probe_seed", type=int, default=None,
                    help="Random seed for probe sampling (defaults to --seed).")
    ap.add_argument("--probe_only", action="store_true",
                    help="Only compute probe scores (skip candidate scoring/write).")
    ap.add_argument("--use_cache_only", action="store_true",
                    help="Only load cached score artifacts when available (no model forward for candidate scoring).")
    ap.add_argument("--score_cache_root", type=str, default=None,
                    help="Optional cache root override for --use_cache_only score mode. "
                         "Expected layout: <root>/<train_file_name>/step_XXXX/. Supports '{task}' placeholder.")
    ap.add_argument("--candidate_percentage", type=float, default=1.0,
                    help="Optional sampling percentage for candidate data.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--torch_dtype", type=str, default=None,
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--no_chat_format", action="store_true",
                    help="Disable chat-format prompts for validation datasets.")
    ap.add_argument("--chat_format", type=str, default="tulu")
    meta_group = ap.add_mutually_exclusive_group()
    meta_group.add_argument("--use_warmup_meta", dest="use_warmup_meta", action="store_true",
                            help="(score) Use warmup meta.json to configure model/LoRA/tokenization.")
    meta_group.add_argument("--no_use_warmup_meta", dest="use_warmup_meta", action="store_false",
                            help="(score) Do not use warmup meta.json.")
    ap.set_defaults(use_warmup_meta=True)
    ap.add_argument("--allow_meta_override", action="store_true",
                    help="(score) Allow CLI args to override warmup meta values.")

    # Optional LoRA (default on to match LESS warmup)
    lora_group = ap.add_mutually_exclusive_group()
    lora_group.add_argument("--lora", dest="lora", action="store_true",
                            help="Enable LoRA adapters (default).")
    lora_group.add_argument("--no_lora", dest="lora", action="store_false",
                            help="Disable LoRA adapters.")
    ap.set_defaults(lora=True)
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=512)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--lora_target_modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument(
        "--lora_rank_pattern",
        nargs="*",
        default=[],
        metavar="MODULE=RANK",
        help="Optional PEFT per-module rank overrides, e.g. o_proj=2.",
    )
    ap.add_argument(
        "--lora_alpha_pattern",
        nargs="*",
        default=[],
        metavar="MODULE=ALPHA",
        help="Optional PEFT per-module alpha overrides, e.g. o_proj=8.",
    )
    ap.add_argument("--freeze_to_final_n_layers", type=int, default=0,
                    help="(warmup) If >0 and --no_lora is used, freeze all parameters except "
                         "the final N transformer blocks.")
    final_norm_group = ap.add_mutually_exclusive_group()
    final_norm_group.add_argument("--train_final_norm", dest="train_final_norm", action="store_true",
                                  help="(warmup) When freezing to final layers, keep final norm trainable (default).")
    final_norm_group.add_argument("--no_train_final_norm", dest="train_final_norm", action="store_false",
                                  help="(warmup) When freezing to final layers, keep final norm frozen.")
    ap.set_defaults(train_final_norm=True)
    lm_head_group = ap.add_mutually_exclusive_group()
    lm_head_group.add_argument("--train_lm_head", dest="train_lm_head", action="store_true",
                               help="(warmup) When freezing to final layers, keep lm_head trainable (default).")
    lm_head_group.add_argument("--no_train_lm_head", dest="train_lm_head", action="store_false",
                               help="(warmup) When freezing to final layers, keep lm_head frozen.")
    ap.set_defaults(train_lm_head=True)
    ap.add_argument("--warmup_init_model_path", type=str, default=None,
                    help="(warmup) Optional model/checkpoint path used as initialization before val warmup.")
    ap.add_argument("--warmup_init_merge_adapter_then_reinit_lora", action="store_true",
                    help="(warmup) If warmup_init_model_path is a PEFT adapter checkpoint, merge it into base model "
                         "then apply LoRA from current CLI args. This enables starting from prior training while "
                         "still using golden LoRA hyperparameters.")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="Weight decay for warmup optimizer.")
    ap.add_argument("--save_warmup_optimizer_state", action="store_true",
                    help="(warmup) Save name-keyed optimizer state for each step to warmup_ckpts.")
    ap.add_argument(
        "--save_warmup_initial_state",
        action="store_true",
        help="(warmup) Save the initial trainable-parameter state as step_0.pt. "
             "This is required for exact trajectory-increment alignment calibration.",
    )
    ap.add_argument("--benchmark_eval_dir", type=str, default=None,
                    help="(warmup) If set, compute TyDiQA test-set loss at each checkpoint "
                         "(no-grad forward pass on 9 one-shot examples). "
                         "Should be the data root containing eval/tydiqa/tydiqa-one-shot.json.")
    ap.add_argument("--track_intra_batch_grad_cosine", action="store_true",
                    help="(warmup) Track intra-batch gradient cosine similarity using two disjoint half-batches.")
    ap.add_argument("--grad_cosine_interval", type=int, default=0,
                    help="(warmup) Track every N training batches; <=0 disables.")
    ap.add_argument("--track_rolling_grad_cosine", action="store_true",
                    help="(warmup) Track rolling virtual-batch gradient cosine without changing training updates.")
    ap.add_argument("--grad_cosine_virtual_k", type=int, default=16,
                    help="(warmup) Rolling mode: accumulate K micro-batch gradients per side (A vs B).")
    ap.add_argument("--grad_cosine_log_jsonl", type=str, default=None,
                    help="(warmup) Optional JSONL path for gradient-cosine events. "
                         "Defaults to <ckpt_dir>/intra_batch_grad_cosine.jsonl.")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Allow loading models/tokenizers with custom code from the Hub.")

    args = ap.parse_args()
    try:
        args.lora_rank_pattern = _parse_module_int_pattern(
            args.lora_rank_pattern,
            name="--lora_rank_pattern",
            minimum=1,
        )
        args.lora_alpha_pattern = _parse_module_int_pattern(
            args.lora_alpha_pattern,
            name="--lora_alpha_pattern",
            minimum=1,
        )
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    unknown_rank = sorted(set(args.lora_rank_pattern) - set(args.lora_target_modules))
    unknown_alpha = sorted(set(args.lora_alpha_pattern) - set(args.lora_target_modules))
    if unknown_rank or unknown_alpha:
        ap.error(
            "LoRA rank/alpha pattern keys must also be present in "
            f"--lora_target_modules; rank_only={unknown_rank}, alpha_only={unknown_alpha}"
        )
    return args


def _prepare_candidate_dataset(path: str,
                               tokenizer,
                               max_seq_length: int,
                               percentage: float,
                               seed: int,
                               chat_format: str = "tulu"):
    raw = load_dataset("json", data_files=[path])["train"]
    if percentage < 1.0:
        sample_size = int(len(raw) * percentage)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(raw), generator=g)[:sample_size].tolist()
        raw = raw.select(idx)
    encoded = encode_data(raw, tokenizer, max_seq_length, chat_format=chat_format)
    # Joint-pool diagnostic panels carry source/index provenance, and selected
    # subsets carry influence metadata. DataCollatorForSeq2Seq cannot collate
    # those strings/scalars; retain only model inputs rather than maintaining a
    # brittle deny-list of known metadata fields.
    model_columns = {"input_ids", "labels", "attention_mask"}
    drop_cols = [c for c in encoded.column_names if c not in model_columns]
    if drop_cols:
        encoded = encoded.remove_columns(drop_cols)
    encoded.set_format(type="pt")
    return encoded


def _prepare_candidate_dataset_count(path: str,
                                     tokenizer,
                                     max_seq_length: int,
                                     num_samples: int,
                                     seed: int,
                                     chat_format: str = "tulu"):
    raw = load_dataset("json", data_files=[path])["train"]
    if num_samples is not None and num_samples > 0 and num_samples < len(raw):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(raw), generator=g)[:num_samples].tolist()
        raw = raw.select(idx)
    encoded = encode_data(raw, tokenizer, max_seq_length, chat_format=chat_format)
    model_columns = {"input_ids", "labels", "attention_mask"}
    drop_cols = [c for c in encoded.column_names if c not in model_columns]
    if drop_cols:
        encoded = encoded.remove_columns(drop_cols)
    encoded.set_format(type="pt")
    return encoded


class _LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Batch candidate rows by length while retaining the original row order.

    Candidate loss scoring is order-sensitive at the artifact boundary: the
    score at row ``i`` must still refer to row ``i`` in the source JSONL.  The
    model, however, only needs examples in the same batch to be padded to a
    common length.  Sorting the batches by sequence length reduces padding
    without changing any per-example loss.  ``_less_restore_order`` is exposed
    on the resulting DataLoader so the scorer can scatter losses back before
    writing the score vector.

    This sampler is intentionally deterministic.  Equal-length rows retain
    their source order, and batches are formed only after the global stable
    length sort.
    """

    def __init__(self, dataset, batch_size: int):
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        lengths = []
        for index in range(len(dataset)):
            row = dataset[index]
            input_ids = row.get("input_ids") if hasattr(row, "get") else row["input_ids"]
            lengths.append(len(input_ids))
        self.order = sorted(range(len(dataset)), key=lambda index: (lengths[index], index))
        self.batches = [
            self.order[start : start + self.batch_size]
            for start in range(0, len(self.order), self.batch_size)
        ]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def _make_candidate_loader(dataset,
                           batch_size: int,
                           collate_fn,
                           length_bucketed: bool = False):
    """Build a candidate loader, optionally using deterministic length buckets."""
    if not length_bucketed:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    sampler = _LengthBucketBatchSampler(dataset, batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
    )
    # DataLoader permits application attributes; this is consumed only by the
    # scorer and avoids putting provenance fields into the model batch.
    loader._less_restore_order = sampler.batches
    print(
        f"[loss_gap] length-bucketed candidate loader: rows={len(dataset)} "
        f"batches={len(sampler)} batch_size={batch_size}",
        flush=True,
    )
    return loader


def _sample_hf_dataset(ds: Dataset, num_samples: int, seed: int) -> Dataset:
    if num_samples is None or num_samples <= 0 or num_samples >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:num_samples].tolist()
    return ds.select(idx)


def _resolve_tydiqa_one_shot_file(data_dir: str, zh: bool = False) -> Optional[str]:
    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    candidate_dirs = [
        os.path.join(data_dir, "eval", "tydiqa"),
        os.path.join(data_dir, "eval", "tydiqa", "dev"),
        os.path.join(data_dir, "eval", "tydiqa", "test"),
    ]
    candidate_files = [os.path.join(d, file_name) for d in candidate_dirs]
    for p in candidate_files:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    try:
        pattern = os.path.join(data_dir, "eval", "tydiqa", "**", "*one-shot*.json*")
        matches = glob.glob(pattern, recursive=True)
        matches = [m for m in matches if "one-shot" in os.path.basename(m)]
        if matches:
            return matches[0]
    except Exception:
        return None
    return None


def _build_task_group_ids(
    target_task: str,
    data_dir: str,
    dataset_len: int,
    split_strategy: str,
    mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT,
) -> Optional[List[str]]:
    if split_strategy != "task_group":
        return None
    if dataset_len <= 0:
        return None

    if target_task == "tydiqa":
        path = _resolve_tydiqa_one_shot_file(data_dir=data_dir, zh=False)
        if path is not None:
            try:
                examples = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(examples, dict):
                    langs = list(examples.keys())
                    if len(langs) == dataset_len:
                        return [f"tydiqa:{lang}" for lang in langs]
            except Exception:
                pass
        # Fallback: each example treated as one group when language metadata is unavailable.
        return [f"tydiqa_idx_{i}" for i in range(dataset_len)]

    if target_task == "mmlu":
        n_shot = resolve_mmlu_n_shot(mmlu_n_shot)
        return [f"mmlu_subject_{i // n_shot}" for i in range(dataset_len)]

    if target_task == "bbh":
        # Current get_bbh_dataset emits 3 examples per BBH subtask in task-major order.
        return [f"bbh_task_{i // 3}" for i in range(dataset_len)]

    return None


def _validate_disjoint_split(total_n: int, warmup_idx: List[int], probe_idx: List[int]) -> None:
    if total_n < 2:
        raise ValueError(f"target validation split requires at least 2 examples, got {total_n}")
    w = [int(i) for i in warmup_idx]
    p = [int(i) for i in probe_idx]
    ws = set(w)
    ps = set(p)
    if not ws:
        raise ValueError("warmup split is empty")
    if not ps:
        raise ValueError("probe split is empty")
    if ws & ps:
        raise ValueError("warmup/probe splits overlap")
    all_idx = set(range(total_n))
    if (ws | ps) != all_idx:
        raise ValueError("warmup/probe splits do not cover full dataset")


def _build_disjoint_split_indices(
    total_n: int,
    warmup_ratio: float,
    seed: int,
    group_ids: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    if total_n < 2:
        raise ValueError(f"target validation split requires at least 2 examples, got {total_n}")
    if warmup_ratio <= 0.0 or warmup_ratio >= 1.0:
        raise ValueError(f"target_valid_warmup_ratio must be in (0,1), got {warmup_ratio}")

    warmup_target = int(round(total_n * warmup_ratio))
    warmup_target = max(1, min(total_n - 1, warmup_target))

    rng = random.Random(int(seed))
    all_indices = list(range(total_n))

    if group_ids is not None and len(group_ids) == total_n:
        group_to_indices: Dict[str, List[int]] = {}
        for idx, gid in enumerate(group_ids):
            key = str(gid)
            group_to_indices.setdefault(key, []).append(idx)
        groups = list(group_to_indices.keys())
        rng.shuffle(groups)

        selected_groups: List[str] = []
        selected_count = 0
        for g in groups:
            g_count = len(group_to_indices[g])
            if selected_count >= warmup_target:
                break
            if selected_count == 0:
                selected_groups.append(g)
                selected_count += g_count
                continue
            # Greedy close-to-target selection.
            overshoot = abs((selected_count + g_count) - warmup_target)
            keep = abs(selected_count - warmup_target)
            if overshoot <= keep:
                selected_groups.append(g)
                selected_count += g_count

        if not selected_groups:
            selected_groups = [groups[0]]

        warmup_set = set()
        for g in selected_groups:
            warmup_set.update(group_to_indices[g])
        probe_set = set(all_indices) - warmup_set

        if not probe_set:
            # Move one group out to ensure non-empty probe split.
            move_group = selected_groups[-1]
            for idx in group_to_indices[move_group]:
                warmup_set.discard(idx)
                probe_set.add(idx)
            if not warmup_set:
                idx = min(probe_set)
                probe_set.discard(idx)
                warmup_set.add(idx)

        warmup_idx = sorted(warmup_set)
        probe_idx = sorted(probe_set)
    else:
        perm = list(all_indices)
        rng.shuffle(perm)
        warmup_idx = sorted(perm[:warmup_target])
        probe_idx = sorted(perm[warmup_target:])

    _validate_disjoint_split(total_n=total_n, warmup_idx=warmup_idx, probe_idx=probe_idx)
    return warmup_idx, probe_idx


def _resolve_target_valid_split_file(
    ckpt_dir: str,
    target_task: str,
    split_file_arg: Optional[str],
) -> str:
    if split_file_arg is None or str(split_file_arg).strip() == "":
        return os.path.join(ckpt_dir, "target_valid_split.json")
    p = str(split_file_arg).format(task=target_task)
    if os.path.isabs(p):
        return p
    return os.path.join(ckpt_dir, p)


def _load_target_valid_split(split_path: str) -> Dict:
    with open(split_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid split JSON: {split_path}")
    return obj


def _prepare_target_valid_split(
    target_task: str,
    data_dir: str,
    dataset_len: int,
    ckpt_dir: str,
    mode: str,
    split_strategy: str,
    warmup_ratio: float,
    split_seed: int,
    split_file_arg: Optional[str],
    mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT,
) -> Dict:
    split_path = _resolve_target_valid_split_file(
        ckpt_dir=ckpt_dir, target_task=target_task, split_file_arg=split_file_arg
    )
    explicit_split_file = bool(split_file_arg is not None and str(split_file_arg).strip())
    current_mmlu_subjects = None
    if target_task == "mmlu":
        subject_env = os.environ.get("TACS_MMLU_SUBJECTS", "").strip()
        current_mmlu_subjects = subject_env.split() if subject_env else None

    split_obj: Optional[Dict] = None
    if os.path.exists(split_path):
        split_obj = _load_target_valid_split(split_path)
        old_n = int(split_obj.get("total_n", -1))
        old_mmlu_n_shot = split_obj.get("mmlu_n_shot")
        old_mmlu_subjects = split_obj.get("mmlu_subjects")
        current_mmlu_n_shot = int(resolve_mmlu_n_shot(mmlu_n_shot)) if target_task == "mmlu" else None
        mismatch = None
        if old_n != dataset_len:
            mismatch = f"saved total_n={old_n}, current={dataset_len}"
        elif (
            target_task == "mmlu"
            and old_mmlu_n_shot is not None
            and int(old_mmlu_n_shot) != current_mmlu_n_shot
        ):
            mismatch = (
                f"saved mmlu_n_shot={old_mmlu_n_shot}, "
                f"current={current_mmlu_n_shot}"
            )
        elif target_task == "mmlu" and current_mmlu_subjects:
            if old_mmlu_subjects is None:
                mismatch = "filtered MMLU run requires split provenance field mmlu_subjects"
            elif list(old_mmlu_subjects) != current_mmlu_subjects:
                mismatch = (
                    f"saved mmlu_subjects={old_mmlu_subjects}, "
                    f"current={current_mmlu_subjects}"
                )
        if mismatch is not None:
            if mode == "warmup" and not explicit_split_file:
                print(
                    f"[loss_gap] target_valid split mismatch in {split_path}: "
                    f"{mismatch}; regenerating.",
                    flush=True,
                )
                split_obj = None
            else:
                raise ValueError(
                    f"target_valid split mismatch in {split_path}: {mismatch}. "
                    "Explicit split files fail closed instead of regenerating."
                )

    if split_obj is None:
        if mode != "warmup":
            raise FileNotFoundError(
                f"target_valid split file not found for score mode: {split_path}. "
                "Run warmup with --target_valid_split first."
            )
        group_ids = _build_task_group_ids(
            target_task=target_task,
            data_dir=data_dir,
            dataset_len=dataset_len,
            split_strategy=split_strategy,
            mmlu_n_shot=mmlu_n_shot,
        )
        warmup_idx, probe_idx = _build_disjoint_split_indices(
            total_n=dataset_len,
            warmup_ratio=warmup_ratio,
            seed=split_seed,
            group_ids=group_ids,
        )
        split_obj = {
            "version": 1,
            "target_task": target_task,
            "split_strategy": split_strategy,
            "warmup_ratio": float(warmup_ratio),
            "split_seed": int(split_seed),
            "mmlu_n_shot": int(resolve_mmlu_n_shot(mmlu_n_shot)) if target_task == "mmlu" else None,
            "mmlu_subjects": current_mmlu_subjects,
            "total_n": int(dataset_len),
            "warmup_indices": [int(x) for x in warmup_idx],
            "probe_indices": [int(x) for x in probe_idx],
            "warmup_n": int(len(warmup_idx)),
            "probe_n": int(len(probe_idx)),
            "grouped": bool(group_ids is not None),
        }
        if group_ids is not None:
            split_obj["warmup_groups"] = sorted({str(group_ids[i]) for i in warmup_idx})
            split_obj["probe_groups"] = sorted({str(group_ids[i]) for i in probe_idx})
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_obj, f, indent=2)
        print(
            f"[loss_gap] wrote target_valid split to {split_path}: "
            f"warmup_n={len(warmup_idx)} probe_n={len(probe_idx)} "
            f"strategy={split_strategy} seed={split_seed}",
            flush=True,
        )

    warmup_idx = [int(x) for x in split_obj.get("warmup_indices", [])]
    probe_idx = [int(x) for x in split_obj.get("probe_indices", [])]
    _validate_disjoint_split(total_n=dataset_len, warmup_idx=warmup_idx, probe_idx=probe_idx)
    split_obj["split_path"] = split_path
    split_obj["warmup_indices"] = warmup_idx
    split_obj["probe_indices"] = probe_idx
    split_obj["warmup_n"] = len(warmup_idx)
    split_obj["probe_n"] = len(probe_idx)
    return split_obj


def _prepare_mmlu_split_dataset(data_dir: str,
                                tokenizer,
                                max_length: int,
                                num_samples: int,
                                seed: int,
                                split: str = "test",
                                use_chat_format: bool = True,
                                chat_format: str = "tulu",
                                mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT):
    import pandas as pd
    from tacs.data_selection.get_validation_dataset import tokenize

    mmlu_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_dir, "test"))
            if "_test.csv" in f
        ]
    )
    if not subjects:
        raise FileNotFoundError(f"No MMLU test files found under {mmlu_dir}/test")

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(train_df, subject, k):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        for i in range(k):
            prompt += format_example(train_df, i, include_answer=True)
        return prompt

    rng = random.Random(seed)
    n_shot = resolve_mmlu_n_shot(mmlu_n_shot)
    # Build index list of (subject, row_index)
    all_pairs = []
    test_cache = {}
    dev_cache = {}
    for subject in subjects:
        test_df = pd.read_csv(os.path.join(mmlu_dir, "test", subject + "_test.csv"), header=None)
        dev_df = pd.read_csv(os.path.join(mmlu_dir, "dev", subject + "_dev.csv"), header=None)
        if len(dev_df) < n_shot:
            raise ValueError(
                f"MMLU subject {subject} only has {len(dev_df)} dev examples, "
                f"cannot build {n_shot}-shot prompts."
            )
        test_cache[subject] = test_df
        dev_cache[subject] = dev_df
        target_df = dev_df if split == "dev" else test_df
        for i in range(len(target_df)):
            all_pairs.append((subject, i))
    if num_samples <= 0:
        num_samples = min(100, len(all_pairs))
    if num_samples < len(all_pairs):
        all_pairs = rng.sample(all_pairs, num_samples)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject, i in all_pairs:
        test_df = test_cache[subject]
        dev_df = dev_cache[subject]
        target_df = dev_df if split == "dev" else test_df
        prompt_end = format_example(target_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df.iloc[:n_shot], subject, n_shot)
        prompt = train_prompt + prompt_end
        answer = " " + target_df.iloc[i, target_df.shape[1] - 1]

        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                prompt = apply_chat_template_prompt(tokenizer, prompt + "\nThe answer is:", add_generation_prompt=True)
            elif fmt == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                prompt = f"<s> [INST] {prompt} [/INST] The answer is:"

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def _per_sample_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Standard causal LM shift.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab), shift_labels.view(-1), reduction="none"
    )
    loss_per_token = loss_flat.view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    denom = mask.sum(dim=1).clamp(min=1)
    loss_per_sample = (loss_per_token * mask).sum(dim=1) / denom
    return loss_per_sample


def _get_max_saved_index(output_dir: str, prefix: str = "losses") -> int:
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix + "-") and f.endswith(".pt")]
    indices = []
    for fname in files:
        try:
            idx = int(fname.split(".")[0].split("-")[1])
        except Exception:
            continue
        indices.append(idx)
    return max(indices) if indices else -1


def _load_cached_losses(cache_dir: str, expected_len: int = None) -> torch.Tensor:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Missing cache directory: {cache_dir}")
    loss_files = [f for f in os.listdir(cache_dir) if f.startswith("losses-") and f.endswith(".pt")]
    if not loss_files:
        raise FileNotFoundError(f"No cached loss files in {cache_dir}")
    loss_files.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    chunks = []
    for fname in loss_files:
        path = os.path.join(cache_dir, fname)
        chunk = torch.load(path, map_location="cpu")
        chunks.append(chunk)
    loss_vec = torch.cat(chunks, dim=0)
    if expected_len is not None and len(loss_vec) != expected_len:
        raise RuntimeError(
            f"Cached loss length {len(loss_vec)} != dataset length {expected_len} in {cache_dir}"
        )
    return loss_vec


def _parse_score_shard(args, total_batches: int):
    m = re.fullmatch(r"(\d+)/(\d+)", str(args.score_shard).strip())
    if not m:
        raise ValueError(f"--score_shard must be k/K, got {args.score_shard!r}")
    if not args.skip_score_write:
        raise ValueError("--score_shard scores part of the pool; use it with --skip_score_write")
    if args.length_bucket_candidates:
        raise ValueError("--score_shard needs the unbucketed candidate loader")
    rng = shard_batch_range(total_batches, int(m.group(1)), int(m.group(2)), args.loss_save_interval)
    if rng is None:
        raise ValueError(f"score shard {args.score_shard} is empty for {total_batches} batches")
    print(f"[loss_gap] score shard {args.score_shard}: batches ({rng[0]}, {rng[1]}] of {total_batches}", flush=True)
    return rng


def _compute_losses_for_dataset(model,
                                dataloader: DataLoader,
                                device: torch.device,
                                use_amp: bool,
                                amp_dtype: torch.dtype,
                                cache_dir: str = None,
                                step_id: int = None,
                                meta: dict = None,
                                save_interval: int = 160,
                                batch_range=None) -> torch.Tensor:
    # batch_range=(start, end]: score only 1-based batches start < i <= end (one score shard).
    model.eval()
    losses = []
    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None
    restore_order = getattr(dataloader, "_less_restore_order", None)
    if batch_range is not None and restore_order is not None:
        raise ValueError("score shards need the unbucketed candidate loader")
    if restore_order is not None:
        # The legacy cache format stores only contiguous loss chunks and has
        # no row-index sidecar.  Do not write an order-ambiguous cache for a
        # length-bucketed loader; the final score tensor is still written
        # atomically by the caller.
        if cache_dir is not None:
            print(
                "[loss_gap] length-bucketed loader: disabling resumable loss "
                f"chunks for {cache_dir}",
                flush=True,
            )
        cache_dir = None
        step_id = None
    use_cache = cache_dir is not None and step_id is not None
    save_interval = max(1, int(save_interval))
    max_index = -1
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, "meta.json")
        if meta is None:
            meta = {}
        meta = dict(meta)
        meta["loss_save_interval"] = save_interval
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for key, value in meta.items():
                    if key in existing and existing[key] != value:
                        raise RuntimeError(
                            f"Score cache metadata mismatch for {cache_dir}: {key}={existing[key]} != {value}"
                        )
            except Exception as exc:
                raise RuntimeError(f"Failed to validate cache meta at {meta_path}: {exc}") from exc
        else:
            _atomic_json_dump(meta, meta_path, indent=2)
        saved_files = [f for f in os.listdir(cache_dir) if f.startswith("losses-") and f.endswith(".pt")]
        saved_files = sorted(
            saved_files,
            key=lambda x: int(x.split(".")[0].split("-")[1])
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
                # gap detected; ignore later files to avoid skipping data
                break
            path = os.path.join(cache_dir, fname)
            try:
                chunk = torch.load(path, map_location="cpu")
            except Exception:
                # Corrupt/partial chunk; remove and stop to recompute.
                try:
                    os.remove(path)
                except Exception:
                    pass
                break
            losses.append(chunk)
            max_index = idx
    from tqdm.auto import tqdm
    pbar = tqdm(
        enumerate(dataloader),
        total=total_batches,
        desc=f"score step={step_id}",
        ncols=100,
    )
    pending = []
    last_done = None
    # Inference mode removes autograd bookkeeping during the potentially very
    # large candidate pass.  It is equivalent to no_grad here because scoring
    # never mutates model parameters.
    with torch.inference_mode():
        for batch_idx, batch in pbar:
            batch_idx += 1
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
            last_idx = max_index
            try:
                last_idx = batch_idx
            except Exception:
                last_idx = max_index
            if batch_range is not None:
                last_idx = last_done
            tmp_path = os.path.join(cache_dir, f"losses-{last_idx}.pt.tmp")
            final_path = os.path.join(cache_dir, f"losses-{last_idx}.pt")
            torch.save(torch.cat(pending, dim=0), tmp_path)
            os.replace(tmp_path, final_path)
            pending = []
    if restore_order is None:
        loss_vec = torch.cat(losses, dim=0)
    else:
        if len(losses) != len(restore_order):
            raise RuntimeError(
                "length-bucketed scorer produced a different number of loss "
                f"chunks ({len(losses)}) than batches ({len(restore_order)})"
            )
        total_rows = len(dataloader.dataset)
        loss_vec = torch.empty(total_rows, dtype=losses[0].dtype)
        for indices, chunk in zip(restore_order, losses):
            if len(indices) != len(chunk):
                raise RuntimeError(
                    "length-bucketed loss chunk/index mismatch: "
                    f"{len(chunk)} losses for {len(indices)} indices"
                )
            loss_vec[torch.as_tensor(indices, dtype=torch.long)] = chunk
    try:
        expected_len = len(dataloader.dataset)
        if batch_range is None and len(loss_vec) != expected_len:
            print(
                f"[loss_gap] warning: loss length {len(loss_vec)} != dataset length {expected_len}",
                flush=True,
            )
    except Exception:
        pass
    return loss_vec


def _dense_batch_embeddings(
    model,
    batch: Dict[str, torch.Tensor],
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    forward_inputs = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        forward_inputs["attention_mask"] = batch["attention_mask"]
    if use_amp:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            outputs = model(
                **forward_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    else:
        outputs = model(
            **forward_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden = None
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden = outputs.hidden_states[-1]
    elif hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    if hidden is None:
        raise RuntimeError(
            "dense_embed_sim requires hidden states from model forward; "
            "received no hidden state tensor."
        )

    if "attention_mask" in forward_inputs and forward_inputs["attention_mask"] is not None:
        mask = forward_inputs["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
    else:
        mask = torch.ones(
            hidden.shape[0], hidden.shape[1], 1, device=hidden.device, dtype=hidden.dtype
        )
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = (hidden * mask).sum(dim=1) / denom
    return F.normalize(pooled.float(), p=2, dim=1, eps=1e-12)


def _compute_dense_target_centroid(
    model,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    desc: str,
) -> torch.Tensor:
    model.eval()
    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None
    from tqdm.auto import tqdm

    emb_sum: Optional[torch.Tensor] = None
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=total_batches, desc=desc, ncols=100):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            emb = _dense_batch_embeddings(model, batch, use_amp, amp_dtype, device)
            batch_sum = emb.sum(dim=0).cpu()
            emb_sum = batch_sum if emb_sum is None else (emb_sum + batch_sum)
            count += int(emb.shape[0])
    if count <= 0 or emb_sum is None:
        raise RuntimeError("dense_embed_sim target validation dataset is empty.")
    centroid = emb_sum / float(count)
    return F.normalize(centroid, p=2, dim=0, eps=1e-12).cpu()


def _compute_dense_scores_for_dataset(
    model,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    target_centroid: torch.Tensor,
    desc: str,
) -> torch.Tensor:
    model.eval()
    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None
    from tqdm.auto import tqdm

    target = target_centroid.to(device=device, dtype=torch.float32)
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=total_batches, desc=desc, ncols=100):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            emb = _dense_batch_embeddings(model, batch, use_amp, amp_dtype, device)
            score = torch.matmul(emb, target)
            chunks.append(score.detach().cpu())
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0).float()


def _update_running_stats(stats: Dict[str, torch.Tensor],
                          loss_vec: torch.Tensor,
                          step_idx: int):
    # loss_vec is CPU tensor in dataset order for this step.
    if step_idx == 0:
        stats["first"] = loss_vec.clone()
        stats["last"] = loss_vec.clone()
        stats["sum"] = loss_vec.clone()
        stats["count"] = 1
        return
    stats["last"] = loss_vec.clone()
    stats["sum"] += loss_vec
    stats["count"] += 1


def _normalize_loss_by_first(loss_vec: torch.Tensor,
                             stats: Dict[str, torch.Tensor],
                             step_id: int,
                             eps: float) -> torch.Tensor:
    if "first_raw" not in stats:
        stats["first_raw"] = loss_vec.clone()
        stats["first_step_id"] = step_id
    denom = stats["first_raw"]
    return loss_vec / (denom + eps)


def _init_cov_stats(n: int) -> Dict[str, torch.Tensor]:
    return {
        "sum_lz": torch.zeros(n),
        "sum_lzlv": torch.zeros(n),
        "count": 0,
    }


def _init_var_stats(n: int) -> Dict[str, torch.Tensor]:
    return {
        "sum": torch.zeros(n),
        "sumsq": torch.zeros(n),
        "count": 0,
    }


def main():
    args = parse_args()
    if len(args.train_files) != len(args.train_file_names):
        raise ValueError("train_files and train_file_names must have the same length.")
    if args.target_valid_split and not (0.0 < args.target_valid_warmup_ratio < 1.0):
        raise ValueError(
            f"--target_valid_warmup_ratio must be in (0,1) when --target_valid_split is set; "
            f"got {args.target_valid_warmup_ratio}"
        )
    args.train_files = _resolve_train_files(args.train_files)
    if args.mode == "score" and args.score_metric == "dense_embed_sim" and args.lora:
        print(
            "[loss_gap] dense_embed_sim detected; forcing --no_lora for a base-model embedding baseline.",
            flush=True,
        )
        args.lora = False
    if args.freeze_to_final_n_layers < 0:
        raise ValueError(
            f"--freeze_to_final_n_layers must be >= 0, got {args.freeze_to_final_n_layers}"
        )
    if args.freeze_to_final_n_layers > 0 and args.lora:
        raise ValueError(
            "--freeze_to_final_n_layers is only supported with --no_lora "
            "(LoRA already constrains trainable parameters)."
        )
    if args.warmup_max_batches_per_checkpoint < 0:
        raise ValueError(
            "--warmup_max_batches_per_checkpoint must be >= 0, got "
            f"{args.warmup_max_batches_per_checkpoint}"
        )
    if args.warmup_gradient_accumulation_steps < 1:
        raise ValueError(
            "--warmup_gradient_accumulation_steps must be >= 1, got "
            f"{args.warmup_gradient_accumulation_steps}"
        )
    if args.warmup_data_sample_size < 0:
        raise ValueError(
            f"--warmup_data_sample_size must be >= 0, got {args.warmup_data_sample_size}"
        )
    if args.warmup_continuous_batches and args.warmup_max_batches_per_checkpoint <= 0:
        raise ValueError(
            "--warmup_continuous_batches requires --warmup_max_batches_per_checkpoint > 0"
        )
    if args.warmup_gradient_accumulation_steps > 1:
        if args.mode != "warmup":
            raise ValueError(
                "gradient-accumulated validation warmup is currently supported only in mode=warmup"
            )
        if not args.warmup_continuous_batches or args.warmup_max_batches_per_checkpoint <= 0:
            raise ValueError(
                "--warmup_gradient_accumulation_steps > 1 requires both "
                "--warmup_continuous_batches and --warmup_max_batches_per_checkpoint > 0"
            )
        if args.track_intra_batch_grad_cosine or args.track_rolling_grad_cosine:
            raise ValueError(
                "gradient-cosine tracking is not supported with accumulated warmup updates"
            )
    if (
        args.warmup_data_file
        and args.target_valid_split
        and args.mode in ("warmup", "warmup_and_score")
    ):
        raise ValueError(
            "--target_valid_split cannot be applied to --warmup_data_file. "
            "Use the explicit proxy only for warmup and keep canonical validation for score probes."
        )
    if (args.warmup_group_file is None) != (args.warmup_group_label is None):
        raise ValueError(
            "--warmup_group_file and --warmup_group_label must be supplied together."
        )
    if args.warmup_group_file is not None and args.mode != "warmup":
        raise ValueError("--warmup_group_file is supported only in mode=warmup.")
    if args.warmup_group_file is not None and args.warmup_data_file:
        raise ValueError(
            "--warmup_group_file cannot be combined with --warmup_data_file; "
            "grouping is defined over canonical validation rows."
        )
    if args.warmup_group_file is not None and len(args.target_task_names) != 1:
        raise ValueError(
            "Grouped warmup requires exactly one target task per process so the "
            "group label is unambiguous."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.bf16 or args.fp16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    warmup_lr = args.warmup_lr if args.warmup_lr is not None else args.learning_rate * args.lr_scale
    num_steps = args.warmup_epochs if args.warmup_epochs is not None else args.warmup_steps
    epochs_per_ckpt = max(1, int(args.warmup_epochs_per_ckpt))
    total_epochs = num_steps * epochs_per_ckpt
    if args.mode in ("warmup", "warmup_and_score"):
        print(
            f"[loss_gap] mode={args.mode} warmup_lr={warmup_lr} steps={num_steps} "
            f"epochs_per_ckpt={epochs_per_ckpt} total_epochs={total_epochs} "
            f"checkpoint_format={args.warmup_checkpoint_format}",
            flush=True,
        )
    else:
        print(
            f"[loss_gap] mode=score score_metric={args.score_metric} "
            f"(warmup args not used in score mode)",
            flush=True,
        )

    cache_only = args.mode == "score" and args.use_cache_only
    probe_seed = args.probe_seed if args.probe_seed is not None else args.seed
    target_valid_split_seed = (
        args.target_valid_split_seed if args.target_valid_split_seed is not None else args.seed
    )
    probe_requested = (args.probe_mmlu_test_n and args.probe_mmlu_test_n > 0) or (
        args.probe_mmlu_valid_n and args.probe_mmlu_valid_n > 0
    ) or (args.probe_target_valid_n and args.probe_target_valid_n > 0) or (
        args.probe_cot_n and args.probe_cot_n > 0
    )
    tokenizer = None

    def _resolve_torch_dtype():
        torch_dtype = None
        if args.torch_dtype:
            if args.torch_dtype == "auto":
                torch_dtype = "auto"
            elif args.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif args.torch_dtype == "float16":
                torch_dtype = torch.float16
            elif args.torch_dtype == "float32":
                torch_dtype = torch.float32
        elif args.bf16:
            torch_dtype = torch.bfloat16
        elif args.fp16:
            torch_dtype = torch.float16
        return torch_dtype

    def build_model():
        torch_dtype = _resolve_torch_dtype()
        init_path = None
        if args.mode == "warmup" and args.warmup_init_model_path:
            init_path = os.path.expanduser(args.warmup_init_model_path)
            _validate_model_name_or_path(init_path)

        if init_path is not None:
            if torch_dtype == "auto":
                # load_checkpoint_model expects an actual torch dtype.
                torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.bfloat16
            print(f"[loss_gap] warmup initialization from: {init_path}", flush=True)
            model = load_checkpoint_model(
                init_path,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )
            if args.warmup_init_merge_adapter_then_reinit_lora and isinstance(model, PeftModel):
                if not hasattr(model, "merge_and_unload"):
                    raise RuntimeError(
                        "warmup_init_merge_adapter_then_reinit_lora was requested but "
                        "PeftModel.merge_and_unload() is unavailable."
                    )
                print(
                    "[loss_gap] merging init PEFT adapter into base model, then re-applying current LoRA args.",
                    flush=True,
                )
                model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code)

        # Resize embeddings if needed (mirrors train.py behavior).
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            try:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                # Compatibility with Transformers releases predating
                # mean_resizing. The added pad token is loss-masked and the
                # embedding matrix remains frozen for LoRA warmup/scoring.
                model.resize_token_embeddings(len(tokenizer))
            if isinstance(model, PeftModel):
                model.get_input_embeddings().weight.requires_grad = False
                model.get_output_embeddings().weight.requires_grad = False
        if args.lora and not isinstance(model, PeftModel):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                rank_pattern=args.lora_rank_pattern,
                alpha_pattern=args.lora_alpha_pattern,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        elif args.lora and isinstance(model, PeftModel):
            # When initializing from an adapter checkpoint, current --lora_* args may be ignored.
            try:
                peft_cfg = None
                active_adapter = getattr(model, "active_adapter", None)
                if hasattr(model, "peft_config") and isinstance(model.peft_config, dict):
                    if active_adapter in model.peft_config:
                        peft_cfg = model.peft_config[active_adapter]
                    elif "default" in model.peft_config:
                        peft_cfg = model.peft_config["default"]
                    elif len(model.peft_config) > 0:
                        peft_cfg = next(iter(model.peft_config.values()))
                if peft_cfg is not None:
                    mismatch = []
                    if getattr(peft_cfg, "r", None) != args.lora_r:
                        mismatch.append(f"r(init={getattr(peft_cfg, 'r', None)} cli={args.lora_r})")
                    if getattr(peft_cfg, "lora_alpha", None) != args.lora_alpha:
                        mismatch.append(
                            f"alpha(init={getattr(peft_cfg, 'lora_alpha', None)} cli={args.lora_alpha})"
                        )
                    if float(getattr(peft_cfg, "lora_dropout", 0.0)) != float(args.lora_dropout):
                        mismatch.append(
                            f"dropout(init={getattr(peft_cfg, 'lora_dropout', None)} cli={args.lora_dropout})"
                        )
                    init_targets = getattr(peft_cfg, "target_modules", None)
                    if isinstance(init_targets, (list, tuple, set)):
                        init_targets_norm = sorted(str(x) for x in init_targets)
                    elif init_targets is None:
                        init_targets_norm = None
                    else:
                        init_targets_norm = [str(init_targets)]
                    cli_targets_norm = sorted(str(x) for x in args.lora_target_modules)
                    if init_targets_norm is not None and init_targets_norm != cli_targets_norm:
                        mismatch.append(
                            f"targets(init={init_targets_norm} cli={cli_targets_norm})"
                        )
                    init_rank_pattern = dict(getattr(peft_cfg, "rank_pattern", {}) or {})
                    init_alpha_pattern = dict(getattr(peft_cfg, "alpha_pattern", {}) or {})
                    if init_rank_pattern != args.lora_rank_pattern:
                        mismatch.append(
                            f"rank_pattern(init={init_rank_pattern} cli={args.lora_rank_pattern})"
                        )
                    if init_alpha_pattern != args.lora_alpha_pattern:
                        mismatch.append(
                            f"alpha_pattern(init={init_alpha_pattern} cli={args.lora_alpha_pattern})"
                        )
                    if mismatch:
                        print(
                            "[loss_gap] warning: initialized PEFT config differs from current --lora_* args; "
                            "current args will not take effect unless using "
                            "--warmup_init_merge_adapter_then_reinit_lora.\n"
                            + "  " + "; ".join(mismatch),
                            flush=True,
                        )
            except Exception as exc:
                print(f"[loss_gap] warning: unable to inspect PEFT init config: {exc}", flush=True)
        if args.freeze_to_final_n_layers > 0:
            trainable_meta = _apply_freeze_to_final_n_layers(
                model=model,
                final_n_layers=args.freeze_to_final_n_layers,
                train_final_norm=bool(args.train_final_norm),
                train_lm_head=bool(args.train_lm_head),
            )
        else:
            trainable_meta = _summarize_trainable_params(model)
            trainable_meta["trainable_scope_mode"] = "default"
        if args.mode in ("warmup", "warmup_and_score"):
            print(
                "[loss_gap] trainable scope: "
                f"mode={trainable_meta.get('trainable_scope_mode')} "
                f"trainable={trainable_meta.get('trainable_params')} / "
                f"{trainable_meta.get('total_params')} "
                f"({100.0 * float(trainable_meta.get('trainable_ratio', 0.0)):.4f}%)",
                flush=True,
            )
            if trainable_meta.get("trainable_scope_mode") == "freeze_to_final_n_layers":
                print(
                    "[loss_gap] final-layer freeze details: "
                    f"final_n={trainable_meta.get('final_n_layers_effective')}/"
                    f"{trainable_meta.get('total_transformer_layers')} "
                    f"start_layer={trainable_meta.get('first_trainable_layer_idx')} "
                    f"train_final_norm={trainable_meta.get('train_final_norm')} "
                    f"train_lm_head={trainable_meta.get('train_lm_head')}",
                    flush=True,
                )
        setattr(model, "_less_trainable_meta", trainable_meta)
        model.to(device)
        return model

    def load_model_from_checkpoint_dir(ckpt_path: str):
        """
        Load a model directly from a Hugging Face checkpoint/adaptor directory.
        This is used in score mode when ckpt_dir contains checkpoint-* dirs
        instead of step_*.pt state_dict files.
        """
        torch_dtype = _resolve_torch_dtype()
        if torch_dtype == "auto":
            # load_checkpoint_model expects an actual torch dtype.
            torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.bfloat16
        model = load_checkpoint_model(ckpt_path, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code)
        # Resize embeddings if tokenizer got expanded.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            try:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        return model

    for target_task in args.target_task_names:
        print(f"[loss_gap] target_task={target_task}", flush=True)
        output_dir = os.path.join(args.output_path, target_task)
        os.makedirs(output_dir, exist_ok=True)

        ckpt_dir = args.ckpt_dir
        if ckpt_dir is None:
            ckpt_dir = os.path.join(output_dir, "warmup_ckpts")

        warmup_meta = _load_warmup_meta(ckpt_dir)
        if args.mode == "score" and args.score_metric != "dense_embed_sim" and args.use_warmup_meta and warmup_meta:
            _apply_meta_to_args(warmup_meta, args)
        if args.mode == "score" and args.score_metric != "dense_embed_sim":
            _check_lora_meta_compat(warmup_meta, args)
            if warmup_meta.get("model_name_or_path") and args.model_name_or_path and warmup_meta.get("model_name_or_path") != args.model_name_or_path:
                print(
                    "[loss_gap] warning: model_name_or_path differs from warmup meta. "
                    f"warmup={warmup_meta.get('model_name_or_path')} scoring={args.model_name_or_path}",
                    flush=True,
                )
        score_cache_root = os.path.join(output_dir, "score_cache")
        if cache_only and args.score_cache_root:
            score_cache_root = args.score_cache_root
            if "{task}" in score_cache_root:
                score_cache_root = score_cache_root.format(task=target_task)
            elif not os.path.isabs(score_cache_root):
                score_cache_root = os.path.join(output_dir, score_cache_root)
            print(f"[loss_gap] cache-only score_cache_root={score_cache_root}", flush=True)

        use_chat_format = not args.no_chat_format
        # Normalize data_dir to avoid double "eval/eval" if user passes a path ending in /eval.
        data_dir = args.data_dir
        if os.path.basename(os.path.normpath(data_dir)) == "eval":
            data_dir = os.path.dirname(os.path.normpath(data_dir))

        data_collator = None
        needs_tokenizer = (not cache_only) or probe_requested
        if needs_tokenizer:
            _validate_model_name_or_path(args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
            add_padding_to_tokenizer(tokenizer)
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
        elif "{task}" in ckpt_dir:
            ckpt_dir = ckpt_dir.format(task=target_task)

        if args.mode == "warmup":
            os.makedirs(ckpt_dir, exist_ok=True)
            val_ds, warmup_data_meta = _prepare_warmup_dataset(
                target_task=target_task,
                data_dir=data_dir,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                use_chat_format=use_chat_format,
                chat_format=args.chat_format,
                mmlu_n_shot=args.mmlu_n_shot,
                warmup_data_file=args.warmup_data_file,
                warmup_data_sample_size=args.warmup_data_sample_size,
                seed=args.seed,
            )
            if args.warmup_group_file is not None:
                group_file = os.path.expandvars(
                    os.path.expanduser(str(args.warmup_group_file))
                ).format(task=target_task)
                assignments = load_validation_group_assignments(
                    group_file,
                    expected_n=len(val_ds),
                    task=target_task,
                )
                group_label = str(args.warmup_group_label)
                selected_indices = [
                    idx for idx, assignment in enumerate(assignments)
                    if str(assignment) == group_label
                ]
                if not selected_indices:
                    raise ValueError(
                        f"warmup group {group_label!r} has no rows in {group_file}"
                    )
                val_ds = val_ds.select(selected_indices)
                warmup_data_meta = dict(warmup_data_meta)
                warmup_data_meta.update(
                    {
                        "kind": "canonical_validation_group",
                        "group_file": os.path.abspath(group_file),
                        "group_label": group_label,
                        "group_rows": len(selected_indices),
                        "full_validation_rows": len(assignments),
                        "group_indices": selected_indices,
                    }
                )
                print(
                    f"[loss_gap] selected validation group {group_label!r}: "
                    f"{len(selected_indices)}/{len(assignments)} rows from {group_file}",
                    flush=True,
                )
            if len(val_ds) == 0:
                raise ValueError("Warmup validation dataset is empty.")
            target_valid_split = None
            if args.target_valid_split:
                target_valid_split = _prepare_target_valid_split(
                    target_task=target_task,
                    data_dir=data_dir,
                    dataset_len=len(val_ds),
                    ckpt_dir=ckpt_dir,
                    mode=args.mode,
                    split_strategy=args.target_valid_split_strategy,
                    warmup_ratio=args.target_valid_warmup_ratio,
                    split_seed=target_valid_split_seed,
                    split_file_arg=args.target_valid_split_file,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                val_ds = val_ds.select(target_valid_split["warmup_indices"])
                print(
                    f"[loss_gap] target_valid split warmup subset: "
                    f"warmup_n={target_valid_split['warmup_n']} probe_n={target_valid_split['probe_n']} "
                    f"path={target_valid_split['split_path']}",
                    flush=True,
                )
            if len(val_ds) == 0:
                raise ValueError("Warmup validation dataset is empty after applying target_valid split.")
            val_loader = DataLoader(
                val_ds,
                batch_size=args.warmup_batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )

            model = build_model()
            trainable_meta = getattr(model, "_less_trainable_meta", _summarize_trainable_params(model))
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError(
                    "No trainable parameters found for warmup. "
                    "Check LoRA/freeze configuration."
                )
            if args.warmup_optim == "adamw":
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=warmup_lr, weight_decay=args.weight_decay
                )
            else:
                optimizer = torch.optim.SGD(
                    trainable_params, lr=warmup_lr, weight_decay=args.weight_decay
                )
            try:
                scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
            except AttributeError:
                scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
            total_batches = None
            try:
                batches_per_checkpoint = len(val_loader) * epochs_per_ckpt
                if args.warmup_max_batches_per_checkpoint > 0:
                    batches_per_checkpoint = min(
                        batches_per_checkpoint,
                        args.warmup_max_batches_per_checkpoint,
                    )
                total_batches = batches_per_checkpoint * num_steps
            except Exception:
                total_batches = None
            val_losses: List[float] = []
            optimizer_updates_by_step: List[int] = []
            microbatches_by_step: List[int] = []
            examples_seen_by_step: List[int] = []
            benchmark_test_losses: List[float] = []
            # Build benchmark loader from held-out test set if requested.
            benchmark_loader = None
            if args.benchmark_eval_dir:
                try:
                    target = args.target_task_names[0] if args.target_task_names else "tydiqa"
                    if target == "bbh":
                        bench_ds = get_bbh_test_dataset(
                            args.benchmark_eval_dir,
                            tokenizer,
                            args.max_seq_length,
                            use_chat_format=use_chat_format,
                            chat_format=args.chat_format,
                            max_examples=200,
                            seed=42,
                        )
                        bench_label = "BBH test"
                    elif target == "mmlu":
                        mmlu_data_dir = os.path.join(args.benchmark_eval_dir, "eval", "mmlu")
                        bench_ds = get_mmlu_test_dataset(
                            mmlu_data_dir,
                            tokenizer,
                            args.max_seq_length,
                            use_chat_format=use_chat_format,
                            chat_format=args.chat_format,
                        )
                        bench_label = "MMLU dev"
                    else:
                        bench_ds = get_tydiqa_goldp_dataset(
                            args.benchmark_eval_dir,
                            tokenizer,
                            args.max_seq_length,
                            use_chat_format=use_chat_format,
                            chat_format=args.chat_format,
                            max_examples=200,
                            seed=42,
                        )
                        bench_label = "TyDiQA goldp-dev"
                    benchmark_loader = DataLoader(
                        bench_ds,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=data_collator,
                    )
                    print(
                        f"[loss_gap] benchmark_eval: loaded {len(bench_ds)} {bench_label} examples (held-out)",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[loss_gap] warning: failed to load benchmark dataset: {exc}", flush=True)
            grad_cosine_events: List[Dict[str, object]] = []
            grad_cosine_interval = int(args.grad_cosine_interval)
            if (args.track_intra_batch_grad_cosine or args.track_rolling_grad_cosine) and grad_cosine_interval <= 0:
                grad_cosine_interval = 1
            grad_cosine_log_jsonl = (
                args.grad_cosine_log_jsonl
                if args.grad_cosine_log_jsonl
                else os.path.join(ckpt_dir, "intra_batch_grad_cosine.jsonl")
            )
            lora_named_params = _iter_trainable_lora_params(model) if (args.track_intra_batch_grad_cosine or args.track_rolling_grad_cosine) else []
            global_batch_idx = 0
            continuous_val_iter = iter(val_loader) if args.warmup_continuous_batches else None
            if args.save_warmup_initial_state:
                if args.warmup_checkpoint_format != "trainable_state_dict":
                    raise ValueError(
                        "--save_warmup_initial_state requires "
                        "--warmup_checkpoint_format trainable_state_dict"
                    )
                initial_state = {
                    name: param.detach().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                if not initial_state:
                    raise RuntimeError(
                        "--save_warmup_initial_state requested but no trainable parameters found"
                    )
                initial_path = os.path.join(ckpt_dir, "step_0.pt")
                if not os.path.exists(initial_path):
                    _atomic_torch_save(initial_state, initial_path)
                    print(f"[loss_gap] saved initial checkpoint {initial_path}", flush=True)
            pbar = tqdm(total=total_batches, desc="[loss_gap] warmup", unit="update")
            for step_idx in range(num_steps):
                model.train()
                step_loss_sum = 0.0
                step_batches = 0
                step_microbatches = 0
                step_examples = 0
                accumulation_steps = int(args.warmup_gradient_accumulation_steps)
                if accumulation_steps > 1:
                    # This branch is deliberately limited to the continuous,
                    # explicitly capped scale-control protocol validated above.
                    # It keeps the physical batch at one while averaging many
                    # unique proxy rows into each optimizer update.
                    while step_batches < args.warmup_max_batches_per_checkpoint:
                        optimizer.zero_grad()
                        for _ in range(accumulation_steps):
                            try:
                                batch = next(continuous_val_iter)
                            except StopIteration:
                                continuous_val_iter = iter(val_loader)
                                batch = next(continuous_val_iter)
                            batch = {k: v.to(device) for k, v in batch.items()}
                            global_batch_idx += 1
                            step_microbatches += 1
                            step_examples += int(batch["input_ids"].shape[0])
                            if use_amp:
                                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                                    outputs = model(**batch)
                                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                            else:
                                outputs = model(**batch)
                                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                            scaled_loss = loss / accumulation_steps
                            if scaler.is_enabled():
                                scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()
                            step_loss_sum += float(loss.item())
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        step_batches += 1
                        if pbar is not None:
                            pbar.update(1)
                else:
                    for _ in range(epochs_per_ckpt):
                        val_iter = continuous_val_iter if args.warmup_continuous_batches else iter(val_loader)
                        while True:
                            if (
                                args.warmup_max_batches_per_checkpoint > 0
                                and step_batches >= args.warmup_max_batches_per_checkpoint
                            ):
                                break
                            try:
                                batch = next(val_iter)
                            except StopIteration:
                                if args.warmup_continuous_batches:
                                    continuous_val_iter = iter(val_loader)
                                    val_iter = continuous_val_iter
                                    continue
                                break
                            batch = {k: v.to(device) for k, v in batch.items()}
                            global_batch_idx += 1
                            step_microbatches += 1
                            step_examples += int(batch["input_ids"].shape[0])

                            if (args.track_intra_batch_grad_cosine or args.track_rolling_grad_cosine) and grad_cosine_interval > 0:
                                if (global_batch_idx % grad_cosine_interval) == 0:
                                    if args.track_rolling_grad_cosine:
                                        cosine, status = _compute_rolling_k_grad_cosine(
                                            model=model,
                                            first_batch=batch,
                                            batch_iter=val_iter,
                                            named_params=lora_named_params,
                                            use_amp=use_amp,
                                            amp_dtype=amp_dtype,
                                            scaler=scaler,
                                            k=int(args.grad_cosine_virtual_k),
                                        )
                                    else:
                                        cosine, status = _compute_intra_batch_grad_cosine(
                                            model=model,
                                            batch=batch,
                                            named_params=lora_named_params,
                                            use_amp=use_amp,
                                            amp_dtype=amp_dtype,
                                            scaler=scaler,
                                        )
                                    event = {
                                        "global_batch": int(global_batch_idx),
                                        "warmup_step": int(step_idx + 1),
                                        "cosine": cosine,
                                        "status": status,
                                        "mode": "rolling_k" if args.track_rolling_grad_cosine else "intra_batch_half",
                                        "k": int(args.grad_cosine_virtual_k) if args.track_rolling_grad_cosine else None,
                                    }
                                    grad_cosine_events.append(event)
                                    try:
                                        with open(grad_cosine_log_jsonl, "a", encoding="utf-8") as f:
                                            f.write(json.dumps(event) + "\n")
                                    except Exception:
                                        pass
                                    if cosine is not None:
                                        print(
                                            f"[loss_gap] grad_cosine[{event['mode']}] step={step_idx+1} "
                                            f"global_batch={global_batch_idx} cosine={cosine:.6f}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"[loss_gap] grad_cosine[{event['mode']}] step={step_idx+1} "
                                            f"global_batch={global_batch_idx} skipped status={status}",
                                            flush=True,
                                        )

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
                            step_loss_sum += float(loss.item())
                            step_batches += 1
                            if pbar is not None:
                                pbar.update(1)
                        if (
                            args.warmup_max_batches_per_checkpoint > 0
                            and step_batches >= args.warmup_max_batches_per_checkpoint
                        ):
                            break
                if step_batches == 0:
                    raise RuntimeError(
                        "Warmup checkpoint received zero optimizer updates. "
                        "Check the validation dataset and batch cap."
                    )
                avg_loss = step_loss_sum / max(step_microbatches, 1)
                val_losses.append(float(avg_loss))
                optimizer_updates_by_step.append(int(step_batches))
                microbatches_by_step.append(int(step_microbatches))
                examples_seen_by_step.append(int(step_examples))
                print(
                    f"[loss_gap] step={step_idx+1}/{num_steps} val_loss={avg_loss:.6f} "
                    f"optimizer_updates={step_batches} microbatches={step_microbatches} "
                    f"examples_seen={step_examples}",
                    flush=True,
                )

                step_num = step_idx + 1
                if args.warmup_checkpoint_format == "state_dict":
                    ckpt_path = os.path.join(ckpt_dir, f"step_{step_num}.pt")
                    _atomic_torch_save(model.state_dict(), ckpt_path)
                elif args.warmup_checkpoint_format == "trainable_state_dict":
                    ckpt_path = os.path.join(ckpt_dir, f"step_{step_num}.pt")
                    trainable_state = {
                        name: param.detach().cpu()
                        for name, param in model.named_parameters()
                        if param.requires_grad
                    }
                    if len(trainable_state) == 0:
                        raise RuntimeError(
                            "warmup_checkpoint_format=trainable_state_dict but no trainable parameters found."
                        )
                    _atomic_torch_save(trainable_state, ckpt_path)
                else:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{step_num}")
                    tmp_ckpt_path = f"{ckpt_path}.tmp"
                    if os.path.isdir(tmp_ckpt_path):
                        shutil.rmtree(tmp_ckpt_path, ignore_errors=True)
                    # For LoRA/PEFT this writes compact adapter checkpoints.
                    model.save_pretrained(tmp_ckpt_path, safe_serialization=True)
                    if os.path.isdir(ckpt_path):
                        shutil.rmtree(ckpt_path, ignore_errors=True)
                    os.replace(tmp_ckpt_path, ckpt_path)
                print(f"[loss_gap] saved checkpoint {ckpt_path}", flush=True)

                # ── benchmark test-set loss (TyDiQA one-shot, no grad) ────────────
                if benchmark_loader is not None:
                    model.eval()
                    bench_loss_sum = 0.0
                    bench_batches = 0
                    with torch.no_grad():
                        for bench_batch in benchmark_loader:
                            bench_batch = {k: v.to(device) for k, v in bench_batch.items()}
                            if use_amp:
                                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                                    bench_out = model(**bench_batch)
                            else:
                                bench_out = model(**bench_batch)
                            b_loss = bench_out.loss if hasattr(bench_out, "loss") else bench_out[0]
                            b_val = float(b_loss.item())
                            if not (b_val != b_val):  # skip NaN
                                bench_loss_sum += b_val
                                bench_batches += 1
                    model.train()
                    bench_avg = bench_loss_sum / max(bench_batches, 1)
                    benchmark_test_losses.append(bench_avg)
                    print(
                        f"[loss_gap] step={step_idx+1}/{num_steps} benchmark_test_loss={bench_avg:.6f}",
                        flush=True,
                    )

                if args.save_warmup_optimizer_state:
                    try:
                        opt_state = _extract_optimizer_state_with_names(optimizer, model)
                        latest_opt_path = os.path.join(ckpt_dir, "optimizer_with_names.pt")
                        torch.save(opt_state, latest_opt_path)
                        print(
                            f"[loss_gap] saved optimizer state {latest_opt_path} (step {step_idx+1})",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[loss_gap] warning: failed to save optimizer state at step {step_idx+1}: {exc}",
                            flush=True,
                        )

                meta_path = os.path.join(ckpt_dir, "meta.json")
                warmup_meta = {
                    "warmup_steps": args.warmup_steps,
                    "warmup_epochs": num_steps,
                    "warmup_epochs_per_ckpt": epochs_per_ckpt,
                    "warmup_total_epochs": total_epochs,
                    "warmup_lr": warmup_lr,
                    "warmup_optim": args.warmup_optim,
                    "model_name_or_path": args.model_name_or_path,
                    "lora": args.lora,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "lora_rank_pattern": args.lora_rank_pattern,
                    "lora_alpha_pattern": args.lora_alpha_pattern,
                    "freeze_to_final_n_layers": int(args.freeze_to_final_n_layers),
                    "train_final_norm": bool(args.train_final_norm),
                    "train_lm_head": bool(args.train_lm_head),
                    "trainable_scope_mode": trainable_meta.get("trainable_scope_mode"),
                    "trainable_params": trainable_meta.get("trainable_params"),
                    "total_params": trainable_meta.get("total_params"),
                    "trainable_ratio": trainable_meta.get("trainable_ratio"),
                    "trainable_param_name_samples": trainable_meta.get(
                        "trainable_param_name_samples", []
                    ),
                    "final_n_layers_effective": trainable_meta.get("final_n_layers_effective"),
                    "total_transformer_layers": trainable_meta.get("total_transformer_layers"),
                    "first_trainable_layer_idx": trainable_meta.get("first_trainable_layer_idx"),
                    "warmup_init_model_path": args.warmup_init_model_path,
                    "warmup_init_merge_adapter_then_reinit_lora": bool(
                        args.warmup_init_merge_adapter_then_reinit_lora
                    ),
                    "max_seq_length": args.max_seq_length,
                    "chat_format": args.chat_format,
                    "use_chat_format": use_chat_format,
                    "bf16": bool(args.bf16),
                    "fp16": bool(args.fp16),
                    "torch_dtype": args.torch_dtype,
                    "weight_decay": args.weight_decay,
                    "warmup_batch_size": args.warmup_batch_size,
                    "warmup_gradient_accumulation_steps": int(
                        args.warmup_gradient_accumulation_steps
                    ),
                    "warmup_max_batches_per_checkpoint": int(
                        args.warmup_max_batches_per_checkpoint
                    ),
                    "warmup_continuous_batches": bool(args.warmup_continuous_batches),
                    "optimizer_updates_by_step": optimizer_updates_by_step,
                    "optimizer_updates_total": int(sum(optimizer_updates_by_step)),
                    "warmup_microbatches_by_step": microbatches_by_step,
                    "warmup_microbatches_total": int(sum(microbatches_by_step)),
                    "warmup_examples_seen_by_step": examples_seen_by_step,
                    "warmup_examples_seen_total": int(sum(examples_seen_by_step)),
                    "warmup_full_proxy_passes_floor": (
                        int(sum(examples_seen_by_step) // len(val_ds))
                        if args.warmup_data_file and len(val_ds) > 0
                        else None
                    ),
                    "warmup_checkpoint_format": args.warmup_checkpoint_format,
                    "save_warmup_optimizer_state": bool(args.save_warmup_optimizer_state),
                    "save_warmup_initial_state": bool(args.save_warmup_initial_state),
                    "track_intra_batch_grad_cosine": bool(args.track_intra_batch_grad_cosine),
                    "track_rolling_grad_cosine": bool(args.track_rolling_grad_cosine),
                    "grad_cosine_interval": int(grad_cosine_interval),
                    "grad_cosine_virtual_k": int(args.grad_cosine_virtual_k),
                    "grad_cosine_log_jsonl": grad_cosine_log_jsonl if (args.track_intra_batch_grad_cosine or args.track_rolling_grad_cosine) else None,
                    "grad_cosine_event_count": len(grad_cosine_events),
                    "grad_cosine_events": grad_cosine_events,
                    "seed": args.seed,
                    "target_task": target_task,
                    "data_dir": data_dir,
                    "validation_examples": int(len(val_ds)),
                    "warmup_data": warmup_data_meta,
                    "warmup_group_file": (
                        os.path.abspath(
                            os.path.expandvars(
                                os.path.expanduser(str(args.warmup_group_file))
                            ).format(task=target_task)
                        )
                        if args.warmup_group_file is not None
                        else None
                    ),
                    "warmup_group_label": args.warmup_group_label,
                    "mmlu_group_name": (
                        os.environ.get("TACS_MMLU_GROUP_NAME") if target_task == "mmlu" else None
                    ),
                    "mmlu_subjects": (
                        os.environ.get("TACS_MMLU_SUBJECTS", "").split()
                        if target_task == "mmlu" else None
                    ),
                    "mmlu_group_manifest_sha256": (
                        os.environ.get("TACS_MMLU_GROUP_MANIFEST_SHA256")
                        if target_task == "mmlu" else None
                    ),
                    "target_valid_split": bool(args.target_valid_split),
                    "target_valid_split_strategy": (
                        args.target_valid_split_strategy if args.target_valid_split else None
                    ),
                    "target_valid_warmup_ratio": (
                        float(args.target_valid_warmup_ratio) if args.target_valid_split else None
                    ),
                    "target_valid_split_seed": (
                        int(target_valid_split.get("split_seed"))
                        if (args.target_valid_split and target_valid_split is not None)
                        else None
                    ),
                    "target_valid_split_file": (
                        target_valid_split.get("split_path")
                        if (args.target_valid_split and target_valid_split is not None)
                        else None
                    ),
                    "target_valid_split_warmup_n": (
                        int(target_valid_split.get("warmup_n"))
                        if (args.target_valid_split and target_valid_split is not None)
                        else None
                    ),
                    "target_valid_split_probe_n": (
                        int(target_valid_split.get("probe_n"))
                        if (args.target_valid_split and target_valid_split is not None)
                        else None
                    ),
                    "val_loss_by_step": val_losses,
                    "benchmark_test_loss_by_step": benchmark_test_losses,
                    "completed_steps": step_idx + 1,
                }
                _atomic_json_dump(warmup_meta, meta_path, indent=2)
            if pbar is not None:
                pbar.close()

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if args.mode == "warmup_and_score":
            # Combined warmup + scoring: train on val, score candidates after
            # each epoch. No checkpoint saving — keeps model in memory.
            # Ideal for full-model (no LoRA) warmup where checkpoints are huge.
            os.makedirs(ckpt_dir, exist_ok=True)
            score_cache_root = os.path.join(output_dir, "score_cache")

            # Load val dataset
            val_ds, warmup_data_meta = _prepare_warmup_dataset(
                target_task=target_task,
                data_dir=data_dir,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                use_chat_format=use_chat_format,
                chat_format=args.chat_format,
                mmlu_n_shot=args.mmlu_n_shot,
                warmup_data_file=args.warmup_data_file,
                warmup_data_sample_size=args.warmup_data_sample_size,
                seed=args.seed,
            )
            if len(val_ds) == 0:
                raise ValueError("Warmup validation dataset is empty.")
            if args.target_valid_split:
                target_valid_split = _prepare_target_valid_split(
                    target_task=target_task,
                    data_dir=data_dir,
                    dataset_len=len(val_ds),
                    ckpt_dir=ckpt_dir,
                    mode="warmup",
                    split_strategy=args.target_valid_split_strategy,
                    warmup_ratio=args.target_valid_warmup_ratio,
                    split_seed=target_valid_split_seed,
                    split_file_arg=args.target_valid_split_file,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                val_ds = val_ds.select(target_valid_split["warmup_indices"])
            val_loader = DataLoader(
                val_ds,
                batch_size=args.warmup_batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
            print(f"[warmup_and_score] val dataset: {len(val_ds)} examples", flush=True)

            # Load candidate datasets
            candidate_names = list(args.train_file_names)
            candidate_loaders: Dict[str, DataLoader] = {}
            for name, path in zip(args.train_file_names, args.train_files):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"train_file not found: {path}")
                ds = _prepare_candidate_dataset(
                    path,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    percentage=args.candidate_percentage,
                    seed=args.seed,
                    chat_format=args.chat_format,
                )
                candidate_loaders[name] = _make_candidate_loader(
                    ds,
                    batch_size=args.loss_batch_size,
                    collate_fn=data_collator,
                    length_bucketed=args.length_bucket_candidates,
                )
                print(f"[warmup_and_score] candidate '{name}': {len(ds)} examples", flush=True)

            # Build model
            model = build_model()
            trainable_meta = getattr(model, "_less_trainable_meta", _summarize_trainable_params(model))
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in model.parameters())
            print(
                f"[warmup_and_score] trainable params: {n_trainable:,} / {n_total:,} "
                f"({n_trainable/n_total*100:.2f}%)",
                flush=True,
            )
            if len(trainable_params) == 0:
                raise RuntimeError("No trainable parameters found.")

            if args.warmup_optim == "adamw":
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=warmup_lr, weight_decay=args.weight_decay
                )
            else:
                optimizer = torch.optim.SGD(
                    trainable_params, lr=warmup_lr, weight_decay=args.weight_decay
                )
            try:
                scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
            except AttributeError:
                scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

            val_losses: List[float] = []
            optimizer_updates_by_step: List[int] = []

            # Determine which steps to score
            score_step_ids = set()
            if args.score_ckpt_ids:
                score_step_ids = set(args.score_ckpt_ids)
            else:
                score_step_ids = set(range(1, num_steps + 1))

            for step_idx in range(num_steps):
                step_num = step_idx + 1

                # --- Train one epoch on val data ---
                model.train()
                step_loss_sum = 0.0
                step_batches = 0
                for _ in range(epochs_per_ckpt):
                    for batch in val_loader:
                        if (
                            args.warmup_max_batches_per_checkpoint > 0
                            and step_batches >= args.warmup_max_batches_per_checkpoint
                        ):
                            break
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
                        step_loss_sum += float(loss.item())
                        step_batches += 1
                    if (
                        args.warmup_max_batches_per_checkpoint > 0
                        and step_batches >= args.warmup_max_batches_per_checkpoint
                    ):
                        break
                if step_batches == 0:
                    raise RuntimeError(
                        "Warmup checkpoint received zero optimizer updates. "
                        "Check the validation dataset and batch cap."
                    )
                avg_loss = step_loss_sum / max(step_batches, 1)
                val_losses.append(float(avg_loss))
                optimizer_updates_by_step.append(int(step_batches))
                print(
                    f"[warmup_and_score] step={step_num}/{num_steps} "
                    f"val_loss={avg_loss:.6f} optimizer_updates={step_batches}",
                    flush=True,
                )

                # --- Score candidates at this step (if requested) ---
                if step_num in score_step_ids:
                    print(f"[warmup_and_score] scoring candidates at step={step_num}", flush=True)
                    for name in candidate_names:
                        loader = candidate_loaders[name]
                        cache_dir = os.path.join(score_cache_root, name, f"step_{step_num:04d}")
                        meta = {
                            "dataset_len": len(loader.dataset),
                            "batch_size": loader.batch_size,
                            "seed": args.seed,
                            "candidate_percentage": args.candidate_percentage,
                            "max_seq_length": args.max_seq_length,
                        }
                        loss_vec = _compute_losses_for_dataset(
                            model,
                            loader,
                            device,
                            use_amp,
                            amp_dtype,
                            cache_dir=cache_dir,
                            step_id=step_num,
                            meta=meta,
                            save_interval=args.loss_save_interval,
                        )
                        print(
                            f"[warmup_and_score]   {name}: {len(loss_vec)} losses, "
                            f"mean={loss_vec.mean():.6f}",
                            flush=True,
                        )
                    model.train()  # back to train mode

            # Save meta
            meta_path = os.path.join(ckpt_dir, "meta.json")
            warmup_meta = {
                "mode": "warmup_and_score",
                "warmup_steps": args.warmup_steps,
                "warmup_epochs": num_steps,
                "warmup_epochs_per_ckpt": epochs_per_ckpt,
                "warmup_total_epochs": total_epochs,
                "warmup_lr": warmup_lr,
                "warmup_optim": args.warmup_optim,
                "model_name_or_path": args.model_name_or_path,
                "lora": args.lora,
                "lora_r": args.lora_r if args.lora else None,
                "lora_alpha": args.lora_alpha if args.lora else None,
                "lora_dropout": args.lora_dropout if args.lora else None,
                "lora_target_modules": args.lora_target_modules if args.lora else None,
                "lora_rank_pattern": args.lora_rank_pattern if args.lora else None,
                "lora_alpha_pattern": args.lora_alpha_pattern if args.lora else None,
                "trainable_params": n_trainable,
                "total_params": n_total,
                "score_step_ids": sorted(score_step_ids),
                "candidate_names": candidate_names,
                "seed": args.seed,
                "target_task": target_task,
                "validation_examples": int(len(val_ds)),
                "warmup_data": warmup_data_meta,
                "mmlu_group_name": (
                    os.environ.get("TACS_MMLU_GROUP_NAME") if target_task == "mmlu" else None
                ),
                "mmlu_subjects": (
                    os.environ.get("TACS_MMLU_SUBJECTS", "").split()
                    if target_task == "mmlu" else None
                ),
                "mmlu_group_manifest_sha256": (
                    os.environ.get("TACS_MMLU_GROUP_MANIFEST_SHA256")
                    if target_task == "mmlu" else None
                ),
                "val_loss_by_step": val_losses,
                "warmup_max_batches_per_checkpoint": int(
                    args.warmup_max_batches_per_checkpoint
                ),
                "optimizer_updates_by_step": optimizer_updates_by_step,
                "optimizer_updates_total": int(sum(optimizer_updates_by_step)),
                "warmup_checkpoint_format": "none (on-the-fly scoring)",
            }
            _atomic_json_dump(warmup_meta, meta_path, indent=2)
            print(f"[warmup_and_score] saved meta to {meta_path}", flush=True)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if args.mode == "score" and args.score_metric == "dense_embed_sim":
            score_metric = args.score_metric
            candidate_names = list(args.train_file_names)
            candidate_datasets: Dict[str, torch.utils.data.Dataset] = {}
            candidate_loaders: Dict[str, DataLoader] = {}
            if not cache_only and not args.probe_only:
                for name, path in zip(args.train_file_names, args.train_files):
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"train_file not found: {path}")
                    ds = _prepare_candidate_dataset(
                        path,
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        percentage=args.candidate_percentage,
                        seed=args.seed,
                        chat_format=args.chat_format,
                    )
                    candidate_datasets[name] = ds
                for name, ds in candidate_datasets.items():
                    candidate_loaders[name] = _make_candidate_loader(
                        ds,
                        batch_size=args.loss_batch_size,
                        collate_fn=data_collator,
                        length_bucketed=args.length_bucket_candidates,
                    )
            else:
                for name, path in zip(args.train_file_names, args.train_files):
                    if not os.path.exists(path):
                        print(
                            f"[loss_gap] warning: train_file not found (ignored in cache-only): {path}",
                            flush=True,
                        )

            probe_loaders: Dict[str, DataLoader] = {}
            if probe_requested:
                if args.probe_mmlu_test_n and args.probe_mmlu_test_n > 0:
                    mmlu_ds = _prepare_mmlu_split_dataset(
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        max_length=args.max_seq_length,
                        num_samples=args.probe_mmlu_test_n,
                        seed=probe_seed,
                        split="test",
                        use_chat_format=use_chat_format,
                        chat_format=args.chat_format,
                        mmlu_n_shot=args.mmlu_n_shot,
                    )
                    probe_loaders["mmlu_test"] = DataLoader(
                        mmlu_ds,
                        batch_size=args.loss_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                    )
                if args.probe_mmlu_valid_n and args.probe_mmlu_valid_n > 0:
                    mmlu_ds = _prepare_mmlu_split_dataset(
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        max_length=args.max_seq_length,
                        num_samples=args.probe_mmlu_valid_n,
                        seed=probe_seed,
                        split="dev",
                        use_chat_format=use_chat_format,
                        chat_format=args.chat_format,
                        mmlu_n_shot=args.mmlu_n_shot,
                    )
                    probe_loaders["mmlu_valid"] = DataLoader(
                        mmlu_ds,
                        batch_size=args.loss_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                    )
                if args.probe_target_valid_n and args.probe_target_valid_n > 0:
                    val_ds = get_dataset(
                        target_task,
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        max_length=args.max_seq_length,
                        use_chat_format=use_chat_format,
                        chat_format=args.chat_format,
                        mmlu_n_shot=args.mmlu_n_shot,
                    )
                    if args.target_valid_split:
                        target_valid_split = _prepare_target_valid_split(
                            target_task=target_task,
                            data_dir=data_dir,
                            dataset_len=len(val_ds),
                            ckpt_dir=ckpt_dir,
                            mode=args.mode,
                            split_strategy=args.target_valid_split_strategy,
                            warmup_ratio=args.target_valid_warmup_ratio,
                            split_seed=target_valid_split_seed,
                            split_file_arg=args.target_valid_split_file,
                            mmlu_n_shot=args.mmlu_n_shot,
                        )
                        val_ds = val_ds.select(target_valid_split["probe_indices"])
                        print(
                            f"[loss_gap] target_valid split probe subset: "
                            f"warmup_n={target_valid_split['warmup_n']} probe_n={target_valid_split['probe_n']} "
                            f"path={target_valid_split['split_path']}",
                            flush=True,
                        )
                    if args.probe_target_valid_n < len(val_ds):
                        val_ds = _sample_hf_dataset(val_ds, args.probe_target_valid_n, probe_seed)
                    probe_loaders["target_valid"] = DataLoader(
                        val_ds,
                        batch_size=args.loss_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                    )
                if args.probe_cot_n and args.probe_cot_n > 0:
                    cot_path = args.probe_cot_file
                    if cot_path is None:
                        for name, path in zip(args.train_file_names, args.train_files):
                            if name.lower() == "cot":
                                cot_path = path
                                break
                    if cot_path is None:
                        raise ValueError(
                            "--probe_cot_n set but no COT file provided and no 'cot' in train_file_names"
                        )
                    cot_ds = _prepare_candidate_dataset_count(
                        cot_path,
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        num_samples=args.probe_cot_n,
                        seed=probe_seed,
                        chat_format=args.chat_format,
                    )
                    probe_loaders["cot_sample"] = DataLoader(
                        cot_ds,
                        batch_size=args.loss_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                    )

            model = None
            if (not cache_only) or probe_requested:
                model = build_model()

            dense_target_cache_dir = os.path.join(output_dir, "score_cache", "_dense_embed_sim")
            dense_target_cache_path = os.path.join(dense_target_cache_dir, "target_centroid.pt")
            target_centroid = None
            if os.path.exists(dense_target_cache_path):
                target_centroid = torch.load(dense_target_cache_path, map_location="cpu").float().view(-1)
            if target_centroid is None:
                if model is None:
                    raise FileNotFoundError(
                        f"dense_embed_sim target centroid cache not found: {dense_target_cache_path}. "
                        "Re-run without --use_cache_only at least once to populate cache."
                    )
                val_ds = get_dataset(
                    target_task,
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    use_chat_format=use_chat_format,
                    chat_format=args.chat_format,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                if args.target_valid_split:
                    target_valid_split = _prepare_target_valid_split(
                        target_task=target_task,
                        data_dir=data_dir,
                        dataset_len=len(val_ds),
                        ckpt_dir=ckpt_dir,
                        mode=args.mode,
                        split_strategy=args.target_valid_split_strategy,
                        warmup_ratio=args.target_valid_warmup_ratio,
                        split_seed=target_valid_split_seed,
                        split_file_arg=args.target_valid_split_file,
                        mmlu_n_shot=args.mmlu_n_shot,
                    )
                    val_ds = val_ds.select(target_valid_split["warmup_indices"])
                    print(
                        f"[loss_gap] dense target_valid split uses warmup subset: "
                        f"warmup_n={target_valid_split['warmup_n']} probe_n={target_valid_split['probe_n']} "
                        f"path={target_valid_split['split_path']}",
                        flush=True,
                    )
                if len(val_ds) == 0:
                    raise ValueError("dense_embed_sim target validation dataset is empty.")
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.loss_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                target_centroid = _compute_dense_target_centroid(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    desc=f"dense target {target_task}",
                )
                os.makedirs(dense_target_cache_dir, exist_ok=True)
                torch.save(target_centroid, dense_target_cache_path)
                _atomic_json_dump(
                    {
                        "target_task": target_task,
                        "dataset_len": len(val_ds),
                        "max_seq_length": args.max_seq_length,
                        "chat_format": args.chat_format,
                        "use_chat_format": bool(use_chat_format),
                        "model_name_or_path": args.model_name_or_path,
                    },
                    os.path.join(dense_target_cache_dir, "meta.json"),
                    indent=2,
                )
                print(
                    f"[loss_gap] dense_embed_sim cached target centroid at {dense_target_cache_path}",
                    flush=True,
                )

            if not args.probe_only:
                for name in candidate_names:
                    dense_cache_dir = os.path.join(output_dir, "score_cache", name, "dense_embed_sim")
                    dense_scores_path = os.path.join(dense_cache_dir, "scores.pt")
                    dense_meta_path = os.path.join(dense_cache_dir, "meta.json")
                    if cache_only:
                        if not os.path.exists(dense_scores_path):
                            raise FileNotFoundError(
                                f"dense_embed_sim score cache not found: {dense_scores_path}. "
                                "Re-run without --use_cache_only to build caches."
                            )
                        score = torch.load(dense_scores_path, map_location="cpu").float().view(-1)
                        expected_len = None
                        if os.path.exists(dense_meta_path):
                            try:
                                with open(dense_meta_path, "r", encoding="utf-8") as f:
                                    cached_meta = json.load(f)
                                expected_len = cached_meta.get("dataset_len")
                            except Exception:
                                expected_len = None
                        if expected_len is not None and len(score) != int(expected_len):
                            raise RuntimeError(
                                f"dense_embed_sim cached score length {len(score)} != dataset length {expected_len} "
                                f"for source {name}"
                            )
                    else:
                        loader = candidate_loaders.get(name)
                        if loader is None:
                            raise RuntimeError(
                                f"dense_embed_sim loader missing for source '{name}'. "
                                "This should not happen in non-cache mode."
                            )
                        score = _compute_dense_scores_for_dataset(
                            model=model,
                            dataloader=loader,
                            device=device,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            target_centroid=target_centroid,
                            desc=f"dense score {name}",
                        )
                        os.makedirs(dense_cache_dir, exist_ok=True)
                        torch.save(score, dense_scores_path)
                        _atomic_json_dump(
                            {
                                "source": name,
                                "dataset_len": len(loader.dataset),
                                "batch_size": loader.batch_size,
                                "seed": args.seed,
                                "candidate_percentage": args.candidate_percentage,
                                "max_seq_length": args.max_seq_length,
                                "chat_format": args.chat_format,
                                "use_chat_format": bool(use_chat_format),
                                "score_metric": score_metric,
                            },
                            dense_meta_path,
                            indent=2,
                        )
                        print(
                            f"[loss_gap] dense_embed_sim cached candidate scores at {dense_scores_path}",
                            flush=True,
                        )
                    if not args.skip_score_write:
                        score_path = os.path.join(output_dir, f"{name}_influence_score.pt")
                        torch.save(score, score_path)
                        print(
                            f"[loss_gap] saved dense_embed_sim scores to {score_path} (n={len(score)})",
                            flush=True,
                        )

            if probe_requested:
                probe_out = {
                    "seed": probe_seed,
                    "score_metric": score_metric,
                    "target_task": target_task,
                }
                for pname, loader in probe_loaders.items():
                    probe_score = _compute_dense_scores_for_dataset(
                        model=model,
                        dataloader=loader,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        target_centroid=target_centroid,
                        desc=f"dense probe {pname}",
                    )
                    if len(probe_score) == 0:
                        stats = {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "scores": []}
                    else:
                        stats = {
                            "n": len(probe_score),
                            "mean": float(probe_score.mean().item()),
                            "median": float(probe_score.median().item()),
                            "min": float(probe_score.min().item()),
                            "max": float(probe_score.max().item()),
                            "scores": probe_score.tolist(),
                        }
                    probe_out[pname] = stats
                    print(
                        f"[loss_gap] probe {pname}: n={stats['n']} mean={stats['mean']:.6f} "
                        f"median={stats['median']:.6f} min={stats['min']:.6f} max={stats['max']:.6f}",
                        flush=True,
                    )
                probe_path = os.path.join(output_dir, "probe_scores.json")
                _atomic_json_dump(probe_out, probe_path, indent=2)
                print(f"[loss_gap] wrote probe scores to {probe_path}", flush=True)

            if model is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            continue

        # ckpt_entries: (step_id, path, kind)
        # kind: "state" for step_*.pt files, "hf_dir" for checkpoint-* directories,
        #       "cache_only" when synthesized from --score_ckpt_ids in cache-only mode.
        ckpt_entries = []
        if cache_only and args.score_ckpt_ids is not None and (
            not os.path.isdir(ckpt_dir)
            or not any(
                f.startswith("step_") or f.startswith("checkpoint-")
                for f in os.listdir(ckpt_dir)
            )
        ):
            # No checkpoint files needed — synthesize entries from requested IDs.
            for sid in args.score_ckpt_ids:
                ckpt_entries.append((sid, "", "cache_only"))
            print(
                f"[loss_gap] cache_only: using score_ckpt_ids {args.score_ckpt_ids} "
                f"(no checkpoint files required)",
                flush=True,
            )
        else:
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(
                    f"No checkpoint directory found at {ckpt_dir}. Run with --mode warmup first."
                )
            for fname in os.listdir(ckpt_dir):
                if fname.startswith("step_") and fname.endswith(".pt"):
                    try:
                        step_id = int(fname.split("_")[1].split(".")[0])
                    except Exception:
                        continue
                    ckpt_entries.append((step_id, os.path.join(ckpt_dir, fname), "state"))
            if not ckpt_entries:
                hf_ckpts = []
                for fname in os.listdir(ckpt_dir):
                    if not fname.startswith("checkpoint-"):
                        continue
                    p = os.path.join(ckpt_dir, fname)
                    if not os.path.isdir(p):
                        continue
                    try:
                        raw_id = int(fname.split("checkpoint-")[1])
                    except Exception:
                        continue
                    hf_ckpts.append((raw_id, p))
                hf_ckpts.sort(key=lambda x: x[0])
                for i, (raw_id, p) in enumerate(hf_ckpts, start=1):
                    ckpt_entries.append((i, p, "hf_dir"))
                if hf_ckpts:
                    print(
                        "[loss_gap] using HF checkpoint directories mapped to sequential steps: "
                        + ", ".join([f"{i}:{os.path.basename(p)}" for i, (_, p) in enumerate(hf_ckpts, start=1)]),
                        flush=True,
                    )
            if not ckpt_entries:
                raise FileNotFoundError(
                    f"No warmup checkpoints found in {ckpt_dir}. "
                    "Expected either step_*.pt files or checkpoint-* directories."
                )
        ckpt_entries.sort(key=lambda x: x[0])
        if args.score_ckpt_ids is not None:
            desired = set(args.score_ckpt_ids)
            found = {item[0] for item in ckpt_entries}
            missing_ids = sorted(desired - found)
            if missing_ids:
                raise FileNotFoundError(
                    f"Requested checkpoints {missing_ids} not found in {ckpt_dir} "
                    f"(available: {sorted(found)}). For step 0, run warmup with --save_warmup_initial_state."
                )
            ckpt_entries = [item for item in ckpt_entries if item[0] in desired]
            if args.score_normalize_by_first and ckpt_entries and ckpt_entries[0][0] != 1:
                print(
                    f"[loss_gap] note: score_normalize_by_first will use checkpoint "
                    f"{ckpt_entries[0][0]} as denominator (not epoch 1).",
                    flush=True,
                )

        score_metric = args.score_metric
        val_loss_by_step_selected: List[float] = []
        val_loss_by_step_map: Dict[int, float] = {}
        mean_val_loss: Optional[float] = None
        if score_metric == "cov_with_val":
            if not warmup_meta or "val_loss_by_step" not in warmup_meta:
                raise ValueError(
                    "score_metric=cov_with_val requires warmup meta with val_loss_by_step. "
                    "Re-run warmup or ensure meta.json includes validation losses."
                )
            all_val_losses = warmup_meta.get("val_loss_by_step", [])
            for step_id, _, _ in ckpt_entries:
                idx = step_id - 1
                if idx < 0 or idx >= len(all_val_losses):
                    raise ValueError(
                        f"val_loss_by_step missing step {step_id}. "
                        f"Have {len(all_val_losses)} entries in meta."
                    )
                val = float(all_val_losses[idx])
                val_loss_by_step_selected.append(val)
                val_loss_by_step_map[step_id] = val
            if not val_loss_by_step_selected:
                raise ValueError("No checkpoints selected for cov_with_val scoring.")
            mean_val_loss = float(sum(val_loss_by_step_selected) / len(val_loss_by_step_selected))

        candidate_names = list(args.train_file_names)
        candidate_datasets: Dict[str, torch.utils.data.Dataset] = {}
        candidate_loaders: Dict[str, DataLoader] = {}
        if not cache_only and not args.probe_only:
            for name, path in zip(args.train_file_names, args.train_files):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"train_file not found: {path}")
                ds = _prepare_candidate_dataset(
                    path,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    percentage=args.candidate_percentage,
                    seed=args.seed,
                    chat_format=args.chat_format,
                )
                candidate_datasets[name] = ds
            for name, ds in candidate_datasets.items():
                candidate_loaders[name] = _make_candidate_loader(
                    ds,
                    batch_size=args.loss_batch_size,
                    collate_fn=data_collator,
                    length_bucketed=args.length_bucket_candidates,
                )
        else:
            for name, path in zip(args.train_file_names, args.train_files):
                if not os.path.exists(path):
                    print(f"[loss_gap] warning: train_file not found (ignored in cache-only): {path}", flush=True)

        stats_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
        cov_stats_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
        var_stats_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
        traj_stats_by_name: Dict[str, Dict[str, Any]] = {}  # for final_drop, min_loss, avg_loss, max_drop
        norm_stats_by_name: Dict[str, Dict[str, torch.Tensor]] = {} if args.score_normalize_by_first else {}

        model = None
        has_state_ckpts = any(kind == "state" for _, _, kind in ckpt_entries)
        if (not cache_only or probe_requested) and has_state_ckpts:
            model = build_model()

        probe_loaders: Dict[str, DataLoader] = {}
        probe_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        probe_cov_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        probe_var_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        probe_traj_stats: Dict[str, Dict[str, Any]] = {}
        probe_norm_stats: Dict[str, Dict[str, torch.Tensor]] = {} if args.score_normalize_by_first else {}
        probe_raw_trajectories: Dict[str, List[torch.Tensor]] = {}
        probe_scored_trajectories: Dict[str, List[torch.Tensor]] = {}
        probe_meta: Dict[str, Dict[str, int]] = {}
        if probe_requested:
            if args.probe_mmlu_test_n and args.probe_mmlu_test_n > 0:
                mmlu_ds = _prepare_mmlu_split_dataset(
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    num_samples=args.probe_mmlu_test_n,
                    seed=probe_seed,
                    split="test",
                    use_chat_format=use_chat_format,
                    chat_format=args.chat_format,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                probe_loaders["mmlu_test"] = DataLoader(
                    mmlu_ds,
                    batch_size=args.loss_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                probe_meta["mmlu_test"] = {"n": len(mmlu_ds)}
            if args.probe_mmlu_valid_n and args.probe_mmlu_valid_n > 0:
                mmlu_ds = _prepare_mmlu_split_dataset(
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    num_samples=args.probe_mmlu_valid_n,
                    seed=probe_seed,
                    split="dev",
                    use_chat_format=use_chat_format,
                    chat_format=args.chat_format,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                probe_loaders["mmlu_valid"] = DataLoader(
                    mmlu_ds,
                    batch_size=args.loss_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                probe_meta["mmlu_valid"] = {"n": len(mmlu_ds)}
            if args.probe_target_valid_n and args.probe_target_valid_n > 0:
                val_ds = get_dataset(
                    target_task,
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    use_chat_format=use_chat_format,
                    chat_format=args.chat_format,
                    mmlu_n_shot=args.mmlu_n_shot,
                )
                if args.target_valid_split:
                    target_valid_split = _prepare_target_valid_split(
                        target_task=target_task,
                        data_dir=data_dir,
                        dataset_len=len(val_ds),
                        ckpt_dir=ckpt_dir,
                        mode=args.mode,
                        split_strategy=args.target_valid_split_strategy,
                        warmup_ratio=args.target_valid_warmup_ratio,
                        split_seed=target_valid_split_seed,
                        split_file_arg=args.target_valid_split_file,
                        mmlu_n_shot=args.mmlu_n_shot,
                    )
                    val_ds = val_ds.select(target_valid_split["probe_indices"])
                    print(
                        f"[loss_gap] target_valid split probe subset: "
                        f"warmup_n={target_valid_split['warmup_n']} probe_n={target_valid_split['probe_n']} "
                        f"path={target_valid_split['split_path']}",
                        flush=True,
                    )
                if args.probe_target_valid_n < len(val_ds):
                    val_ds = _sample_hf_dataset(val_ds, args.probe_target_valid_n, probe_seed)
                probe_loaders["target_valid"] = DataLoader(
                    val_ds,
                    batch_size=args.loss_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                probe_meta["target_valid"] = {"n": len(val_ds)}
            if args.probe_cot_n and args.probe_cot_n > 0:
                cot_path = args.probe_cot_file
                if cot_path is None:
                    for name, path in zip(args.train_file_names, args.train_files):
                        if name.lower() == "cot":
                            cot_path = path
                            break
                if cot_path is None:
                    raise ValueError("--probe_cot_n set but no COT file provided and no 'cot' in train_file_names")
                cot_ds = _prepare_candidate_dataset_count(
                    cot_path,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    num_samples=args.probe_cot_n,
                    seed=probe_seed,
                    chat_format=args.chat_format,
                )
                probe_loaders["cot_sample"] = DataLoader(
                    cot_ds,
                    batch_size=args.loss_batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                probe_meta["cot_sample"] = {"n": len(cot_ds)}
        for step_id, ckpt_path, ckpt_kind in ckpt_entries:
            if (not cache_only) or probe_requested:
                if ckpt_kind == "state":
                    if model is None:
                        model = build_model()
                    state = torch.load(ckpt_path, map_location="cpu")
                    allow_partial_state = (
                        str(warmup_meta.get("warmup_checkpoint_format", "")).strip()
                        == "trainable_state_dict"
                    )
                    strict_ok = False
                    try:
                        model.load_state_dict(state, strict=True)
                        strict_ok = True
                    except RuntimeError as e:
                        if any(k.startswith("module.") for k in state.keys()):
                            stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
                            try:
                                model.load_state_dict(stripped, strict=True)
                                state = stripped
                                strict_ok = True
                            except Exception:
                                pass
                        if not strict_ok:
                            # Try to collect mismatch info for a clearer error message.
                            missing, unexpected = model.load_state_dict(state, strict=False)
                            if unexpected or (missing and not allow_partial_state):
                                hint = (
                                    "Common cause: LoRA config mismatch between warmup and scoring.\n"
                                    f"Warmup meta lora_r={warmup_meta.get('lora_r')}, "
                                    f"lora_alpha={warmup_meta.get('lora_alpha')}, "
                                    f"lora_target_modules={warmup_meta.get('lora_target_modules')}, "
                                    f"lora_rank_pattern={warmup_meta.get('lora_rank_pattern')}, "
                                    f"lora_alpha_pattern={warmup_meta.get('lora_alpha_pattern')}.\n"
                                    f"Scoring args lora_r={args.lora_r}, lora_alpha={args.lora_alpha}, "
                                    f"lora_target_modules={args.lora_target_modules}, "
                                    f"lora_rank_pattern={args.lora_rank_pattern}, "
                                    f"lora_alpha_pattern={args.lora_alpha_pattern}."
                                )
                                raise RuntimeError(
                                    f"Failed to load checkpoint state_dict for {ckpt_path}.\n"
                                    f"Missing keys (sample): {missing[:5]}\n"
                                    f"Unexpected keys (sample): {unexpected[:5]}\n"
                                    + hint
                                ) from e
                            print(
                                f"[loss_gap] partial state_dict load for {ckpt_path}: "
                                f"loaded trainable-only checkpoint "
                                f"(missing={len(missing)}, unexpected={len(unexpected)}).",
                                flush=True,
                            )
                    model.to(device)
                else:
                    # For HF checkpoint-* directory mode, load the checkpoint per step.
                    if model is not None:
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    model = load_model_from_checkpoint_dir(ckpt_path)
            print(f"[loss_gap] scoring checkpoint step={step_id} {ckpt_path}", flush=True)
            if not args.probe_only:
                for name in candidate_names:
                    loader = candidate_loaders.get(name)
                    cache_dir = os.path.join(score_cache_root, name, f"step_{step_id:04d}")
                    meta = {
                        "dataset_len": len(loader.dataset) if loader is not None else None,
                        "batch_size": loader.batch_size if loader is not None else None,
                        "seed": args.seed,
                        "candidate_percentage": args.candidate_percentage,
                        "max_seq_length": args.max_seq_length,
                    }
                    if cache_only:
                        expected_len = meta["dataset_len"]
                        # Try to pull length from cache meta if dataset wasn't loaded.
                        if expected_len is None:
                            meta_path = os.path.join(cache_dir, "meta.json")
                            if os.path.exists(meta_path):
                                try:
                                    with open(meta_path, "r", encoding="utf-8") as f:
                                        cached_meta = json.load(f)
                                    expected_len = cached_meta.get("dataset_len")
                                except Exception:
                                    expected_len = None
                        loss_vec = _load_cached_losses(cache_dir, expected_len=expected_len)
                    else:
                        batch_range = _parse_score_shard(args, len(loader)) if args.score_shard else None
                        loss_vec = _compute_losses_for_dataset(
                            model,
                            loader,
                            device,
                            use_amp,
                            amp_dtype,
                            cache_dir=cache_dir,
                            step_id=step_id,
                            meta=meta,
                            save_interval=args.loss_save_interval,
                            batch_range=batch_range,
                        )
                    if args.save_endpoint_losses:
                        if cache_only or args.score_shard:
                            raise ValueError("--save_endpoint_losses requires full non-cache scoring")
                        endpoint_dir = os.path.join(output_dir, "endpoint_losses", name)
                        endpoint_path = os.path.join(endpoint_dir, f"step_{step_id:04d}.pt")
                        os.makedirs(endpoint_dir, exist_ok=True)
                        raw_loss = loss_vec.detach().float().cpu().view(-1).clone()
                        if os.path.exists(endpoint_path):
                            existing = torch.load(endpoint_path, map_location="cpu", weights_only=True)
                            if not torch.equal(existing, raw_loss):
                                raise RuntimeError(f"Endpoint loss mismatch: {endpoint_path}")
                        else:
                            _atomic_torch_save(raw_loss, endpoint_path)
                        print(f"[loss_gap] saved endpoint losses {endpoint_path} n={len(raw_loss)}", flush=True)
                    if args.score_normalize_by_first:
                        norm_stats = norm_stats_by_name.setdefault(name, {})
                        loss_vec = _normalize_loss_by_first(
                            loss_vec,
                            norm_stats,
                            step_id=step_id,
                            eps=args.score_normalize_eps,
                        )
                    if score_metric == "loss_gap":
                        if name not in stats_by_name:
                            n = len(loss_vec)
                            stats_by_name[name] = {
                                "first": torch.zeros(n),
                                "last": torch.zeros(n),
                                "sum": torch.zeros(n),
                                "count": 0,
                            }
                        _update_running_stats(stats_by_name[name], loss_vec, step_id - 1)
                    elif score_metric == "cov_with_val":
                        if name not in cov_stats_by_name:
                            cov_stats_by_name[name] = _init_cov_stats(len(loss_vec))
                        cov_stats = cov_stats_by_name[name]
                        cov_stats["sum_lz"] += loss_vec
                        cov_stats["sum_lzlv"] += loss_vec * float(val_loss_by_step_map[step_id])
                        cov_stats["count"] += 1
                    elif score_metric in ("final_drop", "min_loss", "avg_loss", "max_drop"):
                        if name not in traj_stats_by_name:
                            n = len(loss_vec)
                            traj_stats_by_name[name] = {
                                "first": None, "last": None,
                                "sum": torch.zeros(n), "min": torch.full((n,), float("inf")),
                                "prev": None, "max_step_drop": torch.full((n,), float("-inf")),
                                "count": 0,
                            }
                        ts = traj_stats_by_name[name]
                        if ts["first"] is None:
                            ts["first"] = loss_vec.clone()
                        ts["last"] = loss_vec.clone()
                        ts["sum"] += loss_vec
                        ts["min"] = torch.minimum(ts["min"], loss_vec)
                        if ts["prev"] is not None:
                            step_drop = ts["prev"] - loss_vec
                            ts["max_step_drop"] = torch.maximum(ts["max_step_drop"], step_drop)
                        ts["prev"] = loss_vec.clone()
                        ts["count"] += 1
                    else:
                        if name not in var_stats_by_name:
                            var_stats_by_name[name] = _init_var_stats(len(loss_vec))
                        var_stats = var_stats_by_name[name]
                        var_stats["sum"] += loss_vec
                        var_stats["sumsq"] += loss_vec * loss_vec
                        var_stats["count"] += 1
            if probe_requested:
                for pname, loader in probe_loaders.items():
                    loss_vec = _compute_losses_for_dataset(
                        model,
                        loader,
                        device,
                        use_amp,
                        amp_dtype,
                        cache_dir=None,
                        step_id=step_id,
                        meta=None,
                        save_interval=args.loss_save_interval,
                    )
                    if args.save_probe_trajectory:
                        probe_raw_trajectories.setdefault(pname, []).append(
                            loss_vec.detach().float().cpu().clone()
                        )
                    if args.score_normalize_by_first:
                        norm_stats = probe_norm_stats.setdefault(pname, {})
                        loss_vec = _normalize_loss_by_first(
                            loss_vec,
                            norm_stats,
                            step_id=step_id,
                            eps=args.score_normalize_eps,
                        )
                    if args.save_probe_trajectory:
                        probe_scored_trajectories.setdefault(pname, []).append(
                            loss_vec.detach().float().cpu().clone()
                        )
                    if score_metric == "loss_gap":
                        if pname not in probe_stats:
                            n = len(loss_vec)
                            probe_stats[pname] = {
                                "first": torch.zeros(n),
                                "last": torch.zeros(n),
                                "sum": torch.zeros(n),
                                "count": 0,
                            }
                        _update_running_stats(probe_stats[pname], loss_vec, step_id - 1)
                    elif score_metric == "cov_with_val":
                        if pname not in probe_cov_stats:
                            probe_cov_stats[pname] = _init_cov_stats(len(loss_vec))
                        cov_stats = probe_cov_stats[pname]
                        cov_stats["sum_lz"] += loss_vec
                        cov_stats["sum_lzlv"] += loss_vec * float(val_loss_by_step_map[step_id])
                        cov_stats["count"] += 1
                    elif score_metric in ("final_drop", "min_loss", "avg_loss", "max_drop"):
                        if pname not in probe_traj_stats:
                            n = len(loss_vec)
                            probe_traj_stats[pname] = {
                                "first": None, "last": None,
                                "sum": torch.zeros(n), "min": torch.full((n,), float("inf")),
                                "prev": None, "max_step_drop": torch.full((n,), float("-inf")),
                                "count": 0,
                            }
                        ts = probe_traj_stats[pname]
                        if ts["first"] is None:
                            ts["first"] = loss_vec.clone()
                        ts["last"] = loss_vec.clone()
                        ts["sum"] += loss_vec
                        ts["min"] = torch.minimum(ts["min"], loss_vec)
                        if ts["prev"] is not None:
                            step_drop = ts["prev"] - loss_vec
                            ts["max_step_drop"] = torch.maximum(ts["max_step_drop"], step_drop)
                        ts["prev"] = loss_vec.clone()
                        ts["count"] += 1
                    else:
                        if pname not in probe_var_stats:
                            probe_var_stats[pname] = _init_var_stats(len(loss_vec))
                        var_stats = probe_var_stats[pname]
                        var_stats["sum"] += loss_vec
                        var_stats["sumsq"] += loss_vec * loss_vec
                        var_stats["count"] += 1

        if not args.skip_score_write and not args.probe_only:
            for name in candidate_names:
                if score_metric == "loss_gap":
                    stats = stats_by_name[name]
                    score = stats["sum"] - (float(stats["count"]) * stats["last"])
                elif score_metric == "cov_with_val":
                    stats = cov_stats_by_name[name]
                    count = float(stats["count"])
                    if count <= 0:
                        raise RuntimeError(f"cov_with_val has no steps for dataset {name}")
                    mean_lval = float(mean_val_loss) if mean_val_loss is not None else 0.0
                    score = (stats["sum_lzlv"] / count) - ((stats["sum_lz"] / count) * mean_lval)
                elif score_metric in ("final_drop", "min_loss", "avg_loss", "max_drop"):
                    ts = traj_stats_by_name[name]
                    if score_metric == "final_drop":
                        score = ts["first"] - ts["last"]
                    elif score_metric == "min_loss":
                        score = -ts["min"]  # negate: lower min → higher score
                    elif score_metric == "avg_loss":
                        score = -ts["sum"] / max(ts["count"], 1)  # negate: lower avg → higher score
                    elif score_metric == "max_drop":
                        score = ts["max_step_drop"]
                        score = torch.where(score.isinf(), torch.zeros_like(score), score)
                else:
                    stats = var_stats_by_name[name]
                    count = float(stats["count"])
                    if count <= 1:
                        score = torch.zeros_like(stats["sum"])
                    else:
                        mean = stats["sum"] / count
                        score = (stats["sumsq"] - (stats["sum"] * mean)) / (count - 1.0)
                score_path = os.path.join(output_dir, f"{name}_influence_score.pt")
                _atomic_torch_save(score, score_path)
                print(f"[loss_gap] saved scores to {score_path} (n={len(score)})", flush=True)
        if probe_requested:
            probe_out = {
                "ckpt_ids": [step_id for step_id, _, _ in ckpt_entries],
                "seed": probe_seed,
                "normalize_by_first": bool(args.score_normalize_by_first),
                "normalize_eps": float(args.score_normalize_eps),
                "score_metric": score_metric,
            }
            if score_metric == "cov_with_val":
                probe_out["val_loss_by_step"] = val_loss_by_step_selected
            if score_metric == "loss_gap":
                for pname, stats in probe_stats.items():
                    score = stats["sum"] - (float(stats["count"]) * stats["last"])
                    probe_out[pname] = {
                        "n": len(score),
                        "mean": float(score.mean().item()),
                        "median": float(score.median().item()),
                        "min": float(score.min().item()),
                        "max": float(score.max().item()),
                        "scores": score.tolist(),
                    }
            elif score_metric == "cov_with_val":
                for pname, cov_stats in probe_cov_stats.items():
                    count = float(cov_stats["count"])
                    if count <= 0:
                        raise RuntimeError(f"cov_with_val has no steps for probe {pname}")
                    mean_lval = float(mean_val_loss) if mean_val_loss is not None else 0.0
                    score = (cov_stats["sum_lzlv"] / count) - ((cov_stats["sum_lz"] / count) * mean_lval)
                    probe_out[pname] = {
                        "n": len(score),
                        "mean": float(score.mean().item()),
                        "median": float(score.median().item()),
                        "min": float(score.min().item()),
                        "max": float(score.max().item()),
                        "scores": score.tolist(),
                    }
            elif score_metric in ("final_drop", "min_loss", "avg_loss", "max_drop"):
                for pname, ts in probe_traj_stats.items():
                    if score_metric == "final_drop":
                        score = ts["first"] - ts["last"]
                    elif score_metric == "min_loss":
                        score = -ts["min"]
                    elif score_metric == "avg_loss":
                        score = -ts["sum"] / max(ts["count"], 1)
                    elif score_metric == "max_drop":
                        score = ts["max_step_drop"]
                        score = torch.where(score.isinf(), torch.zeros_like(score), score)
                    probe_out[pname] = {
                        "n": len(score),
                        "mean": float(score.mean().item()),
                        "median": float(score.median().item()),
                        "min": float(score.min().item()),
                        "max": float(score.max().item()),
                        "scores": score.tolist(),
                    }
            else:
                for pname, var_stats in probe_var_stats.items():
                    count = float(var_stats["count"])
                    if count <= 1:
                        score = torch.zeros_like(var_stats["sum"])
                    else:
                        mean = var_stats["sum"] / count
                        score = (var_stats["sumsq"] - (var_stats["sum"] * mean)) / (count - 1.0)
                    probe_out[pname] = {
                        "n": len(score),
                        "mean": float(score.mean().item()),
                        "median": float(score.median().item()),
                        "min": float(score.min().item()),
                        "max": float(score.max().item()),
                        "scores": score.tolist(),
                    }
            probe_path = os.path.join(output_dir, "probe_scores.json")
            _atomic_json_dump(probe_out, probe_path, indent=2)
            if args.save_probe_trajectory:
                trajectory_path = os.path.join(output_dir, "probe_trajectories.pt")
                trajectory_payload = {
                    "version": 1,
                    "target_task": target_task,
                    "ckpt_ids": [step_id for step_id, _, _ in ckpt_entries],
                    "seed": int(probe_seed),
                    "score_metric": score_metric,
                    "normalize_by_first": bool(args.score_normalize_by_first),
                    "normalize_eps": float(args.score_normalize_eps),
                    "raw_losses": {
                        name: torch.stack(rows, dim=0)
                        for name, rows in probe_raw_trajectories.items()
                    },
                    "scored_losses": {
                        name: torch.stack(rows, dim=0)
                        for name, rows in probe_scored_trajectories.items()
                    },
                    "probe_meta": probe_meta,
                }
                _atomic_torch_save(trajectory_payload, trajectory_path)
                print(
                    f"[loss_gap] wrote probe trajectories to {trajectory_path}",
                    flush=True,
                )
            if score_metric == "loss_gap":
                probe_print_names = probe_stats.keys()
            elif score_metric == "cov_with_val":
                probe_print_names = probe_cov_stats.keys()
            elif score_metric in ("final_drop", "min_loss", "avg_loss", "max_drop"):
                probe_print_names = probe_traj_stats.keys()
            else:
                probe_print_names = probe_var_stats.keys()
            for pname in probe_print_names:
                meta = probe_out[pname]
                print(
                    f"[loss_gap] probe {pname}: n={meta['n']} mean={meta['mean']:.6f} "
                    f"median={meta['median']:.6f} min={meta['min']:.6f} max={meta['max']:.6f}",
                    flush=True,
                )
            print(f"[loss_gap] wrote probe scores to {probe_path}", flush=True)
        if model is not None:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
