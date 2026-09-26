#!/usr/bin/env python3
"""Helpers for canonical retrain naming and metadata extraction."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional


def sanitize_token(raw: str, default: str = "na") -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def abbreviate_token(text: str, max_len: int = 32) -> str:
    token = sanitize_token(text)
    if len(token) <= max_len:
        return token
    suffix = short_hash(token, 10)
    keep = max(8, max_len - len(suffix) - 1)
    return f"{token[:keep]}_{suffix}"


def model_key_from_path(model_name_or_path: str) -> str:
    token = sanitize_token(model_name_or_path.replace("/", "_"))
    # Keep common Llama-3.2-3B identifier compact and stable.
    token = token.replace("meta-llama_llama-3.2-3b", "l32_3b")
    return token


def _extract_traj_rank(path_text: str) -> str:
    m = re.search(r"(?:/|_|^)rank_([0-9]+)(?:/|_|$)", path_text)
    if m:
        return m.group(1)
    m = re.search(r"(?:/|_|^)traj-r([0-9]+)(?:/|_|$)", path_text)
    return m.group(1) if m else "na"


def _extract_traj_lr(path_text: str) -> str:
    m = re.search(r"(?:/|_|^)lr_([0-9.eE+-]+)(?:/|_|$)", path_text)
    if m:
        return sanitize_token(m.group(1))
    m = re.search(r"(?:/|_|^)trajlr-([0-9.eE+-]+)(?:/|_|$)", path_text)
    if m:
        return sanitize_token(m.group(1))
    m = re.search(r"(?:/|_|^)wlr([0-9.eE+-]+)(?:/|_|$)", path_text)
    if m:
        return sanitize_token(m.group(1))
    return "na"


def parse_selected_dataset(train_file: str) -> Dict[str, str]:
    path = Path(train_file)
    text = str(path)
    selection_tag = "external"
    task = sanitize_token(path.parent.name)
    selection_amount = sanitize_token(path.stem)

    marker = "/selected_data_runs/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        parts = [p for p in rel.split("/") if p]
        if len(parts) >= 2:
            selection_tag = sanitize_token(parts[0])
            # Use file basename for amount regardless of intermediate directories.
            selection_amount = sanitize_token(path.stem)
            # Infer task from the nearest known task directory in path.
            for seg in reversed(parts[:-1]):
                seg_tok = sanitize_token(seg)
                if seg_tok in {"tydiqa", "mmlu", "bbh"}:
                    task = seg_tok
                    break

    traj_rank = _extract_traj_rank(text)
    traj_lr = _extract_traj_lr(text)
    return {
        "selection_tag": selection_tag,
        "task_from_path": task,
        "selection_amount": selection_amount,
        "traj_rank": traj_rank,
        "traj_lr": traj_lr,
    }


def build_canonical_retrain_name(
    *,
    task: str,
    train_file: str,
    base_model: str,
    learning_rate: str,
    num_train_epochs: str,
    use_lora: int,
    lora_r: str,
    lora_alpha: str,
    lora_dropout: str,
    seed: str,
    suffix: str = "",
    name_format: str = "short",
    max_len: int = 0,
) -> Dict[str, Any]:
    ds = parse_selected_dataset(train_file)
    task_token = sanitize_token(task or ds["task_from_path"])
    model_key = model_key_from_path(base_model)
    lr_token = sanitize_token(str(learning_rate))
    ep_token = sanitize_token(str(num_train_epochs))
    seed_token = sanitize_token(str(seed))
    lora_flag = "1" if int(use_lora) != 0 else "0"
    lora_r_token = sanitize_token(str(lora_r))
    lora_alpha_token = sanitize_token(str(lora_alpha))
    lora_drop_token = sanitize_token(str(lora_dropout)).replace(".", "")
    traj_rank = sanitize_token(ds["traj_rank"])
    traj_lr = sanitize_token(ds["traj_lr"])

    cfg_token = (
        f"l{lora_flag}r{lora_r_token}a{lora_alpha_token}d{lora_drop_token}"
        f"lr{lr_token}ep{ep_token}s{seed_token}"
    )
    fp_src = "|".join(
        [
            task_token,
            train_file,
            model_key,
            cfg_token,
            ds["selection_tag"],
            ds["selection_amount"],
            traj_rank,
            traj_lr,
        ]
    )
    fp = short_hash(fp_src, 8)

    if name_format == "full":
        name = (
            f"retrain__m-{model_key}__t-{task_token}"
            f"__ds-{sanitize_token(ds['selection_tag'])}"
            f"__amt-{sanitize_token(ds['selection_amount'])}"
            f"__traj-r{traj_rank}lr{traj_lr}"
            f"__cfg-{cfg_token}"
        )
    else:
        ds_short = abbreviate_token(ds["selection_tag"], 28)
        name = (
            f"rtr__t-{task_token}"
            f"__ds-{ds_short}"
            f"__amt-{sanitize_token(ds['selection_amount'])}"
            f"__traj-r{traj_rank}lr{traj_lr}"
            f"__cfg-{cfg_token}"
            f"__h{fp}"
        )

    if suffix:
        name += f"__x-{sanitize_token(suffix)}"

    if max_len > 0 and len(name) > max_len:
        over_hash = short_hash(name, 10)
        keep = max(16, max_len - len(over_hash) - 3)
        name = f"{name[:keep]}__{over_hash}"

    return {
        "name": name,
        "task": task_token,
        "model_key": model_key,
        "selection_tag": ds["selection_tag"],
        "selection_amount": ds["selection_amount"],
        "traj_rank": traj_rank,
        "traj_lr": traj_lr,
        "learning_rate": lr_token,
        "num_train_epochs": ep_token,
        "use_lora": lora_flag,
        "lora_r": lora_r_token,
        "lora_alpha": lora_alpha_token,
        "lora_dropout": lora_drop_token,
        "seed": seed_token,
        "fingerprint": fp,
    }


def parse_retrain_train_log(train_log: Path) -> Dict[str, Any]:
    text = train_log.read_text(encoding="utf-8", errors="ignore")

    def first(pattern: str, default: str = "") -> str:
        m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
        if not m:
            return default
        if m.lastindex and m.lastindex >= 1:
            return m.group(1).strip()
        return m.group(0).strip()

    data_match = re.search(
        r"DataArguments\(train_files=\[(.*?)\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    train_files = []
    if data_match:
        train_files = re.findall(r"'([^']+)'", data_match.group(1))

    run_task = first(r"target_task_names=\[?('?)([a-zA-Z0-9_-]+)\1\]?", "")

    meta = {
        "model_name_or_path": first(r"model_name_or_path='([^']+)'", "na"),
        "learning_rate": first(r"learning_rate=([0-9.eE+-]+)", "na"),
        "num_train_epochs": first(r"num_train_epochs=([0-9.eE+-]+)", "na"),
        "seed": first(r"seed=([0-9]+)", "3"),
        "lora": first(r"lora=(True|False)", "True"),
        "lora_r": first(r"lora_r=([0-9]+)", "16"),
        "lora_alpha": first(r"lora_alpha=([0-9.eE+-]+)", "512"),
        "lora_dropout": first(r"lora_dropout=([0-9.eE+-]+)", "0.1"),
        "train_files": train_files,
        "raw_task_hint": run_task,
    }
    return meta


def infer_task_from_run_dir(run_dir: Path, train_file: Optional[str]) -> str:
    text = run_dir.name.lower()
    for task in ("tydiqa", "mmlu", "bbh"):
        if task in text:
            return task
    if train_file:
        tf = train_file.lower()
        for task in ("tydiqa", "mmlu", "bbh"):
            if f"/{task}/" in tf:
                return task
    return "na"
