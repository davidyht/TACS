"""
Relaunch training from a checkpoint: estimate validation gradient norm, fine-tune on validation,
then continue training on train set for remaining epochs, evaluate final validation loss,
and report (L' - L) / ||grad||_2^2.

Minimal, conservative implementation: only trains LoRA params by default and reuses
existing data loaders from tacs.data_selection.* to maintain format compatibility.
"""
import argparse
import json
import os
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import sys

# utility imports from repo
from tacs.train.utils import set_seed, estimate_full_gradient_norm, estimate_full_gradient, evaluate_loss_on_dataloader
from tacs.data_selection.get_validation_dataset import get_dataset as get_val_dataset, get_dataloader as get_val_dataloader
from tacs.data_selection.get_training_dataset import get_training_dataset
from tacs.data_selection.get_info import load_model


def _validate_checkpoint_path(path: str, label: str) -> None:
    if path is None or str(path).strip() == "":
        raise ValueError(f"{label} is empty. Provide a valid checkpoint directory.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} does not exist: {path}\n"
            "If this is a local checkpoint, check the path. If you intended a HF repo id, "
            "relaunch expects a local checkpoint directory."
        )


def try_load_baseline_L(checkpoint_path: str, val_dataloader, device: torch.device) -> Optional[float]:
    # try common metrics file names
    metrics_files = ["metrics.json", "trainer_state.json", "eval_results.json"]
    for fname in metrics_files:
        p = os.path.join(checkpoint_path, fname)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    j = json.load(f)
                # try to find common keys
                for key in ("eval_loss", "validation_loss", "loss"):
                    if key in j:
                        try:
                            return float(j[key])
                        except Exception:
                            continue
            except Exception:
                continue
    return None
def _format_seconds(s: float) -> str:
    # human readable hh:mm:ss
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m:d}m{sec:02d}s"
    return f"{sec:d}s"


def _compute_avg_lr(checkpoint_path: str, epoch_end: int) -> Optional[float]:
    args_path = os.path.join(checkpoint_path, "training_args.bin")
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    try:
        train_args = torch.load(args_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"relaunch: failed to read training_args.bin ({e}); skipping avg_lr", flush=True)
        return None
    try:
        with open(state_path, "r") as f:
            trainer_state = json.load(f)
    except Exception as e:
        print(f"relaunch: failed to read trainer_state.json ({e}); skipping avg_lr", flush=True)
        return None

    max_steps = int(trainer_state.get("max_steps", 0) or 0)
    num_epochs = float(trainer_state.get("num_train_epochs", 0) or 0)
    if max_steps <= 0 or num_epochs <= 0:
        print("relaunch: trainer_state.json missing max_steps/num_train_epochs; skipping avg_lr", flush=True)
        return None

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
        return None
    return sum(lrs) / float(len(lrs))
def main():
    parser = argparse.ArgumentParser(description="Relaunch training from checkpoint with perturbation for influence estimation")

    # === Required arguments ===
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint-k (the checkpoint to perturb and continue from)")
    parser.add_argument("--train_files", type=str, default=None, help="Space-separated training data files (can be set via config)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--k", type=int, required=True, help="Checkpoint epoch index k")
    parser.add_argument("--T", type=int, required=True, help="Total training epochs T")

    # === Validation data ===
    parser.add_argument("--validation_task", type=str, default=None, help="Validation task: bbh, tydiqa, or mmlu")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to custom validation jsonl file")
    parser.add_argument("--data_dir", type=str, default="..", help="Root data directory containing eval/ subfolder")
    parser.add_argument("--val_sample_size", type=int, default=None, help="Sample this many examples from validation set")
    parser.add_argument("--val_sample_percentage", type=float, default=None, help="Sample this fraction of validation set")
    parser.add_argument("--val_sample_seed", type=int, default=42, help="Seed for validation set sampling (use same value across runs for consistency)")

    # === Perturbation settings ===
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Perturbation strength for first-step gradient replacement")
    parser.add_argument("--simple_adam_perturb", action="store_true", help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--adam_perturb_steps", type=int, default=1, help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--lr_scale", type=float, default=0.1, help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--perturb_lr", type=float, default=None, help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--use_precond", action="store_true", help="(deprecated) Ignored; kept for compatibility")

    # === Denominator computation ===
    parser.add_argument("--use_empirical_denom", action="store_true", help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--denom_lr_scale", type=float, default=None, help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--empirical_epsilon", type=float, default=1e-3, help="(deprecated) Ignored; kept for compatibility")
    parser.add_argument("--grad_est_size", type=int, default=50, help="Number of examples for gradient estimation (0 = use all)")
    parser.add_argument("--double_perturb", action="store_true", help="Use symmetric +/- epsilon perturbations to compute metric; skips baseline_L")
    parser.add_argument("--double_perturb_epsilon", type=float, default=1e-3, help="(deprecated) Ignored; kept for compatibility")

    # === Baseline ===
    parser.add_argument("--baseline_checkpoint_path", type=str, default=None, help="Path to checkpoint-T for baseline_L computation")
    parser.add_argument("--compute_baseline_by_relaunch", action="store_true", help="If enabled, compute baseline_L by running an extra continuation to epoch T without perturbation (doubles training time)")

    # === Training hyperparameters (for continuation mode) ===
    parser.add_argument("--base_lr", type=float, default=2e-5, help="(deprecated) Ignored for training; LR comes from scheduler.pt")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--continue_epochs", type=int, default=None, help="Override number of epochs to continue")
    parser.add_argument("--sample_percentage", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for tokenization (must match warmup)")
    parser.add_argument("--sample_data_seed", type=int, default=None, help="Seed for data sampling (must match warmup's sample_data_seed, default 42 if not set)")
    parser.add_argument("--data_seed", type=int, default=None, help="Seed for data shuffling (match warmup's data_seed)")
    parser.add_argument("--resume_then_perturb", action="store_true", help="Resume from checkpoint (restore trainer/optimizer/scheduler/rng), then apply perturbation before continuing")
    parser.add_argument("--no_resume_then_perturb", action="store_false", dest="resume_then_perturb", help="Disable resume_from_checkpoint behavior")
    parser.set_defaults(resume_then_perturb=True)

    # === Device and precision ===
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training (different from sample_data_seed)")
    parser.add_argument("--bf16", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False)

    # === Config file ===
    parser.add_argument("--config", type=str, default=None, help="JSON config file to override arguments")

    # === Deprecated/unused (kept for compatibility) ===
    parser.add_argument("--val_finetune_steps", type=int, default=None, help="(deprecated)")

    args = parser.parse_args()

    # If a config JSON is provided, load and override matching args to centralize settings
    if args.config is not None:
        try:
            with open(args.config, 'r') as cf:
                cfg = json.load(cf)
            # map simple keys
            for k, v in cfg.items():
                key = k.lower()
                if hasattr(args, key):
                    try:
                        # normalize some common types: lists -> space-joined strings for train_files
                        if key == 'train_files' and isinstance(v, (list, tuple)):
                            setattr(args, key, ' '.join(v))
                        else:
                            setattr(args, key, v)
                    except Exception:
                        pass
        except Exception as e:
            print(f"relaunch: failed to read config {args.config}: {e}", flush=True)

    # normalize train_files into a space-separated string if user provided list via config
    if isinstance(args.train_files, (list, tuple)):
        args.train_files = ' '.join(args.train_files)

    _validate_checkpoint_path(args.checkpoint_path, "checkpoint_path")
    if args.baseline_checkpoint_path:
        _validate_checkpoint_path(args.baseline_checkpoint_path, "baseline_checkpoint_path")
    if args.data_dir is None or str(args.data_dir).strip() == "":
        raise ValueError("data_dir is empty. Provide --data_dir pointing to the repo data root.")

    # Print concise parsed args for reproducibility + core seeds/hparams
    try:
        print(f"relaunch: checkpoint={args.checkpoint_path} output_dir={args.output_dir} k={args.k} T={args.T}", flush=True)
        print(f"relaunch: seeds training_seed={args.seed} sample_data_seed={args.sample_data_seed if args.sample_data_seed is not None else 42} val_sample_seed={args.val_sample_seed}", flush=True)
        print(f"relaunch: hparams epsilon={args.epsilon} warmup_ratio={args.warmup_ratio} sched={args.lr_scheduler_type} weight_decay={args.weight_decay} optim={args.optim}", flush=True)
        print(f"relaunch: precision bf16={args.bf16} fp16={args.fp16}", flush=True)
        print(f"relaunch: data max_seq_length={args.max_seq_length} sample_percentage={args.sample_percentage}", flush=True)
    except Exception:
        pass

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # seed
    set_seed(args.seed)

    # load model and tokenizer using existing helper to handle peft/lora
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # tokenizer diagnostics
    try:
        print(f"relaunch: tokenizer model_max_length={getattr(tokenizer, 'model_max_length', None)} padding_side={getattr(tokenizer, 'padding_side', None)} truncation_side={getattr(tokenizer, 'truncation_side', None)}", flush=True)
    except Exception:
        pass

    model = load_model(args.checkpoint_path)
    # model/params diagnostics
    try:
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"relaunch: trainable_params_count={len(trainable)} sample_trainable_names={trainable[:5]}", flush=True)
    except Exception:
        pass
    # ensure embeddings resized if tokenizer bigger
    try:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # prepare validation dataloader
    import glob
    if args.validation_task is not None:
        # If user asked for a larger validation sample for tydiqa, try to locate the full dev files and sample from them
        if args.validation_task == "tydiqa" and (args.val_sample_size is not None or args.val_sample_percentage is not None):
            # search for files under data_dir/.. that look like full dev datasets (prefer 'dev' subdir)
            # look in dev and test subfolders first, then any tydiqa folder
            cand_patterns = [
                os.path.join(args.data_dir, "**", "tydiqa", "test", "*.json*"),
                os.path.join(args.data_dir, "**", "tydiqa", "dev", "*.json*"),
                os.path.join(args.data_dir, "**", "tydiqa", "*.json*"),
                os.path.join(args.data_dir, "**", "tydiqa", "test", "*.jsonl*"),
                os.path.join(args.data_dir, "**", "tydiqa", "dev", "*.jsonl*"),
                os.path.join(args.data_dir, "**", "tydiqa", "*.jsonl*")
            ]
            candidates = []
            for p in cand_patterns:
                candidates += glob.glob(p, recursive=True)
            # filter out one-shot files
            candidates = [c for c in candidates if "one-shot" not in os.path.basename(c)]
            # resolve to absolute paths and print diagnostics so user can see which files were found
            candidates = [os.path.abspath(c) for c in candidates]
            if len(candidates) > 0:
                print(f"relaunch: found {len(candidates)} candidate tydiqa files under {args.data_dir}", flush=True)
            else:
                print("relaunch: no candidate tydiqa files found under data_dir", flush=True)
            if len(candidates) == 0:
                # fallback to default get_val_dataset behavior
                val_ds = get_val_dataset(args.validation_task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_seq_length)
                val_loader = get_val_dataloader(val_ds, tokenizer, batch_size=1)
            else:
                # prefer the candidate with the largest number of examples (try to load quickly to count)
                best = None
                best_count = -1
                for c in candidates:
                    try:
                        if c.endswith('.jsonl') or c.endswith('.jsonl.gz'):
                            from datasets import load_dataset
                            tmp = load_dataset('json', data_files=c)['train']
                            cnt = len(tmp)
                        else:
                            with open(c, 'r') as f:
                                jj = json.load(f)
                            if isinstance(jj, dict) and all(isinstance(v, list) for v in jj.values()):
                                cnt = sum(len(v) for v in jj.values())
                            elif isinstance(jj, list):
                                cnt = len(jj)
                            else:
                                cnt = 0
                    except Exception:
                        cnt = 0
                    if cnt > best_count:
                        best_count = cnt
                        best = c
                vf_full = best if best is not None else candidates[0]
                print(f"relaunch: using tydiqa full file {vf_full} for sampling validation set (estimated examples={best_count})")
                # load raw examples
                raw_examples = []
                try:
                    # try jsonl first
                    if vf_full.endswith('.jsonl') or vf_full.endswith('.jsonl.gz'):
                        from datasets import load_dataset
                        ds_tmp = load_dataset('json', data_files=vf_full)['train']
                        raw_examples = [dict(x) for x in ds_tmp]
                    else:
                        with open(vf_full, 'r') as f:
                            j = json.load(f)
                        # j may be dict(lang -> list) or list of examples
                        if isinstance(j, dict) and all(isinstance(v, list) for v in j.values()):
                            # flatten across languages (existing processed tydiqa files)
                            for lang in j:
                                for ex in j[lang]:
                                    ex['_lang'] = lang
                                    raw_examples.append(ex)
                        elif isinstance(j, list):
                            # already a flat list of examples
                            raw_examples = j
                        elif isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
                            # SQuAD / original tydiqa gold-p format: {'data': [articles...]}
                            # Flatten into our expected example dicts with keys: context, question, answers
                            for article in j.get("data", []):
                                for para in article.get("paragraphs", []):
                                    context = para.get("context", "")
                                    for qa in para.get("qas", []):
                                        ex = {
                                            "context": context,
                                            "question": qa.get("question", ""),
                                            "answers": qa.get("answers", [])
                                        }
                                        raw_examples.append(ex)
                        else:
                            # unknown structure; fallback to one-shot behavior
                            val_ds = get_val_dataset(args.validation_task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_seq_length)
                            val_loader = get_val_dataloader(val_ds, tokenizer, batch_size=1)
                            raw_examples = []
                except Exception as e:
                    print(f"relaunch: failed to load full tydiqa file {vf_full}: {e}")
                    val_ds = get_val_dataset(args.validation_task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_seq_length)
                    val_loader = get_val_dataloader(val_ds, tokenizer, batch_size=1)
                    raw_examples = []

                if len(raw_examples) > 0:
                    # decide sample size
                    total = len(raw_examples)
                    print(f"relaunch: found {total} total examples in full tydiqa file; sampling {('size='+str(args.val_sample_size)) if args.val_sample_size else ('pct='+str(args.val_sample_percentage))}")
                    if args.val_sample_size is not None:
                        requested = args.val_sample_size
                    elif args.val_sample_percentage is not None:
                        requested = max(1, int(total * args.val_sample_percentage))
                    else:
                        requested = min(3000, total)
                    if requested > total:
                        print(f"relaunch: requested {requested} samples but only {total} available; using all {total} examples")
                        want = total
                    else:
                        want = requested
                    import random as _random
                    # Use val_sample_seed for validation sampling (NOT args.seed which is for training)
                    # This ensures different checkpoints evaluated with same val_sample_seed get the SAME validation subset
                    _random.seed(args.val_sample_seed)
                    sampled = _random.sample(raw_examples, want)
                    print(f"relaunch: sampled {want} examples using val_sample_seed={args.val_sample_seed}")

                    # tokenize sampled examples using tydiqa template
                    from tacs.data_selection.get_validation_dataset import tokenize as val_tokenize
                    from datasets import Dataset as HFDataset
                    dataset_dict = {"input_ids": [], "attention_mask": [], "labels": []}
                    # try to preserve language field if present
                    for ex in sampled:
                        lang = ex.get('_lang', 'english')
                        encoding_templates_with_context = {
                            "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
                        }
                        prompt, p_template, q_template, a_template = encoding_templates_with_context.get(lang, encoding_templates_with_context["english"])
                        prompt += p_template + " " + format(ex.get("context", "")) + "\n" + q_template + " " + format(ex.get("question", "")) + "\n"
                        answer = " " + format(ex.get("answers", [{"text": ""}])[0].get("text", ""))
                        full_input_ids, labels, attention_mask = val_tokenize(tokenizer, prompt, answer, args.max_seq_length)
                        dataset_dict["input_ids"].append(full_input_ids)
                        dataset_dict["labels"].append(labels)
                        dataset_dict["attention_mask"].append(attention_mask)
                    hf_ds = HFDataset.from_dict(dataset_dict)
                    hf_ds.set_format(type="pt")
                    from tacs.data_selection.get_validation_dataset import get_dataloader as get_dataloader_fn
                    val_loader = get_dataloader_fn(hf_ds, tokenizer, batch_size=1)
                    print(f"relaunch: built validation dataloader with {want} examples")
        else:
            try:
                val_ds = get_val_dataset(args.validation_task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_seq_length)
                val_loader = get_val_dataloader(val_ds, tokenizer, batch_size=1)
            except FileNotFoundError:
                # Try to find commonly placed files (e.g., eval/tydiqa/dev/tydiqa-one-shot.json)
                if args.validation_task == "tydiqa":
                    # search recursively for tydiqa-one-shot files under data_dir
                    search_pattern = os.path.join(args.data_dir, "**", "tydiqa-one-shot*.json")
                    matches = glob.glob(search_pattern, recursive=True)
                    if len(matches) > 0:
                        vf = matches[0]
                        from tacs.data_selection.get_validation_dataset import get_dataloader as get_dataloader_fn
                        try:
                            with open(vf, 'r') as f:
                                raw = json.load(f)
                        except Exception as e:
                            raise RuntimeError(f"Failed to read validation file {vf}: {e}")

                        if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
                            from datasets import Dataset as HFDataset
                            from tacs.data_selection.get_validation_dataset import tokenize as val_tokenize

                            encoding_templates_with_context = {
                                "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
                                "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
                                "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
                                "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
                                "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
                                "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
                                "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
                                "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
                                "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
                            }

                            zh = False
                            dataset_dict = {"input_ids": [], "attention_mask": [], "labels": []}
                            for lang in raw:
                                if len(raw[lang]) == 0:
                                    continue
                                example = raw[lang][0]
                                prompt, p_template, q_template, a_template = encoding_templates_with_context.get(lang, encoding_templates_with_context["english"])
                                prompt_text = prompt + p_template + " " + format(example.get("context", "")) + "\n" + q_template + " " + format(example.get("question", "")) + "\n"
                                answer = " " + format(example.get("answers", [])[0].get("text", "")) if example.get("answers") else ""
                                full_input_ids, labels, attention_mask = val_tokenize(tokenizer, prompt_text if not zh else prompt_text, answer, args.max_seq_length)
                                dataset_dict["input_ids"].append(full_input_ids.tolist())
                                dataset_dict["labels"].append(labels.tolist())
                                dataset_dict["attention_mask"].append(attention_mask)

                            hf_ds = HFDataset.from_dict(dataset_dict)
                            hf_ds.set_format(type="pt")
                            val_loader = get_dataloader_fn(hf_ds, tokenizer, batch_size=1)
                            # continue
                        else:
                            vs = get_training_dataset(vf, tokenizer, args.max_seq_length, sample_percentage=1.0, seed=args.seed or 0)
                            val_loader = get_dataloader_fn(vs, tokenizer, batch_size=1)
                    else:
                        raise
                else:
                    # not tydiqa or not found — re-raise to show original error
                    raise
    elif args.validation_file is not None:
            # validation_file can be either:
            # - a training-style jsonl/json with 'prompt'/'completion' or 'messages' columns (use get_training_dataset)
            # - a task-specific json (e.g., tydiqa one-shot) which we must parse into examples
            vf = args.validation_file
            from tacs.data_selection.get_validation_dataset import get_dataloader as get_dataloader_fn
            if vf.endswith('.jsonl') or vf.endswith('.jsonl.gz'):
                vs = get_training_dataset(args.validation_file, tokenizer, args.max_seq_length, sample_percentage=1.0, seed=args.seed or 0)
                val_loader = get_dataloader_fn(vs, tokenizer, batch_size=1)
            elif vf.endswith('.json'):
                # try to detect structure
                try:
                    with open(vf, 'r') as f:
                        raw = json.load(f)
                except Exception as e:
                    raise RuntimeError(f"Failed to read validation file {vf}: {e}")

                # If raw is a dict of languages -> list (tydiqa style), build dataset via tokenize
                if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
                    from datasets import Dataset as HFDataset
                    from tacs.data_selection.get_validation_dataset import tokenize as val_tokenize

                    encoding_templates_with_context = {
                        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
                        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
                        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
                        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
                        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
                        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
                        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
                        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
                        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
                    }

                    # support zh flag if present in args (not passed via file)
                    zh = False
                    dataset_dict = {"input_ids": [], "attention_mask": [], "labels": []}
                    for lang in raw:
                        # many tydiqa files map language -> list of examples; pick first example per lang as in other code
                        if len(raw[lang]) == 0:
                            continue
                        example = raw[lang][0]
                        prompt, p_template, q_template, a_template = encoding_templates_with_context.get(lang, encoding_templates_with_context["english"])
                        prompt_text = prompt + p_template + " " + format(example.get("context", "")) + "\n" + q_template + " " + format(example.get("question", "")) + "\n"
                        answer = " " + format(example.get("answers", [])[0].get("text", "")) if example.get("answers") else ""
                        # use chat format consistent with other helpers
                        full_input_ids, labels, attention_mask = val_tokenize(tokenizer, prompt_text if not zh else prompt_text, answer, args.max_seq_length)
                        dataset_dict["input_ids"].append(full_input_ids.tolist())
                        dataset_dict["labels"].append(labels.tolist())
                        dataset_dict["attention_mask"].append(attention_mask)

                    # convert lists of tensors to flat lists
                    # note: tokenize returns tensors; ensure correct shapes
                    # create HF dataset
                    hf_ds = HFDataset.from_dict(dataset_dict)
                    hf_ds.set_format(type="pt")
                    val_loader = get_dataloader_fn(hf_ds, tokenizer, batch_size=1)
                else:
                    # fallback: try to treat as training-style json
                    vs = get_training_dataset(args.validation_file, tokenizer, args.max_seq_length, sample_percentage=1.0, seed=args.seed or 0)
                    val_loader = get_dataloader_fn(vs, tokenizer, batch_size=1)
            else:
                raise ValueError("Unsupported validation_file extension; expected .jsonl or .json")
    else:
        raise ValueError("Provide either --validation_task or --validation_file")

    # baseline L (skip when using double perturbation)
    baseline_L = None
    if not args.double_perturb:
        # If user provided an explicit baseline checkpoint, load that model and compute baseline by evaluating on the validation dataloader
        if args.baseline_checkpoint_path is not None:
            try:
                print(f"relaunch: computing baseline_L by loading baseline checkpoint {args.baseline_checkpoint_path}", flush=True)
                baseline_model = load_model(args.baseline_checkpoint_path)
                baseline_model.to(device)
                baseline_L = evaluate_loss_on_dataloader(baseline_model, val_loader, device)
                # free baseline model from GPU
                try:
                    del baseline_model
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"relaunch: baseline_L (from baseline_checkpoint) = {baseline_L}", flush=True)
            except Exception as e:
                print(f"relaunch: failed to compute baseline from baseline_checkpoint_path: {e}; falling back to checkpoint metrics or recompute", flush=True)
                baseline_L = None

        if baseline_L is None:
            # Option A: compute by extra relaunch to T without perturbation
            if args.compute_baseline_by_relaunch:
                print("relaunch: compute_baseline_by_relaunch=True — running extra continuation without perturbation to epoch T", flush=True)
                # Clone model weights to avoid contaminating the main run
                model_baseline = load_model(args.checkpoint_path)
                model_baseline.to(device)
                # Build train dataset with same settings
                if args.train_files is None:
                    raise ValueError("--train_files is required when compute_baseline_by_relaunch=True")
                train_files = args.train_files.split()
                sample_pct = args.sample_percentage
                data_seed = args.sample_data_seed if args.sample_data_seed is not None else 42
                max_seq_len = args.max_seq_length
                print(f"relaunch: [baseline] loading training dataset with sample_data_seed={data_seed}, max_seq_length={max_seq_len}, sample_percentage={sample_pct}", flush=True)
                train_dataset_bl = get_training_dataset(train_files, tokenizer, max_seq_length=max_seq_len, sample_percentage=sample_pct, seed=data_seed)
                try:
                    if "dataset" in train_dataset_bl.features:
                        train_dataset_bl = train_dataset_bl.remove_columns(["dataset", "id", "messages"])
                except Exception:
                    pass

                from transformers import TrainingArguments, Trainer
                from transformers import DataCollatorForSeq2Seq

                steps_per_epoch_approx = max(1, len(train_dataset_bl) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
                total_steps_for_T_epochs = steps_per_epoch_approx * args.T
                # Infer checkpoint epoch k (same logic as main flow)
                effective_k_bl = args.k
                trainer_state_path_bl = os.path.join(args.checkpoint_path, "trainer_state.json")
                try:
                    if os.path.exists(trainer_state_path_bl):
                        with open(trainer_state_path_bl, "r") as f:
                            ts = json.load(f)
                        if "epoch" in ts and isinstance(ts["epoch"], (int, float)):
                            effective_k_bl = int(round(ts["epoch"]))
                        elif "global_step" in ts and isinstance(ts["global_step"], (int, float)):
                            global_step = int(ts["global_step"])
                            effective_k_bl = int(global_step // steps_per_epoch_approx)
                except Exception:
                    effective_k_bl = args.k

                start_step_bl = effective_k_bl * steps_per_epoch_approx
                remaining_steps_bl = max(0, total_steps_for_T_epochs - start_step_bl)
                print(f"relaunch: [baseline] steps_per_epoch={steps_per_epoch_approx}, start_step={start_step_bl}, remaining_steps={remaining_steps_bl}")
                # Baseline run: continue from k to T (no perturbation), same length as main run
                training_args_bl = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, "baseline_run"),
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_steps=remaining_steps_bl,
                    learning_rate=0.0,
                    logging_strategy="no",
                    save_strategy="no",
                    do_eval=False,
                    lr_scheduler_type=args.lr_scheduler_type,
                    warmup_ratio=0,
                    weight_decay=args.weight_decay,
                    optim=args.optim,
                    bf16=args.bf16,
                    fp16=args.fp16,
                    seed=args.seed or 0,
                    data_seed=args.seed or 0,
                    ignore_data_skip=True,
                    dataloader_num_workers=0,
                )
                data_collator_bl = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_baseline, padding="longest")
                trainer_bl = Trainer(
                    model=model_baseline,
                    args=training_args_bl,
                    train_dataset=train_dataset_bl,
                    tokenizer=tokenizer,
                    data_collator=data_collator_bl,
                )
                # Restore optimizer/scheduler states to truly continue from k
                ckpt_opt_path_bl = os.path.join(args.checkpoint_path, "optimizer.pt")
                ckpt_sched_path_bl = os.path.join(args.checkpoint_path, "scheduler.pt")
                trainer_bl.create_optimizer_and_scheduler(num_training_steps=total_steps_for_T_epochs)
                if os.path.exists(ckpt_opt_path_bl):
                    try:
                        opt_state = torch.load(ckpt_opt_path_bl, map_location="cpu", weights_only=False)
                        if "state" in opt_state and opt_state["state"]:
                            first_key = next(iter(opt_state["state"].keys()))
                            if not (isinstance(first_key, int) or (isinstance(first_key, str) and str(first_key).isdigit())):
                                # convert name-keyed to index-keyed
                                name2idx = {name: idx for idx, (name, _) in enumerate(model_baseline.named_parameters())}
                                new_state = {}
                                for name, val in opt_state["state"].items():
                                    if name in name2idx:
                                        new_state[name2idx[name]] = val
                                    else:
                                        for full_name, idx in name2idx.items():
                                            if name in full_name or full_name.endswith(name):
                                                new_state[idx] = val
                                                break
                                opt_state["state"] = new_state
                        trainer_bl.optimizer.load_state_dict(opt_state)
                        print("relaunch: [baseline] optimizer state restored")
                    except Exception as e:
                        print(f"relaunch: [baseline] failed to load optimizer state: {e}")
                if os.path.exists(ckpt_sched_path_bl):
                    try:
                        sched_state = torch.load(ckpt_sched_path_bl, map_location="cpu", weights_only=False)
                        trainer_bl.lr_scheduler.load_state_dict(sched_state)
                        cur_lr_bl = trainer_bl.lr_scheduler.get_last_lr()[0]
                        print(f"relaunch: [baseline] scheduler restored. Current LR={cur_lr_bl:.2e}")
                    except Exception as e:
                        print(f"relaunch: [baseline] failed to load scheduler state: {e}")
                start_bl = time.time()
                print("relaunch: [baseline] starting extra continuation from k→T (no perturbation)", flush=True)
                trainer_bl.train()
                dur_bl = time.time() - start_bl
                try:
                    trainer_bl.save_state()
                except Exception:
                    pass
                model_baseline = trainer_bl.model
                model_baseline.eval()
                baseline_L = evaluate_loss_on_dataloader(model_baseline, val_loader, device)
                print(f"relaunch: [baseline] completed. dur={_format_seconds(dur_bl)} baseline_L={baseline_L}", flush=True)
                try:
                    del trainer_bl
                    del model_baseline
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                # Option B: load from checkpoint files or compute at checkpoint-k
                baseline_L = try_load_baseline_L(args.checkpoint_path, val_loader, device)
                if baseline_L is None:
                    model.to(device)
                    baseline_L = evaluate_loss_on_dataloader(model, val_loader, device)
                else:
                    model.to(device)

    # =====================================================================
    # Helper function for Adam-based perturbation
    # =====================================================================
    def run_adam_steps_on_validation(model_to_perturb, val_loader, optimizer_state_path, lr, num_steps, device, verbose=True):
        """
        Run a few Adam steps on the validation set, preserving optimizer momentum.
        Returns the model after perturbation.
        """
        from torch.optim import AdamW
        model_to_perturb.train()
        model_to_perturb.to(device)

        # Create optimizer for trainable params
        trainable_params = [(n, p) for n, p in model_to_perturb.named_parameters() if p.requires_grad]
        params = [p for n, p in trainable_params]
        optimizer = AdamW(params, lr=lr)

        # Load optimizer state to preserve Adam's m and v
        if os.path.exists(optimizer_state_path):
            if verbose:
                print(f"relaunch: loading optimizer state from {optimizer_state_path}", flush=True)
            opt_state = torch.load(optimizer_state_path, map_location="cpu", weights_only=False)

            # Debug: print optimizer.pt structure
            if verbose:
                print(f"relaunch: optimizer.pt keys: {list(opt_state.keys())}", flush=True)
                if "param_groups" in opt_state:
                    pg = opt_state["param_groups"]
                    print(f"relaunch: param_groups count: {len(pg)}", flush=True)
                    if len(pg) > 0:
                        print(f"relaunch: param_groups[0] keys: {list(pg[0].keys())}", flush=True)
                        print(f"relaunch: param_groups[0]['lr']: {pg[0].get('lr', 'N/A')}", flush=True)
                if "state" in opt_state:
                    state_keys = list(opt_state["state"].keys())
                    print(f"relaunch: state has {len(state_keys)} entries, first few keys: {state_keys[:5]}", flush=True)
                    if len(state_keys) > 0:
                        first_key = state_keys[0]
                        first_state = opt_state["state"][first_key]
                        print(f"relaunch: state[{first_key}] keys: {list(first_state.keys())}", flush=True)
                        if 'step' in first_state:
                            print(f"relaunch: state[{first_key}]['step']: {first_state['step']}", flush=True)
                        if 'exp_avg' in first_state:
                            exp_avg = first_state['exp_avg']
                            print(f"relaunch: state[{first_key}]['exp_avg'] shape={exp_avg.shape}, mean={exp_avg.float().mean().item():.6e}, std={exp_avg.float().std().item():.6e}", flush=True)
                        if 'exp_avg_sq' in first_state:
                            exp_avg_sq = first_state['exp_avg_sq']
                            print(f"relaunch: state[{first_key}]['exp_avg_sq'] shape={exp_avg_sq.shape}, mean={exp_avg_sq.float().mean().item():.6e}, std={exp_avg_sq.float().std().item():.6e}", flush=True)

            if "state" in opt_state and opt_state["state"]:
                loaded_state = opt_state["state"]
                first_key = next(iter(loaded_state.keys()))
                state_loaded_count = 0

                if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
                    # Integer-keyed
                    for i, (n, p) in enumerate(trainable_params):
                        key = i
                        if key in loaded_state:
                            optimizer.state[p] = {
                                'step': loaded_state[key].get('step', torch.tensor(0)),
                                'exp_avg': loaded_state[key]['exp_avg'].to(p.device),
                                'exp_avg_sq': loaded_state[key]['exp_avg_sq'].to(p.device),
                            }
                            state_loaded_count += 1
                else:
                    # Name-keyed
                    for i, (n, p) in enumerate(trainable_params):
                        if n in loaded_state:
                            optimizer.state[p] = {
                                'step': loaded_state[n].get('step', torch.tensor(0)),
                                'exp_avg': loaded_state[n]['exp_avg'].to(p.device),
                                'exp_avg_sq': loaded_state[n]['exp_avg_sq'].to(p.device),
                            }
                            state_loaded_count += 1
                        else:
                            for saved_name, saved_state in loaded_state.items():
                                if n in saved_name or saved_name in n:
                                    optimizer.state[p] = {
                                        'step': saved_state.get('step', torch.tensor(0)),
                                        'exp_avg': saved_state['exp_avg'].to(p.device),
                                        'exp_avg_sq': saved_state['exp_avg_sq'].to(p.device),
                                    }
                                    state_loaded_count += 1
                                    break

                if verbose:
                    print(f"relaunch: restored optimizer state for {state_loaded_count}/{len(trainable_params)} params, lr={lr}", flush=True)
                    # Verify restoration by checking optimizer.state
                    if len(optimizer.state) > 0:
                        sample_param = list(optimizer.state.keys())[0]
                        sample_state = optimizer.state[sample_param]
                        print(f"relaunch: [verify] optimizer.state sample - step={sample_state.get('step', 'N/A')}", flush=True)
                        if 'exp_avg' in sample_state:
                            ea = sample_state['exp_avg']
                            print(f"relaunch: [verify] exp_avg: shape={ea.shape}, mean={ea.float().mean().item():.6e}, has_nan={torch.isnan(ea).any().item()}", flush=True)
                        if 'exp_avg_sq' in sample_state:
                            eas = sample_state['exp_avg_sq']
                            print(f"relaunch: [verify] exp_avg_sq: shape={eas.shape}, mean={eas.float().mean().item():.6e}, min={eas.float().min().item():.6e}, has_nan={torch.isnan(eas).any().item()}", flush=True)
            else:
                if verbose:
                    print(f"relaunch: no state found in optimizer.pt, using fresh optimizer", flush=True)
        else:
            print(f"relaunch: WARNING - optimizer.pt not found, using fresh optimizer", flush=True)

        # Run Adam steps on validation batches
        val_iter = iter(val_loader)
        step_count = 0
        for _ in range(num_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

            # Debug: check batch before forward
            if verbose and step_count == 0:
                print(f"relaunch: [debug] batch keys: {list(batch.keys())}", flush=True)
                if 'input_ids' in batch:
                    ids = batch['input_ids']
                    print(f"relaunch: [debug] input_ids shape={ids.shape}, min={ids.min().item()}, max={ids.max().item()}", flush=True)
                if 'labels' in batch:
                    lbls = batch['labels']
                    non_neg = lbls[lbls >= 0]
                    print(f"relaunch: [debug] labels shape={lbls.shape}, num_valid={len(non_neg)}, has -100={-100 in lbls.tolist()[0] if len(lbls.shape) > 1 else -100 in lbls.tolist()}", flush=True)

            outputs = model_to_perturb(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Debug: check loss before backward
            if verbose and step_count == 0:
                print(f"relaunch: [debug] loss value before backward: {loss.item() if not torch.isnan(loss) else 'NaN'}, requires_grad={loss.requires_grad}", flush=True)

            optimizer.zero_grad()
            loss.backward()

            # Debug: check gradients before step
            if verbose and step_count == 0:
                grad_norms = []
                grad_has_nan = False
                grad_is_none_count = 0
                for n, p in trainable_params[:5]:  # Check first 5 params
                    if p.grad is not None:
                        gn = p.grad.float().norm().item()
                        grad_norms.append(gn)
                        if torch.isnan(p.grad).any():
                            grad_has_nan = True
                    else:
                        grad_is_none_count += 1
                        grad_norms.append("None")
                print(f"relaunch: [debug] first 5 param grad norms: {grad_norms}, has_nan={grad_has_nan}, none_count={grad_is_none_count}", flush=True)

                # Check model weights before step
                weight_sample = list(trainable_params)[0][1]
                print(f"relaunch: [debug] weight before step: mean={weight_sample.float().mean().item():.6e}, has_nan={torch.isnan(weight_sample).any().item()}", flush=True)

            optimizer.step()

            # Debug: check weights after step
            if verbose and step_count == 0:
                weight_sample = list(trainable_params)[0][1]
                print(f"relaunch: [debug] weight after step: mean={weight_sample.float().mean().item():.6e}, has_nan={torch.isnan(weight_sample).any().item()}", flush=True)

            step_count += 1
            if verbose:
                print(f"relaunch: Adam step {step_count}/{num_steps}, loss={loss.item():.4f}", flush=True)

        model_to_perturb.eval()
        return model_to_perturb

    # =====================================================================
    # Gradient estimation and relaunch with first-step gradient replacement
    # =====================================================================
    if args.simple_adam_perturb or args.use_precond or args.use_empirical_denom:
        print("relaunch: deprecated flags (simple_adam_perturb/use_precond/use_empirical_denom) are ignored; using first-step gradient replacement", flush=True)
        args.simple_adam_perturb = False
        args.use_precond = False
        args.use_empirical_denom = False

    if args.double_perturb_epsilon is not None:
        if float(args.double_perturb_epsilon) != 1e-3:
            print("relaunch: --double_perturb_epsilon is deprecated; use --epsilon instead", flush=True)

    if args.base_lr is not None or args.lr_scale is not None or args.perturb_lr is not None:
        print("relaunch: --base_lr/--lr_scale/--perturb_lr are ignored for training; LR is restored from scheduler.pt", flush=True)

    epsilon = float(args.epsilon)

    # If requested, build a smaller validation loader for grad estimation
    grad_loader = val_loader
    try:
        if args.grad_est_size is not None and int(args.grad_est_size) > 0:
            want = int(args.grad_est_size)
            print(f"relaunch: building small grad-est loader with up to {want} examples (val_sample_seed={args.val_sample_seed})", flush=True)
            ds = getattr(val_loader, 'dataset', None)
            if ds is None:
                ds = locals().get('hf_ds', None) or locals().get('val_ds', None)
            if ds is not None:
                import random as _random
                _random.seed(args.val_sample_seed)
                total = len(ds)
                want = min(want, total)
                indices = _random.sample(range(total), want)
                if hasattr(ds, 'select'):
                    small_ds = ds.select(indices)
                    from tacs.data_selection.get_validation_dataset import get_dataloader as get_dataloader_fn
                    grad_loader = get_dataloader_fn(small_ds, tokenizer, batch_size=1)
                else:
                    from torch.utils.data import Subset, DataLoader
                    collate = getattr(val_loader, 'collate_fn', None)
                    small_ds = Subset(ds, indices)
                    grad_loader = DataLoader(small_ds, batch_size=1, shuffle=False, collate_fn=collate)
                print(f"relaunch: small grad-est loader built with {want} examples", flush=True)
            else:
                print("relaunch: underlying dataset not accessible from val_loader; using full val_loader for grad estimation", flush=True)
                grad_loader = val_loader
    except Exception:
        grad_loader = val_loader

    avg_grads, grad_norm_sq, grad_meta = estimate_full_gradient(
        model,
        grad_loader,
        device,
        verbose=True,
        transfer_interval=10,
        heartbeat_interval=10,
    )
    print(f"relaunch: grad_norm_sq={grad_norm_sq}", flush=True)
    avg_lr = _compute_avg_lr(args.checkpoint_path, args.k)
    if avg_lr is None:
        print("relaunch: WARNING - avg_lr unavailable; metric denominator will be NaN", flush=True)
    else:
        print(f"relaunch: avg_lr_epoch_end={args.k} avg_lr={avg_lr:.6e}", flush=True)

    # determine how many epochs to continue training
    effective_k = args.k
    trainer_state_path = os.path.join(args.checkpoint_path, "trainer_state.json")
    try:
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                ts = json.load(f)
            if "epoch" in ts and isinstance(ts["epoch"], (int, float)):
                effective_k = int(round(ts["epoch"]))
            elif "global_step" in ts and isinstance(ts["global_step"], (int, float)):
                global_step = int(ts["global_step"])
                try:
                    train_files = args.train_files.split()
                    sample_pct = args.sample_percentage
                    data_seed = args.sample_data_seed if args.sample_data_seed is not None else 42
                    max_seq_len = args.max_seq_length
                    small_ds = get_training_dataset(train_files, tokenizer, max_seq_length=max_seq_len, sample_percentage=sample_pct, seed=data_seed)
                    total_examples = len(small_ds)
                    effective_batch = max(1, args.per_device_train_batch_size) * max(1, args.gradient_accumulation_steps)
                    steps_per_epoch = max(1, math.ceil(total_examples / float(effective_batch)))
                    effective_k = int(global_step // steps_per_epoch)
                except Exception:
                    effective_k = args.k
    except Exception:
        effective_k = args.k

    if args.continue_epochs is not None:
        remaining_epochs = int(args.continue_epochs)
    else:
        remaining_epochs = max(0, args.T - effective_k)

    print(f"relaunch: provided k={args.k}, inferred checkpoint_epoch={effective_k}, T={args.T}, continue_epochs_override={args.continue_epochs}, remaining_epochs={remaining_epochs}", flush=True)

    if args.train_files is None:
        raise ValueError("--train_files is required for training continuation. Provide via command line or config file.")

    train_files = args.train_files.split()
    sample_pct = args.sample_percentage
    data_seed = args.sample_data_seed if args.sample_data_seed is not None else 42
    max_seq_len = args.max_seq_length
    print(f"relaunch: loading training dataset with sample_data_seed={data_seed}, max_seq_length={max_seq_len}, sample_percentage={sample_pct}", flush=True)
    train_dataset = get_training_dataset(train_files, tokenizer, max_seq_length=max_seq_len, sample_percentage=sample_pct, seed=data_seed)
    try:
        print(f"relaunch: sampled train dataset size={len(train_dataset)} (sample_percentage={sample_pct})", flush=True)
    except Exception:
        pass
    try:
        if "dataset" in train_dataset.features:
            train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])
    except Exception:
        pass

    steps_per_epoch_approx = max(
        1,
        math.ceil(len(train_dataset) / float(args.per_device_train_batch_size * args.gradient_accumulation_steps)),
    )
    total_steps_for_T_epochs = steps_per_epoch_approx * args.T
    start_step = effective_k * steps_per_epoch_approx
    remaining_steps = total_steps_for_T_epochs - start_step
    print(f"relaunch: steps_per_epoch={steps_per_epoch_approx}, start_step={start_step}, remaining_steps={remaining_steps}")

    from transformers import TrainingArguments, Trainer, TrainerCallback
    from transformers import DataCollatorForSeq2Seq

    class FirstStepGradReplaceCallback(TrainerCallback):
        def __init__(self, avg_grads_local, epsilon_local, sign=1.0):
            self.avg_grads = avg_grads_local
            self.epsilon = float(epsilon_local)
            self.sign = float(sign)
            self._done = False
            self.eta_t = None

        def on_train_begin(self, args_cb, state, control, **kwargs):
            if self._done:
                return control
            model_cb = kwargs.get("model", None)
            optimizer_cb = kwargs.get("optimizer", None)
            if model_cb is None or optimizer_cb is None:
                return control

            orig_step = optimizer_cb.step
            avg_grads = self.avg_grads
            epsilon = self.epsilon
            sign = self.sign

            def wrapped_step(*step_args, **step_kwargs):
                if not self._done:
                    idx = 0
                    for name, p in model_cb.named_parameters():
                        if p.requires_grad and (("lora" in name) or ("Lora" in name)):
                            if idx < len(avg_grads) and p.grad is not None:
                                g = avg_grads[idx].to(p.device).to(p.dtype)
                                p.grad.add_(g, alpha=epsilon * sign)
                            idx += 1
                    try:
                        if optimizer_cb.param_groups:
                            self.eta_t = float(optimizer_cb.param_groups[0].get("lr", float("nan")))
                    except Exception:
                        self.eta_t = float("nan")
                    self._done = True
                return orig_step(*step_args, **step_kwargs)

            optimizer_cb.step = wrapped_step
            return control

    def build_trainer(model_to_train, callbacks, max_steps, ignore_data_skip):
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=0.0,
            logging_strategy="no",
            save_strategy="no",
            do_eval=False,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim=args.optim,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed or 0,
            data_seed=args.data_seed if args.data_seed is not None else (args.seed or 0),
            ignore_data_skip=ignore_data_skip,
            dataloader_num_workers=0,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_to_train, padding="longest")
        trainer_local = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        return trainer_local

    def run_one_relaunch(sign, tag):
        model_run = load_model(args.checkpoint_path)
        model_run.to(device)
        cb = FirstStepGradReplaceCallback(avg_grads, epsilon, sign=sign)
        max_steps = total_steps_for_T_epochs
        trainer_run = build_trainer(model_run, [cb], max_steps=max_steps, ignore_data_skip=False)
        start_train = time.time()
        if not args.resume_then_perturb:
            print("relaunch: resume_then_perturb not set; forcing resume_from_checkpoint to align scheduler.pt", flush=True)
        train_result = trainer_run.train(resume_from_checkpoint=args.checkpoint_path)
        dur_train_local = time.time() - start_train
        model_run = trainer_run.model
        model_run.eval()
        L_final = evaluate_loss_on_dataloader(model_run, val_loader, device)
        eta_local = cb.eta_t if cb.eta_t is not None else float("nan")
        train_metrics_local = train_result.metrics if hasattr(train_result, "metrics") else {}
        return L_final, eta_local, dur_train_local, train_metrics_local

    if args.double_perturb:
        L_plus, eta_plus, dur_plus, train_metrics_plus = run_one_relaunch(1.0, "plus")
        L_minus, eta_minus, dur_minus, train_metrics_minus = run_one_relaunch(-1.0, "minus")
        eta_t = eta_plus if not math.isnan(eta_plus) else eta_minus
        if not math.isnan(eta_plus) and not math.isnan(eta_minus):
            eta_t = (eta_plus + eta_minus) / 2.0
        denom = 2.0 * epsilon * float(avg_lr) * float(grad_norm_sq) if (grad_norm_sq and avg_lr) else float("nan")
        metric = (float(L_plus) - float(L_minus)) / denom if denom else float("nan")
        result = {
            "checkpoint": args.checkpoint_path,
            "k": args.k,
            "T": args.T,
            "baseline_L": None,
            "L_prime_plus": float(L_plus),
            "L_prime_minus": float(L_minus),
            "L_prime": float((L_plus + L_minus) / 2.0),
            "epsilon": epsilon,
            "eta_t": eta_t,
            "grad_norm_sq": float(grad_norm_sq) if grad_norm_sq is not None else float("nan"),
            "metric": metric,
            "denominator": denom,
            "avg_lr_epoch_end": args.k,
            "avg_lr": float(avg_lr) if avg_lr is not None else None,
            "dur_val_finetune_s": 0.0,
            "dur_train_s": dur_plus + dur_minus,
            "train_metrics_plus": train_metrics_plus,
            "train_metrics_minus": train_metrics_minus,
        }
        outp = os.path.join(args.output_dir, f"ckpt_{args.k}_relaunch_result.json")
        with open(outp, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    L_prime, eta_t, dur_train, train_metrics = run_one_relaunch(1.0, "single")
    baseline_L_used = float(baseline_L) if baseline_L is not None else float("nan")
    denom = epsilon * float(avg_lr) * float(grad_norm_sq) if (grad_norm_sq and avg_lr) else float("nan")
    metric = (float(L_prime) - baseline_L_used) / denom if denom else float("nan")
    result = {
        "checkpoint": args.checkpoint_path,
        "k": args.k,
        "T": args.T,
        "baseline_L": baseline_L_used,
        "L_prime": float(L_prime),
        "epsilon": epsilon,
        "eta_t": eta_t,
        "grad_norm_sq": float(grad_norm_sq) if grad_norm_sq is not None else float("nan"),
        "metric": metric,
        "denominator": denom,
        "avg_lr_epoch_end": args.k,
        "avg_lr": float(avg_lr) if avg_lr is not None else None,
        "dur_val_finetune_s": 0.0,
        "dur_train_s": dur_train,
        "train_metrics": train_metrics,
    }

    outp = os.path.join(args.output_dir, f"ckpt_{args.k}_relaunch_result.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
