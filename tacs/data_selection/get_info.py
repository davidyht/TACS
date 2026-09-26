"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os
import pdb
from copy import deepcopy
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tacs.data_selection.get_training_dataset import get_training_dataset
from tacs.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_dataset)
from tacs.data_selection.merge_grad_shard_outputs import load_shard_bounds, shard_bounds


def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16,
               trust_remote_code: bool = False) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=trust_remote_code)
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=trust_remote_code)

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


if __name__ == "__main__":
    from tacs.data_selection.collect_grad_reps import collect_grads, collect_reps, get_loss
    parser = argparse.ArgumentParser(
        description='Script for getting validation gradients')
    parser.add_argument('--task', type=str, default=None,
                        help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
    parser.add_argument("--train_file", type=str,
                        default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
    parser.add_argument(
        "--info_type", choices=["grads", "reps", "loss"], help="The type of information")
    parser.add_argument("--model_path", type=str,
                        default=None, help="The path to the model")
    parser.add_argument("--state_dict_path", type=str, default=None,
                        help="Optional path to a raw model state_dict (.pt) to load after model init.")
    parser.add_argument("--optimizer_state_path", type=str, default=None,
                        help="Optional explicit optimizer state path for Adam gradients.")
    parser.add_argument("--allow_missing_optimizer_state", action="store_true",
                        help="For Adam gradients, continue with zero-initialized optimizer moments if optimizer state is missing.")
    parser.add_argument("--max_samples", type=int,
                        default=None, help="The maximum number of samples")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"], help="The torch data type")
    parser.add_argument("--output_path", type=str,
                        default=None, help="The path to the output")
    parser.add_argument("--data_dir", type=str,
                        default=None, help="The path to the data")
    parser.add_argument("--gradient_projection_dimension", nargs='+',
                        help="The dimension of the projection, can be a list", type=int, default=[8192])
    parser.add_argument("--gradient_type", type=str, default="adam",
                        choices=["adam", "sign", "sgd"], help="The type of gradient")
    parser.add_argument("--project_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Projection dtype for gradients (auto uses float32 for Adam, float16 otherwise).")
    parser.add_argument("--chat_format", type=str,
                        default="tulu", help="The chat format")
    parser.add_argument("--use_chat_format", type=bool,
                        default=True, help="Whether to use chat format")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="The maximum length")
    parser.add_argument("--zh", default=False, action="store_true",
                        help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
    parser.add_argument("--initialize_lora", default=False, action="store_true",
                        help="Whether to initialize the base model with lora, only works when is_peft is False")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="The value of lora_r hyperparameter")
    parser.add_argument("--lora_alpha", type=float, default=32,
                        help="The value of lora_alpha hyperparameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="The value of lora_dropout hyperparameter")
    parser.add_argument("--lora_target_modules", nargs='+', default=[
                        "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow loading models/tokenizers with custom code from the Hub.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Optional tokenizer source. Defaults to --model_path, with automatic PEFT base-model fallback.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split --train_file rows into this many contiguous shards and process one of them. "
                             "Gradients are per-example (batch size 1, eval mode, fixed projector seed), so "
                             "concatenating shard outputs in shard order equals the unsharded result.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Which contiguous shard to process (0-based); see merge_grad_shard_outputs.py.")
    parser.add_argument("--shard_indices", type=str, default=None,
                        help="Comma-separated shard indices to process sequentially in ONE process "
                             "(e.g. '0,1,2,3'), amortizing the per-process CUDA warmup ramp across "
                             "them (~3.3x measured on the 2nd shard onward). Requires --num_shards > 1. "
                             "--output_path is then the PARENT directory: each shard writes to "
                             "<output_path>/shard_<k>_of_<num_shards>, matching the existing layout. "
                             "Grads only. When omitted, --shard_index is used and behaviour is unchanged.")
    parser.add_argument("--shard_bounds_file", type=str, default=None,
                        help="Optional JSON of row boundaries for --num_shards (length-balanced contiguous "
                             "shards from make_length_balanced_shards.py); default is equal row counts.")

    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard {args.shard_index} of {args.num_shards}")
    if args.num_shards > 1 and args.train_file is None:
        raise ValueError("--num_shards applies to --train_file only")
    if args.shard_bounds_file is not None and args.num_shards < 2:
        raise ValueError("--shard_bounds_file requires --num_shards > 1")
    # Validate --shard_indices up front, before a 3B model and a 100k-row tokenization
    # are loaded, so a bad index fails in milliseconds rather than minutes.
    SHARD_PLAN = None
    if args.shard_indices:
        if args.num_shards < 2:
            raise ValueError("--shard_indices requires --num_shards > 1")
        if args.info_type != "grads":
            raise ValueError("--shard_indices is supported for --info_type grads only")
        SHARD_PLAN = [int(s) for s in str(args.shard_indices).split(",") if str(s).strip() != ""]
        if not SHARD_PLAN:
            raise ValueError("--shard_indices parsed to an empty list")
        if len(set(SHARD_PLAN)) != len(SHARD_PLAN):
            raise ValueError(f"--shard_indices contains duplicates: {SHARD_PLAN}")
        for _si in SHARD_PLAN:
            if not (0 <= _si < args.num_shards):
                raise ValueError(f"invalid shard {_si} of {args.num_shards} in --shard_indices")
    assert args.task is not None or args.train_file is not None

    if args.model_path is None or str(args.model_path).strip() == "":
        raise ValueError("model_path is empty. Provide --model_path (checkpoint dir or HF model id).")
    candidate_model_path = os.path.expanduser(args.model_path)
    if (os.path.isabs(candidate_model_path) or candidate_model_path.startswith(".")) and not os.path.exists(candidate_model_path):
        raise FileNotFoundError(f"model_path looks like a local path but was not found: {candidate_model_path}")

    tokenizer_src = args.tokenizer_name_or_path if (args.tokenizer_name_or_path and str(args.tokenizer_name_or_path).strip() != "") else args.model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=args.trust_remote_code)
    except Exception as tok_exc:
        # Adapter checkpoints often do not carry tokenizer files; fall back to PEFT base model tokenizer.
        peft_cfg_path = os.path.join(args.model_path, "adapter_config.json")
        if args.tokenizer_name_or_path is None and os.path.exists(peft_cfg_path):
            try:
                peft_cfg = LoraConfig.from_pretrained(args.model_path)
                base_tok_src = peft_cfg.base_model_name_or_path
                print(
                    f"Tokenizer load from adapter dir failed ({tok_exc}); "
                    f"retrying with base model tokenizer: {base_tok_src}"
                )
                tokenizer = AutoTokenizer.from_pretrained(base_tok_src, trust_remote_code=args.trust_remote_code)
            except Exception:
                raise tok_exc
        else:
            raise tok_exc
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = load_model(args.model_path, dtype, trust_remote_code=args.trust_remote_code)
    if args.info_type == "grads":
        # reduce activation memory pressure for gradient extraction
        try:
            if hasattr(model, "config") and getattr(model.config, "use_cache", False):
                model.config.use_cache = False
                print("Disabled model.config.use_cache for gradient extraction.")
        except Exception as exc:
            print(f"Warning: failed to disable use_cache: {exc}")
        # Enable activation checkpointing unless explicitly disabled.
        try:
            if os.environ.get("LESS_ENABLE_GRADIENT_CHECKPOINTING", "1") not in {"0", "false", "False"}:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    print("Enabled gradient checkpointing for gradient extraction.")
        except Exception as exc:
            print(f"Warning: failed to enable gradient checkpointing: {exc}")

    # pad token is not added by default for pretrained models
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.initialize_lora:
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if args.state_dict_path is not None and str(args.state_dict_path).strip() != "":
        state_dict_path = os.path.expanduser(args.state_dict_path)
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"state_dict_path not found: {state_dict_path}")
        state_obj = torch.load(state_dict_path, map_location="cpu")
        if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            state_obj = state_obj["state_dict"]
        if not isinstance(state_obj, dict):
            raise ValueError(f"Expected a dict-like state_dict at {state_dict_path}, got {type(state_obj)}")
        missing, unexpected = model.load_state_dict(state_obj, strict=False)
        print(
            f"Loaded state_dict from {state_dict_path}: "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()

    # Ensure LoRA parameters are marked trainable regardless of naming/casing.
    trainable_before = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    trainable_after = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    if trainable_after != trainable_before:
        print(f"Enabled LoRA trainable params: {trainable_before} -> {trainable_after}")

    adam_optimizer_state = None
    if args.info_type == "grads" and args.gradient_type == "adam":
        # Some checkpoints store optimizer state under different filenames
        # (e.g. optimizer.bin, optimizer.pt, optimizer.pth). Try a list of
        # candidate names and pick the first existing file. Provide a clear
        # error message if none are found.
        # Prefer the converted optimizer keyed by parameter names if present.
        candidate_names = [
            "optimizer_with_names.pt",
            "optimizer.pt",
            "optimizer.bin",
            "optimizer.pth",
            "optimizer",
        ]
        optimizer_path = None
        if args.optimizer_state_path is not None and str(args.optimizer_state_path).strip() != "":
            p = os.path.expanduser(args.optimizer_state_path)
            if os.path.exists(p):
                optimizer_path = p
            elif args.allow_missing_optimizer_state:
                print(f"Warning: optimizer_state_path not found ({p}); falling back to zero moments.")
            else:
                raise FileNotFoundError(f"optimizer_state_path not found: {p}")
        elif os.path.isdir(args.model_path):
            for name in candidate_names:
                p = os.path.join(args.model_path, name)
                if os.path.exists(p):
                    optimizer_path = p
                    break
        if optimizer_path is None:
            if args.allow_missing_optimizer_state:
                print(
                    "Warning: no optimizer state found for Adam gradients; "
                    "using zero-initialized moments."
                )
                adam_optimizer_state = {}
            else:
                raise FileNotFoundError(
                    f"No optimizer state file found in {args.model_path}.\n"
                    f"Tried: {candidate_names}.\n"
                    "If you only have a different filename, pass --optimizer_state_path.\n"
                    "Or pass --allow_missing_optimizer_state to use zero moments."
                )
        else:
            print(f"Loading optimizer state from {optimizer_path}")
            adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")

    if args.task is not None:
        dataset = get_dataset(args.task,
                              data_dir=args.data_dir,
                              tokenizer=tokenizer,
                              chat_format=args.chat_format,
                              use_chat_format=args.use_chat_format,
                              max_length=args.max_length,
                              zh=args.zh)
        dataloader = get_dataloader(dataset, tokenizer=tokenizer)
    else:
        assert args.train_file is not None
        train_files = args.train_file
        if isinstance(train_files, str) and " " in train_files:
            train_files = train_files.split()
        dataset = get_training_dataset(
            train_files, tokenizer, args.max_length, sample_percentage=1.0, chat_format=args.chat_format)
        columns = deepcopy(dataset.column_names)
        columns.remove("input_ids")
        columns.remove("labels")
        columns.remove("attention_mask")
        dataset = dataset.remove_columns(columns)
        full_dataset = dataset
        if SHARD_PLAN:
            # The multi-shard loop below slices full_dataset itself. Skip the single-shard
            # select/dataloader here: --shard_index still defaults to 0, so leaving this in
            # builds a throwaway DataLoader and logs a duplicate "Shard 0/N" line that looks
            # like shard 0 ran twice.
            dataloader = None
        else:
            if args.num_shards > 1:
                if args.shard_bounds_file is not None:
                    bounds = load_shard_bounds(args.shard_bounds_file, len(dataset), args.num_shards)
                    start, end = bounds[args.shard_index], bounds[args.shard_index + 1]
                else:
                    start, end = shard_bounds(len(dataset), args.shard_index, args.num_shards)
                print(f"Shard {args.shard_index}/{args.num_shards}: rows [{start}, {end}) of {len(dataset)}")
                dataset = dataset.select(range(start, end))
            dataloader = get_dataloader(dataset, tokenizer=tokenizer)

    # Multi-shard-per-process plan (opt-in via --shard_indices).
    #
    # Measured 2026-09-17: the CUDA/kernel warmup ramp is held PER PROCESS. A fresh
    # process always restarts it (two identical 800-batch runs took 654s and 663s),
    # but a SECOND shard inside the SAME process starts at plateau -- 400 batches took
    # 436s as the first shard and 130s as the second, a 3.3x speedup, despite the second
    # shard holding longer examples. So a shell loop over shards buys nothing (it spawns
    # a new interpreter each time) while an in-process loop captures the full gain.
    #
    # Each shard still writes to its own shard_<k>_of_<n> directory, so the on-disk
    # layout and merge_grad_shard_outputs contract are unchanged.
    shard_plan = SHARD_PLAN  # validated immediately after parse_args()

    if shard_plan and args.info_type == "grads":
        if args.project_dtype == "auto":
            project_dtype = torch.float32 if args.gradient_type == "adam" else torch.float16
        elif args.project_dtype == "float16":
            project_dtype = torch.float16
        elif args.project_dtype == "bfloat16":
            project_dtype = torch.bfloat16
        else:
            project_dtype = torch.float32

        for _pos, _si in enumerate(shard_plan):
            if args.shard_bounds_file is not None:
                _bounds = load_shard_bounds(args.shard_bounds_file, len(full_dataset), args.num_shards)
                _start, _end = _bounds[_si], _bounds[_si + 1]
            else:
                _start, _end = shard_bounds(len(full_dataset), _si, args.num_shards)
            print(f"Shard {_si}/{args.num_shards} (pos {_pos} of {len(shard_plan)}): "
                  f"rows [{_start}, {_end}) of {len(full_dataset)}", flush=True)
            _shard_out = os.path.join(args.output_path, f"shard_{_si}_of_{args.num_shards}")
            os.makedirs(_shard_out, exist_ok=True)
            _loader = get_dataloader(full_dataset.select(range(_start, _end)), tokenizer=tokenizer)
            collect_grads(_loader,
                          model,
                          _shard_out,
                          proj_dim=args.gradient_projection_dimension,
                          gradient_type=args.gradient_type,
                          adam_optimizer_state=adam_optimizer_state,
                          max_samples=args.max_samples,
                          project_dtype=project_dtype)
        raise SystemExit(0)

    if args.info_type == "reps":
        collect_reps(dataloader, model, args.output_path,
                     max_samples=args.max_samples)
    elif args.info_type == "grads":
        if args.project_dtype == "auto":
            project_dtype = torch.float32 if args.gradient_type == "adam" else torch.float16
        elif args.project_dtype == "float16":
            project_dtype = torch.float16
        elif args.project_dtype == "bfloat16":
            project_dtype = torch.bfloat16
        else:
            project_dtype = torch.float32
        collect_grads(dataloader,
                      model,
                      args.output_path,
                      proj_dim=args.gradient_projection_dimension,
                      gradient_type=args.gradient_type,
                      adam_optimizer_state=adam_optimizer_state,
                      max_samples=args.max_samples,
                      project_dtype=project_dtype)
    elif args.info_type == "loss":
        get_loss(dataloader, model, args.output_path)
