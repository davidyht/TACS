import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge a PEFT adapter directory into a temporary full model for vLLM eval."
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not (model_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"{model_dir} does not look like a PEFT adapter directory")

    peft_cfg = PeftConfig.from_pretrained(str(model_dir))
    base_model_name = peft_cfg.base_model_name_or_path
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") not in {"", "0", "false", "False"}

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        model,
        str(model_dir),
        device_map="auto",
        local_files_only=local_files_only,
    ).merge_and_unload()
    model.eval()

    tokenizer_source = str(model_dir) if (model_dir / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "source_adapter_dir": str(model_dir),
        "base_model_name_or_path": base_model_name,
    }
    (output_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
