"""Merge a saved LoRA checkpoint into a standalone model for vLLM."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model")
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if not (args.adapter_dir / "adapter_config.json").is_file():
        parser.error(f"LoRA checkpoint not found: {args.adapter_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged_model = PeftModel.from_pretrained(base_model, args.adapter_dir).merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
