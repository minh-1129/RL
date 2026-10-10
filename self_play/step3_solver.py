
import argparse
import os
import json

from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import torch

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)


from rewards_solver import format_reward_func, accuracy_reward_func

@dataclass
class MyScriptArguments(ScriptArguments):
    generated_data_path: str = field(
        default="solver_train.json",
        metadata={"help": "Path to the generated data"},
    )


def main():

    parser = TrlParser((MyScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print(f"Loading dataset from {script_args.generated_data_path}...")


    with open(script_args.generated_data_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if "answer" in item:
            item["answer"] = str(item["answer"])

    train_dataset = Dataset.from_list(data)
    print(f"\nSuccessfully loaded {len(train_dataset)} examples...")
    print(train_dataset)
    print("\n")


    SYSTEM_PROMPT_SOLVER = (
        "A conversation between user and tool-calling assistant. The user asks a question, and the assistant uses tools to solve it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> and <tool_call_answer></tool_call_answer> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\n<tool_call_answer>[{\"name\": \"<tool_name>\", \"arguments\": {\"arg1\": \"value\", \"arg2\": \"value2\", ...}}, ...]</tool_call_answer>."
    )

    
    def convert_answer_to_string(example):
        """Convert answer field from dict/list to JSON string without newlines"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_SOLVER},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
            "model_name_or_path": model_args.model_name_or_path,
        }
    
    train_dataset = train_dataset.map(convert_answer_to_string)

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        peft_config=get_peft_config(model_args),
        reward_funcs=[format_reward_func, accuracy_reward_func],
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "adapter-final"))


if __name__ == "__main__":
    main()
    

