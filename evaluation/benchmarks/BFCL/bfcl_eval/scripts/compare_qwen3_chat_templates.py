"""Compare BFCL's manual Qwen prompt renderer with Qwen's tokenizer template.

Run from the ``evaluation/benchmarks/BFCL`` directory after installing BFCL and
Transformers:

    python -m bfcl_eval.scripts.compare_qwen3_chat_templates

On first use, Transformers downloads ``Qwen/Qwen3-0.6B`` unless it is already
in the Hugging Face cache.  Use ``--local-files-only`` to require a cached copy.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
import textwrap
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def cases() -> dict[str, list[dict[str, Any]]]:
    """Representative message histories produced by the BFCL prompt flow."""
    system = {
        "role": "system",
        "content": (
            "You are an expert in composing functions.\n\n"
            "Here is a list of functions in JSON format that you can invoke.\n"
            '[{"name": "get_weather", "parameters": {"type": "dict"}}]'
        ),
    }
    question = {"role": "user", "content": "What is the weather in Hanoi?"}

    return {
        "initial request": [system, question],
        "assistant response": [
            system,
            question,
            {"role": "assistant", "content": '[get_weather(city="Hanoi")]'},
        ],
        "one tool result": [
            system,
            question,
            {"role": "assistant", "content": '[get_weather(city="Hanoi")]'},
            {"role": "tool", "content": "25 C and clear"},
        ],
        "two tool results": [
            system,
            {"role": "user", "content": "Get weather for Hanoi and Da Nang."},
            {
                "role": "assistant",
                "content": '[get_weather(city="Hanoi"), get_weather(city="Da Nang")]',
            },
            {"role": "tool", "content": "Hanoi: 25 C and clear"},
            {"role": "tool", "content": "Da Nang: 28 C and cloudy"},
        ],
        "reasoning history": [
            system,
            question,
            {
                "role": "assistant",
                "reasoning_content": "I need current data, so I should call the tool.",
                "content": '[get_weather(city="Hanoi")]',
            },
            {"role": "tool", "content": "25 C and clear"},
        ],
    }


def manual_render(messages: list[dict[str, Any]]) -> str:
    """Execute the exact renderer currently defined in ``qwen.py``.

    Importing ``QwenHandler`` pulls in the complete BFCL evaluator dependency
    tree.  The method itself has no dependency on handler state, so extracting
    its source lets this focused check require only Transformers.
    """
    source_path = Path(__file__).parents[1] / "model_handler" / "local_inference" / "qwen.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "QwenHandler":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "_format_prompt":
                    method_source = ast.get_source_segment(source, method)
                    if method_source is None:
                        raise RuntimeError("Could not extract QwenHandler._format_prompt.")
                    renderer_namespace: dict[str, Any] = {}
                    exec(textwrap.dedent(method_source), renderer_namespace)
                    return renderer_namespace["_format_prompt"](object(), messages, [])

    raise RuntimeError("Could not find QwenHandler._format_prompt in qwen.py.")


def template_render(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    pl = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("-------------------------------------------------")
    print(pl)
    print("------")
    return pl


def print_diff(expected: str, actual: str) -> None:
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile="manual QwenHandler._format_prompt",
        tofile="tokenizer.apply_chat_template",
    )
    sys.stdout.writelines(diff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading the tokenizer from Hugging Face.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print both complete prompts, including for matching cases.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    mismatch_count = 0
    for name, messages in cases().items():
        manual = manual_render(messages)
        templated = template_render(tokenizer, messages)
        matches = manual == templated
        print(f"{'MATCH' if matches else 'DIFF'}: {name}")

        if args.show_prompts:
            print("--- manual ---")
            print(manual)
            print("--- tokenizer ---")
            print(templated)

        if not matches:
            mismatch_count += 1
            print_diff(manual, templated)

    print(f"\n{len(cases()) - mismatch_count}/{len(cases())} cases matched exactly.")
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
