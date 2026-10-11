
import argparse
import os

from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import torch
from transformers import AutoTokenizer

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

from rewards_generator import (
    reward_format_accuracy,
    reward_validity_accuracy,
    reward_curriculum,
)

@dataclass
class MyScriptArguments(ScriptArguments):
    number_of_generated_data: int = field(
        default=100,
        metadata={"help": "Number of generated data"},
    )


import random
import json
from datasets import Dataset

DOMAINS = [
    # --- Primary Functional Domains (Explicitly listed in docs) ---
    "finance",              # Stock trading, mortgage calculation, banking APIs
    "travel",               # Flight booking, hotel reservation, airport info
    "math",                 # Algebra, statistics, calculator functions
    "sports",               # Sports scores (e.g., Soccer/Football, NBA)
    "weather",              # Real-time weather data
    "system",               # File System (ls, cd, cat), OS commands
    "database",             # SQL queries (SELECT, INSERT, etc.)
    "vehicle_control",      # Car status, engine control, EV charging
    "communication",        # Messaging (Slack, Email, SMS)
    "entertainment",        # Movies (TMDB), Music (Spotify)
    "retail_ecommerce",     # Inventory management, order status, product search
    "scheduling",           # Calendar management, meeting booking
    "cloud_infrastructure", # VM management, AWS/Cloud resource handling
    "geolocation",          # Maps, routing, distance estimation
    
    # --- Agentic & Technical Domains (Live Updates) ---
    "web_search",           # Multi-hop reasoning, internet search (SerpAPI)
    "memory_management",    # Key-value store, vector database, recursive summarization
    "programming",          # Code execution, debugging helpers
    "iot",                  # Internet of Things (smart home device control)
    "social_media",         # Twitter/X, Reddit API interactions
    "logistics",            # Shipping tracking, supply chain
    "real_estate",          # Property listing, housing data
    
    # --- Specialized & "Live" API Domains ---
    "food_ordering",        # Restaurant delivery services
    "healthcare",           # Basic medical info (via public APIs)
    "education",            # Course searching, academic references
    "productivity",         # Note-taking (Notion), task management
    "insurance",            # Purchase insurance, policy checks
    "cybersecurity",        # Basic security checks, auth verification
    "legal",                # Regulatory data lookup
    "government",           # Public service APIs
    "news",                 # News aggregation
    "translation",          # Language translation services
    "utilities",            # Energy usage, utility billing
    "customer_support"      # FAQ retrieval, ticket creation
]

DOMAIN_WEIGHTS = {
    "finance": 0.03125,
    "healthcare": 0.03125,
    "productivity": 0.03125,
    "retail_ecommerce": 0.03125,
    "scheduling": 0.03125,
    "database": 0.03125,
    "cloud_infrastructure": 0.03125,
    "system": 0.03125,
    "programming": 0.03125,
    "geolocation": 0.03125,
    "logistics": 0.03125,
    "communication": 0.03125,
    "iot": 0.03125,
    "cybersecurity": 0.03125,
    "insurance": 0.03125,
    "legal": 0.03125,
    "news": 0.03125,
    "weather": 0.03125,
    "sports": 0.03125,
    "entertainment": 0.03125,
    "education": 0.03125,
    "real_estate": 0.03125,
    "food_ordering": 0.03125,
    "translation": 0.03125,
    "utilities": 0.03125,
    "government": 0.03125,
    "memory_management": 0.03125,
    "web_search": 0.03125,
    "social_media": 0.03125,
    "math": 0.03125,
    "vehicle_control": 0.03125,
    "travel": 0.03125,
    # "customer_support": 0.03125  # Uncomment if this is in your DOMAINS list
}



def weighted_choice(weight_dict):
    items = list(weight_dict.items())
    keys, weights = zip(*items)
    return random.choices(keys, weights=weights, k=1)[0]

def sample_spec():

    context_type = random.choices(["single_turn", "multi_turn"], weights=[0.9, 0.1], k=1)[0]

    if context_type == "multi_turn":
        num_calls = 1
    else:
        num_calls = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]

    if num_calls > 1:
        tool_menu_size = random.choices([3, 4, 5], weights=[0.3, 0.4, 0.3], k=1)[0]
    else:
        bucket = random.choices(["SMALL5", "LARGE"], weights=[0.4, 0.6], k=1)[0]
        tool_menu_size = random.randint(2, 4) if bucket == "SMALL5" else random.randint(5, 8)

    domain = weighted_choice(DOMAIN_WEIGHTS)

    return {
        "domain": domain,
        "num_calls": num_calls,
        "tool_menu_size": tool_menu_size,
        "context_type": context_type,
    }



def main():
    parser = TrlParser((MyScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    N = script_args.number_of_generated_data
    specs = [sample_spec() for _ in range(N)]

    dummy_examples = [{"question": ""} for _ in range(N)]
    train_dataset = Dataset.from_list(dummy_examples)
    train_dataset = train_dataset.add_column("gen_spec", [json.dumps(s) for s in specs])

    SYSTEM_PROMPT_GENERATOR_TEMPLATE = """You are an expert task generator for tool-calling agents.

    FIRST, in your private scratch-pad, reason step-by-step to design a realistic, non-trivial task that cannot be solved without correctly calling one or sometimes multiple tools.

    CONTROL SPEC (MUST FOLLOW EXACTLY):
    - Domain: {domain}
    - Context type: {context_type}  (single_turn or multi_turn)
    - Number of available tools: {tool_menu_size} (<available_tools>)
    - Number of gold tool calls: {num_calls} (<tool_call_answer>)

    RULES TO SATISFY THE SPEC:
    1) You MUST output exactly {tool_menu_size} tools in <available_tools>.
    2) You MUST output exactly {num_calls} tool calls (JSON list length) in <tool_call_answer>.
    3) Domain must be {domain}. Do not drift into other domains.
    4) If context_type=multi_turn, embed a short conversation in <question> like: "# Conversation\\nUser: ...\\nAgent: ...\\nUser: ...\\nAgent: ..."
    5) Tool arguments must be flat primitives only (no lists, no nested objects).
    6) The function values (<value1>, <value2>, ...) MUST be present inside user question (<question>...</question>), otherwise agent cannot solve the task.

    THEN, without revealing your reasoning, output the following four blocks in the exact format, NOTHING ELSE:

    <think>
    Your private reasoning here.
    </think>

    <question>
    Write a natural user question (no bullet points, no meta-instructions, no placeholders).
    It must be a natural question, be in domain "{domain}", and mention the exact argument values that appear in <tool_call_answer>.
    </question>

    <available_tools>
    A JSON list of tools. Each tool MUST include: "name", "description", and "parameters". "parameters" MUST be a JSON object mapping param_name -> {{ "type": "...", "description": "..." }} and OPTIONALLY include top-level "required": ["param1", ...] which can be empty list if no required parameters.
    [
        {{
            "name": "<tool_name>",
            "description": "<short description>",
            "parameters": {{
            "<param1>": {{"type": "<param1_type>", "description": "<param1_description>"}},
            "<param2>": {{"type": "<param2_type>", "description": "<param2_description>"}},
            ...
            }},
            "required": [<param1>, ...],
        }},
        ...
    ]
    </available_tools>

    <tool_call_answer>
    [
    {{\"name\": \"<tool_name>\", \"arguments\": {{\"<param>\": <value>, ...}}}}
    ]
    </tool_call_answer>"""

    USER_PROMPT_GENERATOR = "Generate a new tool-calling task now. Follow the CONTROL SPEC exactly and remember to format the output exactly as instructed."

    SYSTEM_PROMPT_SOLVER = (
        "A conversation between user and tool-calling assistant. The user asks a question, and the assistant uses tools to solve it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> and <tool_call_answer></tool_call_answer> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\n<tool_call_answer>[{\"name\": \"<tool_name>\", \"arguments\": {\"arg1\": \"value\", \"arg2\": \"value2\", ...}}, ...]</tool_call_answer>."
    )

    def make_conversation_generator(example):
        spec = json.loads(example["gen_spec"])
        system_prompt = SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**spec)
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT_GENERATOR},
            ],
            "gen_spec": example["gen_spec"],
        }

    train_dataset = train_dataset.map(make_conversation_generator)
    train_dataset = train_dataset.remove_columns(["question"])
    train_dataset = train_dataset.add_column("solver_prompt", [SYSTEM_PROMPT_SOLVER] * len(train_dataset))
    train_dataset = train_dataset.add_column("model_name_or_path", [model_args.model_name_or_path] * len(train_dataset))


    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        reward_funcs=[
            reward_format_accuracy,
            reward_validity_accuracy,
            reward_curriculum,
        ],
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "adapter-final"))


if __name__ == "__main__":
    main()
    
