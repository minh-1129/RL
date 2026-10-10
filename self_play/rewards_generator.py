import json
import re
import os
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Set
import unicodedata

from openai import OpenAI

P_LOW = 0.25
P_HIGH = 0.75
P_TARGET = 0.5
K_SOLVER_SAMPLES = 8
TEMP_SOLVER = 0.7
MAX_TOKENS_SOLVER = 2048

W_FORMAT = 0.25
W_GOLD_VALID = 0.25
W_DIFFICULTY = 0.5

HARD_GATE_ON_GOLD_INVALID = False

LAMBDA_REP = 0.20
TAU_REP = 0.30
NGRAM_N = 3

ALPHA_TAGS_PARSED = 1.0 / 3.0
ALPHA_TOOLS_VALID_JSON = 1.0 / 3.0
ALPHA_GOLD_VALID_JSON = 1.0 / 3.0

ALPHA_GOLD_TOOL_EXISTS = 0.4
ALPHA_GOLD_ARGS_VALID = 0.4
ALPHA_GOLD_VALUES_VALID = 0.2

ALPHA_DIFFICULTY_RAW = 0.5
ALPHA_SEMANTIC_COHERENCE = 0.5
ALPHA_REPETITION_PENALTY = 1.0

TAG_PATTERNS = {
    "think": re.compile(r"<think>(.*?)</think>", re.DOTALL),
    "question": re.compile(r"<question>(.*?)</question>", re.DOTALL),
    "available_tools": re.compile(r"<available_tools>(.*?)</available_tools>", re.DOTALL),
    "tool_call_answer": re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL),
}


def _last_tag(text: str, tag: str) -> Optional[str]:
    m = TAG_PATTERNS[tag].findall(text)
    if not m:
        return None
    return m[-1].strip()


def extract_generator_fields(completion_text: str, think_flag: bool = False) -> Optional[Dict[str, str]]:
    """Extract tagged sections from a generator completion.

    Parses the completion text for the required XML-like tags
    (`<question>`, `<available_tools>`, `<tool_call_answer>`, and
    optionally `<think>` when `think_flag` is True) and returns a mapping
    from tag name to stripped string content. Returns None if any required
    tag is missing or empty.
    """
    if think_flag:
        required = ["think", "question", "available_tools", "tool_call_answer"]
    else:
        required = ["question", "available_tools", "tool_call_answer"]
    out = {}

    for k in required:
        v = _last_tag(completion_text, k)
        if v is None or v == "":
            return None
        out[k] = v.strip()

    return out


def _json_loads_relaxed(s: str) -> Optional[Any]:
    """Best-effort JSON deserializer for model outputs.

    Strips optional triple-backtick fences (with or without a `json` hint),
    then attempts to parse the remaining string as JSON. Returns the parsed
    object on success or None if parsing fails.
    """
    s2 = s.strip()
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s2, flags=re.IGNORECASE)
    try:
        return json.loads(s2)
    except Exception:
        return None


def _canonical_json(x: Any) -> str:
    """Serialize a Python object into a canonical JSON string.

    Uses sorted keys and compact separators to ensure that semantically
    equivalent objects produce identical string representations. This is
    primarily used for stable equality checks and cache keys.
    """
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_tool_call(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize a raw tool-call object into a `{name, arguments}` dict.

    Accepts several common tool-call shapes, including OpenAI-style
    `{"function": {"name": ..., "arguments": ...}}`, as well as flat
    `{"name": ..., "arguments": ...}` or dicts where all non-name keys
    are treated as arguments. Returns None if the structure cannot be
    interpreted as a valid tool call.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        obj = obj[0]

    if not isinstance(obj, dict):
        return None

    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _json_loads_relaxed(args)
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args}

    name = obj.get("name") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None

    args = obj.get("arguments")
    if isinstance(args, str):
        args = _json_loads_relaxed(args)
    if isinstance(args, dict):
        return {"name": name, "arguments": args}

    flat = {k: v for k, v in obj.items() if k not in ("name", "tool_name")}
    return {"name": name, "arguments": flat}


def parse_available_tools(tools_text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse and validate the `<available_tools>` JSON payload.

    Attempts to deserialize `tools_text` and checks that it is a non-empty
    list of dicts where each entry has a non-empty string `name` field and
    an optional `parameters` field that, if present, is a dict. Returns the
    validated list or None if the structure is invalid.
    """
    tools = _json_loads_relaxed(tools_text)
    if not isinstance(tools, list) or len(tools) == 0:
        return None
    for t in tools:
        if not isinstance(t, dict) or "name" not in t:
            return None
        if not isinstance(t["name"], str) or t["name"].strip() == "":
            return None
        if "parameters" in t and not isinstance(t["parameters"], dict):
            return None
    return tools


def validate_args_against_schema(args: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> bool:
    """Check that a tool-call argument dict satisfies a JSON schema.

    Treats `schema` as a minimal JSON Schema–like object where only the
    `required` key is honored. Returns True if either no schema is given
    or all required keys are present in `args`, otherwise False.
    """
    if schema is None:
        return True
    if not isinstance(schema, dict):
        return True
    req = schema.get("required")
    if isinstance(req, list):
        for k in req:
            if k not in args:
                return False
    return True


def tool_index(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a name-to-spec index from a list of tool specifications."""
    return {t["name"]: t for t in tools}


def build_solver_user_message(question: str, tools_text: str) -> str:
    """Construct the user message passed to the solver model."""
    return question.strip() + "\n\n<available_tools>\n" + tools_text.strip() + "\n</available_tools>\n"


def extract_solver_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract and normalize the solver's `<tool_call_answer>` from text."""
    tca = _last_tag(response_text, "tool_call_answer")
    if tca is None:
        return None
    obj = _json_loads_relaxed(tca)
    return normalize_tool_call(obj)


@lru_cache(maxsize=8)
def get_client(base_url: str) -> OpenAI:
    """Create or retrieve a cached OpenAI-compatible client for a base URL."""
    return OpenAI(base_url=base_url, api_key="EMPTY")


def solver_sample_tool_calls(
    base_url: str,
    model: str,
    solver_system_prompt: str,
    user_message: str,
    k: int = K_SOLVER_SAMPLES,
    temperature: float = TEMP_SOLVER,
    max_tokens: int = MAX_TOKENS_SOLVER,
) -> List[Optional[Dict[str, Any]]]:
    """Query the solver model multiple times and collect normalized tool calls."""
    client = get_client(base_url)
    outs: List[Optional[Dict[str, Any]]] = []
    for _ in range(k):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": solver_system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        txt = resp.choices[0].message.content or ""
        outs.append(extract_solver_tool_call(txt))
    return outs


class SolverCache:
    """
    Cache solver calls to avoid redundant inference when using
    separate reward_difficulty_raw and reward_consistency functions.
    """
    _cache: Dict[str, List[Optional[Dict[str, Any]]]] = {}
    
    @classmethod
    def get_key(cls, user_message: str, gold: Dict[str, Any]) -> str:
        return f"{hash(user_message)}_{_canonical_json(gold)}"
    
    @classmethod
    def get_or_call(
        cls,
        base_url: str,
        model: str,
        solver_system_prompt: str,
        user_message: str,
        gold: Dict[str, Any],
        k: int = K_SOLVER_SAMPLES,
        temperature: float = TEMP_SOLVER,
        max_tokens: int = MAX_TOKENS_SOLVER,
    ) -> List[Optional[Dict[str, Any]]]:
        key = cls.get_key(user_message, gold)
        
        if key not in cls._cache:
            cls._cache[key] = solver_sample_tool_calls(
                base_url=base_url,
                model=model,
                solver_system_prompt=solver_system_prompt,
                user_message=user_message,
                k=k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        cls._cache.clear()


def solver_sample_tool_calls_cached(
    base_url: str,
    model: str,
    solver_system_prompt: str,
    user_message: str,
    gold: Dict[str, Any],
    k: int = K_SOLVER_SAMPLES,
    temperature: float = TEMP_SOLVER,
    max_tokens: int = MAX_TOKENS_SOLVER,
) -> List[Optional[Dict[str, Any]]]:
    """Cached version of solver_sample_tool_calls."""
    return SolverCache.get_or_call(
        base_url=base_url,
        model=model,
        solver_system_prompt=solver_system_prompt,
        user_message=user_message,
        gold=gold,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def consistency_score(calls: List[Optional[Dict[str, Any]]]) -> float:
    canon = [_canonical_json(c) for c in calls if c is not None]
    if len(canon) == 0:
        return 0.0
    freq = {}
    for s in canon:
        freq[s] = freq.get(s, 0) + 1
    mode = max(freq.values())
    return mode / len(calls)


def success_prob_against_gold(
    calls: List[Optional[Dict[str, Any]]],
    gold: Dict[str, Any],
) -> float:
    g = _canonical_json(gold)
    good = 0
    for c in calls:
        if c is not None and _canonical_json(c) == g:
            good += 1
    return good / len(calls) if calls else 0.0


def difficulty_reward_bandpass(
    p_success: float,
    p_low: float = 0.25,
    p_high: float = 0.75,
    k: int = K_SOLVER_SAMPLES,
    sigma: float = 0.12,
) -> float:
    """
    Flat-top bandpass reward for difficulty.
    """
    if p_success < (1.0 / k):
        return 0.0
    
    if p_low <= p_success <= p_high:
        return 1.0
    
    if p_success < p_low:
        diff = p_success - p_low
        return math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    
    if p_success > p_high:
        diff = p_success - p_high
        return math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    
    return 1.0


class CompletionCache:
    """
    Cache parsed fields for a completion to avoid redundant parsing
    across multiple reward functions.
    """
    _cache: Dict[int, Dict[str, Any]] = {}
    
    @classmethod
    def get_or_parse(cls, completion_text: str) -> Dict[str, Any]:
        key = id(completion_text)
        
        if key not in cls._cache:
            fields = extract_generator_fields(completion_text, think_flag=True)
            
            if fields is None:
                cls._cache[key] = {
                    "valid": False,
                    "fields": None,
                    "tools": None,
                    "gold": None,
                }
            else:
                tools = parse_available_tools(fields["available_tools"])
                gold_obj = _json_loads_relaxed(fields["tool_call_answer"])
                gold = normalize_tool_call(gold_obj) if gold_obj is not None else None
                
                cls._cache[key] = {
                    "valid": True,
                    "fields": fields,
                    "tools": tools,
                    "gold": gold,
                }
        
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        cls._cache.clear()



_NUM_WORDS_0_19 = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}

_WORD_TO_NUM = {v: str(k) for k, v in _NUM_WORDS_0_19.items()}
_WORD_TO_NUM.update({v: str(k) for k, v in _TENS.items()})

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]+")

def _norm_text(s: str) -> str:
    """Normalize free-form text for robust matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = s.replace("\u00A0", " ")
    s = _NON_ALNUM_SPACE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def get_grounding_variants(val: Any) -> Set[str]:
    """Generate safe text variants for grounding a value in the question."""
    variants: Set[str] = set()

    s_raw = str(val).strip()
    s = _norm_text(s_raw)
    if s:
        variants.add(s)

    try:
        f = float(str(val))
        if f.is_integer():
            i = int(f)
            variants.add(_norm_text(str(i)))
            if abs(i) >= 1000:
                variants.add(_norm_text(f"{i:,}"))

            if 0 <= i <= 19:
                variants.add(_NUM_WORDS_0_19[i])
            elif i in _TENS:
                variants.add(_TENS[i])
    except Exception:
        pass

    if s in _WORD_TO_NUM:
        variants.add(_WORD_TO_NUM[s])

    return variants

_PLACEHOLDER_PATTERNS = [
    r"\bthe generated user question here\b",
    r"\buser question must be from the specified domain\b",
    r"\bgenerate a new tool-calling task now\b",
    r"\bcontrol spec\b",
    r"\brules to satisfy\b",
    r"\bprivate reasoning\b",
    r"\[the private reasoning here\]",
    r"\[.*generated user question.*\]",
]

def is_placeholder_question(q: str) -> bool:
    """Detect whether a question looks like a template or placeholder."""
    if not q:
        return True
    qn = q.strip().lower()
    if len(qn) < 8:
        return True
    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, qn):
            return True
    return False


def _question_gate_mask(completions) -> List[float]:
    """Compute a gating mask that suppresses rewards for bad questions."""
    mask = []
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        if (not parsed["valid"]) or (parsed["fields"] is None):
            mask.append(0.0)
            continue
        q = parsed["fields"].get("question", "")
        mask.append(0.0 if is_placeholder_question(q) else 1.0)
    return mask


def reward_tags_parsed(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 1: Did all required tags parse successfully?
    
    Returns:
        0.0 if any required tag is missing or empty
        1.0 if all tags parsed successfully
    """
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if parsed["valid"]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def reward_tools_valid_json(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 2: Is <available_tools> a valid JSON list of tool specs?
    
    Returns:
        0.0 if tools don't parse or tags are invalid
        1.0 if tools parse as valid JSON list with proper structure
    """
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"]:
            rewards.append(0.0)
        elif parsed["tools"] is not None:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def reward_gold_valid_json(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 3: Does <tool_call_answer> parse and normalize correctly?
    
    Returns:
        0.0 if gold doesn't parse/normalize or tags are invalid
        1.0 if gold parses and normalizes to valid tool call structure
    """
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"]:
            rewards.append(0.0)
        elif parsed["gold"] is not None:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def reward_gold_tool_exists(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 4: Does the gold tool call reference a tool that exists in available_tools?
    
    Returns:
        0.0 if prerequisites fail or tool name not found
        1.0 if gold tool name exists in available_tools
    """
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"] or parsed["tools"] is None or parsed["gold"] is None:
            rewards.append(0.0)
            continue
        
        tindex = tool_index(parsed["tools"])
        gold_name = parsed["gold"]["name"]
        
        if gold_name in tindex:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def reward_gold_args_valid(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 5: Do the gold arguments satisfy the tool's schema (required params)?
    
    Returns:
        0.0 if prerequisites fail or required args missing
        1.0 if all required arguments are present
    """
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"] or parsed["tools"] is None or parsed["gold"] is None:
            rewards.append(0.0)
            continue
        
        tindex = tool_index(parsed["tools"])
        gold = parsed["gold"]
        tool_spec = tindex.get(gold["name"])
        
        if tool_spec is None:
            rewards.append(0.0)
            continue
        
        schema = tool_spec.get("parameters")
        if validate_args_against_schema(gold["arguments"], schema):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def reward_gold_values_valid(prompts, completions, **kwargs) -> List[float]:
    rewards: List[float] = []

    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)

        if not parsed["valid"] or parsed["gold"] is None or parsed["fields"] is None:
            rewards.append(0.0)
            continue

        q_norm = _norm_text(parsed["fields"]["question"])
        args = parsed["gold"].get("arguments", {})

        if not args:
            rewards.append(1.0)
            continue

        ok = True
        for _, val in args.items():
            if isinstance(val, bool) or val is None:
                continue

            variants = get_grounding_variants(val)
            found = False

            for v in variants:
                if re.search(rf"\b{re.escape(v)}\b", q_norm):
                    found = True
                    break

            if not found:
                ok = False
                break

        rewards.append(1.0 if ok else 0.0)

    return rewards



def reward_difficulty_raw(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 6a: Raw difficulty banding (WITHOUT consistency multiplier).
    
    Peaks at P_TARGET (0.5 by default), penalizes too easy or too hard.
    
    Returns:
        0.0 if prerequisites fail or task is unsolvable
        0.0-1.0 based on how close p_success is to P_TARGET
    """
    base_url = kwargs.get("base_url", os.environ.get("SOLVER_BASE_URL", "http://localhost:5000/v1"))
    solver_system_prompt = kwargs["solver_prompt"][0]
    solver_model = kwargs.get("solver_model", "solver")
    
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"] or parsed["tools"] is None or parsed["gold"] is None:
            rewards.append(0.0)
            continue
        
        fields = parsed["fields"]
        question_text = fields["question"]
        tools_text = fields["available_tools"]
        
        if question_text.strip() == "":
            rewards.append(0.0)
            continue
        
        user_msg = build_solver_user_message(question_text, tools_text)
        solver_calls = solver_sample_tool_calls_cached(
            base_url=base_url,
            model=solver_model,
            solver_system_prompt=solver_system_prompt,
            user_message=user_msg,
            gold=parsed["gold"],
            k=K_SOLVER_SAMPLES,
        )
        
        p_success = success_prob_against_gold(solver_calls, parsed["gold"])
        
        if p_success < (1.0 / K_SOLVER_SAMPLES):
            rewards.append(0.0)
            continue
        
        diff_core = difficulty_reward_bandpass(
            p_success,
            p_low=P_LOW,
            p_high=P_HIGH,
            k=K_SOLVER_SAMPLES,
            sigma=0.12,
        )
        
        rewards.append(float(max(0.0, min(1.0, diff_core))))
        
    return rewards


def reward_consistency(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 6b: Solver consistency score.
    
    Measures how consistent the solver's outputs are across K samples.
    High consistency = unambiguous task, low consistency = ambiguous task.
    
    Returns:
        0.0 if prerequisites fail or all solver outputs are None
        0.0-1.0 based on mode frequency / total samples
    """
    base_url = kwargs.get("base_url", os.environ.get("SOLVER_BASE_URL", "http://localhost:5000/v1"))
    solver_system_prompt = kwargs["solver_prompt"][0]
    solver_model = kwargs.get("solver_model", "solver")
    
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"] or parsed["tools"] is None or parsed["gold"] is None:
            rewards.append(0.0)
            continue
        
        fields = parsed["fields"]
        question_text = fields["question"]
        tools_text = fields["available_tools"]
        
        if question_text.strip() == "":
            rewards.append(0.0)
            continue
        
        user_msg = build_solver_user_message(question_text, tools_text)
        solver_calls = solver_sample_tool_calls_cached(
            base_url=base_url,
            model=solver_model,
            solver_system_prompt=solver_system_prompt,
            user_message=user_msg,
            gold=parsed["gold"],
            k=K_SOLVER_SAMPLES,
        )
        
        consis = consistency_score(solver_calls)
        rewards.append(float(max(0.0, min(1.0, consis))))
    
    return rewards


LLM_JUDGE_SYSTEM_PROMPT = """You are a strict quality control judge for a synthetic data generation pipeline.
You will be given a User Question, Available Tools, and a Tool Call Answer.

Your job is to score the example on a scale of 1 to 5 based on TWO criteria:
1. **Question Quality**: Is the user question realistic, specific, and clear? (CRITICAL)
2. **Semantic Coherence**: Does the tool call actually solve the user's request?

Scoring Rubric:
- 5 (Perfect): The question is specific and realistic (e.g., "Book a flight to Paris on Dec 5th"). The tool call perfectly addresses it.
- 4 (Good): The question is good, but the tool call has minor issues (e.g., slightly different parameter values that still work).
- 3 (Passable): The question is vague or simple. The tool call matches it.
- 2 (Bad Question): The question is generic, placeholder text (e.g., "User request here", "Make a tool call"), or nonsense. **Score 2 or 1 immediately if the question is bad.**
- 1 (Failure): The tool call is completely unrelated, OR the question is clearly a template error (e.g., "A single concrete user request").

**IMPORTANT:** If the User Question looks like an instruction (e.g., "Generate a query...") rather than a natural user request, you MUST give a score of 1 or 2.

Reply with ONLY a single integer from 1 to 5."""

LLM_JUDGE_USER_TEMPLATE = """<question>
{question}
</question>

<available_tools>
{tools}
</available_tools>

<tool_call_answer>
{answer}
</tool_call_answer>

Score (1-5):"""

def _extract_score_from_response(response: str) -> Optional[int]:
    """
    Robust score extraction with multiple fallback strategies.
    Returns integer 1-5 or None if extraction fails.
    """
    if not response:
        return None
    
    response = response.strip()
    
    if response in ("1", "2", "3", "4", "5"):
        return int(response)
    
    if response and response[0].isdigit():
        digit = int(response[0])
        if 1 <= digit <= 5:
            return digit
    
    for char in response:
        if char in "12345":
            return int(char)
    
    spelled = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "1/5": 1, "2/5": 2, "3/5": 3, "4/5": 4, "5/5": 5,
    }
    response_lower = response.lower()
    for word, val in spelled.items():
        if word in response_lower:
            return val
    
    import re
    patterns = [
        r"score[:\s]+([1-5])",
        r"rating[:\s]+([1-5])",
        r"\b([1-5])\s*(?:/\s*5|out of 5)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            return int(match.group(1))
    
    return None


def _call_llm_judge_single(
    client: OpenAI,
    model: str,
    question: str,
    tools: str,
    answer: str,
    max_retries: int = 3,
    max_tokens: int = 16,
) -> int:
    """
    Call LLM judge with retries. Returns score 1-5, defaults to 3 on failure.
    """
    user_prompt = LLM_JUDGE_USER_TEMPLATE.format(
        question=question,
        tools=tools,
        answer=answer,
    )
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            
            content = resp.choices[0].message.content or ""
            score = _extract_score_from_response(content)
            
            if score is not None:
                return score
            
            if attempt < max_retries - 1:
                user_prompt = f"{user_prompt}\n\nIMPORTANT: Reply with ONLY a single digit 1-5, nothing else."
                
        except Exception as e:
            if attempt < max_retries - 1:
                continue
    
    return 3


class LLMJudgeCache:
    """
    Cache LLM judge scores to avoid redundant calls.
    """
    _cache: Dict[str, int] = {}
    
    @classmethod
    def get_key(cls, question: str, tools: str, answer: str) -> str:
        return f"{hash(question)}_{hash(tools)}_{hash(answer)}"
    
    @classmethod
    def get_or_call(
        cls,
        client: OpenAI,
        model: str,
        question: str,
        tools: str,
        answer: str,
        max_retries: int = 3,
    ) -> int:
        key = cls.get_key(question, tools, answer)
        
        if key not in cls._cache:
            cls._cache[key] = _call_llm_judge_single(
                client=client,
                model=model,
                question=question,
                tools=tools,
                answer=answer,
                max_retries=max_retries,
            )
        
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        cls._cache.clear()


def reward_semantic_coherence(prompts, completions, **kwargs) -> List[float]:
    """
    Reward 7: LLM judge for semantic coherence between question and gold answer.
    
    Uses the solver model as a judge to evaluate whether the tool call
    actually addresses the user's question.
    
    Scoring: 1-5 from LLM, normalized to 0-1:
        1 -> 0.0
        2 -> 0.25
        3 -> 0.5
        4 -> 0.75
        5 -> 1.0
    
    Returns:
        0.0 if prerequisites fail
        0.0-1.0 based on LLM judge score
    """
    base_url = kwargs.get("base_url", os.environ.get("SOLVER_BASE_URL", "http://localhost:5000/v1"))
    judge_model = kwargs.get("judge_model", kwargs.get("solver_model", "solver"))
    
    client = get_client(base_url)
    rewards: List[float] = []
    
    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        
        if not parsed["valid"] or parsed["tools"] is None or parsed["gold"] is None:
            rewards.append(0.0)
            continue
        
        fields = parsed["fields"]
        question_text = fields["question"]
        tools_text = fields["available_tools"]
        answer_text = fields["tool_call_answer"]
        
        if question_text.strip() == "":
            rewards.append(0.0)
            continue
        
        score = LLMJudgeCache.get_or_call(
            client=client,
            model=judge_model,
            question=question_text,
            tools=tools_text,
            answer=answer_text,
            max_retries=3,
        )
        
        normalized = (score - 1) / 4.0
        rewards.append(float(max(0.0, min(1.0, normalized))))
    
    return rewards


from collections import deque, defaultdict

def _norm_text_rep(s: str) -> str:
    """Normalize question text specifically for repetition detection."""
    s = s.lower().strip()
    s = s.replace("available_tools", "").replace("tool_call_answer", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\?\!\-\:\;\'\"]+", "", s)
    return s

def _char_ngrams(s: str, n: int = NGRAM_N) -> set:
    """Generate a set of character n-grams from normalized text."""
    s = f" {_norm_text_rep(s)} "
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def _jaccard(a: set, b: set) -> float:
    """Compute the Jaccard similarity between two n-gram sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni > 0 else 0.0

def _distance_question(q_i: str, q_j: str) -> float:
    """Compute a distance between two questions based on character n-grams."""
    A = _char_ngrams(_norm_text_rep(q_i))
    B = _char_ngrams(_norm_text_rep(q_j))
    sim = _jaccard(A, B)
    return 1.0 - sim

def _clusters_from_pairwise(dist_fn, items: List[str], tau: float) -> List[List[int]]:
    """Cluster items using a distance function and a threshold."""
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i+1, n):
            if dist_fn(items[i], items[j]) < tau:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())

def reward_repetition_penalty(prompts, completions, **kwargs) -> List[float]:
    """
    Repetition penalty (negative reward).
    Cluster similar questions inside the current batch; penalize by cluster size.

    Returns per-sample penalty in [-LAMBDA_REP, 0].
    """
    questions: List[str] = []
    valids: List[bool] = []

    for completion in completions:
        comp = completion[0]["content"]
        parsed = CompletionCache.get_or_parse(comp)
        if not parsed["valid"] or parsed["fields"] is None:
            questions.append("")
            valids.append(False)
            continue
        q = parsed["fields"]["question"]
        if q is None or q.strip() == "":
            questions.append("")
            valids.append(False)
            continue
        questions.append(q)
        valids.append(True)

    B = len(questions)
    if B == 0:
        return []

    idx_map = [i for i, ok in enumerate(valids) if ok]
    if len(idx_map) <= 1:
        return [0.0] * B

    valid_questions = [questions[i] for i in idx_map]
    clusters_local = _clusters_from_pairwise(_distance_question, valid_questions, TAU_REP)

    cluster_size_global = [1] * B
    for cl in clusters_local:
        size = len(cl)
        for local_i in cl:
            global_i = idx_map[local_i]
            cluster_size_global[global_i] = size

    penalties = [0.0] * B
    for i in range(B):
        if not valids[i]:
            penalties[i] = 0.0
        else:
            penalties[i] = -float(LAMBDA_REP * (cluster_size_global[i] / B))

    return penalties


def clear_all_caches():
    """Clear all caches between batches."""
    CompletionCache.clear()
    SolverCache.clear()
    LLMJudgeCache.clear()


def clear_completion_cache():
    """Clear the in-memory cache of parsed completion fields."""
    CompletionCache.clear()


def clear_solver_cache():
    """Clear the in-memory cache of solver tool-call samples."""
    SolverCache.clear()


def clear_judge_cache():
    """Clear the in-memory cache of LLM judge scores."""
    LLMJudgeCache.clear()


def reward_format_accuracy(prompts, completions, **kwargs) -> List[float]:
    """Aggregate format-related rewards into a single scalar score."""
    r1 = reward_tags_parsed(prompts, completions, **kwargs)
    r2 = reward_tools_valid_json(prompts, completions, **kwargs)
    r3 = reward_gold_valid_json(prompts, completions, **kwargs)

    a1 = float(kwargs.get("alpha_tags_parsed", ALPHA_TAGS_PARSED))
    a2 = float(kwargs.get("alpha_tools_valid_json", ALPHA_TOOLS_VALID_JSON))
    a3 = float(kwargs.get("alpha_gold_valid_json", ALPHA_GOLD_VALID_JSON))

    out = [a1*x + a2*y + a3*z for x, y, z in zip(r1, r2, r3)]
    gate = _question_gate_mask(completions)
    return [o*g for o, g in zip(out, gate)]


def reward_validity_accuracy(prompts, completions, **kwargs) -> List[float]:
    """Aggregate validity-related rewards into a single scalar score."""
    r1 = reward_gold_tool_exists(prompts, completions, **kwargs)
    r2 = reward_gold_args_valid(prompts, completions, **kwargs)
    r3 = reward_gold_values_valid(prompts, completions, **kwargs)

    a1 = float(kwargs.get("alpha_gold_tool_exists", ALPHA_GOLD_TOOL_EXISTS))
    a2 = float(kwargs.get("alpha_gold_args_valid", ALPHA_GOLD_ARGS_VALID))
    a3 = float(kwargs.get("alpha_gold_values_valid", ALPHA_GOLD_VALUES_VALID))

    out = [a1*x + a2*y + a3*z for x, y, z in zip(r1, r2, r3)]
    gate = _question_gate_mask(completions)
    return [o*g for o, g in zip(out, gate)]


def reward_curriculum(prompts, completions, **kwargs) -> List[float]:
    """Aggregate curriculum-related rewards into a single scalar score."""
    r1 = reward_difficulty_raw(prompts, completions, **kwargs)
    r2 = reward_semantic_coherence(prompts, completions, **kwargs)

    a1 = float(kwargs.get("alpha_difficulty_raw", ALPHA_DIFFICULTY_RAW))
    a2 = float(kwargs.get("alpha_semantic_coherence", ALPHA_SEMANTIC_COHERENCE))

    out = [a1*x + a2*y for x, y in zip(r1, r2)]
    gate = _question_gate_mask(completions)
    return [o*g for o, g in zip(out, gate)]
