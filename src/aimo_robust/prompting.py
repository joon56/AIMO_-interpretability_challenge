"""Prompt construction, forced-answer suffixes and answer parsing per model family."""

from __future__ import annotations

import math
import re

# Same instruction the organizers used when sampling the labels (baselines/prompts/solve.txt).
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, then output "
    "your final answer on the last line in the exact format:\nANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


def is_gpt_oss(model_id: str) -> bool:
    return "gpt-oss" in model_id


def build_prompt(tokenizer, model_id: str, problem: str, reasoning_effort: str) -> str:
    """Chat-formatted prompt string ending at the start of the assistant turn."""
    messages = [
        {"role": "system", "content": SOLVE_PROMPT},
        {"role": "user", "content": problem},
    ]
    kwargs = {}
    if is_gpt_oss(model_id):
        kwargs["reasoning_effort"] = reasoning_effort if reasoning_effort in ("low", "medium", "high") else "medium"
    else:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        # Templates without a system role: fold the instruction into the user turn.
        messages = [{"role": "user", "content": SOLVE_PROMPT + "\n\n" + problem}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def force_answer_suffix(model_id: str, partial: str) -> str:
    """Text appended to a truncated generation so the next tokens are the final answer."""
    if is_gpt_oss(model_id):
        return "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: "
    if "</think>" in partial:
        return "\nANSWER: "
    return "\n\nTime is up, I must commit to my final answer now.\n</think>\n\nANSWER: "


_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*\**\s*([^\n]*)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_INT_RE = re.compile(r"-?\d+")


def _last_boxed(text: str) -> str | None:
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i : j - 1] if depth == 0 else text[i:]


def normalize_answer(raw: str | None) -> str | None:
    """Canonical answer string: integer digits, 'nan', or None if unparseable."""
    if raw is None:
        return None
    s = raw.strip().strip("$*. ").replace(",", "").replace("\\!", "").replace("{", "").replace("}", "")
    s = s.replace("\\text", "").replace("\\mathrm", "").strip()
    low = s.lower()
    if low.startswith("nan") or "no solution" in low or low in ("none", "undefined"):
        return "nan"
    m = _INT_RE.search(s)
    if m:
        return str(int(m.group(0)))
    try:
        v = float(s)
        return "nan" if math.isnan(v) else str(round(v))
    except ValueError:
        return None


def parse_answer(text: str) -> str | None:
    """Final answer from a (possibly partial) generation; final-channel text is preferred."""
    if "<|channel|>final<|message|>" in text:
        text = text.rsplit("<|channel|>final<|message|>", 1)[1]
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    else:
        return None
    hits = _ANSWER_RE.findall(text)
    if hits:
        return normalize_answer(hits[-1])
    return normalize_answer(_last_boxed(text))
