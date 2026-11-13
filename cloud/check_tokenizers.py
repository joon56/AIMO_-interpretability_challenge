"""Round-trip check of AutoTokenizer for every cached model in the current venv (vLLM uses AutoTokenizer)."""

import transformers
from transformers import AutoTokenizer

PROBE = "Hello world 12345 \\frac{1}{2} x^2 + y_1 = 0. Find $17_b$ divides $97_b$, answer 70.\n"
for m in ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "openai/gpt-oss-120b"]:
    tok = AutoTokenizer.from_pretrained(m)
    ids = tok(PROBE, add_special_tokens=False)["input_ids"]
    dec = tok.decode(ids, skip_special_tokens=False)
    print(transformers.__version__, m, type(tok).__name__, "roundtrip_ok" if dec == PROBE else f"MISMATCH: {dec!r}")
