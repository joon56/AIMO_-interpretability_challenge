"""Offline tokenizer/model loading shared by the submission and the dev scripts."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

_ROUNDTRIP = "Hello world 12345 \\frac{1}{2} x^2 + y_1 = 0.\n"


def _roundtrip_ok(tok) -> bool:
    try:
        ids = tok(_ROUNDTRIP, add_special_tokens=False)["input_ids"]
        return tok.decode(ids, skip_special_tokens=False) == _ROUNDTRIP
    except Exception:
        return False


def load_tokenizer(model_id: str, local_files_only: bool = True):
    """Prefer a tokenizer that round-trips text (AutoTokenizer mangles DeepSeek-R1-0528-Qwen3-8B in 5.13)."""
    candidates = []
    for loader in (AutoTokenizer, PreTrainedTokenizerFast):
        try:
            candidates.append(loader.from_pretrained(model_id, local_files_only=local_files_only))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"no tokenizer for {model_id}")
    ranked = sorted(candidates, key=lambda t: (not _roundtrip_ok(t), getattr(t, "chat_template", None) is None))
    tok = ranked[0]
    if getattr(tok, "chat_template", None) is None:
        tok.chat_template = next((t.chat_template for t in ranked[1:] if getattr(t, "chat_template", None)), None)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_id: str, local_files_only: bool = True):
    cuda = torch.cuda.is_available()
    kwargs = {"local_files_only": local_files_only, "dtype": torch.bfloat16 if cuda else torch.float32,
              "device_map": "cuda" if cuda else "cpu"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        # Qwen3.5 small checkpoints are registered as image-text-to-text models.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg
