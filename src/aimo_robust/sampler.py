"""Budgeted self-consistency: K parallel samples (label sampling settings) under a token cap and wall-clock deadline,
then an optional forced-answer pass that makes every unfinished sample commit to an answer."""

from __future__ import annotations

import math
import time
from collections import Counter
from statistics import NormalDist

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .prompting import build_prompt, force_answer_suffix, is_gpt_oss, normalize_answer, parse_answer


class Deadline(StoppingCriteria):
    def __init__(self, t_stop: float):
        self.t_stop = t_stop

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), time.time() >= self.t_stop, dtype=torch.bool, device=input_ids.device)


def stop_ids(model, tok, model_id: str) -> list[int]:
    ids = set()
    for v in (getattr(model.generation_config, "eos_token_id", None), tok.eos_token_id):
        if v is not None:
            ids.update([v] if isinstance(v, int) else v)
    if is_gpt_oss(model_id):
        vocab = tok.get_vocab()
        ids.update(vocab[t] for t in ("<|return|>", "<|call|>") if t in vocab)
    return sorted(ids)


@torch.inference_mode()
def force_answers(model, tok, model_id: str, prompt: str, partials: list[str], t_hard: float, chunk: int = 8) -> list:
    """Greedy-decode a short final answer after each truncated reasoning text; None where time ran out."""
    out = [None] * len(partials)
    for i in range(0, len(partials), chunk):
        if time.time() > t_hard:
            break
        texts = [prompt + p + force_answer_suffix(model_id, p) for p in partials[i : i + chunk]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=12, pad_token_id=tok.pad_token_id,
                             temperature=None, top_p=None, top_k=None)
        for j, row in enumerate(gen[:, enc["input_ids"].shape[1]:]):
            out[i + j] = normalize_answer(tok.decode(row, skip_special_tokens=True).split("\n")[0])
        del gen, enc
    return out


def decoder_layers(model):
    inner = model.model
    return inner.layers if hasattr(inner, "layers") else inner.language_model.layers


@torch.inference_mode()
def sample(model, tok, model_id: str, problem: str, effort: str, k: int, max_new: int, t_stop: float,
           seed: int = 0, force_until: float = 0.0, capture_layer: int = 0) -> list[dict]:
    """K samples; each -> {finished, len, ans, forced[, state]}. Halves K on OOM while time remains.
    force_until > 0: unfinished samples get a forced answer (until that wall-clock time).
    capture_layer > 0: each sample also carries hidden_states[capture_layer] at the final decode step
    (the state of its last generated token; meaningful for samples still running at the end)."""
    prompt = build_prompt(tok, model_id, problem, effort)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    eos = stop_ids(model, tok, model_id)
    eos_set = set(eos)
    last = {}
    handle = None
    if capture_layer:
        def keep(_module, _inp, out):
            last["h"] = (out[0] if isinstance(out, tuple) else out)[:, -1, :]
        handle = decoder_layers(model)[capture_layer - 1].register_forward_hook(keep)
    try:
        while True:
            try:
                torch.manual_seed(seed)
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=40, top_p=0.95,
                                     max_new_tokens=max_new, num_return_sequences=k, eos_token_id=eos,
                                     pad_token_id=tok.pad_token_id,
                                     stopping_criteria=StoppingCriteriaList([Deadline(t_stop)]))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                k //= 2
                if k < 2 or time.time() > t_stop - 20:
                    return []
    finally:
        if handle is not None:
            handle.remove()
    states = last["h"].float().cpu().numpy() if "h" in last else None
    rows = out[:, enc["input_ids"].shape[1]:].tolist()
    del out
    torch.cuda.empty_cache()
    res, partials = [], []
    for row in rows:
        cut = next((i for i, t in enumerate(row) if t in eos_set), None)
        kept = row if cut is None else row[:cut]
        text = tok.decode(kept, skip_special_tokens=False)
        res.append({"finished": cut is not None, "len": len(kept), "ans": parse_answer(text) if cut is not None else None,
                    "forced": None})
        if states is not None:
            res[-1]["state"] = states[len(res) - 1]
        if cut is None:
            partials.append((len(res) - 1, text))
    if force_until and partials:
        try:
            forced = force_answers(model, tok, model_id, prompt, [t for _, t in partials], force_until)
            for (i, _), a in zip(partials, forced):
                res[i]["forced"] = a
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return res


def summarize(res: list[dict]) -> dict:
    """agree: modal share among finished answers over all K (unfinished = disagree).
    agree_all: modal share when unfinished samples contribute their forced answer."""
    k = len(res)
    if k == 0:
        return {"k": 0, "finish": 0.0, "agree": 0.0, "agree_all": 0.0, "modal": None}
    fin = [r["ans"] for r in res if r["finished"] and r["ans"] is not None]
    modal, cnt = Counter(fin).most_common(1)[0] if fin else (None, 0)
    every = [r["ans"] if r["finished"] else r["forced"] for r in res]
    every = [a for a in every if a is not None]
    _, cnt_all = Counter(every).most_common(1)[0] if every else (None, 0)
    return {"k": k, "finish": sum(r["finished"] for r in res) / k, "agree": cnt / k, "agree_all": cnt_all / k,
            "modal": modal}


def censored_mu(lens: list[int], finished: list[bool], sigma: float, mu0: float, tau: float) -> float:
    """MAP estimate of a problem's mean log length: log-normal samples with known within-problem sigma,
    unfinished samples right-censored at their generated length, prior N(mu0, tau) from the model's pool."""
    nd = NormalDist()
    xs = [math.log1p(n) for n, f in zip(lens, finished) if f]
    cs = [math.log1p(n) for n, f in zip(lens, finished) if not f]
    best_mu, best_lp = mu0, -float("inf")
    for i in range(400, 1300):
        mu = i / 100
        lp = -0.5 * ((mu - mu0) / tau) ** 2 + sum(-0.5 * ((x - mu) / sigma) ** 2 for x in xs)
        lp += sum(math.log(max(1e-300, 1 - nd.cdf((c - mu) / sigma))) for c in cs)
        if lp > best_lp:
            best_mu, best_lp = mu, lp
    return best_mu
