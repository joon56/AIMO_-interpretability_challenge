"""Per-case robustness prediction: hidden-state probes (+ optional budgeted self-consistency).

Artifact layout (one .npz per config, no pickle). Probe blocks, suffix "" = solve-rate, "_len" = reasoning length:
    layers{s} int[1], pool{s} "last"|"mean", mu{s}, sd{s} float[d], pca{s} float[r, d], w{s} float[r], b{s}
    -> z_solve / z_len = pca @ ((h - mu) / sd) @ w + b
Decision:
    cal_feats str[m], cal_w float[m], cal_b        probe-only robust logit
    comb_feats str[m], comb_w float[m], comb_b      optional, used when present; features from
                                                    {z_solve, z_len, agree_logit, agree_all_logit, finish, len_beh,
                                                     len_prog}
    prog_*: progress probe (hidden state of a running sample at token cap prog_cap -> problem mean log length)
    fuse_a [a0, a1, a2], fuse_center: len_fused = a0 + a1 * z_len_raw + a2 * len_mix_raw - fuse_center
    center{s}: per-config pool median of the probe output; beh_mu_center / beh_tau / beh_sigma: pool median and spread
    of mean log length and within-problem std (censored log-normal MAP: len_beh = mu_hat - beh_mu_center)
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .modeling import load_model, load_tokenizer
from .prompting import build_prompt

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
T0 = time.time()
BUDGET_S = float(os.environ.get("AIMO_BUDGET_S", "140"))
# (K samples, max_new_tokens, sampling window s). Window also capped by BUDGET_S - 25 s after process start
# (then up to BUDGET_S - 10 s for the forced-answer pass). Token caps sized from eval-HW step times
# (queue run 434b7038 at batch 32, pod bench) so the cap, not the clock, normally binds.
SAMPLING = {
    "Qwen/Qwen3.5-4B": (32, 3072, 100),  # 26.6 ms/step @32 (linear-attn torch fallback scales with batch)
    "Skywork/Skywork-OR1-Math-7B": (64, 4096, 100),  # 23.5 ms/step @64
    "allenai/Olmo-3-7B-Think": (16, 2560, 100),  # 61 ms/step @32, attention-bound (32 KV heads)
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": (32, 3072, 100),  # 27.2 ms/step @32, 40.4 @64
    "openai/gpt-oss-120b": (40, 1536, 95),  # 50.9 ms/step @32, 78.9 @64 (pod bench); progress probe trained at 1536
}
BLOCKS = {"z_solve": "", "z_len": "_len"}


def safe_name(model_id: str, effort: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model_id}__{effort}")


def load_artifact(model_id: str, effort: str) -> dict | None:
    for eff in (effort, "default"):
        path = ARTIFACT_DIR / f"{safe_name(model_id, eff)}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
    return None


@torch.inference_mode()
def pooled_states(model, tok, model_id: str, problem: str, effort: str, wants: set[tuple[int, str]]) -> dict:
    """One forward pass over the chat prompt -> {(layer, pool): vector} for the requested pairs."""
    prompt = build_prompt(tok, model_id, problem, effort)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0]
    out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False)
    start = prompt.find(problem)
    end = start + len(problem)
    span = ((offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])).nonzero()
    lo, hi = (int(span.min()), int(span.max()) + 1) if len(span) else (0, offsets.shape[0])
    res = {}
    for layer, pool in wants:
        h = out.hidden_states[layer][0]
        res[(layer, pool)] = (h[-1] if pool == "last" else h[lo:hi].float().mean(0)).float().cpu().numpy()
    return res


def probe_z(art: dict, suf: str, vec: np.ndarray, prefix: str = "") -> float:
    x = (vec - art[f"{prefix}mu{suf}"]) / art[f"{prefix}sd{suf}"]
    return float(art[f"{prefix}pca{suf}"] @ x @ art[f"{prefix}w{suf}"] + art[f"{prefix}b{suf}"])


def agree_logit(agree: float, k: int) -> float:
    return math.log((agree * k + 0.5) / (k - agree * k + 0.5))


def predict_one(model_id: str, problem: str, effort: str) -> bool:
    art = load_artifact(model_id, effort)
    if art is None:
        raise RuntimeError(f"no artifact for {model_id} / {effort}")
    tok = load_tokenizer(model_id)
    model = load_model(model_id)
    blocks = {name: suf for name, suf in BLOCKS.items() if f"w{suf}" in art}
    wants = {(int(art[f"layers{suf}"][0]), str(art[f"pool{suf}"])) for suf in blocks.values()}
    states = pooled_states(model, tok, model_id, problem, effort, wants)
    # probe outputs relative to the config's pool median (val/test are balanced per model)
    f = {name: probe_z(art, suf, states[(int(art[f"layers{suf}"][0]), str(art[f"pool{suf}"]))])
         - float(art.get(f"center{suf}", 0.0)) for name, suf in blocks.items()}
    names, w, b = [str(n) for n in art["cal_feats"]], art["cal_w"], float(art["cal_b"])
    if "comb_w" in art and model_id in SAMPLING:
        try:
            from .sampler import censored_mu, sample, summarize

            k, max_new, window = SAMPLING[model_id]
            comb = [str(n) for n in art["comb_feats"]]
            prog = "prog_w" in art
            if prog:
                max_new = int(art["prog_cap"])  # the progress probe was trained on states at exactly this cap
            t_stop = min(time.time() + window, T0 + BUDGET_S - 25)
            force_until = T0 + BUDGET_S - 10 if "agree_all_logit" in comb else 0.0
            res = sample(model, tok, model_id, problem, effort, k, max_new, t_stop, force_until=force_until,
                         capture_layer=int(art["prog_layer"]) if prog else 0)
            s = summarize(res)
            if s["k"]:
                mu = censored_mu([r["len"] for r in res], [r["finished"] for r in res], float(art["beh_sigma"]),
                                 float(art["beh_mu_center"]), float(art["beh_tau"]))
                f.update(agree_logit=agree_logit(s["agree"], s["k"]), agree_all_logit=agree_logit(s["agree_all"], s["k"]),
                         finish=s["finish"], len_beh=mu - float(art["beh_mu_center"]))
                if prog:
                    vals = [math.log1p(r["len"]) if r["finished"] else probe_z(art, "", r["state"], prefix="prog_")
                            for r in res]
                    f["len_prog"] = float(np.mean(vals)) - float(art["prog_center"])
                    if "fuse_a" in art and "z_len" in f:  # per-config fusion of both length estimates (pool-fitted OLS)
                        a = art["fuse_a"]
                        f["len_fused"] = float(a[0] + a[1] * (f["z_len"] + float(art["center_len"]))
                                               + a[2] * (f["len_prog"] + float(art["prog_center"]))
                                               - float(art["fuse_center"]))
                names, w, b = comb, art["comb_w"], float(art["comb_b"])
        except Exception as exc:  # sampling/hook failure: keep the probe-only decision for this case
            print(f"[aimo] sampling failed: {exc!r}", file=sys.stderr, flush=True)
    if any(n not in f for n in names):  # combiner feature unavailable (e.g. sampling failed): probe-only decision
        names, w, b = [str(n) for n in art["cal_feats"]], art["cal_w"], float(art["cal_b"])
    logit = b + float(np.dot(w, [f[n] for n in names]))
    # one line per case to stderr (model, features, timing; never problem text) for eval-log diagnostics
    print(f"[aimo] {model_id} {effort} " + " ".join(f"{k}={v:.3f}" for k, v in f.items())
          + f" logit={logit:.3f} t={time.time() - T0:.1f}s", file=sys.stderr, flush=True)
    return bool(logit >= 0.0) if math.isfinite(logit) else True
