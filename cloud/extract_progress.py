"""Hidden states of partial reasoning traces ("progress states") for the progress probe.

For each stored vLLM sample of a config that is still running at the deployed token cap B, rebuild
prompt + first (B - 1) generated tokens, run one forward pass, and keep the last-position hidden state at a few
layers. This mirrors what the submission's sampler captures with a forward hook at its final decode step.
Output: runs/progress/<safe_cfg>__B<cap>.npz with pids, sample index, states [n, n_layers, d] fp16, layers.
Usage: python cloud/extract_progress.py --model Qwen/Qwen3.5-4B --efforts default --cap 3072
"""

import argparse
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aimo_robust.modeling import load_model, load_tokenizer, text_config  # noqa: E402
from aimo_robust.prompting import build_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--efforts", default="default")
ap.add_argument("--cap", type=int, required=True)
ap.add_argument("--batch", type=int, default=6)
args = ap.parse_args()

problems = {p["pid"]: p for p in map(json.loads, open(ROOT / "data/pool/problems.jsonl", encoding="utf-8"))}
tok = load_tokenizer(args.model, local_files_only=False)
model = load_model(args.model, local_files_only=False)
n_layers = text_config(model).num_hidden_layers
layers = sorted({n_layers // 2, 3 * n_layers // 4, n_layers})  # hidden_states indices
blocks = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.model.language_model.layers
captured = {}


def hook(idx):
    def fn(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[idx] = h[:, -1, :].float().cpu()
    return fn


for L in layers:  # hidden_states[L] is the output of decoder layer L-1
    blocks[L - 1].register_forward_hook(hook(L))

for effort in args.efforts.split(","):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.model}__{effort}")
    rows, texts = {}, defaultdict(list)
    for f in sorted((ROOT / "runs/gen").glob(f"{stem}*.jsonl")):
        if ".texts." in f.name:
            continue
        rows.update({r["pid"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
        tf = f.with_suffix(".texts.jsonl.gz")
        if tf.exists():
            for line in gzip.open(tf, "rt", encoding="utf-8"):
                t = json.loads(line)
                texts[t["pid"]].append(t["text"])
    items = []  # (pid, sample index, token ids of prompt + first cap-1 generated tokens)
    for pid, r in rows.items():
        prompt_ids = tok(build_prompt(tok, args.model, problems[pid]["problem"], effort), add_special_tokens=False)["input_ids"]
        for si, (s, text) in enumerate(zip(r["samples"], texts.get(pid, []))):
            if s["len"] <= args.cap and s["finish"] == "stop":
                continue  # finished before the cap: its length is observed directly at test time
            gen_ids = tok(text, add_special_tokens=False)["input_ids"][: args.cap - 1]
            items.append((pid, si, prompt_ids + gen_ids))
    t0 = time.time()
    states, order = [], []
    groups = defaultdict(list)  # equal-length sequences batch without padding (no position/recurrence artifacts)
    for k, (_, _, seq) in enumerate(items):
        groups[len(seq)].append(k)
    with torch.inference_mode():
        for width, ks in groups.items():
            for i in range(0, len(ks), args.batch):
                part = ks[i : i + args.batch]
                ids = torch.tensor([items[k][2] for k in part], device=model.device)
                model(input_ids=ids, use_cache=False)
                states.append(torch.stack([captured[L] for L in layers], 1).numpy().astype(np.float16))
                order.extend(part)
    items = [items[k] for k in order]
    out = ROOT / "runs/progress" / f"{stem}__B{args.cap}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pids=np.array([x[0] for x in items]), sidx=np.array([x[1] for x in items]),
             states=np.concatenate(states) if states else np.zeros((0, len(layers), 1), np.float16),
             layers=np.array(layers))
    print(f"{stem} cap {args.cap}: {len(items)} running samples, {time.time() - t0:.0f}s -> {out.name}", flush=True)
