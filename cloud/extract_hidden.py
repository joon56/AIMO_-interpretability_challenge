"""Prompt hidden states for every (config, problem): last prompt token and problem-span mean, all layers.

Runs in the runtime-mirror venv (transformers 5.13) with the exact prompt the submission uses.
Output: <out>/<safe_cfg>.npz with pids, last [N, L+1, d] fp16, mean [N, L+1, d] fp16.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aimo_robust.modeling import load_model, load_tokenizer  # noqa: E402
from aimo_robust.prompting import build_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--efforts", default="default", help="comma list; gpt-oss: low,medium,high")
ap.add_argument("--problems", default=str(Path(__file__).resolve().parents[1] / "data/pool/problems.jsonl"))
ap.add_argument("--out", default="hidden")
args = ap.parse_args()

problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
t0 = time.time()
tok = load_tokenizer(args.model, local_files_only=False)
model = load_model(args.model, local_files_only=False)
print(f"loaded {args.model} in {time.time() - t0:.1f}s; tokenizer {type(tok).__name__}", flush=True)

for effort in args.efforts.split(","):
    lasts, means = [], []
    for p in problems:
        prompt = build_prompt(tok, args.model, p["problem"], effort)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        start = prompt.find(p["problem"])
        end = start + len(p["problem"])
        span = ((offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])).nonzero()
        lo, hi = (int(span.min()), int(span.max()) + 1) if len(span) else (0, offsets.shape[0])
        with torch.inference_mode():
            out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states)[:, 0]  # [L+1, T, d]
        lasts.append(hs[:, -1].float().cpu().numpy().astype(np.float16))
        means.append(hs[:, lo:hi].float().mean(1).cpu().numpy().astype(np.float16))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.model}__{effort}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.out) / f"{safe}.npz", pids=np.array([p["pid"] for p in problems]),
             last=np.stack(lasts), mean=np.stack(means))
    print(f"{safe}: {len(problems)} problems, {time.time() - t0:.0f}s", flush=True)
