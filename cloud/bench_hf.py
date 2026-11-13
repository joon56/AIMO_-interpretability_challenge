"""Cold-start + sampling throughput of HF generate in the runtime-mirror venv. Run once per model in a fresh process.

Prints one JSON line per measurement.
"""

import time

T_START = time.time()
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aimo_robust.modeling import load_model, load_tokenizer  # noqa: E402
from aimo_robust.prompting import build_prompt  # noqa: E402

T_IMPORT = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--effort", default="default")
ap.add_argument("--batches", default="1,16,32,64")
ap.add_argument("--new-tokens", type=int, default=1024)
ap.add_argument("--static", action="store_true", help="cache_implementation=static")
args = ap.parse_args()

PROBLEM = ("Find the number of ordered pairs of positive integers $(a, b)$ with $a < b$ and $b$ at most 50 "
           "such that $a + b$ divides $ab$.")  # any prompt works: min_new_tokens below forces every decode step


def emit(**kw):
    print(json.dumps({"model": args.model, **kw}), flush=True)


tok = load_tokenizer(args.model, local_files_only=False)
model = load_model(args.model, local_files_only=False)
torch.cuda.synchronize()
T_LOAD = time.time()
emit(stage="cold", import_s=round(T_IMPORT - T_START, 1), load_s=round(T_LOAD - T_IMPORT, 1),
     mem_gb=round(torch.cuda.memory_allocated() / 2**30, 1), tokenizer=type(tok).__name__)

prompt = build_prompt(tok, args.model, PROBLEM, args.effort)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
for bs in [int(b) for b in args.batches.split(",")]:
    torch.cuda.synchronize()
    t = time.time()
    kw = {"cache_implementation": "static"} if args.static else {}
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=40, top_p=0.95,
                         max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                         num_return_sequences=bs, pad_token_id=tok.pad_token_id, **kw)
    torch.cuda.synchronize()
    dt = time.time() - t
    gen = (out.shape[1] - enc["input_ids"].shape[1]) * bs
    emit(stage="gen", batch=bs, new_tokens=args.new_tokens, seconds=round(dt, 1), tok_s=round(gen / dt),
         step_ms=round(1000 * dt / args.new_tokens, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    del out
    torch.cuda.empty_cache()
