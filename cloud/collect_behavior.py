"""Run the submission's budgeted sampler on official train cases (runtime-mirror venv, eval-identical GPU).

One model load per config (the eval reloads per case; only timing differs, features do not).
Each case gets the same sampling window as in the submission. Output: runs/behavior/<safe_cfg>.jsonl
Usage: python cloud/collect_behavior.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B [--efforts low,medium,high]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aimo_robust.modeling import load_model, load_tokenizer  # noqa: E402
from aimo_robust.predict import SAMPLING  # noqa: E402
from aimo_robust.sampler import sample, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--efforts", default="default")
ap.add_argument("--window", type=float, default=0.0, help="sampling seconds per case (0 = submission default)")
ap.add_argument("--max-new", type=int, default=8192,
                help="collection cap (>= deployed cap; smaller deployed caps are simulated by re-censoring)")
ap.add_argument("--pool", default="", help="optional jsonl of extra problems (pid, problem) to also run")
args = ap.parse_args()

tr = pd.read_parquet(ROOT / "data/raw/train-main-v2/data.parquet")
tok = load_tokenizer(args.model, local_files_only=False)
model = load_model(args.model, local_files_only=False)
k, _, window = SAMPLING[args.model]
window = args.window or window
max_new = args.max_new
out_dir = ROOT / "runs/behavior"
out_dir.mkdir(parents=True, exist_ok=True)

for effort in args.efforts.split(","):
    rows = tr[(tr.model_id == args.model) & (tr.reasoning_effort == effort)]
    probs = [(None, p) for p in rows.problem]
    if args.pool:
        probs += [(r["pid"], r["problem"]) for r in map(json.loads, open(args.pool, encoding="utf-8"))]
    out = out_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.model}__{effort}") + ".jsonl")
    done = {json.loads(l)["problem"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    for pid, problem in probs:
        if problem in done:
            continue
        t0 = time.time()
        # eval timing: sampling until t_stop, then forced answers for up to 15 s (BUDGET-25 .. BUDGET-10)
        res = sample(model, tok, args.model, problem, effort, k, max_new, t0 + window, force_until=t0 + window + 15)
        rec = {"model": args.model, "effort": effort, "pid": pid, "problem": problem, **summarize(res),
               "lens": [r["len"] for r in res], "fin": [r["finished"] for r in res], "ans": [r["ans"] for r in res],
               "forced": [r["forced"] for r in res],
               "seconds": round(time.time() - t0, 1), "k_cfg": k, "max_new": max_new}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{effort} agree={rec['agree']:.2f} all={rec['agree_all']:.2f} finish={rec['finish']:.2f} k={rec['k']} {rec['seconds']}s", flush=True)
