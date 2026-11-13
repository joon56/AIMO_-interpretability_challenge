"""Parse our per-case "[aimo]" stderr lines from an evaluation log (Codabench / queue) into a table.

Usage: python experiments/parse_eval_log.py <log file> [<log file> ...]
Reports per case: model, effort, features, logit, decision, elapsed; plus sampling failures and per-config
robust-prediction rates (a skewed rate on a balanced val set points to a threshold/center shift).
"""

import re
import sys

import pandas as pd

rows, fails = [], 0
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8", errors="replace").read().replace("\\n", "\n")
    fails += len(re.findall(r"\[aimo\] sampling failed", text))
    for m in re.finditer(r"\[aimo\] (\S+) (\S+) ((?:\w+=-?[\d.]+(?:e-?\d+)? ?)+)t=([\d.]+)s", text):
        feats = dict(kv.split("=") for kv in m.group(3).split())
        rows.append({"model": m.group(1).split("/")[-1], "effort": m.group(2), **{k: float(v) for k, v in feats.items()},
                     "t": float(m.group(4))})
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no [aimo] lines found")
df["robust"] = df.logit >= 0
pd.set_option("display.width", 220)
cols = [c for c in ["model", "effort", "z_len", "len_prog", "len_fused", "finish", "logit", "robust", "t"] if c in df]
print(df[cols].round(3).to_string(index=False))
print(f"\ncases {len(df)}, sampling failures {fails}, predicted robust {df.robust.sum()}/{len(df)}, "
      f"time mean {df.t.mean():.0f}s max {df.t.max():.0f}s")
print(df.groupby(["model", "effort"]).agg(n=("robust", "size"), robust=("robust", "sum"),
                                          len_fused=("len_fused", "mean") if "len_fused" in df else ("logit", "mean")).to_string())
