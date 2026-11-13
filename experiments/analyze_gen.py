"""Decision checks on vLLM solve-rate data (D1 oracle, D2 truncation) — no hidden states needed.

Usage: python experiments/analyze_gen.py --gen runs/pod/runs/gen
Prints per config: problems, truncation rate, finish-length stats, solve-rate histogram, and on official
train rows the AUC of empirical solve rate / mean log length for the robust label.
"""

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs/pod/runs/gen"))
args = ap.parse_args()


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


problems = {p["pid"]: p for p in map(json.loads, open(ROOT / "data/pool/problems.jsonl", encoding="utf-8"))}
pid_of = {norm(p["problem"]): pid for pid, p in problems.items()}
gens = {}
for f in sorted(glob.glob(f"{args.gen}/*.jsonl")):
    if f.endswith(".texts.jsonl"):
        continue
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    if rows:  # shards (*.partN.jsonl) of one config are merged
        gens.setdefault((rows[0]["model"], rows[0]["effort"]), {}).update({r["pid"]: r for r in rows})

# pseudo truth for problems without a known answer: vote of finished answers over all configs
truth = {pid: p["answer"] for pid, p in problems.items()}
for pid in [p for p, a in truth.items() if a is None]:
    votes = Counter(s["ans"] for g in gens.values() for s in g.get(pid, {}).get("samples", [])
                    if s["ans"] is not None and s["finish"] == "stop")
    truth[pid] = votes.most_common(1)[0][0] if votes else None

tr = pd.read_parquet(ROOT / "data/raw/train-main-v2/data.parquet")
tr = tr[tr.is_robust.notna()].copy()
tr["pid"] = tr.problem.map(lambda s: pid_of[norm(s)])
tr["y"] = tr.is_robust.astype(bool).astype(int)

print(f"{'config':45s} {'n':>4s} {'trunc':>6s} {'len_med':>7s} {'solve0':>6s} {'solve1':>6s} {'mean':>5s} | train: n AUC(solve) AUC(-len)")
for (m, e), g in gens.items():
    s_all = [s for r in g.values() for s in r["samples"]]
    trunc = np.mean([s["finish"] == "length" for s in s_all])
    lens = [s["len"] for s in s_all if s["finish"] == "stop"]
    rate = {pid: np.mean([s["ans"] == truth[pid] and s["finish"] == "stop" for s in r["samples"]])
            for pid, r in g.items() if truth.get(pid) is not None}
    loglen = {pid: np.mean([np.log1p(s["len"]) for s in r["samples"]]) for pid, r in g.items()}
    v = np.array(list(rate.values()))
    sub = tr[(tr.model_id == m) & (tr.reasoning_effort == e) & tr.pid.isin(rate)]
    aucs = ""
    if sub.y.nunique() == 2:
        aucs = (f"{len(sub):3d} {roc_auc_score(sub.y, sub.pid.map(rate)):.3f} "
                f"{roc_auc_score(sub.y, -sub.pid.map(loglen)):.3f}")
    print(f"{m + ' ' + e:45s} {len(g):4d} {trunc:6.2f} {int(np.median(lens)) if lens else 0:7d} "
          f"{np.mean(v == 0):6.2f} {np.mean(v == 1):6.2f} {v.mean():5.2f} | {aucs}")

# pooled oracle over all configs present
tr["rate"] = [np.nan if (m, e) not in gens else
              np.mean([s["ans"] == truth[p] and s["finish"] == "stop" for s in gens[(m, e)].get(p, {}).get("samples", [])] or [np.nan])
              for m, e, p in zip(tr.model_id, tr.reasoning_effort, tr.pid)]
have = tr.rate.notna()
if have.sum() and tr[have].y.nunique() == 2:
    print(f"\n[D1] pooled oracle AUC(solve rate -> robust) on {have.sum()} train rows: {roc_auc_score(tr[have].y, tr[have].rate):.3f}")
    print(tr[have].groupby(pd.cut(tr[have].rate, [-0.01, 0.25, 0.5, 0.75, 0.99, 1.0]), observed=False).y.agg(["mean", "size"]).T)
