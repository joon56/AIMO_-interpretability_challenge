"""Codabench leaderboard snapshot (public API): competitor summary + our own submissions.

Usage: python experiments/lb.py [--me <codabench username>]
Saves research/lb/<UTC timestamp>.json and prints per-track best-per-owner stats and our rows.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.codabench.org/api/phases/30038/get_leaderboard/?page_size=all"
TASKS = {36237: "Main", 36241: "Small"}

ap = argparse.ArgumentParser()
ap.add_argument("--me", default="")
args = ap.parse_args()

raw = subprocess.run(["curl", "-s", "-m", "120", URL], capture_output=True, text=True, encoding="utf-8").stdout
data = json.loads(raw)
out = ROOT / "research/lb" / (time.strftime("%Y%m%dT%H%M", time.gmtime()) + ".json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(raw, encoding="utf-8")

rows = [{"task": TASKS.get(sc["task_id"], sc["task_id"]), "acc": float(sc["score"]), "owner": s["owner"],
         "id": s["id"], "t": (s.get("created_when") or "")[:16]}
        for s in data["submissions"] for sc in s["scores"] if sc["column_key"] == "accuracy"]
df = pd.DataFrame(rows).drop_duplicates(["task", "id"])
pd.set_option("display.width", 200)
for task, g in df.groupby("task"):
    agg = g.groupby("owner").agg(n=("acc", "size"), best=("acc", "max"), mean=("acc", "mean")).sort_values(
        ["best", "mean"], ascending=False)
    print(f"\n== {task}: {len(g)} subs, {len(agg)} owners, latest {g.t.max()} | best-per-owner p50 "
          f"{agg.best.median():.2f} p90 {agg.best.quantile(.9):.2f} | mean-per-owner (>=3 subs) top: "
          f"{agg[agg.n >= 3]['mean'].nlargest(3).round(2).to_dict()}")
    print(agg.head(10).round(2).to_string())
    recent = g[g.t >= time.strftime("%Y-%m-%dT00:00", time.gmtime(time.time() - 86400))]
    if len(recent):
        print(f" last 24h: {len(recent)} subs, max {recent.acc.max():.2f}")
if args.me:
    mine = df[df.owner == args.me].sort_values("t")
    print(f"\n== ours ({args.me}):\n" + (mine.to_string(index=False) if len(mine) else " none yet"))
