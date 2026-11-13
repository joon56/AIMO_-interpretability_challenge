"""Compare white-box / behavioral / combined decision features on official train rows (repeated grouped CV),
then write comb_* fields into the artifacts.

Inputs: runs/report/train_rows_z.csv (out-of-fold, per-config-centered z_solve / z_len from train_probe.py),
        runs/behavior/*.jsonl (exact submission sampler output on train cases, cloud/collect_behavior.py),
        artifacts (for beh_sigma / beh_mu_center / beh_tau used by the behavioral length estimate).
Usage: python experiments/fit_combiner.py --art runs/artifacts --out runs/artifacts_comb [--use NAME]
"""

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aimo_robust.sampler import censored_mu  # noqa: E402
from cvutil import grouped_cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=str(ROOT / "runs/report/train_rows_z.csv"))
ap.add_argument("--behavior", default=str(ROOT / "runs/behavior"))
ap.add_argument("--art", default=str(ROOT / "runs/artifacts"))
ap.add_argument("--out", default=str(ROOT / "runs/artifacts_comb"))
ap.add_argument("--use", default="", help="feature set to export (default: best CV AUC among white-box sets)")
ap.add_argument("--C", type=float, default=1.0)
args = ap.parse_args()


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


def lg(a, k):
    return math.log((a * k + 0.5) / (k - a * k + 0.5))


arts = {}
for f in Path(args.art).glob("*.npz"):
    with np.load(f, allow_pickle=False) as d:
        arts[f.stem] = {k: d[k] for k in d.files}

rows = pd.read_csv(args.rows)
beh = pd.DataFrame([json.loads(l) for f in glob.glob(f"{args.behavior}/*.jsonl") for l in open(f, encoding="utf-8")])
beh = beh[beh.pid.isna()].rename(columns={"model": "model_id", "effort": "reasoning_effort"})
from aimo_robust.predict import SAMPLING  # noqa: E402


def recensor(lens, fin, cap):
    """Simulate the deployed token cap on samples collected with a larger cap."""
    return [min(n, cap) for n in lens], [f and n <= cap for n, f in zip(lens, fin)]


def finished_answers(ans, lens, fin, cap):
    return [a for a, n, f in zip(ans, lens, fin) if f and n <= cap and a is not None]


cap_of = {m: c for m, (_, c, _) in SAMPLING.items()}
rec = [recensor(L, F, cap_of[m]) for m, L, F in zip(beh.model_id, beh.lens, beh.fin)]
beh["len_beh"] = [
    censored_mu(L, F, float(a["beh_sigma"]), float(a["beh_mu_center"]), float(a["beh_tau"])) - float(a["beh_mu_center"])
    if (a := arts.get(safe(m, e))) is not None and len(L) else np.nan
    for m, e, (L, F) in zip(beh.model_id, beh.reasoning_effort, rec)]
beh["finish"] = [float(np.mean(F)) if len(F) else np.nan for L, F in rec]
beh["agree"] = [max([fa.count(x) for x in set(fa)] or [0]) / max(1, len(L))
                for fa, L in ((finished_answers(A, L0, F0, cap_of[m]), L0)
                              for m, A, L0, F0 in zip(beh.model_id, beh.ans, beh.lens, beh.fin))]
df = rows.merge(beh[["model_id", "reasoning_effort", "problem", "agree", "agree_all", "finish", "k", "len_beh"]],
                on=["model_id", "reasoning_effort", "problem"], how="inner").dropna(subset=["len_beh"])
prog_csv = Path(args.rows).with_name("train_rows_prog.csv")
if prog_csv.exists():
    df = df.merge(pd.read_csv(prog_csv), on=["model_id", "reasoning_effort", "problem"], how="left")
df["agree_logit"] = [lg(a, k) for a, k in zip(df.agree, df.k)]
print(f"rows with behavior: {len(df)} / {len(rows)}; robust rate {df.y.mean():.2f}")

P = ["z_solve", "z_len"]
FEATS = {"probe": P, "probe_len": ["z_len"],
         "beh_len": ["len_beh"], "beh_len_agree": ["len_beh", "agree_logit"],
         "comb_len": ["z_len", "len_beh"], "comb": P + ["len_beh"], "comb_agree": P + ["len_beh", "agree_logit"],
         "comb_len_agree": ["z_len", "len_beh", "agree_logit"]}
if "len_prog" in df and df.len_prog.notna().all():
    FEATS.update({"prog": ["len_prog"], "probe+prog": ["z_len", "len_prog"], "prog+beh": ["len_prog", "len_beh"],
                  "probe+prog+beh": ["z_len", "len_prog", "len_beh"]})
WHITE_BOX = {"probe", "probe_len", "comb_len", "comb", "comb_agree", "comb_len_agree", "prog", "probe+prog",
             "prog+beh", "probe+prog+beh"}
y = df.y.values
scores = {}
for name, cols in FEATS.items():
    auc, bacc, acc, prob = grouped_cv(df[cols].values, y, df.grp.values, C=args.C)
    scores[name] = auc
    per = " ".join(f"{m.split('/')[-1][:6]}-{e[:3]}:{roc_auc_score(d.y, d.p):.2f}"
                   for (m, e), d in df.assign(p=prob).groupby(["model_id", "reasoning_effort"]) if d.y.nunique() == 2)
    print(f"{name:14s} AUC {auc:.3f} balanced {bacc:.3f} acc {acc:.3f} | {per}")

use = args.use or max((n for n in scores if n in WHITE_BOX), key=lambda n: (round(scores[n], 2), -len(FEATS[n])))
cols = FEATS[use]
lr = LogisticRegression(C=args.C, class_weight="balanced").fit(df[cols], y)
print(f"export '{use}' (white-box={use in WHITE_BOX}):", dict(zip(cols, lr.coef_[0].round(3))), "b", round(lr.intercept_[0], 3))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
NEEDS = {"len_prog": "prog_w", "len_beh": "beh_sigma", "z_len": "w_len", "z_solve": "w"}
for stem, data in arts.items():
    data = dict(data)
    if any(NEEDS.get(c, c) not in data for c in cols if c in NEEDS):
        print(f"  {stem}: missing fields for {cols} -> keeps probe-only decision")
    elif any(c not in P for c in cols):
        data.update(comb_feats=np.array(cols), comb_w=lr.coef_[0].astype(np.float64), comb_b=np.array(lr.intercept_[0]))
    else:
        data.update(cal_feats=np.array(cols), cal_w=lr.coef_[0].astype(np.float64), cal_b=np.array(lr.intercept_[0]))
    np.savez(out / f"{stem}.npz", **data)
print("wrote", len(arts), "artifacts to", out)
