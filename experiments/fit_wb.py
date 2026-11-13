"""White-box decision without behavioral features: logistic on per-config-centered [z_len, len_prog]
(prompt-state length probe + progress probe), fitted on official train rows; writes comb_* into artifacts
that have a progress probe, others keep the probe-only cal_* decision.
Usage: python experiments/fit_wb.py --report runs/v1pre/report --art runs/v1pre/art_prog --out runs/v1pre/art_wb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvutil import grouped_cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--art", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feats", default="z_len,len_prog")
ap.add_argument("--C", type=float, default=1.0)
args = ap.parse_args()

cols = args.feats.split(",")
z = pd.read_csv(Path(args.report) / "train_rows_z.csv")
p = pd.read_csv(Path(args.report) / "train_rows_prog.csv")
d = z.merge(p, on=["model_id", "reasoning_effort", "problem"], how="inner").dropna(subset=cols)
auc, bacc, acc, _ = grouped_cv(d[cols].values, d.y.values, d.grp.values, C=args.C)
print(f"{'+'.join(cols)}: n={len(d)} grouped CV AUC {auc:.3f} balanced {bacc:.3f} acc {acc:.3f}")
lr = LogisticRegression(C=args.C, class_weight="balanced").fit(d[cols], d.y)
print("weights", dict(zip(cols, lr.coef_[0].round(3))), "b", round(float(lr.intercept_[0]), 3))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in Path(args.art).glob("*.npz"):
    data = dict(np.load(f, allow_pickle=False))
    if "len_prog" in cols and "prog_w" not in data:
        print(f"  {f.stem}: no progress probe -> probe-only decision")
    else:
        data.update(comb_feats=np.array(cols), comb_w=lr.coef_[0].astype(np.float64), comb_b=np.array(lr.intercept_[0]))
    np.savez(out / f.name, **data)
print("wrote", len(list(out.glob("*.npz"))), "artifacts")
