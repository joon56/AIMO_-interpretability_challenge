"""Per-config fusion of the two white-box length estimates, learned on the label-free problem pool, then one
calibrated robustness decision shared across configs.

For config c: fused = a0 + a1 * z_len_raw + a2 * len_mix_raw (OLS on pool out-of-fold predictions vs the measured
mean log length); len_fused = fused - median_pool(fused). Decision: class-balanced logistic on [len_fused] fitted on
official train rows. Writes fuse_* and comb_* into artifacts.
Usage: python experiments/fuse.py --report runs/v1/report --art runs/v1/art_prog --out runs/v1/art_fused
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvutil import grouped_cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--art", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--C", type=float, default=1.0)
args = ap.parse_args()
R = Path(args.report)


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


probe = pd.read_csv(R / "pool_probe_oof.csv")
probe = probe[probe.target == "len"].rename(columns={"oof": "z_raw", "y": "mlog"})
prog = pd.read_csv(R / "pool_prog_oof.csv")[["model_id", "reasoning_effort", "pid", "len_mix_oof"]]
pool = probe.merge(prog, on=["model_id", "reasoning_effort", "pid"], how="left")
rows = pd.read_csv(R / "train_rows_z.csv").merge(pd.read_csv(R / "train_rows_prog.csv"),
                                                on=["model_id", "reasoning_effort", "problem"], how="left")
arts = {f.stem: dict(np.load(f, allow_pickle=False)) for f in Path(args.art).glob("*.npz")}

fuse = {}
for (m, e), g in pool.groupby(["model_id", "reasoning_effort"]):
    art = arts[safe(m, e)]
    both = g.dropna(subset=["len_mix_oof"])
    X1 = np.c_[np.ones(len(g)), g.z_raw]
    a_z = np.linalg.lstsq(X1, g.mlog, rcond=None)[0]  # z-only fusion (fallback when no progress estimate)
    if len(both) >= 20 and "prog_w" in art:
        X2 = np.c_[np.ones(len(both)), both.z_raw, both.len_mix_oof]
        a = np.linalg.lstsq(X2, both.mlog, rcond=None)[0]
    else:
        a = np.r_[a_z, 0.0]
    fused = np.where(g.len_mix_oof.notna(), a[0] + a[1] * g.z_raw + a[2] * g.len_mix_oof.fillna(0),
                     a_z[0] + a_z[1] * g.z_raw)
    center = float(np.median(fused))
    rho = pd.Series(fused).corr(pd.Series(g.mlog.values), method="spearman")
    fuse[(m, e)] = (a, a_z, center)
    print(f"{m.split('/')[1][:22]:22s} {e:7s} pool n={len(g)} (prog {len(both)}): fused = {a[0]:.2f} + {a[1]:.2f} z "
          f"+ {a[2]:.2f} prog | rho(fused, mlog) {rho:.3f}")


def fused_rel(r):
    a, a_z, center = fuse[(r.model_id, r.reasoning_effort)]
    art = arts[safe(r.model_id, r.reasoning_effort)]
    z_raw = r.z_len + float(art["center_len"])
    if pd.notna(r.len_prog) and a[2] != 0.0:
        return a[0] + a[1] * z_raw + a[2] * (r.len_prog + float(art["prog_center"])) - center
    return a_z[0] + a_z[1] * z_raw - center


rows = rows[rows.apply(lambda r: (r.model_id, r.reasoning_effort) in fuse, axis=1)].copy()
rows["len_fused"] = rows.apply(fused_rel, axis=1)
y = rows.y.values
for cols in (["len_fused"], ["z_len"], ["z_len", "len_prog"]):
    d = rows.dropna(subset=cols)
    auc, bacc, acc, prob = grouped_cv(d[cols].values, d.y.values, d.grp.values, C=args.C)
    per = " ".join(f"{m.split('/')[-1][:6]}-{e[:3]}:{roc_auc_score(g.y, g.p):.2f}"
                   for (m, e), g in d.assign(p=prob).groupby(["model_id", "reasoning_effort"]) if g.y.nunique() == 2)
    print(f"{'+'.join(cols):16s} n={len(d)} AUC {auc:.3f} balanced {bacc:.3f} acc {acc:.3f} | {per}")
rows.to_csv(R / "train_rows_fused.csv", index=False)

lr = LogisticRegression(C=args.C, class_weight="balanced").fit(rows[["len_fused"]], y)
print("decision: logit =", round(float(lr.intercept_[0]), 3), "+", round(float(lr.coef_[0, 0]), 3), "* len_fused")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for (m, e), (a, a_z, center) in fuse.items():
    data = arts[safe(m, e)]
    data.update(fuse_a=np.array(a, dtype=np.float64), fuse_center=np.array(center),
                comb_feats=np.array(["len_fused"]), comb_w=lr.coef_[0].astype(np.float64), comb_b=np.array(lr.intercept_[0]))
    np.savez(out / f"{safe(m, e)}.npz", **data)
print("wrote", len(fuse), "artifacts to", out)
