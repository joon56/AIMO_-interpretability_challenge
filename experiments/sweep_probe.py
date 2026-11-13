"""Exploratory sweep of probe hyperparameters on the problem pool (grouped 5-fold Spearman), per config.

Usage: python experiments/sweep_probe.py --gen runs/pod/runs/gen --hidden runs/pod/runs/hidden --target len
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--hidden", required=True)
ap.add_argument("--target", default="len", choices=["len", "solve"])
args = ap.parse_args()

problems = {p["pid"]: p for p in map(json.loads, open(ROOT / "data/pool/problems.jsonl", encoding="utf-8"))}
for hf in sorted(Path(args.hidden).glob("*.npz")):
    stem = hf.stem
    gfiles = [f for f in Path(args.gen).glob(f"{stem}*.jsonl") if ".texts." not in f.name]
    if not gfiles:
        continue
    g = {}
    for f in gfiles:
        g.update({r["pid"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    d = np.load(hf, allow_pickle=False)
    pids = list(d["pids"])
    idx = [i for i, p in enumerate(pids) if p in g and (args.target == "len" or problems[p]["answer"])]
    if args.target == "len":
        y = np.array([np.mean([np.log1p(s["len"]) for s in g[pids[i]]["samples"]]) for i in idx])
    else:
        y = np.array([np.mean([s["ans"] == problems[pids[i]]["answer"] and s["finish"] == "stop"
                               for s in g[pids[i]]["samples"]]) for i in idx])
    fold = np.array([hash(pids[i]) % 5 for i in idx])
    L = d["last"].shape[1]
    feats = {}
    for pool in ("last", "mean"):
        for layer in sorted({L // 3, L // 2, 2 * L // 3, 3 * L // 4, L - 2, L - 1}):
            feats[f"{pool}{layer}"] = d[pool][idx, layer].astype(np.float32)
    feats["last_mid+last_end"] = np.hstack([feats[f"last{L // 2}"], feats[f"last{L - 1}"]])
    feats["last+mean@3/4"] = np.hstack([feats[f"last{3 * L // 4}"], feats[f"mean{3 * L // 4}"]])
    res = []
    for name, X in feats.items():
        for k in (16, 32, 64):
            for alpha in (1.0, 10.0, 100.0):
                pred = np.zeros(len(y))
                for f in range(5):
                    tr, te = fold != f, fold == f
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
                    pca = PCA(n_components=min(k, tr.sum() - 1), random_state=0).fit((X[tr] - mu) / sd)
                    reg = Ridge(alpha=alpha).fit(pca.transform((X[tr] - mu) / sd), y[tr])
                    pred[te] = reg.predict(pca.transform((X[te] - mu) / sd))
                res.append((spearmanr(pred, y).correlation, name, k, alpha))
    res.sort(key=lambda r: -r[0])
    print(f"{stem}: n={len(y)} top5 {[(round(r[0], 3), r[1], r[2], r[3]) for r in res[:5]]}")
