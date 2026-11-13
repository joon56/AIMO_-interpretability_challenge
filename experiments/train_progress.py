"""Progress probe: hidden state of a still-running sample at the token cap -> problem mean log length.

len_mix(problem) = mean over samples of [log1p(len) if finished before the cap else probe(state)], centered by the
config's pool median. Computed out-of-fold (grouped by problem template) for pool and official train rows.
Inputs: runs/progress/<safe_cfg>__B<cap>.npz (cloud/extract_progress.py), gens, existing artifacts.
Outputs: prog_* fields added to artifacts in --out, report/train_rows_prog.csv.
"""

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs/pod/runs/gen"))
ap.add_argument("--progress", default=str(ROOT / "runs/pod/runs/progress"))
ap.add_argument("--art", default=str(ROOT / "runs/artifacts"))
ap.add_argument("--out", default=str(ROOT / "runs/artifacts_prog"))
ap.add_argument("--report", default=str(ROOT / "runs/report"))
ap.add_argument("--pca", type=int, default=32)
args = ap.parse_args()


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


problems = [json.loads(l) for l in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
pid_of = {norm(p["problem"]): p["pid"] for p in problems}
parent = {p["pid"]: p["pid"] for p in problems}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


texts = [(p["pid"], norm(p["problem"])) for p in problems]
for i, (a, ta) in enumerate(texts):
    for b, tb in texts[i + 1:]:
        if difflib.SequenceMatcher(None, ta[:200], tb[:200]).ratio() > 0.75:
            parent[find(a)] = find(b)
group = {pid: find(pid) for pid in parent}
tr = pd.read_parquet(ROOT / "data/raw/train-main-v2/data.parquet")
tr = tr[tr.is_robust.notna()].copy()
tr["pid"] = tr.problem.map(lambda s: pid_of[norm(s)])
tr["y"] = tr.is_robust.astype(bool).astype(int)


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    pca = PCA(n_components=min(args.pca, len(X) - 1), random_state=0).fit((X - mu) / sd)
    reg = Ridge(alpha=10.0).fit(pca.transform((X - mu) / sd), y)
    b = float(reg.intercept_ - pca.mean_ @ pca.components_.T @ reg.coef_)
    return {"mu": mu, "sd": sd, "pca": pca.components_.astype(np.float32), "w": reg.coef_.astype(np.float32), "b": b}


def apply(pr, X):
    return ((X - pr["mu"]) / pr["sd"]) @ pr["pca"].T @ pr["w"] + pr["b"]


out_rows, pool_rows = [], []
Path(args.out).mkdir(parents=True, exist_ok=True)
for f in Path(args.art).glob("*.npz"):  # carry over configs without progress data unchanged
    np.savez(Path(args.out) / f.name, **dict(np.load(f, allow_pickle=False)))
for pf in sorted(Path(args.progress).glob("*__B*.npz")):
    stem, cap = pf.stem.rsplit("__B", 1)
    cap = int(cap)
    gfiles = [f for f in Path(args.gen).glob(f"{stem}*.jsonl") if ".texts." not in f.name]
    if not gfiles:
        continue
    g = {}
    for f in gfiles:
        g.update({r["pid"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    model, effort = next(iter(g.values()))["model"], next(iter(g.values()))["effort"]
    mlog = {p: float(np.mean([np.log1p(s["len"]) for s in r["samples"]])) for p, r in g.items()}
    d = np.load(pf, allow_pickle=False)
    pids, sidx, S, layers = list(d["pids"]), d["sidx"], d["states"].astype(np.float32), list(d["layers"])
    y = np.array([mlog[p] for p in pids])
    grp = np.array([group[p] for p in pids])
    ug = np.unique(grp)
    fold_of = {gg: i % 5 for i, gg in enumerate(np.random.RandomState(0).permutation(ug))}
    fold = np.array([fold_of[gg] for gg in grp])

    def mix(pred_by_sample):
        """problem -> mean of observed (finished before cap) or predicted log length over its samples"""
        acc = defaultdict(list)
        for (p, si), v in pred_by_sample.items():
            acc[p].append(v)
        res = {}
        for p, r in g.items():
            vals = [np.log1p(s["len"]) for s in r["samples"] if s["finish"] == "stop" and s["len"] <= cap]
            vals += acc.get(p, [])
            res[p] = float(np.mean(vals)) if vals else np.nan
        return res

    best = None
    for li, L in enumerate(layers):
        X = S[:, li]
        if not np.isfinite(X).all():  # fp16 overflow from massive activations (e.g. gpt-oss final layer)
            print(f"{stem} cap {cap} layer {L}: non-finite states, skipped")
            continue
        pred = np.zeros(len(y))
        for f in range(5):
            if (fold == f).any() and (fold != f).sum() > args.pca:
                pred[fold == f] = apply(fit(X[fold != f], y[fold != f]), X[fold == f])
        m = mix({(p, int(si)): v for p, si, v in zip(pids, sidx, pred)})
        keys = [p for p in g if np.isfinite(m[p])]
        rho = spearmanr([m[p] for p in keys], [mlog[p] for p in keys]).correlation
        rho_s = spearmanr(pred, y).correlation
        print(f"{stem} cap {cap} layer {L}: sample-level rho {rho_s:.3f}, problem-level len_mix rho {rho:.3f} "
              f"({len(y)} running samples, {len(keys)} problems)")
        if best is None or rho > best[0]:
            best = (rho, li, L, m)
    rho, li, L, m = best
    center = float(np.nanmedian(list(m.values())))
    sub = tr[(tr.model_id == model) & (tr.reasoning_effort == effort) & tr.pid.isin(m)].copy()
    sub["len_prog"] = sub.pid.map(m) - center
    ok = sub.dropna(subset=["len_prog"])  # problems without progress states (e.g. Olmo second half) have no feature
    if ok.y.nunique() == 2:
        print(f"  -> best layer {L}: train rows n={len(ok)}/{len(sub)} AUC(-len_prog) {roc_auc_score(ok.y, -ok.len_prog):.3f}")
    out_rows.append(sub[["model_id", "reasoning_effort", "problem", "len_prog"]])
    pool_rows.extend({"model_id": model, "reasoning_effort": effort, "pid": p, "len_mix_oof": v, "mlog": mlog[p]}
                     for p, v in m.items())
    pr = fit(S[:, li], y)
    art = Path(args.art) / f"{stem}.npz"
    data = dict(np.load(art, allow_pickle=False)) if art.exists() else {}
    data.update(prog_layer=np.array(L), prog_cap=np.array(cap), prog_mu=pr["mu"], prog_sd=pr["sd"],
                prog_pca=pr["pca"], prog_w=pr["w"], prog_b=np.array(pr["b"]), prog_center=np.array(center))
    np.savez(Path(args.out) / f"{stem}.npz", **data)

if out_rows:
    Path(args.report).mkdir(parents=True, exist_ok=True)
    pd.concat(out_rows).to_csv(Path(args.report) / "train_rows_prog.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(Path(args.report) / "pool_prog_oof.csv", index=False)
    print("wrote", Path(args.report) / "train_rows_prog.csv")
