"""Consensus-difficulty probe: does a probe trained on a cross-model difficulty target improve the fused score?

Own targets are noisy where traces are few (DeepSeek: 2 traces on 144 problems; Olmo: 3-4 on 108). A consensus target
is available on all 322 problems: the mean of per-config z-scored mean log lengths over the four full-coverage configs
whose length tracks difficulty (Qwen, Skywork, gpt-oss medium/high; gpt-oss low is excluded, see T1). For every config
we fit the same probe recipe (standardise, PCA-32, ridge 10) on its own prompt states to predict the consensus, choose
layer/pooling by grouped 5-fold CV, and add the out-of-fold prediction c_hat as a third fusion input:
    s_new = a0 + a1 z_raw + a2 len_mix + a3 c_hat - median        (cross-fitted OLS on the pool, label-free)
Evaluation: pool OOF Spearman with the own mean log length; official train rows (scoring only): AUC, and acc / bal at
the capability-aware threshold recomputed for each score. Usage: python experiments/consensus_probe.py
"""

import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "runs/pod/runs"
R = ROOT / "runs/v1/report"
ART = ROOT / "runs/v3/art"
CONFIGS = [("Qwen/Qwen3.5-4B", "default"), ("Skywork/Skywork-OR1-Math-7B", "default"),
           ("allenai/Olmo-3-7B-Think", "default"), ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
           ("openai/gpt-oss-120b", "low"), ("openai/gpt-oss-120b", "medium"), ("openai/gpt-oss-120b", "high")]
CONS_FROM = [("Qwen/Qwen3.5-4B", "default"), ("Skywork/Skywork-OR1-Math-7B", "default"),
             ("openai/gpt-oss-120b", "medium"), ("openai/gpt-oss-120b", "high")]
RHO = 0.7


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


def short(c):
    return f"{c[0].split('/')[-1][:8]}|{c[1][:4]}"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


# --- problems, template groups and folds (same rules as train_probe.py) ---------------------------------------
problems = [json.loads(l) for l in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
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

# --- per-config targets and reliably-solved sets ---------------------------------------------------------------
gens = {}
for f in sorted(G.glob("gen/*.jsonl")):
    for r in map(json.loads, open(f, encoding="utf-8")):
        gens.setdefault((r["model"], r["effort"]), {})[r["pid"]] = r
truth = {p["pid"]: p["answer"] for p in problems}
for pid in [p for p, a in truth.items() if a is None]:
    votes = Counter()
    for (m, e), g in gens.items():
        for s in g.get(pid, {}).get("samples", []):
            if s["ans"] is not None and s["finish"] == "stop":
                votes[s["ans"]] += 2 if e == "high" else 1
    truth[pid] = votes.most_common(1)[0][0] if votes else None
mlog, inb = {}, {}
for c, g in gens.items():
    for pid, r in g.items():
        if truth.get(pid) is None:
            continue
        mlog[(c, pid)] = float(np.mean([np.log1p(s["len"]) for s in r["samples"]]))
        fin = [s for s in r["samples"] if s["finish"] == "stop"]
        inb[(c, pid)] = len(fin) > 0 and all(s["ans"] == truth[pid] for s in fin)
pids_all = [p["pid"] for p in problems]
Z = {}
for c in CONS_FROM:
    v = np.array([mlog.get((c, p), np.nan) for p in pids_all])
    Z[c] = (v - np.nanmean(v)) / np.nanstd(v)
cons = dict(zip(pids_all, np.nanmean(np.vstack(list(Z.values())), axis=0)))


def fit_probe(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Zx = (X - mu) / sd
    pca = PCA(n_components=min(32, len(X) - 1), random_state=0).fit(Zx)
    return mu, sd, pca, Ridge(alpha=10.0).fit(pca.transform(Zx), y)


def apply_probe(pr, X):
    mu, sd, pca, reg = pr
    return reg.predict(pca.transform((X - mu) / sd))


def oof(X, y, fold):
    out = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        out[te] = apply_probe(fit_probe(X[tr], y[tr]), X[te])
    return out


def folds_for(pids):
    grp = np.array([group[p] for p in pids])
    fmap = {g: i % 5 for i, g in enumerate(np.random.RandomState(0).permutation(np.unique(grp)))}
    return np.array([fmap[g] for g in grp])


# --- consensus probe per config (all 322 problems) -----------------------------------------------------------------
chat = {}
for c in CONFIGS:
    d = np.load(G / f"hidden/{safe(*c)}.npz", allow_pickle=False)
    pids = list(d["pids"])
    y = np.array([cons[p] for p in pids])
    fold = folds_for(pids)
    n_layers = d["last"].shape[1]
    layers = sorted({n_layers // 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3, 3 * n_layers // 4, n_layers - 1})
    best = None
    for pool in ("last", "mean"):
        H = d[pool]
        for L in layers:
            X = H[:, L].astype(np.float32)
            if not np.isfinite(X).all():
                continue
            pred = oof(X, y, fold)
            r = spearmanr(pred, y)[0]
            if best is None or r > best[0]:
                best = (r, pool, L, pred)
    chat[c] = dict(zip(pids, best[3]))
    print(f"[cons] {short(c)} best rho(c_hat, consensus) {best[0]:.3f} ({best[1]} L{best[2]})")

# --- fusion with and without c_hat, cross-fitted on the pool ---------------------------------------------------------
probe = pd.read_csv(R / "pool_probe_oof.csv")
probe = probe[probe.target == "len"][["model_id", "reasoning_effort", "pid", "oof", "y"]]
prog = pd.read_csv(R / "pool_prog_oof.csv")[["model_id", "reasoning_effort", "pid", "len_mix_oof"]]
pool = probe.merge(prog, on=["model_id", "reasoning_effort", "pid"], how="left")
rows = pd.read_csv(R / "train_rows_fused.csv")
arts = {c: dict(np.load(ART / f"{safe(*c)}.npz", allow_pickle=False)) for c in CONFIGS}


def crossfit(X, y, fold):
    out = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        a = np.linalg.lstsq(np.c_[np.ones(tr.sum()), X[tr]], y[tr], rcond=None)[0]
        out[te] = np.c_[np.ones(te.sum()), X[te]] @ a
    a_all = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0]
    return out, a_all


res_rows = []
for c in CONFIGS:
    g = pool[(pool.model_id == c[0]) & (pool.reasoning_effort == c[1])].copy().reset_index(drop=True)
    g["chat"] = g.pid.map(chat[c])
    fold = folds_for(list(g.pid))
    hp = g.len_mix_oof.notna().values  # problems with a progress estimate (fallback: prompt-only fusion)
    r = rows[(rows.model_id == c[0]) & (rows.reasoning_effort == c[1])].copy()
    art = arts[c]
    r["oof"] = r.z_len + float(art["center_len"])
    r["len_mix_oof"] = r.len_prog + float(art["prog_center"])
    r["chat"] = r.pid.map(chat[c])
    rp = r.len_mix_oof.notna().values
    out = {}
    for name, extra in (("base", []), ("cons", ["chat"])):
        cz, cp = ["oof"] + extra, ["oof", "len_mix_oof"] + extra
        pred_z, a_z = crossfit(g[cz].values, g.y.values, fold)
        pred = pred_z.copy()
        if hp.sum() >= 20:
            pred_p, a_p = crossfit(g.loc[hp, cp].values, g.y.values[hp], fold[hp])
            pred[hp] = pred_p
        else:
            a_p = None
        center = float(np.median(pred))
        B = np.array([inb.get((c, p), False) for p in g.pid])
        theta = float(np.quantile(pred[B] - center, RHO))
        s = np.c_[np.ones(len(r)), r[cz].values] @ a_z
        if a_p is not None and rp.any():
            s[rp] = np.c_[np.ones(rp.sum()), r.loc[rp, cp].values] @ a_p
        r[f"s_{name}"] = s - center
        r[f"p_{name}"] = r[f"s_{name}"] <= theta
        out[name] = (spearmanr(pred, g.y)[0], a_p if a_p is not None else a_z, theta)
    res_rows.append(r)
    print(f"[fuse] {short(c)} pool OOF rho base {out['base'][0]:.3f} -> +c_hat {out['cons'][0]:.3f} | "
          f"weights {np.round(out['cons'][1], 2)} | theta {out['base'][2]:+.3f} -> {out['cons'][2]:+.3f}")

rr = pd.concat(res_rows, ignore_index=True)
y = rr.y.values == 1
print(f"\ntrain rows n={len(rr)}")
for name in ("base", "cons"):
    s, p = rr[f"s_{name}"].values, rr[f"p_{name}"].values
    bal = 0.5 * (p[y].mean() + (~p[~y]).mean())
    per = " ".join(f"{short(c)}:{roc_auc_score(g.y, -g[f's_{name}']):.2f}"
                   for c, g in rr.groupby(["model_id", "reasoning_effort"]) if g.y.nunique() == 2)
    print(f"{name:5s} AUC {roc_auc_score(y, -s):.3f} acc {(p == y).mean():.3f} bal {bal:.3f} | {per}")
rr.to_csv(R / "train_rows_consensus.csv", index=False)
