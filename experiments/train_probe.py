"""Hidden-state probes for solve rate and reasoning length -> calibrated robustness decision.

Targets per (config, problem) from vLLM samples (cloud/gen_solve.py):
  solve : logit of smoothed solve rate (finished with the reference answer)
  len   : mean log1p(generated tokens) (truncated samples count at the cap) — continuous difficulty proxy
Inputs: --gen runs/gen/<safe_cfg>.jsonl, --hidden runs/hidden/<safe_cfg>.npz (cloud/extract_hidden.py)
Outputs: --art artifacts/<safe_cfg>.npz (one probe block per target), --report train_rows_z.csv (out-of-fold z's).
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvutil import grouped_cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [("Qwen/Qwen3.5-4B", "default"), ("Skywork/Skywork-OR1-Math-7B", "default"),
           ("allenai/Olmo-3-7B-Think", "default"), ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
           ("openai/gpt-oss-120b", "low"), ("openai/gpt-oss-120b", "medium"), ("openai/gpt-oss-120b", "high")]
TARGETS = ("solve", "len")

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs/gen"))
ap.add_argument("--hidden", default=str(ROOT / "runs/hidden"))
ap.add_argument("--art", default=str(ROOT / "runs/artifacts"))
ap.add_argument("--report", default=str(ROOT / "runs/report"))
ap.add_argument("--pca", type=int, default=32)
ap.add_argument("--alpha", type=float, default=10.0)
args = ap.parse_args()


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


problems = [json.loads(l) for l in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
pid_of = {norm(p["problem"]): p["pid"] for p in problems}

# --- template groups: symbolic siblings share a group ---------------------------------------
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

# --- samples (shards *.partN.jsonl merged) -> targets -----------------------------------------
gens = {}
for m, e in CONFIGS:
    rows = {}
    for f in sorted(Path(args.gen).glob(f"{safe(m, e)}*.jsonl")):
        if ".texts." not in f.name:
            rows.update({r["pid"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    if rows:
        gens[(m, e)] = rows
print("gen configs:", {f"{m.split('/')[1]}|{e}": len(v) for (m, e), v in gens.items()})

truth = {p["pid"]: p["answer"] for p in problems}
for pid in [p for p, a in truth.items() if a is None]:  # pseudo truth: vote of finished answers, gpt-oss high x2
    votes = Counter()
    for (m, e), g in gens.items():
        for s in g.get(pid, {}).get("samples", []):
            if s["ans"] is not None and s["finish"] == "stop":
                votes[s["ans"]] += 2 if e == "high" else 1
    truth[pid] = votes.most_common(1)[0][0] if votes else None

tgt = {}  # (cfg, pid) -> {"solve": logit, "len": mean log len, "rate": raw}
for cfg, g in gens.items():
    for pid, row in g.items():
        if truth.get(pid) is None:
            continue
        ok = [s["ans"] == truth[pid] and s["finish"] == "stop" for s in row["samples"]]
        k, n = sum(ok), len(ok)
        tgt[(cfg, pid)] = {"solve": np.log((k + 0.5) / (n - k + 0.5)), "rate": k / n,
                           "len": float(np.mean([np.log1p(s["len"]) for s in row["samples"]]))}

# --- official labels -------------------------------------------------------------------------
tr = pd.read_parquet(ROOT / "data/raw/train-main-v2/data.parquet")
tr = tr[tr.is_robust.notna()].copy()
tr["pid"] = tr.problem.map(lambda s: pid_of[norm(s)])
tr["cfg"] = list(zip(tr.model_id, tr.reasoning_effort))
tr["y"] = tr.is_robust.astype(bool).astype(int)
tr["grp"] = tr.pid.map(group)
for t in ("rate", "len"):
    tr[f"emp_{t}"] = [tgt.get((c, p), {}).get(t, np.nan) for c, p in zip(tr.cfg, tr.pid)]
pool_stats = {}  # cfg -> median mean-log-length over pool, median within-problem std of log length
for cfg, g in gens.items():
    mls = [np.mean(np.log1p([s["len"] for s in r["samples"]])) for r in g.values()]
    # within-problem spread from finished samples only (cap-truncated samples share one length)
    sds = [np.std(np.log1p(fl)) for r in g.values()
           if len(fl := [s["len"] for s in r["samples"] if s["finish"] == "stop"]) > 1]
    q1, q3 = np.percentile(mls, [25, 75])
    pool_stats[cfg] = (float(np.median(mls)), float(np.median(sds)), float((q3 - q1) / 1.349))
tr["emp_len_rel"] = [l - pool_stats[c][0] if c in pool_stats else np.nan for l, c in zip(tr.emp_len, tr.cfg)]
have = tr.emp_rate.notna()
if have.sum() and tr[have].y.nunique() == 2:
    print(f"[oracle] AUC -relative loglen {roc_auc_score(tr[have].y, -tr[have].emp_len_rel):.3f}; "
          f"rule rel<0: acc {((tr[have].emp_len_rel < 0) == tr[have].y).mean():.3f}")
    print(f"\n[oracle] {have.sum()} train rows: AUC solve-rate {roc_auc_score(tr[have].y, tr[have].emp_rate):.3f}, "
          f"AUC -loglen {roc_auc_score(tr[have].y, -tr[have].emp_len):.3f}")


# --- probes ------------------------------------------------------------------------------------
def fit_probe(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = (X - mu) / sd
    pca = PCA(n_components=min(args.pca, len(X) - 1), random_state=0).fit(Z)
    reg = Ridge(alpha=args.alpha).fit(pca.transform(Z), y)
    b = float(reg.intercept_ - pca.mean_ @ pca.components_.T @ reg.coef_)  # fold PCA centering into bias
    return {"mu": mu, "sd": sd, "pca": pca.components_.astype(np.float32), "w": reg.coef_.astype(np.float32), "b": b}


def apply_probe(pr, X):
    return ((X - pr["mu"]) / pr["sd"]) @ pr["pca"].T @ pr["w"] + pr["b"]


best = {}  # (cfg, target) -> spec
for cfg in CONFIGS:
    f = Path(args.hidden) / f"{safe(*cfg)}.npz"
    if not f.exists() or cfg not in gens:
        continue
    d = np.load(f, allow_pickle=False)
    pids, H = list(d["pids"]), {"last": d["last"].astype(np.float32), "mean": d["mean"].astype(np.float32)}
    idx = [i for i, p in enumerate(pids) if (cfg, p) in tgt]
    P = [pids[i] for i in idx]
    grp = np.array([group[p] for p in P])
    folds = {g: i % 5 for i, g in enumerate(np.random.RandomState(0).permutation(np.unique(grp)))}
    fold = np.array([folds[g] for g in grp])
    n_layers = H["last"].shape[1]
    layers = sorted({n_layers // 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3, 3 * n_layers // 4, n_layers - 1})
    for t in TARGETS:
        y = np.array([tgt[(cfg, p)][t] for p in P])
        res = []
        for pool in ("last", "mean"):
            for L in layers:
                X = H[pool][idx, L]
                pred = np.zeros(len(y))
                for k in range(5):
                    pred[fold == k] = apply_probe(fit_probe(X[fold != k], y[fold != k]), X[fold == k])
                res.append((spearmanr(pred, y).correlation, pool, L, pred))
        res.sort(key=lambda x: -x[0])
        r, pool, L, pred = res[0]
        print(f"[probe] {cfg[0].split('/')[1]:22s} {cfg[1]:7s} {t:5s} n={len(y)} best rho={r:.3f} ({pool} L{L}); "
              f"next {[(round(x[0], 2), x[1], x[2]) for x in res[1:3]]}")
        best[(cfg, t)] = {"pool": pool, "layer": L, "pids": pids, "X": H[pool][:, L], "idx": idx, "P": P, "y": y, "rho": r,
                          "center": float(np.median(pred)), "oof": pred}

# --- calibration on official rows: nested leave-template-out --------------------------------------
cfgs = sorted({c for c, _ in best})
rows = tr[tr.cfg.isin(cfgs)].copy().reset_index(drop=True)
for t in TARGETS:
    z = np.full(len(rows), np.nan)
    for g in rows.grp.unique():
        for j in np.where((rows.grp == g).values)[0]:
            b = best.get((rows.cfg[j], t))
            if b is None:
                continue
            keep = [k for k, p in enumerate(b["P"]) if group[p] != g]
            pr = fit_probe(b["X"][[b["idx"][k] for k in keep]], b["y"][keep])
            z[j] = apply_probe(pr, b["X"][[b["pids"].index(rows.pid[j])]])[0] - b["center"]
    rows[f"z_{t}"] = z  # relative to the config's pool median prediction
if len(rows) and rows.y.nunique() == 2:
    y = rows.y.values
    print(f"\n[robust] {len(rows)} train rows: AUC z_solve {roc_auc_score(y, rows.z_solve):.3f}, "
          f"AUC -z_len {roc_auc_score(y, -rows.z_len):.3f}")
    for cols in (["z_solve"], ["z_len"], ["z_solve", "z_len"]):
        auc, bacc, acc, _ = grouped_cv(rows[cols].values, y, rows.grp.values)
        print(f"[robust] grouped5x20 {'+'.join(cols):15s} AUC {auc:.3f} balanced {bacc:.3f} acc {acc:.3f}")

Path(args.report).mkdir(parents=True, exist_ok=True)
rows.drop(columns=["cfg"]).to_csv(Path(args.report) / "train_rows_z.csv", index=False)
pool_rows = []  # pool-level out-of-fold probe outputs (raw, uncentered) for per-config fusion
for (cfg, t), b in best.items():
    for p, v, yy in zip(b["P"], b["oof"], b["y"]):
        pool_rows.append({"model_id": cfg[0], "reasoning_effort": cfg[1], "pid": p, "target": t, "oof": v, "y": yy})
pd.DataFrame(pool_rows).to_csv(Path(args.report) / "pool_probe_oof.csv", index=False)

# --- final artifacts: probes on all problems; probe-only decision on [z_solve, z_len] ---------------
cols = ["z_solve", "z_len"]
lr = LogisticRegression(C=1.0, class_weight="balanced").fit(rows[cols], rows.y) if rows.y.nunique() == 2 else None
Path(args.art).mkdir(parents=True, exist_ok=True)
for cfg in cfgs:
    data = {}
    for t, suf in (("solve", ""), ("len", "_len")):
        b = best[(cfg, t)]
        pr = fit_probe(b["X"][b["idx"]], b["y"])
        data.update({f"layers{suf}": np.array([b["layer"]]), f"pool{suf}": np.array(b["pool"]), f"mu{suf}": pr["mu"],
                     f"sd{suf}": pr["sd"], f"pca{suf}": pr["pca"], f"w{suf}": pr["w"], f"b{suf}": np.array(pr["b"]),
                     f"center{suf}": np.array(b["center"])})
    data.update(beh_mu_center=np.array(pool_stats[cfg][0]), beh_sigma=np.array(pool_stats[cfg][1]),
                beh_tau=np.array(pool_stats[cfg][2]))
    if lr is None:
        raise SystemExit("calibration needs both labels among covered train rows")
    data.update(cal_feats=np.array(cols), cal_w=lr.coef_[0].astype(np.float64), cal_b=np.array(lr.intercept_[0]))
    np.savez(Path(args.art) / f"{safe(*cfg)}.npz", **data)
print("artifacts:", sorted(p.name for p in Path(args.art).glob("*.npz")))
