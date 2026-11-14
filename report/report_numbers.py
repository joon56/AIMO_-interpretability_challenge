"""All numbers quoted in the report, from the pipeline outputs (no fitting on official labels; labels only score).

Writes report/numbers.json and prints the two tables:
  per-config: pool size, share reliably solved, probe Spearman (prompt / progress / fused vs mean log length, out-of-fold),
              train rows, threshold-free AUC of -s_m on official train rows, theta_m
  ablation  : constant / black-box len_beh / prompt probe z / fused s with v2 and v3 thresholds, on all 80 rows and on
              the 61 rows that have black-box behaviour samples (same rows for every method), with a grouped bootstrap
              CI (resampling template groups) for AUC(s) - AUC(len_beh)
Usage: python report/report_numbers.py
"""

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aimo_robust.predict import SAMPLING  # noqa: E402
from aimo_robust.sampler import censored_mu  # noqa: E402

REP, ART, GEN = ROOT / "runs/v1/report", ROOT / "runs/v3/art", ROOT / "runs/pod/runs"
NAMES = {("Qwen/Qwen3.5-4B", "default"): "Qwen3.5-4B", ("Skywork/Skywork-OR1-Math-7B", "default"): "Skywork-OR1-7B",
         ("allenai/Olmo-3-7B-Think", "default"): "Olmo-3-7B-Think",
         ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"): "DeepSeek-R1-8B",
         ("openai/gpt-oss-120b", "low"): "gpt-oss low", ("openai/gpt-oss-120b", "medium"): "gpt-oss medium",
         ("openai/gpt-oss-120b", "high"): "gpt-oss high"}


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


arts = {f.stem: dict(np.load(f, allow_pickle=False)) for f in ART.glob("*.npz")}
rows = pd.read_csv(REP / "train_rows_fused.csv")
rows["theta"] = [float(arts[safe(m, e)]["cap_theta"]) for m, e in zip(rows.model_id, rows.reasoning_effort)]
rows["theta_z"] = [float(arts[safe(m, e)]["cap_theta_z"]) for m, e in zip(rows.model_id, rows.reasoning_effort)]
y = rows.y.values == 1

# --- black-box behaviour feature on train cases (same sampler as the submission, re-censored to the deployed cap)
beh = pd.DataFrame([json.loads(l) for f in glob.glob(str(GEN / "behavior/*.jsonl")) for l in open(f, encoding="utf-8")])
beh = beh.rename(columns={"model": "model_id", "effort": "reasoning_effort"})  # official train cases only, keyed by pid
cap_of = {m: c for m, (_, c, _) in SAMPLING.items()}
vals = []
for m, e, L, F in zip(beh.model_id, beh.reasoning_effort, beh.lens, beh.fin):
    a, cap = arts[safe(m, e)], cap_of[m]
    L2, F2 = [min(n, cap) for n in L], [f and n <= cap for n, f in zip(L, F)]
    vals.append(censored_mu(L2, F2, float(a["beh_sigma"]), float(a["beh_mu_center"]), float(a["beh_tau"]))
                - float(a["beh_mu_center"]) if L2 else np.nan)
beh["len_beh"] = vals
rows = rows.merge(beh[["model_id", "reasoning_effort", "pid", "len_beh"]],
                  on=["model_id", "reasoning_effort", "pid"], how="left")


def acc_bal(pred, yy):
    return float((pred == yy).mean()), float(0.5 * (pred[yy].mean() + (~pred[~yy]).mean()))


def auc(score, yy):
    return float(roc_auc_score(yy, -score))


def grouped_boot(df, f1, f2, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = df.grp.unique()
    idx = {g: np.where(df.grp.values == g)[0] for g in groups}
    diffs = []
    for _ in range(reps):
        take = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        yy = df.y.values[take] == 1
        if yy.all() or (~yy).all():
            continue
        diffs.append(auc(df[f1].values[take], yy) - auc(df[f2].values[take], yy))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


out = {"per_config": [], "ablation": {}}

# --- per-config table ----------------------------------------------------------------------------------------
probe = pd.read_csv(REP / "pool_probe_oof.csv")
probe = probe[probe.target == "len"]
prog = pd.read_csv(REP / "pool_prog_oof.csv")
for (m, e), name in NAMES.items():
    p = probe[(probe.model_id == m) & (probe.reasoning_effort == e)]
    q = prog[(prog.model_id == m) & (prog.reasoning_effort == e)].dropna(subset=["len_mix_oof"])
    a = arts[safe(m, e)]
    fa = a["fuse_a"]
    pq = p.merge(q[["pid", "len_mix_oof"]], on="pid", how="left")
    a_z = np.linalg.lstsq(np.c_[np.ones(len(pq)), pq.oof], pq.y, rcond=None)[0]
    fused = np.where(pq.len_mix_oof.notna(), fa[0] + fa[1] * pq.oof + fa[2] * pq.len_mix_oof.fillna(0),
                     a_z[0] + a_z[1] * pq.oof)
    r = rows[(rows.model_id == m) & (rows.reasoning_effort == e)]
    out["per_config"].append({
        "config": name, "pool": int(len(p)), "prog_problems": int(len(q)),
        "rho_prompt": float(spearmanr(p.oof, p.y)[0]), "rho_progress": float(spearmanr(q.len_mix_oof, q.mlog)[0]),
        "rho_fused": float(spearmanr(fused, pq.y)[0]), "train_rows": int(len(r)), "train_robust": int(r.y.sum()),
        "auc_fused": auc(r.len_fused.values, r.y.values == 1) if r.y.nunique() == 2 else None,
        "theta": float(a["cap_theta"]), "prog_cap": int(a["prog_cap"]), "prog_layer": int(a["prog_layer"]),
        "layer_len": int(a["layers_len"][0]), "pool_len": str(a["pool_len"]), "k": SAMPLING[m][0]})

# --- ablation --------------------------------------------------------------------------------------------------
def methods(df):
    yy = df.y.values == 1
    res = {"n": int(len(df)), "robust_rate": float(yy.mean())}
    res["constant_robust"] = {"acc": float(yy.mean()), "bal": 0.5}
    if df.len_beh.notna().all():
        res["blackbox_len_beh"] = {"auc": auc(df.len_beh.values, yy),
                                   **dict(zip(("acc", "bal"), acc_bal(df.len_beh.values <= 0, yy)))}
    res["probe_z"] = {"auc": auc(df.z_len.values, yy), **dict(zip(("acc", "bal"), acc_bal(df.z_len.values <= df.theta_z.values, yy)))}
    res["fused_v2"] = {"auc": auc(df.len_fused.values, yy), **dict(zip(("acc", "bal"), acc_bal(df.len_fused.values < 0, yy)))}
    res["fused_v3"] = {"auc": auc(df.len_fused.values, yy),
                       **dict(zip(("acc", "bal"), acc_bal(df.len_fused.values <= df.theta.values, yy)))}
    return res


out["ablation"]["all_rows"] = methods(rows)
sub = rows.dropna(subset=["len_beh"]).copy()
out["ablation"]["behaviour_rows"] = methods(sub)
out["ablation"]["behaviour_rows"]["auc_diff_fused_minus_beh_ci95"] = grouped_boot(sub, "len_fused", "len_beh")
out["ablation"]["behaviour_rows"]["auc_diff_fused_minus_probe_ci95"] = grouped_boot(sub, "len_fused", "z_len")
out["ablation"]["all_rows"]["auc_diff_fused_minus_probe_ci95"] = grouped_boot(rows, "len_fused", "z_len")
out["ablation"]["behaviour_rows"]["configs"] = sorted({NAMES[(m, e)] for m, e in zip(sub.model_id, sub.reasoning_effort)})

Path(ROOT / "report/numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("config            pool prog  rho_prompt rho_prog rho_fused  train(rob) AUC   theta  layer/pool  cap  K")
for c in out["per_config"]:
    print(f"{c['config']:16s} {c['pool']:4d} {c['prog_problems']:4d}   {c['rho_prompt']:.2f}      {c['rho_progress']:.2f}     "
          f"{c['rho_fused']:.2f}     {c['train_rows']:2d} ({c['train_robust']:2d})  "
          f"{(c['auc_fused'] or float('nan')):.2f}  {c['theta']:+.3f}  L{c['layer_len']}/{c['pool_len']}  "
          f"{c['prog_cap']}  {c['k']}")
for k, v in out["ablation"].items():
    print(f"\n[{k}] n={v['n']} robust {v['robust_rate']:.3f}")
    for mth in ("constant_robust", "blackbox_len_beh", "probe_z", "fused_v2", "fused_v3"):
        if mth in v:
            print(f"  {mth:16s} " + " ".join(f"{kk} {vv:.3f}" for kk, vv in v[mth].items()))
    for kk in [x for x in v if x.startswith("auc_diff") or x == "configs"]:
        print(f"  {kk}: {v[kk]}")
