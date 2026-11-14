"""Capability-aware decision thresholds (label-free) vs the v2 rule (len_fused < 0 = per-config pool median).

Val/test cases only contain (model, problem) pairs whose base answer is consistently correct ("those are excluded",
organizers 10/16), and the val board is balanced per problem AND per model. The pool median also counts problems the
model fails, so it misplaces the boundary for configs that solve few pool problems reliably or that answer hard
problems quickly and wrongly (gpt-oss low: over the whole pool, shorter traces are LESS often correct). Per config c:
  B_c = pool problems where >= 1 sample finished and every finished sample is correct (32k-truncated samples are
        treated as unknown, since labels use a 100k budget; --bdef strict also requires every sample to finish)
  theta_c = rho-quantile of the pool out-of-fold feature over B_c (rho ~ natural robust rate among such cases)
Robust iff feature <= theta_c. Official train rows are used only to score the rules, never to fit them.
With --out, writes artifacts = --art copies with comb_b / cal_b moved to theta_c (no code change in predict.py).
Usage: python experiments/capability_threshold.py --art runs/v2/art --rho 0.7 --out runs/v3/art
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--report", default=str(ROOT / "runs/v1/report"))
ap.add_argument("--art", default=str(ROOT / "runs/v2/art"))
ap.add_argument("--gen", default=str(ROOT / "runs/pod/runs/gen"))
ap.add_argument("--val_log", default="",
                help="optional stderr log of a validation run ('[aimo]' lines, features only); not published here")
ap.add_argument("--bdef", choices=("strict", "fin"), default="fin")
ap.add_argument("--rho", type=float, default=0.7)
ap.add_argument("--out", default="")
args = ap.parse_args()
R = Path(args.report)
RHOS = sorted({0.5, 0.6, 0.7, 0.75, 0.8, args.rho})


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


def short(m, e):
    return f"{m.split('/')[-1][:8]}|{e[:4]}"


# --- per-sample correctness from the pool generations (same truth rule as train_probe.py) ---------------
problems = [json.loads(l) for l in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
gens = {}
for f in sorted(Path(args.gen).glob("*.jsonl")):
    if ".texts." in f.name:
        continue
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
in_b = {}  # (m, e, pid) -> problem counts as reliably solved
for (m, e), g in gens.items():
    for pid, r in g.items():
        if truth.get(pid) is None:
            continue
        fin = [s for s in r["samples"] if s["finish"] == "stop"]
        ok = sum(s["ans"] == truth[pid] for s in fin)
        in_b[(m, e, pid)] = (ok == len(r["samples"])) if args.bdef == "strict" else (len(fin) > 0 and ok == len(fin))

# --- pool features (out-of-fold), exactly as fuse.py / train_probe.py compute them --------------------------
probe = pd.read_csv(R / "pool_probe_oof.csv")
probe = probe[probe.target == "len"].rename(columns={"oof": "z_raw", "y": "mlog"})
prog = pd.read_csv(R / "pool_prog_oof.csv")[["model_id", "reasoning_effort", "pid", "len_mix_oof"]]
pool = probe.merge(prog, on=["model_id", "reasoning_effort", "pid"], how="left")
arts = {f.stem: dict(np.load(f, allow_pickle=False)) for f in Path(args.art).glob("*.npz")}
parts = []
for (m, e), g in pool.groupby(["model_id", "reasoning_effort"]):
    art = arts[safe(m, e)]
    a, center = art["fuse_a"], float(art["fuse_center"])
    a_z = np.linalg.lstsq(np.c_[np.ones(len(g)), g.z_raw], g.mlog, rcond=None)[0]
    fused = np.where(g.len_mix_oof.notna(), a[0] + a[1] * g.z_raw + a[2] * g.len_mix_oof.fillna(0),
                     a_z[0] + a_z[1] * g.z_raw) - center
    parts.append(g.assign(len_fused=fused, z_len=g.z_raw - float(art["center_len"]),
                          inb=[in_b.get((m, e, p), False) for p in g.pid]))
pool = pd.concat(parts, ignore_index=True)

theta = {}  # (m, e) -> {feature: {rho: theta}}
print(f"B_c definition: {args.bdef}")
print("config         pool  |B|/pool | len_fused theta(rho): " + " ".join(f"{r:>5}" for r in RHOS)
      + " | z_len theta(rho)")
for (m, e), g in pool.groupby(["model_id", "reasoning_effort"]):
    B = g[g.inb]
    theta[(m, e)] = {f: {r: float(np.quantile(B[f], r)) for r in RHOS} for f in ("len_fused", "z_len")}
    t = theta[(m, e)]
    print(f"{short(m, e):14s} {len(g):4d}  {g.inb.mean():6.2f}  | "
          + " ".join(f"{t['len_fused'][r]:+5.2f}" for r in RHOS) + " | "
          + " ".join(f"{t['z_len'][r]:+5.2f}" for r in RHOS))

# --- official train rows: score rules (no fitting on labels) ------------------------------------------------
rows = pd.read_csv(R / "train_rows_fused.csv")
y = rows.y.values == 1


def rule(df, feat, rho):
    if rho is None:
        return df[feat].values < 0
    return np.array([v <= theta[(m, e)][feat][rho] for v, m, e in zip(df[feat], df.model_id, df.reasoning_effort)])


print(f"\ntrain rows n={len(rows)} (robust {y.mean():.2f}): acc / balanced / predicted-robust | per-config acc")
for feat in ("len_fused", "z_len"):
    for rho in [None] + RHOS:
        p = rule(rows, feat, rho)
        bal = 0.5 * (p[y].mean() + (~p[~y]).mean())
        per = " ".join(f"{short(m, e)}:{(p[g.index] == y[g.index]).mean():.2f}"
                       for (m, e), g in rows.groupby(["model_id", "reasoning_effort"]))
        name = f"{feat}<0 (v2)" if rho is None else f"{feat} cap{rho}"
        print(f"{name:18s} {(p == y).mean():.3f} {bal:.3f} {p.mean():.2f} | {per}")

# --- unlabelled v1 val features: predicted-robust share per config (sanity only, never used to choose) --------
# Needs the stderr log of the v1-main validation run; it is not part of this repository, so the block is optional.
if args.val_log and Path(args.val_log).exists():
    val = []
    for line in open(args.val_log, encoding="utf-8"):
        if line.startswith("[aimo]"):
            tok = line.split()
            kv = dict(t.split("=", 1) for t in tok[3:] if "=" in t)
            val.append({"model_id": tok[1], "reasoning_effort": tok[2], "len_fused": float(kv["len_fused"]),
                        "z_len": float(kv["z_len"])})
    val = pd.DataFrame(val)
    print(f"\nval (unlabelled, n={len(val)}) predicted robust:")
    for feat, rho in (("len_fused", None), ("len_fused", args.rho), ("z_len", None), ("z_len", args.rho)):
        p = rule(val, feat, rho)
        per = " ".join(f"{short(m, e)}:{int(p[g.index].sum())}/{len(g)}"
                       for (m, e), g in val.groupby(["model_id", "reasoning_effort"]))
        print(f"{feat:9s} {'v2' if rho is None else f'cap{rho}':7s} {int(p.sum())}/{len(val)} | {per}")

if args.out:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for (m, e), t in theta.items():
        data = dict(arts[safe(m, e)])
        th_f, th_z = t["len_fused"][args.rho], t["z_len"][args.rho]
        # logit = b + w * feat >= 0  <=>  feat <= theta  (w < 0); the fallback (sampling failed) is probe-only z_len.
        # comb_w keeps fuse.py's slope; cal_w = -2 as in v2 (only the sign matters for the decision)
        w_f = float(data["comb_w"][0])
        data.update(comb_feats=np.array(["len_fused"]), comb_b=np.array(-w_f * th_f),
                    cal_feats=np.array(["z_len"]), cal_w=np.array([-2.0]), cal_b=np.array(2.0 * th_z),
                    cap_rho=np.array(args.rho), cap_theta=np.array(th_f), cap_theta_z=np.array(th_z))
        np.savez(out / f"{safe(m, e)}.npz", **data)
    print(f"\nwrote {len(theta)} artifacts (rho={args.rho}, B_c={args.bdef}) to {out}")
