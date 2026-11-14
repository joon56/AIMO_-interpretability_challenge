"""Ablation artifacts for the white-box award (same code path as the full submission, decision features swapped).

  blackbox : same budgeted samples (K, token cap), decision on len_beh <= 0 only (censored log-normal MAP of the mean
             log reasoning length from finish flags and lengths: text-level information). Progress probe and fusion
             removed (no hook), sampling failure -> constant True (no hidden-state fallback).
  probeonly: prompt-state length probe only (z_len <= cal threshold of --art), no sampling.
Usage: python experiments/make_ablation.py --art runs/v3/art --mode blackbox --out runs/v3b/art
"""

import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True)
ap.add_argument("--mode", choices=("blackbox", "probeonly"), required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(args.art).glob("*.npz")):
    data = dict(np.load(f, allow_pickle=False))
    drop = [k for k in data if k.startswith(("prog_", "fuse_"))]
    if args.mode == "probeonly":
        drop += [k for k in data if k.startswith("comb_")]
    for k in drop:
        del data[k]
    if args.mode == "blackbox":
        data.update(comb_feats=np.array(["len_beh"]), comb_w=np.array([-1.0]), comb_b=np.array(0.0),
                    cal_feats=np.array([], dtype="<U8"), cal_w=np.zeros(0), cal_b=np.array(0.0))
    np.savez(out / f.name, **data)
    print(f.name, args.mode, "decision:", data.get("comb_feats", data["cal_feats"]).tolist(), "dropped", len(drop))
