"""v2 decision rule: move the robustness threshold of the v1 fused artifacts to each model's pool median.

    python experiments/make_v2_artifacts.py --art runs/v1/art_fused --out runs/v2/art

len_fused is already centered on the model's pool median (experiments/fuse.py), and the validation/test sets are
balanced per model, so v2 predicts "robust iff len_fused <= 0" instead of using the intercept fitted on the official
rows (74% robust). The fail-soft fallback (sampling failure) applies the same principle to the prompt-state probe:
"robust iff z_len <= 0". Everything else (probes, fusion coefficients, slope) is unchanged.
"""

import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="runs/v1/art_fused")
ap.add_argument("--out", default="runs/v2/art")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(args.art).glob("*.npz")):
    d = dict(np.load(f, allow_pickle=False))
    d["comb_b"] = np.array(0.0)  # logit = comb_w * len_fused: robust iff len_fused <= 0
    d["cal_feats"] = np.array(["z_len"])  # fallback when sampling fails
    d["cal_w"] = np.array([-2.0])
    d["cal_b"] = np.array(0.0)
    np.savez(out / f.name, **d)
print("wrote", len(list(out.glob("*.npz"))), "artifacts to", out)
