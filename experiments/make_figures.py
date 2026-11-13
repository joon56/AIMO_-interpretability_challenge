"""README figures, generated from the released report CSVs (runs/v1/report). No GPU, no problem text needed.

    python experiments/make_figures.py        # -> docs/figures/*.png
Needs matplotlib in addition to requirements.txt.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvutil import grouped_cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/v1/report"
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

ROBUST, NON = "#2a9d8f", "#e76f51"
PROMPT, PROG, FUSED = "#6c8ebf", "#e9a23b", "#2a9d8f"
CONFIGS = [("Qwen/Qwen3.5-4B", "default", "Qwen3.5-4B"), ("Skywork/Skywork-OR1-Math-7B", "default", "Skywork-OR1-7B"),
           ("allenai/Olmo-3-7B-Think", "default", "Olmo-3-7B"), ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default", "DeepSeek-R1-8B"),
           ("openai/gpt-oss-120b", "low", "gpt-oss low"), ("openai/gpt-oss-120b", "medium", "gpt-oss med"),
           ("openai/gpt-oss-120b", "high", "gpt-oss high")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def jitter(n, seed, width=0.16):
    return np.random.RandomState(seed).uniform(-width, width, n)


rows = pd.read_csv(REPORT / "train_rows_fused.csv")

# 1 -- oracle: measured relative reasoning length vs the official robustness label --------------------------------
d = rows.dropna(subset=["emp_len_rel"])
auc = roc_auc_score(d.y, -d.emp_len_rel)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for k, (lab, col) in enumerate(((1, ROBUST), (0, NON))):
    g = d[d.y == lab]
    small = ~g.model_id.str.contains("gpt-oss")
    ax.scatter(k + jitter(len(g), k)[small.values], g.emp_len_rel[small], c=col, s=46, alpha=0.85, edgecolor="white", lw=0.6)
    ax.scatter(k + jitter(len(g), k + 9)[~small.values], g.emp_len_rel[~small], c=col, s=52, alpha=0.85, marker="^",
               edgecolor="white", lw=0.6)
    ax.hlines(g.emp_len_rel.median(), k - 0.3, k + 0.3, color="black", lw=2)
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_xticks([0, 1], [f"robust (n={int(d.y.sum())})", f"non-robust (n={int((1 - d.y).sum())})"])
ax.set_ylabel("mean log reasoning length\nminus the model's pool median")
ax.set_title(f"Oracle: measured reasoning length separates the labels (AUC {auc:.2f})", fontsize=10.5)
ax.scatter([], [], c="grey", s=46, label="small models (4 configs)")
ax.scatter([], [], c="grey", s=52, marker="^", label="gpt-oss-120b (3 efforts)")
ax.plot([], [], c="black", lw=2, label="median")
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
save(fig, "fig1_oracle_length.png")
print(f"  oracle AUC {auc:.3f} on {len(d)} labelled rows; solve-rate AUC "
      f"{roc_auc_score(rows.dropna(subset=['emp_rate']).y, rows.dropna(subset=['emp_rate']).emp_rate):.3f}")

# 2 -- probe quality on the label-free problem pool (out-of-fold Spearman with the measured mean log length) ----------
probe = pd.read_csv(REPORT / "pool_probe_oof.csv")
probe = probe[probe.target == "len"].rename(columns={"oof": "z_raw", "y": "mlog"})
prog = pd.read_csv(REPORT / "pool_prog_oof.csv")[["model_id", "reasoning_effort", "pid", "len_mix_oof"]]
pool = probe.merge(prog, on=["model_id", "reasoning_effort", "pid"], how="left")
rho = {"prompt": [], "progress": [], "fused": []}
for m, e, _ in CONFIGS:
    g = pool[(pool.model_id == m) & (pool.reasoning_effort == e)]
    both = g.dropna(subset=["len_mix_oof"])
    a_z = np.linalg.lstsq(np.c_[np.ones(len(g)), g.z_raw], g.mlog, rcond=None)[0]
    a = np.linalg.lstsq(np.c_[np.ones(len(both)), both.z_raw, both.len_mix_oof], both.mlog, rcond=None)[0]
    fused = np.where(g.len_mix_oof.notna(), a[0] + a[1] * g.z_raw + a[2] * g.len_mix_oof.fillna(0), a_z[0] + a_z[1] * g.z_raw)
    rho["prompt"].append(spearmanr(g.z_raw, g.mlog).correlation)
    rho["progress"].append(spearmanr(both.len_mix_oof, both.mlog).correlation)
    rho["fused"].append(spearmanr(fused, g.mlog.values).correlation)
fig, ax = plt.subplots(figsize=(10.2, 4.4))
x, w = np.arange(len(CONFIGS)), 0.27
for i, (key, col, lab) in enumerate((("prompt", PROMPT, "prompt-state probe (z_len)"),
                                     ("progress", PROG, "progress probe (len_mix)"), ("fused", FUSED, "fused (len_fused)"))):
    bars = ax.bar(x + (i - 1) * w, rho[key], w, color=col, label=lab)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in rho[key]], fontsize=7, padding=1.5)
ax.set_xticks(x, [c[2] for c in CONFIGS], rotation=20, ha="right")
ax.set_ylabel("Spearman ρ (out-of-fold)")
ax.set_ylim(0, 1.12)
ax.set_title("Hidden-state probes predict each model's reasoning length on unseen problems", fontsize=10.5)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
save(fig, "fig2_probe_quality.png")
print("  rho prompt  ", [round(v, 2) for v in rho["prompt"]])
print("  rho progress", [round(v, 2) for v in rho["progress"]])
print("  rho fused   ", [round(v, 2) for v in rho["fused"]])

# 3 -- decision quality on the official labelled rows (repeated grouped 5-fold CV) -------------------------------------
res = {}
for col in ("z_len", "len_fused"):
    auc_, bacc, acc, prob = grouped_cv(rows[[col]].values, rows.y.values, rows.grp.values)
    res[col] = (auc_, bacc, acc, prob)
fig, ax = plt.subplots(figsize=(10.6, 4.4))
labels = [f"{c[2]}\n(n={((rows.model_id == c[0]) & (rows.reasoning_effort == c[1])).sum()})" for c in CONFIGS] + [f"all rows\n(n={len(rows)})"]
for i, (col, colr, lab) in enumerate((("z_len", PROMPT, "prompt probe only (v1p)"), ("len_fused", FUSED, "fused (v1 / v2)"))):
    vals = []
    for m, e, _ in CONFIGS:
        k = ((rows.model_id == m) & (rows.reasoning_effort == e)).values
        vals.append(roc_auc_score(rows.y[k], res[col][3][k]) if rows.y[k].nunique() == 2 else np.nan)
    vals.append(res[col][0])
    bars = ax.bar(np.arange(len(vals)) + (i - 0.5) * 0.38, vals, 0.38, color=colr, label=lab)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7.5, padding=1.5)
ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
ax.set_ylim(0.3, 1.1)
ax.set_ylabel("ROC AUC on held-out template groups")
ax.set_title("Robustness AUC on the 80 official labelled rows (per-config n is tiny — read with care)", fontsize=10.5)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
save(fig, "fig3_cv_auc.png")
for col in res:
    print(f"  {col:10s} AUC {res[col][0]:.3f} balanced {res[col][1]:.3f} acc {res[col][2]:.3f}")

# 4 -- why the v1 threshold failed: official rows are easier than the (balanced) validation/test populations -------------
thr1 = -0.336 / 2.142  # v1 logistic: logit = -0.336 - 2.142 * len_fused  ->  robust iff len_fused <= -0.157
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for k, (lab, col) in enumerate(((1, ROBUST), (0, NON))):
    g = rows[rows.y == lab]
    ax.scatter(k + jitter(len(g), 20 + k), g.len_fused, c=col, s=44, alpha=0.85, edgecolor="white", lw=0.6)
    ax.hlines(g.len_fused.median(), k - 0.3, k + 0.3, color="black", lw=2)
ax.axhline(thr1, color="#b5651d", ls="--", lw=1.6)
ax.axhline(0.0, color="#264653", ls="-", lw=1.6)
ax.text(1.38, thr1 - 0.03, f"v1: robust iff\nlen_fused ≤ {thr1:.2f}", color="#b5651d", ha="left", va="top", fontsize=8.5)
ax.text(1.38, 0.03, "v2: robust iff\nlen_fused ≤ 0", color="#264653", ha="left", va="bottom", fontsize=8.5)
ax.set_xlim(-0.5, 2.05)
ax.set_xticks([0, 1], [f"robust (n={int(rows.y.sum())})", f"non-robust (n={int((1 - rows.y).sum())})"])
ax.set_ylabel("len_fused (centered per model on the pool median)")
ax.set_title(f"Official rows skew easy: median {rows.len_fused.median():+.2f}, {rows.y.mean():.0%} robust", fontsize=10.5)
save(fig, "fig4_threshold_shift.png")
