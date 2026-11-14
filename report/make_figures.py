"""Figures for the technical report (vector PDF for LaTeX + PNG preview).

fig_pipeline : method overview
fig_main     : (a) official train rows: len_fused per config by label, v2 (0) and v3 (theta_c) thresholds
               (b) pool: share of problems reliably solved vs relative reasoning-length quintile, gpt-oss efforts
Usage: python report/make_figures.py [--report runs/v1/report --art runs/v3/art --gen runs/pod/runs/gen]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--report", default=str(ROOT / "runs/v1/report"))
ap.add_argument("--art", default=str(ROOT / "runs/v3/art"))
ap.add_argument("--gen", default=str(ROOT / "runs/pod/runs/gen"))
ap.add_argument("--out", default=str(ROOT / "report/figures"))
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# palette (validated: dataviz validate_palette.js, light, surface #ffffff, all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "pdf.fonttype": 42, "savefig.dpi": 300, "figure.facecolor": SURF,
    "axes.facecolor": SURF})


def save(fig, name, tight=True):
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)


def check_inside(fig, name):
    # figures saved at their exact size (LaTeX scale 1) clip anything drawn past the canvas, so fail instead.
    # Axis extents are taken from the axes themselves: fig.get_tightbbox shrinks x-labels to 1 px wide.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fig.texts]
    for ax in fig.axes:
        boxes += [c.get_window_extent(r) for c in ax.get_children() if c.get_visible()
                  and (isinstance(c, Legend) or isinstance(c, Text) and c.get_text())]  # empty titles sit off-canvas
        boxes += [axis.get_tightbbox(r) for axis in (ax.xaxis, ax.yaxis) if ax.axison]
    w, h = fig.bbox.width, fig.bbox.height
    bad = [b.bounds for b in boxes if b is not None and (b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > w + 0.5 or b.y1 > h + 0.5)]
    assert not bad, f"{name}: drawn outside the {w:.0f} x {h:.0f} px canvas: {bad}"


def safe(m, e):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{m}__{e}")


NAMES = {("Qwen/Qwen3.5-4B", "default"): "Qwen3.5-4B", ("Skywork/Skywork-OR1-Math-7B", "default"): "Skywork-OR1-7B",
         ("allenai/Olmo-3-7B-Think", "default"): "Olmo-3-7B-Think",
         ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"): "DeepSeek-R1-8B",
         ("openai/gpt-oss-120b", "low"): "gpt-oss low", ("openai/gpt-oss-120b", "medium"): "gpt-oss medium",
         ("openai/gpt-oss-120b", "high"): "gpt-oss high"}


# ---------------------------------------------------------------------------------------------------------
def fig_pipeline():
    # Laid out in points on a canvas exactly \linewidth wide (LaTeX places it at scale 1). Each box is sized from its
    # measured text with a fixed line pitch and equal padding; the column gaps share the remaining width.
    W, FS, FS_NOTE, PITCH, PADX, PADY, M = 5.5 * 72, 6.5, 6.0, 1.3, 5.5, 4.0, 1.0
    labels = {"in": "problem $x$\nmodel $m$",
              "pass": "prompt forward pass\nstate $h$ at layer $L_m$",
              "probe": "length probe$\\to\\hat z$\n(PCA-32, ridge)",
              "samp": "$K$ budgeted samples\nup to $B_m$ tokens",
              "prog": "progress probe$\\to\\hat\\ell$\non the state at $B_m$",
              "fuse": "per-model fusion\n(unlabelled pool)",
              "out": "robust iff\n$s_m(x)\\leq\\theta_m$"}
    note_text = "$\\theta_m$: 0.7-quantile\nof $s_m$ over pool\nproblems that $m$\nreliably solves"
    cols = [["in"], ["pass", "samp"], ["probe", "prog"], ["fuse"], ["out"]]
    fig = plt.figure(figsize=(W / 72, 1.5))
    rend, px = fig.canvas.get_renderer(), fig.dpi / 72

    def block(text, fs):  # per line: text, width, ascent, descent (pt); block height at the fixed line pitch
        met = []
        for line in text.split("\n"):
            t = fig.text(0, 0.5, line, fontsize=fs, va="baseline")
            e, base = t.get_window_extent(rend), 0.5 * fig.bbox.height
            met.append((line, e.width / px, (e.y1 - base) / px, (base - e.y0) / px))
            t.remove()
        return {"met": met, "fs": fs, "w": max(m[1] for m in met),
                "h": met[0][2] + PITCH * fs * (len(met) - 1) + met[-1][3]}

    blk = {k: block(v, FS) for k, v in labels.items()}
    note = block(note_text, FS_NOTE)
    colw = [max(blk[k]["w"] for k in c) + 2 * PADX for c in cols]
    colw[-1] = max(colw[-1], note["w"])  # decision box as wide as the note under it, so their edges line up
    gap = (W - 2 * M - sum(colw)) / (len(cols) - 1)
    assert gap >= 14, f"fig_pipeline: labels too wide, column gap {gap:.1f} pt"
    bh = max(b["h"] for b in blk.values()) + 2 * PADY
    dv = max(1.1 * bh, note["h"] + 3)  # row offset; the note has to end above the bottom row's lower edge
    H = 2 * (dv + M) + bh
    fig.set_size_inches(W / 72, H / 72)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    def text(b, cx, cy, color):  # block centred on (cx, cy)
        y = cy + b["h"] / 2 - b["met"][0][2]
        for line, *_ in b["met"]:
            ax.text(cx, y, line, fontsize=b["fs"], ha="center", va="baseline", color=color)
            y -= PITCH * b["fs"]

    x, ym, pos = M, H / 2, {}
    for c, w in zip(cols, colw):
        for k, y in zip(c, [ym] if len(c) == 1 else [ym + dv, ym - dv]):
            pos[k] = (x + w / 2, y, w)
        x += w + gap
    for k, (cx, cy, w) in pos.items():
        last = k == "out"
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - bh / 2), w, bh, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc="#e6eefa" if last else "#f4f3ef", ec=BLUE if last else AXIS,
                                    lw=0.8 if last else 0.6))
        text(blk[k], cx, cy, INK)

    def arrow(a, b, dya=0.0, dyb=0.0):  # right edge of a -> left edge of b, 2 pt clear of both boxes
        (xa, ya, wa), (xb, yb, wb) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((xa + wa / 2 + 2, ya + dya), (xb - wb / 2 - 2, yb + dyb), arrowstyle="-|>",
                                     mutation_scale=6.5, lw=0.7, color=INK2, shrinkA=0, shrinkB=0))

    off = bh / 4
    arrow("in", "pass", off)
    arrow("in", "samp", -off)
    arrow("pass", "probe")
    arrow("samp", "prog")
    arrow("probe", "fuse", 0, off)
    arrow("prog", "fuse", 0, -off)
    arrow("fuse", "out")
    (xf, _, wf), (xo, _, wo) = pos["fuse"], pos["out"]
    ax.text((xf + wf / 2 + xo - wo / 2) / 2, ym + 1.5, "$s_m$", fontsize=FS, ha="center", va="bottom", color=INK2)
    text(note, xo, ym - bh / 2 - 3 - note["h"] / 2, INK2)
    check_inside(fig, "fig_pipeline")
    save(fig, "fig_pipeline", tight=False)


# ---------------------------------------------------------------------------------------------------------
def load_pool_reliability():
    problems = [json.loads(l) for l in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
    gens = {}
    for f in sorted(Path(args.gen).glob("*.jsonl")):
        if ".texts." not in f.name:
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
    rows = []
    for (m, e), g in gens.items():
        for pid, r in g.items():
            if truth.get(pid) is None:
                continue
            fin = [s for s in r["samples"] if s["finish"] == "stop"]
            rows.append({"cfg": (m, e), "pid": pid, "mlog": float(np.mean([np.log1p(s["len"]) for s in r["samples"]])),
                         "solved": len(fin) > 0 and all(s["ans"] == truth[pid] for s in fin)})
    d = pd.DataFrame(rows)
    d["rel"] = d.mlog - d.groupby("cfg").mlog.transform("median")
    return d


def fig_main():
    rows = pd.read_csv(Path(args.report) / "train_rows_fused.csv")
    arts = {f.stem: dict(np.load(f, allow_pickle=False)) for f in Path(args.art).glob("*.npz")}
    # exact \linewidth canvas; margins (in) leave room for tick labels, axis labels and two legend rows below
    W, H, L, R, T, B, GAP = 5.5, 2.9, 0.92, 0.04, 0.22, 0.82, 0.62
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=L / W, right=1 - R / W, top=1 - T / H, bottom=B / H,
                          wspace=GAP / ((W - L - R - GAP) / 2))
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) train rows, one row per config; rows left of XMIN drawn as '<' at the edge
    XMIN = -1.3
    order = list(NAMES)
    rng = np.random.default_rng(1)
    for i, cfg in enumerate(order):
        g = rows[(rows.model_id == cfg[0]) & (rows.reasoning_effort == cfg[1])]
        yy = len(order) - 1 - i
        for lab, col, mk in ((1, BLUE, "o"), (0, ORANGE, "^")):
            h = g[g.y == lab]
            jit = yy + rng.uniform(-0.17, 0.17, len(h))
            inside = (h.len_fused >= XMIN).values
            a.scatter(h.len_fused[inside], jit[inside], s=13, c=col, marker=mk, edgecolors=SURF, linewidths=0.6,
                      zorder=3, label=("robust" if lab else "non-robust") if i == 0 else None)
            a.scatter(np.full((~inside).sum(), XMIN + 0.03), jit[~inside], s=13, c=col, marker="<",
                      edgecolors=SURF, linewidths=0.6, zorder=3)  # off-scale rows (see caption)
        th = float(arts[safe(*cfg)]["cap_theta"])
        a.plot([th, th], [yy - 0.36, yy + 0.36], color=INK, lw=1.3, zorder=4, solid_capstyle="butt",
               label=r"$\theta_m$ (capability-aware)" if i == 0 else None)
    a.axvline(0, color=MUTED, lw=0.8, zorder=1, label=r"pool median ($\theta_m=0$)")
    a.set_yticks(range(len(order)))
    a.set_yticklabels([NAMES[c] for c in order][::-1])
    a.set_xlim(XMIN, rows.len_fused.max() + 0.08)
    a.set_xlabel(r"fused length score $s_m(x)$, official training cases")
    a.tick_params(axis="y", length=0, pad=6)  # keeps the off-scale '<' markers clear of the labels
    a.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.set_title("(a)", loc="left", fontsize=8)

    # (b) pool: reliably-solved share by relative-length quintile, gpt-oss efforts
    d = load_pool_reliability()
    for (eff, col, mk) in (("low", BLUE, "o"), ("medium", ORANGE, "s"), ("high", AQUA, "D")):
        g = d[d.cfg == ("openai/gpt-oss-120b", eff)].copy()
        g["q"] = pd.qcut(g.rel, 5, labels=False)
        s = g.groupby("q").solved.mean()
        b.plot(s.index + 1, s.values, color=col, lw=1.6, marker=mk, ms=4.2, mec=SURF, mew=0.7, zorder=3, label=eff)
    b.set_xlim(0.7, 5.3)
    b.set_ylim(0, 1)
    b.set_xticks([1, 2, 3, 4, 5])
    b.set_xlabel("length quintile (1 = shortest)")
    b.set_ylabel("share reliably solved")
    b.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    b.set_axisbelow(True)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.set_title("(b)", loc="left", fontsize=8)

    # both legends hang under their panel with aligned tops, 6 pt below the x-labels
    fig.align_xlabels([a, b])
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    top = min(p.xaxis.label.get_window_extent(rend).y0 for p in (a, b)) / fig.bbox.height - 6 / 72 / H
    kw = dict(loc="upper center", bbox_transform=fig.transFigure, frameon=False, handlelength=1.6,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0, borderpad=0)
    for p, extra in ((a, {"ncol": 2}), (b, {"ncol": 3, "title": "gpt-oss-120b reasoning effort", "title_fontsize": 7})):
        box = p.get_position()
        p.legend(bbox_to_anchor=((box.x0 + box.x1) / 2, top), **kw, **extra)
    check_inside(fig, "fig_main")
    save(fig, "fig_main", tight=False)


fig_pipeline()
fig_main()
print("wrote", sorted(p.name for p in OUT.iterdir()))
