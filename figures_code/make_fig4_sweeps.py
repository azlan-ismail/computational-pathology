"""Fig. 4: reduction-strength sweeps (accuracy and training time per mechanism). Usage: python -I make_fig4_sweeps.py"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(ROOT, "figures")
sys.path.insert(0, RESULTS)
from stage4_data import RUNS  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300,
})

panels = [
    ("PCA dimensions", [("F32", 32), ("F64", 64), ("M1", 128), ("F256", 256), ("M8", 576)],
     "M1", ["32", "64", "128", "256", "576\n(none)"]),
    ("$k$ of feature-space $k$-NN graph", [("G3", 3), ("M2", 6), ("G12", 12)], "M2", ["3", "6", "12"]),
    ("Top-$K$ pooling ratio", [("E25", 0.25), ("M3", 0.5), ("E75", 0.75), ("M8", 1.0)],
     "M3", ["0.25", "0.5", "0.75", "1.0\n(none)"]),
]
ref = 100 * RUNS["M8"]["acc"][0]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharey="row")
for c, (title, pts, default, ticks) in enumerate(panels):
    xs = list(range(len(pts)))
    acc = [100 * RUNS[k]["acc"][0] for k, _ in pts]; sd = [100 * RUNS[k]["acc"][1] for k, _ in pts]
    t = [RUNS[k]["t"] for k, _ in pts]
    for r, (ys, err, ylab) in enumerate(((acc, sd, "Test accuracy (%)"), (t, None, "Training time per epoch (s)"))):
        ax = axes[r, c]
        m = [i for i, (k, _) in enumerate(pts) if k != "M8"]      # points where the mechanism is active
        ax.plot([xs[i] for i in m], [ys[i] for i in m], color=BLUE, lw=2, zorder=2)
        if err:
            for i, (k, _) in enumerate(pts):
                ax.errorbar(xs[i], ys[i], yerr=err[i], fmt="none", ecolor=MUTED if k == "M8" else BLUE,
                            elinewidth=1, capsize=2, zorder=2)
            ax.axhline(ref, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
        for i, (k, _) in enumerate(pts):
            dflt = k == default
            if k == "M8":
                ax.plot(xs[i], ys[i], marker="o", ms=5, mfc="white", mec=MUTED, mew=1.4, zorder=3)
            else:
                ax.plot(xs[i], ys[i], marker="o", ms=7 if dflt else 5, color=ORANGE if dflt else BLUE,
                        mec="white", mew=1.2, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels(ticks)
        if c == 0: ax.set_ylabel(ylab)
        if r == 1: ax.set_xlabel(title)
axes[0, 0].set_ylim(76, 85)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig4_sweeps.{ext}"))
print("fig4 written")
