"""Fig. 3: reduction-strength sweeps (accuracy and training time per mechanism)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from make_figs import BLUE, ORANGE, INK, MUTED, OUTDIR   # shared style
from stage4_data import RUNS

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
    fig.savefig(os.path.join(OUTDIR, f"fig3_sweeps.{ext}"))
print("fig3 written")
