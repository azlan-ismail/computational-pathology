"""Fig. 6: qualitative example of edge explanations (Stage 7 output). Usage: python -I make_fig6_qualitative.py

Inputs (stage7/): fig_compact.json (graph, edge selections and probabilities from
stage7_figure_data.json) and tumour.png / normal.png (the 96 x 96 PCam test patches)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(ROOT, "figures")
SRC = os.path.join(RESULTS, "stage7_figure")
os.makedirs(OUTDIR, exist_ok=True)

INK, MUTED = "#1f1f1e", "#6b6a63"
KEEP = "#ffd23f"          # selected edges
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "savefig.dpi": 300})

D = json.load(open(os.path.join(SRC, "fig_compact.json")))

# (title, model, selection key, removal-probability key, keep-probability key)
COLS = [
    ("Patch and entity graph", "M3", None, None, None),
    ("M3: PGExplainer\n(ties broken by index)", "M3", "pgx_index", "p_drop_pgx", "p_keep_pgx"),
    ("M3: gradient saliency", "M3", "kept_sal", "p_drop_sal", "p_keep_sal"),
    ("M3: random", "M3", "kept_rand", "p_drop_rand", "p_keep_rand"),
    ("M8: gradient saliency", "M8", "kept_sal", "p_drop_sal", "p_keep_sal"),
]
ROWS = [("tumour", "Tumour patch"), ("normal", "Normal patch")]


def selection(ex, model, key):
    m = ex["m"][model]
    if key == "pgx_index":
        # Stage 3 ranking: every PGExplainer score is tied (n_unique = 1), so a stable sort
        # keeps the first k edges in index order.
        assert m["n_unique"] == 1
        return set(range(len(m["kept_sal"])))
    return set(m[key])


fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.55))
for r, (name, rlabel) in enumerate(ROWS):
    ex = D[name]
    img = np.asarray(Image.open(os.path.join(SRC, f"{name}.png")).convert("RGB"))
    xy = np.array([[c[1], c[0]] for c in ex["c"]], dtype=float)      # centroids are (row, col)
    segs = np.array([[xy[a], xy[b]] for a, b in ex["e"]])
    for c, (title, model, key, kd, kk) in enumerate(COLS):
        ax = axes[r, c]
        ax.imshow(img, interpolation="nearest")
        if key is None:
            ax.add_collection(LineCollection(segs, colors="white", linewidths=0.9, alpha=0.9))
            note = f"full graph: M3 {ex['m']['M3']['p_full']:.2f}, M8 {ex['m']['M8']['p_full']:.2f}"
        else:
            sel = selection(ex, model, key)
            keep = np.array([i in sel for i in range(len(segs))])
            ax.add_collection(LineCollection(segs[~keep], colors="white", linewidths=0.5, alpha=0.55))
            ax.add_collection(LineCollection(segs[keep], colors=INK, linewidths=2.0))
            ax.add_collection(LineCollection(segs[keep], colors=KEEP, linewidths=1.2))
            m = ex["m"][model]
            note = f"removed {m[kd]:.2f}  |  kept only {m[kk]:.2f}"
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c="white", edgecolors=INK, linewidths=0.5, zorder=3)
        ax.set_xlim(-0.5, 95.5); ax.set_ylim(95.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=MUTED)
        if r == 0:
            ax.set_title(title, fontsize=7.5, color=INK, pad=4)
        if c == 0:
            ax.set_ylabel(rlabel, fontsize=7.5, color=INK, labelpad=4)

fig.subplots_adjust(left=0.035, right=0.995, top=0.88, bottom=0.06, wspace=0.05, hspace=0.22)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig6_qualitative.{ext}"))
print("saved")
