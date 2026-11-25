"""Fig. 1: pipeline with a real PCam test patch (the Stage 7 tumour example). Usage: python -I make_fig1.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(ROOT, "figures")
SRC = os.path.join(RESULTS, "stage7_figure")
INK, MUTED = "#1f1f1e", "#6b6a63"
RED_FILL, RED_EDGE = "#fdeadf", "#b85a2a"       # reduction points
BOX_FILL, BOX_EDGE = "#f3f2ee", "#6b6a63"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "savefig.dpi": 300})

ex = json.load(open(os.path.join(SRC, "fig_compact.json")))["tumour"]
img = np.asarray(Image.open(os.path.join(SRC, "tumour.png")).convert("RGB"))
rc = np.array(ex["c"], dtype=float)                       # (row, col)
xy = rc[:, ::-1]
segs = np.array([[xy[a], xy[b]] for a, b in ex["e"]])

W, H = 7.2, 2.75
fig = plt.figure(figsize=(W, H))
S = 1.12                                                    # panel size (in)
y0 = 0.86                                                   # panel bottom (in)
xs = [0.10, 1.52, 2.94, 4.36]                               # panel left (in)
ry, rh = 0.08, 0.46                                         # bottom row (in)


def axes_in(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def box(x, y, w, h, text, fill=BOX_FILL, edge=BOX_EDGE, dashed=False, size=7.8):
    p = FancyBboxPatch((x / W, y / H), w / W, h / H, boxstyle="round,pad=0.004,rounding_size=0.012",
                       transform=fig.transFigure, fc=fill, ec=edge, lw=0.8, ls="--" if dashed else "-")
    fig.add_artist(p)
    fig.text((x + w / 2) / W, (y + h / 2) / H, text, ha="center", va="center", fontsize=size,
             color=INK, linespacing=1.25)


def arrow(x1, y1, x2, y2, dashed=False, color=INK, rad=0.0):
    a = FancyArrowPatch((x1 / W, y1 / H), (x2 / W, y2 / H), transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=7, lw=0.9, color=color, ls="--" if dashed else "-",
                        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    fig.add_artist(a)


def panel(i, title):
    ax = axes_in(xs[i], y0, S, S)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    fig.text((xs[i] + S / 2) / W, (y0 + S + 0.08) / H, title, ha="center", va="bottom", fontsize=7.8,
             color=INK, linespacing=1.25)
    return ax


# (a) patch, with the central 32 x 32 region that defines the label
ax = panel(0, "H&E patch\n96×96 px")
ax.imshow(img, interpolation="nearest")
ax.add_patch(plt.Rectangle((31.5, 31.5), 32, 32, fill=False, ec="white", lw=0.8, ls=(0, (2, 1.5))))

# (b) entities
ax = panel(1, "Entities: dark regions\n(darkest 20%, ≥ 4 px)")
ax.imshow(img, interpolation="nearest", alpha=0.55)
ax.scatter(xy[:, 0], xy[:, 1], s=11, c="white", edgecolors=INK, linewidths=0.6, zorder=3)
ax.set_xlim(-0.5, 95.5); ax.set_ylim(95.5, -0.5)

# (c) crops -> encoder
ax = panel(2, "32×32 crops → frozen\nMobileNetV3-Small (576-d)")
ax.axis("off")
pad = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode="reflect")
pick = [7, 12, 18, 20, 26, 28, 31, 33, 35]
g, gap = 3, 3
tile = np.full((g * 32 + (g - 1) * gap, g * 32 + (g - 1) * gap, 3), 255, np.uint8)
for n, k in enumerate(pick):
    r, c = int(round(rc[k, 0])), int(round(rc[k, 1]))
    i, j = divmod(n, g)
    tile[i * (32 + gap):i * (32 + gap) + 32, j * (32 + gap):j * (32 + gap) + 32] = pad[r:r + 32, c:c + 32]
ax.imshow(tile, interpolation="nearest")

# (d) graph
ax = panel(3, "Graph: Delaunay edges\nbetween centroids")
ax.imshow(img, interpolation="nearest", alpha=0.55)
ax.add_collection(LineCollection(segs, colors=INK, linewidths=0.6, alpha=0.85))
ax.scatter(xy[:, 0], xy[:, 1], s=11, c="white", edgecolors=INK, linewidths=0.6, zorder=3)
ax.set_xlim(-0.5, 95.5); ax.set_ylim(95.5, -0.5)

mid = y0 + S / 2
for i in range(3):
    arrow(xs[i] + S + 0.05, mid, xs[i + 1] - 0.05, mid)

# (e) classifier stack, from the top of the panels to the bottom row
gx, gw = 5.72, 1.46
bh, top = 0.36, y0 + S
gapb = (top - ry - 4 * bh) / 3
yb = [top - bh - n * (bh + gapb) for n in range(4)]
box(gx, yb[0], gw, bh, "2× SAGEConv (64)\nor 2× GATConv (4×16)")
box(gx, yb[1], gw, bh, "ELR: Top-K pooling (0.5)", RED_FILL, RED_EDGE, dashed=True)
box(gx, yb[2], gw, bh, "Mean readout")
box(gx, yb[3], gw, bh, "MLP 64–32–2\n→ tumor / normal")
ya = yb[0] + bh / 2
arrow(xs[3] + S + 0.05, ya, gx - 0.04, ya)
arrow(gx + gw / 2, yb[0] - 0.02, gx + gw / 2, yb[1] + bh + 0.02, dashed=True, color=RED_EDGE)
arrow(gx + gw / 2, yb[1] - 0.02, gx + gw / 2, yb[2] + bh + 0.02, dashed=True, color=RED_EDGE)
arrow(gx + gw / 2, yb[2] - 0.02, gx + gw / 2, yb[3] + bh + 0.02)
arrow(gx + gw + 0.03, yb[0] + bh / 2, gx + gw + 0.03, yb[2] + bh / 2, rad=-0.5)   # path without pooling
fig.text((gx + gw / 2) / W, (y0 + S + 0.08) / H, "GNN classifier", ha="center", va="bottom", fontsize=7.8,
         color=INK)

# feature and topology reductions
box(xs[2] + 0.02, ry, S - 0.04, rh, "FLR\nPCA 576 → 128", RED_FILL, RED_EDGE)
arrow(xs[2] + S / 2, y0 - 0.04, xs[2] + S / 2, ry + rh + 0.02, dashed=True, color=RED_EDGE)
box(xs[3] + 0.02, ry, S - 0.04, rh, "GLR\nfeature-space\nk-NN (k = 6)", RED_FILL, RED_EDGE)
arrow(xs[3] + S / 2, ry + rh + 0.02, xs[3] + S / 2, y0 - 0.04, dashed=True, color=RED_EDGE)
arrow(xs[2] + S - 0.02, ry + rh / 2, xs[3] + 0.02, ry + rh / 2, dashed=True, color=RED_EDGE)
# reduced node features enter the graph
arrow(xs[2] + S - 0.12, ry + rh + 0.02, xs[3] + 0.16, y0 - 0.04, dashed=True, color=RED_EDGE)
# no-edge baseline
box(xs[0] + 0.02, ry, 2.5, rh,
    "No-edge baseline: same pipeline\nand network, empty edge set;\nwith FLR: entity-set model", dashed=True)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig1_pipeline.{ext}"), bbox_inches="tight", pad_inches=0.03)
print("saved")
