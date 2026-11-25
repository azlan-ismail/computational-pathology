"""Fig. 5: random-referenced fidelity (Stage 3b). Usage: python -I make_fig5_fidelity.py

Input: stage3b_summary_rounded.json (stage3b_results/stage3b_summary.json, values to 3 significant digits)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(ROOT, "figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300,
})
S = json.load(open(os.path.join(RESULTS, "stage3b_summary_rounded.json")))
SP = S["sparsity"]
C = S["configs"]
ORDER = ["M8", "M1", "M2", "M4", "M3", "M5", "M6", "M7", "A8"]
TITLE = {"M8": "M8\nnone", "M1": "M1\nF", "M2": "M2\nG", "M4": "M4\nF+G", "M3": "M3\nE (pool)",
         "M5": "M5\nF+E (pool)", "M6": "M6\nG+E (pool)", "M7": "M7\nF+G+E (pool)", "A8": "A8\nGAT"}
METHODS = (("saliency", BLUE, "o", "Gradient saliency"), ("pgexplainer", ORANGE, "^", "PGExplainer (ties broken at random)"))

fig, axes = plt.subplots(2, len(ORDER), figsize=(7.2, 3.7), sharex=True, sharey="row")
for j, k in enumerate(ORDER):
    for i, (key, lab) in enumerate((("dfid_plus", "$\\Delta$Fid$^+$ (removal)"), ("dfid_minus", "$\\Delta$Fid$^-$ (keep only)"))):
        ax = axes[i, j]
        ax.axhline(0, color=MUTED, lw=0.8)
        for m, col, mk, _ in METHODS:
            v = [C[k]["delta"][f"{m}_{s}_{key}"] for s in SP]
            ax.fill_between(SP, [x[1] for x in v], [x[2] for x in v], color=col, alpha=0.18, lw=0)
            ax.plot(SP, [x[0] for x in v], marker=mk, ms=3, lw=1.4, color=col)
        if i == 0:
            ax.set_title(TITLE[k], fontsize=7, color=INK)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 1:
            ax.set_xticks(SP)
            ax.set_xticklabels([f"{int(100 * s)}" if s != 0.2 else "" for s in SP], fontsize=6)  # 20 unlabeled: too close to 10
        ax.tick_params(axis="y", labelsize=6.5)
fig.supxlabel("Edges kept in the explanation (%)", fontsize=8)
handles = [plt.Line2D([], [], marker=mk, color=col, label=lab) for _, col, mk, lab in METHODS]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig5_fidelity.{ext}"))
print("saved")
