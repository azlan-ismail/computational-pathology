"""Fig. 3: test accuracy against training cost, including the entity-set model and the GAT models.

Usage: python -I make_fig3_cost.py
Reads results/interim_data.py (GraphSAGE configurations, Stage 2/4 logs) and the Stage 5 and Stage 6 summaries."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUTDIR = os.path.join(ROOT, "figures")
sys.path.insert(0, RESULTS)
import interim_data as d  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a4945"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300,
})

# accuracy (mean %, sd) and resources (train s/epoch, peak MB)
acc = {k: (v[0], v[1]) for k, v in d.STAGE2.items()}
res = {k: (v[0], v[1]) for k, v in d.BENCH.items()}
s5 = json.load(open(os.path.join(RESULTS, "stage5_entity_set.json")))
acc["Entity-set"] = (100 * s5["N128"]["test_accuracy"]["mean"], 100 * s5["N128"]["test_accuracy"]["sd"])
res["Entity-set"] = (s5["resources"]["train_epoch_seconds"], s5["resources"]["peak_gpu_mb_train"])
s6 = json.load(open(os.path.join(RESULTS, "stage6_gat.json")))
for k, lab in (("A8", "A8"), ("A_noedges", "A-none")):
    acc[lab] = (100 * s6[k]["test_accuracy"]["mean"], 100 * s6[k]["test_accuracy"]["sd"])
    res[lab] = (s6["resources"][k]["train_epoch_seconds"], s6["resources"][k]["peak_gpu_mb_train"])

POOL = {"M3", "M5", "M6", "M7"}
LABEL = {"B_noedges": "No-edge"}
STYLE = {"B_noedges": (AQUA, "s", True), "Entity-set": (AQUA, "s", False),
         "A8": (GRAY, "D", True), "A-none": (GRAY, "D", False)}
NAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "B_noedges", "Entity-set", "A8", "A-none"]
# label offsets (points) to keep direct labels clear of neighbouring markers
OFFSETS = ({"Entity-set": (0, 7), "M4": (5, -11), "M1": (0, 7), "B_noedges": (0, 7), "M7": (5, 4),
            "M2": (5, -10), "M5": (5, -10), "M8": (5, 3), "A-none": (6, 3), "A8": (6, 3)},
           {"Entity-set": (0, 7), "M1": (6, 3), "M5": (-6, -9), "M4": (6, -2), "M7": (6, 4),
            "B_noedges": (6, -10), "A-none": (6, 3), "A8": (-6, 3), "M8": (-6, 3), "M2": (6, -2), "M3": (6, -2)})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
for ax, idx, xl in ((axes[0], 0, "Training time per epoch (s)"), (axes[1], 1, "Peak GPU memory, training (MB)")):
    for k in NAMES:
        colour, marker, filled = STYLE.get(k, (ORANGE, "^", True) if k in POOL else (BLUE, "o", True))
        ax.errorbar(res[k][idx], acc[k][0], yerr=acc[k][1], fmt=marker, ms=6, color=colour, ecolor=colour,
                    elinewidth=1, capsize=0, mec=colour if not filled else "white", mew=1.2 if not filled else 1,
                    mfc=colour if filled else "white", zorder=3)
        off = OFFSETS[idx].get(k, (5, 3))
        ax.annotate(LABEL.get(k, k), (res[k][idx], acc[k][0]), xytext=off, textcoords="offset points",
                    fontsize=7, color=INK, ha="right" if off[0] < 0 else ("center" if off[0] == 0 else "left"),
                    va="bottom" if off[0] == 0 else "baseline", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel(xl)
axes[0].set_xlim(6, 27)
axes[0].set_xticks([10, 15, 20, 25])
axes[1].set_xlim(0, 290)
axes[0].set_ylabel("Test accuracy (%), mean ± SD over 5 seeds")
for ax in axes:
    ax.axhline(acc["B_patch_mlp"][0], color=MUTED, lw=1, ls=":")
axes[1].annotate("whole-patch MLP (no graph)", (axes[1].get_xlim()[1], acc["B_patch_mlp"][0]),
                 xytext=(-2, -9), textcoords="offset points", ha="right", fontsize=7, color=MUTED)
handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, mec="white", label="GraphSAGE, no pooling"),
           plt.Line2D([], [], marker="^", ls="", color=ORANGE, mec="white", label="GraphSAGE, Top-K pooling"),
           plt.Line2D([], [], marker="s", ls="", color=AQUA, mec="white", label="GraphSAGE, no edges"),
           plt.Line2D([], [], marker="D", ls="", color=GRAY, mec="white", label="GAT, with / without edges")]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.86))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig3_accuracy_cost.{ext}"))
print("saved")
