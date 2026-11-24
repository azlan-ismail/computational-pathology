"""Figures for the manuscript. Usage: python -I make_figs.py [results.json]

Without results.json the interim numbers transcribed from the Kaggle logs are used."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
os.makedirs(OUTDIR, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300,
})
POOL = {"M3", "M5", "M6", "M7"}
LABEL = {"B_noedges": "No-edge", "M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4", "M5": "M5",
         "M6": "M6", "M7": "M7", "M8": "M8"}


def load():
    if len(sys.argv) > 1:
        r = json.load(open(sys.argv[1]))
        acc = {k: (100 * v["test"]["accuracy"]["mean"], 100 * v["test"]["accuracy"]["sd"])
               for k, v in r["configs"].items()}
        res = {k: (v["train_epoch_seconds"]["mean"], v["peak_gpu_mb_train"]["mean"])
               for k, v in r["resources"].items()}
        fid = {}
        for k, v in r["fidelity"].items():
            fid[k] = {m: [(row[m]["fid_plus"], row[m]["fid_minus"]) for row in v["by_sparsity"]]
                      for m in ("pgexplainer", "random")}
        sp = [row["sparsity"] for row in next(iter(r["fidelity"].values()))["by_sparsity"]]
        return acc, res, fid, sp
    sys.path.insert(0, HERE)
    import interim_data as d
    acc = {k: (v[0], v[1]) for k, v in d.STAGE2.items()}
    res = {k: (v[0], v[1]) for k, v in d.BENCH.items()}
    fid = {k: {m: [(t[0], t[1]) for t in rows] for m, rows in v.items()} for k, v in d.FID.items()}
    return acc, res, fid, d.SPARSITY


# label offsets (points) to keep direct labels clear of neighbouring markers
OFFSETS = ({"M4": (-5, 4), "M1": (5, 4), "M2": (5, -10), "M5": (5, -10), "M7": (5, 4)},
           {"M7": (6, 4), "M1": (-6, 3), "M4": (6, -2), "M5": (-6, -9), "M2": (6, -2), "M3": (6, -2), "M8": (-6, 3)})


def fig_cost(acc, res):
    names = [k for k in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "B_noedges"] if k in res]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, idx, xl in ((axes[0], 0, "Training time per epoch (s)"), (axes[1], 1, "Peak GPU memory, training (MB)")):
        for k in names:
            colour, marker = (AQUA, "s") if k == "B_noedges" else ((ORANGE, "^") if k in POOL else (BLUE, "o"))
            ax.errorbar(res[k][idx], acc[k][0], yerr=acc[k][1], fmt=marker, ms=6, color=colour,
                        ecolor=colour, elinewidth=1, capsize=0, mec="white", mew=1)
            off = OFFSETS[idx].get(k, (5, 3))
            ax.annotate(LABEL[k], (res[k][idx], acc[k][0]), xytext=off, textcoords="offset points",
                        fontsize=7, color=INK, ha="right" if off[0] < 0 else "left")
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Test accuracy (%), mean ± SD over 5 seeds")
    for ref, ls in (("B_patch_mlp", ":"),):
        if ref in acc:
            for ax in axes:
                ax.axhline(acc[ref][0], color=MUTED, lw=1, ls=ls)
            axes[1].annotate("whole-patch MLP (no graph)", (axes[1].get_xlim()[1], acc[ref][0]),
                             xytext=(-2, -9), textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Graph model, no pooling"),
               plt.Line2D([], [], marker="^", ls="", color=ORANGE, label="Graph model, Top-K pooling"),
               plt.Line2D([], [], marker="s", ls="", color=AQUA, label="Same network, no edges")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig2_accuracy_cost.{ext}"))
    plt.close(fig)


def fig_fidelity(fid, sp):
    names = [k for k in ["M8", "M1", "M2", "M4", "M3", "M5", "M6", "M7"] if k in fid]
    fig, axes = plt.subplots(2, len(names), figsize=(7.2, 3.6), sharex=True, sharey="row")
    for j, k in enumerate(names):
        for i, (lab, col) in enumerate((("Fid$^+$ (removal)", 0), ("Fid$^-$ (keep only)", 1))):
            ax = axes[i, j]
            for m, colour, marker in (("pgexplainer", BLUE, "o"), ("random", ORANGE, "^")):
                ax.plot(sp, [t[col] for t in fid[k][m]], marker=marker, ms=3.5, lw=1.5, color=colour)
            ax.axhline(0, color=MUTED, lw=0.8)
            if i == 0:
                ax.set_title(k + (" (pool)" if k in POOL else ""), fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 1:
                ax.set_xticks(sp)
                ax.set_xticklabels([f"{int(100 * s)}" for s in sp], fontsize=6.5)
    fig.supxlabel("Edges kept in the explanation (%)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", color=BLUE, label="PGExplainer edges"),
               plt.Line2D([], [], marker="^", color=ORANGE, label="Random edges (same number)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig4_fidelity.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    acc, res, fid, sp = load()
    fig_cost(acc, res)
    fig_fidelity(fid, sp)
    print("figures written to", OUTDIR)
