# Graph-contribution assessment protocol for a hybrid CNN-GNN on PatchCamelyon

Code, notebooks and result summaries for
"Does the graph matter? A graph-contribution assessment protocol for hybrid CNN-GNN models, applied to reduction
trade-offs in histopathology patch classification".

## Contents

| Folder | What it holds |
|---|---|
| `notebooks/` | The six Kaggle notebooks, in run order (GPU T4 x2, Internet on) |
| `scripts/` | The same code as plain Python scripts (`# %%` cells) and the templates they were built from |
| `tests/` | Local tests for each stage and the script-to-notebook build tools |
| `results/` | Result summaries transcribed from the Kaggle run logs |
| `figures_code/` | Scripts that draw Figures 2-4 from `results/` |
| `figures/` | The figures those scripts produce |

## Pipeline

1. `stage1_prepare_pcam` - downloads PatchCamelyon (MD5-checked), detects entities, encodes them with a frozen
   MobileNetV3-Small and stores the entity graphs (8-bit node features, Delaunay edges).
2. `stage2_train_pcam` - hyperparameter tuning, then M1-M8 and three baselines with five seeds; resource benchmarks.
3. `stage3_fidelity_pcam` - PGExplainer fidelity against random edges on 1,000 test graphs.
4. `stage4_sweeps_stats_pcam` - reduction-strength sweeps, hierarchical bootstrap, Holm adjustment.
5. `stage5_entityset_pcam` - the entity-set model (PCA-128 node features, no edges).
6. `stage6_gat_pcam` - the protocol repeated with a graph attention network (GAT): tuning, A8 and A-none with five seeds,
   PGExplainer fidelity against random edges.

Each later notebook takes the outputs of the earlier ones as Kaggle "Notebook output" inputs.

## What is not in this repository

Trained checkpoints, per-patch test predictions and the full `results.json` files are stored as outputs of the
Kaggle notebook versions; they are too large for a Git repository. `results/` contains the summary numbers used
in the manuscript, transcribed from the run logs. The Stage 5 outputs are saved in Version 3 of its notebook, from an interactive run.

## Tests

    python -I tests/test_stage4.py scripts
