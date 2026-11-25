"""Assemble stage6_gat_pcam.py from the tested Stage 5 script and the Stage 3 / Stage 4 templates."""
import re
s5 = open("stage5_entityset_pcam.py").read()
cells = re.split(r"^# %%", s5, flags=re.M)[1:]          # markdown, config, shared, stats, main
shared = cells[2]
t3 = open("stage3_template.py").read()
fid = t3[t3.index("# ----------------------------------------------------------------------------------------------\n# Explanation subsets"):t3.index("# %%\ndef main():")]
t4 = open("stage4_template.py").read()
stats = t4[t4.index("# ----------------------------------------------------------------------------------------------\n# Part B"):t4.index("# %%\ndef main():")]

def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:70])
    return text.replace(a, b)

# ---- model: GAT instead of GraphSAGE
shared = rep(shared, '"""Two SAGEConv layers (64), optional Top-K pooling, global mean pool, 64-32-2 classifier."""',
             '"""Two GATConv layers (4 heads x 16 = 64), optional Top-K pooling, global mean pool, 64-32-2 classifier."""')
shared = rep(shared, "from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool",
             "from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool")
shared = rep(shared, "self.c1 = SAGEConv(in_dim, 64)", "self.c1 = GATConv(in_dim, 16, heads=4)")
shared = rep(shared, "self.c2 = SAGEConv(64, 64)", "self.c2 = GATConv(64, 16, heads=4)")
# ---- tuning may use the configurations of this notebook
shared = rep(shared, "table = {**CORE, **BASELINES}\n        for name in TUNE_CONFIGS:",
             "table = {**CORE, **BASELINES, **SWEEPS}\n        for name in TUNE_CONFIGS:")
# ---- benchmark with the job's own settings
shared = rep(shared, "opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)",
             "opt = torch.optim.AdamW(model.parameters(), lr=job.lr, weight_decay=job.wd)")

shared = rep(shared, "model = net_class()(tr.feats.shape[1], pool_ratio(job.pool), DROPOUT).to(dev)",
             "model = net_class()(tr.feats.shape[1], pool_ratio(job.pool), job.dropout).to(dev)")

# ---- statistics: GAT comparisons, fidelity read from this notebook's own output too
stats = rep(stats, 'sw = {k: v for k, v in probs.items() if k in SWEEPS or k in ("M1", "M2", "M3", "M8")}',
            'sw = {k: v for k, v in probs.items() if k in SWEEPS or k in ("M8", "B_noedges")}')
stats = rep(stats, 'hierarchical_bootstrap(sw, y_test, ["M8"], N_BOOT, BOOT_SEED + 1)',
            'hierarchical_bootstrap(sw, y_test, [r for r in ("A_noedges", "M8", "B_noedges") if r in sw], N_BOOT, BOOT_SEED + 3)')
stats = rep(stats, "    for folder in s3:\n        for path in glob.glob(os.path.join(folder, \"*\", \"fidelity_probs.npz\")):",
            "    for folder in s3 + [OUT]:\n        for path in glob.glob(os.path.join(folder, \"*\", \"fidelity_probs.npz\")):")
stats = rep(stats, 'res["stage4_prep"]', 'res["stage6_prep"]')

head = '''# %% [markdown]
# # Stage 6 - the graph-contribution assessment protocol on a second architecture (GAT)
#
# **Inputs (right-hand panel -> Add Input -> Notebook output):** the finished Stage 1 notebook
# (graphs), the finished Stage 2 notebook (`stage2_results`) and the finished Stage 3 notebook
# (`stage3_results`). Accelerator = GPU T4 x2, Internet = On.
#
# The network of Stage 2 with its two SAGEConv layers replaced by GATConv layers (4 heads x 16).
# 1. Hyperparameters are selected on the validation split (same grid as Stage 2, seed 0) for the
#    GAT with Delaunay edges (`A8`) and the same GAT without edges (`A_noedges`).
# 2. Both are trained with five seeds and benchmarked (Check 1 and Check 3).
# 3. PGExplainer fidelity against random edges is measured for `A8`, seeds 0-2, on the same 1,000
#    test graphs as Stage 3 (Check 2).
# 4. Hierarchical-bootstrap comparisons: every model against `A_noedges`, `M8` and `B_noedges`.
#
# Results are printed between `===RESULTS_JSON_BEGIN===` and `===RESULTS_JSON_END===` and saved to
# `stage6_results/results.json`. First time: `QUICK_TEST = True`.

# %%
QUICK_TEST = True

ARCH = "gat"
EXPERIMENTS = ["tune", "sweeps"]
SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 40
EVALS_PER_EPOCH = 4
PATIENCE = 8
BATCH = 128
LR, WEIGHT_DECAY, DROPOUT = 1e-3, 0.0, 0.3      # replaced by the tuned values before the main runs
TUNE_CONFIGS = ["A8", "A_noedges"]
TUNE_LR, TUNE_WD, TUNE_DROPOUT = [1e-3, 3e-4, 1e-4], [0.0, 1e-3], [0.0, 0.3]
EVAL_BATCH = 512
BENCH_REPS = 3
TIME_BUDGET_H = 9.0
KNN_MAX = 12
N_BOOT = 2000
BOOT_SEED = 2026

# explanation fidelity (as in Stage 3)
FID_CONFIGS = ["A8"]
EXPLAIN_SEEDS = [0, 1, 2]
N_FIT, N_EXPLAIN = 500, 1000
MIN_NODES, MIN_EDGES = 8, 10
PGX_EPOCHS, PGX_LR = 30, 0.003
SPARSITY = [0.1, 0.2, 0.3, 0.5]
SAMPLE_SEED = 12345

OUT = "/kaggle/working/stage6_results"
TMP = "/kaggle/temp/stage6"

CORE = {
    "M1": ("pca128", "delaunay", "none"),
    "M2": ("raw", "knn6", "none"),
    "M3": ("raw", "delaunay", "topk0.5"),
    "M4": ("pca128", "knn6", "none"),
    "M5": ("pca128", "delaunay", "topk0.5"),
    "M6": ("raw", "knn6", "topk0.5"),
    "M7": ("pca128", "knn6", "topk0.5"),
    "M8": ("raw", "delaunay", "none"),
}
BASELINES = {
    "B_noedges": ("raw", "none", "none"),
    "B_patch_linear": ("patch", "-", "linear"),
    "B_patch_mlp": ("patch", "-", "mlp"),
}
SWEEPS = {
    "A8": ("raw", "delaunay", "none"),
    "A_noedges": ("raw", "none", "none"),
}

if QUICK_TEST:
    SWEEP_SEEDS, MAX_EPOCHS, BENCH_REPS, N_BOOT = [0], 2, 1, 200
    TUNE_LR, TUNE_WD, TUNE_DROPOUT = [1e-3], [0.0], [0.0, 0.3]
    EXPLAIN_SEEDS, N_FIT, N_EXPLAIN, PGX_EPOCHS = [0], 40, 60, 2
    SUBSET = {"train": 20000, "valid": 4000, "test": 4000}
else:
    SUBSET = {"train": None, "valid": None, "test": None}

'''

main = '''# %%
def choose_hparams():
    """Lowest mean validation loss over the tuned configurations (validation data only)."""
    hp = {}
    for path in glob.glob(os.path.join(OUT, "T_*", "metrics.json")):
        with open(path) as f:
            r = json.load(f)
        if bool(r.get("quick_test")) != QUICK_TEST:
            continue
        key = (r["hparams"]["lr"], r["hparams"]["weight_decay"], r["hparams"]["dropout"])
        hp.setdefault(key, []).append(r["val"]["nll"])
    n = max(len(v) for v in hp.values())
    full = {k: float(np.mean(v)) for k, v in hp.items() if len(v) == n}
    best = min(full, key=full.get)
    with open(os.path.join(OUT, "tuning_choice.json"), "w") as f:
        json.dump({"lr": best[0], "weight_decay": best[1], "dropout": best[2], "arch": ARCH,
                   "mean_val_nll": {hp_tag(*k): v for k, v in full.items()}}, f, indent=2)
    log(f"tuned on validation: lr={best[0]:g}, weight_decay={best[1]:g}, dropout={best[2]:g} "
        f"(mean val loss {full[best]:.4f}; range {min(full.values()):.4f}-{max(full.values()):.4f})")
    return best


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    ensure_pyg()
    deadline = T_START + TIME_BUDGET_H * 3600
    jobs = all_jobs()
    tune_jobs = [j for j in jobs if j.exp == "tune"]
    main_jobs = [j for j in jobs if j.exp != "tune"]
    log(f"Stage 6 ({ARCH}): {len(tune_jobs)} tuning runs, {len(main_jobs)} main runs, {n_gpus()} GPU(s)")
    folder = find_stage1()
    splits = {k: load_split(folder, k, SUBSET[k]) for k in ("train", "valid", "test")}
    g_max, f_min = splits["train"].g_max, splits["train"].f_min
    run_child(prep_child, splits, [], [])
    data = {k: view(s, s.q, None) for k, s in splits.items()}

    # 1. tuning, then the tuned settings are applied to the main runs
    run_jobs([j for j in tune_jobs if not done(j)], data, "raw", g_max, f_min, deadline)
    lr, wd, do = choose_hparams()
    for j in main_jobs:
        j.lr, j.wd, j.dropout = lr, wd, do

    # 2. resource measurements and main runs
    seen, firsts = set(), []
    for j in main_jobs:
        if j.name not in seen:
            seen.add(j.name); firsts.append(j)
    if not os.path.exists(os.path.join(OUT, "benchmark_raw.json")):
        run_child(bench_child, firsts, data, "raw", g_max, f_min, "raw")
    run_jobs([j for j in main_jobs if not done(j)], data, "raw", g_max, f_min, deadline)
    left = [j.run_id for j in main_jobs if not done(j)]
    log(f"training finished; {len(left)} main runs not done" + (f": {left}" if left else ""))

    # 3. explanation fidelity on the same test graphs as Stage 3
    try:
        fjobs = [j for j in main_jobs if j.name in FID_CONFIGS and j.seed in EXPLAIN_SEEDS and done(j)]
        todo = [j for j in fjobs if not os.path.exists(os.path.join(OUT, j.run_id, "fidelity.json"))]
        if todo:
            fit_ids = pick_graphs(splits["valid"], N_FIT, SAMPLE_SEED)
            test_ids = pick_graphs(splits["test"], N_EXPLAIN, SAMPLE_SEED + 1)
            np.save(os.path.join(OUT, "explained_test_graph_ids.npy"), test_ids)
            log(f"PGExplainer fitted on {len(fit_ids)} validation graphs; {len(test_ids)} test graphs explained")
            ckpt = {j.run_id: os.path.join(OUT, j.run_id) for j in todo}
            views = {"raw": {"valid": data["valid"], "test": data["test"]}}
            check_parent_clean()
            k = max(1, n_gpus())
            procs = []
            for gpu in range(k):
                p = mp.get_context("fork").Process(
                    target=fid_worker, args=(gpu, todo[gpu::k], views, ckpt, fit_ids, test_ids, g_max, f_min, deadline))
                p.start(); procs.append(p)
            for p in procs:
                p.join()
        summarise_fidelity()
        fleft = [j.run_id for j in fjobs if not os.path.exists(os.path.join(OUT, j.run_id, "fidelity.json"))]
        log(f"fidelity finished; {len(fleft)} runs not done" + (f": {fleft}" if fleft else ""))
    except Exception:
        log("fidelity phase FAILED (training results are kept)")
        traceback.print_exc()
        fleft = ["fidelity phase failed"]

    # 4. statistics
    import h5py
    with h5py.File(os.path.join(folder, "pcam_test.h5"), "r") as f:
        y_test = f["y"][:].astype(np.int64)
    log("statistics")
    res = analysis(y_test)
    res["quick_test"] = QUICK_TEST
    res["arch"] = ARCH
    res["tuned"] = {"lr": lr, "weight_decay": wd, "dropout": do}
    res["runs_not_done"] = left
    res["fidelity_not_done"] = fleft
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("===RESULTS_JSON_BEGIN===")
    print(json.dumps(res, separators=(",", ":")))
    print("===RESULTS_JSON_END===")
    log("finished")


if __name__ == "__main__":
    main()
'''
out = head + "# %%" + shared + "# %%\n" + fid + "\n# %%\n" + stats + "\n" + main
open("stage6_gat_pcam.py", "w").write(out)
