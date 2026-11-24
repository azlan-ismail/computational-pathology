# %% [markdown]
# # Stage 4 - reduction-strength sweeps and statistical analysis
#
# **Inputs (right-hand panel -> Add Input -> Notebook output):** the finished Stage 1 notebook
# (graphs), the finished Stage 2 notebook (`stage2_results`) and the finished Stage 3 notebook
# (`stage3_results`). Accelerator = GPU T4 x2, Internet = On.
#
# Part A trains the reduction-strength sweeps with the tuned settings of Stage 2 (3 seeds each):
# PCA to 32, 64 and 256 dimensions (128 is M1), k-NN graphs with k = 3 and 12 (6 is M2), and
# Top-K pooling ratios 0.25 and 0.75 (0.5 is M3), plus their resource measurements.
#
# Part B (CPU only) computes, from the saved per-patch test predictions:
# * mean, SD and 95 % CI of each metric over seeds;
# * differences to M8 (no reduction) and to the no-edge network, with 95 % CIs and Holm-adjusted
#   p-values from a hierarchical bootstrap that resamples test patches (paired across
#   configurations) and seeds (independently per configuration);
# * fidelity of PGExplainer against random edge selection, with paired bootstrap CIs over graphs.
#
# Everything is printed between `===RESULTS_JSON_BEGIN===` and `===RESULTS_JSON_END===` and saved
# to `stage4_results/results.json`. First time: `QUICK_TEST = True`.

# %%
QUICK_TEST = True

EXPERIMENTS = ["sweeps"]
SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2]
MAX_EPOCHS = 40
EVALS_PER_EPOCH = 4
PATIENCE = 8
BATCH = 128
LR = 1e-3                 # tuned in Stage 2
WEIGHT_DECAY = 0.0        # tuned in Stage 2
DROPOUT = 0.3             # tuned in Stage 2
TUNE_CONFIGS, TUNE_LR, TUNE_WD, TUNE_DROPOUT = [], [], [], []
EVAL_BATCH = 512
BENCH_REPS = 3
TIME_BUDGET_H = 9.0
KNN_MAX = 12
N_BOOT = 2000
BOOT_SEED = 2026

OUT = "/kaggle/working/stage4_results"
TMP = "/kaggle/temp/stage4"

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
    "F32": ("pca32", "delaunay", "none"), "F64": ("pca64", "delaunay", "none"),
    "F256": ("pca256", "delaunay", "none"),
    "G3": ("raw", "knn3", "none"), "G12": ("raw", "knn12", "none"),
    "E25": ("raw", "delaunay", "topk0.25"), "E75": ("raw", "delaunay", "topk0.75"),
}

if QUICK_TEST:
    SWEEP_SEEDS, MAX_EPOCHS, BENCH_REPS, N_BOOT = [0], 2, 1, 200
    SWEEPS = {"F32": SWEEPS["F32"], "G3": SWEEPS["G3"]}
    SUBSET = {"train": 20000, "valid": 4000, "test": 4000}
else:
    SUBSET = {"train": None, "valid": None, "test": None}

# %%
# <<SHARED_STAGE2_CODE>>

# %%
# ----------------------------------------------------------------------------------------------
# Part B: statistics (numpy only)
# ----------------------------------------------------------------------------------------------
def t_ci(v, level=0.95):
    """Mean, SD and t-based CI over seeds."""
    from scipy import stats
    v = np.asarray(v, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return {"mean": m, "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(v))}
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * sd / np.sqrt(len(v)))
    return {"mean": m, "sd": sd, "lo": m - h, "hi": m + h, "n": int(len(v))}


def weighted_auc_sorted(y_sorted, w_sorted):
    """AUC (Mann-Whitney) for weighted samples already sorted by score (ascending).

    Ties in the score are not merged; with continuous probabilities they are negligible."""
    wn = w_sorted * (1 - y_sorted)
    wp = w_sorted * y_sorted
    neg_below = np.cumsum(wn) - wn
    tot_p, tot_n = wp.sum(), wn.sum()
    if tot_p == 0 or tot_n == 0:
        return float("nan")
    return float((wp * neg_below).sum() / (tot_p * tot_n))


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def hierarchical_bootstrap(probs, y, refs, n_boot, seed):
    """probs: name -> (seeds, N) array of P(class 1). Returns per-comparison accuracy/AUC differences.

    Each replicate draws one set of test patches (shared by all configurations, so comparisons are
    paired) and, for every configuration separately, a resample of its seeds."""
    rng = np.random.default_rng(seed)
    names = sorted(probs)
    N = len(y)
    correct = {k: ((v >= 0.5).astype(np.int64) == y).astype(np.float64) for k, v in probs.items()}
    order = {k: np.argsort(v, axis=1, kind="stable") for k, v in probs.items()}
    ysort = {k: y[o] for k, o in order.items()}
    acc = {k: np.empty(n_boot) for k in names}
    auc = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        for k in names:
            S = probs[k].shape[0]
            pick = rng.integers(0, S, S)
            acc[k][b] = float((correct[k][pick] @ w).mean() / N)
            auc[k][b] = float(np.mean([weighted_auc_sorted(ysort[k][s], w[order[k][s]]) for s in pick]))
    comps = []
    for ref in refs:
        if ref not in probs:
            continue
        for k in names:
            if k == ref:
                continue
            row = {"config": k, "reference": ref}
            for metric, arr in (("accuracy", acc), ("auc", auc)):
                d = arr[k] - arr[ref]
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                row[metric] = {"diff": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
                               "hi": float(np.percentile(d, 97.5)), "p": float(min(1.0, max(p, 1 / n_boot)))}
            comps.append(row)
    for ref in refs:
        for metric in ("accuracy", "auc"):
            idx = [i for i, r in enumerate(comps) if r["reference"] == ref]
            if idx:
                adj = holm([comps[i][metric]["p"] for i in idx])
                for i, a in zip(idx, adj):
                    comps[i][metric]["p_holm"] = float(a)
    return comps


def fidelity_stats(P_by_seed, sparsity, n_boot, seed):
    """P_by_seed: list of (graphs, columns) matrices (same graphs, columns as in Stage 3).

    Per graph, values are averaged over seeds; CIs come from resampling graphs."""
    rng = np.random.default_rng(seed)
    P = np.mean(np.stack(P_by_seed), axis=0)
    G = P.shape[0]
    full = P[:, 0]
    out, col = [], 1
    per = {}
    for method in ("pgexplainer", "random"):
        for s in sparsity:
            per[(method, s)] = (full - P[:, col + 1], full - P[:, col])   # (Fid+ per graph, Fid- per graph)
            col += 2
    boots = rng.integers(0, G, (n_boot, G))
    for s in sparsity:
        row = {"sparsity": s}
        for m in ("pgexplainer", "random"):
            fp, fm = per[(m, s)]
            row[m] = {"fid_plus": float(fp.mean()), "fid_minus": float(fm.mean())}
        for key, i in (("fid_plus", 0), ("fid_minus", 1)):
            d = per[("pgexplainer", s)][i] - per[("random", s)][i]
            bd = d[boots].mean(axis=1)
            p = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
            row[f"{key}_pgx_minus_random"] = {"diff": float(d.mean()), "lo": float(np.percentile(bd, 2.5)),
                                              "hi": float(np.percentile(bd, 97.5)),
                                              "p": float(min(1.0, max(p, 1 / n_boot)))}
        out.append(row)
    return out


def bench_summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    res = {}
    for k, rs in by.items():
        res[k] = {key: t_ci([r[key] for r in rs if r.get(key) is not None])
                  for key in ("train_epoch_seconds", "gpu_compute_seconds", "batch_assembly_seconds",
                              "peak_gpu_mb_train", "peak_gpu_mb_inference", "inference_ms_per_graph")}
        res[k].update({"parameters": rs[0]["parameters"], "input_dim": rs[0]["input_dim"],
                       "feature_bytes_per_node": rs[0]["feature_bytes_per_node"],
                       "mean_nodes_per_graph": rs[0]["mean_nodes_per_graph"],
                       "mean_directed_edges_per_graph": rs[0]["mean_directed_edges_per_graph"],
                       "gpu": rs[0]["gpu"], "reps": len(rs)})
    return res


def load_runs(folders):
    runs = {}
    for folder in folders:
        for path in glob.glob(os.path.join(folder, "*", "metrics.json")):
            with open(path) as f:
                m = json.load(f)
            if m.get("quick_test") or m.get("experiment") == "tune":
                continue
            d = os.path.dirname(path)
            runs[m["run_id"]] = (m, d)
    return runs


def analysis(y_test):
    s2 = sorted(glob.glob("/kaggle/input/**/stage2_results", recursive=True))
    s3 = sorted(glob.glob("/kaggle/input/**/stage3_results", recursive=True))
    runs = load_runs(s2 + [OUT])
    res = {"n_test": int(len(y_test)), "n_boot": N_BOOT, "configs": {}, "inputs": {"stage2": s2, "stage3": s3}}

    probs = {}
    by_name = {}
    for rid, (m, d) in runs.items():
        by_name.setdefault(m["name"], []).append((m, d))
    for name, lst in sorted(by_name.items()):
        lst.sort(key=lambda t: t[0]["seed"])
        keys = ["accuracy", "auc", "f1", "sensitivity", "specificity", "nll", "brier", "ece"]
        res["configs"][name] = {
            "feat": lst[0][0]["feat"], "graph": lst[0][0]["graph"], "pool": lst[0][0]["pool"],
            "experiment": lst[0][0]["experiment"], "seeds": [t[0]["seed"] for t in lst],
            "hparams": lst[0][0].get("hparams"),
            "test": {k: t_ci([t[0]["test"][k] for t in lst]) for k in keys},
            "val": {k: t_ci([t[0]["val"][k] for t in lst]) for k in ("accuracy", "auc", "nll")},
            "best_epoch": t_ci([t[0].get("best_epoch") or 0 for t in lst]),
            "epochs_run": t_ci([t[0].get("epochs_run") or 0 for t in lst]),
        }
        P = np.stack([np.load(os.path.join(t[1], "test_prob.npy")) for t in lst])
        if P.shape[1] == len(y_test):
            probs[name] = P
    core = {k: v for k, v in probs.items() if k in CORE or k in BASELINES}
    log(f"bootstrap over {len(core)} main configurations, {N_BOOT} replicates")
    res["comparisons_main"] = hierarchical_bootstrap(core, y_test, ["M8", "B_noedges"], N_BOOT, BOOT_SEED)
    sw = {k: v for k, v in probs.items() if k in SWEEPS or k in ("M1", "M2", "M3", "M8")}
    if any(k in SWEEPS for k in sw):
        log(f"bootstrap over {len(sw)} sweep configurations")
        res["comparisons_sweeps"] = hierarchical_bootstrap(sw, y_test, ["M8"], N_BOOT, BOOT_SEED + 1)

    bench_rows = []
    for folder in s2 + [OUT]:
        for path in glob.glob(os.path.join(folder, "benchmark_*.json")):
            with open(path) as f:
                bench_rows += json.load(f)
    res["resources"] = bench_summary(bench_rows)

    fid = {}
    for folder in s3:
        for path in glob.glob(os.path.join(folder, "*", "fidelity_probs.npz")):
            with open(os.path.join(os.path.dirname(path), "fidelity.json")) as f:
                meta = json.load(f)
            if meta.get("quick_test"):
                continue
            z = np.load(path)
            fid.setdefault(meta["name"], []).append((meta["seed"], z["P"], meta))
    res["fidelity"] = {}
    for name, lst in sorted(fid.items()):
        lst.sort(key=lambda t: t[0])
        sparsity = sorted({x["sparsity"] for x in lst[0][2]["fidelity"]})
        res["fidelity"][name] = {
            "seeds": [t[0] for t in lst], "graphs": int(lst[0][1].shape[0]),
            "accuracy_on_explained_graphs": t_ci([t[2]["accuracy_on_explained_graphs"] for t in lst]),
            "checkpoint_max_abs_prob_diff": max(t[2]["checkpoint_check"]["max_abs_prob_diff"] for t in lst),
            "explain_ms_per_graph": t_ci([t[2]["explain_ms_per_graph"] for t in lst]),
            "by_sparsity": fidelity_stats([t[1] for t in lst], sparsity, N_BOOT, BOOT_SEED + 2),
        }
    for path in glob.glob("/kaggle/input/**/stage2_results/prep_info.json", recursive=True):
        with open(path) as f:
            res["stage2_prep"] = json.load(f)
    p4 = os.path.join(OUT, "prep_info.json")
    if os.path.exists(p4):
        with open(p4) as f:
            res["stage4_prep"] = json.load(f)
    return res


# %%
def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    ensure_pyg()
    deadline = T_START + TIME_BUDGET_H * 3600
    jobs = all_jobs()
    todo = [j for j in jobs if not done(j)]
    log(f"Part A: {len(jobs)} sweep runs defined, {len(todo)} still to do, {n_gpus()} GPU(s)")
    folder = find_stage1()
    if todo:
        splits = {k: load_split(folder, k, SUBSET[k]) for k in ("train", "valid", "test")}
        g_max, f_min = splits["train"].g_max, splits["train"].f_min
        dims = sorted({pca_dim(j.feat) for j in todo if pca_dim(j.feat)})
        spaces = sorted({j.feat for j in todo if knn_k(j.graph)})
        run_child(prep_child, splits, dims, spaces)

        def uniq(js):
            seen, out = set(), []
            for j in js:
                if j.name not in seen:
                    seen.add(j.name); out.append(j)
            return out

        def load_knn(space):
            if space not in spaces:
                return {k: None for k in splits}
            return {k: np.load(os.path.join(TMP, f"knn_{space}_{k}.npy")) for k in splits}

        raw_jobs = [j for j in todo if j.feat == "raw"]
        if raw_jobs:
            knn = load_knn("raw")
            data = {k: view(s, s.q, knn[k]) for k, s in splits.items()}
            if not os.path.exists(os.path.join(OUT, "benchmark_raw.json")):
                run_child(bench_child, uniq(raw_jobs), data, "raw", g_max, f_min, "raw")
            run_jobs(raw_jobs, data, "raw", g_max, f_min, deadline)
            del data, knn
        for s in splits.values():
            s.q = None
        gc.collect()
        for d in dims:
            d_jobs = [j for j in todo if j.feat == f"pca{d}"]
            feats = {k: np.load(os.path.join(TMP, f"pca{d}_{k}.npy")) for k in splits}
            knn = load_knn(f"pca{d}")
            data = {k: view(s, feats[k], knn[k]) for k, s in splits.items()}
            if not os.path.exists(os.path.join(OUT, f"benchmark_pca{d}.json")):
                run_child(bench_child, uniq(d_jobs), data, "pca", g_max, f_min, f"pca{d}")
            run_jobs(d_jobs, data, "pca", g_max, f_min, deadline)
            del data, feats, knn
            gc.collect()
    left = [j.run_id for j in jobs if not done(j)]
    log(f"Part A finished; {len(left)} sweep runs not done" + (f": {left}" if left else ""))

    import h5py
    with h5py.File(os.path.join(folder, "pcam_test.h5"), "r") as f:
        y_test = f["y"][:].astype(np.int64)
    log("Part B: statistics")
    res = analysis(y_test)
    res["quick_test"] = QUICK_TEST
    res["sweep_runs_not_done"] = left
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("===RESULTS_JSON_BEGIN===")
    print(json.dumps(res, separators=(",", ":")))
    print("===RESULTS_JSON_END===")
    log("finished")


if __name__ == "__main__":
    main()
