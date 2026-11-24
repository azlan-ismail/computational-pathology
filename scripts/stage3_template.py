# %% [markdown]
# # Stage 3 - explanation fidelity (PGExplainer) from the saved Stage 2 checkpoints
#
# **Inputs (right-hand panel -> Add Input -> Notebook output):** the finished Stage 1 notebook
# (graphs) and the finished Stage 2 notebook (`stage2_results`, one folder per trained run).
# Accelerator = GPU T4 x2, Internet = On.
#
# For every configuration and seed listed below it:
# 1. rebuilds the standardisation, PCA and nearest-neighbour graphs exactly as Stage 2 did
#    (fitted on the training nodes) and checks that the reloaded checkpoint reproduces the test
#    predictions saved by Stage 2;
# 2. fits PGExplainer on a fixed set of validation graphs (target = true label);
# 3. explains a fixed, class-balanced set of test graphs (the same graphs for every model) and keeps
#    the top 10/20/30/50 % of edges as the explanation;
# 4. measures Fidelity+ (prediction change when the explanation edges are removed) and Fidelity-
#    (prediction change when only the explanation edges are kept), both as probability of the
#    true class and as accuracy, plus the characterization score;
# 5. repeats step 4 with randomly chosen edges of the same size as a reference.
#
# Runs are resumable (a finished `fidelity.json` is skipped). First time: `QUICK_TEST = True`.

# %%
QUICK_TEST = True

CONFIGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
EXPLAIN_SEEDS = [0, 1, 2]
N_FIT = 500                 # validation graphs used to fit PGExplainer
N_EXPLAIN = 1000            # test graphs explained (half per class, same graphs for every model)
MIN_NODES = 8               # graphs smaller than this are not sampled
MIN_EDGES = 10              # nor graphs with fewer undirected Delaunay edges
PGX_EPOCHS = 30
PGX_LR = 0.003
SPARSITY = [0.1, 0.2, 0.3, 0.5]   # fraction of edges kept as the explanation
SAMPLE_SEED = 12345
TIME_BUDGET_H = 10.0

# settings read by the shared Stage 2 code
KNN_MAX = 12
EVAL_BATCH = 512
BATCH = 128
LR = 1e-3
MAX_EPOCHS, PATIENCE, BENCH_REPS = 1, 1, 1
EXPERIMENTS, SEEDS, SWEEP_SEEDS = [], [], []
SUBSET = {"train": None, "valid": None, "test": None}
OUT = "/kaggle/working/stage3_results"
TMP = "/kaggle/temp/stage3"
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
BASELINES, SWEEPS = {}, {}

if QUICK_TEST:
    CONFIGS, EXPLAIN_SEEDS, N_FIT, N_EXPLAIN, PGX_EPOCHS = ["M8", "M7"], [0], 40, 60, 2

# %%
# <<SHARED_STAGE2_CODE>>

# %%
# ----------------------------------------------------------------------------------------------
# Explanation subsets and fidelity (numpy only)
# ----------------------------------------------------------------------------------------------
def undirected_scores(scores, sym):
    """Delaunay batches hold every edge twice: first half (i, j), second half (j, i) in the same
    order. Score each undirected edge by the mean of its two directed scores."""
    if not sym:
        return np.asarray(scores, dtype=np.float64)
    h = len(scores) // 2
    return 0.5 * (np.asarray(scores[:h], np.float64) + np.asarray(scores[h:], np.float64))


def top_units(unit_scores, frac):
    """Indices of the round(frac * n) highest-scoring units (at least one); ties by index."""
    n = len(unit_scores)
    k = min(n, max(1, int(round(frac * n))))
    return np.sort(np.argsort(-np.asarray(unit_scores), kind="stable")[:k])


def units_to_edges(units, n_units, sym):
    keep = np.zeros(n_units, dtype=bool)
    keep[units] = True
    return np.concatenate([keep, keep]) if sym else keep


def variant_batch(x, ei, keeps):
    """Copy 0 of the graph keeps all edges; copy i+1 keeps only the edges where keeps[i] is True."""
    n = x.shape[0]
    parts = [ei] + [ei[:, k] + (i + 1) * n for i, k in enumerate(keeps)]
    return (np.concatenate([x] * (1 + len(keeps))), np.concatenate(parts, axis=1),
            np.repeat(np.arange(1 + len(keeps), dtype=np.int64), n))


def variant_layout():
    """Column order of the probability matrix written for every run."""
    cols = ["full"]
    for method in ("pgexplainer", "random"):
        for s in SPARSITY:
            cols += [f"{method}_keep{s}", f"{method}_drop{s}"]
    return cols


def charact(fid_plus, fid_minus, w_plus=0.5, w_minus=0.5):
    """Characterization score (GraphFramEx); undefined unless fid+ > 0 and fid- < 1."""
    if not (fid_plus > 0 and fid_minus < 1):
        return float("nan")
    return (w_plus + w_minus) / (w_plus / fid_plus + w_minus / (1 - fid_minus))


def fidelity_rows(P):
    """P: (graphs, columns) probability of the true class, columns as in variant_layout()."""
    full = P[:, 0]
    full_ok = (full > 0.5).astype(np.float64)
    rows, col = [], 1
    for method in ("pgexplainer", "random"):
        for s in SPARSITY:
            keep, drop = P[:, col], P[:, col + 1]
            col += 2
            fpa = float(np.mean(full_ok - (drop > 0.5)))
            fma = float(np.mean(full_ok - (keep > 0.5)))
            rows.append({
                "method": method, "sparsity": s, "graphs": int(len(full)),
                "fid_plus_prob": float(np.mean(full - drop)), "fid_minus_prob": float(np.mean(full - keep)),
                "fid_plus_acc": fpa, "fid_minus_acc": fma, "charact_acc": charact(fpa, fma),
                "fid_plus_prob_se": float(np.std(full - drop, ddof=1) / np.sqrt(len(full))) if len(full) > 1 else float("nan"),
                "fid_minus_prob_se": float(np.std(full - keep, ddof=1) / np.sqrt(len(full))) if len(full) > 1 else float("nan"),
            })
    return rows


def pick_graphs(d, n, seed):
    """Class-balanced sample of graphs that are large enough to have a meaningful explanation."""
    nodes = d.ptr[1:] - d.ptr[:-1]
    edges = d.edge_ptr[1:] - d.edge_ptr[:-1]
    ok = (nodes >= MIN_NODES) & (edges >= MIN_EDGES)
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        idx = np.flatnonzero(ok & (d.y == c))
        out.append(rng.choice(idx, size=min(n // 2, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


class H5Rows:
    """Row slices of an HDF5 dataset, opening the file for each read (safe after fork)."""

    def __init__(self, path, name):
        import h5py
        self.path, self.name = path, name
        with h5py.File(path, "r") as f:
            self.shape = f[name].shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        import h5py
        a, b = sl.start or 0, min(sl.stop, self.shape[0])
        with h5py.File(self.path, "r") as f:
            return f[self.name][a:b]


def lazy_train(folder):
    import h5py
    path = os.path.join(folder, "pcam_train.h5")
    with h5py.File(path, "r") as f:
        assert f.attrs.get("complete", 0) == 1 and not bool(f.attrs.get("quick_test", 0))
        n = int(f["ptr"][-1])
        g_max, f_min = float(f["feats_q"].attrs["G_MAX"]), float(f["feats_q"].attrs["F_MIN"])
    log(f"train split: {n:,} nodes (read in chunks for the statistics only)")
    return SimpleNamespace(name="train", q=H5Rows(path, "feats_q"), N=n, g_max=g_max, f_min=f_min)


def find_stage2():
    found = {}
    for path in glob.glob("/kaggle/input/**/stage2_results/*/metrics.json", recursive=True):
        with open(path) as f:
            m = json.load(f)
        if not m.get("quick_test"):
            found[m["run_id"]] = os.path.dirname(path)
    return found


# ----------------------------------------------------------------------------------------------
# GPU side
# ----------------------------------------------------------------------------------------------
def explain_run(job, data, kind, norm, dev, fit_ids, test_ids, ckpt_dir):
    from torch_geometric.explain import Explainer, PGExplainer
    import torch_geometric

    d_va, d_te = data["valid"], data["test"]
    model = net_class()(d_te.feats.shape[1], pool_ratio(job.pool)).to(dev)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 1. the reloaded model must give the predictions Stage 2 saved
    _, tprob = evaluate(model, d_te, job, kind, norm, dev)
    saved = np.load(os.path.join(ckpt_dir, "test_prob.npy"))
    check = {"max_abs_prob_diff": float(np.abs(tprob - saved).max()),
             "reproduced_test_accuracy": float(((tprob >= 0.5) == d_te.y).mean()),
             "saved_test_accuracy": float(((saved >= 0.5) == d_te.y).mean())}
    if check["max_abs_prob_diff"] > 1e-2:
        log(f"WARNING {job.run_id}: reloaded predictions differ from Stage 2 ({check})")

    # 2. fit PGExplainer
    torch.manual_seed(job.seed)
    explainer = Explainer(
        model=model, algorithm=PGExplainer(epochs=PGX_EPOCHS, lr=PGX_LR).to(dev),
        explanation_type="phenomenon", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))
    fit = [to_dev(make_batch(d_va, np.array([g]), job.graph), kind, norm, dev) for g in fit_ids]
    fit = [b for b in fit if b[1].shape[1] > 0]
    t0, losses = time.time(), []
    for ep in range(PGX_EPOCHS):
        tot = 0.0
        for x, ei, bvec, y in fit:
            tot += float(explainer.algorithm.train(ep, model, x, ei, target=y, b=bvec, B=1))
        losses.append(tot / len(fit))
    fit_s = time.time() - t0

    # 3-5. explain test graphs, score explanation and random subsets
    rng = np.random.default_rng(10_000 + job.seed)
    sym = job.graph == "delaunay"
    P, n_units, t0 = [], [], time.time()
    for g in test_ids:
        xb, eib, bb, yb = make_batch(d_te, np.array([g]), job.graph)
        x, ei, bvec, y = to_dev((xb, eib, bb, yb), kind, norm, dev)
        scores = explainer(x, ei, target=y, b=bvec, B=1).edge_mask.detach().float().cpu().numpy()
        us = undirected_scores(scores, sym)
        rand = rng.random(len(us))
        keeps = []
        for method_scores in (us, rand):
            for s in SPARSITY:
                k = units_to_edges(top_units(method_scores, s), len(us), sym)
                keeps += [k, ~k]
        X, EI, BV = variant_batch(xb, eib, keeps)
        xt, eit, bt, _ = to_dev((X, EI, BV, yb), kind, norm, dev)
        with torch.no_grad():
            p = torch.softmax(model(xt, eit, bt, len(keeps) + 1), 1)[:, int(yb[0])]
        P.append(p.float().cpu().numpy())
        n_units.append(len(us))
    explain_s = time.time() - t0
    P = np.stack(P)

    folder = os.path.join(OUT, job.run_id)
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, "fidelity_probs.npz"), P=P, test_ids=test_ids, y=d_te.y[test_ids],
             columns=np.array(variant_layout()), edges_ranked=np.array(n_units))
    res = {"run_id": job.run_id, "name": job.name, "seed": job.seed, "feat": job.feat, "graph": job.graph,
           "pool": job.pool, "quick_test": QUICK_TEST, "checkpoint_check": check,
           "explainer_fit_graphs": len(fit), "explained_graphs": int(len(test_ids)),
           "pgx_epochs": PGX_EPOCHS, "pgx_lr": PGX_LR, "pgx_loss_by_epoch": losses,
           "explainer_fit_seconds": round(fit_s, 1), "explain_ms_per_graph": 1000 * explain_s / len(test_ids),
           "accuracy_on_explained_graphs": float((P[:, 0] > 0.5).mean()),
           "fidelity": fidelity_rows(P), "torch_geometric": torch_geometric.__version__}
    with open(os.path.join(folder, "fidelity.json"), "w") as f:   # written last: marks the run as done
        json.dump(res, f, indent=2)
    r = {(x["method"], x["sparsity"]): x for x in res["fidelity"]}
    log(f"{job.run_id}: checkpoint diff {check['max_abs_prob_diff']:.2e}; at 20% of edges "
        f"Fid+ {r[('pgexplainer', 0.2)]['fid_plus_prob']:.4f} (random {r[('random', 0.2)]['fid_plus_prob']:.4f}), "
        f"Fid- {r[('pgexplainer', 0.2)]['fid_minus_prob']:.4f} (random {r[('random', 0.2)]['fid_minus_prob']:.4f})")


def fid_worker(gpu, jobs, views, stage2, fit_ids, test_ids, g_max, f_min, deadline):
    torch.set_num_threads(2)
    dev = device(gpu)
    norm = make_norm(g_max, f_min, dev)
    for job in jobs:
        if time.time() > deadline:
            log(f"time budget reached; {job.run_id} left for the next session")
            continue
        try:
            kind = "raw" if job.feat == "raw" else "pca"
            explain_run(job, views[job.feat], kind, norm, dev, fit_ids, test_ids, stage2[job.run_id])
        except Exception:
            log(f"{job.run_id} FAILED")
            traceback.print_exc()


def summarise_fidelity():
    import csv
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT, "*", "fidelity.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        return
    keys = ["fid_plus_prob", "fid_minus_prob", "fid_plus_acc", "fid_minus_acc", "charact_acc"]
    with open(os.path.join(OUT, "fidelity_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "name", "seed", "method", "sparsity", "graphs"] + keys + ["checkpoint_max_abs_prob_diff"])
        for r in runs:
            for x in r["fidelity"]:
                w.writerow([r["run_id"], r["name"], r["seed"], x["method"], x["sparsity"], x["graphs"]]
                           + [x[k] for k in keys] + [r["checkpoint_check"]["max_abs_prob_diff"]])
    with open(os.path.join(OUT, "fidelity_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "method", "sparsity", "seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "sd")])
        print(f"\n{'config':7s} {'method':12s} {'keep':>5s} {'seeds':>5s} {'Fid+ prob':>10s} {'Fid- prob':>10s} "
              f"{'Fid+ acc':>9s} {'Fid- acc':>9s} {'charact':>8s}")
        for name in sorted({r["name"] for r in runs}):
            for method in ("pgexplainer", "random"):
                for s in SPARSITY:
                    xs = [x for r in runs if r["name"] == name for x in r["fidelity"]
                          if x["method"] == method and x["sparsity"] == s]
                    row = [name, method, s, len(xs)]
                    means = {}
                    for k in keys:
                        v = np.array([x[k] for x in xs], dtype=float)
                        v = v[np.isfinite(v)]
                        means[k] = float(v.mean()) if len(v) else float("nan")
                        row += [means[k], float(v.std(ddof=1)) if len(v) > 1 else float("nan")]
                    w.writerow(row)
                    print(f"{name:7s} {method:12s} {s:5.2f} {len(xs):5d} {means['fid_plus_prob']:10.4f} "
                          f"{means['fid_minus_prob']:10.4f} {means['fid_plus_acc']:9.4f} "
                          f"{means['fid_minus_acc']:9.4f} {means['charact_acc']:8.4f}")


# %%
def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    ensure_pyg()
    deadline = T_START + TIME_BUDGET_H * 3600
    for path in glob.glob("/kaggle/input/**/stage3_results/*/fidelity.json", recursive=True):
        with open(path) as f:
            if bool(json.load(f).get("quick_test")) != QUICK_TEST:
                continue
        dst = os.path.join(OUT, os.path.basename(os.path.dirname(path)))
        if not os.path.exists(os.path.join(dst, "fidelity.json")):
            shutil.copytree(os.path.dirname(path), dst, dirs_exist_ok=True)

    stage2 = find_stage2()
    jobs = []
    for name in CONFIGS:
        feat, graph, pool = CORE[name]
        for seed in EXPLAIN_SEEDS:
            jobs.append(SimpleNamespace(name=name, feat=feat, graph=graph, pool=pool, seed=seed,
                                        run_id=f"{name}_s{seed}", exp="core"))
    missing = [j.run_id for j in jobs if j.run_id not in stage2]
    if missing:
        raise RuntimeError(f"Stage 2 checkpoints not found for {missing}. Attach the output of the "
                           "finished Stage 2 run (not its quick test).")
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j.run_id, "fidelity.json"))]
    log(f"{len(jobs)} explanation runs defined, {len(todo)} still to do, {n_gpus()} GPU(s)")
    if todo:
        folder = find_stage1()
        ev = {k: load_split(folder, k, None) for k in ("valid", "test")}
        tr = lazy_train(folder)
        dims = sorted({pca_dim(j.feat) for j in todo if pca_dim(j.feat)})
        spaces = sorted({j.feat for j in todo if knn_k(j.graph)})
        run_child(prep_child, {"train": tr, **ev}, dims, spaces, ["valid", "test"])
        fit_ids = pick_graphs(ev["valid"], N_FIT, SAMPLE_SEED)
        test_ids = pick_graphs(ev["test"], N_EXPLAIN, SAMPLE_SEED + 1)
        np.save(os.path.join(OUT, "explained_test_graph_ids.npy"), test_ids)
        log(f"PGExplainer fitted on {len(fit_ids)} validation graphs; {len(test_ids)} test graphs explained")

        def knn_for(space):
            if space not in spaces:
                return {k: None for k in ev}
            return {k: np.load(os.path.join(TMP, f"knn_{space}_{k}.npy")) for k in ev}

        views = {}
        if any(j.feat == "raw" for j in todo):
            kr = knn_for("raw")
            views["raw"] = {k: view(s, s.q, kr[k]) for k, s in ev.items()}
        for d in dims:
            kp = knn_for(f"pca{d}")
            views[f"pca{d}"] = {k: view(s, np.load(os.path.join(TMP, f"pca{d}_{k}.npy")), kp[k])
                                for k, s in ev.items()}
        g_max, f_min = tr.g_max, tr.f_min
        check_parent_clean()
        k = max(1, n_gpus())
        procs = []
        for gpu in range(k):
            p = mp.get_context("fork").Process(
                target=fid_worker, args=(gpu, todo[gpu::k], views, stage2, fit_ids, test_ids, g_max, f_min, deadline))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
    summarise_fidelity()
    left = [j.run_id for j in jobs if not os.path.exists(os.path.join(OUT, j.run_id, "fidelity.json"))]
    log(f"finished; {len(left)} runs not done" + (f": {left}" if left else ""))


if __name__ == "__main__":
    main()
