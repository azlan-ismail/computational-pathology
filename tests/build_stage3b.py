"""Build scripts/stage3b_fidelity_fixed_pcam.py from the Stage 3 script.

Stage 3b repeats the explanation-fidelity analysis with three corrections found by the Stage 7
diagnostics:
1. explanation edges are ranked with ties broken at random (Stage 3 broke ties by edge index, and
   PGExplainer's masks are often all tied, so its 'explanations' were index-ordered edge blocks);
2. the degeneracy of every explanation is recorded (distinct scores, largest score, share tied at
   the maximum), so the protocol can report whether an explainer produced a ranking at all;
3. a second, gradient-based explainer (saliency: |d logit_true / d edge_mask|) is added.
It covers M1-M8 (GraphSAGE, Stage 2 checkpoints) and A8 (GAT, Stage 6 checkpoints), seeds 0-2,
the same 500 fitting and 1,000 test graphs and the same random reference stream as Stage 3, and
finishes with the bootstrap summary that the manuscript needs.

Usage: python -I tests/build_stage3b.py   (from the repository root)
"""
src = open("scripts/stage3_fidelity_pcam.py").read()

header = '''# %% [markdown]
# # Stage 3b - explanation fidelity, corrected (random tie-breaking, degeneracy check, saliency)
#
# **Inputs (right-hand panel -> Add Input -> Notebook output):** the finished Stage 1 notebook
# (graphs), the finished Stage 2 notebook (`stage2_results`) and the finished Stage 6 notebook
# (`stage6_results`, GAT checkpoints). Accelerator = GPU T4 x2, Internet = On.
#
# Corrects Stage 3: PGExplainer masks were frequently all tied (collapsed towards zero), and the
# tie-breaking by edge index turned the 'explanation' into a contiguous block of edges. Here ties
# are broken at random, the degeneracy of every explanation is recorded, and gradient saliency is
# added as a second explainer. Same graphs, seeds and random reference as Stage 3.
# Output: `stage3b_results/stage3b_summary.json` (all numbers for the manuscript).

# %%
QUICK_TEST = False
'''

body_start = src.index("CONFIGS = [")
main_start = src.index("# %%\ndef main():")
body = src[body_start:main_start]

def rep(a, b, count=1):
    global body
    assert body.count(a) >= 1, a[:70]
    body = body.replace(a, b, count)

rep('CONFIGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]',
    'CONFIGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "A8"]')
rep('OUT = "/kaggle/working/stage3_results"', 'OUT = "/kaggle/working/stage3b_results"')
rep('TMP = "/kaggle/temp/stage3"', 'TMP = "/kaggle/temp/stage3b"')
rep('    "M8": ("raw", "delaunay", "none"),\n}', '    "M8": ("raw", "delaunay", "none"),\n    "A8": ("raw", "delaunay", "none"),     # GAT (Stage 6)\n}')
rep('if QUICK_TEST:\n    CONFIGS, EXPLAIN_SEEDS, N_FIT, N_EXPLAIN, PGX_EPOCHS = ["M8", "M7"], [0], 40, 60, 2\n', '')
rep('METHODS_PLACEHOLDER', 'METHODS_PLACEHOLDER') if 'METHODS_PLACEHOLDER' in body else None
body = body.replace('SAMPLE_SEED = 12345\n', 'SAMPLE_SEED = 12345\nMETHODS = ("pgexplainer", "saliency", "random")\nN_BOOT = 2000\n', 1)
assert "METHODS = (" in body

# network: GraphSAGE (M configs) or GAT (A8)
old_net = body[body.index("def net_class():"):body.index("def check_parent_clean():")]
new_net = '''def net_class(arch="sage"):
    """Stage 2 network (two SAGEConv) or Stage 6 network (two GATConv, 4 heads x 16)."""
    key = "cls_" + arch
    if key not in _NET:
        from torch_geometric.nn import GATConv, SAGEConv, TopKPooling, global_mean_pool

        class HybridNet(nn.Module):
            def __init__(self, in_dim, ratio, dropout=0.0):
                super().__init__()
                self.drop = nn.Dropout(dropout)
                if arch == "gat":
                    self.c1, self.c2 = GATConv(in_dim, 16, heads=4), GATConv(64, 16, heads=4)
                else:
                    self.c1, self.c2 = SAGEConv(in_dim, 64), SAGEConv(64, 64)
                self.pool = TopKPooling(64, ratio=ratio) if ratio > 0 else None
                self.l1 = nn.Linear(64, 32)
                self.l2 = nn.Linear(32, 2)

            def forward(self, x, ei, b, B):
                x = F.relu(self.c1(x, ei))
                x = F.relu(self.c2(x, ei))
                if self.pool is not None:
                    x, ei, _, b, _, _ = self.pool(x, ei, None, b)
                h = self.drop(global_mean_pool(x, b, B))
                return self.l2(self.drop(F.relu(self.l1(h))))

        _NET[key] = HybridNet
    return _NET[key]


'''
body = body.replace(old_net, new_net, 1)

# layout and fidelity rows for three methods
old = body[body.index("def variant_layout():"):body.index("def charact(")]
new = '''def variant_layout():
    """Column order of the probability matrix written for every run."""
    cols = ["full"]
    for method in METHODS:
        for s in SPARSITY:
            cols += [f"{method}_keep{s}", f"{method}_drop{s}"]
    return cols


def ranked_top(scores, frac, tie):
    """Indices of the round(frac * n) highest scores; ties broken by the random numbers in `tie`."""
    n = len(scores)
    k = min(n, max(1, int(round(frac * n))))
    return np.sort(np.lexsort((tie, -np.asarray(scores, dtype=np.float64)))[:k])


def degeneracy(scores):
    s = np.asarray(scores, dtype=np.float64)
    return {"n_unique": int(len(np.unique(s))), "max": float(s.max()) if len(s) else 0.0,
            "frac_at_max": float(np.mean(s == s.max())) if len(s) else 1.0}


'''
body = body.replace(old, new, 1)
body = body.replace('''    rows, col = [], 1
    for method in ("pgexplainer", "random"):''', '''    rows, col = [], 1
    for method in METHODS:''', 1)
body = body.replace('''            for method in ("pgexplainer", "random"):
                for s in SPARSITY:
                    xs = [x for r in runs''', '''            for method in METHODS:
                for s in SPARSITY:
                    xs = [x for r in runs''', 1)
assert body.count('("pgexplainer", "random")') == 0, body.count('("pgexplainer", "random")')

# checkpoints from Stage 2 and Stage 6
old = body[body.index("def find_stage2():"):body.index("# ----------------------------------------------------------------------------------------------\n# GPU side")]
new = '''def find_stage2():
    """Checkpoint folders of Stage 2 (M configs) and Stage 6 (A8), full runs only."""
    found = {}
    for pat in ("/kaggle/input/**/stage2_results/*/metrics.json", "/kaggle/input/**/stage6_results/*/metrics.json"):
        for path in glob.glob(pat, recursive=True):
            with open(path) as f:
                m = json.load(f)
            if not m.get("quick_test"):
                found[m["run_id"]] = os.path.dirname(path)
    return found


'''
body = body.replace(old, new, 1)

# explanation run
old = body[body.index("def explain_run("):body.index("def fid_worker(")]
new = '''def saliency_scores(model, x, ei, bvec, target):
    """|d logit_true / d m_e| with every edge mask m_e = 1 (one backward pass)."""
    from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
    mask = torch.ones(ei.shape[1], device=x.device, requires_grad=True)
    set_masks(model, mask, ei, apply_sigmoid=False)
    try:
        with torch.enable_grad():
            out = model(x, ei, bvec, 1)
            out[0, int(target)].backward()
        g = mask.grad.detach().abs().float().cpu().numpy()
    finally:
        clear_masks(model)
    return g


def explain_run(job, data, kind, norm, dev, fit_ids, test_ids, ckpt_dir):
    from torch_geometric.explain import Explainer, PGExplainer
    import torch_geometric

    d_va, d_te = data["valid"], data["test"]
    arch = "gat" if job.name.startswith("A") else "sage"
    model = net_class(arch)(d_te.feats.shape[1], pool_ratio(job.pool)).to(dev)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, tprob = evaluate(model, d_te, job, kind, norm, dev)
    saved = np.load(os.path.join(ckpt_dir, "test_prob.npy"))
    check = {"max_abs_prob_diff": float(np.abs(tprob - saved).max()),
             "reproduced_test_accuracy": float(((tprob >= 0.5) == d_te.y).mean()),
             "saved_test_accuracy": float(((saved >= 0.5) == d_te.y).mean())}
    if check["max_abs_prob_diff"] > 1e-2:
        log(f"WARNING {job.run_id}: reloaded predictions differ from the saved ones ({check})")

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

    rng = np.random.default_rng(10_000 + job.seed)       # random reference: same stream as Stage 3
    tie_rng = np.random.default_rng(30_000 + job.seed)   # tie-breaking for the explainers
    sym = job.graph == "delaunay"
    P, n_units, deg, t0 = [], [], {"pgexplainer": [], "saliency": []}, time.time()
    for g in test_ids:
        xb, eib, bb, yb = make_batch(d_te, np.array([g]), job.graph)
        x, ei, bvec, y = to_dev((xb, eib, bb, yb), kind, norm, dev)
        pgx = explainer(x, ei, target=y, b=bvec, B=1).edge_mask.detach().double().cpu().numpy()
        sal = saliency_scores(model, x, ei, bvec, int(yb[0])).astype(np.float64)
        us = {"pgexplainer": undirected_scores(pgx, sym), "saliency": undirected_scores(sal, sym)}
        n = len(us["pgexplainer"])
        rand = rng.random(n)
        tie = tie_rng.random(n)
        for m in ("pgexplainer", "saliency"):
            dg = degeneracy(us[m])
            dg["top_is_index_order"] = bool(np.array_equal(top_units(us[m], 0.5), np.arange(len(top_units(us[m], 0.5)))))
            deg[m].append(dg)
        keeps = []
        for method in METHODS:
            for s in SPARSITY:
                idx = top_units(rand, s) if method == "random" else ranked_top(us[method], s, tie)
                k = units_to_edges(idx, n, sym)
                keeps += [k, ~k]
        X, EI, BV = variant_batch(xb, eib, keeps)
        xt, eit, bt, _ = to_dev((X, EI, BV, yb), kind, norm, dev)
        with torch.no_grad():
            p = torch.softmax(model(xt, eit, bt, len(keeps) + 1), 1)[:, int(yb[0])]
        P.append(p.float().cpu().numpy())
        n_units.append(n)
    explain_s = time.time() - t0
    P = np.stack(P)

    folder = os.path.join(OUT, job.run_id)
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, "fidelity_probs.npz"), P=P, test_ids=test_ids, y=d_te.y[test_ids],
             columns=np.array(variant_layout()), edges_ranked=np.array(n_units))
    deg_sum = {m: {"graphs_all_tied": int(sum(d["n_unique"] == 1 for d in v)),
                   "graphs_top_is_index_order": int(sum(d["top_is_index_order"] for d in v)),
                   "median_unique": float(np.median([d["n_unique"] for d in v])),
                   "median_frac_at_max": float(np.median([d["frac_at_max"] for d in v])),
                   "largest_score": float(max(d["max"] for d in v)),
                   "median_largest_score": float(np.median([d["max"] for d in v]))} for m, v in deg.items()}
    res = {"run_id": job.run_id, "name": job.name, "seed": job.seed, "arch": arch, "feat": job.feat,
           "graph": job.graph, "pool": job.pool, "checkpoint_check": check, "degeneracy": deg_sum,
           "explainer_fit_graphs": len(fit), "explained_graphs": int(len(test_ids)),
           "pgx_epochs": PGX_EPOCHS, "pgx_lr": PGX_LR, "pgx_loss_by_epoch": losses,
           "explainer_fit_seconds": round(fit_s, 1), "explain_ms_per_graph": 1000 * explain_s / len(test_ids),
           "fidelity": fidelity_rows(P), "torch_geometric": torch_geometric.__version__}
    with open(os.path.join(folder, "fidelity.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"{job.run_id}: checkpoint diff {check['max_abs_prob_diff']:.1e}; PGExplainer all-tied graphs "
        f"{deg_sum['pgexplainer']['graphs_all_tied']}, saliency all-tied {deg_sum['saliency']['graphs_all_tied']}")


'''
body = body.replace(old, new, 1)

analysis = r'''# %%
# ----------------------------------------------------------------------------------------------
# Summary for the manuscript: per configuration, seed-averaged per-graph differences to random,
# with 95 % bootstrap intervals over the 1,000 graphs
# ----------------------------------------------------------------------------------------------
def analyse():
    cols = variant_layout()
    out = {"sparsity": SPARSITY, "n_boot": N_BOOT, "configs": {}}
    rng = np.random.default_rng(2024)
    for name in CONFIGS:
        Ps, degs, ids = [], [], None
        for seed in EXPLAIN_SEEDS:
            folder = os.path.join(OUT, f"{name}_s{seed}")
            if not os.path.exists(os.path.join(folder, "fidelity.json")):
                continue
            z = np.load(os.path.join(folder, "fidelity_probs.npz"))
            if ids is None:
                ids = z["test_ids"]
            assert np.array_equal(ids, z["test_ids"])
            Ps.append(z["P"])
            with open(os.path.join(folder, "fidelity.json")) as f:
                degs.append(json.load(f)["degeneracy"])
        if not Ps:
            continue
        P = np.stack(Ps)                       # seeds x graphs x columns
        full = P[:, :, 0]
        c = {"seeds": len(Ps), "degeneracy_by_seed": degs, "fid": {}, "delta": {}}
        G = P.shape[1]
        boot = rng.integers(0, G, size=(N_BOOT, G))
        for s in SPARSITY:
            for method in METHODS:
                fp = (full - P[:, :, cols.index(f"{method}_drop{s}")]).mean(0)
                fm = (full - P[:, :, cols.index(f"{method}_keep{s}")]).mean(0)
                c["fid"][f"{method}_{s}"] = {"fid_plus": float(fp.mean()), "fid_minus": float(fm.mean())}
            fp_r = (full - P[:, :, cols.index(f"random_drop{s}")]).mean(0)
            fm_r = (full - P[:, :, cols.index(f"random_keep{s}")]).mean(0)
            for method in ("pgexplainer", "saliency"):
                fp = (full - P[:, :, cols.index(f"{method}_drop{s}")]).mean(0)
                fm = (full - P[:, :, cols.index(f"{method}_keep{s}")]).mean(0)
                for key, dv in (("dfid_plus", fp - fp_r), ("dfid_minus", fm - fm_r)):
                    bm = dv[boot].mean(1)
                    c["delta"][f"{method}_{s}_{key}"] = [float(dv.mean()), float(np.percentile(bm, 2.5)),
                                                         float(np.percentile(bm, 97.5))]
        out["configs"][name] = c
    with open(os.path.join(OUT, "stage3b_summary.json"), "w") as f:
        json.dump(out, f)
    for name, c in out["configs"].items():
        d = c["delta"]
        log(f"{name}: dFid+ 50% PGX {d['pgexplainer_0.5_dfid_plus'][0]:+.4f} "
            f"[{d['pgexplainer_0.5_dfid_plus'][1]:+.4f}, {d['pgexplainer_0.5_dfid_plus'][2]:+.4f}], "
            f"saliency {d['saliency_0.5_dfid_plus'][0]:+.4f} "
            f"[{d['saliency_0.5_dfid_plus'][1]:+.4f}, {d['saliency_0.5_dfid_plus'][2]:+.4f}]")


'''

main = src[main_start:]
main = main.replace('glob.glob("/kaggle/input/**/stage3_results/*/fidelity.json", recursive=True)',
                    'glob.glob("/kaggle/input/**/stage3b_results/*/fidelity.json", recursive=True)', 1)
main = main.replace('raise RuntimeError(f"Stage 2 checkpoints not found for {missing}. Attach the output of the "\n                           "finished Stage 2 run (not its quick test).")',
                    'raise RuntimeError(f"Checkpoints not found for {missing}. Attach the finished Stage 2 and Stage 6 outputs.")', 1)
main = main.replace("    summarise_fidelity()\n    left =", "    summarise_fidelity()\n    analyse()\n    left =", 1)
assert "analyse()" in main and "stage3b_results" in main and "Stage 6 outputs" in main
# GAT runs keep their checkpoints' dropout irrelevant at eval; quick-test filter is not needed
main = main.replace("            if bool(json.load(f).get(\"quick_test\")) != QUICK_TEST:\n                continue\n", "            pass\n", 1)

out = header + "\n" + body + analysis + main
open("scripts/stage3b_fidelity_fixed_pcam.py", "w").write(out)
compile(out, "stage3b", "exec")
print("written scripts/stage3b_fidelity_fixed_pcam.py", len(out.splitlines()), "lines")
