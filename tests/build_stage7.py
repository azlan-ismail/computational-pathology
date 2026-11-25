"""Build scripts/stage7_figure_pcam.py from the Stage 3 script.

Stage 7 reuses the Stage 3 code verbatim (data loading, standardisation, model, PGExplainer fit,
edge scoring and random reference) and replaces only the header and main(). It explains the same
1,000 test graphs with seed 0 of M8 and M3, chooses one representative tumour and one normal patch,
downloads the official PCam test images to draw them, and writes everything the figure needs to
stage7_figure_data.json (plus a preview PNG).

Usage: python -I tests/build_stage7.py   (run from the repository root)
"""
import re

src = open("scripts/stage3_fidelity_pcam.py").read()

header = '''# %% [markdown]
# # Stage 7 - qualitative example of edge explanations (figure data)
#
# **Inputs (right-hand panel -> Add Input -> Notebook output):** the finished Stage 1 notebook
# (graphs) and the finished Stage 2 notebook (`stage2_results`). Accelerator = GPU T4 x2,
# Internet = On (the official PCam test images are downloaded from Zenodo, about 0.8 GB).
#
# For seed 0 of M8 (no reduction) and M3 (Top-K pooling) it repeats the Stage 3 procedure exactly:
# same standardisation, same 500 validation graphs to fit PGExplainer, same 1,000 test graphs, same
# random reference. It then picks, among the explained test graphs that both models classify
# correctly, the tumour patch and the normal patch whose M3 saliency-minus-random drop at 50 % of
# edges is closest to the median (a representative example, not the best one), and saves the patch
# image, entity centroids, Delaunay edges, edge scores (PGExplainer with random tie-breaking and gradient
# saliency), tie diagnostics and the predicted probabilities to
# `stage7_results/stage7_figure_data.json`. Run time: about 15 minutes.

# %%
QUICK_TEST = False
'''

body_start = src.index("CONFIGS = [")
main_start = src.index("# %%\ndef main():")
body = src[body_start:main_start]
# Stage 7 needs only the two models, seed 0, and has no quick-test variant
body = body.replace('CONFIGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]', 'CONFIGS = ["M8", "M3"]')
body = body.replace("EXPLAIN_SEEDS = [0, 1, 2]", "EXPLAIN_SEEDS = [0]")
body = body.replace('OUT = "/kaggle/working/stage3_results"', 'OUT = "/kaggle/working/stage7_results"')
body = body.replace('TMP = "/kaggle/temp/stage3"', 'TMP = "/kaggle/temp/stage7"')
assert 'CONFIGS = ["M8", "M3"]' in body and "EXPLAIN_SEEDS = [0]" in body and "stage7_results" in body

new_main = r'''# %%
# ----------------------------------------------------------------------------------------------
# Stage 7: figure data
# ----------------------------------------------------------------------------------------------
KEEP = 0.5                  # fraction of edges kept as the explanation in the figure (as in Table 4)
RAW_DIR = "/kaggle/temp/pcam_raw"
ZENODO = ["https://zenodo.org/records/2546921/files/{name}?download=1",
          "https://zenodo.org/record/2546921/files/{name}"]
TEST_X = "camelyonpatch_level_2_split_test_x.h5.gz"
TEST_X_MD5 = "d8c2d60d490dbd479f8199bdfa0cf6ec"     # torchvision/datasets/pcam.py, as in Stage 1


class _HashingReader:
    def __init__(self, raw):
        import hashlib
        self.raw, self.md5 = raw, hashlib.md5()

    def read(self, size=-1):
        b = self.raw.read(size)
        self.md5.update(b)
        return b


def fetch_test_images():
    """Official PCam test images, MD5-checked, or an attached copy under /kaggle/input."""
    import gzip
    import urllib.request
    name = TEST_X[:-3]
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, name)
    if os.path.exists(dest):
        return dest
    last = None
    for attempt in range(3):
        for tmpl in ZENODO:
            try:
                log(f"downloading {TEST_X} (attempt {attempt + 1})")
                req = urllib.request.Request(tmpl.format(name=TEST_X), headers={"User-Agent": "pcam-stage7"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    hr = _HashingReader(resp)
                    with gzip.GzipFile(fileobj=hr) as gz, open(dest + ".part", "wb") as out:
                        shutil.copyfileobj(gz, out, length=16 * 1024 * 1024)
                    while hr.read(1024 * 1024):
                        pass
                if hr.md5.hexdigest() != TEST_X_MD5:
                    raise IOError(f"MD5 mismatch: {hr.md5.hexdigest()}")
                os.replace(dest + ".part", dest)
                return dest
            except Exception as exc:
                last = exc
                log(f"  failed: {exc}")
                time.sleep(10)
    raise RuntimeError(f"Could not download {TEST_X}: {last}")


def saliency_scores(model, x, ei, bvec, target):
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


def figure_child(job, data, norm_args, fit_ids, test_ids, ckpt_dir, out_path):
    """Stage 3 explain_run, keeping the per-graph edge scores and probabilities at KEEP."""
    from torch_geometric.explain import Explainer, PGExplainer
    dev = device(0)
    norm = make_norm(*norm_args, dev)
    kind = "raw"
    d_va, d_te = data["valid"], data["test"]
    model = net_class()(d_te.feats.shape[1], pool_ratio(job.pool)).to(dev)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, tprob = evaluate(model, d_te, job, kind, norm, dev)
    saved = np.load(os.path.join(ckpt_dir, "test_prob.npy"))
    diff = float(np.abs(tprob - saved).max())

    torch.manual_seed(job.seed)
    explainer = Explainer(
        model=model, algorithm=PGExplainer(epochs=PGX_EPOCHS, lr=PGX_LR).to(dev),
        explanation_type="phenomenon", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))
    fit = [to_dev(make_batch(d_va, np.array([g]), job.graph), kind, norm, dev) for g in fit_ids]
    fit = [b for b in fit if b[1].shape[1] > 0]
    for ep in range(PGX_EPOCHS):
        for x, ei, bvec, y in fit:
            explainer.algorithm.train(ep, model, x, ei, target=y, b=bvec, B=1)

    rng = np.random.default_rng(10_000 + job.seed)        # same random stream as Stage 3
    tie_rng = np.random.default_rng(20_000 + job.seed)    # separate stream for tie-breaking
    rows = {}
    for g in test_ids:
        xb, eib, bb, yb = make_batch(d_te, np.array([g]), job.graph)
        x, ei, bvec, y = to_dev((xb, eib, bb, yb), kind, norm, dev)
        scores = explainer(x, ei, target=y, b=bvec, B=1).edge_mask.detach().float().cpu().numpy()
        us = undirected_scores(scores, True)
        rand = rng.random(len(us))
        tb = tie_rng.random(len(us))
        k_n = min(len(us), max(1, int(round(KEEP * len(us)))))
        order_tb = np.lexsort((tb, -us))[:k_n]                 # ties broken at random
        keep_tb = units_to_edges(np.sort(order_tb), len(us), True)
        top_idx = top_units(us, KEEP)
        keeps = []
        for method_scores in (us, rand):
            for s in SPARSITY:
                k = units_to_edges(top_units(method_scores, s), len(us), True)
                keeps += [k, ~k]
        sal = undirected_scores(saliency_scores(model, x, ei, bvec, int(yb[0])).astype(np.float64), True)
        tb2 = tie_rng.random(len(us))
        keep_sal = units_to_edges(np.sort(np.lexsort((tb2, -sal))[:k_n]), len(us), True)
        keeps += [keep_tb, ~keep_tb, keep_sal, ~keep_sal]
        X, EI, BV = variant_batch(xb, eib, keeps)
        xt, eit, bt, _ = to_dev((X, EI, BV, yb), kind, norm, dev)
        with torch.no_grad():
            p = torch.softmax(model(xt, eit, bt, len(keeps) + 1), 1)[:, int(yb[0])].float().cpu().numpy()
        cols = variant_layout() + ["pgx_tiebreak_keep", "pgx_tiebreak_drop", "sal_keep", "sal_drop"]
        rows[int(g)] = {"scores": us.tolist(), "random": rand.tolist(),
                        "p_full": float(p[0]),
                        "p_drop_pgx": float(p[cols.index(f"pgexplainer_drop{KEEP}")]),
                        "p_drop_rand": float(p[cols.index(f"random_drop{KEEP}")]),
                        "p_keep_pgx": float(p[cols.index(f"pgexplainer_keep{KEEP}")]),
                        "p_keep_rand": float(p[cols.index(f"random_keep{KEEP}")]),
                        "p_drop_pgx_tb": float(p[cols.index("pgx_tiebreak_drop")]),
                        "p_keep_pgx_tb": float(p[cols.index("pgx_tiebreak_keep")]),
                        "p_drop_sal": float(p[cols.index("sal_drop")]), "p_keep_sal": float(p[cols.index("sal_keep")]),
                        "sal_scores": sal.tolist(), "sal_n_unique": int(len(np.unique(sal))),
                        "kept_pgx_tb": np.flatnonzero(keep_tb[:len(us)]).tolist(),
                        "kept_sal": np.flatnonzero(keep_sal[:len(us)]).tolist(),
                        "kept_rand": top_units(rand, KEEP).tolist(),
                        "n_unique": int(len(np.unique(us))), "s_min": float(us.min()), "s_max": float(us.max()),
                        "frac_at_max": float(np.mean(us == us.max())),
                        "top_is_index_order": bool(np.array_equal(top_idx, np.arange(len(top_idx))))}
    with open(out_path, "w") as f:
        json.dump({"run_id": job.run_id, "checkpoint_max_abs_prob_diff": diff, "rows": rows}, f)


def main():
    import base64
    import io
    import h5py
    from PIL import Image
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    ensure_pyg()
    stage2 = find_stage2()
    jobs = [SimpleNamespace(name=n, feat=CORE[n][0], graph=CORE[n][1], pool=CORE[n][2], seed=0,
                            run_id=f"{n}_s0", exp="core") for n in CONFIGS]
    missing = [j.run_id for j in jobs if j.run_id not in stage2]
    if missing:
        raise RuntimeError(f"Stage 2 checkpoints not found for {missing}. Attach the finished Stage 2 output.")
    folder = find_stage1()
    ev = {k: load_split(folder, k, None) for k in ("valid", "test")}
    tr = lazy_train(folder)
    run_child(prep_child, {"train": tr, **ev}, [], [], ["valid", "test"])
    fit_ids = pick_graphs(ev["valid"], N_FIT, SAMPLE_SEED)
    test_ids = pick_graphs(ev["test"], N_EXPLAIN, SAMPLE_SEED + 1)
    views = {k: view(s, s.q, None) for k, s in ev.items()}
    res = {}
    for job in jobs:
        path = os.path.join(TMP, f"{job.run_id}.json")
        run_child(figure_child, job, views, (tr.g_max, tr.f_min), fit_ids, test_ids, stage2[job.run_id], path)
        with open(path) as f:
            res[job.name] = json.load(f)
        log(f"{job.run_id}: explained {len(res[job.name]['rows'])} graphs "
            f"(checkpoint diff {res[job.name]['checkpoint_max_abs_prob_diff']:.1e})")

    # mean drops over the 1,000 graphs (seed 0), as a check against the Stage 3 fidelity table
    summary = {}
    for name in CONFIGS:
        r = res[name]["rows"].values()
        summary[name] = {"fid_plus_pgx": float(np.mean([x["p_full"] - x["p_drop_pgx"] for x in r])),
                         "fid_plus_rand": float(np.mean([x["p_full"] - x["p_drop_rand"] for x in r]))}
    for name in CONFIGS:
        r = list(res[name]["rows"].values())
        summary[name].update({
            "fid_plus_pgx_tiebreak": float(np.mean([x["p_full"] - x["p_drop_pgx_tb"] for x in r])),
            "fid_minus_pgx": float(np.mean([x["p_full"] - x["p_keep_pgx"] for x in r])),
            "fid_minus_rand": float(np.mean([x["p_full"] - x["p_keep_rand"] for x in r])),
            "fid_minus_pgx_tiebreak": float(np.mean([x["p_full"] - x["p_keep_pgx_tb"] for x in r])),
            "fid_plus_sal": float(np.mean([x["p_full"] - x["p_drop_sal"] for x in r])),
            "fid_minus_sal": float(np.mean([x["p_full"] - x["p_keep_sal"] for x in r])),
            "graphs_sal_all_tied": int(sum(x["sal_n_unique"] == 1 for x in r)),
            "graphs_all_scores_tied": int(sum(x["n_unique"] == 1 for x in r)),
            "graphs_top_is_index_order": int(sum(x["top_is_index_order"] for x in r)),
            "median_unique_scores": float(np.median([x["n_unique"] for x in r])),
            "median_frac_at_max": float(np.median([x["frac_at_max"] for x in r])),
            "score_min": float(min(x["s_min"] for x in r)), "score_max": float(max(x["s_max"] for x in r)),
            "graphs": len(r)})
    log(f"seed-0 Fid+ at {KEEP:.0%} kept: {summary}")
    with open(os.path.join(OUT, "stage7_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # representative examples: correct for both models, M3 (PGX - random) drop closest to the median
    d_te = ev["test"]
    chosen = {}
    for c in (1, 0):
        cand = [int(g) for g in test_ids if d_te.y[g] == c
                and all(res[n]["rows"][str(int(g))]["p_full"] > 0.5 for n in CONFIGS)]
        gap = np.array([(lambda x: (x["p_full"] - x["p_drop_sal"]) - (x["p_full"] - x["p_drop_rand"]))(
            res["M3"]["rows"][str(g)]) for g in cand])
        chosen["tumour" if c == 1 else "normal"] = cand[int(np.argmin(np.abs(gap - np.median(gap))))]
    log(f"chosen test graphs: {chosen}")

    img_path = fetch_test_images()
    out = {"keep": KEEP, "seed": 0, "summary_seed0": summary, "examples": {}}
    with h5py.File(img_path, "r") as fx, h5py.File(os.path.join(folder, "pcam_test.h5"), "r") as fg:
        for label, g in chosen.items():
            img = fx["x"][g]
            a, b = int(fg["ptr"][g]), int(fg["ptr"][g + 1])
            ea, eb = int(fg["edge_ptr"][g]), int(fg["edge_ptr"][g + 1])
            buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
            out["examples"][label] = {
                "test_index": g, "label": int(d_te.y[g]),
                "image_png_b64": base64.b64encode(buf.getvalue()).decode(),
                "centroids": fg["centroids"][a:b].tolist(),
                "edges": fg["delaunay_edges"][ea:eb].tolist(),
                "models": {n: res[n]["rows"][str(g)] for n in CONFIGS}}
    with open(os.path.join(OUT, "stage7_figure_data.json"), "w") as f:
        json.dump(out, f)

    # quick preview so the result can be checked in the notebook
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(chosen), 1 + 2 * len(CONFIGS), figsize=(3 * (1 + 2 * len(CONFIGS)), 3 * len(chosen)))
    for r, (label, ex) in enumerate(out["examples"].items()):
        img = np.array(Image.open(io.BytesIO(base64.b64decode(ex["image_png_b64"]))))
        c = np.array(ex["centroids"], float); e = np.array(ex["edges"], int)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{label} (test {ex['test_index']})")
        col = 1
        for n in CONFIGS:
            m = ex["models"][n]
            for method, key in (("saliency", "kept_sal"), ("random", "kept_rand")):
                ax = axes[r, col]; col += 1
                ax.imshow(img, alpha=0.5)
                keep = np.zeros(len(e), bool); keep[m[key]] = True
                for (i, j), on in zip(e, keep):
                    ax.plot(c[[i, j], 1], c[[i, j], 0], color="crimson" if on else "0.6", lw=1.4 if on else 0.5)
                drop = m["p_full"] - (m["p_drop_sal"] if key == "kept_sal" else m["p_drop_rand"])
                ax.set_title(f"{n} {method}: drop {drop:.3f}", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stage7_preview.png"), dpi=120)
    log("finished: stage7_results/stage7_figure_data.json and stage7_preview.png written")


if __name__ == "__main__":
    main()
'''

out = header + "\n" + body + new_main
open("scripts/stage7_figure_pcam.py", "w").write(out)
compile(out, "stage7", "exec")
print("written scripts/stage7_figure_pcam.py", len(out.splitlines()), "lines")
