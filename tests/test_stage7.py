import importlib.util, json, os, sys, types
import fakeh5, numpy as np, h5py
spec = importlib.util.spec_from_file_location("s7", sys.argv[1]); s7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s7)
rng = np.random.default_rng(0)
G = 2000
n = rng.integers(8, 40, G); ptr = np.concatenate([[0], np.cumsum(n)])
cents, edges, ec = [], [], []
from scipy.spatial import Delaunay
for g in range(G):
    c = rng.integers(0, 96, (n[g], 2)); cents.append(c)
    tri = Delaunay(c.astype(float)); s = tri.simplices
    e = np.unique(np.sort(np.concatenate([s[:, [0,1]], s[:, [1,2]], s[:, [0,2]]]), 1), axis=0); edges.append(e); ec.append(len(e))
os.makedirs("in/s1", exist_ok=True)
with h5py.File("in/s1/pcam_test.h5", "w") as f:
    f["ptr"] = ptr; f["edge_ptr"] = np.concatenate([[0], np.cumsum(ec)]); f["delaunay_edges"] = np.concatenate(edges).astype(np.int16)
    f["centroids"] = np.concatenate(cents).astype(np.uint8); f["y"] = np.arange(G) % 2
with h5py.File("in/test_x.h5", "w") as f:
    f["x"] = rng.integers(0, 255, (G, 96, 96, 3)).astype(np.uint8)
d = types.SimpleNamespace(edges=None, ptr=ptr, edge_ptr=np.concatenate([[0], np.cumsum(ec)]), y=np.arange(G) % 2, G=G, q=None, name="test", N=int(ptr[-1]), g_max=1.0, f_min=0.0)
s7.OUT, s7.TMP = "out", "tmp"
s7.ensure_pyg = lambda: None
s7.find_stage2 = lambda: {"M8_s0": "x", "M3_s0": "x"}
s7.find_stage1 = lambda: "in/s1"
s7.load_split = lambda folder, k, nn: d
s7.lazy_train = lambda folder: types.SimpleNamespace(g_max=1.0, f_min=0.0)
s7.fetch_test_images = lambda: "in/test_x.h5"
def fake_run_child(fn, *a):
    if fn.__name__ == "prep_child": return
    job, views, na, fit_ids, test_ids, ck, path = a
    rows = {}
    r = np.random.default_rng(1)
    for g in test_ids:
        ne = int(d.edge_ptr[g+1] - d.edge_ptr[g])
        pf = float(r.uniform(0.3, 1))
        rows[int(g)] = {"scores": r.random(ne).tolist(), "random": r.random(ne).tolist(), "p_full": pf,
                        "p_drop_pgx": pf - float(r.uniform(0, .2)), "p_drop_rand": pf - float(r.uniform(0, .02)),
                        "p_keep_pgx": pf, "p_keep_rand": pf, "p_drop_pgx_tb": pf, "p_keep_pgx_tb": pf, "n_unique": 1, "s_min": 0.0, "s_max": 0.0, "frac_at_max": 1.0, "top_is_index_order": True, "p_drop_sal": pf - 0.05, "p_keep_sal": pf, "sal_scores": r.random(ne).tolist(), "sal_n_unique": ne, "kept_pgx_tb": list(range(ne//2)), "kept_sal": list(range(0, ne, 2)), "kept_rand": list(range(1, ne, 2))}
    json.dump({"run_id": job.run_id, "checkpoint_max_abs_prob_diff": 0.0, "rows": rows}, open(path, "w"))
s7.run_child = fake_run_child
s7.main()
o = json.load(open("out/stage7_figure_data.json"))
for k, ex in o["examples"].items():
    g = ex["test_index"]; assert ex["label"] == (1 if k == "tumour" else 0)
    assert len(ex["models"]["M8"]["scores"]) == len(ex["edges"]) == int(d.edge_ptr[g+1]-d.edge_ptr[g])
    assert len(ex["centroids"]) == n[g]
print("OK", {k: v["test_index"] for k, v in o["examples"].items()}, os.path.getsize("out/stage7_figure_data.json"), "bytes")
