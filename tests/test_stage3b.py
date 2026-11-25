import importlib.util, json, os, sys
import numpy as np
spec = importlib.util.spec_from_file_location("s3b", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.torch is None
cols = m.variant_layout()
assert len(cols) == 1 + 3 * 2 * len(m.SPARSITY) and cols[1] == "pgexplainer_keep0.1" and cols[-1] == "random_drop0.5"
# ranked_top: ties broken by the tie vector, not by index
s = np.zeros(10); tie = np.arange(10)[::-1] / 10.0
assert list(m.ranked_top(s, 0.3, tie)) == [7, 8, 9]
s2 = np.array([0.1, 0.9, 0.5, 0.9]); assert list(m.ranked_top(s2, 0.5, np.array([0.0, 0.2, 0.0, 0.1]))) == [1, 3]
assert m.degeneracy(np.zeros(5))["n_unique"] == 1
# fidelity_rows: random columns are read where variant_layout puts them
G = 50; P = np.full((G, len(cols)), 0.9); P[:, cols.index("random_drop0.5")] = 0.4
rows = m.fidelity_rows(P)
r = {(x["method"], x["sparsity"]): x for x in rows}
assert abs(r[("random", 0.5)]["fid_plus_prob"] - 0.5) < 1e-9 and abs(r[("pgexplainer", 0.5)]["fid_plus_prob"]) < 1e-9
# analyse on synthetic runs: saliency better than random at 50% for M3, nothing for M8
m.OUT = "out"; m.CONFIGS = ["M8", "M3"]
rng = np.random.default_rng(0)
ids = np.arange(1000)
for name in m.CONFIGS:
    for seed in m.EXPLAIN_SEEDS:
        P = 0.9 + 0.01 * rng.standard_normal((1000, len(cols)))
        if name == "M3":
            P[:, cols.index("saliency_drop0.5")] -= 0.1
        os.makedirs(f"out/{name}_s{seed}", exist_ok=True)
        np.savez(f"out/{name}_s{seed}/fidelity_probs.npz", P=P, test_ids=ids)
        json.dump({"degeneracy": {"pgexplainer": {"graphs_all_tied": 556}}}, open(f"out/{name}_s{seed}/fidelity.json", "w"))
m.analyse()
o = json.load(open("out/stage3b_summary.json"))
d3 = o["configs"]["M3"]["delta"]["saliency_0.5_dfid_plus"]; d8 = o["configs"]["M8"]["delta"]["saliency_0.5_dfid_plus"]
assert 0.09 < d3[0] < 0.11 and d3[1] < d3[0] < d3[2] and d8[1] < 0 < d8[2], (d3, d8)
print("OK", d3, d8, os.path.getsize("out/stage3b_summary.json"))
