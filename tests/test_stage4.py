import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("s4", sys.argv[1] + "/stage4_sweeps_stats_pcam.py")
s4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s4)
rng = np.random.default_rng(0)
# weighted AUC equals unweighted AUC (sklearn) for unit weights, and for integer weights equals AUC on repeated data
from sklearn.metrics import roc_auc_score
y = rng.integers(0, 2, 3000); p = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1)
o = np.argsort(p, kind="stable")
assert abs(s4.weighted_auc_sorted(y[o], np.ones(3000)) - roc_auc_score(y, p)) < 1e-9
w = rng.integers(0, 4, 3000).astype(float)
rep = np.repeat(np.arange(3000), w.astype(int))
assert abs(s4.weighted_auc_sorted(y[o], w[o]) - roc_auc_score(y[rep], p[rep])) < 1e-9
# Holm
adj = s4.holm([0.01, 0.04, 0.03, 0.5])
assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
# hierarchical bootstrap: identical configs -> diff ~0 with CI covering 0; clearly better config -> positive
N = 4000; y = rng.integers(0, 2, N)
def model(skill, seeds=5):
    return np.stack([np.clip(np.where(y == 1, 0.5 + skill, 0.5 - skill) + rng.normal(0, 0.25, N), 0, 1) for _ in range(seeds)])
probs = {"M8": model(0.1), "A": model(0.1), "B": model(0.2)}
comps = s4.hierarchical_bootstrap(probs, y, ["M8"], 300, 1)
r = {c["config"]: c for c in comps}
assert r["A"]["accuracy"]["lo"] < 0 < r["A"]["accuracy"]["hi"], r["A"]
assert r["B"]["accuracy"]["lo"] > 0 and r["B"]["auc"]["lo"] > 0 and r["B"]["accuracy"]["p_holm"] < 0.05
# t_ci
t = s4.t_ci([1, 2, 3, 4, 5]); assert abs(t["mean"] - 3) < 1e-12 and t["lo"] < 3 < t["hi"]
# fidelity_stats: build P where pgx removal hurts more than random
G = 500; cols = 1 + 2 * 2 * len(s4.SPARSITY_FOR_TEST) if hasattr(s4, "SPARSITY_FOR_TEST") else 17
sp = [0.1, 0.2, 0.3, 0.5]
P = np.full((G, 17), 0.8)
P[:, 2] = 0.6   # pgexplainer drop at 0.1 -> Fid+ 0.2
fs = s4.fidelity_stats([P, P], sp, 200, 0)
assert abs(fs[0]["pgexplainer"]["fid_plus"] - 0.2) < 1e-9 and abs(fs[0]["random"]["fid_plus"]) < 1e-12
assert fs[0]["fid_plus_pgx_minus_random"]["lo"] > 0.19
print("stage 4 statistics checks passed")
