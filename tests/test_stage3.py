"""numpy-only checks for the Stage 3 helpers. Usage: python -I test_stage3.py <pkg_dir>"""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location("stage3", sys.argv[1] + "/stage3_fidelity_pcam.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
assert s3.torch is None
rng = np.random.default_rng(3)

# Delaunay batch from the shared code: directed edge i and i + h are the same undirected edge
G = 30
n = rng.integers(8, 30, G)
ptr = np.concatenate([[0], np.cumsum(n)])
edges, ec = [], []
for g in range(G):
    e = np.array([(i, j) for i in range(n[g]) for j in range(i + 1, n[g]) if rng.random() < 0.3], np.int16)
    edges.append(e); ec.append(len(e))
d = SimpleNamespace(feats=rng.integers(0, 255, (ptr[-1], 4)).astype(np.uint8), ptr=ptr,
                    edge_ptr=np.concatenate([[0], np.cumsum(ec)]), edges=np.concatenate(edges),
                    knn=None, y=np.arange(G) % 2, G=G, N=int(ptr[-1]))
for g in range(G):
    x, ei, bvec, y = s3.make_batch(d, np.array([g]), "delaunay")
    h = ei.shape[1] // 2
    assert np.array_equal(ei[0, :h], ei[1, h:]) and np.array_equal(ei[1, :h], ei[0, h:])
    sc = rng.random(2 * h)
    us = s3.undirected_scores(sc, True)
    assert np.allclose(us, (sc[:h] + sc[h:]) / 2)
    for s in s3.SPARSITY:
        u = s3.top_units(us, s)
        assert len(u) == max(1, int(round(s * h)))
        assert us[u].min() >= np.delete(us, u).max() if len(u) < h else True
        k = s3.units_to_edges(u, h, True)
        kept = {tuple(sorted(p)) for p in ei[:, k].T.tolist()}
        assert len(kept) == len(u) and k.sum() == 2 * len(u)
        dropped = {tuple(sorted(p)) for p in ei[:, ~k].T.tolist()}
        assert not (kept & dropped) and len(kept) + len(dropped) == h
    keeps = [s3.units_to_edges(s3.top_units(us, 0.2), h, True)]
    keeps.append(~keeps[0])
    X, EI, BV = s3.variant_batch(x, ei, keeps)
    assert X.shape[0] == 3 * len(x) and EI.shape[1] == 2 * h + 2 * h
    assert (BV[EI[0]] == BV[EI[1]]).all()
    assert np.array_equal(EI[:, :2 * h], ei)

# directed (k-NN) graphs are ranked edge by edge
u = s3.top_units(np.array([0.1, 0.9, 0.5, 0.7]), 0.5)
assert list(u) == [1, 3] and list(s3.units_to_edges(u, 4, False)) == [False, True, False, True]

# fidelity arithmetic on a hand-made matrix: columns full, then keep/drop per method and sparsity
cols = s3.variant_layout()
assert len(cols) == 1 + 2 * 2 * len(s3.SPARSITY)
P = np.full((4, len(cols)), 0.9)
P[:, cols.index("pgexplainer_drop0.2")] = [0.2, 0.3, 0.9, 0.9]   # removing explanation hurts 2 of 4
P[:, cols.index("pgexplainer_keep0.2")] = [0.8, 0.8, 0.8, 0.4]   # keeping only explanation: 1 of 4 flips
rows = {(r["method"], r["sparsity"]): r for r in s3.fidelity_rows(P)}
r = rows[("pgexplainer", 0.2)]
assert abs(r["fid_plus_acc"] - 0.5) < 1e-12 and abs(r["fid_minus_acc"] - 0.25) < 1e-12
assert abs(r["fid_plus_prob"] - np.mean(0.9 - np.array([0.2, 0.3, 0.9, 0.9]))) < 1e-12
assert abs(r["charact_acc"] - 1 / (0.5 / 0.5 + 0.5 / 0.75)) < 1e-12
assert rows[("random", 0.1)]["fid_plus_prob"] == 0.0
assert np.isnan(s3.charact(0.0, 0.1))

# sampling: class-balanced, large-enough graphs, reproducible
ids = s3.pick_graphs(d, 10, 7)
assert np.array_equal(ids, s3.pick_graphs(d, 10, 7)) and (d.y[ids] == 0).sum() == (d.y[ids] == 1).sum()
assert ((d.ptr[ids + 1] - d.ptr[ids]) >= s3.MIN_NODES).all()
print("stage 3 numpy checks passed")
