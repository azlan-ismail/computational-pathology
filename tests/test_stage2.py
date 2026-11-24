"""numpy-only checks for stage 2 batching and metrics. Usage: python -I test_stage2.py <pkg_dir>"""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location("stage2", sys.argv[1] + "/stage2_train_pcam.py")
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)
assert s2.torch is None, "this test is meant for an environment without torch"

rng = np.random.default_rng(1)

# ---- ranges_index against a naive version
for _ in range(200):
    m = rng.integers(1, 12)
    starts = rng.integers(0, 1000, m)
    lens = rng.integers(0, 6, m)
    want = np.concatenate([np.arange(s, s + l) for s, l in zip(starts, lens)] + [np.zeros(0, np.int64)])
    assert np.array_equal(s2.ranges_index(starts, lens), want)

# ---- synthetic split: graphs with 1..40 nodes, including 1- and 2-node graphs
G = 60
n = rng.integers(1, 40, G); n[0], n[1] = 1, 2
ptr = np.concatenate([[0], np.cumsum(n)])
N = int(ptr[-1])
feats = rng.integers(0, 255, (N, 8)).astype(np.uint8)
edges, ecount, knn = [], [], np.full((N, s2.KNN_MAX), -1, np.int16)
for g in range(G):
    e = np.array([(i, j) for i in range(n[g]) for j in range(i + 1, n[g]) if rng.random() < 0.2],
                 dtype=np.int16).reshape(-1, 2)
    edges.append(e); ecount.append(len(e))
    for i in range(n[g]):
        others = [j for j in range(n[g]) if j != i]
        pick = rng.permutation(others)[:s2.KNN_MAX]
        knn[ptr[g] + i, :len(pick)] = pick
d = SimpleNamespace(feats=feats, ptr=ptr, edge_ptr=np.concatenate([[0], np.cumsum(ecount)]),
                    edges=np.concatenate(edges), knn=knn, y=rng.integers(0, 2, G), G=G, N=N)

for graph in ("delaunay", "knn6", "knn3", "knn12", "none"):
    for _ in range(20):
        gids = rng.permutation(G)[:rng.integers(1, 20)]
        x, ei, bvec, y = s2.make_batch(d, gids, graph)
        # naive construction
        xs, bs, es, off = [], [], [], 0
        for b, g in enumerate(gids):
            xs.append(feats[ptr[g]:ptr[g + 1]]); bs += [b] * n[g]
            if graph == "delaunay":
                e = edges[g].astype(np.int64) + off
                es.append(np.concatenate([e, e[:, ::-1]]))
            elif graph.startswith("knn"):
                k = int(graph[3:])
                for i in range(n[g]):
                    for nb in knn[ptr[g] + i, :k]:
                        if nb >= 0:
                            es.append(np.array([[nb + off, i + off]]))
            off += n[g]
        assert np.array_equal(x, np.concatenate(xs)) and np.array_equal(bvec, np.array(bs))
        assert np.array_equal(y, d.y[gids]) and ei.dtype == np.int64 and ei.shape[0] == 2
        want = np.concatenate(es).reshape(-1, 2) if es else np.zeros((0, 2), np.int64)
        a = {tuple(r) for r in ei.T.tolist()}; w = {tuple(r) for r in want.tolist()}
        assert a == w and ei.shape[1] == len(want), graph
        if ei.size:
            assert ei.min() >= 0 and ei.max() < len(x)
            assert (bvec[ei[0]] == bvec[ei[1]]).all(), "edge crosses graphs"
            if graph.startswith("knn"):
                assert (ei[0] != ei[1]).all()

# ---- metrics
y = rng.integers(0, 2, 5000)
p = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.2, 5000), 0.001, 0.999)
m = s2.metrics_from_probs(p, y)
pred = p >= 0.5
assert abs(m["accuracy"] - (pred == y).mean()) < 1e-12
assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 5000 and 0 <= m["ece"] <= 1
assert abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-12
perfect = s2.metrics_from_probs(np.where(y == 1, 0.999, 0.001), y)
assert perfect["accuracy"] == 1.0 and perfect["ece"] < 0.01 and perfect["auc"] in (1.0, None)
half = s2.metrics_from_probs(np.full(5000, 0.9), y)       # confident but 50% right -> ECE ~ 0.4
assert abs(half["ece"] - abs(0.9 - (y == 1).mean())) < 1e-9

# ---- job table
jobs = s2.all_jobs()
assert len({j.run_id for j in jobs}) == len(jobs) and all(hasattr(j, "lr") for j in jobs)
assert s2.knn_k("knn12") == 12 and s2.knn_k("delaunay") == 0 and s2.pool_ratio("topk0.25") == 0.25
assert s2.pca_dim("pca128") == 128 and s2.pca_dim("raw") == 0
print("stage 2 numpy checks passed;", len(jobs), "runs in quick-test job table; auc available:", m["auc"] is not None)
