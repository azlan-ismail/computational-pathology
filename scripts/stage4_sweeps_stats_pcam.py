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
# Shared with Stage 2 (data loading, batching, model, preparation); copied verbatim.
import gc
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

# Worker processes are forked from this one and use the GPUs, so this process must never start
# CUDA itself. This setting makes torch.cuda.is_available() safe to call before forking.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:          # allows the numpy-only helpers to be tested without PyTorch
    torch = None

T_START = time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------------------------
def hp_tag(lr, wd, do):
    return f"lr{lr:g}_wd{wd:g}_do{do:g}"


def all_jobs():
    jobs = []
    if "tune" in EXPERIMENTS:
        table = {**CORE, **BASELINES}
        for name in TUNE_CONFIGS:
            feat, graph, pool = table[name]
            for lr in TUNE_LR:
                for wd in TUNE_WD:
                    for do in TUNE_DROPOUT:
                        jobs.append(SimpleNamespace(
                            name=f"T_{name}", base=name, feat=feat, graph=graph, pool=pool, seed=0,
                            lr=lr, wd=wd, dropout=do, exp="tune",
                            run_id=f"T_{name}_{hp_tag(lr, wd, do)}_s0"))
    for exp, table, seeds in (("core", CORE, SEEDS), ("baselines", BASELINES, SEEDS),
                              ("sweeps", SWEEPS, SWEEP_SEEDS)):
        if exp not in EXPERIMENTS:
            continue
        for name, (feat, graph, pool) in table.items():
            for seed in seeds:
                jobs.append(SimpleNamespace(name=name, base=name, feat=feat, graph=graph, pool=pool,
                                            seed=seed, lr=LR, wd=WEIGHT_DECAY, dropout=DROPOUT, exp=exp,
                                            run_id=f"{name}_s{seed}"))
    return jobs


def knn_k(graph):
    return int(graph[3:]) if graph.startswith("knn") else 0


def pool_ratio(pool):
    return float(pool[4:]) if pool.startswith("topk") else 0.0


def pca_dim(feat):
    return int(feat[3:]) if feat.startswith("pca") else 0


def done(job):
    return os.path.exists(os.path.join(OUT, job.run_id, "metrics.json"))


# ----------------------------------------------------------------------------------------------
# Batching (numpy only)
# ----------------------------------------------------------------------------------------------
def ranges_index(starts, lens):
    """Concatenation of arange(s, s + l) for each (s, l); zero-length ranges are skipped."""
    starts = np.asarray(starts, dtype=np.int64)
    lens = np.asarray(lens, dtype=np.int64)
    keep = lens > 0
    starts, lens = starts[keep], lens[keep]
    if len(lens) == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.ones(int(lens.sum()), dtype=np.int64)
    idx[0] = starts[0]
    bounds = np.cumsum(lens)[:-1]
    idx[bounds] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(idx)


def make_batch(d, gids, graph):
    """Assemble a mini-batch of graphs.

    d: split data with feats (N, F), ptr, edge_ptr, edges (E, 2) local undirected pairs,
       knn (N, KNN_MAX) local neighbour indices or None, y.
    Returns x (n, F), edge_index (2, e) int64 with messages flowing row 0 -> row 1,
    batch vector (n,), labels (B,).
    """
    gids = np.asarray(gids, dtype=np.int64)
    n = d.ptr[gids + 1] - d.ptr[gids]
    node_idx = ranges_index(d.ptr[gids], n)
    x = d.feats[node_idx]
    offs = np.cumsum(n) - n
    bvec = np.repeat(np.arange(len(gids), dtype=np.int64), n)
    if graph == "delaunay":
        ec = d.edge_ptr[gids + 1] - d.edge_ptr[gids]
        e = d.edges[ranges_index(d.edge_ptr[gids], ec)].astype(np.int64) + np.repeat(offs, ec)[:, None]
        ei = np.concatenate([e.T, e.T[::-1]], axis=1)          # both directions
    elif graph.startswith("knn"):
        nb = d.knn[node_idx, :knn_k(graph)].astype(np.int64)
        valid = nb >= 0
        src = (nb + np.repeat(offs, n)[:, None])[valid]          # neighbour ...
        dst = np.broadcast_to(np.arange(len(node_idx), dtype=np.int64)[:, None], nb.shape)[valid]
        ei = np.stack([src, dst])                                # ... sends to the node
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
    return x, np.ascontiguousarray(ei), bvec, d.y[gids]


# ----------------------------------------------------------------------------------------------
# Metrics (numpy only)
# ----------------------------------------------------------------------------------------------
def ece_score(p_pos, y, bins=15):
    conf = np.maximum(p_pos, 1 - p_pos)
    pred = (p_pos >= 0.5).astype(np.int64)
    edges = np.linspace(0.5, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def metrics_from_probs(p_pos, y):
    p_pos = np.asarray(p_pos, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pred = (p_pos >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    eps = 1e-12
    out = {
        "n": int(len(y)), "accuracy": (tp + tn) / max(len(y), 1),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "nll": float(-np.mean(y * np.log(p_pos + eps) + (1 - y) * np.log(1 - p_pos + eps))),
        "brier": float(np.mean((p_pos - y) ** 2)), "ece": ece_score(p_pos, y),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p_pos)) if len(np.unique(y)) == 2 else None
    except Exception:
        out["auc"] = None
    return out


# ----------------------------------------------------------------------------------------------
# Loading Stage 1 output
# ----------------------------------------------------------------------------------------------
def find_stage1():
    hits = glob.glob("/kaggle/input/**/pcam_train.h5", recursive=True)
    if not hits:
        raise FileNotFoundError("Stage 1 output not found. Use Add Input -> Notebook output and "
                                "attach the finished Stage 1 notebook.")
    return os.path.dirname(sorted(hits)[0])


def load_split(folder, split, n_graphs):
    import h5py
    s = SimpleNamespace(name=split)
    with h5py.File(os.path.join(folder, f"pcam_{split}.h5"), "r") as f:
        assert f.attrs.get("complete", 0) == 1, f"{split}: Stage 1 file is not complete"
        if not QUICK_TEST:
            assert not bool(f.attrs.get("quick_test", 0)), (
                f"{split}: this is a Stage 1 quick-test file; attach the output of the full run")
        G = f["y"].shape[0]
        g = G if n_graphs is None else min(G, n_graphs)
        s.ptr = f["ptr"][:g + 1].astype(np.int64)
        s.edge_ptr = f["edge_ptr"][:g + 1].astype(np.int64)
        N, E = int(s.ptr[-1]), int(s.edge_ptr[-1])
        s.q = np.empty((N, 576), dtype=np.uint8)
        step = 2_000_000
        for a in range(0, N, step):
            b = min(N, a + step)
            f["feats_q"].read_direct(s.q, np.s_[a:b], np.s_[a:b])
        s.edges = f["delaunay_edges"][:E]
        s.y = f["y"][:g].astype(np.int64)
        s.patch = f["patch_feats"][:g]
        s.g_max = float(f["feats_q"].attrs["G_MAX"]); s.f_min = float(f["feats_q"].attrs["F_MIN"])
    s.G, s.N = g, N
    log(f"loaded {split}: {g:,} graphs, {N:,} nodes, {E:,} Delaunay edges")
    return s


def view(s, feats, knn):
    """Light object handed to the batching code: one feature array + one neighbour table."""
    return SimpleNamespace(feats=feats, knn=knn, ptr=s.ptr, edge_ptr=s.edge_ptr, edges=s.edges,
                           y=s.y, G=s.G, N=s.N, name=s.name)


# ----------------------------------------------------------------------------------------------
# Child-process helpers. The main process never touches CUDA, so forked children can.
# ----------------------------------------------------------------------------------------------
def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def run_child(fn, *args):
    def wrapped():
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    check_parent_clean()
    p = mp.get_context("fork").Process(target=wrapped)
    p.start(); p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{fn.__name__} failed in the child process (see traceback above)")


def device(i=0):
    return torch.device(f"cuda:{i}" if n_gpus() > 0 else "cpu")


def dequant(q, g_max, f_min):
    return (q.float() * (g_max / 255.0)).square_() + f_min


# ----------------------------------------------------------------------------------------------
# Preparation: standardisation + PCA (training nodes only), projections, neighbour tables
# ----------------------------------------------------------------------------------------------
def prep_child(splits, dims, knn_spaces, targets=None):
    """targets: splits to project and build neighbour tables for (default: all of them)."""
    dev = device(0)
    tr = splits["train"]
    stats_path = os.path.join(TMP, "stats.npz")
    t0 = time.time()
    s1 = torch.zeros(576, dtype=torch.float64, device=dev)
    s2 = torch.zeros(576, 576, dtype=torch.float64, device=dev)
    CH = 200_000
    for a in range(0, tr.N, CH):
        x = dequant(torch.from_numpy(tr.q[a:a + CH]).to(dev), tr.g_max, tr.f_min).double()
        s1 += x.sum(0); s2 += x.T @ x
    mean = s1 / tr.N
    cov = s2 / tr.N - torch.outer(mean, mean)
    var = torch.diagonal(cov).clamp_min(0)
    std = var.sqrt()
    dead = std < 1e-6
    std_safe = torch.where(dead, torch.ones_like(std), std)
    corr = cov / torch.outer(std_safe, std_safe)
    evals, evecs = torch.linalg.eigh(corr)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0), evecs[:, order]
    # deterministic sign: largest-magnitude loading of each component is positive
    sign = torch.sign(evecs[evecs.abs().argmax(0), torch.arange(576, device=dev)])
    evecs = evecs * sign
    fit_s = time.time() - t0
    mean32, std32, W = mean.float(), std_safe.float(), evecs.float()
    info = {"pca_fit_seconds": round(fit_s, 1), "train_nodes": int(tr.N), "dead_features": int(dead.sum()),
            "explained_variance_ratio": {str(d): float(evals[:d].sum() / evals.sum()) for d in dims},
            "projection_seconds": {}, "knn_seconds": {}}

    targets = list(splits) if targets is None else list(targets)
    for d in dims:
        t0 = time.time()
        for name in targets:
            s = splits[name]
            mm = np.lib.format.open_memmap(os.path.join(TMP, f"pca{d}_{name}.npy"), mode="w+",
                                           dtype=np.float16, shape=(s.N, d))
            for a in range(0, s.N, CH):
                x = (dequant(torch.from_numpy(s.q[a:a + CH]).to(dev), s.g_max, s.f_min) - mean32) / std32
                mm[a:a + CH] = (x @ W[:, :d]).cpu().numpy().astype(np.float16)
            mm.flush(); del mm
        info["projection_seconds"][str(d)] = round(time.time() - t0, 1)
        log(f"prep: PCA-{d} projections written ({info['explained_variance_ratio'][str(d)]:.4f} of variance)")

    GB = 256
    for space in knn_spaces:
        t0 = time.time()
        for name in targets:
            s = splits[name]
            z = None if space == "raw" else np.load(os.path.join(TMP, f"{space}_{name}.npy"), mmap_mode="r")
            out = np.lib.format.open_memmap(os.path.join(TMP, f"knn_{space}_{name}.npy"), mode="w+",
                                            dtype=np.int16, shape=(s.N, KNN_MAX))
            for g0 in range(0, s.G, GB):
                g1 = min(s.G, g0 + GB)
                a, b = int(s.ptr[g0]), int(s.ptr[g1])
                n = torch.from_numpy(s.ptr[g0 + 1:g1 + 1] - s.ptr[g0:g1]).to(dev)
                if space == "raw":
                    x = (dequant(torch.from_numpy(s.q[a:b]).to(dev), s.g_max, s.f_min) - mean32) / std32
                else:
                    x = torch.from_numpy(np.asarray(z[a:b])).to(dev).float()
                B, nmax = len(n), int(n.max())
                valid = torch.arange(nmax, device=dev)[None, :] < n[:, None]          # (B, nmax)
                pad = torch.zeros(B, nmax, x.shape[1], device=dev)
                pad[valid] = x
                dist = torch.cdist(pad, pad)
                dist.masked_fill_(~valid[:, None, :], float("inf"))
                dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
                k = min(KNN_MAX, nmax)
                dv, di = torch.topk(dist, k, dim=2, largest=False)
                di = di.masked_fill(torch.isinf(dv), -1)
                if k < KNN_MAX:
                    di = torch.cat([di, di.new_full((B, nmax, KNN_MAX - k), -1)], dim=2)
                out[a:b] = di[valid].cpu().numpy().astype(np.int16)
            out.flush(); del out
        info["knn_seconds"][space] = round(time.time() - t0, 1)
        log(f"prep: neighbour tables for {space} written")

    os.makedirs(OUT, exist_ok=True)
    for path in (stats_path, os.path.join(OUT, "pca_stats.npz")):     # second copy kept as output
        np.savez(path, mean=mean32.cpu().numpy(), std=std32.cpu().numpy(),
                 evals=evals.cpu().numpy(), components=W[:, :max(dims + [1])].cpu().numpy())
    with open(os.path.join(OUT, "prep_info.json"), "w") as f:
        json.dump(info, f, indent=2)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def ensure_pyg():
    """Install PyTorch Geometric if it is missing. It is only imported inside worker processes."""
    import importlib.util
    if importlib.util.find_spec("torch_geometric") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"])


_NET = {}


def net_class():
    """Two SAGEConv layers (64), optional Top-K pooling, global mean pool, 64-32-2 classifier."""
    if "cls" not in _NET:
        from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool

        class HybridNet(nn.Module):
            def __init__(self, in_dim, ratio, dropout=0.0):
                super().__init__()
                self.drop = nn.Dropout(dropout)
                self.c1 = SAGEConv(in_dim, 64)
                self.c2 = SAGEConv(64, 64)
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

        _NET["cls"] = HybridNet
    return _NET["cls"]


def check_parent_clean():
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was started in the main process, so worker processes cannot use the "
                           "GPUs. Restart the session (power button) and run all cells again.")


def to_dev(batch, kind, norm, dev):
    x, ei, bvec, y = batch
    x = torch.from_numpy(x).to(dev)
    if kind == "raw":
        x = (dequant(x, norm.g_max, norm.f_min) - norm.mean) / norm.std
    else:
        x = x.float()
    return x, torch.from_numpy(ei).to(dev), torch.from_numpy(bvec).to(dev), torch.from_numpy(y).to(dev)


def make_norm(g_max, f_min, dev):
    st = np.load(os.path.join(TMP, "stats.npz"))
    return SimpleNamespace(g_max=g_max, f_min=f_min, mean=torch.from_numpy(st["mean"]).to(dev),
                           std=torch.from_numpy(st["std"]).to(dev))


def evaluate(model, d, job, kind, norm, dev):
    model.eval()
    probs, loss = [], 0.0
    with torch.no_grad():
        for a in range(0, d.G, EVAL_BATCH):
            gids = np.arange(a, min(d.G, a + EVAL_BATCH))
            x, ei, bvec, y = to_dev(make_batch(d, gids, job.graph), kind, norm, dev)
            logits = model(x, ei, bvec, len(gids))
            loss += float(F.cross_entropy(logits, y, reduction="sum"))
            probs.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
    return loss / d.G, np.concatenate(probs)


def save_run(job, model_state, hist, val_p, test_p, val_y, test_y, extra):
    folder = os.path.join(OUT, job.run_id)
    os.makedirs(folder, exist_ok=True)
    torch.save(model_state, os.path.join(folder, "best.pt"))
    np.save(os.path.join(folder, "val_prob.npy"), val_p.astype(np.float32))
    np.save(os.path.join(folder, "test_prob.npy"), test_p.astype(np.float32))
    with open(os.path.join(folder, "history.json"), "w") as f:
        json.dump(hist, f)
    m = {"run_id": job.run_id, "name": job.name, "seed": job.seed, "feat": job.feat, "graph": job.graph,
         "pool": job.pool, "experiment": job.exp, "quick_test": QUICK_TEST,
         "hparams": {"lr": job.lr, "weight_decay": job.wd, "dropout": job.dropout, "batch": BATCH,
                     "evals_per_epoch": EVALS_PER_EPOCH, "patience_checks": PATIENCE, "optimizer": "AdamW"},
         "val": metrics_from_probs(val_p, val_y), "test": metrics_from_probs(test_p, test_y), **extra}
    with open(os.path.join(folder, "metrics.json"), "w") as f:     # written last: marks the run as done
        json.dump(m, f, indent=2)
    log(f"{job.run_id}: test acc {m['test']['accuracy']:.4f}, auc {m['test']['auc']}, "
        + (f"val loss {extra['best_val_loss']:.4f}, " if "best_val_loss" in extra else "")
        + f"best epoch {extra.get('best_epoch')}, epochs run {extra.get('epochs_run')}")


def train_run(job, data, kind, norm, dev):
    torch.manual_seed(job.seed)
    rng = np.random.default_rng(job.seed)
    tr, va, te = data["train"], data["valid"], data["test"]
    model = net_class()(tr.feats.shape[1], pool_ratio(job.pool), job.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=job.lr, weight_decay=job.wd)
    n_batches = math.ceil(tr.G / BATCH)
    checks = sorted({round(n_batches * (i + 1) / EVALS_PER_EPOCH) for i in range(EVALS_PER_EPOCH)})
    best_loss, best_state, best_at, bad, hist, stop = float("inf"), None, -1.0, 0, [], False
    t_train = 0.0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(tr.G)
        t0, tot, correct, seen = time.time(), 0.0, 0, 0
        for bi, a in enumerate(range(0, tr.G, BATCH)):
            gids = perm[a:a + BATCH]
            x, ei, bvec, y = to_dev(make_batch(tr, gids, job.graph), kind, norm, dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x, ei, bvec, len(gids))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(gids); correct += int((logits.argmax(1) == y).sum()); seen += len(gids)
            if bi + 1 in checks:
                t_train += time.time() - t0
                vloss, vprob = evaluate(model, va, job, kind, norm, dev)
                model.train()
                at = round(ep + (bi + 1) / n_batches, 3)
                hist.append({"epoch": at, "train_loss": tot / seen, "train_acc": correct / seen,
                             "val_loss": vloss, "val_acc": float(((vprob >= 0.5) == va.y).mean())})
                tot, correct, seen = 0.0, 0, 0
                if vloss < best_loss - 1e-4:
                    best_loss, best_at, bad = vloss, at, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        stop = True
                        break
                t0 = time.time()
        if stop:
            break
    if best_state is None:      # validation loss never finite; keep the final weights and record it
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, vprob = evaluate(model, va, job, kind, norm, dev)
    _, tprob = evaluate(model, te, job, kind, norm, dev)
    epochs_run = hist[-1]["epoch"] if hist else 0.0
    save_run(job, best_state, hist, vprob, tprob, va.y, te.y,
             {"best_epoch": best_at, "epochs_run": epochs_run, "stopped_early": stop,
              "best_val_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters()),
              "train_seconds_per_epoch": t_train / max(epochs_run, 1e-9),
              "torch": torch.__version__, "torch_geometric": __import__("torch_geometric").__version__,
              "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"})


def worker(gpu, jobs, data, kind, g_max, f_min, deadline):
    torch.set_num_threads(2)
    dev = device(gpu)
    norm = make_norm(g_max, f_min, dev)
    for job in jobs:
        if time.time() > deadline:
            log(f"time budget reached; {job.run_id} left for the next session")
            continue
        if done(job):
            continue
        try:
            train_run(job, data, kind, norm, dev)
        except Exception:
            log(f"{job.run_id} FAILED")
            traceback.print_exc()


def run_jobs(jobs, data, kind, g_max, f_min, deadline):
    jobs = [j for j in jobs if not done(j)]
    if not jobs:
        return
    check_parent_clean()
    k = max(1, n_gpus())
    procs = []
    for gpu in range(k):
        p = mp.get_context("fork").Process(target=worker,
                                           args=(gpu, jobs[gpu::k], data, kind, g_max, f_min, deadline))
        p.start(); procs.append(p)
    for p in procs:
        p.join()


# ----------------------------------------------------------------------------------------------
# Resource measurements: one configuration at a time, one GPU, nothing else running
# ----------------------------------------------------------------------------------------------
def bench_child(jobs, data, kind, g_max, f_min, tag):
    dev = device(0)
    norm = make_norm(g_max, f_min, dev)
    tr, te = data["train"], data["test"]
    rows = []
    for job in jobs:
        for rep in range(BENCH_REPS):
            torch.manual_seed(1000 + rep)
            rng = np.random.default_rng(1000 + rep)
            model = net_class()(tr.feats.shape[1], pool_ratio(job.pool), DROPOUT).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            perm = rng.permutation(tr.G)
            model.train()
            for a in range(0, min(tr.G, 20 * BATCH), BATCH):      # warm-up, not timed
                x, ei, bvec, y = to_dev(make_batch(tr, perm[a:a + BATCH], job.graph), kind, norm, dev)
                opt.zero_grad(); F.cross_entropy(model(x, ei, bvec, len(y)), y).backward(); opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
            t_data, n_edges, n_nodes, t0 = 0.0, 0, 0, time.perf_counter()
            for a in range(0, tr.G, BATCH):
                t = time.perf_counter()
                batch = make_batch(tr, perm[a:a + BATCH], job.graph)
                t_data += time.perf_counter() - t
                n_edges += batch[1].shape[1]; n_nodes += batch[0].shape[0]
                x, ei, bvec, y = to_dev(batch, kind, norm, dev)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x, ei, bvec, len(y)), y).backward()
                opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            epoch_s = time.perf_counter() - t0
            peak_train = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            evaluate(model, te, job, kind, norm, dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            infer_s = time.perf_counter() - t0
            rows.append({
                "name": job.name, "feat": job.feat, "graph": job.graph, "pool": job.pool, "rep": rep,
                "train_graphs": int(tr.G), "train_epoch_seconds": epoch_s,
                "batch_assembly_seconds": t_data, "gpu_compute_seconds": epoch_s - t_data,
                "peak_gpu_mb_train": peak_train,
                "peak_gpu_mb_inference": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None,
                "test_graphs": int(te.G), "inference_seconds": infer_s,
                "inference_ms_per_graph": 1000 * infer_s / te.G,
                "parameters": sum(p.numel() for p in model.parameters()),
                "input_dim": int(tr.feats.shape[1]),
                "feature_bytes_per_node": int(tr.feats.shape[1] * tr.feats.dtype.itemsize),
                "mean_nodes_per_graph": n_nodes / tr.G, "mean_directed_edges_per_graph": n_edges / tr.G,
                "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            })
            del model, opt
        r = [x for x in rows if x["name"] == job.name]
        log(f"bench {job.name}: epoch {np.mean([x['train_epoch_seconds'] for x in r]):.1f} s, "
            f"peak GPU {np.mean([x['peak_gpu_mb_train'] or 0 for x in r]):.0f} MB, "
            f"inference {np.mean([x['inference_ms_per_graph'] for x in r]):.3f} ms/graph")
    with open(os.path.join(OUT, f"benchmark_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


# ----------------------------------------------------------------------------------------------
# Whole-patch baselines (no graph): frozen CNN features of the 96x96 patch
# ----------------------------------------------------------------------------------------------
def patch_child(jobs, splits, deadline):
    dev = device(0)
    X = {k: torch.from_numpy(s.patch.astype(np.float32)).to(dev) for k, s in splits.items()}
    Y = {k: torch.from_numpy(s.y).to(dev) for k, s in splits.items()}
    mu, sd = X["train"].mean(0), X["train"].std(0).clamp_min(1e-6)
    X = {k: (v - mu) / sd for k, v in X.items()}

    def predict(model, k):
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(X[k][a:a + 8192]) for a in range(0, len(X[k]), 8192)])
        return float(F.cross_entropy(logits, Y[k])), torch.softmax(logits, 1)[:, 1].cpu().numpy()

    for job in jobs:
        if done(job) or time.time() > deadline:
            continue
        torch.manual_seed(job.seed)
        g = torch.Generator(device="cpu").manual_seed(job.seed)
        model = (nn.Sequential(nn.Dropout(job.dropout), nn.Linear(576, 2)) if job.pool == "linear" else
                 nn.Sequential(nn.Linear(576, 64), nn.ReLU(), nn.Dropout(job.dropout), nn.Linear(64, 32),
                               nn.ReLU(), nn.Dropout(job.dropout), nn.Linear(32, 2))).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=job.lr, weight_decay=job.wd)
        best_loss, best_state, best_epoch, bad, hist = float("inf"), None, -1, 0, []
        n = len(X["train"])
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for a in range(0, n, 512):
                idx = perm[a:a + 512]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(X["train"][idx]), Y["train"][idx]).backward()
                opt.step()
            vloss, vprob = predict(model, "valid")
            hist.append({"epoch": ep + 1, "val_loss": vloss,
                         "val_acc": float(((vprob >= 0.5) == splits["valid"].y).mean())})
            if vloss < best_loss - 1e-4:
                best_loss, best_epoch, bad = vloss, ep + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        save_run(job, best_state, hist, predict(model, "valid")[1], predict(model, "test")[1],
                 splits["valid"].y, splits["test"].y,
                 {"best_epoch": best_epoch, "epochs_run": len(hist), "stopped_early": bad >= PATIENCE,
                  "parameters": sum(p.numel() for p in model.parameters())})


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------
def summarise():
    import csv
    runs = []
    for path in sorted(glob.glob(os.path.join(OUT, "*", "metrics.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        return
    keys = ["accuracy", "auc", "f1", "sensitivity", "specificity", "nll", "brier", "ece"]
    tune = [r for r in runs if r.get("experiment") == "tune"]
    if tune:
        print(f"\n{'tuning run (selection on validation only)':42s} {'val nll':>8s} {'val auc':>8s} "
              f"{'val acc':>8s} {'best ep':>8s} {'epochs':>7s}")
        for r in sorted(tune, key=lambda r: (r["name"], r["val"]["nll"])):
            print(f"{r['run_id']:42s} {r['val']['nll']:8.4f} {r['val']['auc']:8.4f} "
                  f"{100 * r['val']['accuracy']:8.2f} {r.get('best_epoch')!s:>8s} {r.get('epochs_run')!s:>7s}")
        hp = {}
        for r in tune:
            key = (r["hparams"]["lr"], r["hparams"]["weight_decay"], r["hparams"]["dropout"])
            hp.setdefault(key, []).append(r["val"]["nll"])
        best = min(hp, key=lambda k: np.mean(hp[k]))
        print(f"lowest mean validation loss over the tuned configurations: lr={best[0]:g}, "
              f"weight_decay={best[1]:g}, dropout={best[2]:g}")
        with open(os.path.join(OUT, "tuning_choice.json"), "w") as f:
            json.dump({"lr": best[0], "weight_decay": best[1], "dropout": best[2],
                       "mean_val_nll": {hp_tag(*k): float(np.mean(v)) for k, v in hp.items()}}, f, indent=2)
    runs = [r for r in runs if r.get("experiment") != "tune"]
    if not runs:
        return
    with open(os.path.join(OUT, "all_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "name", "seed", "feat", "graph", "pool", "best_epoch", "epochs_run"]
                   + [f"test_{k}" for k in keys] + ["val_accuracy", "val_nll"])
        for r in runs:
            w.writerow([r["run_id"], r["name"], r["seed"], r["feat"], r["graph"], r["pool"],
                        r.get("best_epoch"), r.get("epochs_run")] + [r["test"].get(k) for k in keys]
                       + [r["val"]["accuracy"], r["val"]["nll"]])
    names = sorted({r["name"] for r in runs})
    print(f"\n{'config':16s} {'runs':>4s} {'test acc mean':>14s} {'sd':>7s} {'auc mean':>9s} {'sd':>7s} {'ece':>7s} {'epochs':>7s}")
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "feat", "graph", "pool", "runs"] + [f"test_{k}_{s}" for k in keys for s in ("mean", "sd")]
                   + ["epochs_run_mean"])
        for nme in names:
            rs = [r for r in runs if r["name"] == nme]
            row = [nme, rs[0]["feat"], rs[0]["graph"], rs[0]["pool"], len(rs)]
            st = {}
            for k in keys:
                v = np.array([r["test"][k] for r in rs if r["test"].get(k) is not None], dtype=float)
                st[k] = (float(v.mean()) if len(v) else float("nan"),
                         float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
                row += list(st[k])
            ep = float(np.mean([r.get("epochs_run", 0) for r in rs]))
            w.writerow(row + [ep])
            print(f"{nme:16s} {len(rs):4d} {100 * st['accuracy'][0]:14.2f} {100 * st['accuracy'][1]:7.2f} "
                  f"{st['auc'][0]:9.4f} {st['auc'][1]:7.4f} {st['ece'][0]:7.4f} {ep:7.1f}")


def restore_previous():
    """Copy finished runs from an attached earlier version of this notebook's output."""
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for path in glob.glob("/kaggle/input/**/stage2_results/*/metrics.json", recursive=True):
        try:
            with open(path) as f:
                m = json.load(f)
            if bool(m.get("quick_test")) != QUICK_TEST or "hparams" not in m:
                continue
            if m["experiment"] != "tune" and (m["hparams"]["lr"], m["hparams"]["weight_decay"],
                                              m["hparams"]["dropout"]) != (LR, WEIGHT_DECAY, DROPOUT):
                continue
        except Exception:
            continue
        src = os.path.dirname(path)
        dst = os.path.join(OUT, os.path.basename(src))
        if not os.path.exists(os.path.join(dst, "metrics.json")):
            shutil.copytree(src, dst, dirs_exist_ok=True); n += 1
    for path in glob.glob("/kaggle/input/**/stage2_results/*.json", recursive=True):
        dst = os.path.join(OUT, os.path.basename(path))
        if not os.path.exists(dst) and not QUICK_TEST:
            shutil.copy(path, dst)
    if n:
        log(f"restored {n} finished runs from attached output")

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
