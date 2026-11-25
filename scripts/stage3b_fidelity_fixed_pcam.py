# %% [markdown]
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

CONFIGS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "A8"]
EXPLAIN_SEEDS = [0, 1, 2]
N_FIT = 500                 # validation graphs used to fit PGExplainer
N_EXPLAIN = 1000            # test graphs explained (half per class, same graphs for every model)
MIN_NODES = 8               # graphs smaller than this are not sampled
MIN_EDGES = 10              # nor graphs with fewer undirected Delaunay edges
PGX_EPOCHS = 30
PGX_LR = 0.003
SPARSITY = [0.1, 0.2, 0.3, 0.5]   # fraction of edges kept as the explanation
SAMPLE_SEED = 12345
METHODS = ("pgexplainer", "saliency", "random")
N_BOOT = 2000
TIME_BUDGET_H = 10.0

# settings read by the shared Stage 2 code
KNN_MAX = 12
EVAL_BATCH = 512
BATCH = 128
LR = 1e-3
MAX_EPOCHS, PATIENCE, BENCH_REPS = 1, 1, 1
EXPERIMENTS, SEEDS, SWEEP_SEEDS = [], [], []
SUBSET = {"train": None, "valid": None, "test": None}
OUT = "/kaggle/working/stage3b_results"
TMP = "/kaggle/temp/stage3b"
CORE = {
    "M1": ("pca128", "delaunay", "none"),
    "M2": ("raw", "knn6", "none"),
    "M3": ("raw", "delaunay", "topk0.5"),
    "M4": ("pca128", "knn6", "none"),
    "M5": ("pca128", "delaunay", "topk0.5"),
    "M6": ("raw", "knn6", "topk0.5"),
    "M7": ("pca128", "knn6", "topk0.5"),
    "M8": ("raw", "delaunay", "none"),
    "A8": ("raw", "delaunay", "none"),     # GAT (Stage 6)
}
BASELINES, SWEEPS = {}, {}


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


def net_class(arch="sage"):
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
        f"val loss {extra.get('best_val_loss', float('nan')):.4f}, "
        f"best epoch {extra.get('best_epoch')}, epochs run {extra.get('epochs_run')}")


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
    for method in METHODS:
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
    """Checkpoint folders of Stage 2 (M configs) and Stage 6 (A8), full runs only."""
    found = {}
    for pat in ("/kaggle/input/**/stage2_results/*/metrics.json", "/kaggle/input/**/stage6_results/*/metrics.json"):
        for path in glob.glob(pat, recursive=True):
            with open(path) as f:
                m = json.load(f)
            if not m.get("quick_test"):
                found[m["run_id"]] = os.path.dirname(path)
    return found


# ----------------------------------------------------------------------------------------------
# GPU side
# ----------------------------------------------------------------------------------------------
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
            for method in METHODS:
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


# %%
def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    ensure_pyg()
    deadline = T_START + TIME_BUDGET_H * 3600
    for path in glob.glob("/kaggle/input/**/stage3b_results/*/fidelity.json", recursive=True):
        with open(path) as f:
            pass
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
        raise RuntimeError(f"Checkpoints not found for {missing}. Attach the finished Stage 2 and Stage 6 outputs.")
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
    analyse()
    left = [j.run_id for j in jobs if not os.path.exists(os.path.join(OUT, j.run_id, "fidelity.json"))]
    log(f"finished; {len(left)} runs not done" + (f": {left}" if left else ""))


if __name__ == "__main__":
    main()
