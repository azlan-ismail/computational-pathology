# %% [markdown]
# # Stage 1 - PatchCamelyon graph preparation
#
# Run this once on Kaggle before any training.
#
# **Notebook settings (right-hand panel):** Accelerator = GPU T4 x2 (or P100), Internet = On.
#
# What it does, for each official split (test, valid, train):
# 1. downloads the official PatchCamelyon file from Zenodo and checks its MD5 against the
#    published value;
# 2. detects candidate entities in every patch (same rule as the manuscript: grayscale,
#    20th-percentile threshold, connected components of at least 4 pixels);
# 3. encodes a 32x32 crop around every entity with frozen ImageNet MobileNetV3-Small (576 features);
# 4. builds the reference Delaunay edges from the entity centroids;
# 5. also encodes each whole patch once (needed for the CNN-only baseline);
# 6. writes one HDF5 file per split plus `stage1_report.json` to `/kaggle/working/pcam_graphs`.
#
# Storage: PatchCamelyon yields about 85 entities per patch, i.e. about 28 million nodes. To fit
# Kaggle's 20 GB output limit the 576 node features are stored as 8-bit values after a square-root
# transform (`q = round(255 * sqrt(f + 0.375) / sqrt(F_CAP + 0.375))`); the training code inverts
# this. The quick test measures how much precision this costs and prints it in the report.
#
# Nothing here is random, so a second run gives the same files.
# The run is resumable: a finished split is skipped if the session is restarted.
#
# First time: leave `QUICK_TEST = True`, press Run All (about 5 minutes; test split only), and
# check that the last cell prints a report. Then set `QUICK_TEST = False` and use **Save Version -> Save & Run All
# (Commit)** so it runs unattended.

# %%
QUICK_TEST = True          # True: first 2,000 patches of the test split only. False: everything.

OUT_DIR = "/kaggle/working/pcam_graphs"   # kept when the notebook version is saved (20 GB limit)
RAW_DIR = "/kaggle/temp/pcam_raw"               # scratch space for the downloaded images, deleted afterwards
OUTPUT_BUDGET_GB = 18.5                   # stop early if the features would not fit in /kaggle/working
F_MIN = -0.375             # lowest possible feature value (minimum of the hardswish activation)
F_CAP = 16.0               # feature values above this are clipped when stored (counted in the report)

CROP = 32                  # crop side around each centroid, pixels
RESIZE = 96                # crops are resized to this before the CNN
MIN_COMPONENT = 4          # smallest connected component kept, pixels
PERCENTILE = 20            # grayscale percentile used as the threshold
SUPER_BATCH = 2048         # patches read from disk at a time
GPU_BATCH = 2048           # crops per CNN forward pass

ZENODO = [
    "https://zenodo.org/records/2546921/files/{name}?download=1",
    "https://zenodo.org/record/2546921/files/{name}",
]
MD5 = {  # MD5 of each .h5.gz file. Values as recorded in torchvision/datasets/pcam.py. The table in
         # https://github.com/basveeling/pcam lists the same six hashes but with the valid and test
         # rows exchanged; the files served by Zenodo match the torchvision assignment.
    "camelyonpatch_level_2_split_train_x.h5.gz": "1571f514728f59376b705fc836ff4b63",
    "camelyonpatch_level_2_split_train_y.h5.gz": "35c2d7259d906cfc8143347bb8e05be7",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    "camelyonpatch_level_2_split_test_x.h5.gz": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    "camelyonpatch_level_2_split_test_y.h5.gz": "60a7035772fbdb7f34eb86d4420cf66a",
}
EXPECTED = {"train": 262144, "valid": 32768, "test": 32768}

# %%
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import urllib.request

import numpy as np
import scipy
import scipy.ndimage as ndi
from scipy.spatial import Delaunay

GRAY_W = np.array([0.2989, 0.5870, 0.1140])
HALF = CROP // 2


# ----------------------------------------------------------------------------------------------
# CPU side: entity detection, crops, Delaunay edges (pure numpy / scipy)
# ----------------------------------------------------------------------------------------------
def detect_entities(img):
    """Centroids (row, col) of candidate entities in one uint8 HxWx3 patch.

    Same rule as src/dataset.py::extract_cell_crops in the original repository: grayscale,
    pixels darker than the 20th percentile, 4-connected components with at least 4 pixels,
    integer-truncated centre of mass.
    """
    gray = np.dot(img[..., :3].astype(np.float32) / 255.0, GRAY_W)
    mask = gray < np.percentile(gray, PERCENTILE)
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, lab, idx)
    keep = idx[sizes >= MIN_COMPONENT]
    if keep.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    com = np.asarray(ndi.center_of_mass(mask, lab, keep), dtype=np.float64).reshape(-1, 2)
    return com.astype(np.int64)  # truncation, like int() in the original


def cut_crops(img, cents):
    """uint8 crops (n, CROP, CROP, 3) centred on each centroid, with reflection padding."""
    pad = np.pad(img, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="reflect")
    out = np.empty((len(cents), CROP, CROP, 3), dtype=np.uint8)
    for k, (r, c) in enumerate(cents):
        out[k] = pad[r:r + CROP, c:c + CROP, :]
    return out


def delaunay_edges(cents):
    """Undirected Delaunay edges (i < j) between centroids, as an (E, 2) int16 array.

    Fallbacks: one node -> no edges; two nodes -> one edge; degenerate layouts that Qhull
    rejects (e.g. all centroids on one line) -> a chain through the points in sorted order.
    """
    n = len(cents)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int16)
    if n == 2:
        return np.array([[0, 1]], dtype=np.int16)
    try:
        tri = Delaunay(cents.astype(np.float64))
        s = tri.simplices
        e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]], axis=0)
    except Exception:
        order = np.lexsort((cents[:, 1], cents[:, 0]))
        e = np.stack([order[:-1], order[1:]], axis=1)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return e.astype(np.int16)


def process_chunk(imgs):
    """Worker: returns per-patch node counts, centroids, crops, edges for a block of patches."""
    counts, placeholder, cents_all, crops_all, edges_all, ecounts = [], [], [], [], [], []
    for img in imgs:
        cents = detect_entities(img)
        if len(cents) == 0:
            # No valid entity: one placeholder node built from an all-zero crop, no edges
            # (the original code feeds a zero image through the CNN in this case).
            placeholder.append(1)
            cents = np.array([[img.shape[0] // 2, img.shape[1] // 2]], dtype=np.int64)
            crops = np.zeros((1, CROP, CROP, 3), dtype=np.uint8)
            edges = np.zeros((0, 2), dtype=np.int16)
        else:
            placeholder.append(0)
            crops = cut_crops(img, cents)
            edges = delaunay_edges(cents)
        counts.append(len(cents))
        cents_all.append(cents.astype(np.uint8))
        crops_all.append(crops)
        edges_all.append(edges)
        ecounts.append(len(edges))
    return (np.asarray(counts, dtype=np.int64), np.asarray(placeholder, dtype=np.uint8),
            np.concatenate(cents_all), np.concatenate(crops_all),
            np.concatenate(edges_all), np.asarray(ecounts, dtype=np.int64))


# ----------------------------------------------------------------------------------------------
# 8-bit feature storage
# ----------------------------------------------------------------------------------------------
G_MAX = float(np.sqrt(F_CAP - F_MIN))


def quantize(f):
    """float features -> uint8 codes (square-root companding, clipped to [F_MIN, F_CAP])."""
    g = np.sqrt(np.clip(f.astype(np.float32), F_MIN, F_CAP) - F_MIN)
    return np.rint(g * (255.0 / G_MAX)).astype(np.uint8)


def dequantize(q):
    """uint8 codes -> float32 features (inverse of quantize, up to rounding)."""
    g = q.astype(np.float32) * (G_MAX / 255.0)
    return g * g + F_MIN


def knn_sets(x, k=6):
    """Indices of the k nearest other nodes (Euclidean) for each row of x."""
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def precision_report(f32, counts, max_graphs=300):
    """How much the 8-bit storage changes the features (run in the quick test only)."""
    f32 = f32.astype(np.float32)
    back = dequantize(quantize(f32))
    err = back - f32
    std = f32.std(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    ratio = rmse / np.maximum(std, 1e-6)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    overlap, top1 = [], []
    for g in range(min(max_graphs, len(counts))):
        a, b = f32[ptr[g]:ptr[g + 1]], back[ptr[g]:ptr[g + 1]]
        if len(a) < 8:
            continue
        ka, kb = knn_sets(a), knn_sets(b)
        overlap.append(np.mean([len(set(r) & set(t)) / 6.0 for r, t in zip(ka, kb)]))
        top1.append(float((ka[:, 0] == kb[:, 0]).mean()))
    pct = np.percentile(f32, [0, 1, 50, 99, 99.99, 100])
    return {
        "feature_min_p1_p50_p99_p99.99_max": [float(v) for v in pct],
        "fraction_clipped_above_cap": float((f32 > F_CAP).mean()),
        "fraction_below_floor": float((f32 < F_MIN).mean()),
        "per_feature_std_mean": float(std.mean()),
        "rmse_over_std_mean": float(ratio.mean()),
        "rmse_over_std_p95_across_features": float(np.percentile(ratio, 95)),
        "rmse_over_std_max": float(ratio.max()),
        "max_abs_error": float(np.abs(err).max()),
        "knn6_neighbour_overlap_mean": float(np.mean(overlap)) if overlap else None,
        "knn_nearest_neighbour_agreement": float(np.mean(top1)) if top1 else None,
        "graphs_used_for_knn": len(overlap),
    }


# ----------------------------------------------------------------------------------------------
# Download with checksum
# ----------------------------------------------------------------------------------------------
class _HashingReader:
    def __init__(self, raw):
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0

    def read(self, size=-1):
        b = self.raw.read(size)
        self.md5.update(b)
        self.n += len(b)
        return b


def find_local(name_h5):
    """Use a copy already attached under /kaggle/input if there is one (unverified)."""
    for root, _, files in os.walk("/kaggle/input"):
        if name_h5 in files:
            return os.path.join(root, name_h5)
    return None


def fetch(name_gz, dest_h5):
    """Stream-download name_gz, decompress to dest_h5, and verify the MD5 of the .gz stream."""
    last = None
    for attempt in range(3):
        for tmpl in ZENODO:
            url = tmpl.format(name=name_gz)
            try:
                print(f"  downloading {name_gz} (attempt {attempt + 1})", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "pcam-stage1"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    hr = _HashingReader(resp)
                    with gzip.GzipFile(fileobj=hr) as gz, open(dest_h5 + ".part", "wb") as out:
                        shutil.copyfileobj(gz, out, length=16 * 1024 * 1024)
                    while hr.read(1024 * 1024):   # drain anything gzip did not consume
                        pass
                got = hr.md5.hexdigest()
                if got != MD5[name_gz]:
                    raise IOError(f"MD5 mismatch for {name_gz}: got {got}, expected {MD5[name_gz]}")
                os.replace(dest_h5 + ".part", dest_h5)
                return "md5 verified (hash list: torchvision/datasets/pcam.py)"
            except Exception as exc:  # try the other URL form, then retry
                last = exc
                print(f"    failed: {exc}", flush=True)
                time.sleep(10)
    raise RuntimeError(f"Could not download {name_gz}: {last}")


def get_file(split, kind):
    name_h5 = f"camelyonpatch_level_2_split_{split}_{kind}.h5"
    local = find_local(name_h5)
    if local:
        return local, "found under /kaggle/input (third-party copy, MD5 not checked)", False
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, name_h5)
    need_gb = 8.0 if (split == "train" and kind == "x") else 1.5
    free_gb = shutil.disk_usage(RAW_DIR).free / 1e9
    if not os.path.exists(dest) and free_gb < need_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free in {RAW_DIR}; {need_gb} GB needed for "
                           f"{name_h5}. Change RAW_DIR to a location with more space.")
    if os.path.exists(dest):
        return dest, "already downloaded in this session", True
    return dest, fetch(name_h5 + ".gz", dest), True


# ----------------------------------------------------------------------------------------------
# GPU side: frozen MobileNetV3-Small encoder
# ----------------------------------------------------------------------------------------------
def build_encoder():
    import torch
    import torch.nn as nn
    import torchvision.models as models

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    enc = nn.Sequential(m.features, m.avgpool, nn.Flatten()).to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return enc, dev, mean, std


def encode(enc, dev, mean, std, u8):
    """u8: (n, h, w, 3) uint8 -> (n, 576) float32. Resize to 96x96 (bilinear), then normalise."""
    import torch
    import torch.nn.functional as F

    outs = []
    with torch.no_grad():
        for i in range(0, len(u8), GPU_BATCH):
            x = torch.from_numpy(u8[i:i + GPU_BATCH]).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
            if x.shape[-1] != RESIZE or x.shape[-2] != RESIZE:
                x = F.interpolate(x, size=(RESIZE, RESIZE), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            outs.append(enc(x).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 576), dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# One split
# ----------------------------------------------------------------------------------------------
def prepare_split(split, pool, encoder, report):
    import h5py

    out_path = os.path.join(OUT_DIR, f"pcam_{split}.h5")
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            if f.attrs.get("complete", 0) == 1 and bool(f.attrs.get("quick_test", 0)) == QUICK_TEST:
                print(f"[{split}] already complete, skipping")
                report["splits"][split] = json.loads(f.attrs.get("report", "{}"))
                return
        os.remove(out_path)

    t_dl = time.time()
    x_path, x_src, x_tmp = get_file(split, "x")
    y_path, y_src, y_tmp = get_file(split, "y")
    download_seconds = time.time() - t_dl
    enc, dev, mean, std = encoder
    t0 = time.time()
    keep_float, clipped = [], 0

    with h5py.File(x_path, "r") as fx, h5py.File(y_path, "r") as fy:
        X = fx["x"]
        y = np.asarray(fy["y"]).reshape(-1).astype(np.uint8)
        total = X.shape[0]
        assert X.shape[1:] == (96, 96, 3), X.shape
        assert total == EXPECTED[split] == len(y), (total, len(y))
        n = min(total, 2000) if QUICK_TEST else total

        counts, placeholder, cents, edges, ecounts = [], [], [], [], []
        n_nodes = 0
        with h5py.File(out_path, "w") as fo:
            d_feat = fo.create_dataset("feats_q", shape=(0, 576), maxshape=(None, 576),
                                       dtype="uint8", chunks=(8192, 576))
            d_feat.attrs["encoding"] = "f = (q * G_MAX / 255) ** 2 + F_MIN"
            d_feat.attrs["F_MIN"], d_feat.attrs["F_CAP"], d_feat.attrs["G_MAX"] = F_MIN, F_CAP, G_MAX
            d_patch = fo.create_dataset("patch_feats", shape=(n, 576), dtype="float16")
            for start in range(0, n, SUPER_BATCH):
                imgs = X[start:min(start + SUPER_BATCH, n)]
                parts = np.array_split(imgs, max(1, min(len(imgs), pool._processes * 4)))
                for c, ph, ce, cr, ed, ec in pool.map(process_chunk, parts):
                    f32 = encode(enc, dev, mean, std, cr)
                    clipped += int((f32 > F_CAP).sum())
                    if QUICK_TEST:
                        keep_float.append(f32.astype(np.float16))
                    d_feat.resize(n_nodes + len(f32), axis=0)
                    d_feat[n_nodes:n_nodes + len(f32)] = quantize(f32)
                    n_nodes += len(f32)
                    counts.append(c); placeholder.append(ph); cents.append(ce)
                    edges.append(ed); ecounts.append(ec)
                d_patch[start:start + len(imgs)] = encode(enc, dev, mean, std, imgs).astype(np.float16)

                done = start + len(imgs)
                if start == 0 and not QUICK_TEST:
                    per_patch = n_nodes / done
                    proj = per_patch * sum(EXPECTED.values()) * 590 / 1e9   # 576 feature bytes + edges etc.
                    print(f"[{split}] {per_patch:.1f} nodes per patch -> about {proj:.1f} GB of "
                          f"output for the whole dataset", flush=True)
                    if proj > OUTPUT_BUDGET_GB:
                        raise RuntimeError(
                            f"Projected feature size {proj:.1f} GB exceeds {OUTPUT_BUDGET_GB} GB. "
                            "Stop here and report this number; the storage layout needs changing.")
                if (start // SUPER_BATCH) % 10 == 0:
                    rate = done / (time.time() - t0)
                    print(f"[{split}] {done:,}/{n:,} patches, {n_nodes:,} nodes, "
                          f"{rate:.0f} patches/s", flush=True)

            counts = np.concatenate(counts)
            ecounts = np.concatenate(ecounts)
            fo.create_dataset("ptr", data=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
            fo.create_dataset("edge_ptr", data=np.concatenate([[0], np.cumsum(ecounts)]).astype(np.int64))
            fo.create_dataset("delaunay_edges", data=np.concatenate(edges).astype(np.int16))
            fo.create_dataset("centroids", data=np.concatenate(cents).astype(np.uint8))
            fo.create_dataset("placeholder", data=np.concatenate(placeholder).astype(np.uint8))
            fo.create_dataset("y", data=y[:n])
            assert fo["ptr"][-1] == n_nodes == fo["feats_q"].shape[0] == fo["centroids"].shape[0]
            assert fo["edge_ptr"][-1] == fo["delaunay_edges"].shape[0]
            fo.attrs["quick_test"] = int(QUICK_TEST)
            fo.attrs["complete"] = 1

    q = np.percentile(counts, [0, 5, 25, 50, 75, 95, 100])
    report["splits"][split] = {
        "patches": int(n), "nodes": int(n_nodes), "undirected_delaunay_edges": int(ecounts.sum()),
        "nodes_per_patch_mean": float(counts.mean()),
        "nodes_per_patch_min_p5_p25_p50_p75_p95_max": [float(v) for v in q],
        "placeholder_patches": int(np.concatenate(placeholder).sum()),
        "positive_fraction": float(y[:n].mean()),
        "images_source": x_src, "labels_source": y_src,
        "download_seconds": round(download_seconds, 1),
        "processing_seconds": round(time.time() - t0, 1),
        "nodes_per_second": round(n_nodes / max(time.time() - t0, 1e-9)),
        "feature_values_clipped_above_cap": clipped,
        "output_gb": round(os.path.getsize(out_path) / 1e9, 3),
    }
    if QUICK_TEST and keep_float:
        report["storage_precision_check"] = precision_report(np.concatenate(keep_float), counts)
    with h5py.File(out_path, "a") as fo:
        fo.attrs["report"] = json.dumps(report["splits"][split])
    print(f"[{split}] done in {(time.time() - t0) / 60:.1f} min: {n_nodes:,} nodes, "
          f"{report['splits'][split]['output_gb']} GB", flush=True)
    for path, tmp in ((x_path, x_tmp), (y_path, y_tmp)):
        if tmp and not QUICK_TEST:
            os.remove(path)   # free scratch space before the next split


# %%
def main():
    import h5py
    import torch
    import torchvision

    os.makedirs(OUT_DIR, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2))
    pool = mp.get_context("fork").Pool(workers)       # created before CUDA is initialised
    encoder = build_encoder()
    dev = encoder[1]
    report = {
        "quick_test": QUICK_TEST,
        "settings": {"crop": CROP, "resize": RESIZE, "min_component": MIN_COMPONENT,
                     "percentile": PERCENTILE, "encoder": "torchvision mobilenet_v3_small IMAGENET1K_V1",
                     "node_feature_storage": "uint8, square-root companded", "F_MIN": F_MIN,
                     "F_CAP": F_CAP, "patch_feature_dtype": "float16", "train_augmentation": "none"},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                        "cpu_count": os.cpu_count()},
        "splits": {},
    }
    if dev.type != "cuda":
        print("WARNING: no GPU found. Turn on the accelerator in the notebook settings.")
    try:
        # small splits first, so problems show early; the quick test only needs the test split
        for split in (("test",) if QUICK_TEST else ("test", "valid", "train")):
            prepare_split(split, pool, encoder, report)
    finally:
        pool.close()
        pool.join()
    report["free_gb_in_working"] = round(shutil.disk_usage(OUT_DIR).free / 1e9, 1)
    with open(os.path.join(OUT_DIR, "stage1_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
