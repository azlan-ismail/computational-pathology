"""CPU-side checks for stage 1 (no torch / h5py needed). Usage: python -I test_stage1.py <pkg_dir>"""
import importlib.util
import sys

import numpy as np
import scipy.ndimage as ndi

spec = importlib.util.spec_from_file_location("stage1", sys.argv[1] + "/stage1_prepare_pcam.py")
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)


def original_centroids(img_u8):
    """Verbatim logic of extract_cell_crops in the authors' src/dataset.py (numpy only)."""
    img_np = img_u8.astype(np.float32) / 255.0
    gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    thresh = np.percentile(gray, 20)
    binary_mask = gray < thresh
    labeled_array, num_features = ndi.label(binary_mask)
    centroids = []
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        if np.sum(mask) >= 4:
            com = ndi.center_of_mass(mask)
            centroids.append((int(com[0]), int(com[1])))
    return centroids, img_np


def original_crops(img_np, centroids):
    padded = np.pad(img_np, ((16, 16), (16, 16), (0, 0)), mode="reflect")
    return [padded[r + 16 - 16:r + 16 + 16, c + 16 - 16:c + 16 + 16, :] for r, c in centroids]


rng = np.random.default_rng(0)
n_nodes = []
for t in range(300):
    # blobby synthetic "tissue": smoothed noise, plus a few degenerate cases
    if t == 0:
        img = np.full((96, 96, 3), 200, np.uint8)                 # constant image
    elif t == 1:
        img = np.full((96, 96, 3), 220, np.uint8); img[40:44, 40:44] = 10   # single entity
    elif t == 2:
        img = np.full((96, 96, 3), 220, np.uint8)
        for c in (10, 30, 50, 70):                                # collinear entities
            img[20:23, c:c + 3] = 5
    else:
        base = ndi.gaussian_filter(rng.random((96, 96)), sigma=rng.uniform(1.0, 3.0))
        base = (base - base.min()) / (np.ptp(base) + 1e-9)
        img = np.clip(base[..., None] * 255 + rng.normal(0, 8, (96, 96, 3)), 0, 255).astype(np.uint8)

    ref_c, img_np = original_centroids(img)
    got = s1.detect_entities(img)
    assert [tuple(map(int, r)) for r in got] == ref_c, f"centroid mismatch on image {t}"
    if ref_c:
        ref_crops = original_crops(img_np, ref_c)
        crops = s1.cut_crops(img, got)
        assert crops.shape == (len(ref_c), 32, 32, 3)
        for a, b in zip(crops, ref_crops):
            assert np.array_equal(a.astype(np.float32) / 255.0, b), f"crop mismatch on image {t}"
        e = s1.delaunay_edges(got)
        assert e.dtype == np.int16 and (e.size == 0 or (e[:, 0] < e[:, 1]).all())
        assert e.size == 0 or e.max() < len(got)
        assert len(np.unique(e, axis=0)) == len(e)
        if len(got) == 2:
            assert len(e) == 1
        if len(got) >= 3:
            assert len(e) >= len(np.unique(got, axis=0)) - 1, "graph cannot be connected"
    n_nodes.append(len(ref_c))

# chunk-level bookkeeping
imgs = np.stack([np.clip(ndi.gaussian_filter(rng.random((96, 96)), 2)[..., None] * 900
                         + rng.normal(0, 5, (96, 96, 3)), 0, 255).astype(np.uint8) for _ in range(40)]
                + [np.full((96, 96, 3), 128, np.uint8)])
counts, ph, cents, crops, edges, ecounts = s1.process_chunk(imgs)
assert len(counts) == len(ph) == len(ecounts) == 41
assert counts.sum() == len(cents) == len(crops) and ecounts.sum() == len(edges)
assert ph[-1] == 1 and counts[-1] == 1 and ecounts[-1] == 0 and not crops[-1].any()
assert cents.dtype == np.uint8 and crops.dtype == np.uint8
ptr = np.concatenate([[0], np.cumsum(counts)]); eptr = np.concatenate([[0], np.cumsum(ecounts)])
for g in range(41):
    e = edges[eptr[g]:eptr[g + 1]]
    assert e.size == 0 or e.max() < counts[g]
print("all checks passed;", "nodes per synthetic image: min", min(n_nodes), "median",
      int(np.median(n_nodes)), "max", max(n_nodes), "| collinear case nodes:", n_nodes[2])

# ---- 8-bit storage round trip
f = np.concatenate([rng.gamma(1.2, 0.5, (5000, 576)).astype(np.float32) - 0.375,
                    np.array([[-0.375] * 576, [16.0] * 576, [40.0] * 576], np.float32)])
q = s1.quantize(f)
assert q.dtype == np.uint8 and q[-3].max() == 0 and q[-2].min() == 255 and q[-1].min() == 255
b = s1.dequantize(q)
assert abs(b[-3]).max() - 0.375 < 1e-6 and abs(b[-2] - 16.0).max() < 1e-4
assert np.array_equal(s1.quantize(b), q), "quantize(dequantize(q)) must be stable"
step = np.abs(b[:-1] - f[:-1]).max()
assert step < 0.07, step   # worst case is G_MAX**2 / 255 at the cap
cnt = np.full(50, 100)
rep = s1.precision_report(f[:5000], cnt)
assert rep["graphs_used_for_knn"] == 50 and 0 <= rep["knn6_neighbour_overlap_mean"] <= 1
assert rep["fraction_clipped_above_cap"] == 0.0
print("8-bit storage checks passed; synthetic rmse/std", round(rep["rmse_over_std_mean"], 4),
      "max abs err", round(float(step), 4), "knn overlap", round(rep["knn6_neighbour_overlap_mean"], 3))
