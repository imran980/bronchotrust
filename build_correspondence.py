"""Build inter-subject correspondence on the ATM22 corpus.

For each subject, produce a fixed-row-count (N, 3) point matrix where
row k represents the same anatomical position across every subject.
This is the matrix the SSM (step 3) feeds to PCA.

Procedure per subject:
  1. Look up the 5 named bifurcation landmarks (TRACHEA, CARINA,
     RMB_BIF, LMB_BIF, BI_BIF).
  2. Walk the unique tree path between each adjacent landmark pair
     (T-C, C-RMB, C-LMB, RMB-BI). Resample each path to K equispaced
     arc-length samples (excluding the two endpoint landmarks).
  3. Similarity-align (rotation + translation + uniform scale) the
     subject's 5 landmarks onto the template's via the Umeyama-style
     Procrustes solution. Apply that transform to all sampled points.

Canonical row ordering (same for every subject; N = 5 + 4*K):
  0                TRACHEA
  1 .. K           T_C[0..K-1]
  K+1              CARINA
  K+2 .. 2K+1      C_RMB[0..K-1]
  2K+2             RMB_BIF
  2K+3 .. 3K+2     RMB_BI[0..K-1]
  3K+3             BI_BIF
  3K+4 .. 4K+3     C_LMB[0..K-1]
  4K+4             LMB_BIF

Outputs written into atm22_corpus.h5:
  /subjects/<sid>/corresponded_points   (N, 3) float32, mm
  /subjects/<sid>/correspondence_residual_mm  scalar = mean landmark
                                              error after Procrustes
  /metadata/correspondence_template     subject ID used as reference
  /metadata/point_labels                (N,) string  per-row labels

Usage:
  python build_correspondence.py --corpus atm22_corpus.h5 \\
                                 --samples_per_branch 20
"""
from __future__ import annotations

import argparse

import h5py
import networkx as nx
import numpy as np
from tqdm import tqdm

LABEL_DTYPE = h5py.string_dtype()
EDGES = [
    ("TRACHEA", "CARINA"),
    ("CARINA", "RMB_BIF"),
    ("RMB_BIF", "BI_BIF"),
    ("CARINA", "LMB_BIF"),
]
LANDMARKS = ["TRACHEA", "CARINA", "RMB_BIF", "BI_BIF", "LMB_BIF"]


def _build_graph(n_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(a), int(b)) for a, b in edges)
    return G


def _resample_polyline(points, n_samples):
    """Resample (M, 3) polyline to (n_samples, 3) by arc length,
    excluding both endpoints (caller emits those as named landmarks)."""
    if len(points) < 2 or n_samples <= 0:
        return np.zeros((max(n_samples, 0), 3), dtype=points.dtype)
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-9:
        return np.tile(points[0], (n_samples, 1))
    targets = np.linspace(0.0, total, n_samples + 2)[1:-1]
    out = np.zeros((n_samples, 3), dtype=points.dtype)
    for i, target in enumerate(targets):
        idx = int(np.searchsorted(cum, target) - 1)
        idx = max(0, min(idx, len(seg_len) - 1))
        t = (target - cum[idx]) / max(seg_len[idx], 1e-9)
        out[i] = points[idx] + t * seg[idx]
    return out


def _gather_subject_points(g, K):
    """Stack landmarks + along-branch samples in canonical row order.
    Returns (N, 3) float32 array, or None if any landmark is missing."""
    nodes = g["centerline_nodes"][:]
    edges = g["centerline_edges"][:]
    label_idx = g["label_node_idx"][:]
    label_name = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in g["label_name"][:]]
    name_to_idx = {n: int(i) for i, n in zip(label_idx, label_name)}
    if not set(LANDMARKS).issubset(name_to_idx):
        return None

    G = _build_graph(len(nodes), edges)

    def along(src_name, dst_name):
        path = nx.shortest_path(G, source=name_to_idx[src_name],
                                target=name_to_idx[dst_name])
        return _resample_polyline(nodes[path], K)

    parts = [
        nodes[name_to_idx["TRACHEA"]][None, :],
        along("TRACHEA", "CARINA"),
        nodes[name_to_idx["CARINA"]][None, :],
        along("CARINA", "RMB_BIF"),
        nodes[name_to_idx["RMB_BIF"]][None, :],
        along("RMB_BIF", "BI_BIF"),
        nodes[name_to_idx["BI_BIF"]][None, :],
        along("CARINA", "LMB_BIF"),
        nodes[name_to_idx["LMB_BIF"]][None, :],
    ]
    return np.vstack(parts).astype(np.float32)


def _row_labels(K):
    out = ["TRACHEA"]
    out += [f"T_C[{i}]" for i in range(K)]
    out.append("CARINA")
    out += [f"C_RMB[{i}]" for i in range(K)]
    out.append("RMB_BIF")
    out += [f"RMB_BI[{i}]" for i in range(K)]
    out.append("BI_BIF")
    out += [f"C_LMB[{i}]" for i in range(K)]
    out.append("LMB_BIF")
    return out


def _landmark_rows(K):
    return [0, 1 + K, 2 + 2 * K, 3 + 3 * K, 4 + 4 * K]


def procrustes_similarity(source, target):
    """Umeyama similarity: returns (R, scale, src_mean, tgt_mean) so
    that target ~= scale * (source - src_mean) @ R.T + tgt_mean."""
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    s = source - src_mean
    t = target - tgt_mean
    H = s.T @ t
    U, Sigma, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    denom = float((s ** 2).sum())
    scale = (np.array([1.0, 1.0, d]) * Sigma).sum() / denom if denom > 0 else 1.0
    return R, float(scale), src_mean, tgt_mean


def apply_similarity(points, R, scale, src_mean, tgt_mean):
    return scale * ((points - src_mean) @ R.T) + tgt_mean


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--samples_per_branch", type=int, default=20)
    args = ap.parse_args()

    K = args.samples_per_branch
    row_labels = _row_labels(K)
    landmark_rows = _landmark_rows(K)
    N = len(row_labels)
    print(f"Correspondence: {N} points per subject "
          f"({K} samples × 4 branches + 5 landmarks)")

    with h5py.File(args.corpus, "a") as h5:
        if "template_subject_id" not in h5["metadata"]:
            raise SystemExit("Run select_template.py first.")
        tmpl_raw = h5["metadata/template_subject_id"][()]
        template_id = tmpl_raw.decode() if isinstance(tmpl_raw, bytes) \
            else str(tmpl_raw)
        print(f"Template: {template_id}")

        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        template_pts = _gather_subject_points(
            h5[f"subjects/{template_id}"], K)
        if template_pts is None:
            raise SystemExit("Template subject lacks required landmarks.")
        template_landmarks = template_pts[landmark_rows]

        skipped, residuals = [], []
        for sid in tqdm(ids, desc="corresponding"):
            g = h5[f"subjects/{sid}"]
            pts = _gather_subject_points(g, K)
            if pts is None:
                skipped.append(sid)
                continue
            R, scale, src_mean, tgt_mean = procrustes_similarity(
                pts[landmark_rows], template_landmarks)
            aligned = apply_similarity(pts, R, scale, src_mean, tgt_mean)
            resid = float(np.linalg.norm(
                aligned[landmark_rows] - template_landmarks,
                axis=1).mean())
            residuals.append(resid)
            for k in ("corresponded_points", "correspondence_residual_mm"):
                if k in g:
                    del g[k]
            g.create_dataset("corresponded_points",
                             data=aligned.astype(np.float32),
                             compression="gzip", compression_opts=6)
            g.create_dataset("correspondence_residual_mm",
                             data=np.float32(resid))

        meta = h5["metadata"]
        for k in ("correspondence_template", "point_labels"):
            if k in meta:
                del meta[k]
        meta.create_dataset("correspondence_template",
                            data=np.array(template_id, dtype=LABEL_DTYPE))
        meta.create_dataset("point_labels",
                            data=np.array(row_labels, dtype=LABEL_DTYPE))

    res = np.array(residuals)
    print(f"\nCorresponded {len(residuals)} subjects. Skipped {len(skipped)}.")
    if len(res):
        print(f"Landmark residual (mm) after similarity alignment: "
              f"mean={res.mean():.2f}  median={np.median(res):.2f}  "
              f"max={res.max():.2f}")
    for sid in skipped:
        print(f"  skipped: {sid}")


if __name__ == "__main__":
    main()
