"""Non-rigid registration of every subject's airway surface to the template.

Builds on build_correspondence.py: every subject already has 85
Procrustes-aligned centerline correspondences stored as
corresponded_points. This script:
  1. Recomputes the per-subject similarity transform (rotation +
     translation + uniform scale) from the 5 named landmarks and
     applies it to the subject's full surface, putting subject and
     template into the same coordinate frame.
  2. Fits a regularized 3D TPS warp using all 85 corresponded
     centerline points as anchors (source = template's points,
     target = subject's points, both already in template frame).
  3. Applies that warp to the template's surface vertices, producing
     a (V_template, 3) deformed mesh whose vertex k is the same
     anatomical surface point on every subject.
  4. Refines with ICP: sample template vertices, find nearest
     pre-aligned subject vertices, add as soft anchors, refit TPS.
     Centerline anchors stay near-hard; ICP anchors are smoother.

This is the variant of TPS that doesn't explode. The naive version —
TPS with 5 landmark anchors in different coordinate frames, no
pre-alignment — extrapolates by kilometers in the distal lobes
because most surface vertices lie far outside the anchor convex hull
and there's no rigid prior. Pre-alignment + dense centerline anchors
fixes both.

Inputs from the corpus:
  /subjects/*/surface_verts, surface_faces
  /subjects/*/corresponded_points (from build_correspondence.py)
  /subjects/*/label_node_idx, label_name (from label_bifurcations.py)
  /metadata/template_subject_id
  /metadata/point_labels (to recover K = samples_per_branch)

Outputs written into the corpus:
  /subjects/<sid>/corresponded_surface_verts  (V_template, 3) float32
                                              in the template's frame
  /subjects/<sid>/surface_registration_residual_mm  scalar mean
                                                    nearest-neighbor
                                                    distance post-ICP
  /metadata/surface_template_faces            (F_template, 3) int32
  /metadata/surface_template_n_verts          int

Usage:
  python register_surface.py --corpus atm22_corpus.h5 \\
      --icp_iters 3 --icp_samples 500 --reg 1.0
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from build_correspondence import (
    _gather_subject_points, _landmark_rows,
    procrustes_similarity, apply_similarity,
)


def _infer_K_from_row_labels(point_labels):
    """Recover samples_per_branch from the stored row labels so we
    don't have to pass it as a flag."""
    K = sum(1 for s in point_labels if s.startswith("T_C["))
    if K == 0:
        raise SystemExit("Could not infer samples_per_branch from "
                         "metadata/point_labels — has build_"
                         "correspondence.py been run?")
    return K


def fit_tps_3d(source, target, regularization):
    """Solve 3D TPS warp source -> target. Kernel phi(r) = |r|
    (the 3D biharmonic spline). Regularization can be a scalar or an
    (N,) array of per-anchor lambdas — lets us mix hard centerline
    anchors with soft ICP-derived anchors."""
    N = len(source)
    K = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    if np.isscalar(regularization):
        if regularization > 0:
            K = K + regularization * np.eye(N)
    else:
        K = K + np.diag(regularization)
    P = np.hstack([source, np.ones((N, 1))])
    L = np.zeros((N + 4, N + 4))
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.T
    Y = np.zeros((N + 4, 3))
    Y[:N] = target
    sol = np.linalg.solve(L, Y)
    return {
        "control": source,
        "weights": sol[:N],
        "affine": sol[N:N + 3],
        "translation": sol[N + 3],
    }


def apply_tps_3d(points, warp, chunk=4096):
    control = warp["control"]
    weights = warp["weights"]
    A = warp["affine"]
    t = warp["translation"]
    out = np.empty_like(points)
    for i in range(0, len(points), chunk):
        pts = points[i:i + chunk]
        D = np.linalg.norm(pts[:, None, :] - control[None, :, :], axis=2)
        out[i:i + chunk] = D @ weights + pts @ A + t
    return out


def register_one(template_verts, template_pts,
                 subject_verts, subject_pts,
                 icp_iters, icp_samples, reg, icp_max_dist):
    """All inputs are in the template's frame.

    template_verts: (V, 3) template surface mesh vertices
    template_pts:   (N, 3) template centerline anchors (== template's
                    own corresponded_points)
    subject_verts:  (Vs, 3) subject surface, Procrustes-aligned to
                    template frame
    subject_pts:    (N, 3) subject centerline anchors in template
                    frame (the stored corresponded_points)
    """
    warp = fit_tps_3d(template_pts, subject_pts, regularization=1e-3)
    warped = apply_tps_3d(template_verts, warp)

    subject_tree = cKDTree(subject_verts)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(template_verts),
                            size=min(icp_samples, len(template_verts)),
                            replace=False)

    prev_err = float("inf")
    last_err = float("inf")
    for _ in range(icp_iters):
        dists, nearest = subject_tree.query(warped[sample_idx])
        valid = dists < icp_max_dist
        if not valid.any():
            break
        src = np.vstack([template_pts,
                         template_verts[sample_idx[valid]]])
        dst = np.vstack([subject_pts,
                         subject_verts[nearest[valid]]])
        per_anchor_reg = np.concatenate([
            np.full(len(template_pts), 1e-3),
            np.full(int(valid.sum()), reg),
        ])
        warp = fit_tps_3d(src, dst, regularization=per_anchor_reg)
        warped = apply_tps_3d(template_verts, warp)
        last_err = float(subject_tree.query(warped[sample_idx])[0].mean())
        if last_err > prev_err * 0.999:
            break
        prev_err = last_err

    if last_err == float("inf"):
        last_err = float(subject_tree.query(warped[sample_idx])[0].mean())
    return warped.astype(np.float32), last_err


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--icp_iters", type=int, default=3)
    ap.add_argument("--icp_samples", type=int, default=500)
    ap.add_argument("--reg", type=float, default=1.0,
                    help="Regularization (lambda) on ICP-derived "
                         "anchors. Larger = smoother warp.")
    ap.add_argument("--icp_max_dist", type=float, default=20.0,
                    help="Reject ICP correspondences with nearest-"
                         "neighbor distance above this (mm).")
    args = ap.parse_args()

    with h5py.File(args.corpus, "a") as h5:
        if "template_subject_id" not in h5["metadata"]:
            raise SystemExit("Run select_template.py first.")
        if "point_labels" not in h5["metadata"]:
            raise SystemExit("Run build_correspondence.py first.")
        point_labels = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in h5["metadata/point_labels"][:]]
        K = _infer_K_from_row_labels(point_labels)
        landmark_rows = _landmark_rows(K)

        tmpl_raw = h5["metadata/template_subject_id"][()]
        template_id = (tmpl_raw.decode() if isinstance(tmpl_raw, bytes)
                       else str(tmpl_raw))
        tg = h5[f"subjects/{template_id}"]
        template_verts = tg["surface_verts"][:].astype(np.float64)
        template_faces = tg["surface_faces"][:].astype(np.int32)
        if "corresponded_points" not in tg:
            raise SystemExit("Template subject has no corresponded_points.")
        template_pts = tg["corresponded_points"][:].astype(np.float64)
        template_landmarks = template_pts[landmark_rows]
        V = len(template_verts)
        print(f"Template: {template_id} ({V} verts, {len(template_faces)} faces) "
              f"with {len(template_pts)} centerline anchors")

        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]

        residuals, skipped = [], []
        for sid in tqdm(ids, desc="registering"):
            g = h5[f"subjects/{sid}"]
            if "corresponded_points" not in g:
                skipped.append(sid)
                continue
            subject_verts_raw = g["surface_verts"][:].astype(np.float64)
            subject_pts_aligned = g["corresponded_points"][:].astype(np.float64)

            # Recover the same Procrustes transform build_correspondence
            # used, and apply it to the subject's full surface so both
            # surfaces share the template frame.
            subj_pts_raw = _gather_subject_points(g, K)
            if subj_pts_raw is None:
                skipped.append(sid)
                continue
            subj_pts_raw = subj_pts_raw.astype(np.float64)
            R, scale, src_mean, tgt_mean = procrustes_similarity(
                subj_pts_raw[landmark_rows], template_landmarks)
            subject_verts = apply_similarity(
                subject_verts_raw, R, scale, src_mean, tgt_mean)

            warped, err = register_one(
                template_verts, template_pts,
                subject_verts, subject_pts_aligned,
                icp_iters=args.icp_iters,
                icp_samples=args.icp_samples,
                reg=args.reg,
                icp_max_dist=args.icp_max_dist)
            residuals.append(err)
            for k in ("corresponded_surface_verts",
                      "surface_registration_residual_mm"):
                if k in g:
                    del g[k]
            g.create_dataset("corresponded_surface_verts", data=warped,
                             compression="gzip", compression_opts=6)
            g.create_dataset("surface_registration_residual_mm",
                             data=np.float32(err))

        meta = h5["metadata"]
        for k in ("surface_template_faces", "surface_template_n_verts"):
            if k in meta:
                del meta[k]
        meta.create_dataset("surface_template_faces", data=template_faces,
                            compression="gzip", compression_opts=6)
        meta.attrs["surface_template_n_verts"] = V

    r = np.array(residuals)
    print(f"\nRegistered {len(r)} subjects. Skipped {len(skipped)}.")
    if len(r):
        print(f"Surface registration residual (mm): "
              f"mean={r.mean():.2f}  median={np.median(r):.2f}  "
              f"max={r.max():.2f}")
    for sid in skipped:
        print(f"  skipped: {sid}")


if __name__ == "__main__":
    main()
