"""Render the corpus + SSM to .ply files for offline 3D viewing.

Read-only against the corpus. Produces files designed to be opened
in MeshLab (or any .ply-compatible viewer) — no headless GL or
rendering is performed on the server. Drag the .ply files into the
viewer; they all share the template's coordinate frame so multiple
files can be overlaid.

Outputs under <out_dir>/:
  template_surface.ply              template's airway surface
  template_centerline.ply           template centerline as colored
                                    points (see centerline_legend.txt)
  centerline_legend.txt
  ssm/mean_surface.ply              SSM mean shape (surface variant)
  ssm/mode_<k>_minus2sigma.ply      first --n_modes modes at -2 sigma
  ssm/mode_<k>_plus2sigma.ply       ... and at +2 sigma
  registration/<sid>_subject_in_template_frame.ply
                                    subject's mesh Procrustes-aligned
                                    into the template's frame
  registration/<sid>_warped_template.ply
                                    template warped to that subject
  registration/<sid>_residuals_colored.ply
                                    same warped template, colored by
                                    nearest-neighbor distance to
                                    subject (green=0mm, red>=20mm)

Usage:
  python visualize_corpus.py --corpus atm22_corpus.h5 --out_dir viz
  python visualize_corpus.py --corpus atm22_corpus.h5 \\
      --n_modes 4 --n_check_subjects 3
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from atm22_corpus import Corpus
from build_correspondence import (
    _gather_subject_points, _landmark_rows,
    procrustes_similarity, apply_similarity,
)

LANDMARK_COLORS = {
    "TRACHEA": [255, 60, 60, 255],     # red
    "CARINA":  [255, 165, 0, 255],     # orange
    "RMB_BIF": [60, 200, 80, 255],     # green
    "LMB_BIF": [80, 100, 230, 255],    # blue
    "BI_BIF":  [200, 80, 200, 255],    # magenta
}


def _save_mesh(verts, faces, path, vertex_colors=None):
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                        vertex_colors=vertex_colors)
    m.export(path)


def _save_point_cloud(points, colors, path):
    trimesh.PointCloud(vertices=points, colors=colors).export(path)


def _heatmap_rgba(values, vmin=0.0, vmax=20.0):
    """Map a scalar field to a (N, 4) RGBA array: green at vmin, red at vmax."""
    v = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgba = np.zeros((len(v), 4), dtype=np.uint8)
    rgba[:, 0] = (255 * v).astype(np.uint8)
    rgba[:, 1] = (255 * (1.0 - v)).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--n_modes", type=int, default=4)
    ap.add_argument("--n_check_subjects", type=int, default=3)
    ap.add_argument("--max_residual_color_mm", type=float, default=20.0)
    args = ap.parse_args()

    corpus = Corpus(args.corpus)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "ssm"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "registration"), exist_ok=True)

    # ---- Template ----
    template_id = corpus.template_subject_id
    if template_id is None:
        raise SystemExit("No template_subject_id in corpus.")
    print(f"Template: {template_id}")
    template = corpus.load(template_id)

    _save_mesh(template.surface_verts, template.surface_faces,
               os.path.join(args.out_dir, "template_surface.ply"))

    pts = template.centerline_nodes
    colors = np.full((len(pts), 4), [200, 200, 200, 255], dtype=np.uint8)
    colors[template.is_bifurcation] = [80, 200, 255, 255]
    if template.label_node_idx is not None:
        for idx, name in zip(template.label_node_idx, template.label_name):
            if name in LANDMARK_COLORS:
                colors[int(idx)] = LANDMARK_COLORS[name]
    _save_point_cloud(pts, colors,
                      os.path.join(args.out_dir, "template_centerline.ply"))

    with open(os.path.join(args.out_dir, "centerline_legend.txt"), "w") as f:
        f.write("template_centerline.ply coloring:\n")
        f.write("  light grey  - shaft (degree 2)\n")
        f.write("  cyan        - unnamed bifurcation (degree >= 3)\n")
        for name, col in LANDMARK_COLORS.items():
            f.write(f"  {name:<10s}- RGB({col[0]},{col[1]},{col[2]})\n")

    # ---- SSM modes ----
    ssm = corpus.load_ssm(surface=True)
    faces = corpus.surface_template_faces
    if ssm is None or faces is None:
        print("No surface SSM found, skipping mean/modes.")
    else:
        _save_mesh(ssm["mean_shape"], faces,
                   os.path.join(args.out_dir, "ssm", "mean_surface.ply"))
        n_modes = min(args.n_modes, len(ssm["modes"]))
        print(f"Rendering first {n_modes} SSM modes at +-2 sigma")
        for k in range(n_modes):
            sigma = float(np.sqrt(ssm["eigenvalues"][k]))
            for sign, lbl in ((-2.0, "minus2sigma"), (2.0, "plus2sigma")):
                shape = ssm["mean_shape"] + sign * sigma * ssm["modes"][k]
                _save_mesh(shape, faces,
                           os.path.join(args.out_dir, "ssm",
                                        f"mode_{k}_{lbl}.ply"))

    # ---- Registration check ----
    rng = np.random.default_rng(0)
    candidates = [s for s in corpus.subject_ids if s != template_id]
    n_check = min(args.n_check_subjects, len(candidates))
    if n_check > 0 and faces is not None:
        check_sids = rng.choice(candidates, size=n_check, replace=False)
        print(f"Rendering registration overlay for {n_check} subjects: "
              f"{list(check_sids)}")
        K = sum(1 for s in corpus.point_labels if s.startswith("T_C["))
        landmark_rows = _landmark_rows(K)
        template_landmarks = template.corresponded_points[landmark_rows] \
            .astype(np.float64)

        with h5py.File(args.corpus, "r") as h5:
            for sid in check_sids:
                s = corpus.load(sid)
                if s.corresponded_surface_verts is None:
                    continue
                subj_pts_raw = _gather_subject_points(
                    h5[f"subjects/{sid}"], K)
                if subj_pts_raw is None:
                    continue
                R, scale, src_mean, tgt_mean = procrustes_similarity(
                    subj_pts_raw[landmark_rows].astype(np.float64),
                    template_landmarks)
                subj_in_tpl = apply_similarity(
                    s.surface_verts.astype(np.float64),
                    R, scale, src_mean, tgt_mean)

                _save_mesh(subj_in_tpl, s.surface_faces,
                           os.path.join(args.out_dir, "registration",
                                        f"{sid}_subject_in_template_frame.ply"))
                _save_mesh(s.corresponded_surface_verts, faces,
                           os.path.join(args.out_dir, "registration",
                                        f"{sid}_warped_template.ply"))

                tree = cKDTree(subj_in_tpl)
                dists, _ = tree.query(s.corresponded_surface_verts)
                _save_mesh(
                    s.corresponded_surface_verts, faces,
                    os.path.join(args.out_dir, "registration",
                                 f"{sid}_residuals_colored.ply"),
                    vertex_colors=_heatmap_rgba(
                        dists, vmax=args.max_residual_color_mm))

    print(f"\nDone. Output in {args.out_dir}/")
    print("Open in MeshLab: File > Import Mesh, multi-select to overlay.")


if __name__ == "__main__":
    main()
