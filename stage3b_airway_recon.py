"""STAGE 3b: MVS + Poisson + render of the AIRWAY-side sub-model (models/0).

The Stage-3 reconstruction kept sfm/ = trachea+cylinder side (17 imgs).
The airway side is sub-model models/0 (15 imgs, 891 sparse points, mean
reproj 2.45 px). Run MVS on it, render trajectory + dense cloud, and
check whether any 3D points are seen by the blade-segment images (f0, f14).
If blade points exist they are the independent in-model scale anchor.

Output: runs/gated2/2_v2/recon_airway/
  sfm/                    (copy of models/0)
  dense/                  (MVS workspace, fused.ply)
  dense/mesh_poisson.ply  (VIEWING ONLY)
  results.json            (registered list, blade-point count, dense N)
  trajectory_3view.png    (side/top/front render)
"""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
GATED = PROJECT / "runs/gated2/2_v2"
SRC_MODEL = GATED / "recon/sfm/models/0"
IMAGES = GATED / "recon/images"
OUT = GATED / "recon_airway"
COLMAP = Path("/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap")

BLADE_FRAME_INDICES = {0, 14}


def main():
    assert SRC_MODEL.exists(), f"missing {SRC_MODEL}"
    assert IMAGES.exists(), f"missing {IMAGES}"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sfm = OUT / "sfm"
    sfm.mkdir()
    # Copy reconstruction binaries
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = SRC_MODEL / fn
        if p.exists():
            shutil.copy(str(p), str(sfm / fn))
    print(f"copied airway sub-model -> {sfm}")

    # MVS
    dense = OUT / "dense"
    dense.mkdir()
    print("MVS image_undistorter ...")
    subprocess.run([str(COLMAP), "image_undistorter",
                     "--image_path", str(IMAGES),
                     "--input_path", str(sfm),
                     "--output_path", str(dense),
                     "--output_type", "COLMAP"], capture_output=True)
    print("MVS patch_match_stereo (geom) ...")
    subprocess.run([str(COLMAP), "patch_match_stereo",
                     "--workspace_path", str(dense),
                     "--workspace_format", "COLMAP",
                     "--PatchMatchStereo.geom_consistency", "true"],
                   capture_output=True)
    fused = dense / "fused.ply"
    print("MVS stereo_fusion ...")
    subprocess.run([str(COLMAP), "stereo_fusion",
                     "--workspace_path", str(dense),
                     "--workspace_format", "COLMAP",
                     "--input_type", "geometric",
                     "--output_path", str(fused)], capture_output=True)
    n_dense = 0
    if fused.exists():
        with open(fused, "rb") as fh:
            for _ in range(40):
                line = fh.readline().decode("ascii", errors="ignore")
                if line.startswith("element vertex"):
                    n_dense = int(line.split()[-1]); break
                if line.startswith("end_header"):
                    break
    print(f"MVS dense: {n_dense}")

    # Poisson view-only
    sub = """
import open3d as o3d, numpy as np, sys
fused, out = sys.argv[1], sys.argv[2]
pcd = o3d.io.read_point_cloud(fused)
if len(pcd.points) < 100: sys.exit(2)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
try: pcd.orient_normals_consistent_tangent_plane(k=20)
except Exception: pass
mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
density = np.asarray(density)
thresh = np.quantile(density, 0.05)
mesh.remove_vertices_by_index(np.where(density < thresh)[0].tolist())
o3d.io.write_triangle_mesh(out, mesh)
print(len(mesh.vertices))
"""
    out_mesh = dense / "mesh_poisson.ply"
    res = subprocess.run([sys.executable, "-c", sub, str(fused), str(out_mesh)],
                          capture_output=True, text=True)
    try:
        n_mesh_v = int(res.stdout.strip().splitlines()[-1])
    except Exception:
        n_mesh_v = 0
    print(f"Poisson verts (view only): {n_mesh_v}")

    # Inspect reconstruction: blade-point check + cameras
    sub = r"""
import pycolmap, json, sys, re, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
out = {"n_reg": int(rec.num_reg_images()),
        "n_sparse": int(rec.num_points3D())}
errs = [float(p.error) for p in rec.points3D.values()]
out["mean_reprojection_error_px"] = float(np.mean(errs)) if errs else None
# Image-by-image
imgs = []
blade_image_ids = set()
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    cfw = img.cam_from_world()
    M = np.array(cfw.matrix())
    R = M[:3,:3]; t = M[:3,3]
    C = (-R.T @ t).tolist()
    a = (R.T @ np.array([0,0,1])).tolist()
    imgs.append({"name": img.name, "frame_idx": fi,
                  "image_id": int(img.image_id), "C": C, "axis": a})
    if fi in {0, 14}:
        blade_image_ids.add(int(img.image_id))
imgs.sort(key=lambda r: r["frame_idx"])
out["registered"] = imgs
out["blade_image_ids"] = sorted(blade_image_ids)
# Blade points: 3D points with >=1 observation from blade images
blade_pt_ids = []
per_pt_blade_obs = {}
for pid, p in rec.points3D.items():
    obs = {int(el.image_id) for el in p.track.elements}
    n_blade = len(obs & blade_image_ids)
    if n_blade > 0:
        blade_pt_ids.append(int(pid))
        per_pt_blade_obs[int(pid)] = int(n_blade)
out["n_points_seen_by_blade"] = len(blade_pt_ids)
out["n_points_total"] = int(rec.num_points3D())
# Among blade points: count of those also seen by non-blade
non_blade_ids = set(int(i.image_id) for i in rec.images.values()) - blade_image_ids
n_both = 0; n_blade_only = 0
for pid in blade_pt_ids:
    obs = {int(el.image_id) for el in rec.points3D[pid].track.elements}
    if obs & non_blade_ids:
        n_both += 1
    else:
        n_blade_only += 1
out["n_blade_pts_also_seen_by_non_blade"] = n_both
out["n_blade_pts_only_in_blade"] = n_blade_only
# Camera params (verify pinned)
cam = next(iter(rec.cameras.values()))
out["camera_model"] = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
out["camera_params"] = list(cam.params)
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(sfm)],
                          capture_output=True, text=True)
    rec_info = json.loads(res.stdout.strip().splitlines()[-1])
    print(f"airway sub-model: reg={rec_info['n_reg']}  sparse={rec_info['n_sparse']}  "
          f"err={rec_info['mean_reprojection_error_px']:.2f} px")
    print(f"  blade image_ids: {rec_info['blade_image_ids']}")
    print(f"  blade points: {rec_info['n_points_seen_by_blade']} / "
          f"{rec_info['n_points_total']} sparse, {rec_info['n_blade_pts_also_seen_by_non_blade']} "
          f"also seen by non-blade")

    # Render 3-view
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(fused))
    pts = np.asarray(pcd.points)
    if len(pts) > 100000:
        ix = np.random.default_rng(0).choice(len(pts), 100000, replace=False)
        pts = pts[ix]
    reg = rec_info["registered"]
    centers = np.array([r["C"] for r in reg], dtype=float)
    fis = np.array([r["frame_idx"] for r in reg], dtype=int)
    is_blade_cam = np.isin(fis, list(BLADE_FRAME_INDICES))

    # Color points by camera trajectory PCA-1 parameter
    if len(centers) >= 2:
        ctr = centers.mean(0)
        _, _, Vt = np.linalg.svd(centers - ctr, full_matrices=False)
        e1 = Vt[0]
        order = np.argsort(fis)
        c_sorted = centers[order]; fi_sorted = fis[order]
        s_cam = (c_sorted - ctr) @ e1
        if s_cam[-1] < s_cam[0]:
            e1 = -e1; s_cam = -s_cam
        s_pts = (pts - ctr) @ e1
        s_norm = (s_pts - s_cam.min()) / max(s_cam.max() - s_cam.min(), 1e-9)
        s_norm = np.clip(s_norm, 0, 1)
    else:
        s_norm = np.zeros(len(pts))
        order = np.arange(len(centers))
        fi_sorted = fis

    # Get 3D positions of blade-tied sparse points for highlight overlay
    sub_pts = r"""
import pycolmap, json, sys, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
blade_image_ids = set(int(x) for x in json.loads(sys.argv[2]))
pts = []
for pid, p in rec.points3D.items():
    obs = {int(el.image_id) for el in p.track.elements}
    if obs & blade_image_ids:
        pts.append(list(p.xyz))
print(json.dumps(pts))
"""
    res = subprocess.run([sys.executable, "-c", sub_pts, str(sfm),
                           json.dumps(rec_info["blade_image_ids"])],
                          capture_output=True, text=True)
    blade_pts_3d = np.array(json.loads(res.stdout.strip().splitlines()[-1]))

    fig = plt.figure(figsize=(18, 6))
    titles = ["side (XZ)", "top (XY)", "front (YZ)"]
    proj = [("X", "Z", 0, 2), ("X", "Y", 0, 1), ("Y", "Z", 1, 2)]
    for k, (lx, ly, ix, iy) in enumerate(proj):
        ax = fig.add_subplot(1, 3, k + 1)
        if len(pts) > 0:
            ax.scatter(pts[:, ix], pts[:, iy], c=s_norm,
                       s=0.4, cmap="viridis", alpha=0.55, marker='.',
                       linewidths=0)
        if len(blade_pts_3d) > 0:
            ax.scatter(blade_pts_3d[:, ix], blade_pts_3d[:, iy], c="red",
                       s=20, edgecolors="black", linewidths=0.4, marker="o",
                       label=f"blade-tracked sparse ({len(blade_pts_3d)})",
                       zorder=4)
        # Trajectory line in chronological order
        ax.plot(centers[order][:, ix], centers[order][:, iy], "-",
                c="white", lw=2.0)
        ax.plot(centers[order][:, ix], centers[order][:, iy], "-",
                c="orange", lw=1.0)
        # Blade cam markers in chrono order
        ax.scatter(centers[order][is_blade_cam[order], ix],
                   centers[order][is_blade_cam[order], iy],
                   c="red", s=70, marker="^", edgecolors="black",
                   linewidths=0.6, label="blade cams (f0,f14)")
        ax.scatter(centers[order][~is_blade_cam[order], ix],
                   centers[order][~is_blade_cam[order], iy],
                   c="black", s=14, marker="o", edgecolors="white",
                   linewidths=0.3, label="non-blade cams")
        ax.set_title(f"{titles[k]}  ({lx},{ly})")
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"gated2 / 2_v2 AIRWAY-side sub-model (models/0)  |  "
        f"reg={rec_info['n_reg']}/15  sparse={rec_info['n_sparse']}  "
        f"dense={len(np.asarray(pcd.points))}  mean reproj={rec_info['mean_reprojection_error_px']:.2f} px\n"
        f"red ▲ = blade cameras (f0, f14);  red ● = sparse points tracked into a blade frame  "
        f"(n={rec_info['n_points_seen_by_blade']}/{rec_info['n_sparse']})",
        fontsize=11)
    fig.tight_layout()
    out_png = OUT / "trajectory_3view.png"
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved 3-view -> {out_png}")

    diag = {
        "source_submodel": str(SRC_MODEL),
        "n_registered": rec_info["n_reg"],
        "n_sparse_points": rec_info["n_sparse"],
        "n_dense_points": int(n_dense),
        "n_poisson_verts_VIEW_ONLY": int(n_mesh_v),
        "mean_reprojection_error_px": rec_info["mean_reprojection_error_px"],
        "registered_frame_indices":
            sorted(r["frame_idx"] for r in rec_info["registered"]),
        "blade_image_ids": rec_info["blade_image_ids"],
        "n_points_seen_by_blade": rec_info["n_points_seen_by_blade"],
        "n_blade_pts_also_seen_by_non_blade":
            rec_info["n_blade_pts_also_seen_by_non_blade"],
        "n_blade_pts_only_in_blade": rec_info["n_blade_pts_only_in_blade"],
        "camera_model": rec_info["camera_model"],
        "camera_params": rec_info["camera_params"],
        "trajectory_3view": str(out_png),
        "mesh_poisson_ply_VIEW_ONLY": str(out_mesh),
    }
    (OUT / "results.json").write_text(json.dumps(diag, indent=2))
    print(f"saved {OUT/'results.json'}")


if __name__ == "__main__":
    main()
