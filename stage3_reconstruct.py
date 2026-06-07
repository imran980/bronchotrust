"""STAGE 3: pinned-intrinsics reconstruction of the gated2 30-frame set.

Pipeline:
  1. Copy 30 Stage-1b PNGs from runs/gated2/2_v2/stage1_frames/ to images/,
     apply content_mask.png bezel (preserves Stage-0 dark-drop logic).
  2. HLoc SuperPoint+LightGlue, exhaustive matcher.
  3. COLMAP incremental mapper: init_min_tri_angle=4, min_num_matches=8,
     pinned OPENCV intrinsics (NEW from Stage 2), refine_focal=False,
     refine_extra_params=False, refine_principal_point=False.
  4. Pass 1 default-init; pass 2 forced glottic init pair (highest-inlier
     pair among segment="glottis" frames f15, f35, f76).
  5. multiple_models=True; report ALL sub-models found (connectivity check).
  6. MVS dense (geom_consistency) on the largest sub-model.
  7. Poisson surface for VIEWING ONLY (not measurement).
  8. Connectivity + cylinder analysis + trajectory continuity report.
  9. 3-view render of dense cloud + trajectory.

Outputs (runs/gated2/2_v2/recon/):
  images/, masks/, sfm/, sfm/models/, dense/, sfm_text/
  recon_results.json    : standard + scale-anchor + linkage health
  trajectory_3view.png  : side / top / front render
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
GATED = PROJECT / "runs/gated2/2_v2"
RECON = GATED / "recon"
COLMAP = Path("/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap")

VID = "2_v2"
SEG_TUBE = "transition_tube"
INSIDE_TUBE_THRESH = 3470  # frame_idx >= this => inside-bore camera
LINKAGE_LO, LINKAGE_HI = 3300, 3469


def load_inputs():
    sel = json.loads((GATED / "stage1_selection.json").read_text())["selection"]
    intr = json.loads((GATED / "intrinsics_pinned.json").read_text())
    mask = cv2.imread(str(GATED / "content_mask.png"), cv2.IMREAD_GRAYSCALE)
    return sel, intr, mask


def preprocess(out_base: Path, selection, mask):
    img_dir = out_base / "images"
    msk_dir = out_base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    bezel = (mask > 0)
    extracted = []
    for s in selection:
        fi = s["frame_idx"]; seg = s["segment"]
        src = GATED / "stage1_frames" / f"seg_{seg}_f{fi:05d}.png"
        if not src.exists():
            print(f"  MISSING source {src.name}")
            continue
        img = cv2.imread(str(src))
        img[~bezel] = 0
        # Canonical filename: 2_v2_f<fi:06d>.png (so frame-idx regex works)
        name = f"{VID}_f{fi:06d}.png"
        cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (bezel.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        extracted.append({"frame_idx": fi, "segment": seg, "filename": name})
    print(f"  preprocessed {len(extracted)} / {len(selection)} frames")
    return extracted


def cam_params_str(intr):
    params = [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
              intr["k1"], intr["k2"], 0.0, 0.0]
    return ",".join(f"{p:.10g}" for p in params), params


def run_hloc(out_base, image_list):
    from hloc import extract_features, match_features, pairs_from_exhaustive
    images_dir = out_base / "images"
    pairs_path = out_base / "pairs-exhaustive.txt"
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    print("  HLoc SuperPoint ...")
    feats = extract_features.main(
        feature_conf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    print("  HLoc LightGlue ...")
    matches = match_features.main(
        matcher_conf, pairs_path, feature_conf["output"], out_base)
    return feats, matches, pairs_path


def find_glottic_init(db_path: Path, glottic_files, min_gap: int = 12):
    import sqlite3

    def fi(n):
        m = re.search(r"f(\d+)\.png$", n)
        return int(m.group(1)) if m else -1

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    id_name = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")
    pairs = cur.fetchall()
    conn.close()
    best_baseline = None; best_anywhere = None
    for pair_id, rows in pairs:
        i1 = pair_id // 2147483647
        i2 = pair_id - i1 * 2147483647
        n1 = id_name.get(i1); n2 = id_name.get(i2)
        if n1 not in glottic_files or n2 not in glottic_files:
            continue
        gap = abs(fi(n1) - fi(n2))
        if best_anywhere is None or rows > best_anywhere[0]:
            best_anywhere = (rows, i1, i2, n1, n2, gap)
        if gap >= min_gap and (best_baseline is None or rows > best_baseline[0]):
            best_baseline = (rows, i1, i2, n1, n2, gap)
    pick = best_baseline if best_baseline is not None else best_anywhere
    return pick[:5] if pick else None


def reconstruct(out_base, image_list, intr, feats, matches, sfm_pairs,
                glottic_files):
    from hloc import reconstruction as hloc_recon
    sfm_dir = out_base / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_base / "images"
    cam_params, _ = cam_params_str(intr)
    image_options = {"camera_model": "OPENCV", "camera_params": cam_params}
    mapper_options = {
        "min_num_matches": 8,
        "ba_local_max_num_iterations": 50,
        "ba_global_max_num_iterations": 100,
        "multiple_models": True, "max_num_models": 50, "min_model_size": 3,
        "ba_refine_focal_length": False,
        "ba_refine_extra_params": False,
        "ba_refine_principal_point": False,
        "mapper": {
            "init_min_tri_angle": 4.0,
            "init_max_error": 8.0,
            "init_min_num_inliers": 15,
            "init_max_forward_motion": 0.99,
            "abs_pose_max_error": 20.0,
            "abs_pose_min_num_inliers": 15,
            "abs_pose_min_inlier_ratio": 0.1,
            "filter_max_reproj_error": 8.0,
            "filter_min_tri_angle": 1.0,
        },
    }
    print("  Pass 1 (default init, pinned) ...")
    model = hloc_recon.main(
        sfm_dir, images_dir, sfm_pairs, feats, matches,
        image_list=image_list,
        camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
        image_options=image_options,
        verbose=False, mapper_options=mapper_options)
    n_reg_pre = model.num_reg_images() if model else 0
    n_pts_pre = model.num_points3D() if model else 0
    print(f"  Pass 1 registered: {n_reg_pre}  sparse: {n_pts_pre}")

    init = find_glottic_init(sfm_dir / "database.db", glottic_files)
    if init is None:
        print("  no glottic init found; keeping pass 1")
        return n_reg_pre, n_pts_pre

    n_inl, i1, i2, n1, n2 = init
    print(f"  Pass 2 forced init: {n1} <-> {n2}  inliers={n_inl}")
    pass1_dir = out_base / "sfm_pass1"
    if pass1_dir.exists():
        shutil.rmtree(pass1_dir)
    pass1_dir.mkdir()
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        p = sfm_dir / fn
        if p.exists():
            shutil.copy(str(p), str(pass1_dir / fn))
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        p = sfm_dir / fn
        if p.exists():
            p.unlink()
    models_dir = sfm_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)

    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 8
    opts.multiple_models = True
    opts.max_num_models = 50
    opts.min_model_size = 3
    opts.init_image_id1 = i1
    opts.init_image_id2 = i2
    opts.ba_refine_focal_length = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_principal_point = False
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.init_max_error = 8.0
    opts.mapper.init_min_num_inliers = 15
    opts.mapper.init_max_forward_motion = 0.99
    opts.mapper.abs_pose_max_error = 20.0
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.mapper.abs_pose_min_inlier_ratio = 0.1
    opts.mapper.filter_max_reproj_error = 8.0
    opts.mapper.filter_min_tri_angle = 1.0
    try:
        recons = pycolmap.incremental_mapping(
            database_path=str(sfm_dir / "database.db"),
            image_path=str(images_dir),
            output_path=str(models_dir),
            options=opts)
    except Exception as e:
        print(f"  Pass 2 error: {e}; using pass 1")
        for fn in ["cameras.bin", "images.bin", "points3D.bin",
                    "frames.bin", "rigs.bin"]:
            src = pass1_dir / fn
            if src.exists():
                shutil.copy(str(src), str(sfm_dir / fn))
        return n_reg_pre, n_pts_pre
    if not recons:
        print("  Pass 2 produced no models; restoring pass 1")
        for fn in ["cameras.bin", "images.bin", "points3D.bin",
                    "frames.bin", "rigs.bin"]:
            src = pass1_dir / fn
            if src.exists():
                shutil.copy(str(src), str(sfm_dir / fn))
        return n_reg_pre, n_pts_pre
    best_id = max(recons.keys(), key=lambda k: recons[k].num_reg_images())
    best = recons[best_id]
    best.write_binary(str(sfm_dir))
    n_reg_p2 = best.num_reg_images()
    n_pts_p2 = best.num_points3D()
    if n_reg_p2 >= n_reg_pre:
        print(f"  Pass 2 registered: {n_reg_p2}  sparse: {n_pts_p2}  -> using")
        return n_reg_p2, n_pts_p2
    print(f"  Pass 2 ({n_reg_p2}) < pass 1 ({n_reg_pre}); restoring pass 1")
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        src = pass1_dir / fn
        if src.exists():
            shutil.copy(str(src), str(sfm_dir / fn))
    return n_reg_pre, n_pts_pre


def run_mvs(out_base: Path):
    sfm = out_base / "sfm"
    dense = out_base / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    dense.mkdir(parents=True)
    print("  MVS image_undistorter ...")
    subprocess.run([str(COLMAP), "image_undistorter",
                     "--image_path", str(out_base / "images"),
                     "--input_path", str(sfm),
                     "--output_path", str(dense),
                     "--output_type", "COLMAP"], capture_output=True)
    print("  MVS patch_match_stereo (geom) ...")
    subprocess.run([str(COLMAP), "patch_match_stereo",
                     "--workspace_path", str(dense),
                     "--workspace_format", "COLMAP",
                     "--PatchMatchStereo.geom_consistency", "true"],
                   capture_output=True)
    out_ply = dense / "fused.ply"
    print("  MVS stereo_fusion ...")
    subprocess.run([str(COLMAP), "stereo_fusion",
                     "--workspace_path", str(dense),
                     "--workspace_format", "COLMAP",
                     "--input_type", "geometric",
                     "--output_path", str(out_ply)], capture_output=True)
    n_dense = 0
    if out_ply.exists():
        with open(out_ply, "rb") as fh:
            for _ in range(40):
                line = fh.readline().decode("ascii", errors="ignore")
                if line.startswith("element vertex"):
                    n_dense = int(line.split()[-1]); break
                if line.startswith("end_header"):
                    break
    return n_dense


def run_poisson(out_base):
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
    fused = out_base / "dense" / "fused.ply"
    out_mesh = out_base / "dense" / "mesh_poisson.ply"
    res = subprocess.run([sys.executable, "-c", sub, str(fused), str(out_mesh)],
                          capture_output=True, text=True)
    try:
        return int(res.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def analyze(out_base: Path, selection, intr):
    """Connectivity + cylinder + trajectory continuity + standard metrics."""
    sub = r"""
import pycolmap, json, sys, re, numpy as np
from pathlib import Path
sfm = Path(sys.argv[1])
models_dir = sfm / "models"
out = {}
# Top-level (best) model
rec = pycolmap.Reconstruction(str(sfm))
out["best"] = {
    "n_reg": int(rec.num_reg_images()),
    "n_sparse": int(rec.num_points3D()),
}
errs = [float(p.error) for p in rec.points3D.values()]
out["best"]["mean_reprojection_error_px"] = float(np.mean(errs)) if errs else None
# Registered image list with frame indices + camera centers
reg = []
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    cfw = img.cam_from_world()
    M = np.array(cfw.matrix())
    R = M[:3,:3]; t = M[:3,3]
    C = (-R.T @ t).tolist()
    a = (R.T @ np.array([0,0,1])).tolist()
    reg.append({"name": img.name, "frame_idx": fi,
                 "image_id": int(img.image_id),
                 "C": C, "axis": a})
reg.sort(key=lambda r: r["frame_idx"])
out["best"]["registered"] = reg
# Final camera params (verify pinned)
cam = next(iter(rec.cameras.values()))
out["best"]["camera_model"] = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
out["best"]["camera_params"] = list(cam.params)
# All sub-models in models_dir
out["all_submodels"] = []
if models_dir.exists():
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir(): continue
        try:
            r = pycolmap.Reconstruction(str(sub))
        except Exception as e:
            out["all_submodels"].append({"path": str(sub), "error": str(e)}); continue
        names = []
        for img in r.images.values():
            mm = re.search(r'f(\d+)\.png', img.name)
            names.append({"name": img.name,
                           "frame_idx": int(mm.group(1)) if mm else -1})
        names.sort(key=lambda x: x["frame_idx"])
        out["all_submodels"].append({
            "path": str(sub),
            "n_reg": int(r.num_reg_images()),
            "n_sparse": int(r.num_points3D()),
            "images": names,
        })
# Cylinder points: 3D points that have an observation from any
# inside-tube image (frame_idx >= 3470). Distinct from airway points.
inside_ids = set(); airway_ids = set()
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    if fi >= 3470:
        inside_ids.add(int(img.image_id))
    elif fi >= 0:
        airway_ids.add(int(img.image_id))
n_cyl = 0; n_air = 0; n_both = 0
for pid, p in rec.points3D.items():
    sees = {int(el.image_id) for el in p.track.elements}
    in_cyl = bool(sees & inside_ids)
    in_air = bool(sees & airway_ids)
    if in_cyl: n_cyl += 1
    if in_air: n_air += 1
    if in_cyl and in_air: n_both += 1
out["cylinder"] = {
    "n_inside_tube_cameras": len(inside_ids),
    "n_airway_cameras": len(airway_ids),
    "n_pts_seen_by_inside_tube": n_cyl,
    "n_pts_seen_by_airway": n_air,
    "n_pts_seen_by_both": n_both,
}
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(out_base / "sfm")],
                          capture_output=True, text=True)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    return out


def render_3view(out_base: Path, analysis, dense_ply: Path):
    """Side / top / front renders of dense MVS cloud + camera trajectory.
    Cylinder points (seen by inside-tube cameras) highlighted in magenta;
    airway points colored by camera-trajectory-parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    # Load dense cloud
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print("  dense cloud empty; skip render")
        return
    # Subsample for plot
    if len(pts) > 80000:
        ix = np.random.default_rng(0).choice(len(pts), 80000, replace=False)
        pts = pts[ix]

    # Camera centers from analysis
    reg = analysis["best"]["registered"]
    centers = np.array([r["C"] for r in reg], dtype=float)
    fis = np.array([r["frame_idx"] for r in reg], dtype=int)

    # Identify cylinder vs airway points: nearest-camera by frame_idx.
    # Approximate: project each point onto camera-trajectory PCA-1, then
    # interpolate the frame index. If inferred fi >= 3470 -> cylinder.
    if len(centers) >= 2:
        ctr = centers.mean(0)
        _, _, Vt = np.linalg.svd(centers - ctr, full_matrices=False)
        e1 = Vt[0]
        # Sign convention: first registered camera should have lowest fi
        order = np.argsort(fis)
        c_sorted = centers[order]; fi_sorted = fis[order]
        s_cam = (c_sorted - ctr) @ e1
        if s_cam[-1] < s_cam[0]:
            e1 = -e1
            s_cam = -s_cam
        # For each point, find nearest camera by s
        s_pts = (pts - ctr) @ e1
        nearest_ix = np.searchsorted(s_cam, s_pts, side="left")
        nearest_ix = np.clip(nearest_ix, 0, len(s_cam) - 1)
        nearest_fi = fi_sorted[nearest_ix]
        is_cyl = (nearest_fi >= 3470)
        # Color airway by camera-trajectory-position (0..1)
        s_norm = (s_pts - s_cam.min()) / max(s_cam.max() - s_cam.min(), 1e-9)
        s_norm = np.clip(s_norm, 0, 1)
    else:
        is_cyl = np.zeros(len(pts), dtype=bool)
        s_norm = np.zeros(len(pts))

    cyl_pts = pts[is_cyl]; air_pts = pts[~is_cyl]; air_s = s_norm[~is_cyl]

    fig = plt.figure(figsize=(18, 6))
    titles = ["side (XZ)", "top (XY)", "front (YZ)"]
    proj = [("X", "Z", 0, 2), ("X", "Y", 0, 1), ("Y", "Z", 1, 2)]
    for k, (lx, ly, ix, iy) in enumerate(proj):
        ax = fig.add_subplot(1, 3, k + 1)
        if len(air_pts) > 0:
            ax.scatter(air_pts[:, ix], air_pts[:, iy], c=air_s,
                       s=0.3, cmap="viridis", alpha=0.55, marker='.',
                       linewidths=0)
        if len(cyl_pts) > 0:
            ax.scatter(cyl_pts[:, ix], cyl_pts[:, iy], c="magenta",
                       s=0.6, alpha=0.85, marker='.', linewidths=0)
        # Camera trajectory
        ax.plot(centers[order][:, ix], centers[order][:, iy], "-",
                c="white", lw=2.0)
        ax.plot(centers[order][:, ix], centers[order][:, iy], "-",
                c="red", lw=1.0)
        # Label cylinder cameras
        in_tube = fi_sorted >= 3470
        ax.scatter(centers[order][in_tube, ix], centers[order][in_tube, iy],
                   c="yellow", s=30, marker="^", edgecolors="black",
                   linewidths=0.5, label="inside-tube cams")
        ax.scatter(centers[order][~in_tube, ix],
                   centers[order][~in_tube, iy],
                   c="red", s=8, marker="o", edgecolors="black",
                   linewidths=0.3, label="airway cams")
        ax.set_title(f"{titles[k]}  ({lx},{ly})")
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"gated2 / 2_v2 pinned recon  |  reg={analysis['best']['n_reg']}/30  "
        f"sparse={analysis['best']['n_sparse']}  dense={len(np.asarray(pcd.points))}\n"
        f"magenta = cylinder points (inferred via inside-tube cam proximity); "
        f"yellow ▲ = inside-tube cameras",
        fontsize=11)
    fig.tight_layout()
    out_png = out_base / "trajectory_3view.png"
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 3-view: {out_png}")


def main():
    print("=== STAGE 3: gated2/2_v2 pinned reconstruction ===")
    sel, intr, mask = load_inputs()
    print(f"  loaded {len(sel)} selected frames")
    print(f"  pinned intrinsics: fx={intr['fx']:.3f} fy={intr['fy']:.3f}  "
          f"k1={intr['k1']:.4f} k2={intr['k2']:.4f}  RMS={intr['rms_reproj_px']:.4f}")
    assert intr["rms_reproj_px"] < 0.5, "calibration RMS gate failed"

    RECON.mkdir(parents=True, exist_ok=True)
    for f in RECON.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)

    extracted = preprocess(RECON, sel, mask)
    image_list = sorted([e["filename"] for e in extracted])
    glottic_files = {e["filename"] for e in extracted if e["segment"] == "glottis"}
    print(f"  glottic init pool: {sorted(glottic_files)}")

    feats, matches, sfm_pairs = run_hloc(RECON, image_list)
    n_reg, n_pts = reconstruct(RECON, image_list, intr, feats, matches,
                                  sfm_pairs, glottic_files)
    n_dense = run_mvs(RECON)
    n_mesh_v = run_poisson(RECON)
    print(f"  MVS dense: {n_dense}   Poisson verts (VIEW ONLY): {n_mesh_v}")

    analysis = analyze(RECON, sel, intr)

    # Export sparse text
    text_dir = RECON / "sfm_text"
    text_dir.mkdir(exist_ok=True)
    subprocess.run([str(COLMAP), "model_converter",
                     "--input_path", str(RECON / "sfm"),
                     "--output_path", str(text_dir),
                     "--output_type", "TXT"], capture_output=True)

    # Render
    dense_ply = RECON / "dense" / "fused.ply"
    if dense_ply.exists():
        try:
            render_3view(RECON, analysis, dense_ply)
        except Exception as e:
            print(f"  render error: {e}")

    # Linkage report
    reg_idx = {r["frame_idx"] for r in analysis["best"]["registered"]}
    all_idx = {s["frame_idx"] for s in sel}
    failed = sorted(all_idx - reg_idx)
    transition_target = sorted(s["frame_idx"] for s in sel
                                 if s["segment"] == SEG_TUBE)
    transition_reg = sorted(fi for fi in transition_target if fi in reg_idx)
    transition_failed = sorted(fi for fi in transition_target if fi not in reg_idx)
    inside_tube_target = sorted(s["frame_idx"] for s in sel
                                 if s["frame_idx"] >= INSIDE_TUBE_THRESH)
    inside_tube_reg = sorted(fi for fi in inside_tube_target if fi in reg_idx)
    linkage_target = sorted(s["frame_idx"] for s in sel
                             if LINKAGE_LO <= s["frame_idx"] <= LINKAGE_HI)
    linkage_reg = sorted(fi for fi in linkage_target if fi in reg_idx)
    n_submodels = len(analysis.get("all_submodels", []))
    largest_sub = max((sm.get("n_reg", 0) for sm in analysis.get("all_submodels", [])),
                       default=0)

    # Pin verification
    cam_params = analysis["best"]["camera_params"]
    pin_check = {
        "model": analysis["best"]["camera_model"],
        "fx_final": cam_params[0] if len(cam_params) >= 1 else None,
        "fy_final": cam_params[1] if len(cam_params) >= 2 else None,
        "cx_final": cam_params[2] if len(cam_params) >= 3 else None,
        "cy_final": cam_params[3] if len(cam_params) >= 4 else None,
        "k1_final": cam_params[4] if len(cam_params) >= 5 else None,
        "k2_final": cam_params[5] if len(cam_params) >= 6 else None,
        "delta_fx_px": (cam_params[0] - intr["fx"]) if len(cam_params) >= 1 else None,
        "delta_fy_px": (cam_params[1] - intr["fy"]) if len(cam_params) >= 2 else None,
        "delta_k1": (cam_params[4] - intr["k1"]) if len(cam_params) >= 5 else None,
        "delta_k2": (cam_params[5] - intr["k2"]) if len(cam_params) >= 6 else None,
    }

    diag = {
        "mode": "pinned (NEW OPENCV)",
        "n_selected": len(sel),
        "n_registered": int(analysis["best"]["n_reg"]),
        "n_failed_to_register": len(failed),
        "failed_frames": failed,
        "n_sparse_points": int(analysis["best"]["n_sparse"]),
        "n_dense_points": int(n_dense),
        "n_poisson_verts_VIEW_ONLY": int(n_mesh_v),
        "mean_reprojection_error_px": analysis["best"]["mean_reprojection_error_px"],
        "transition_tube": {
            "target_indices": transition_target,
            "n_target": len(transition_target),
            "n_registered": len(transition_reg),
            "registered_indices": transition_reg,
            "failed_indices": transition_failed,
        },
        "inside_tube_only_f3470plus": {
            "target_indices": inside_tube_target,
            "n_target": len(inside_tube_target),
            "n_registered": len(inside_tube_reg),
            "registered_indices": inside_tube_reg,
        },
        "linkage_zone_f3300_3469": {
            "target_indices": linkage_target,
            "n_target": len(linkage_target),
            "n_registered": len(linkage_reg),
            "registered_indices": linkage_reg,
        },
        "cylinder_points": analysis["cylinder"],
        "connectivity": {
            "n_submodels": n_submodels,
            "best_submodel_n_reg": int(analysis["best"]["n_reg"]),
            "largest_submodel_n_reg": int(largest_sub),
            "all_submodels": analysis.get("all_submodels", []),
        },
        "pin_verification": pin_check,
        "pinned_intrinsics_source": str(GATED / "intrinsics_pinned.json"),
    }
    (RECON / "recon_results.json").write_text(json.dumps(diag, indent=2))
    print(f"\nsaved {RECON/'recon_results.json'}")


if __name__ == "__main__":
    main()
