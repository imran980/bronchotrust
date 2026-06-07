"""Locked pipeline on UNTRIMMED 2_v2 with user-specified curated frames
and corrected anatomical landmarks (2026-06-04).

Curated frames (30 total, user-defined):
  glottic   : 0, 4, 8, 12, 16, 20, 24, 28, 32, 36
  subglottic: 88, 100, 112, 124, 136, 148, 160, 172, 184, 196
  pull-back : 2010, 2060, 2110, 2160, 2210, 2260, 2310, 2360, 2410, 2460

Landmarks (camera positions):
  L1 glottis              = camera nearest to f=20
  L2 prox subglottis      = camera nearest to f=140
  L3 distal pre-carina    = camera nearest to f=2250

Pipeline (FROZEN, not tuned):
  preprocess: raw color + bezel mask 1px erode; NO CLAHE; NO specular mask
  features: SuperPoint-max via HLoc
  matcher: exhaustive (LightGlue)
  init: forced -- highest-inlier pair among the 10 GLOTTIC frames (f<=36)
  mapper: init_min_tri_angle=4, init_max_error=8, min_num_matches=8,
          init_max_forward_motion=0.99, abs_pose_max_error=20
  MVS: COLMAP CUDA patch_match_stereo + stereo_fusion
  Poisson: Open3D depth=9, 5% density crop

DCE (camera-as-center radial measurement):
  - Slab +/- 5 mm of landmark camera along its viewing tangent
  - All recon points (sparse + dense MVS) within slab kept
  - Project onto plane perp to tangent, origin = camera center
  - radial = |projected_pt|
  - median radial = wall radius proxy
  - DCE = 2 * median_radial (then * scale -> mm)
  - IQR = Q3 - Q1 of radial
  - Polar histogram: 36 bins of 10 deg each, count of pts per bin

Scale anchor:
  Karl Storz Parsons Size 2 inner aperture width = BLADE_MM (default 13.0)
  Detected from 3D points observed in the f=0 reconstruction with L>220,
  PC2 (P2.5-P97.5) robust extent = aperture width in scene units.

Outputs (runs/untrimmed/2_v2/):
  hero.png        : 3 raw frames + 3 polar histograms + 3 3D views
  results.json    : per-landmark DCE, IQR, n_pts, angular coverage
  mesh_metric.ply : reconstructed mesh in mm (cropped to within 50 mm of traj)
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
SRC = Path("/home/mi3dr/dataset/validation-videos/First 13 Videos")
COLMAP = Path("/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap")
OUT_BASE = PROJECT / "runs/untrimmed/2_v2"
VID = "2_v2"
BLADE_MM = 13.0

CURATED_FRAMES = (
    # 6 glottic (vocal cords visible)
    [0, 8, 16, 24, 32, 40] +
    # 4 blade-visible (suspension laryngoscope blade walls flanking lumen)
    [52, 58, 64, 70] +
    # 10 subglottic
    [88, 100, 112, 124, 136, 148, 160, 172, 184, 196] +
    # 10 pull-back
    [2010, 2060, 2110, 2160, 2210, 2260, 2310, 2360, 2410, 2460]
)
GLOTTIC_FRAMES_SET = set(CURATED_FRAMES[:6])
BLADE_FRAMES_SET = set(CURATED_FRAMES[6:10])

LANDMARK_TARGETS = {
    "L1 glottis":             20,
    "L2 prox subglottis":     140,
    "L3 distal pre-carina":   2250,
}


def video_path():
    for ext in ("mp4", "MP4"):
        p = SRC / f"{VID}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(VID)


def preprocess_raw_with_bezel(frame_indices):
    """Save raw color frames + bezel masks. NO CLAHE, NO specular."""
    intr = json.loads(
        (PROJECT / f"runs/barbour/{VID}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r_fov = float(intr["fov_radius"])
    Wimg, Himg = intr["image_size"]

    img_dir = OUT_BASE / "images"; msk_dir = OUT_BASE / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); msk_dir.mkdir(parents=True)

    yy, xx = np.indices((Himg, Wimg), dtype=np.float32)
    r_inner = r_fov - 1
    bezel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r_inner ** 2

    cap = cv2.VideoCapture(str(video_path()))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = []
    for fi in frame_indices:
        if fi >= n_total:
            print(f"  WARN: f={fi} beyond video length {n_total}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            print(f"  WARN: cannot read f={fi}")
            continue
        f[~bezel] = 0
        name = f"{VID}_f{fi:06d}.png"
        cv2.imwrite(str(img_dir / name), f,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (bezel.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        extracted.append({"frame_idx": fi, "filename": name})
    cap.release()
    print(f"  preprocessed {len(extracted)}/{len(frame_indices)} frames")

    dist_full = intr.get("dist_full")
    if dist_full and len(dist_full) >= 4:
        k1, k2, p1, p2 = (float(dist_full[i]) for i in range(4))
    else:
        k1, k2, p1, p2 = intr["k1"], intr["k2"], 0.0, 0.0
    return extracted, {
        "model": "OPENCV", "width": Wimg, "height": Himg,
        "params": [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                    k1, k2, p1, p2],
    }


def find_glottis_init_pair_from_db(db_path, glottic_filenames,
                                     min_frame_gap=12):
    """Highest-inlier glottic pair, but only among pairs with frame_gap
    >= min_frame_gap to ensure sufficient baseline for triangulation.
    Falls back to absolute highest-inlier glottic pair if none qualify."""
    import sqlite3
    def fi(n):
        m = re.search(r"f(\d+)\.png$", n)
        return int(m.group(1)) if m else -1
    conn = sqlite3.connect(str(db_path)); cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    id_name = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")
    pairs = cur.fetchall()
    conn.close()
    best_baseline = None  # pair with sufficient baseline + most inliers
    best_anywhere = None  # any glottic pair, most inliers
    for pair_id, rows in pairs:
        i1 = pair_id // 2147483647
        i2 = pair_id - i1 * 2147483647
        n1 = id_name.get(i1); n2 = id_name.get(i2)
        if n1 not in glottic_filenames or n2 not in glottic_filenames:
            continue
        gap = abs(fi(n1) - fi(n2))
        if best_anywhere is None or rows > best_anywhere[0]:
            best_anywhere = (rows, i1, i2, n1, n2, gap)
        if gap >= min_frame_gap:
            if best_baseline is None or rows > best_baseline[0]:
                best_baseline = (rows, i1, i2, n1, n2, gap)
    pick = best_baseline if best_baseline is not None else best_anywhere
    if pick is None: return None
    return pick[:5]  # drop gap from return tuple


def run_hloc_pass1(image_list):
    """First pass: extract + match + default-init recon (populates DB)."""
    from hloc import (extract_features, match_features,
                      pairs_from_exhaustive)
    from hloc import reconstruction as hloc_recon
    images_dir = OUT_BASE / "images"
    sfm_pairs = OUT_BASE / "pairs-exhaustive.txt"
    sfm_dir = OUT_BASE / "sfm"; sfm_dir.mkdir(parents=True, exist_ok=True)
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    print("  HLoc SuperPoint extraction ...")
    features_path = extract_features.main(
        feature_conf, images_dir, OUT_BASE, image_list=image_list)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    print("  HLoc LightGlue exhaustive matching ...")
    matches_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], OUT_BASE)
    mapper_options = {
        "min_num_matches": 8,
        "ba_local_max_num_iterations": 50,
        "ba_global_max_num_iterations": 100,
        "multiple_models": True, "max_num_models": 50,
        "min_model_size": 3,
        # *** PIN INTRINSICS: do NOT refine focal length / distortion ***
        "ba_refine_focal_length": False,
        "ba_refine_extra_params": False,
        "ba_refine_principal_point": False,
        "mapper": {
            "init_min_tri_angle": 4.0, "init_max_error": 8.0,
            "init_min_num_inliers": 15,
            "init_max_forward_motion": 0.99,
            "abs_pose_max_error": 20.0,
            "abs_pose_min_num_inliers": 15,
            "abs_pose_min_inlier_ratio": 0.1,
            "filter_max_reproj_error": 8.0,
            "filter_min_tri_angle": 1.0,
        },
    }
    # *** Pin OPENCV intrinsics from per-session calibration ***
    intr = json.loads(
        (PROJECT / f"runs/barbour/{VID}/intrinsics.json").read_text())
    dist_full = intr.get("dist_full") or []
    if len(dist_full) >= 4:
        k1, k2, p1, p2 = (float(dist_full[i]) for i in range(4))
    else:
        k1, k2, p1, p2 = intr.get("k1", 0.0), intr.get("k2", 0.0), 0.0, 0.0
    cam_params_str = ",".join(f"{p:.10g}" for p in
                                [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                                 k1, k2, p1, p2])
    image_options = {"camera_model": "OPENCV",
                      "camera_params": cam_params_str}
    print(f"  Pass 1: pycolmap reconstruction (default init, "
          f"pinned OPENCV f={intr['fx']:.1f}px) ...")
    try:
        model = hloc_recon.main(
            sfm_dir, images_dir, sfm_pairs, features_path, matches_path,
            image_list=image_list,
            camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
            image_options=image_options,
            verbose=False, mapper_options=mapper_options)
        return model.num_reg_images() if model else 0
    except Exception as e:
        print(f"  Pass 1 error: {e}")
        return 0


def run_pass2_forced_init(init_pair, image_list):
    """Pass 2: forced glottis init via pycolmap (DB already populated)."""
    import pycolmap
    sfm_dir = OUT_BASE / "sfm"
    n_inl, i1, i2, n1, n2 = init_pair
    print(f"  Pass 2 init: {n1} <-> {n2}  inliers={n_inl}")

    # Clear prior reconstruction binaries (keep DB)
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        p = sfm_dir / fn
        if p.exists(): p.unlink()
    models_dir = sfm_dir / "models"
    if models_dir.exists(): shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 8
    opts.multiple_models = True
    opts.max_num_models = 50
    opts.min_model_size = 3
    opts.init_image_id1 = i1
    opts.init_image_id2 = i2
    # *** PIN INTRINSICS ***
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
            image_path=str(OUT_BASE / "images"),
            output_path=str(models_dir),
            options=opts)
        if not recons:
            print("  Pass 2 produced no models"); return 0
        best_id = max(recons.keys(), key=lambda k: recons[k].num_reg_images())
        best = recons[best_id]
        best.write_binary(str(sfm_dir))
        return best.num_reg_images()
    except Exception as e:
        print(f"  Pass 2 error: {e}")
        return 0


def run_mvs():
    sfm_dir = OUT_BASE / "sfm"
    dense = OUT_BASE / "dense"
    if dense.exists(): shutil.rmtree(dense)
    dense.mkdir(parents=True)
    print("  MVS image_undistorter ...")
    subprocess.run([str(COLMAP), "image_undistorter",
                     "--image_path", str(OUT_BASE / "images"),
                     "--input_path", str(sfm_dir),
                     "--output_path", str(dense),
                     "--output_type", "COLMAP"], capture_output=True)
    print("  MVS patch_match_stereo ...")
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
                if line.startswith("end_header"): break
    return n_dense


def run_poisson():
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
    out_mesh = OUT_BASE / "dense" / "mesh_poisson.ply"
    fused = OUT_BASE / "dense" / "fused.ply"
    res = subprocess.run([sys.executable, "-c", sub, str(fused), str(out_mesh)],
                          capture_output=True, text=True)
    try: return int(res.stdout.strip().splitlines()[-1])
    except Exception: return 0


def export_geometry_and_blade():
    """Subprocess (pycolmap) -> geometry npz + blade detection from f=0."""
    sub = """
import pycolmap, cv2, numpy as np, sys, re, json
from pathlib import Path
out_base = Path(sys.argv[1])
images_dir = out_base / 'images'
sfm = out_base / 'sfm'
rec = pycolmap.Reconstruction(str(sfm))
centers, fids, image_ids = [], [], []
for img in rec.images.values():
    centers.append(np.array(img.projection_center()))
    m = re.search(r'f(\\d+)\\.png', img.name)
    fids.append(int(m.group(1)) if m else -1)
    image_ids.append(img.image_id)
order = np.argsort(fids)
centers = np.array([centers[i] for i in order])
fids = np.array([fids[i] for i in order])
pts_ids = list(rec.points3D.keys())
pts_xyz = np.array([list(p.xyz) for p in rec.points3D.values()])
np.savez(out_base / '_geometry.npz', centers=centers, fids=fids, pts=pts_xyz)

# Blade detection: 3D points observed in image f=0 with L>220 (raw)
img_f0 = None
for img in rec.images.values():
    m = re.search(r'f(\\d+)\\.png', img.name)
    if m and int(m.group(1)) == 0:
        img_f0 = img; break
result = {'n_pts': len(pts_xyz)}
if img_f0 is not None:
    L0 = cv2.cvtColor(cv2.imread(str(images_dir / img_f0.name)),
                      cv2.COLOR_BGR2LAB)[..., 0]
    # For each 3D point, check if it has an observation in f=0
    blade_pt_idxs = []
    for pi, pt in enumerate(rec.points3D.values()):
        for elem in pt.track.elements:
            if elem.image_id == img_f0.image_id:
                kp = img_f0.points2D[elem.point2D_idx]
                x = int(round(kp.xy[0])); y = int(round(kp.xy[1]))
                if 0 <= x < L0.shape[1] and 0 <= y < L0.shape[0]:
                    if L0[y, x] > 220:
                        blade_pt_idxs.append(pi)
                break
    result['blade_n_pts'] = len(blade_pt_idxs)
    if len(blade_pt_idxs) >= 5:
        blade_pts = pts_xyz[blade_pt_idxs]
        b_cen = blade_pts.mean(0)
        Xb = blade_pts - b_cen
        _, _, Vt = np.linalg.svd(Xb, full_matrices=False)
        proj = Xb @ Vt.T
        pc_robust = (np.percentile(proj, 97.5, axis=0)
                     - np.percentile(proj, 2.5, axis=0))
        # Save blade points for diagnostic
        np.savez(out_base / '_blade_points.npz', blade_pts=blade_pts,
                 pc_axes=Vt, pc_robust=pc_robust)
        result['blade_pc1'] = float(pc_robust[0])
        result['blade_pc2'] = float(pc_robust[1])
        result['blade_pc3'] = float(pc_robust[2])
        result['flatness_pc1_pc2'] = float(pc_robust[0] / max(pc_robust[1], 1e-9))
else:
    result['blade_n_pts'] = 0
    result['note'] = 'no f=0 reconstruction'
print(json.dumps(result))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(OUT_BASE)],
                          capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": res.stderr[-500:]}


def central_tangent(C, i):
    if i == 0: v = C[1] - C[0]
    elif i == len(C) - 1: v = C[-1] - C[-2]
    else: v = C[i + 1] - C[i - 1]
    return v / max(np.linalg.norm(v), 1e-9)


def landmark_dce_polar(centers, pts_all, L_idx, slab_units, scale_mm_per_unit):
    T = central_tangent(centers, L_idx)
    cam = centers[L_idx]
    rel = pts_all - cam
    axial = rel @ T
    mask = np.abs(axial) <= slab_units
    n_slab = int(mask.sum())
    if n_slab < 5:
        return {"n_slab": n_slab, "radius_med_units": None,
                "iqr_units": None, "dce_units": None,
                "dce_mm": None, "iqr_mm": None,
                "polar_hist": None,
                "n_angular_bins_covered": 0}
    in_plane = pts_all[mask] - cam
    in_plane = in_plane - np.outer(in_plane @ T, T)
    # Build 2D basis
    up = np.array([0.0, 0.0, 1.0])
    if abs(T @ up) > 0.95: up = np.array([0.0, 1.0, 0.0])
    e1 = up - (up @ T) * T; e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(T, e1)
    proj_2d = np.stack([in_plane @ e1, in_plane @ e2], axis=1)
    radial = np.linalg.norm(proj_2d, axis=1)
    r_med = float(np.median(radial))
    q25 = float(np.percentile(radial, 25))
    q75 = float(np.percentile(radial, 75))
    iqr = q75 - q25
    # Polar histogram (36 bins of 10 deg)
    theta = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
    theta_deg = (np.degrees(theta) + 360) % 360
    bins = np.zeros(36, dtype=np.int64)
    for t in theta_deg:
        b = int(t // 10) % 36
        bins[b] += 1
    n_covered = int((bins > 0).sum())
    return {
        "n_slab": n_slab,
        "radius_med_units": r_med,
        "radius_q25_units": q25,
        "radius_q75_units": q75,
        "iqr_units": iqr,
        "dce_units": 2 * r_med,
        "dce_mm": 2 * r_med * scale_mm_per_unit,
        "iqr_mm": iqr * scale_mm_per_unit,
        "radius_med_mm": r_med * scale_mm_per_unit,
        "polar_hist": bins.tolist(),
        "n_angular_bins_covered": n_covered,
        "polar_proj_2d_units": proj_2d.tolist(),  # for hero figure
        "radial_units": radial.tolist(),
    }


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for f in OUT_BASE.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)

    print(f"\n=== UNTRIMMED {VID} | locked pipeline | 30 curated frames ===")
    print(f"  glottic:    {CURATED_FRAMES[:10]}")
    print(f"  subglottic: {CURATED_FRAMES[10:20]}")
    print(f"  pull-back:  {CURATED_FRAMES[20:30]}")

    extracted, intr_doc = preprocess_raw_with_bezel(CURATED_FRAMES)
    image_list = sorted([e["filename"] for e in extracted])
    glottic_files = {e["filename"] for e in extracted
                     if e["frame_idx"] <= 36}

    n_reg_pre = run_hloc_pass1(image_list)
    print(f"  Pass 1 (default init) registered: {n_reg_pre}")
    # Snapshot pass-1 reconstruction in case pass 2 (forced glottis init)
    # collapses to a smaller model.
    sfm_dir = OUT_BASE / "sfm"
    pass1_dir = OUT_BASE / "sfm_pass1"
    if pass1_dir.exists(): shutil.rmtree(pass1_dir)
    pass1_dir.mkdir()
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        p = sfm_dir / fn
        if p.exists(): shutil.copy(str(p), str(pass1_dir / fn))

    init_pair = find_glottis_init_pair_from_db(
        sfm_dir / "database.db", glottic_files, min_frame_gap=12)
    if init_pair is None:
        print("  No glottic pair found in DB — using pass 1 result.")
        n_reg = n_reg_pre
    else:
        n_reg_p2 = run_pass2_forced_init(init_pair, image_list)
        print(f"  Pass 2 (forced glottis init) registered: {n_reg_p2}")
        if n_reg_p2 >= n_reg_pre:
            n_reg = n_reg_p2
            print(f"  Using pass 2 result ({n_reg_p2} >= {n_reg_pre})")
        else:
            # Restore pass 1
            print(f"  Pass 2 ({n_reg_p2}) < pass 1 ({n_reg_pre}); "
                  f"restoring pass 1")
            for fn in ["cameras.bin", "images.bin", "points3D.bin",
                        "frames.bin", "rigs.bin"]:
                src_p = pass1_dir / fn
                if src_p.exists():
                    shutil.copy(str(src_p), str(sfm_dir / fn))
            n_reg = n_reg_pre
    if n_reg < 5:
        print("  Reconstruction too small to proceed.")
        return

    n_dense = run_mvs()
    print(f"  MVS dense points: {n_dense}")
    n_mesh = run_poisson()
    print(f"  Poisson mesh verts: {n_mesh}")

    geo = export_geometry_and_blade()
    print(f"  geometry: {geo}")
    if "blade_pc2" not in geo or not geo.get("blade_pc2"):
        print("  WARN: no blade detection from f=0 — fallback to global blade")
        # Optional fallback: use whole-recon high-L blade. Skipping for now.
        scale = None
    else:
        scale = BLADE_MM / geo["blade_pc2"]
        print(f"  Scale (f=0 blade PC2 = {geo['blade_pc2']:.3f}u, "
              f"BLADE_MM = {BLADE_MM} mm): {scale:.4f} mm/u")
        print(f"  Blade flatness PC1/PC2 = {geo['flatness_pc1_pc2']:.2f}")

    g = np.load(OUT_BASE / "_geometry.npz")
    centers, fids, sparse = g["centers"], g["fids"], g["pts"]
    import open3d as o3d
    dense = np.empty((0, 3))
    fused = OUT_BASE / "dense" / "fused.ply"
    if fused.exists():
        pcd = o3d.io.read_point_cloud(str(fused))
        dense = np.asarray(pcd.points)
    pts_all = np.concatenate([sparse, dense]) if len(dense) else sparse
    print(f"  total points (sparse+dense): {len(pts_all)}")

    if scale is None: scale = 1.0  # fallback for DCE in scene units only

    # Find nearest-curated landmark cameras
    landmarks_info = {}
    slab_mm = 5.0; slab_units = slab_mm / scale
    for lab, target_f in LANDMARK_TARGETS.items():
        L_idx = int(np.argmin(np.abs(fids - target_f)))
        cam_f = int(fids[L_idx])
        r = landmark_dce_polar(centers, pts_all, L_idx, slab_units, scale)
        r["target_frame"] = target_f
        r["camera_frame"] = cam_f
        r["landmark"] = lab
        landmarks_info[lab] = r
        if r["dce_mm"] is not None:
            print(f"  {lab:<24} target=f{target_f}  cam=f{cam_f}  "
                  f"n={r['n_slab']:>4}  r_med={r['radius_med_mm']:.2f} mm  "
                  f"DCE={r['dce_mm']:.2f} mm  IQR={r['iqr_mm']:.2f} mm  "
                  f"angular_cov={r['n_angular_bins_covered']}/36")
        else:
            print(f"  {lab:<24} target=f{target_f}  cam=f{cam_f}  "
                  f"insufficient points (n={r['n_slab']})")

    # Metric mesh for downstream comparison
    mesh_in = OUT_BASE / "dense" / "mesh_poisson.ply"
    if mesh_in.exists() and scale != 1.0:
        mesh = o3d.io.read_triangle_mesh(str(mesh_in))
        verts = np.asarray(mesh.vertices).copy()
        # Crop to within 50 mm of any camera (in scene units)
        crop_units = 50.0 / scale
        keep = np.zeros(len(verts), dtype=bool)
        for c in centers:
            keep |= np.sqrt(((verts - c) ** 2).sum(axis=1)) < crop_units
        print(f"  metric mesh crop: kept {keep.sum()}/{len(verts)}")
        old_to_new = -np.ones(len(verts), dtype=np.int64)
        new_idx = 0
        for i, k in enumerate(keep):
            if k: old_to_new[i] = new_idx; new_idx += 1
        new_verts = verts[keep]
        old_tris = np.asarray(mesh.triangles)
        mask_tris = np.array([keep[t[0]] and keep[t[1]] and keep[t[2]]
                              for t in old_tris])
        new_tris = old_to_new[old_tris[mask_tris]]
        verts_mm = new_verts * scale - (centers * scale).mean(0)
        m2 = o3d.geometry.TriangleMesh()
        m2.vertices = o3d.utility.Vector3dVector(verts_mm)
        m2.triangles = o3d.utility.Vector3iVector(new_tris.astype(np.int32))
        m2.compute_vertex_normals()
        out_metric = OUT_BASE / "mesh_metric.ply"
        o3d.io.write_triangle_mesh(str(out_metric), m2)
        bbox_ext = verts_mm.max(0) - verts_mm.min(0)
        print(f"  metric mesh: {len(verts_mm)} verts, bbox (mm): {bbox_ext.round(1).tolist()}")

    # Save results.json
    # Strip polar_proj_2d_units and radial_units to keep file size small
    landmarks_save = {}
    for lab, r in landmarks_info.items():
        clean = {k: v for k, v in r.items()
                  if k not in ("polar_proj_2d_units", "radial_units")}
        landmarks_save[lab] = clean
    results = {
        "vid": VID,
        "n_curated_input": len(image_list),
        "n_registered": int(n_reg),
        "n_sparse_points": int(len(sparse)),
        "n_mvs_dense_points": int(n_dense),
        "n_mesh_verts": int(n_mesh),
        "blade_pc1_pc2_pc3_units": [
            geo.get("blade_pc1"), geo.get("blade_pc2"), geo.get("blade_pc3")],
        "blade_flatness_pc1_pc2": geo.get("flatness_pc1_pc2"),
        "blade_n_pts_observed_in_f0": geo.get("blade_n_pts"),
        "blade_mm_assumed": BLADE_MM,
        "scale_mm_per_unit": scale,
        "landmarks": landmarks_save,
    }
    out_json = OUT_BASE / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n  saved {out_json}")

    # Cache polar diagnostics + DCE for hero figure
    np.savez(OUT_BASE / "_dce_cache.npz",
              **{f"{lab}_proj": np.array(landmarks_info[lab].get("polar_proj_2d_units", []))
                 for lab in landmarks_info if landmarks_info[lab].get("polar_proj_2d_units")},
              **{f"{lab}_radial": np.array(landmarks_info[lab].get("radial_units", []))
                 for lab in landmarks_info if landmarks_info[lab].get("radial_units")})
    print(f"  cached polar data to _dce_cache.npz")


if __name__ == "__main__":
    main()
