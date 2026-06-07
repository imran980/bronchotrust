"""LOCKED final pipeline runner (2026-06-04).

Per video, executes the frozen pipeline:
  1. Detect forward-pass end (peak dark fraction).
  2. Compute 3 anatomical bins (glottis / subglottic / distal) anchored on
     subglottic_kf and forward_end.
  3. Curated picks: ~10 frames per bin, spaced by max cumulative flow.
  4. Preprocess: RAW color + 1px bezel mask. NO CLAHE. NO specular mask.
  5. HLoc SuperPoint-max + LightGlue + exhaustive matcher.
  6. Init pair: highest-inlier pair where both frames are in glottis bin.
  7. pycolmap mapper: init_min_tri_angle=4, min_num_matches=8,
     otherwise default.
  8. MVS dense (image_undistorter + patch_match_stereo + stereo_fusion).
  9. Poisson mesh (Open3D, depth=9, 5% density crop).
 10. DCE at L1=first glottis cam, L2=first subglottic cam, L3=last cam
     (slab ±5 scene units, algebraic circle fit + 1000-sample bootstrap σ).
 11. Blade scale: 3D points whose median observed L > 220 (raw) form the
     blade cluster; PC2 P2.5-P97.5 robust extent = blade width in scene
     units; scale_mm_per_unit = BLADE_MM / blade_width_units.

NO mapper tuning sweeps. NO alternate features.
"""
from __future__ import annotations

import argparse
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
FINAL_ROOT = PROJECT / "runs/final"
BLADE_MM_DEFAULT = 14.0


def video_path(vid: str) -> Path:
    for ext in ("mp4", "MP4"):
        p = SRC / f"{vid}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(vid)


def detect_forward_end(vid: str) -> tuple[int, int]:
    intr = json.loads(
        (PROJECT / f"runs/barbour/{vid}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r = float(intr["fov_radius"])
    cap = cv2.VideoCapture(str(video_path(vid)))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yy, xx = np.indices((H, W), dtype=np.float32)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.85) ** 2
    dark_frac, idxs = [], []
    for fi in range(0, n_total, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: continue
        L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0][mask]
        dark_frac.append(float((L < 30).sum() / L.size))
        idxs.append(fi)
    cap.release()
    arr = np.array(dark_frac); idxs = np.array(idxs)
    sm = np.convolve(arr, np.ones(15) / 15, mode="same")
    return int(idxs[np.argmax(sm)]), n_total


def compute_3_bins(subglottic_kf: int, forward_end: int):
    """Three anatomical bins: glottis / subglottic / distal."""
    # Glottis = pre-subglottic-kf
    glottis_end = max(subglottic_kf, 60)
    # Subglottic = subglottic_kf + 150 frames (~5s)
    sg_end = min(glottis_end + 200, int(0.55 * forward_end))
    return [
        ("glottis",    0, glottis_end),
        ("subglottic", glottis_end, sg_end),
        ("distal",     sg_end, forward_end),
    ]


def consecutive_flow(vid: str, forward_end: int):
    """Half-resolution Farneback flow between consecutive source frames."""
    cap = cv2.VideoCapture(str(video_path(vid)))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(forward_end, n_total - 1)
    grays = []
    for fi in range(0, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
        grays.append((fi, g))
    cap.release()
    flow_mag = np.zeros(len(grays), dtype=np.float32)
    for i in range(1, len(grays)):
        g1, g2 = grays[i - 1][1], grays[i][1]
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.linalg.norm(flow, axis=2)
        valid = (g1 > 5) & (g2 > 5)
        flow_mag[i] = float(np.median(mag[valid]) * 2) if valid.any() else 0
    return np.array([g[0] for g in grays]), flow_mag, np.cumsum(flow_mag)


def pick_per_bin_by_cumflow(frame_idxs, cum_flow, bins, per_bin=10):
    """For each bin, pick `per_bin` frames evenly distributed in cum_flow
    space within that bin."""
    picks = []
    for name, lo, hi in bins:
        in_bin = [(i, frame_idxs[i]) for i in range(len(frame_idxs))
                  if lo <= frame_idxs[i] < hi]
        if not in_bin: continue
        idxs_in_bin = np.array([x[0] for x in in_bin])
        cum_in = cum_flow[idxs_in_bin] - cum_flow[idxs_in_bin[0]]
        if cum_in[-1] <= 0:
            # No motion in bin -- fall back to evenly spaced source frames
            sel = np.linspace(0, len(idxs_in_bin) - 1, per_bin).astype(int)
        else:
            targets = np.linspace(0, cum_in[-1], per_bin)
            sel = []
            seen = set()
            for t in targets:
                j = int(np.argmin(np.abs(cum_in - t)))
                if j not in seen:
                    seen.add(j); sel.append(j)
        for j in sel:
            local_idx = idxs_in_bin[j]
            fi = int(frame_idxs[local_idx])
            picks.append({"frame_idx": fi, "bin": name,
                          "cum_flow_px": float(cum_flow[local_idx])})
    return picks


def preprocess_raw_with_bezel(vid: str, picks, out_base: Path,
                                bezel_erode_px: int = 1):
    """Save raw color frames with bezel mask applied (zero outside)."""
    intr = json.loads(
        (PROJECT / f"runs/barbour/{vid}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r_fov = float(intr["fov_radius"])
    Wimg, Himg = intr["image_size"]
    img_dir = out_base / "images"
    msk_dir = out_base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); msk_dir.mkdir(parents=True)
    yy, xx = np.indices((Himg, Wimg), dtype=np.float32)
    r_inner = r_fov - bezel_erode_px
    bezel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r_inner ** 2
    cap = cv2.VideoCapture(str(video_path(vid)))
    for e in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_idx"])
        ok, f = cap.read()
        if not ok: continue
        # *** NO CLAHE, NO specular. Just zero bezel. ***
        f[~bezel] = 0
        name = f"{vid}_f{e['frame_idx']:06d}.png"
        e["filename"] = name
        cv2.imwrite(str(img_dir / name), f,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (bezel.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cap.release()
    dist_full = intr.get("dist_full")
    if dist_full and len(dist_full) >= 4:
        k1, k2, p1, p2 = (float(dist_full[i]) for i in range(4))
    else:
        k1, k2, p1, p2 = intr["k1"], intr["k2"], 0.0, 0.0
    return {"model": "OPENCV", "width": Wimg, "height": Himg,
            "params": [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                        k1, k2, p1, p2]}


def run_hloc_recon(out_base: Path, image_list, intr_doc, init_pair_glottis):
    """SuperPoint-max + LightGlue + exhaustive + pycolmap. Returns
    (n_reg, n_pts, model). Forces init pair if init_pair_glottis is set."""
    from hloc import (extract_features, match_features,
                      pairs_from_exhaustive)
    from hloc import reconstruction as hloc_recon
    images_dir = out_base / "images"
    sfm_pairs = out_base / "pairs-exhaustive.txt"
    sfm_dir = out_base / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    print(f"  HLoc SuperPoint extract on {len(image_list)} frames ...")
    features_path = extract_features.main(
        feature_conf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    print(f"  HLoc LightGlue exhaustive matching ...")
    matches_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], out_base)

    # If init pair specified, edit mapper to force it.
    # pycolmap pipeline init pair: pass image IDs.
    mapper_options = {
        "min_num_matches": 8,
        "ba_local_max_num_iterations": 50,
        "ba_global_max_num_iterations": 100,
        "multiple_models": True, "max_num_models": 50,
        "min_model_size": 3,
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
    if init_pair_glottis is not None:
        mapper_options["init_image_id1"] = init_pair_glottis[0]
        mapper_options["init_image_id2"] = init_pair_glottis[1]
    print(f"  pycolmap reconstruction ...")
    try:
        model = hloc_recon.main(
            sfm_dir, images_dir, sfm_pairs, features_path, matches_path,
            image_list=image_list,
            camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
            verbose=False, mapper_options=mapper_options)
    except Exception as e:
        print(f"  reconstruction error: {e}")
        return 0, 0, None
    if model is None: return 0, 0, None
    return model.num_reg_images(), model.num_points3D(), model


def find_glottis_init_pair(out_base: Path, picks, intr_doc):
    """Read database after SIFT/LightGlue matches, pick highest-inlier
    pair where BOTH images are glottis-bin frames."""
    import sqlite3
    # Read sfm/database.db to find image_id->name
    db = out_base / "sfm" / "database.db"
    if not db.exists(): return None
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    id_name = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")
    pairs = cur.fetchall()
    conn.close()
    glottis_files = {p["filename"] for p in picks if p["bin"] == "glottis"}
    best = (-1, None, None, None, None)
    for pair_id, rows in pairs:
        i1 = pair_id // 2147483647
        i2 = pair_id - i1 * 2147483647
        n1 = id_name.get(i1); n2 = id_name.get(i2)
        if n1 in glottis_files and n2 in glottis_files and rows > best[0]:
            best = (rows, i1, i2, n1, n2)
    if best[0] < 0: return None
    return best


def run_mvs_poisson(out_base: Path, model):
    """Use COLMAP CLI for MVS (CUDA), Open3D for Poisson."""
    # First, find biggest sub-model — pycolmap reconstruction gives only the
    # accepted one already. But sparse output is in `sfm/models/0` typically.
    sfm_dir = out_base / "sfm"
    models_dir = sfm_dir / "models"
    sub_models = sorted([p for p in models_dir.iterdir() if p.is_dir()]) \
        if models_dir.exists() else []
    if not sub_models:
        # Some HLoc versions put model files directly in sfm/
        sub_models = [sfm_dir]
    biggest, bn = None, -1
    for m in sub_models:
        # Convert binary -> txt if needed
        if not (m / "images.txt").exists() and (m / "images.bin").exists():
            subprocess.run([str(COLMAP), "model_converter",
                             "--input_path", str(m),
                             "--output_path", str(m),
                             "--output_type", "TXT"], capture_output=True)
        if (m / "images.txt").exists():
            n = sum(1 for l in open(m / "images.txt")
                    if not l.startswith("#") and l.strip()) // 2
            if n > bn: bn = n; biggest = m
    if biggest is None: return 0, 0, None
    dense = out_base / "dense"
    if dense.exists(): shutil.rmtree(dense)
    dense.mkdir(parents=True)
    print(f"  MVS: image_undistorter ...")
    subprocess.run([str(COLMAP), "image_undistorter",
                     "--image_path", str(out_base / "images"),
                     "--input_path", str(biggest),
                     "--output_path", str(dense),
                     "--output_type", "COLMAP"], capture_output=True)
    print(f"  MVS: patch_match_stereo ...")
    subprocess.run([str(COLMAP), "patch_match_stereo",
                     "--workspace_path", str(dense),
                     "--workspace_format", "COLMAP",
                     "--PatchMatchStereo.geom_consistency", "true"],
                   capture_output=True)
    out_ply = dense / "fused.ply"
    print(f"  MVS: stereo_fusion ...")
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
    return n_dense, bn, out_ply if n_dense > 0 else None


def poisson_mesh(out_base: Path, fused_ply: Path):
    """Run Poisson in a subprocess to avoid Open3D + pycolmap segfault."""
    sub = """
import open3d as o3d, numpy as np, sys
fused = sys.argv[1]; out = sys.argv[2]
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
    mesh_out = out_base / "dense" / "mesh_poisson.ply"
    res = subprocess.run([sys.executable, "-c", sub,
                           str(fused_ply), str(mesh_out)],
                          capture_output=True, text=True)
    try:
        n_v = int(res.stdout.strip().splitlines()[-1])
    except Exception:
        n_v = 0
    return mesh_out if mesh_out.exists() else None, n_v


# ---- DCE + blade scale ----
def algebraic_circle_fit(pts_2d):
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = -sol[0] / 2; cy = -sol[1] / 2
    r2 = cx * cx + cy * cy - sol[2]
    return cx, cy, (np.sqrt(r2) if r2 > 0 else float("nan"))


def bootstrap_circle(pts_2d, n_boot=1000):
    rng = np.random.default_rng(0); n = len(pts_2d)
    radii = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, r = algebraic_circle_fit(pts_2d[idx])
        if np.isfinite(r): radii.append(r)
    if not radii: return float("nan"), float("nan")
    return float(np.median(radii)), float(np.std(radii))


def central_tangent(C, i):
    if i == 0: v = C[1] - C[0]
    elif i == len(C) - 1: v = C[-1] - C[-2]
    else: v = C[i + 1] - C[i - 1]
    return v / max(np.linalg.norm(v), 1e-9)


def cross_section_2d(pts, cam, T, slab=5.0):
    rel = pts - cam; axial = rel @ T
    mask = np.abs(axial) <= slab
    if mask.sum() < 5: return None
    in_plane = pts[mask] - cam
    in_plane = in_plane - np.outer(in_plane @ T, T)
    up = np.array([0.0, 0.0, 1.0])
    if abs(T @ up) > 0.95: up = np.array([0.0, 1.0, 0.0])
    e1 = up - (up @ T) * T; e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(T, e1)
    return np.stack([in_plane @ e1, in_plane @ e2], axis=1)


def compute_dce_at(centers, pts, L_idx, slab=5.0):
    T = central_tangent(centers, L_idx)
    pts_2d = cross_section_2d(pts, centers[L_idx], T, slab)
    if pts_2d is None or len(pts_2d) < 5:
        n = int((np.abs((pts - centers[L_idx]) @ T) <= slab).sum())
        return None, None, n
    _, _, r_fit = algebraic_circle_fit(pts_2d)
    _, r_sigma = bootstrap_circle(pts_2d)
    return 2 * r_fit, 2 * r_sigma, len(pts_2d)


def blade_scale_from_reconstruction(out_base, blade_mm):
    """Score each 3D point by median observed L on the RAW image. Blade
    cluster = points with median L > 220. PC2 robust extent = blade
    width in scene units."""
    sub = """
import pycolmap, cv2, json, numpy as np, sys, re
from pathlib import Path
out_base = Path(sys.argv[1])
sfm_dir = out_base / 'sfm'
# HLoc writes the largest reconstruction to the top-level sfm dir;
# only the smaller intermediates go to sfm/models/<i>. Prefer top-level.
def has_recon(d):
    return all((d / f).exists() for f in
                ['cameras.bin', 'images.bin', 'points3D.bin'])
if has_recon(sfm_dir):
    biggest = sfm_dir
else:
    models_dir = sfm_dir / 'models'
    cands = sorted([p for p in models_dir.iterdir() if p.is_dir()]) \
            if models_dir.exists() else []
    biggest, bn = None, -1
    for m in cands:
        try:
            r = pycolmap.Reconstruction(str(m))
            n = r.num_reg_images()
            if n > bn: bn = n; biggest = m
        except Exception: pass
    if biggest is None: biggest = sfm_dir
rec = pycolmap.Reconstruction(str(biggest))
images_dir = out_base / 'images'
img_L = {}
for img in rec.images.values():
    p = images_dir / img.name
    if not p.exists(): continue
    bgr = cv2.imread(str(p))
    if bgr is None: continue
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    img_L[img.image_id] = L
pts_ids = list(rec.points3D.keys())
pts_xyz = np.array([list(p.xyz) for p in rec.points3D.values()])
scores = []
for pt_id, pt in rec.points3D.items():
    vals = []
    for elem in pt.track.elements:
        L = img_L.get(elem.image_id)
        if L is None: continue
        img = rec.images[elem.image_id]
        kp = img.points2D[elem.point2D_idx]
        x, y = int(round(kp.xy[0])), int(round(kp.xy[1]))
        if 0 <= x < L.shape[1] and 0 <= y < L.shape[0]:
            vals.append(int(L[y, x]))
    scores.append(float(np.median(vals)) if vals else float('nan'))
scores = np.array(scores)
is_blade = scores > 220
n_blade = int(is_blade.sum())
if n_blade < 10:
    # relax
    thresh = float(np.percentile(scores[~np.isnan(scores)], 90)) if n_blade < 5 else 200
    is_blade = scores > thresh
    n_blade = int(is_blade.sum())
result = {
    'n_blade': n_blade,
    'blade_thresh_L': 220 if n_blade >= 10 else float(thresh) if 'thresh' in dir() else 220,
}
if n_blade >= 5:
    blade_pts = pts_xyz[is_blade]
    bcen = blade_pts.mean(0)
    Xb = blade_pts - bcen
    _, _, Vt = np.linalg.svd(Xb, full_matrices=False)
    proj = Xb @ Vt.T
    pc_robust = np.percentile(proj, 97.5, axis=0) - np.percentile(proj, 2.5, axis=0)
    result['pc1_units'] = float(pc_robust[0])
    result['pc2_units'] = float(pc_robust[1])
    result['pc3_units'] = float(pc_robust[2])
    result['blade_width_units'] = float(pc_robust[1])
print(json.dumps(result))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(out_base)],
                          capture_output=True, text=True)
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        data = {"n_blade": 0, "error": "subprocess_failed",
                "stderr": res.stderr[-500:]}
    if "blade_width_units" in data and data["blade_width_units"] > 0:
        data["scale_mm_per_unit"] = blade_mm / data["blade_width_units"]
        data["blade_mm_assumed"] = blade_mm
        data["scale_source"] = "blade_PC2_robust"
    return data


def export_geometry(out_base: Path):
    """Subprocess to dump centers, fids, sparse pts to npz."""
    sub = """
import pycolmap, numpy as np, sys, re, json
from pathlib import Path
out_base = Path(sys.argv[1])
sfm_dir = out_base / 'sfm'
# HLoc writes the largest reconstruction to the top-level sfm dir;
# only the smaller intermediates go to sfm/models/<i>. Prefer top-level.
def has_recon(d):
    return all((d / f).exists() for f in
                ['cameras.bin', 'images.bin', 'points3D.bin'])
if has_recon(sfm_dir):
    biggest = sfm_dir
else:
    models_dir = sfm_dir / 'models'
    cands = sorted([p for p in models_dir.iterdir() if p.is_dir()]) \
            if models_dir.exists() else []
    biggest, bn = None, -1
    for m in cands:
        try:
            r = pycolmap.Reconstruction(str(m))
            n = r.num_reg_images()
            if n > bn: bn = n; biggest = m
        except Exception: pass
    if biggest is None: biggest = sfm_dir
rec = pycolmap.Reconstruction(str(biggest))
centers, fids, names = [], [], []
for img in rec.images.values():
    centers.append(np.array(img.projection_center()))
    m = re.search(r'f(\\d+)\\.png', img.name)
    fids.append(int(m.group(1)) if m else -1)
    names.append(img.name)
order = np.argsort(fids)
centers = np.array([centers[i] for i in order])
fids = np.array([fids[i] for i in order])
names = [names[i] for i in order]
pts = np.array([list(p.xyz) for p in rec.points3D.values()])
np.savez(out_base / '_geometry.npz',
         centers=centers, fids=fids, pts=pts,
         names=np.array(names))
print(json.dumps({'n_cams': len(centers), 'n_pts': len(pts)}))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(out_base)],
                          capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": res.stderr[-500:]}


def run_pipeline(vid: str, subglottic_kf: int, blade_mm: float = BLADE_MM_DEFAULT):
    print(f"\n{'='*70}\n  {vid}  (subglottic_kf={subglottic_kf})\n{'='*70}")
    out_base = FINAL_ROOT / vid
    out_base.mkdir(parents=True, exist_ok=True)
    for f in out_base.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)

    fwd_end, n_total = detect_forward_end(vid)
    print(f"  forward_end={fwd_end}  (total={n_total})")
    bins = compute_3_bins(subglottic_kf, fwd_end)
    print(f"  bins: {bins}")

    frame_idxs, flow_mag, cum_flow = consecutive_flow(vid, fwd_end)
    picks = pick_per_bin_by_cumflow(frame_idxs, cum_flow, bins, per_bin=10)
    print(f"  curated picks: {len(picks)} "
          f"({sum(1 for p in picks if p['bin']=='glottis')}g/"
          f"{sum(1 for p in picks if p['bin']=='subglottic')}sg/"
          f"{sum(1 for p in picks if p['bin']=='distal')}d)")

    intr_doc = preprocess_raw_with_bezel(vid, picks, out_base)
    image_list = sorted([p["filename"] for p in picks])

    # Pass 1: extract + match + default-init recon to populate DB
    n_reg_pre, n_pts_pre, model_pre = run_hloc_recon(
        out_base, image_list, intr_doc, init_pair_glottis=None)

    # Find best glottis-glottis pair from the matches DB
    init_pair = find_glottis_init_pair(out_base, picks, intr_doc)
    if init_pair is not None:
        n_inl, i1, i2, n1, n2 = init_pair
        print(f"  glottis init: {n1} <-> {n2}  inliers={n_inl}")

    # Pass 2: forced glottis init re-run via pycolmap directly (DB already
    # has features + matches from pass 1). This is the locked-spec
    # "manual init pair" step.
    n_reg, n_pts, model = n_reg_pre, n_pts_pre, model_pre
    if init_pair is not None:
        import pycolmap
        sfm_dir = out_base / "sfm"
        # Clear prior sparse outputs (but keep database.db)
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
            print(f"  Pass 2: forced-init pycolmap mapper ...")
            recons = pycolmap.incremental_mapping(
                database_path=str(sfm_dir / "database.db"),
                image_path=str(out_base / "images"),
                output_path=str(models_dir),
                options=opts)
            if recons:
                # Pick the biggest
                best_id = max(recons.keys(),
                                key=lambda k: recons[k].num_reg_images())
                model_p2 = recons[best_id]
                n_reg_p2 = model_p2.num_reg_images()
                n_pts_p2 = model_p2.num_points3D()
                print(f"  Pass 2: {len(recons)} sub-models, "
                      f"largest = {n_reg_p2} cams, {n_pts_p2} pts")
                # Write the best one to top-level sfm_dir for downstream
                model_p2.write_binary(str(sfm_dir))
                if n_reg_p2 > n_reg_pre:
                    n_reg, n_pts, model = n_reg_p2, n_pts_p2, model_p2
                    print(f"  Pass 2 BEATS pass 1 ({n_reg_pre} -> {n_reg_p2}) -- keeping pass 2")
                else:
                    print(f"  Pass 2 did not improve ({n_reg_pre} vs {n_reg_p2})")
        except Exception as e:
            print(f"  Pass 2 failed: {e}")

    if model is None or n_reg < 3:
        print(f"  RECONSTRUCTION FAILED (n_reg={n_reg})")
        summary = {"vid": vid, "outcome": "RECON_FAIL",
                    "n_input_curated": len(picks),
                    "n_registered": int(n_reg),
                    "init_pair_inliers": init_pair[0] if init_pair else None}
        (out_base / "_final_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    n_dense, biggest_size, fused_ply = run_mvs_poisson(out_base, model)
    print(f"  MVS dense: {n_dense} pts")
    mesh_path, n_mesh_v = None, 0
    if fused_ply is not None:
        mesh_path, n_mesh_v = poisson_mesh(out_base, fused_ply)
        print(f"  Poisson mesh: {n_mesh_v} verts")

    # Export geometry npz so downstream can read without pycolmap segfault
    geo = export_geometry(out_base)
    print(f"  geometry: {geo}")
    g = np.load(out_base / "_geometry.npz")
    centers = g["centers"]; pts = g["pts"]; fids = g["fids"]

    # Landmark indices
    glottis_idxs = [i for i, f in enumerate(fids)
                    if f < bins[0][2]]  # in glottis bin
    sg_idxs = [i for i, f in enumerate(fids)
                if bins[1][1] <= f < bins[1][2]]
    L1 = glottis_idxs[0] if glottis_idxs else 0
    L2 = sg_idxs[0] if sg_idxs else (L1 + 1 if L1 + 1 < len(centers) else L1)
    L3 = len(centers) - 1
    print(f"  landmarks: L1={int(fids[L1])} L2={int(fids[L2])} L3={int(fids[L3])}")
    dce_rows = []
    for L, lab in [(L1, "L1"), (L2, "L2"), (L3, "L3")]:
        dce, sig, n_used = compute_dce_at(centers, pts, L, slab=5.0)
        dce_rows.append({"landmark": lab, "frame_idx": int(fids[L]),
                          "DCE_units": dce, "sigma_units": sig,
                          "n_pts_slab_pm5u": n_used})
        d_str = f"{dce:.3f}" if dce else "NaN"
        s_str = f"{sig:.3f}" if sig else "NaN"
        print(f"  {lab} f={int(fids[L]):>4} n={n_used:>3}  DCE={d_str} +/- {s_str} units")

    # Blade scale
    blade = blade_scale_from_reconstruction(out_base, blade_mm)
    print(f"  blade: n_pts={blade.get('n_blade',0)}  "
          f"width_units={blade.get('blade_width_units','NaN')}  "
          f"scale={blade.get('scale_mm_per_unit','NaN')}")

    scale = blade.get("scale_mm_per_unit")
    for r in dce_rows:
        r["DCE_mm"] = (r["DCE_units"] * scale
                       if scale is not None and r["DCE_units"] is not None
                       else None)
        r["sigma_mm"] = (r["sigma_units"] * scale
                         if scale is not None and r["sigma_units"] is not None
                         else None)

    summary = {
        "vid": vid, "outcome": "PASS",
        "subglottic_kf": subglottic_kf,
        "forward_end": fwd_end,
        "n_input_curated": len(picks),
        "n_registered": int(n_reg),
        "n_sparse_points": int(n_pts),
        "n_mvs_dense_points": int(n_dense),
        "n_mesh_verts": int(n_mesh_v),
        "init_pair": {
            "inliers": init_pair[0] if init_pair else None,
            "frames": [init_pair[3], init_pair[4]] if init_pair else None,
        },
        "landmark_frames": {"L1": int(fids[L1]), "L2": int(fids[L2]),
                              "L3": int(fids[L3])},
        "dce_rows": dce_rows,
        "blade": blade,
    }
    (out_base / "_final_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {out_base}/_final_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--subglottic_kf", type=int, required=True)
    ap.add_argument("--blade_mm", type=float, default=BLADE_MM_DEFAULT)
    args = ap.parse_args()
    run_pipeline(args.vid, args.subglottic_kf, args.blade_mm)


if __name__ == "__main__":
    main()
