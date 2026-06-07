"""Pinned-vs-unpinned intrinsics diagnostic on untrimmed 2_v2.

Variations:
  --pinned   : OPENCV camera with calibrated fx,fy,cx,cy,k1,k2,p1=p2=0,
               BA refine_focal_length=False, refine_extra_params=False,
               refine_principal_point=False.
  --unpinned : OPENCV camera with same calibrated initial values, but BA
               refine_focal_length=True, refine_extra_params=True (default
               COLMAP behaviour).
  --out <dir>: output directory. Reconstruction, MVS, mesh all go inside.

Pipeline identical to barbour_untrimmed_2v2.py minus the DCE / hero phases
(this script stops after Poisson mesh and saves results.json + sparse TXT
+ mesh PLY + calibration JSON).
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
VID = "2_v2"

CURATED_FRAMES = (
    [0, 8, 16, 24, 32, 40] +
    [52, 58, 64, 70] +
    [88, 100, 112, 124, 136, 148, 160, 172, 184, 196] +
    [2010, 2060, 2110, 2160, 2210, 2260, 2310, 2360, 2410, 2460]
)
GLOTTIC_FRAMES = set(CURATED_FRAMES[:10])  # glottic + blade-visible -> good init pool


def video_path() -> Path:
    for ext in ("mp4", "MP4"):
        p = SRC / f"{VID}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(VID)


def preprocess(out_base: Path):
    """Raw color frames + bezel mask, no CLAHE, no specular."""
    intr = json.loads(
        (PROJECT / f"runs/barbour/{VID}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]
    r_fov = float(intr["fov_radius"])
    Wimg, Himg = intr["image_size"]
    img_dir = out_base / "images"
    msk_dir = out_base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    yy, xx = np.indices((Himg, Wimg), dtype=np.float32)
    bezel = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_fov - 1) ** 2
    cap = cv2.VideoCapture(str(video_path()))
    extracted = []
    for fi in CURATED_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        frame[~bezel] = 0
        name = f"{VID}_f{fi:06d}.png"
        cv2.imwrite(str(img_dir / name), frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (bezel.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        extracted.append({"frame_idx": fi, "filename": name})
    cap.release()
    print(f"  preprocessed {len(extracted)} / {len(CURATED_FRAMES)} frames")
    return extracted, intr


def cam_params_str(intr):
    """OPENCV params (fx,fy,cx,cy,k1,k2,p1,p2). p1=p2=0 because calibration
    only fitted k1,k2."""
    params = [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
              intr["k1"], intr["k2"], 0.0, 0.0]
    return ",".join(f"{p:.10g}" for p in params), params


def find_glottic_init(db_path: Path, glottic_files: set, min_gap: int = 12):
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
    best_baseline = None
    best_anywhere = None
    for pair_id, rows in pairs:
        i1 = pair_id // 2147483647
        i2 = pair_id - i1 * 2147483647
        n1 = id_name.get(i1)
        n2 = id_name.get(i2)
        if n1 not in glottic_files or n2 not in glottic_files:
            continue
        gap = abs(fi(n1) - fi(n2))
        if best_anywhere is None or rows > best_anywhere[0]:
            best_anywhere = (rows, i1, i2, n1, n2, gap)
        if gap >= min_gap and (best_baseline is None or rows > best_baseline[0]):
            best_baseline = (rows, i1, i2, n1, n2, gap)
    pick = best_baseline if best_baseline is not None else best_anywhere
    return pick[:5] if pick else None


def run_hloc_extract_match(out_base, image_list, intr):
    from hloc import extract_features, match_features, pairs_from_exhaustive
    images_dir = out_base / "images"
    sfm_pairs = out_base / "pairs-exhaustive.txt"
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    print("  HLoc SuperPoint extraction ...")
    feats = extract_features.main(
        feature_conf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    print("  HLoc LightGlue matching ...")
    matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], out_base)
    return feats, matches, sfm_pairs


def reconstruct(out_base, image_list, intr, feats, matches, sfm_pairs,
                refine_focal: bool):
    """Pass 1: default-init recon (populates DB)."""
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
        "multiple_models": True, "max_num_models": 50,
        "min_model_size": 3,
        "ba_refine_focal_length": refine_focal,
        "ba_refine_extra_params": refine_focal,
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
    print(f"  Pass 1 (refine_focal={refine_focal}) ...")
    model = hloc_recon.main(
        sfm_dir, images_dir, sfm_pairs, feats, matches,
        image_list=image_list,
        camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
        image_options=image_options,
        verbose=False, mapper_options=mapper_options)
    n_reg_pre = model.num_reg_images() if model else 0
    print(f"  Pass 1 registered: {n_reg_pre}")

    # Find glottic init pair from DB and pass 2 with forced init
    init = find_glottic_init(sfm_dir / "database.db",
                              {p for p in
                                {f"{VID}_f{fi:06d}.png" for fi in
                                  CURATED_FRAMES[:10]}})
    if init is None:
        print("  no glottic init found; keeping pass 1")
        return model.num_reg_images(), model.num_points3D(), model
    n_inl, i1, i2, n1, n2 = init
    print(f"  Pass 2 init: {n1} <-> {n2}  inliers={n_inl}")

    # Snapshot pass 1
    pass1_dir = out_base / "sfm_pass1"
    if pass1_dir.exists():
        shutil.rmtree(pass1_dir)
    pass1_dir.mkdir()
    for fn in ["cameras.bin", "images.bin", "points3D.bin",
                "frames.bin", "rigs.bin"]:
        p = sfm_dir / fn
        if p.exists():
            shutil.copy(str(p), str(pass1_dir / fn))

    # Clear sparse + rerun with forced init
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
    opts.ba_refine_focal_length = refine_focal
    opts.ba_refine_extra_params = refine_focal
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
        if not recons:
            print("  Pass 2 produced no models -> restoring pass 1")
            for fn in ["cameras.bin", "images.bin", "points3D.bin",
                        "frames.bin", "rigs.bin"]:
                src = pass1_dir / fn
                if src.exists():
                    shutil.copy(str(src), str(sfm_dir / fn))
            return n_reg_pre, model.num_points3D(), model
        best_id = max(recons.keys(),
                       key=lambda k: recons[k].num_reg_images())
        best = recons[best_id]
        best.write_binary(str(sfm_dir))
        n_reg_p2 = best.num_reg_images()
        if n_reg_p2 >= n_reg_pre:
            print(f"  Pass 2 registered: {n_reg_p2} -> using")
            return n_reg_p2, best.num_points3D(), best
        else:
            print(f"  Pass 2 ({n_reg_p2}) < pass 1 ({n_reg_pre}); restoring")
            for fn in ["cameras.bin", "images.bin", "points3D.bin",
                        "frames.bin", "rigs.bin"]:
                src = pass1_dir / fn
                if src.exists():
                    shutil.copy(str(src), str(sfm_dir / fn))
            return n_reg_pre, model.num_points3D(), model
    except Exception as e:
        print(f"  Pass 2 error: {e}; using pass 1")
        return n_reg_pre, model.num_points3D(), model


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
                    n_dense = int(line.split()[-1])
                    break
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


def export_sparse_text_and_meta(out_base: Path, intr, refine_focal: bool,
                                  n_reg: int, n_pts: int,
                                  n_dense: int, n_mesh_v: int):
    sfm = out_base / "sfm"
    text_dir = out_base / "sfm_text"
    text_dir.mkdir(exist_ok=True)
    subprocess.run([str(COLMAP), "model_converter",
                     "--input_path", str(sfm),
                     "--output_path", str(text_dir),
                     "--output_type", "TXT"], capture_output=True)

    # Mean reprojection error & camera params via pycolmap subprocess
    sub = r"""
import pycolmap, json, sys, numpy as np
from pathlib import Path
rec = pycolmap.Reconstruction(sys.argv[1])
mr_err = float(rec.compute_mean_reprojection_error()) if hasattr(rec, "compute_mean_reprojection_error") else None
if mr_err is None:
    errs = []
    for p in rec.points3D.values():
        errs.append(float(p.error))
    mr_err = float(np.mean(errs)) if errs else None
out = {"mean_reprojection_error_px": mr_err}
for cam in rec.cameras.values():
    out["camera_model"] = cam.model.name if hasattr(cam.model, 'name') else str(cam.model)
    out["camera_params"] = list(cam.params)
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(sfm)],
                          capture_output=True, text=True)
    cam_info = json.loads(res.stdout.strip().splitlines()[-1])

    # Mesh AABB extent
    sub2 = """
import open3d as o3d, sys, json, numpy as np
m = o3d.io.read_triangle_mesh(sys.argv[1])
v = np.asarray(m.vertices)
ext = (v.max(0) - v.min(0)).tolist() if len(v) > 0 else None
print(json.dumps({"bbox_extent": ext, "bbox_diag": float(np.linalg.norm(v.max(0)-v.min(0))) if len(v)>0 else None,
                   "n_verts": len(v)}))
"""
    res = subprocess.run([sys.executable, "-c", sub2,
                           str(out_base / "dense" / "mesh_poisson.ply")],
                          capture_output=True, text=True)
    mesh_info = json.loads(res.stdout.strip().splitlines()[-1])

    diag = {
        "mode": "unpinned" if refine_focal else "pinned",
        "n_registered_images": n_reg,
        "n_sparse_points": n_pts,
        "n_dense_points": n_dense,
        "n_mesh_verts": n_mesh_v,
        "final_camera_model": cam_info["camera_model"],
        "final_camera_params": cam_info["camera_params"],
        "final_fx": (cam_info["camera_params"][0]
                      if cam_info["camera_model"] == "OPENCV" else None),
        "final_fy": (cam_info["camera_params"][1]
                      if cam_info["camera_model"] == "OPENCV" else None),
        "final_k1": (cam_info["camera_params"][4]
                      if cam_info["camera_model"] == "OPENCV"
                      and len(cam_info["camera_params"]) >= 5 else None),
        "final_k2": (cam_info["camera_params"][5]
                      if cam_info["camera_model"] == "OPENCV"
                      and len(cam_info["camera_params"]) >= 6 else None),
        "calibrated_fx": intr["fx"],
        "calibrated_fy": intr["fy"],
        "calibrated_k1": intr["k1"],
        "calibrated_k2": intr["k2"],
        "calibration_rms_px": intr["rms_reproj_px"],
        "mean_reprojection_error_px": cam_info["mean_reprojection_error_px"],
        "mesh_aabb_extent": mesh_info["bbox_extent"],
        "mesh_bbox_diag": mesh_info["bbox_diag"],
    }
    (out_base / "results.json").write_text(json.dumps(diag, indent=2))
    # Save calibration JSON
    (out_base / "calibration.json").write_text(json.dumps({
        "fx": intr["fx"], "fy": intr["fy"],
        "cx": intr["cx"], "cy": intr["cy"],
        "k1": intr["k1"], "k2": intr["k2"],
        "p1": 0.0, "p2": 0.0,
        "rms_reproj_px": intr["rms_reproj_px"],
        "image_size": intr["image_size"],
        "model": "OPENCV (k1,k2,p1=0,p2=0)",
    }, indent=2))
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--refine_intrinsics", action="store_true",
                    help="If set, BA refines focal + distortion (unpinned).")
    args = ap.parse_args()
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    for f in out_base.glob("*"):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)
    print(f"\n=== {VID} {'UNPINNED' if args.refine_intrinsics else 'PINNED'} -> {out_base} ===")
    extracted, intr = preprocess(out_base)
    if intr["rms_reproj_px"] >= 0.5:
        print(f"  CALIBRATION RMS {intr['rms_reproj_px']:.3f} >= 0.5 -- STOP")
        return 1
    image_list = sorted([e["filename"] for e in extracted])
    feats, matches, sfm_pairs = run_hloc_extract_match(out_base, image_list, intr)
    n_reg, n_pts, model = reconstruct(out_base, image_list, intr,
                                        feats, matches, sfm_pairs,
                                        refine_focal=args.refine_intrinsics)
    n_dense = run_mvs(out_base)
    print(f"  MVS dense: {n_dense}")
    n_mesh_v = run_poisson(out_base)
    print(f"  Poisson mesh: {n_mesh_v}")
    diag = export_sparse_text_and_meta(out_base, intr, args.refine_intrinsics,
                                          n_reg, n_pts, n_dense, n_mesh_v)
    print(f"\nresults.json: {json.dumps({k: v for k, v in diag.items() if not isinstance(v, list)}, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
