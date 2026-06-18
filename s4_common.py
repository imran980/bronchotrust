"""Shared Stage-4/5 helpers for the GlueMap-vs-COLMAP A/B on 15_v2.

Pure cv2/numpy/pycolmap/subprocess -> importable in BOTH the `depth-eval`
(COLMAP arm: hloc+pycolmap) and `gluemap` (GlueMap arm) conda envs.

Both arms consume IDENTICAL inputs: the curated frames undistorted once with
the pinned OPENCV model into PINHOLE images (K = fx,fy,cx,cy). Only the SfM
engine differs. MVS (COLMAP patch_match_stereo) is identical for both, so the
dense clouds are comparable in shape (each in its own arbitrary scene scale;
the deliverable -- % obstruction area ratio -- is scale-free).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

COLMAP_BIN = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"


def undistort_curated(stage1_img_dir, intr, mask, out_img_dir):
    """Undistort curated frames with pinned OPENCV -> PINHOLE images (newK=K)."""
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    W, H = int(intr["width"]), int(intr["height"])
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    D = np.array([intr["k1"], intr["k2"],
                  intr.get("p1", 0.0), intr.get("p2", 0.0)])
    um = cv2.undistort((mask > 0).astype(np.uint8) * 255, K, D, None, K)
    ubez = um > 127
    names = []
    for p in sorted(Path(stage1_img_dir).glob("f*.png")):
        u = cv2.undistort(cv2.imread(str(p)), K, D, None, K)
        u[~ubez] = 0
        cv2.imwrite(str(out_img_dir / p.name), u, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        names.append(p.name)
    pinhole = {"model": "PINHOLE", "width": W, "height": H,
               "fx": intr["fx"], "fy": intr["fy"],
               "cx": intr["cx"], "cy": intr["cy"],
               "params": [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]}
    cv2.imwrite(str(out_img_dir.parent / "undist_mask.png"),
                (ubez.astype(np.uint8) * 255))
    return pinhole, sorted(names)


def write_gt_model_text(out_dir, names, pinhole):
    """COLMAP TEXT model with one PINHOLE camera (for GlueMap --gt_intrinsics_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = pinhole["params"]
    W, H = pinhole["width"], pinhole["height"]
    # GlueMap's reprojection code uses camera.focal_length (single-focal only),
    # so use SIMPLE_PINHOLE with f = mean(fx, fy) (here fx~=fy to 0.05%).
    f = 0.5 * (fx + fy)
    (out_dir / "cameras.txt").write_text(
        f"# Camera list\n1 SIMPLE_PINHOLE {W} {H} {f:.10g} {cx:.10g} {cy:.10g}\n")
    lines = ["# Image list"]
    for i, nm in enumerate(names, start=1):
        lines.append(f"{i} 1 0 0 0 0 0 0 1 {nm}")
        lines.append("")  # empty 2D-point line
    (out_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "points3D.txt").write_text("# 3D point list\n")
    return out_dir


def run_mvs(sfm_dir, images_dir, dense_dir):
    """COLMAP MVS: undistort -> patch_match_stereo(geom) -> stereo_fusion."""
    dense_dir = Path(dense_dir)
    shutil.rmtree(dense_dir, ignore_errors=True)
    dense_dir.mkdir(parents=True)
    logs = {}
    r1 = subprocess.run([COLMAP_BIN, "image_undistorter",
                         "--image_path", str(images_dir),
                         "--input_path", str(sfm_dir),
                         "--output_path", str(dense_dir),
                         "--output_type", "COLMAP"],
                        capture_output=True, text=True)
    logs["undistort_rc"] = r1.returncode
    r2 = subprocess.run([COLMAP_BIN, "patch_match_stereo",
                         "--workspace_path", str(dense_dir),
                         "--workspace_format", "COLMAP",
                         "--PatchMatchStereo.geom_consistency", "true"],
                        capture_output=True, text=True)
    logs["pms_rc"] = r2.returncode
    out_ply = dense_dir / "fused.ply"
    r3 = subprocess.run([COLMAP_BIN, "stereo_fusion",
                         "--workspace_path", str(dense_dir),
                         "--workspace_format", "COLMAP",
                         "--input_type", "geometric",
                         "--output_path", str(out_ply)],
                        capture_output=True, text=True)
    logs["fusion_rc"] = r3.returncode
    n = 0
    if out_ply.exists():
        with open(out_ply, "rb") as fh:
            for _ in range(60):
                ln = fh.readline().decode("ascii", "ignore")
                if ln.startswith("element vertex"):
                    n = int(ln.split()[-1])
                    break
                if ln.startswith("end_header"):
                    break
    logs["tail_undist"] = (r1.stderr or "")[-300:]
    logs["tail_fusion"] = (r3.stderr or "")[-300:]
    return n, out_ply, logs


def cone_stats(centers, axes):
    centers = np.asarray(centers, float)
    A = np.asarray(axes, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    if len(A) < 2:
        return None
    ang = np.degrees(np.arccos(np.clip(A @ A.T, -1, 1)))
    mv = A.mean(0)
    mv /= (np.linalg.norm(mv) + 1e-12)
    cc = centers - centers.mean(0)
    al = cc @ mv
    lat = np.linalg.norm(cc - np.outer(al, mv), axis=1)
    asp = float(al.max() - al.min())
    return {"n": int(len(A)), "cone_total_extent_deg": float(ang.max()),
            "along_axis_span": asp,
            "lateral_over_along_ratio": (2 * float(lat.max()) / asp)
            if asp > 1e-6 else float("inf")}


def analyze(sfm_dir, seg_of):
    """Per-registered-image pose + cone + per-point triangulation angles."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(sfm_dir))
    reg, cc = [], {}
    for img in rec.images.values():
        m = re.search(r"f(\d+)\.png", img.name)
        fi = int(m.group(1)) if m else -1
        M = np.array(img.cam_from_world().matrix())
        R, t = M[:3, :3], M[:3, 3]
        C = -R.T @ t
        cc[img.image_id] = C
        reg.append({"frame_idx": fi, "segment": seg_of.get(fi, "?"),
                    "image_id": int(img.image_id), "C": C.tolist(),
                    "axis": (R.T @ np.array([0, 0, 1.0])).tolist()})
    reg.sort(key=lambda r: r["frame_idx"])
    errs = [float(p.error) for p in rec.points3D.values()]
    tri = []
    for p in rec.points3D.values():
        obs = [el.image_id for el in p.track.elements if el.image_id in cc]
        if len(obs) < 2:
            continue
        X = np.array(p.xyz)
        V = np.array([cc[i] - X for i in obs])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        tri.append(float(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max()))
    cam = next(iter(rec.cameras.values()))
    return {
        "n_registered": len(reg), "registered": reg,
        "n_sparse": int(rec.num_points3D()),
        "mean_reproj_px": float(np.mean(errs)) if errs else None,
        "camera_model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
        "camera_params": list(cam.params),
        "tri_angle_median": float(np.median(tri)) if tri else None,
        "tri_angle_p75": float(np.percentile(tri, 75)) if tri else None,
        "tri_angle_max": float(np.max(tri)) if tri else None,
        "cone_all": cone_stats([r["C"] for r in reg], [r["axis"] for r in reg]),
        "cone_subcord": cone_stats(
            [r["C"] for r in reg if r["segment"].startswith("subglot")],
            [r["axis"] for r in reg if r["segment"].startswith("subglot")]),
    }
