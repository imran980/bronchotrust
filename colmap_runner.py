"""
COLMAP subprocess runner — invoked by reconstruct.py's COLMAPBackbone.run().

Bronchoscopy frames are low-contrast and have a dark circular bezel around
the scope's optical cone. We apply two preprocessing steps that COLMAP gets
but the foundation backbones (MASt3R / DUSt3R) do not, because:
    * CLAHE on L (LAB) lifts feature contrast on textureless mucosa.
    * A circular ROI mask tells COLMAP to ignore the bezel (no features
      detected outside the optical cone). COLMAP supports per-image masks
      via --ImageReader.mask_path with `<name>.png.png` convention (a
      binary PNG: white=valid, black=ignored).

Pipeline (sparse only — dense MVS deferred):
    1. feature_extractor (SIFT, single-camera shared intrinsics)
    2. exhaustive_matcher
    3. mapper                       -> sparse model in <out>/colmap_sparse/0/
    4. model_converter --output_type TXT for our parser to read cameras.txt
       + images.txt + points3D.txt

Outputs (six-file contract):
    poses.npz        N x 4 x 4 c2w (from images.txt, qvec/tvec are w2c -> invert)
    points.ply       sparse colored point cloud (from points3D.txt)
    intrinsics.yaml  single-camera focal + distortion COLMAP estimated
    confidence.npy   per-point reprojection-error inverse as a confidence proxy
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------- preprocessing ----------

def _circular_bezel_mask(img_gray: np.ndarray) -> np.ndarray:
    """Detect the brightest (mucosa) circular region; return uint8 0/255 mask.
    Falls back to all-white if the bezel detector fails — better to let
    COLMAP see the full frame than to mask too aggressively."""
    H, W = img_gray.shape
    # Median-blur + OTSU; the bezel is typically uniformly black.
    blur = cv2.medianBlur(img_gray, 7)
    _, bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if n <= 1:
        return np.full((H, W), 255, np.uint8)
    # Pick the largest non-background component
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < 0.2 * H * W:
        return np.full((H, W), 255, np.uint8)  # bezel detection unconvincing
    mask = (labels == largest).astype(np.uint8) * 255
    # Light erode so feature support stays clear of the bezel edge.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.erode(mask, kern, iterations=1)
    return mask


def _clahe_rgb(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge((L, a, b)), cv2.COLOR_LAB2BGR)


def _preprocess(frames_in: Path, frames_out: Path, masks_out: Path) -> int:
    """Write CLAHE'd frames and bezel masks. Returns frame count."""
    frames_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(frames_in.iterdir()):
        if src.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = _circular_bezel_mask(gray)
        img_eq = _clahe_rgb(img)
        cv2.imwrite(str(frames_out / src.name), img_eq)
        # COLMAP wants '<image_name>.png' for masks (e.g. 000001.png.png).
        cv2.imwrite(str(masks_out / (src.name + ".png")), mask)
        n += 1
    return n


# ---------- output parsing ----------

def _qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _read_colmap_txt(sparse_dir: Path):
    """Parse cameras.txt, images.txt, points3D.txt."""
    cameras = {}
    for line in (sparse_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model, W, H = parts[1], int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        cameras[cam_id] = (model, W, H, params)
    images = []  # (name, R_w2c, t_w2c, cam_id)
    lines = (sparse_dir / "images.txt").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        parts = line.split()
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        t = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        cam_id = int(parts[8])
        name = parts[9]
        images.append((name, _qvec2R(q), t, cam_id))
        i += 2  # skip the 2D-points line
    images.sort(key=lambda r: r[0])
    pts = []
    for line in (sparse_dir / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # ID, X, Y, Z, R, G, B, ERROR, TRACK[]
        xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
        err = float(parts[7])
        pts.append((xyz, rgb, err))
    return cameras, images, pts


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement
    pcd = np.empty(len(points), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    pcd["x"], pcd["y"], pcd["z"] = points[:, 0], points[:, 1], points[:, 2]
    pcd["red"], pcd["green"], pcd["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(pcd, "vertex")], text=False).write(str(path))


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--colmap", default=shutil.which("colmap") or "colmap")
    ap.add_argument("--single_camera", type=int, default=1,
                    help="Treat all frames as one camera (intrinsics shared).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "_work"
    work.mkdir(exist_ok=True)
    frames_pp = work / "frames"
    masks_pp = work / "masks"
    db_path = work / "database.db"
    if db_path.exists():
        db_path.unlink()
    sparse_dir = work / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    sparse_txt = work / "sparse_txt"

    t0 = time.time()
    n_frames = _preprocess(args.frames_dir, frames_pp, masks_pp)
    print(f"[colmap_runner] preprocessed {n_frames} frames in {time.time() - t0:.1f}s")
    if n_frames == 0:
        raise FileNotFoundError(f"No frames found in {args.frames_dir}")

    def run(stage_args):
        print(f"[colmap_runner] {' '.join(stage_args[:3])}")
        subprocess.run([args.colmap] + stage_args, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # 1. SIFT feature extraction (CPU is fine, GPU may not be wired in conda build)
    t0 = time.time()
    run(["feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(frames_pp),
         "--ImageReader.mask_path", str(masks_pp),
         "--ImageReader.single_camera", str(args.single_camera),
         "--ImageReader.camera_model", "SIMPLE_RADIAL",
         "--SiftExtraction.use_gpu", "0"])
    print(f"[colmap_runner] feature_extractor {time.time() - t0:.1f}s")

    # 2. Exhaustive matcher (small N from subsampled videos; OK)
    t0 = time.time()
    run(["exhaustive_matcher",
         "--database_path", str(db_path),
         "--SiftMatching.use_gpu", "0"])
    print(f"[colmap_runner] exhaustive_matcher {time.time() - t0:.1f}s")

    # 3. Mapper -> sparse model
    t0 = time.time()
    run(["mapper",
         "--database_path", str(db_path),
         "--image_path", str(frames_pp),
         "--output_path", str(sparse_dir)])
    print(f"[colmap_runner] mapper {time.time() - t0:.1f}s")

    # Mapper writes sparse/0/, sparse/1/, ... — pick the largest.
    submodels = sorted(sparse_dir.iterdir())
    if not submodels:
        print("[colmap_runner] no sparse model produced — registration failed.")
        # Still produce empty contract files so downstream parsing doesn't crash.
        np.savez(args.out_dir / "poses.npz",
                 c2w=np.zeros((0, 4, 4)), timestamps=np.zeros((0,)))
        np.save(args.out_dir / "confidence.npy", np.zeros((0,), dtype=np.float32))
        return 0
    sizes = [(d, sum(1 for _ in (d / "images.bin").parent.glob("*"))) for d in submodels]
    best = sorted(submodels, key=lambda d: (d / "images.bin").stat().st_size if (d / "images.bin").exists() else 0)[-1]
    print(f"[colmap_runner] best sparse submodel: {best.name}")

    # 4. Convert to TXT for parsing
    sparse_txt.mkdir(exist_ok=True)
    run(["model_converter",
         "--input_path", str(best),
         "--output_path", str(sparse_txt),
         "--output_type", "TXT"])

    cameras, images, pts = _read_colmap_txt(sparse_txt)
    print(f"[colmap_runner] registered {len(images)} of {n_frames} frames; "
          f"{len(pts)} sparse points")

    # Build six-file contract.
    poses = np.zeros((len(images), 4, 4), dtype=np.float64)
    for k, (name, R_w2c, t_w2c, cam_id) in enumerate(images):
        # COLMAP stores world-to-camera. Invert for c2w.
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        poses[k, :3, :3] = R_c2w
        poses[k, :3, 3] = t_c2w
        poses[k, 3, 3] = 1.0
    timestamps = np.arange(len(images), dtype=np.float64)
    np.savez(args.out_dir / "poses.npz", c2w=poses, timestamps=timestamps)

    if pts:
        xyz = np.array([p[0] for p in pts], dtype=np.float32)
        rgb = np.array([p[1] for p in pts], dtype=np.uint8)
        err = np.array([p[2] for p in pts], dtype=np.float32)
        _save_ply(args.out_dir / "points.ply", xyz, rgb)
        # Confidence proxy: 1 / (1 + reprojection_error). Higher is better.
        conf = (1.0 / (1.0 + err)).astype(np.float32)
        np.save(args.out_dir / "confidence.npy", conf)

    if cameras:
        cam_id = next(iter(cameras))
        model, W, H, params = cameras[cam_id]
        (args.out_dir / "intrinsics.yaml").write_text(
            f"# COLMAP-estimated intrinsics (model={model}).\n"
            f"width: {W}\nheight: {H}\n"
            f"calibration: {params}\n"
        )

    print(f"[colmap_runner] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
