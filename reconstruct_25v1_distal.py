"""Targeted re-reconstruction of the 25_V1 DISTAL subglottis to close the residual arc gap.
Denser + sharpness-filtered distal frames (slightly wider connected window), CLAHE, PINNED OPENCV
intrinsics, EXHAUSTIVE matching, dense MVS, and BOTH geometric + photometric fusion. Frames come
from the exact extracted PNGs on disk (runs/batch4/25-V1/images) -> no POS_FRAMES drift; one
connected model. depth-eval env drives; colmap-cuda binary for SfM/MVS.
Usage: python reconstruct_25v1_distal.py [prep|colmap|all]
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2, re

ROOT = Path("/home/mi3dr/projects/bronchotrust")
SRC = ROOT / "runs/batch4/25-V1/images"
WORK = ROOT / "runs/barbour30/airwayfit/recon_25v1_distal"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
# pinned 25_V1 OPENCV intrinsics (from runs/batch4/25-V1/sparse/0)
PARAMS = [847.470798, 846.672489, 945.047446, 536.649292, -0.085283, -0.071004, 0.0, 0.0]
WIN = (390, 470)              # slightly wider than the accepted distal window 405-452, still connected
MIN_SHARP = 1100             # drop the few blurriest frames


def prep():
    img_dir = WORK / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.png"): p.unlink()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kept, dropped = [], []
    for f in sorted(SRC.glob("*.png")):
        idx = int(re.search(r"f(\d+)", f.name).group(1))
        if not (WIN[0] <= idx <= WIN[1]): continue
        bgr = cv2.imread(str(f)); g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        if sharp < MIN_SHARP: dropped.append((idx, round(sharp))); continue
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_dir / f.name), out, [cv2.IMWRITE_PNG_COMPRESSION, 1]); kept.append(idx)
    (WORK / "frames.json").write_text(json.dumps({"kept": kept, "dropped": dropped, "window": WIN}, indent=2))
    print(f"prep: kept {len(kept)} frames ({min(kept)}-{max(kept)}), dropped {len(dropped)} blurry {dropped}", flush=True)
    return kept


def sh(cmd): return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def ply_n(p):
    if not Path(p).exists(): return 0
    with open(p, "rb") as fh:
        for _ in range(60):
            ln = fh.readline().decode("ascii", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
            if ln.startswith("end_header"): return 0
    return 0


def colmap():
    import pycolmap
    img_dir = WORK / "images"; names = sorted(p.name for p in img_dir.glob("*.png"))
    params = ",".join(f"{p:.10g}" for p in PARAMS); db = WORK / "database.db"
    if db.exists(): db.unlink()
    sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", params, "--SiftExtraction.max_image_size", "1920",
        "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"])
    print("  features done", flush=True)
    sh(["exhaustive_matcher", "--database_path", str(db), "--SiftMatching.guided_matching", "1"])
    print("  exhaustive matching done", flush=True)
    sparse = WORK / "sparse"; sparse.mkdir(exist_ok=True)
    sh(["mapper", "--database_path", str(db), "--image_path", str(img_dir), "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4"])
    models = sorted([d for d in sparse.iterdir() if d.is_dir()])
    if not models: print("  MAPPER FAILED: no model"); return
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rec = pycolmap.Reconstruction(str(best))
    print(f"  mapper: {len(models)} model(s); best {best.name} reg={rec.num_reg_images()}/{len(names)} sparse={rec.num_points3D()}", flush=True)
    if best.name != "0":
        (sparse / "0").mkdir(exist_ok=True)
        for f in best.glob("*"): shutil.copy(str(f), str(sparse / "0" / f.name))
    dense = WORK / "dense"; dense.mkdir(exist_ok=True)
    sh(["image_undistorter", "--image_path", str(img_dir), "--input_path", str(sparse / "0"),
        "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1920"])
    sh(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1400"])
    print("  patch_match_stereo done", flush=True)
    out = {}
    for typ in ("geometric", "photometric"):
        fp = dense / f"fused_{typ}.ply"
        sh(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
            "--input_type", typ, "--output_path", str(fp),
            "--StereoFusion.min_num_pixels", "3"])
        out[typ] = ply_n(fp); print(f"  fused {typ}: {out[typ]} pts", flush=True)
    summ = {"window": WIN, "n_frames": len(names), "n_registered": rec.num_reg_images(),
            "n_sparse": rec.num_points3D(), "fused_geometric": out.get("geometric"),
            "fused_photometric": out.get("photometric"), "camera_params": list(next(iter(rec.cameras.values())).params)}
    (WORK / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== 25_V1 DISTAL RE-RECON ===")
    for k in ("n_frames", "n_registered", "n_sparse", "fused_geometric", "fused_photometric"): print(f"  {k}: {summ[k]}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    WORK.mkdir(parents=True, exist_ok=True)
    if which in ("prep", "all"): prep()
    if which in ("colmap", "all"): colmap()


if __name__ == "__main__":
    main()
