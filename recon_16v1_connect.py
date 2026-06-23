"""16_v1 lower-airway CONNECTIVITY attempt (sparse only, NO MVS).
The default batch_dense run fragmented (4 models, best 1014-1079) and dropped the
distal trachea + mainstem. Goal: bridge the carina seam into ONE model covering
distal trachea -> carina -> mainstem so a non-circular DCE ratio test is possible.
Levers for the lowest-texture video: more SIFT features (max_image_size 1920,
lower peak threshold, affine-shape + DSP), exhaustive matching, and a CONNECTIVITY-
lenient mapper (low init_min_tri_angle, low min_num_matches). KEEP ALL models and
report each model's frame coverage so we can see if trachea+mainstem ever connect.
Reuses the already-extracted CLAHE frames in recon/images. STOP after mapper.
depth-eval env (colmap-cuda binary)."""
from __future__ import annotations
import json, shutil, subprocess, re
from pathlib import Path
import cv2, numpy as np, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/ct_validation_16v1/recon_connect"
IMG_SRC = ROOT / "runs/ct_validation_16v1/recon/images"   # 441 CLAHE frames f900-1340
VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/16_v1.mp4")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
PARAMS = ",".join(f"{p:.10g}" for p in json.loads(
    (ROOT / "runs/retriage_first15/_calib/16_v1/intrinsics_pinned.json").read_text())["params_colmap"])


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4) > 0


def run(cmd):
    print(">>", cmd[0], "...", flush=True)
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-500:], flush=True)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    msk = OUT / "masks"
    if not msk.exists():
        msk.mkdir(parents=True); bez = bezel(VIDEO).astype(np.uint8) * 255
        for p in sorted(IMG_SRC.glob("f*.png")):
            cv2.imwrite(str(msk / (p.name + ".png")), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    db = OUT / "db.db"; db.unlink(missing_ok=True)
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(IMG_SRC),
         "--ImageReader.mask_path", str(msk), "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1", "--ImageReader.camera_params", PARAMS,
         "--SiftExtraction.max_image_size", "1920", "--SiftExtraction.max_num_features", "16384",
         "--SiftExtraction.peak_threshold", "0.002", "--SiftExtraction.edge_threshold", "12",
         "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"])
    run(["exhaustive_matcher", "--database_path", str(db),
         "--SiftMatching.guided_matching", "1", "--SiftMatching.max_ratio", "0.9",
         "--TwoViewGeometry.min_num_inliers", "8"])
    sp = OUT / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    run(["mapper", "--database_path", str(db), "--image_path", str(IMG_SRC), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.init_min_tri_angle", "1.5", "--Mapper.min_num_matches", "5",
         "--Mapper.abs_pose_min_num_inliers", "8", "--Mapper.filter_min_tri_angle", "1.0",
         "--Mapper.max_reg_trials", "5"])
    models = sorted([d for d in sp.iterdir() if d.is_dir()],
                    key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
    print(f"\n{len(models)} model(s):", flush=True)
    rep = []
    for m in models:
        rec = pycolmap.Reconstruction(str(m))
        frs = sorted(int(re.search(r'f(\d+)', im.name).group(1)) for im in rec.images.values())
        # coverage of the 3 anatomical zones
        tr = sum(1 for f in frs if f < 1056)        # distal trachea (above carina)
        ca = sum(1 for f in frs if 1056 <= f <= 1160)  # carina
        ms = sum(1 for f in frs if f > 1160)         # mainstem
        rep.append({"model": m.name, "reg": rec.num_reg_images(), "sparse": rec.num_points3D(),
                    "frame_min": frs[0], "frame_max": frs[-1],
                    "distal_trachea(<1056)": tr, "carina(1056-1160)": ca, "mainstem(>1160)": ms})
        print(f"  model {m.name}: reg={rec.num_reg_images():3d} sparse={rec.num_points3D():6d} "
              f"frames {frs[0]}..{frs[-1]}  [trachea={tr} carina={ca} mainstem={ms}]", flush=True)
    (OUT / "models_report.json").write_text(json.dumps(rep, indent=2))
    # flag a through-path model (covers BOTH trachea and mainstem)
    through = [r for r in rep if r["distal_trachea(<1056)"] >= 5 and r["mainstem(>1160)"] >= 5]
    print(f"\nthrough-path model (trachea+mainstem connected)?: "
          f"{'YES -> '+through[0]['model'] if through else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
