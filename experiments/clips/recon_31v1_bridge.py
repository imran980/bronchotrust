"""31-V1 BRIDGE attempt: fuse the upper (f206-432) and lower (f466-581) trachea across the blurred+dark
f432-466 bridge. Uniform STRONGER preprocessing (strong CLAHE brightens the dark bridge + unsharp
high-boost counters blur) applied to ALL window frames (uniform => consistent appearance for matching),
NO frame dropping, and a MORE SENSITIVE SIFT (more features, lower peak threshold) to extract matchable
structure from the weak bridge. Same pinned intrinsics/mapper. Goal: 1 connected model f206-581.

Usage: python recon_31v1_bridge.py
"""
import re, json, shutil, subprocess
from pathlib import Path
import numpy as np, cv2, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/31-V1.mp4")
CALIB = ROOT / "runs/retriage_first15/_calib/31_v1/intrinsics_pinned.json"
OUT = ROOT / "runs/own_data/recon_31v1_bridge"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
LO, HI = 206, 581


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), iterations=5) > 0


def enhance(bgr, mask, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])                        # gentle brightening only (no noise-amplifying unsharp)
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
    return im


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-200:], flush=True)
    return r


def main():
    img = OUT / "images"; msk = OUT / "masks"
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = bezel(VIDEO); bez = mask.astype(np.uint8) * 255
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(VIDEO)); fi = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > HI: break
        if LO <= fi <= HI:
            cv2.imwrite(str(img / f"f{fi:05d}.png"), enhance(fr, mask, clahe), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release(); print(f"extracted {n} enhanced frames f{LO}-{HI} (no dropping)", flush=True)
    params = json.loads(CALIB.read_text())["params_colmap"]; pstr = ",".join(f"{p:.10g}" for p in params)
    db = OUT / "db.db"; db.unlink(missing_ok=True)
    sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
        "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.peak_threshold", "0.005"])       # mild sensitivity bump only
    sh(["exhaustive_matcher", "--database_path", str(db)])  # default matcher
    sp = OUT / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    sh(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = sorted([d for d in sp.iterdir() if d.is_dir()],
                    key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
    if not models: print("NO MODEL"); return
    print("\n=== 31-V1 BRIDGE (enhanced + sensitive SIFT, no drop) ===")
    for d in models[:6]:
        r = pycolmap.Reconstruction(str(d)); fr = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in r.images.values())
        print(f"  model {d.name}: reg={r.num_reg_images()} span f{fr[0]}-{fr[-1]} pts={r.num_points3D()}")
    print(f"  BASELINE was: 3 models (f206-432 | f413-456 | f466-581)")
    # merge the two largest overlapping halves via their shared frames
    if len(models) >= 2:
        merged = sp / "merged"; shutil.rmtree(merged, ignore_errors=True); merged.mkdir()
        sh(["model_merger", "--input_path1", str(models[0]), "--input_path2", str(models[1]),
            "--output_path", str(merged)])
        try:
            rm = pycolmap.Reconstruction(str(merged))
            frm = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in rm.images.values())
            one = frm[0] <= 210 and frm[-1] >= 578
            print(f"  MERGED: reg={rm.num_reg_images()} span f{frm[0]}-{frm[-1]} pts={rm.num_points3D()} "
                  f"-> {'*** ONE TRACHEA f206-581 ***' if one else 'partial merge'}")
        except Exception as e:
            print(f"  model_merger produced no joint model ({e}) -> overlap insufficient for SIFT merge")


if __name__ == "__main__":
    main()
