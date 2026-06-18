"""Preprocessing ablation on 7-V1: raw / CLAHE / grayscale-CLAHE / specular-masked
/ sharpened / higher-resolution. For each: verified matches/pair + sparse points.

Same subset of frames (f110-450 stride 2), same bezel mask, same COLMAP sequential
matching + mapper. Only the preprocessing (and SIFT max_image_size for hires) varies.
Run in depth-eval env (uses colmap-cuda binary)."""
from __future__ import annotations
import json, sqlite3, shutil, subprocess
from pathlib import Path
import cv2, numpy as np, pycolmap

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/7-V1.MP4")
INTR = json.loads(Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib/7_v1/intrinsics_pinned.json").read_text())
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/ablation_7v1")
LO, HI, STRIDE = 110, 450, 2
DARK_L = 12.0
PARAMS = ",".join(f"{p:.10g}" for p in INTR["params_colmap"])
MAXID = 2147483647


def content_mask(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4)
    return m > 0


def extract_raw(mask):
    raw = OUT / "_raw"; raw.mkdir(parents=True, exist_ok=True)
    for p in raw.glob("*.png"): p.unlink()
    cap = cv2.VideoCapture(str(DATA)); fi = 0; vc = 0; names = []
    while True:
        ok, fr = cap.read()
        if not ok or fi > HI: break
        if LO <= fi <= HI:
            L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255
            if L >= DARK_L:
                if vc % STRIDE == 0:
                    cv2.imwrite(str(raw / f"f{fi:05d}.png"), fr, [cv2.IMWRITE_PNG_COMPRESSION, 1]); names.append(f"f{fi:05d}.png")
                vc += 1
        fi += 1
    cap.release()
    return raw, sorted(names)


def clahe_l(img, c):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run(cmd):
    return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def db_match_stats(db):
    c = sqlite3.connect(db); cur = c.cursor()
    kp = np.array([r[0] for r in cur.execute("SELECT rows FROM keypoints").fetchall()])
    ver = np.array([r[0] for r in cur.execute("SELECT rows FROM two_view_geometries WHERE rows>0").fetchall()] or [0])
    c.close()
    return (float(np.median(kp)), int((ver > 0).sum()), float(ver.mean()), float(np.median(ver)))


def variant(name, raw, names, mask, maxsz=1600, mode="raw"):
    vd = OUT / name; img = vd / "images"; msk = vd / "masks"
    for d in (img, msk):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    bez = (mask.astype(np.uint8)) * 255
    for nm in names:
        im = cv2.imread(str(raw / nm)); spec = None
        if mode == "clahe": im = clahe_l(im, c)
        elif mode == "gray_clahe":
            g = c.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)); im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif mode == "sharpen":
            im = cv2.addWeighted(im, 1.6, cv2.GaussianBlur(im, (0, 0), 3), -0.6, 0)
        elif mode == "specular":
            g0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); spec = cv2.dilate((g0 > 235).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
            im = clahe_l(im, c)
        im[~mask] = 0
        cv2.imwrite(str(img / nm), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        m = bez.copy()
        if spec is not None: m[spec > 0] = 0
        cv2.imwrite(str(msk / (nm + ".png")), m, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    db = vd / "db.db"
    if db.exists(): db.unlink()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", PARAMS,
         "--SiftExtraction.max_image_size", str(maxsz)])
    run(["sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", "30", "--SequentialMatching.quadratic_overlap", "1"])
    kp_med, n_ver, ver_mean, ver_med = db_match_stats(db)
    sp = vd / "sparse"; sp.mkdir(exist_ok=True)
    run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_principal_point", "0"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images()) if models else None
    reg = sparse = 0
    if best is not None:
        r = pycolmap.Reconstruction(str(best)); reg = r.num_reg_images(); sparse = r.num_points3D()
    row = {"variant": name, "kp_median": round(kp_med), "verified_pairs": n_ver,
           "verified_matches_per_pair_mean": round(ver_mean, 1), "verified_matches_per_pair_median": round(ver_med),
           "n_models": len(models), "reg_images": reg, "sparse_points": sparse}
    print(f"  {name:<16} kp_med={row['kp_median']:<5} ver/pair mean={row['verified_matches_per_pair_mean']:<7} "
          f"med={row['verified_matches_per_pair_median']:<5} | models={row['n_models']} reg={reg} SPARSE={sparse}", flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = content_mask(DATA)
    raw, names = extract_raw(mask)
    print(f"subset: {len(names)} frames f{LO}-{HI} stride {STRIDE}\n")
    specs = [("raw", 1600, "raw"), ("clahe", 1600, "clahe"), ("gray_clahe", 1600, "gray_clahe"),
             ("specular_masked", 1600, "specular"), ("sharpened", 1600, "sharpen"), ("hires", 1920, "raw")]
    rows = []
    for nm, sz, mode in specs:
        rows.append(variant(nm, raw, names, mask, sz, mode))
    (OUT / "ablation.json").write_text(json.dumps({"n_frames": len(names), "rows": rows}, indent=2))
    print(f"\n{'variant':<16}{'kp_med':>8}{'ver/pair_mean':>15}{'ver/pair_med':>14}{'sparse':>9}{'reg':>6}")
    for r in rows:
        print(f"{r['variant']:<16}{r['kp_median']:>8}{r['verified_matches_per_pair_mean']:>15}{r['verified_matches_per_pair_median']:>14}{r['sparse_points']:>9}{r['reg_images']:>6}")
    print(f"\nsaved {OUT/'ablation.json'}")


if __name__ == "__main__":
    main()
