"""TEST: does cutting 7-V1's pullback/reversal frames into ONE smooth forward run improve the recon?
Selects the monotonic-forward ENVELOPE of frames from the existing 7-V1 poses (drops frames that
backtrack), re-extracts EXACTLY those video frames with the SAME preprocessing as batch_dense.py
(bezel + CLAHE + dark-skip), and runs the IDENTICAL COLMAP SfM (pinned OPENCV, exhaustive, same mapper).
Sparse only here. Reports selection + registration + tube-shape (roundness s2/s3, angular coverage)
so we can compare against the full-window baseline before spending MVS.

Usage: python smooth_run_7v1.py [select|recon|eval]
"""
from __future__ import annotations
import sys, re, json, shutil, subprocess
from pathlib import Path
import numpy as np, cv2, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
BASE = ROOT / "runs/own_data/recon_7v1_verify/dense/sparse"          # full-window baseline
OUT = ROOT / "runs/own_data/recon_7v1_smooth"
VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/7-V1.MP4")
CALIB = ROOT / "runs/retriage_first15/_calib/7_v1/intrinsics_pinned.json"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
DARK_L = 12.0
TOL_FRAC = 0.02                                                       # allow tiny dips in the envelope


def fidx(name): return int(re.search(r"(\d+)", name).group(1))


def select():
    rec = pycolmap.Reconstruction(str(BASE))
    rows = sorted(([fidx(im.name), np.asarray(im.projection_center())] for im in rec.images.values()),
                  key=lambda r: r[0])
    fr = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    a = Cc @ Vt[0]
    if a[-1] < a[0]: a = -a
    tol = TOL_FRAC * (a.max() - a.min())
    keep = []; run_max = -1e18
    for i in range(len(a)):
        if a[i] >= run_max - tol:
            keep.append(i); run_max = max(run_max, a[i])
    kept_fr = fr[keep]
    print(f"baseline: {len(fr)} registered frames f{fr.min()}-{fr.max()}")
    print(f"monotonic-envelope keeps {len(kept_fr)} frames ({len(kept_fr)/len(fr):.0%}), "
          f"span f{kept_fr.min()}-{kept_fr.max()}")
    return set(int(x) for x in kept_fr)


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4) > 0


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  COLMAP ERR", cmd[0], r.stderr[-300:], flush=True)
    return r


def recon(keep):
    img = OUT / "images"; msk = OUT / "masks"
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    params = json.loads(CALIB.read_text())["params_colmap"]; pstr = ",".join(f"{p:.10g}" for p in params)
    mask = bezel(VIDEO); bez = mask.astype(np.uint8) * 255
    c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(VIDEO)); fi = 0; n = 0; hi = max(keep)
    while True:
        ok, fr = cap.read()
        if not ok or fi > hi: break
        if fi in keep and cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
            im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
            cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release(); print(f"extracted {n} CLAHE frames (selected)", flush=True)
    db = OUT / "db.db"; db.unlink(missing_ok=True)
    sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
        "--SiftExtraction.max_image_size", "1600"])
    sh(["exhaustive_matcher", "--database_path", str(db)])
    sp = OUT / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    sh(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    if not models: print("NO MODEL"); return
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rb = pycolmap.Reconstruction(str(best))
    print(f"mapper: {len(models)} model(s), best reg={rb.num_reg_images()}/{n}", flush=True)
    if best.name != "0":
        (best).rename(sp / "0_keep")
        for d in sp.iterdir():
            if d.is_dir() and d.name != "0_keep": shutil.rmtree(d)
        (sp / "0_keep").rename(sp / "0")


def tube_shape(sparse_dir, label):
    rec = pycolmap.Reconstruction(str(sparse_dir))
    P = np.array([p.xyz for p in rec.points3D.values()])
    Pc = P - P.mean(0); _, sv, Vt = np.linalg.svd(Pc, full_matrices=False)
    elong = sv[0] / sv[1]; roundness = sv[1] / sv[2]
    perp = Pc - np.outer(Pc @ Vt[0], Vt[0]); az = np.degrees(np.arctan2(perp @ Vt[2], perp @ Vt[1]))
    binc = np.histogram(az, bins=36, range=(-180, 180))[0]
    pop = (binc > 0).mean(); dens_half = binc[np.argsort(binc)[::-1][:18]].sum() / binc.sum()
    print(f"[{label:16s}] reg={rec.num_reg_images():3d} pts={len(P):6d} | elong s1/s2={elong:4.2f} "
          f"roundness s2/s3={roundness:4.2f} | ang-cover={pop:.2f} densest-180deg={dens_half:.2f}")
    return dict(label=label, reg=rec.num_reg_images(), pts=len(P), elong=float(elong),
                roundness=float(roundness), ang_cover=float(pop), dens_half=float(dens_half))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("select", "recon", "all"):
        keep = select()
    if which in ("recon", "all"):
        recon(keep)
    if which in ("eval", "all"):
        print("\n=== TUBE SHAPE: baseline full-window vs smooth-run subset ===")
        r = [tube_shape(BASE, "7V1 full-window"), tube_shape(OUT / "sparse/0", "7V1 smooth-run")]
        (OUT / "shape_compare.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
