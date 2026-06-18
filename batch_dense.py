"""Dense Barbour recon for one video/window: CLAHE+bezel -> COLMAP (OPENCV pinned)
-> EXHAUSTIVE match -> mapper -> MVS (image_undistort + patch_match geom + fusion)
-> fused.ply. Cleans the stereo workspace + extracted frames afterward to stay under
quota (keeps dense0/fused.ply + sparse/0). depth-eval env (colmap-cuda binary).

Usage: batch_dense.py --video <name> --session <calib> --lo L --hi H [--stride 1] --out <dir>"""
from __future__ import annotations
import argparse, json, shutil, subprocess, re
from pathlib import Path
import cv2, numpy as np, pycolmap

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
CAL = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
DARK_L = 12.0


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
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  COLMAP ERR", cmd[0], r.stderr[-300:], flush=True)
    return r


def ply_n(p):
    with open(p, "rb") as f:
        for _ in range(40):
            ln = f.readline().decode("latin1", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--session", required=True)
    ap.add_argument("--lo", type=int, required=True); ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); img = out / "images"; msk = out / "masks"
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    video = DATA / a.video
    params = json.loads((CAL / a.session / "intrinsics_pinned.json").read_text())["params_colmap"]
    pstr = ",".join(f"{p:.10g}" for p in params)

    mask = bezel(video); bez = mask.astype(np.uint8) * 255
    c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(video)); fi = 0; vc = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > a.hi: break
        if a.lo <= fi <= a.hi:
            if cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
                if vc % a.stride == 0:
                    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
                    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
                    cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
                vc += 1
        fi += 1
    cap.release()
    print(f"[{a.video}] extracted {n} CLAHE frames f{a.lo}-{a.hi} stride{a.stride}", flush=True)

    db = out / "db.db"; db.unlink(missing_ok=True)
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
         "--SiftExtraction.max_image_size", "1600"])
    run(["exhaustive_matcher", "--database_path", str(db)])
    sp = out / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    if not models:
        print(f"[{a.video}] NO MODEL — registration failed"); return
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rb = pycolmap.Reconstruction(str(best)); reg = rb.num_reg_images()
    fr = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in rb.images.values())
    # ensure best model is at sparse/0
    if best.name != "0":
        shutil.rmtree(sp / "0_keep", ignore_errors=True); best.rename(sp / "0_keep")
        for d in sp.iterdir():
            if d.is_dir() and d.name != "0_keep": shutil.rmtree(d)
        (sp / "0_keep").rename(sp / "0")
    print(f"[{a.video}] mapper: {len(models)} models, best reg={reg}/{n} frames {fr[0]}..{fr[-1]}", flush=True)

    dense = out / "dense0"; shutil.rmtree(dense, ignore_errors=True)
    run(["image_undistorter", "--image_path", str(img), "--input_path", str(sp / "0"),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"])
    fused = dense / "fused.ply"
    run(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
    npts = ply_n(fused) if fused.exists() else 0
    print(f"[{a.video}] DENSE {npts} fused points", flush=True)
    # cleanup to stay under quota: keep dense0/fused.ply + sparse/0; drop stereo maps + frames + db
    shutil.rmtree(dense / "stereo", ignore_errors=True)
    shutil.rmtree(dense / "images", ignore_errors=True)
    shutil.rmtree(msk, ignore_errors=True); db.unlink(missing_ok=True)
    (out / "dense_summary.json").write_text(json.dumps(
        {"video": a.video, "session": a.session, "window": [a.lo, a.hi], "stride": a.stride,
         "n_frames": n, "n_models": len(models), "reg": reg, "frame_span": [fr[0], fr[-1]], "dense_pts": npts}, indent=2))
    print(f"[{a.video}] done; cleaned stereo. fused.ply + sparse/0 kept in {out}", flush=True)


if __name__ == "__main__":
    main()
