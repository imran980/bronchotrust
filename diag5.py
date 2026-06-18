"""Feature/track diagnostic on 2-V2, 16_v1, 18-V1, 25-V1, 32-V2 (NO MVS/mesh).
Same recipe as the 7-V1 ablation winner: CLAHE + bezel mask, COLMAP SIFT,
sequential overlap30 quadratic, mapper -> sparse. Reports per video:
kp_med, verified matches/pair median, sparse pts/frame, track-len median,
>10-view tracks, mean reproj err. Dense window = stride-2 valid frames from the
subglottic-descent anchor (eyeballed from contact sheets), <=170 frames.

Intrinsics: own pinned OPENCV (2_v2,25_v1); 32_v2 borrows 32_v1 (same patient);
16_v1,18_v1 have NO calib -> SIMPLE_RADIAL focal ESTIMATED (flagged). Run in depth-eval env."""
from __future__ import annotations
import json, sqlite3, shutil, subprocess
from pathlib import Path
import cv2, numpy as np, pycolmap

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
CAL = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/diag5")
DARK_L, STRIDE, NMAX = 12.0, 2, 170

CFG = {
    "2-V2.MP4":  {"anchor": 720, "calib": "2_v2",  "note": "own calib"},
    "16_v1.mp4": {"anchor": 300, "calib": None,    "note": "NO calib -> focal estimated"},
    "18-V1.MP4": {"anchor": 200, "calib": None,    "note": "NO calib -> focal estimated"},
    "25-V1.MP4": {"anchor": 173, "calib": "25_v1", "note": "own calib (sub_in)"},
    "32-V2.MP4": {"anchor": 191, "calib": "32_v1", "note": "calib borrowed 32_v1 (same patient)"},
}


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4)
    return m > 0


def clahe_l(img, c):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract(video, anchor, mask):
    c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    bez = mask.astype(np.uint8) * 255
    cap = cv2.VideoCapture(str(video)); fi = 0; vc = 0; got = []
    while True:
        ok, fr = cap.read()
        if not ok or len(got) >= NMAX: break
        if fi >= anchor:
            L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255
            if L >= DARK_L:
                if vc % STRIDE == 0:
                    im = clahe_l(fr, c); im[~mask] = 0
                    got.append((f"f{fi:05d}.png", im))
                vc += 1
        fi += 1
    cap.release()
    return got, bez


def run(cmd):
    return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def stats(db, model):
    c = sqlite3.connect(db); cur = c.cursor()
    kp = np.array([r[0] for r in cur.execute("SELECT rows FROM keypoints").fetchall()])
    ver = np.array([r[0] for r in cur.execute("SELECT rows FROM two_view_geometries WHERE rows>0").fetchall()] or [0])
    c.close()
    r = pycolmap.Reconstruction(str(model))
    reg = r.num_reg_images(); sp = r.num_points3D()
    tl = np.array([len(p.track.elements) for p in r.points3D.values()]) if sp else np.array([0])
    errs = np.array([p.error for p in r.points3D.values()]) if sp else np.array([0.0])
    n_gt10 = int((tl > 10).sum())
    return {"kp_med": round(float(np.median(kp))), "ver_med": round(float(np.median(ver))),
            "ver_mean": round(float(ver.mean()), 1), "reg": reg, "sparse": sp,
            "sparse_per_frame": round(sp / max(reg, 1), 1), "track_med": round(float(np.median(tl)), 1),
            "gt10_tracks": n_gt10, "gt10_frac": round(100 * n_gt10 / max(sp, 1), 1),
            "reproj_px": round(float(errs.mean()), 2)}


def diag(name, cfg):
    video = DATA / name; vd = OUT / Path(name).stem; img = vd / "images"; msk = vd / "masks"
    for d in (img, msk):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = bezel(video)
    got, bez = extract(video, cfg["anchor"], mask)
    for nm, im in got:
        cv2.imwrite(str(img / nm), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(msk / (nm + ".png")), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # contact sheet of selected frames
    cols = 12; rows = (len(got) + cols - 1) // cols; cv = np.zeros((rows * 90, cols * 160, 3), np.uint8)
    for i, (nm, im) in enumerate(got):
        t = cv2.resize(im, (160, 90)); cv2.putText(t, nm[1:6], (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        r, cc = divmod(i, cols); cv[r * 90:r * 90 + 90, cc * 160:cc * 160 + 160] = t
    cv2.imwrite(str(vd / "selected.png"), cv)
    db = vd / "db.db"
    if db.exists(): db.unlink()
    fe = ["feature_extractor", "--database_path", str(db), "--image_path", str(img),
          "--ImageReader.mask_path", str(msk), "--ImageReader.single_camera", "1", "--SiftExtraction.max_image_size", "1600"]
    refine = "1"
    if cfg["calib"]:
        params = json.loads((CAL / cfg["calib"] / "intrinsics_pinned.json").read_text())["params_colmap"]
        fe += ["--ImageReader.camera_model", "OPENCV", "--ImageReader.camera_params", ",".join(f"{p:.10g}" for p in params)]
        refine = "0"
    else:
        fe += ["--ImageReader.camera_model", "SIMPLE_RADIAL"]
    run(fe)
    run(["sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", "30", "--SequentialMatching.quadratic_overlap", "1"])
    sp = vd / "sparse"; sp.mkdir(exist_ok=True)
    run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", refine, "--Mapper.ba_refine_extra_params", refine, "--Mapper.ba_refine_principal_point", "0"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    if not models:
        print(f"  {name:<11} NO MODEL (registration failed)"); return {"video": name, **cfg, "n_frames": len(got), "fail": True}
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    s = stats(db, best); s.update({"video": name, "anchor": cfg["anchor"], "calib_note": cfg["note"],
                                   "n_frames": len(got), "n_models": len(models)})
    print(f"  {name:<11} n={len(got):<3} kp_med={s['kp_med']:<5} ver/pair_med={s['ver_med']:<5} "
          f"sp/fr={s['sparse_per_frame']:<6} trk_med={s['track_med']:<4} >10v={s['gt10_tracks']:<6}({s['gt10_frac']}%) "
          f"reproj={s['reproj_px']}px | reg={s['reg']} models={s['n_models']} [{cfg['note']}]", flush=True)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [diag(n, c) for n, c in CFG.items()]
    (OUT / "diag5.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'video':<11}{'n':>4}{'kp_med':>8}{'ver/pair':>10}{'sp/frame':>10}{'trk_med':>9}{'>10v':>8}{'>10v%':>7}{'reproj':>8}")
    for r in rows:
        if r.get("fail"): print(f"{r['video']:<11}{r['n_frames']:>4}   --- registration failed ---"); continue
        print(f"{r['video']:<11}{r['n_frames']:>4}{r['kp_med']:>8}{r['ver_med']:>10}{r['sparse_per_frame']:>10}"
              f"{r['track_med']:>9}{r['gt10_tracks']:>8}{r['gt10_frac']:>7}{r['reproj_px']:>8}")
    print("\n7-V1 ref (CLAHE, 171fr): kp_med=4452 ver/pair_med=206 sp/frame~56 (full-model0: sp/fr~100, trk_med=6, >10v=31%, reproj=1.37)")
    print(f"saved {OUT/'diag5.json'}")


if __name__ == "__main__":
    main()
