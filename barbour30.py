"""Barbour-recipe 30-frame local-reconstruction audit (stable COLMAP only).
Question: can small manually-selected 30-frame landmark-local batches produce cleaner
landmark CSA/DCE surfaces than our long-window reconstructions?

Per batch: pinned intrinsics (same calib as baseline), NO focal/PP/distortion refine,
EXHAUSTIVE matching inside the 30 frames, raw OR CLAHE frames. Two stages:
  --stage sparse : extract + sparse + connectivity/parallax metrics (cheap screen of all).
  --stage dense  : for viable batches, dense MVS + landmark cross-section CSA/DCE measured
                   THREE ways (polar-median polygon, ellipse fit, double-Poisson mesh slab)
                   + A/B/centerline/landmark/reproj figures. Cleans stereo after.
Landmark centers picked manually from labeled contact sheets (runs/barbour30/sheet_*.png).
depth-eval env (colmap-cuda binary)."""
from __future__ import annotations
import argparse, json, shutil, subprocess, re
from pathlib import Path
import cv2, numpy as np, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
CAL = ROOT / "runs/retriage_first15/_calib"
OUT = ROOT / "runs/barbour30"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
DARK_L = 12.0

# manual landmark centers (frame index) from the contact sheets
LANDMARKS = {
    "2-V2":  {"video": "2-V2.MP4",  "session": "2_v2", "glottis_long": 875,
              "centers": {"glottis": 878, "prox_subglottis": 918, "dist_subglottis": 985, "trachea_ref": 1120}},
    "25-V1": {"video": "25-V1.MP4", "session": "25_v1", "glottis_long": 335,
              "centers": {"glottis": 345, "prox_subglottis": 390, "dist_subglottis": 455, "trachea_ref": 560}},
    "32-V2": {"video": "32-V2.MP4", "session": "32_v1", "glottis_long": 297,
              "centers": {"glottis": 300, "prox_subglottis": 330, "dist_subglottis": 360, "trachea_ref": 470}},
}
POSITIONS = {"early": (-30, -1), "centered": (-15, 14), "late": (0, 29)}  # 30-frame windows rel. to center


def gen_batches(video_keys, preprocs=("raw", "clahe")):
    out = []
    for vk in video_keys:
        L = LANDMARKS[vk]
        for lm, C in L["centers"].items():
            for pos, (a, b) in POSITIONS.items():
                for pp in preprocs:
                    out.append({"vk": vk, "video": L["video"], "session": L["session"], "landmark": lm,
                                "center": C, "position": pos, "lo": C + a, "hi": C + b, "preproc": pp,
                                "name": f"{vk}_{lm}_{pos}_{pp}"})
    return out


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4) > 0


_BEZEL_CACHE = {}
def get_bezel(video):
    if video not in _BEZEL_CACHE: _BEZEL_CACHE[video] = bezel(DATA / video)
    return _BEZEL_CACHE[video]


def extract(b, bdir):
    img = bdir / "images"; msk = bdir / "masks"
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = get_bezel(b["video"]); bez = mask.astype(np.uint8) * 255
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(DATA / b["video"])); fi = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > b["hi"]: break
        if b["lo"] <= fi <= b["hi"]:
            if cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
                if b["preproc"] == "clahe":
                    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    im = fr.copy()
                im[~mask] = 0
                cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release()
    return img, msk, n


def run(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-250:], flush=True)
    return r


def cam_geo(rec):
    C, A = [], []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix())
        C.append(-M[:3, :3].T @ M[:3, 3]); A.append(M[:3, :3].T @ np.array([0, 0, 1.0]))
    C = np.array(C); A = np.array(A); A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    cone = float(np.degrees(np.arccos(np.clip(A @ A.T, -1, 1))).max()) if len(A) > 1 else 0.0
    # parallax: lateral vs along-axis translation spread
    Cc = C - C.mean(0)
    if len(C) > 2:
        u, s, vt = np.linalg.svd(Cc, full_matrices=False)
        along = s[0]; lateral = np.sqrt(s[1] ** 2 + s[2] ** 2) if len(s) > 2 else s[1]
        par = float(lateral / (along + 1e-9))
    else:
        par = 0.0
    med = np.median(C, 0); mad = np.median(np.abs(C - med), 0) + 1e-9
    gross = np.any(np.abs(C - med) > 6 * mad * 1.4826, axis=1)
    return cone, par, float(gross.mean()), C


def sparse_one(b):
    bdir = OUT / "batches" / b["name"]; bdir.mkdir(parents=True, exist_ok=True)
    params = json.loads((CAL / b["session"] / "intrinsics_pinned.json").read_text())["params_colmap"]
    pstr = ",".join(f"{p:.10g}" for p in params)
    img, msk, n = extract(b, bdir)
    db = bdir / "db.db"; db.unlink(missing_ok=True)
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
         "--SiftExtraction.max_image_size", "1600"])
    run(["exhaustive_matcher", "--database_path", str(db)])
    sp = bdir / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    res = {**{k: b[k] for k in ("name", "vk", "landmark", "center", "position", "lo", "hi", "preproc")},
           "n_extracted": n, "n_models": len(models)}
    if not models:
        res.update({"reg": 0, "viable": False});
    else:
        best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
        if best.name != "0":
            for d in sp.iterdir():
                if d.is_dir() and d != best: shutil.rmtree(d)
            best.rename(sp / "0_keep"); (sp / "0_keep").rename(sp / "0")
        else:
            for d in sp.iterdir():
                if d.is_dir() and d.name != "0": shutil.rmtree(d)
        rec = pycolmap.Reconstruction(str(sp / "0"))
        errs = np.array([p.error for p in rec.points3D.values()]) if rec.num_points3D() else np.array([0.0])
        cone, par, orat, C = cam_geo(rec)
        res.update({"reg": rec.num_reg_images(), "sparse": rec.num_points3D(),
                    "reproj_px": round(float(errs.mean()), 3), "cone_deg": round(cone, 1),
                    "parallax_lat_along": round(par, 3), "outlier_ratio": round(orat, 3),
                    "viable": bool(rec.num_reg_images() >= 25 and len(models) == 1 and float(errs.mean()) < 3.0)})
    db.unlink(missing_ok=True); shutil.rmtree(msk, ignore_errors=True)
    (bdir / "sparse_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sparse"], default="sparse")
    ap.add_argument("--videos", default="2-V2,25-V1")
    ap.add_argument("--only", default=None, help="run a single batch name (test)")
    a = ap.parse_args()
    batches = gen_batches(a.videos.split(","))
    if a.only: batches = [b for b in batches if b["name"] == a.only]
    rows = []
    for i, b in enumerate(batches):
        r = sparse_one(b)
        rows.append(r)
        print(f"[{i+1}/{len(batches)}] {b['name']:42} reg={r.get('reg',0):2d}/{r['n_extracted']:2d} "
              f"models={r['n_models']} cone={r.get('cone_deg','-')} reproj={r.get('reproj_px','-')} "
              f"par={r.get('parallax_lat_along','-')} out={r.get('outlier_ratio','-')} "
              f"{'VIABLE' if r.get('viable') else 'weak'}", flush=True)
    (OUT / "sparse_screen.json").write_text(json.dumps(rows, indent=2))
    nv = sum(1 for r in rows if r.get("viable"))
    print(f"\nSPARSE SCREEN done: {nv}/{len(rows)} viable -> {OUT}/sparse_screen.json")


if __name__ == "__main__":
    main()
