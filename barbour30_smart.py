"""Smart 30-frame selection for the Barbour local audit (stable COLMAP).
From a pool around a landmark, pick 30 frames that maximize:
  - same landmark visible      -> pool constrained to +/-POOL around the landmark center
  - lateral/angled viewpoint    -> farthest-point sampling in a flow-derived pseudo-viewpoint
    change (parallax)              space; lateral motion weighted ABOVE forward (zoom) motion
  - sharp frames                -> Laplacian-variance gate (drop blur)
  - no mucus/glare              -> specular-saturation + low-contrast(haze) gate
  - not all from one forward    -> diversity sampling collapses dwell clusters to a few frames
    dwell
Then run COLMAP (pinned intrinsics, exhaustive). Writes a batch compatible with
barbour30_dense.py + appends to sparse_screen.json. depth-eval env (colmap-cuda)."""
from __future__ import annotations
import argparse, json, shutil, subprocess, re
from pathlib import Path
import cv2, numpy as np, pycolmap
from barbour30 import bezel, run as _run, cam_geo, LANDMARKS, DATA, CAL, OUT, COLMAP, DARK_L

TARGETS = [  # (video_key, landmark, pool half-width) — moderate pool: more parallax than the
             # contiguous +/-15 window, but enough frame overlap for low-texture matching
    ("2-V2", "prox_subglottis", 30), ("2-V2", "dist_subglottis", 30),
    ("25-V1", "prox_subglottis", 30), ("25-V1", "dist_subglottis", 30),
    ("2-V2", "glottis", 30), ("25-V1", "glottis", 30),
]


def frame_metrics(video, lo, hi, mask):
    """Per pool-frame: sharpness, glare, contrast, brightness + downscaled masked gray for flow."""
    cap = cv2.VideoCapture(str(DATA / video)); fi = 0
    idx, sharp, glare, contr, bright, grays, bgr = [], [], [], [], [], [], {}
    while True:
        ok, fr = cap.read()
        if not ok or fi > hi: break
        if lo <= fi <= hi:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); L = lab[:, :, 0]
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            m = mask
            Lm = L[m]
            idx.append(fi)
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F)[m].var()))
            glare.append(float((Lm > 240).mean()))
            contr.append(float(Lm.std()))
            bright.append(float(Lm.mean() * 100 / 255))
            gd = cv2.resize((g * (m)).astype(np.uint8), (240, 135))
            grays.append(gd); bgr[fi] = fr
        fi += 1
    cap.release()
    return (np.array(idx), np.array(sharp), np.array(glare), np.array(contr), np.array(bright), grays)


def viewpoint_traj(grays):
    """Pseudo-viewpoint per frame from cumulative optical flow: lateral mean-flow (weighted)
    + forward divergence (down-weighted). Dwells -> ~no change -> cluster."""
    H, W = grays[0].shape; cy, cx = H / 2, W / 2
    yy, xx = np.mgrid[0:H, 0:W]; rx = (xx - cx); ry = (yy - cy)
    rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    vp = [np.zeros(3)]
    for i in range(1, len(grays)):
        fl = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        valid = grays[i] > 3
        fx = np.median(fl[..., 0][valid]) if valid.any() else 0.0
        fy = np.median(fl[..., 1][valid]) if valid.any() else 0.0
        rad = (fl[..., 0] * rx + fl[..., 1] * ry) / rn
        fwd = np.median(rad[valid]) if valid.any() else 0.0
        vp.append(vp[-1] + np.array([fx, fy, 0.3 * fwd]))  # lateral full weight, forward 0.3
    V = np.array(vp); V = V - V.mean(0)
    s = V.std(0) + 1e-6
    return V / s


def fps_select(V, qmask, seed, n=30):
    """Farthest-point sampling among quality-passed frames (indices where qmask True)."""
    cand = np.where(qmask)[0]
    if len(cand) <= n: return list(cand)
    sel = [seed if seed in cand else cand[len(cand) // 2]]
    d = np.full(len(V), np.inf)
    while len(sel) < n:
        last = sel[-1]
        d = np.minimum(d, np.linalg.norm(V - V[last], axis=1))
        dd = d.copy(); dd[~qmask] = -1
        for s in sel: dd[s] = -1
        nxt = int(np.argmax(dd))
        if dd[nxt] < 0: break
        sel.append(nxt)
    return sorted(sel)


def select(vk, landmark, pool, n=30):
    L = LANDMARKS[vk]; video = L["video"]; C = L["centers"][landmark]
    lo, hi = C - pool, C + pool
    mask = bezel(DATA / video)
    idx, sharp, glare, contr, bright, grays = frame_metrics(video, lo, hi, mask)
    V = viewpoint_traj(grays)
    # quality gate: sharp enough, low glare, sane brightness, enough contrast (not hazy/mucus)
    sok = sharp >= np.percentile(sharp, 35)
    gok = glare < 0.04
    bok = (bright >= DARK_L) & (bright <= 92)
    cok = contr >= np.percentile(contr, 25)
    q = sok & gok & bok & cok
    if q.sum() < n:  # relax progressively
        q = sok & (glare < 0.08) & bok
    if q.sum() < n:
        q = bright >= DARK_L
    seed = int(np.argmin(np.abs(idx - C)))
    sel = fps_select(V, q, seed, n)
    frames = sorted(int(idx[i]) for i in sel)
    info = {"vk": vk, "landmark": landmark, "center": C, "pool": [int(lo), int(hi)], "n_pool": len(idx),
            "n_quality_passed": int(q.sum()), "frames": frames,
            "viewpoint_span": [round(float((V[sel].max(0) - V[sel].min(0))[k]), 2) for k in range(3)],
            "sharp_med": round(float(np.median(sharp[sel])), 1), "glare_med": round(float(np.median(glare[sel])), 4)}
    return info, {f: grays for f in []}  # montage built by caller from video


def montage(vk, landmark, frames, name):
    video = LANDMARKS[vk]["video"]; want = set(frames); got = {}
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)); mask = bezel(DATA / video)
    cap = cv2.VideoCapture(str(DATA / video)); fi = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > max(frames): break
        if fi in want:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
            h, w = im.shape[:2]; s = min(w, h); im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            im = cv2.resize(im, (200, 200)); cv2.putText(im, f"{fi}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            got[fi] = im
        fi += 1
    cap.release()
    cols = 10; rows = (len(frames) + cols - 1) // cols; sh = np.zeros((rows * 200, cols * 200, 3), np.uint8)
    for k, f in enumerate(sorted(frames)):
        r, c = divmod(k, cols); sh[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200] = got[f]
    cv2.imwrite(str(OUT / f"smart_{name}_montage.png"), sh)


def extract_list(vk, frames, bdir, preproc="clahe"):
    video = LANDMARKS[vk]["video"]; img = bdir / "images"; msk = bdir / "masks"
    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = bezel(DATA / video); bez = mask.astype(np.uint8) * 255; clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    want = set(frames); cap = cv2.VideoCapture(str(DATA / video)); fi = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > max(frames): break
        if fi in want:
            if preproc == "clahe":
                lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0]); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else: im = fr.copy()
            im[~mask] = 0
            cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release(); return img, msk, n


def recon(vk, landmark, info, preproc="clahe"):
    name = f"{vk}_{landmark}_smart_{preproc}"
    bdir = OUT / "batches" / name; bdir.mkdir(parents=True, exist_ok=True)
    sess = LANDMARKS[vk]["session"]; pstr = ",".join(f"{p:.10g}" for p in json.loads((CAL / sess / "intrinsics_pinned.json").read_text())["params_colmap"])
    img, msk, nfr = extract_list(vk, info["frames"], bdir, preproc)
    db = bdir / "db.db"; db.unlink(missing_ok=True)
    _run(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
          "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
          "--SiftExtraction.max_image_size", "1600"])
    _run(["exhaustive_matcher", "--database_path", str(db)])
    sp = bdir / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    _run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
          "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_principal_point", "0",
          "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = [d for d in sp.iterdir() if d.is_dir()]
    res = {"name": name, "vk": vk, "landmark": landmark, "center": info["center"], "position": "smart",
           "lo": info["pool"][0], "hi": info["pool"][1], "preproc": preproc, "n_extracted": nfr, "n_models": len(models),
           "selection": info}
    if models:
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
        cone, par, orat, _ = cam_geo(rec)
        res.update({"reg": rec.num_reg_images(), "sparse": rec.num_points3D(), "reproj_px": round(float(errs.mean()), 3),
                    "cone_deg": round(cone, 1), "parallax_lat_along": round(par, 3), "outlier_ratio": round(orat, 3),
                    "viable": bool(rec.num_reg_images() >= 25 and len(models) == 1 and float(errs.mean()) < 3.0)})
    else:
        res.update({"reg": 0, "viable": False})
    db.unlink(missing_ok=True); shutil.rmtree(msk, ignore_errors=True)
    (bdir / "sparse_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    targets = TARGETS if not a.only else [t for t in TARGETS if f"{t[0]}_{t[1]}" == a.only]
    # contiguous-centered reference for comparison
    screen = {r["name"]: r for r in json.loads((OUT / "sparse_screen.json").read_text())}
    out = []
    for vk, lm, pool in targets:
        info, _ = select(vk, lm, pool)
        montage(vk, lm, info["frames"], f"{vk}_{lm}")
        res = recon(vk, lm, info)
        ref = screen.get(f"{vk}_{lm}_centered_clahe", {})
        print(f"{vk}/{lm:16} SMART reg={res.get('reg',0)}/30 models={res['n_models']} cone={res.get('cone_deg','-')} "
              f"par={res.get('parallax_lat_along','-')} reproj={res.get('reproj_px','-')} "
              f"vp_span={info['viewpoint_span']} qpass={info['n_quality_passed']}/{info['n_pool']} "
              f"{'VIABLE' if res.get('viable') else 'weak'}  | contiguous-ref cone={ref.get('cone_deg','-')} par={ref.get('parallax_lat_along','-')}", flush=True)
        out.append(res)
    (OUT / "smart_screen.json").write_text(json.dumps(out, indent=2))
    print(f"\nsmart selection done -> {OUT}/smart_screen.json")


if __name__ == "__main__":
    main()
