"""Cohort re-triage on the CORRECTED metric (triangulation angle), First 15 Videos.

Poses-only geometry screen. Per video: pin OPENCV intrinsics from its session
calibration (reuse stage2_calib_compare.py), sequential-extract ~36 frames over
the upper-airway descent (hash-verified), SuperPoint+LightGlue + COLMAP mapper
(pinned, no refine), then report the corrected metrics.

DECISION VARIABLE = per-point triangulation angle (angle at the 3-D point
subtended by the two observing camera centers), NOT the optical-axis cone.
The viewing-cone (max optical-axis angle) is reported for continuity only.

Locked: pinned OPENCV (ba_refine_focal_length=0, extra=0); sequential cv2.read()
+ per-frame SHA1 (never POS_FRAMES); DTS warnings = cosmetic; POSES ONLY (no MVS).

Run in depth-eval env. Outputs runs/retriage_first15/.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path
import cv2, numpy as np, pycolmap

DATA = Path("/home/mi3dr/dataset/validation-videos")
VID = DATA / "First 15 Videos"
CAL = DATA / "First 13 Calibration Videos"
FFMPEG = "/home/mi3dr/.conda/envs/depth-eval/bin/ffmpeg"
PYEXE = "/home/mi3dr/.conda/envs/depth-eval/bin/python"
STAGE2 = "/home/mi3dr/projects/bronchotrust/stage2_calib_compare.py"
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15")

# First-15 file -> calibration session
SESSION = {
    "2-V2.MP4": "2_v2", "5_v1_1.mp4": "5_v1", "5_v1_2.mp4": "5_v1",
    "7-V1.MP4": "7_v1", "10_v2.mp4": "10_v2", "13_v2.mp4": "13_v2",
    "15_v2.mp4": "15_v2", "16_v1.mp4": "16_v1", "18-V1.MP4": "18_v1",
    "25-V1.MP4": "25_v1", "31-V1.MP4": "31_v1", "32-V2.MP4": "32_v1",  # APPROX
    "33-V1.MP4": "33_v1",
}
APPROX = {"32-V2.MP4"}
ALL = list(SESSION)
DARK_L = 12.0
EARLY_VALID = 700     # screen the upper-airway descent (sub-cord lives here)
NPICK = 36
COSMETIC = ("non monotonic", "non-monotonic", "monoton", "dts to muxer")
REAL = ("corrupt", "concealing", "error while decoding", "invalid data", "no frame")


def find_calib(session):
    for p in CAL.glob(f"{session} Calibration Video.*"):
        return p
    return None


def calibrate(session):
    cdir = OUT / "_calib" / session
    pinned = cdir / "intrinsics_pinned.json"
    if pinned.exists():
        return json.loads(pinned.read_text())
    cv = find_calib(session)
    if cv is None:
        return None
    cdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([PYEXE, STAGE2, "--video", str(cv), "--out", str(cdir)],
                       capture_output=True, text=True)
    if not pinned.exists():
        (cdir / "calib_fail.log").write_text(r.stdout[-3000:] + "\nERR\n" + r.stderr[-2000:])
        return None
    return json.loads(pinned.read_text())


def integrity(video):
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-f", "null", "-"],
                       capture_output=True, text=True, timeout=1800)
    lines = [l for l in r.stderr.strip().split("\n") if l.strip()]
    real = sum(1 for l in lines if any(k in l.lower() for k in REAL)
               and not any(k in l.lower() for k in COSMETIC))
    cos = sum(1 for l in lines if any(k in l.lower() for k in COSMETIC))
    return {"n_err": len(lines), "real": real, "cosmetic": cos,
            "sample_real": [l for l in lines if any(k in l.lower() for k in REAL)
                            and not any(k in l.lower() for k in COSMETIC)][:5]}


def sha(fr):
    return hashlib.sha1(cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA).tobytes()).hexdigest()


def scan_and_pick(video):
    """Pass 1: sequential stats + mask. Pick ~NPICK over first EARLY_VALID valid frames."""
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cum = np.zeros((H, W), np.int32)
    L = []; shas = []; diff = []; gds = None; last = None
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cum += (fr.mean(2) > 8).astype(np.int32)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        L.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0].mean() * 100 / 255))
        shas.append(sha(fr))
        gd = cv2.resize(g, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        diff.append(0.0 if last is None else float(np.abs(gd.astype(np.int16) - last.astype(np.int16)).mean()))
        last = gd
        fi += 1
    cap.release()
    n = fi
    L = np.array(L); diff = np.array(diff)
    mask = cum >= max(int(0.20 * n), 5)
    m = cv2.morphologyEx((mask.astype(np.uint8)) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=4)
    valid = L >= DARK_L
    vidx = np.where(valid)[0]
    if len(vidx) == 0:
        return None
    cand = vidx[:EARLY_VALID]
    # flow-spaced pick
    d = diff[cand].copy(); d[0] = 0
    cum_d = np.cumsum(d); tot = cum_d[-1]
    if tot > 1e-6:
        tg = np.linspace(0, tot, NPICK)
        picks = sorted(set(int(cand[np.clip(np.searchsorted(cum_d, t), 0, len(cand) - 1)]) for t in tg))
    else:
        picks = sorted(set(int(x) for x in np.linspace(cand[0], cand[-1], NPICK).round()))
    return {"n_total": n, "n_valid": int(valid.sum()), "WH": [W, H],
            "mask": m > 0, "sha": shas, "picks": picks}


def extract(video, picks, mask, sha_ref):
    img_dir = OUT / "_work" / video.stem / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.png"):
        p.unlink()
    bez = mask
    want = sorted(picks); ni = 0; fi = 0; names = []; matches = 0
    cap = cv2.VideoCapture(str(video))
    while ni < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if fi == want[ni]:
            matches += int(sha(fr) == sha_ref[fi])
            fr[~bez] = 0
            nm = f"f{fi:05d}.png"
            cv2.imwrite(str(img_dir / nm), fr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            names.append(nm); ni += 1
        fi += 1
    cap.release()
    return img_dir, sorted(names), matches


def run_sfm(img_dir, names, intr):
    from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction as hr
    work = img_dir.parent
    fconf = extract_features.confs["superpoint_max"]; mconf = match_features.confs["superpoint+lightglue"]
    feats = extract_features.main(fconf, img_dir, work, image_list=names)
    pairs = work / "pairs.txt"; pairs_from_exhaustive.main(pairs, image_list=names)
    matches = match_features.main(mconf, pairs, fconf["output"], work)
    params = ",".join(f"{p:.10g}" for p in [intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["k1"], intr["k2"], 0.0, 0.0])
    sfm = work / "sfm"; sfm.mkdir(exist_ok=True)
    hr.main(sfm, img_dir, pairs, feats, matches, image_list=names,
            camera_mode=hr.pycolmap.CameraMode.SINGLE,
            image_options={"camera_model": "OPENCV", "camera_params": params},
            verbose=False, mapper_options={
                "min_num_matches": 8, "multiple_models": True, "max_num_models": 50, "min_model_size": 3,
                "ba_refine_focal_length": False, "ba_refine_extra_params": False, "ba_refine_principal_point": False,
                "mapper": {"init_min_tri_angle": 4.0, "init_max_error": 8.0, "init_min_num_inliers": 15,
                           "init_max_forward_motion": 0.99, "abs_pose_max_error": 20.0,
                           "abs_pose_min_num_inliers": 15, "abs_pose_min_inlier_ratio": 0.1,
                           "filter_max_reproj_error": 8.0, "filter_min_tri_angle": 1.0}})
    return sfm


def metrics(sfm, intr):
    rec = pycolmap.Reconstruction(str(sfm))
    C, A, cc = [], [], {}
    for img in rec.images.values():
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        ctr = -R.T @ t; C.append(ctr); cc[img.image_id] = ctr
        A.append(R.T @ np.array([0, 0, 1.0]))
    n = len(C)
    if n < 2:
        return {"n_reg": n, "status": "insufficient_registration"}
    C = np.array(C); A = np.array(A); A /= np.linalg.norm(A, axis=1, keepdims=True)
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    sd = (Cc @ Vt.T).std(0, ddof=1)
    along = float(sd[0]); lateral = float(np.sqrt(sd[1] ** 2 + sd[2] ** 2))
    ang = np.degrees(np.arccos(np.clip(A @ A.T, -1, 1)))[np.triu_indices(n, 1)]
    tri = []
    for p in rec.points3D.values():
        obs = [el.image_id for el in p.track.elements if el.image_id in cc]
        if len(obs) < 2:
            continue
        X = np.array(p.xyz); V = np.array([cc[i] - X for i in obs]); V /= np.linalg.norm(V, axis=1, keepdims=True)
        tri.append(float(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max()))
    tri = np.array(tri)
    cam = next(iter(rec.cameras.values())); cp = list(cam.params)
    return {"n_reg": n, "n_sparse": int(rec.num_points3D()),
            "lat_along": round(lateral / along, 4) if along > 1e-9 else None,
            "max_optical_axis_deg": round(float(ang.max()), 3),
            "median_optical_axis_deg": round(float(np.median(ang)), 3),
            "median_tri_deg": round(float(np.median(tri)), 3) if len(tri) else None,
            "p95_tri_deg": round(float(np.percentile(tri, 95)), 3) if len(tri) else None,
            "sfm_fx": round(cp[0], 3), "pin_fx": round(intr["fx"], 3),
            "delta_fx": round(cp[0] - intr["fx"], 7)}


def triage_one(fname):
    video = VID / fname
    session = SESSION[fname]
    row = {"video": fname, "session": session, "approx_intrinsics": fname in APPROX}
    intr = calibrate(session)
    if intr is None:
        row["status"] = "NO_CALIBRATION"; return row
    row["calib_rms"] = round(intr.get("rms_reproj_px", -1), 4)
    ig = integrity(video)
    row["integrity"] = ig
    row["intact"] = ig["real"] == 0
    sp = scan_and_pick(video)
    if sp is None:
        row["status"] = "NO_VALID_FRAMES"; return row
    row["n_frames"] = sp["n_total"]; row["n_valid"] = sp["n_valid"]
    img_dir, names, hm = extract(video, sp["picks"], sp["mask"], sp["sha"])
    row["n_picked"] = len(names); row["hash_match"] = f"{hm}/{len(names)}"
    try:
        sfm = run_sfm(img_dir, names, intr)
        row.update(metrics(sfm, intr))
        row["status"] = "OK"
    except Exception as e:
        row["status"] = f"SFM_FAIL:{type(e).__name__}:{str(e)[:120]}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(ALL))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows").mkdir(exist_ok=True)
    vids = [v for v in args.videos.split(",") if v in SESSION]
    print(f"triaging {len(vids)} videos from First 15 Videos\n")
    rows = []
    for v in vids:
        print(f"\n########## {v} ##########", flush=True)
        try:
            r = triage_one(v)
        except Exception as e:
            r = {"video": v, "status": f"FATAL:{type(e).__name__}:{str(e)[:150]}"}
        (OUT / "rows" / f"{v}.json").write_text(json.dumps(r, indent=2))
        rows.append(r)
        print(json.dumps({k: r[k] for k in r if k != "integrity"}, indent=0), flush=True)
    (OUT / "triage_table.json").write_text(json.dumps(rows, indent=2))
    # table
    print("\n\n=== CORRECTED RE-TRIAGE TABLE (First 15 Videos) ===")
    h = f"{'video':<11}{'reg/N':>7}{'lat/along':>10}{'cone°(rep)':>11}{'medTri°':>9}{'p95Tri°':>9}{'verdict':>13}"
    print(h); print("-" * len(h))
    for r in rows:
        if r.get("status") != "OK":
            print(f"{r['video']:<11}{'-':>7}{'-':>10}{'-':>11}{'-':>9}{'-':>9}{r.get('status','?')[:13]:>13}")
            continue
        mt = r.get("median_tri_deg") or 0; p95 = r.get("p95_tri_deg") or 0; la = r.get("lat_along") or 0
        verdict = "CANDIDATE" if (mt >= 10 and p95 >= 15 and la >= 0.2) else "not viable"
        print(f"{r['video']:<11}{str(r['n_reg'])+'/'+str(r['n_picked']):>7}{la:>10.3f}"
              f"{r.get('max_optical_axis_deg',0):>11.2f}{mt:>9.2f}{p95:>9.2f}{verdict:>13}")
    print("\n15_v2 (trimmed, FROZEN ref): reg 34/36 lat/along 0.053 cone 17.5 medTri 2.2 p95Tri 5.9 -> NOT viable")
    print(f"saved {OUT/'triage_table.json'}")


if __name__ == "__main__":
    main()
