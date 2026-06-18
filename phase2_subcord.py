"""PHASE 2 (top-tier triage): dual-pass subglottic reconstruction, POSES ONLY.

5 videos, approved anatomy windows. Curated frame set = dense sub_in + dense
sub_out + sparse tracheal bridge connecting the two passes. Pinned OPENCV
intrinsics (no refine), SuperPoint+LightGlue, COLMAP mapper only (no GlueMap,
no MVS/Poisson/centerline/cross-section/scale).

Metrics computed over the SUB-CORD cameras / sub-cord-observing tracks (that is
where subglottic parallax must live). Reports raw geometry only; no thresholds,
no quality/density ranking. Final table sorted by median triangulation angle
(then p95 tri, then lateral/along).

Run in depth-eval env. Outputs runs/phase2_subcord/.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path
import cv2, numpy as np, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
CAL = Path("/home/mi3dr/dataset/validation-videos/First 13 Calibration Videos")
PYEXE = "/home/mi3dr/.conda/envs/depth-eval/bin/python"
STAGE2 = "/home/mi3dr/projects/bronchotrust/stage2_calib_compare.py"
CALDIR = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/phase2_subcord")

WINDOWS = {
    "7-V1.MP4":  {"session": "7_v1",  "sub_in": [130, 220],  "sub_out": [1846, 1992]},
    "25-V1.MP4": {"session": "25_v1", "sub_in": [173, 248],  "sub_out": [1697, 1777]},
    "31-V1.MP4": {"session": "31_v1", "sub_in": [174, 327],  "sub_out": [1521, 1625]},
    "32-V2.MP4": {"session": "32_v1", "sub_in": [191, 305],  "sub_out": [2229, 2281], "approx": True},
    "33-V1.MP4": {"session": "33_v1", "sub_in": [75, 140],   "sub_out": [1522, 1669]},
}
DARK_L = 12.0
K_IN, K_OUT, K_BRIDGE = 14, 14, 10
SEG_COL = {"sub_in": "#3cb43c", "sub_out": "#19c819", "bridge": "#3c78e6"}


def find_calib(session):
    for p in CAL.glob(f"{session} Calibration Video.*"):
        return p
    return None


def calibrate(session):
    cdir = CALDIR / session
    pinned = cdir / "intrinsics_pinned.json"
    if pinned.exists():
        return json.loads(pinned.read_text())
    cv = find_calib(session)
    if cv is None:
        return None
    cdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([PYEXE, STAGE2, "--video", str(cv), "--out", str(cdir)],
                   capture_output=True, text=True)
    return json.loads(pinned.read_text()) if pinned.exists() else None


def content_mask(video, n=50):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
                  np.ones((5, 5), np.uint8), iterations=4)
    return m > 0, (W, H)


def sha(fr):
    return hashlib.sha1(cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA).tobytes()).hexdigest()


def scan(video, mask, last_idx):
    cap = cv2.VideoCapture(str(video))
    L, diff, shas = [], [], []
    last = None; fi = 0
    while fi <= last_idx:
        ok, fr = cap.read()
        if not ok:
            break
        L.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255))
        gd = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (mask.shape[1] // 6, mask.shape[0] // 6), interpolation=cv2.INTER_AREA)
        diff.append(0.0 if last is None else float(np.abs(gd.astype(np.int16) - last.astype(np.int16)).mean()))
        last = gd; shas.append(sha(fr)); fi += 1
    cap.release()
    return np.array(L), np.array(diff), shas


def flow_pick(valid, diff, lo, hi, k):
    cand = np.array([i for i in range(lo, hi + 1) if i < len(valid) and valid[i]], int)
    if len(cand) <= k:
        return cand.tolist()
    d = diff[cand].copy(); d[0] = 0
    cd = np.cumsum(d); tot = cd[-1]
    if tot < 1e-6:
        return sorted(set(int(x) for x in np.linspace(cand[0], cand[-1], k).round()))
    tg = np.linspace(0, tot, k)
    return sorted(set(int(cand[np.clip(np.searchsorted(cd, t), 0, len(cand) - 1)]) for t in tg))


def extract(video, picks, mask, sha_ref, od):
    img_dir = od / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.png"):
        p.unlink()
    want = sorted(picks); ni = 0; fi = 0; names = []; matches = 0
    cap = cv2.VideoCapture(str(video))
    while ni < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if fi == want[ni]:
            matches += int(sha(fr) == sha_ref[fi])
            fr[~mask] = 0
            cv2.imwrite(str(img_dir / f"f{fi:05d}.png"), fr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            names.append(f"f{fi:05d}.png"); ni += 1
        fi += 1
    cap.release()
    return img_dir, sorted(names), matches


def run_sfm(img_dir, names, intr, od):
    from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction as hr
    fconf = extract_features.confs["superpoint_max"]; mconf = match_features.confs["superpoint+lightglue"]
    feats = extract_features.main(fconf, img_dir, od, image_list=names)
    pairs = od / "pairs.txt"; pairs_from_exhaustive.main(pairs, image_list=names)
    matches = match_features.main(mconf, pairs, fconf["output"], od)
    params = ",".join(f"{p:.10g}" for p in [intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["k1"], intr["k2"], 0.0, 0.0])
    sfm = od / "sfm"; sfm.mkdir(exist_ok=True)
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


def seg_of(fi, win):
    if win["sub_in"][0] <= fi <= win["sub_in"][1]:
        return "sub_in"
    if win["sub_out"][0] <= fi <= win["sub_out"][1]:
        return "sub_out"
    return "bridge"


def geom(centers, axes):
    C = np.asarray(centers, float); A = np.asarray(axes, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    n = len(C)
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    sd = (Cc @ Vt.T).std(0, ddof=1)
    lat_along = float(np.sqrt(sd[1] ** 2 + sd[2] ** 2) / sd[0]) if sd[0] > 1e-9 else None
    ang = np.degrees(np.arccos(np.clip(A @ A.T, -1, 1)))[np.triu_indices(n, 1)]
    return lat_along, float(ang.max()), float(np.median(ang))


def analyze(sfm, win):
    rec = pycolmap.Reconstruction(str(sfm))
    cc, seg = {}, {}
    C, A = [], []
    for img in rec.images.values():
        m = re.search(r"f(\d+)\.png", img.name); fi = int(m.group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        ctr = -R.T @ t; cc[img.image_id] = ctr; seg[img.image_id] = seg_of(fi, win)
        C.append(ctr); A.append(R.T @ np.array([0, 0, 1.0]))
    sub_ids = {i for i, s in seg.items() if s in ("sub_in", "sub_out")}
    # sub-cord camera geometry
    sc_C = [cc[i] for i in sub_ids]; sc_A = [A[list(cc).index(i)] for i in sub_ids]
    g = geom(sc_C, sc_A) if len(sc_C) >= 2 else (None, None, None)
    # triangulation over tracks observed by >=1 sub-cord camera
    tri = []
    for p in rec.points3D.values():
        obs = [el.image_id for el in p.track.elements if el.image_id in cc]
        if len(obs) < 2 or not (set(obs) & sub_ids):
            continue
        X = np.array(p.xyz); V = np.array([cc[i] - X for i in obs]); V /= np.linalg.norm(V, axis=1, keepdims=True)
        tri.append(float(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max()))
    tri = np.array(tri)
    errs = [float(p.error) for p in rec.points3D.values()]
    return {"n_reg": len(cc), "n_sparse": int(rec.num_points3D()),
            "reproj_px": float(np.mean(errs)) if errs else None,
            "subcord_lat_along": g[0], "subcord_max_axis_deg": g[1], "subcord_median_axis_deg": g[2],
            "subcord_median_tri_deg": float(np.median(tri)) if len(tri) else None,
            "subcord_p95_tri_deg": float(np.percentile(tri, 95)) if len(tri) else None,
            "subcord_n_points": int(len(tri)),
            "subcord_n_cams_reg": len(sub_ids),
            "_cc": {i: cc[i].tolist() for i in cc}, "_seg": {i: seg[i] for i in seg}, "_tri": tri.tolist()}


def plots(od, fname, ana, picks, win, img_dir):
    cc = {int(k): np.array(v) for k, v in ana["_cc"].items()}
    seg = {int(k): v for k, v in ana["_seg"].items()}
    tri = np.array(ana["_tri"])
    # trajectory
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    C = np.array(list(cc.values())); segs = [seg[i] for i in cc]
    for proj, (ix, iy, lx, ly) in zip(ax, [(0, 2, "X", "Z"), (0, 1, "X", "Y")]):
        for i, s in zip(cc, segs):
            proj.scatter(cc[i][ix], cc[i][iy], c=SEG_COL[s], s=45, edgecolors="k", linewidths=.4, zorder=3)
        proj.set_xlabel(lx); proj.set_ylabel(ly); proj.set_aspect("equal", "datalim"); proj.grid(alpha=.3)
        proj.set_title(f"{lx}{ly}")
    import matplotlib.patches as mp
    ax[0].legend(handles=[mp.Patch(color=SEG_COL[s], label=s) for s in SEG_COL], fontsize=8, loc="best")
    fig.suptitle(f"{fname} camera trajectory  reg={ana['n_reg']}  subcord lat/along={ana['subcord_lat_along']}")
    fig.tight_layout(); fig.savefig(od / "trajectory.png", dpi=130); plt.close(fig)
    # tri hist
    fig, a = plt.subplots(figsize=(7, 4.5))
    if len(tri):
        a.hist(tri, bins=30, color="#c0392b", alpha=.85)
        a.axvline(np.median(tri), c="k", ls="--", label=f"median {np.median(tri):.1f}")
        a.axvline(np.percentile(tri, 95), c="b", ls=":", label=f"p95 {np.percentile(tri,95):.1f}")
        a.legend()
    a.set_title(f"{fname} subglottic triangulation angle ({len(tri)} pts)")
    a.set_xlabel("triangulation angle (deg)")
    fig.tight_layout(); fig.savefig(od / "tri_hist.png", dpi=130); plt.close(fig)
    # selected-frames contact sheet
    cols = 10; thumbs = sorted(picks)
    ims = {int(re.search(r'f(\d+)', p.name).group(1)): cv2.imread(str(p)) for p in img_dir.glob("f*.png")}
    tw = 200; th = int(round(tw * ims[thumbs[0]].shape[0] / ims[thumbs[0]].shape[1]))
    rows = (len(thumbs) + cols - 1) // cols; lh = 26
    canvas = np.full((rows * (th + lh), cols * tw, 3), 245, np.uint8); f = cv2.FONT_HERSHEY_SIMPLEX
    for idx, fi in enumerate(thumbs):
        rr, ccc = divmod(idx, cols); y0 = rr * (th + lh); x0 = ccc * tw
        canvas[y0:y0 + th, x0:x0 + tw] = cv2.resize(ims[fi], (tw, th), interpolation=cv2.INTER_AREA)
        col = tuple(int(c) for c in bytes.fromhex(SEG_COL[seg_of(fi, win)][1:]))[::-1]
        canvas[y0 + th:y0 + th + lh, x0:x0 + tw] = col
        cv2.putText(canvas, f"{fi}", (x0 + 4, y0 + th + lh - 7), f, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"{fi}", (x0 + 4, y0 + th + lh - 7), f, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(od / "selected_frames.png"), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def process(fname, tag=""):
    win = WINDOWS[fname]; video = DATA / fname; od = OUT / (Path(fname).stem + tag)
    od.mkdir(parents=True, exist_ok=True)
    intr = calibrate(win["session"])
    row = {"video": fname, "session": win["session"], "approx": win.get("approx", False)}
    if intr is None:
        row["status"] = "NO_CALIBRATION"; return row
    row["calib_rms"] = round(intr.get("rms_reproj_px", -1), 4)
    mask, _ = content_mask(video)
    L, diff, shas = scan(video, mask, win["sub_out"][1] + 5)
    valid = L >= DARK_L
    picks = sorted(set(
        flow_pick(valid, diff, win["sub_in"][0], win["sub_in"][1], K_IN) +
        flow_pick(valid, diff, win["sub_out"][0], win["sub_out"][1], K_OUT) +
        flow_pick(valid, diff, win["sub_in"][1] + 1, win["sub_out"][0] - 1, K_BRIDGE)))
    img_dir, names, hm = extract(video, picks, mask, shas, od)
    row["n_picked"] = len(names); row["hash_match"] = f"{hm}/{len(names)}"
    row["n_sub_in"] = sum(1 for p in picks if seg_of(p, win) == "sub_in")
    row["n_sub_out"] = sum(1 for p in picks if seg_of(p, win) == "sub_out")
    row["n_bridge"] = sum(1 for p in picks if seg_of(p, win) == "bridge")
    try:
        sfm = run_sfm(img_dir, names, intr, od)
        ana = analyze(sfm, win)
        # connectivity: count sub-models
        models = od / "sfm" / "models"
        n_comp = len([d for d in models.iterdir() if d.is_dir()]) if models.exists() else 1
        ana["n_components"] = max(n_comp, 1); ana["frames_in_largest"] = ana["n_reg"]
        plots(od, fname, ana, picks, win, img_dir)
        for k in ("_cc", "_seg", "_tri"):
            ana.pop(k, None)
        row.update(ana); row["status"] = "OK"
    except Exception as e:
        row["status"] = f"SFM_FAIL:{type(e).__name__}:{str(e)[:140]}"
    (od / "metrics.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=0), flush=True)
    return row


def main():
    global K_IN, K_OUT, K_BRIDGE
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", default=",".join(WINDOWS))
    ap.add_argument("--k-in", type=int, default=K_IN)
    ap.add_argument("--k-out", type=int, default=K_OUT)
    ap.add_argument("--k-bridge", type=int, default=K_BRIDGE)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    K_IN, K_OUT, K_BRIDGE = args.k_in, args.k_out, args.k_bridge
    print(f"K_IN={K_IN} K_OUT={K_OUT} K_BRIDGE={K_BRIDGE} tag='{args.tag}'")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.videos.split(","):
        if v not in WINDOWS:
            continue
        print(f"\n######## {v} ########", flush=True)
        rows.append(process(v, args.tag))
    ok = [r for r in rows if r.get("status") == "OK"]
    ok.sort(key=lambda r: (r.get("subcord_median_tri_deg") or -1,
                           r.get("subcord_p95_tri_deg") or -1,
                           r.get("subcord_lat_along") or -1), reverse=True)
    (OUT / "phase2_table.json").write_text(json.dumps(rows, indent=2))
    print("\n\n=== PHASE 2 TABLE (sorted: median tri, then p95 tri, then lat/along) ===")
    h = (f"{'video':<9}{'reg/N':>7}{'reproj':>8}{'lat/along':>10}{'maxAx':>7}{'medAx':>7}"
         f"{'medTri':>8}{'p95Tri':>8}{'pts':>6}{'comp':>5}{'largest':>8}")
    print(h); print("-" * len(h))
    for r in ok + [r for r in rows if r.get("status") != "OK"]:
        if r.get("status") != "OK":
            print(f"{r['video']:<9}  {r.get('status')}"); continue
        print(f"{r['video']:<9}{str(r['n_reg'])+'/'+str(r['n_picked']):>7}{r['reproj_px']:>8.2f}"
              f"{(r['subcord_lat_along'] or 0):>10.3f}{(r['subcord_max_axis_deg'] or 0):>7.2f}"
              f"{(r['subcord_median_axis_deg'] or 0):>7.2f}{(r['subcord_median_tri_deg'] or 0):>8.2f}"
              f"{(r['subcord_p95_tri_deg'] or 0):>8.2f}{r['subcord_n_points']:>6}"
              f"{r['n_components']:>5}{r['frames_in_largest']:>8}")
    print(f"\nsaved {OUT/'phase2_table.json'}")


if __name__ == "__main__":
    main()
