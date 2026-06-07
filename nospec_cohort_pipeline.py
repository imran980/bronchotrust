"""End-to-end no-specular Stage D pipeline, parameterized per video.

Per-video steps:
  1. Detect forward-pass end via smoothed dark-fraction peak.
  2. Compute 4 anatomical bins anchored to subglottic_kf + forward_end.
  3. Stage A: Farneback flow on every frame in [0..forward_end] (half-res
     for speed). Pick frames per bin separated by >= MIN_CUM_FLOW.
  4. Stage B: lumen-geometry similarity filter (median-pm-30%).
  5. Stage D no-specular: CLAHE-3.5 + 1px bezel mask (NO specular).
  6. HLoc SuperPoint(max) extraction + LightGlue exhaustive matching +
     pycolmap reconstruction. Mapper.min_num_matches=8, init_min_tri=4,
     etc.
  7. DCE at L1/L2/L3 (slab +/-5 scene units), bootstrap sigma.

Per video: writes runs/barbour/<vid>/colmap_recon_stageD_noSpec/
including _noSpec_summary.json and _dce.json.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path("/home/mi3dr/projects/bronchotrust")
SRC = Path("/home/mi3dr/dataset/validation-videos/First 13 Videos")


def video_path(vid: str) -> Path:
    for ext in ("mp4", "MP4"):
        p = SRC / f"{vid}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(vid)


def detect_forward_end(vid: str) -> tuple[int, int]:
    """Smooth dark fraction (L<30) per-frame across central FOV; return
    (forward_end, total_frames). forward_end = arg-max of smoothed dark
    fraction (the deepest tunnel-ahead moment)."""
    intr = json.loads(
        (PROJECT / f"runs/barbour/{vid}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r = float(intr["fov_radius"])
    cap = cv2.VideoCapture(str(video_path(vid)))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yy, xx = np.indices((H, W), dtype=np.float32)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.85) ** 2
    dark_frac = []; idxs = []
    for fi in range(0, n_total, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: continue
        L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0][mask]
        dark_frac.append(float((L < 30).sum() / L.size))
        idxs.append(fi)
    cap.release()
    arr = np.array(dark_frac); idxs = np.array(idxs)
    sm = np.convolve(arr, np.ones(15) / 15, mode="same")
    forward_end = int(idxs[np.argmax(sm)])
    return forward_end, n_total


def compute_bins(subglottic_kf: int, forward_end: int):
    """Return 4 anatomical bins. Subglottic_kf is the manually-picked
    post-vocal-cord landmark. Ensure each bin spans a reasonable frame
    range so flow-based picks can land in it."""
    # Guarantee vocal_cord covers at least 50 frames if at all possible
    vc_end = max(subglottic_kf, 50)
    sg_end = min(vc_end + max(150, int(0.15 * forward_end)),
                  int(0.55 * forward_end))
    mt_end = max(int(0.85 * forward_end), sg_end + 100)
    return [
        ("vocal_cord", 0, vc_end),
        ("subglottic", vc_end, sg_end),
        ("mid_trachea", sg_end, mt_end),
        ("main_carina", mt_end, forward_end),
    ]


def stage_a_picks(vid: str, forward_end: int, bins, per_bin=8,
                   min_cum_flow=100.0, base_dir: Path = None):
    base = base_dir or (PROJECT / f"runs/barbour/{vid}/colmap_recon_stageA")
    base.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path(vid)))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(forward_end, n_total - 1)
    grays = []
    for fi in range(0, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
        grays.append((fi, g))
    cap.release()
    flow_mag = np.zeros(len(grays), dtype=np.float32)
    for i in range(1, len(grays)):
        g1, g2 = grays[i - 1][1], grays[i][1]
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.linalg.norm(flow, axis=2)
        valid = (g1 > 5) & (g2 > 5)
        flow_mag[i] = float(np.median(mag[valid]) * 2) if valid.any() else 0
    cum_flow = np.cumsum(flow_mag)

    picks = []
    for name, lo, hi in bins:
        bin_idxs = [i for i, (fi, _) in enumerate(grays) if lo <= fi < hi]
        last_cum = -np.inf; bin_picks = []
        for i in bin_idxs:
            if cum_flow[i] - last_cum >= min_cum_flow:
                bin_picks.append((grays[i][0], float(cum_flow[i])))
                last_cum = cum_flow[i]
        bin_picks = bin_picks[:per_bin]
        for fi, cf in bin_picks:
            picks.append({"vid": vid, "frame_idx": int(fi),
                          "bin": name, "cum_flow_px": cf,
                          "filename": f"{vid}_f{int(fi):06d}.png"})

    np.savez(base / "_stageA_flow.npz",
             flow_mag=flow_mag, cum_flow=cum_flow,
             frame_idxs=np.array([g[0] for g in grays]))
    (base / "_stageA_picks.json").write_text(json.dumps({
        "vid": vid, "forward_end": forward_end,
        "bins": [{"name": b, "lo": l, "hi": h} for b, l, h in bins],
        "per_bin_target": per_bin, "min_cum_flow_px": min_cum_flow,
        "n_picks": len(picks), "picks": picks,
    }, indent=2))
    return picks


def stage_b_lumen_filter(vid: str, picks, max_dev_pct=30.0,
                          base_dir: Path = None):
    base = base_dir or (PROJECT / f"runs/barbour/{vid}/colmap_recon_stageA")
    intr = json.loads(
        (PROJECT / f"runs/barbour/{vid}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r_fov = float(intr["fov_radius"])
    cap = cv2.VideoCapture(str(video_path(vid)))
    rows = []
    for e in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_idx"])
        ok, frame = cap.read()
        if not ok: continue
        H, W = frame.shape[:2]
        yy, xx = np.indices((H, W), dtype=np.float32)
        fov = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_fov * 0.92) ** 2
        L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
        thresh = float(np.percentile(L[fov], 10))
        dark = ((L < thresh) & fov).astype(np.uint8)
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8)
        if n_lab < 2:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[big, cv2.CC_STAT_AREA])
        eq_r = float(np.sqrt(area / np.pi))
        rows.append({**e, "lumen_eq_r_px": eq_r, "kept": True})
    cap.release()
    valid_r = np.array([r["lumen_eq_r_px"] for r in rows])
    if len(valid_r) == 0:
        return []
    med_r = float(np.median(valid_r))
    for r in rows:
        dev = abs(r["lumen_eq_r_px"] - med_r) / max(med_r, 1e-9) * 100
        r["lumen_dev_pct"] = float(dev)
        if dev > max_dev_pct:
            r["kept"] = False
    (base / "_stageB_filter.json").write_text(json.dumps({
        "vid": vid, "median_lumen_eq_r_px": med_r,
        "max_dev_pct": max_dev_pct, "n_kept": sum(1 for r in rows if r["kept"]),
        "n_total": len(rows), "frames": rows,
    }, indent=2))
    return [r for r in rows if r["kept"]]


def preprocess_no_specular(vid: str, kept_entries, base: Path,
                            clip=3.5, bezel_erode_px=1):
    intr = json.loads(
        (PROJECT / f"runs/barbour/{vid}/intrinsics.json").read_text())
    cx, cy = intr["fov_center"]; r_fov = float(intr["fov_radius"])
    Wimg, Himg = intr["image_size"]
    img_dir = base / "images"; msk_dir = base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); msk_dir.mkdir(parents=True)
    yy, xx = np.indices((Himg, Wimg), dtype=np.float32)
    r_inner = r_fov - bezel_erode_px
    bezel_inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r_inner ** 2
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(video_path(vid)))
    for e in kept_entries:
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_idx"])
        ok, f = cap.read()
        if not ok: continue
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        L_clahe = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = L_clahe
        enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enh[~bezel_inside] = 0
        cv2.imwrite(str(img_dir / e["filename"]), enh,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{e['filename']}.png"),
                    (bezel_inside.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cap.release()
    dist_full = intr.get("dist_full")
    if dist_full and len(dist_full) >= 4:
        k1, k2, p1, p2 = float(dist_full[0]), float(dist_full[1]), \
                         float(dist_full[2]), float(dist_full[3])
    else:
        k1, k2, p1, p2 = intr["k1"], intr["k2"], 0.0, 0.0
    return {"model": "OPENCV", "width": Wimg, "height": Himg,
            "params": [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                        k1, k2, p1, p2]}


def algebraic_circle_fit(pts_2d):
    x = pts_2d[:, 0]; y = pts_2d[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = -sol[0] / 2.0; cy = -sol[1] / 2.0
    r2 = cx * cx + cy * cy - sol[2]
    if r2 <= 0: return float("nan"), float("nan"), float("nan")
    return float(cx), float(cy), float(np.sqrt(r2))


def bootstrap_circle(pts_2d, n_boot=1000):
    rng = np.random.default_rng(0); n = len(pts_2d)
    radii = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, r = algebraic_circle_fit(pts_2d[idx])
        if np.isfinite(r): radii.append(r)
    if not radii: return float("nan"), float("nan")
    return float(np.median(radii)), float(np.std(radii))


def central_tangent(centers, i):
    if i == 0: v = centers[1] - centers[0]
    elif i == len(centers) - 1: v = centers[-1] - centers[-2]
    else: v = centers[i + 1] - centers[i - 1]
    return v / max(np.linalg.norm(v), 1e-9)


def fit_dce(pts, cam, T, slab):
    rel = pts - cam; axial = rel @ T
    mask = np.abs(axial) <= slab
    n_slab = int(mask.sum())
    if n_slab < 5: return float("nan"), float("nan"), n_slab
    in_plane = pts[mask] - cam
    in_plane = in_plane - np.outer(in_plane @ T, T)
    up = np.array([0.0, 0.0, 1.0])
    if abs(T @ up) > 0.95: up = np.array([0.0, 1.0, 0.0])
    e1 = up - (up @ T) * T; e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(T, e1)
    pts_2d = np.stack([in_plane @ e1, in_plane @ e2], axis=1)
    _, _, r_fit = algebraic_circle_fit(pts_2d)
    _, r_sigma = bootstrap_circle(pts_2d)
    return 2 * r_fit, 2 * r_sigma, n_slab


def run_pipeline(vid: str, subglottic_kf: int):
    print(f"\n{'='*70}\n {vid}  (subglottic_kf={subglottic_kf})\n{'='*70}")
    forward_end, n_total = detect_forward_end(vid)
    print(f"[{vid}] forward_end={forward_end}  (total={n_total})")

    out_base = PROJECT / f"runs/barbour/{vid}/colmap_recon_stageD_noSpec"
    out_base.mkdir(parents=True, exist_ok=True)
    for f in out_base.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)

    bins = compute_bins(subglottic_kf, forward_end)
    print(f"[{vid}] bins: {bins}")

    stageA_dir = PROJECT / f"runs/barbour/{vid}/colmap_recon_stageA_v2"
    picks = stage_a_picks(vid, forward_end, bins, per_bin=8,
                           min_cum_flow=50.0, base_dir=stageA_dir)
    print(f"[{vid}] Stage A: {len(picks)} picks")
    kept = stage_b_lumen_filter(vid, picks, base_dir=stageA_dir)
    print(f"[{vid}] Stage B: {len(kept)} kept after lumen filter")
    if len(kept) < 6:
        print(f"[{vid}] FAIL: <6 frames usable")
        return {"vid": vid, "fail_reason": "too_few_curated_frames",
                "n_kept": len(kept)}

    intr_doc = preprocess_no_specular(vid, kept, out_base)
    print(f"[{vid}] preprocessed (CLAHE-3.5, 1px bezel, no specular)")

    # ---- HLoc + pycolmap ----
    from hloc import extract_features, match_features, pairs_from_exhaustive
    from hloc import reconstruction as hloc_recon
    image_list = sorted([e["filename"] for e in kept])
    images_dir = out_base / "images"
    sfm_pairs = out_base / "pairs-exhaustive.txt"
    sfm_dir = out_base / "sfm"; sfm_dir.mkdir(parents=True, exist_ok=True)
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    print(f"[{vid}] HLoc SuperPoint ...")
    features_path = extract_features.main(
        feature_conf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    print(f"[{vid}] HLoc LightGlue matching ...")
    matches_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], out_base)
    print(f"[{vid}] pycolmap reconstruction ...")
    try:
        model = hloc_recon.main(
            sfm_dir, images_dir, sfm_pairs, features_path, matches_path,
            image_list=image_list,
            camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
            verbose=False,
            mapper_options={
                "min_num_matches": 8,
                "ba_local_max_num_iterations": 50,
                "ba_global_max_num_iterations": 100,
                "multiple_models": True, "max_num_models": 50,
                "min_model_size": 3,
                "mapper": {
                    "init_min_tri_angle": 4.0, "init_max_error": 8.0,
                    "init_min_num_inliers": 15,
                    "init_max_forward_motion": 0.99,
                    "abs_pose_max_error": 20.0,
                    "abs_pose_min_num_inliers": 15,
                    "abs_pose_min_inlier_ratio": 0.1,
                    "filter_max_reproj_error": 8.0,
                    "filter_min_tri_angle": 1.0,
                },
            })
    except Exception as e:
        print(f"[{vid}] reconstruction error: {e}")
        return {"vid": vid, "fail_reason": "recon_exception", "exc": str(e)}
    if model is None:
        return {"vid": vid, "fail_reason": "recon_none"}

    n_reg = model.num_reg_images(); n_pts = model.num_points3D()
    pct = 100 * n_reg / len(image_list)
    print(f"[{vid}] registered: {n_reg}/{len(image_list)} ({pct:.1f}%)")
    print(f"[{vid}] sparse 3D points: {n_pts}")

    # Landmarks + DCE
    centers, fids = [], []
    for img in model.images.values():
        centers.append(np.array(img.projection_center()))
        m = re.search(r"f(\d+)\.png", img.name)
        fids.append(int(m.group(1)) if m else -1)
    order = np.argsort(fids)
    centers = np.array([centers[i] for i in order])
    fids = np.array([fids[i] for i in order])
    pts = np.array([list(p.xyz) for p in model.points3D.values()])
    np.savez(out_base / "_geometry.npz", centers=centers, fids=fids, pts=pts)

    bin_of = lambda f: next(
        (n for n, lo, hi in bins if lo <= f < hi), "after")
    cam_bins = [bin_of(int(f)) for f in fids]
    vc = [i for i, b in enumerate(cam_bins) if b == "vocal_cord"]
    sg = [i for i, b in enumerate(cam_bins) if b == "subglottic"]
    L1 = vc[0] if vc else 0
    L2 = sg[0] if sg else (L1 + 1 if L1 + 1 < len(centers) else L1)
    L3 = len(centers) - 1
    dce_rows = []
    for L, lab in [(L1, "L1 glottis"), (L2, "L2 prox subglottis"),
                     (L3, "L3 distal pre-carina")]:
        T = central_tangent(centers, L)
        dce, sig, n_slab = fit_dce(pts, centers[L], T, slab=5.0)
        dce_rows.append({"landmark": lab, "frame_idx": int(fids[L]),
                          "n_pts_slab_pm5u": n_slab,
                          "DCE_units": float(dce) if np.isfinite(dce) else None,
                          "sigma_units": float(sig) if np.isfinite(sig) else None})
        print(f"  {lab:<24} f={int(fids[L]):>4}  n={n_slab:>3}  "
              f"DCE={dce:.3f} +/- {sig:.3f} units")

    summary = {
        "vid": vid, "subglottic_kf": subglottic_kf,
        "forward_end": forward_end,
        "bins": [{"name": n, "lo": l, "hi": h} for n, l, h in bins],
        "n_curated_input": len(image_list),
        "n_registered": n_reg, "pct_registered": pct,
        "n_sparse_points": n_pts,
        "landmark_frames": {"L1": int(fids[L1]), "L2": int(fids[L2]),
                              "L3": int(fids[L3])},
        "dce_rows": dce_rows,
    }
    (out_base / "_noSpec_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--subglottic_kf", type=int, required=True)
    args = ap.parse_args()
    run_pipeline(args.vid, args.subglottic_kf)


if __name__ == "__main__":
    main()
