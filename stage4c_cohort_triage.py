"""Cohort reconstructability triage on the First 15 Videos folder.

Lightweight per-video SfM (poses only, NO MVS):
  - auto-pick sub-cord frames: 20 evenly-spaced indices in window [80, 500],
    drop those whose content-ROI LAB-L mean < 30
  - SuperPoint+LightGlue (HLoc), exhaustive matcher
  - COLMAP incremental mapping with PINNED per-session OPENCV intrinsics,
    init_min_tri_angle=4.0, min_num_matches=8, default init (no forced pair)
  - keep the largest sub-model

For each video report:
  n_picked, n_registered, viewing-cone total angular extent (max pairwise
  axis angle), camera position bbox diagonal, lateral-vs-along-axis
  translation ratio, mean reprojection error.

Verdict per video:
  VIABLE        : cone > 15 deg AND lateral/along > 0.30
  MARGINAL      : 5 <= cone <= 15 deg
  NOT_RECON     : cone < 5 deg  (parallax-zero, same failure mode as 2_v2)
  NO_CALIBRATION: per-session intrinsics missing
"""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
DATASET = Path("/home/mi3dr/dataset/validation-videos")
VIDEOS_15 = DATASET / "First 15 Videos"
VIDEOS_TRIMMED = DATASET / "First 13 Videos Trimmed"
OUT_ROOT = PROJECT / "runs/gated2/cohort_triage"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
COLMAP = Path("/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap")

COHORT = ["2_v1", "2_v2", "5_v1", "7_v1", "10_v2", "11_v2", "13_v2", "15_v2"]

# Sub-cord window is now RELATIVE to first sustained-lit frame.
SUB_CORD_REL_LO = 30          # skip cord-passage
SUB_CORD_REL_HI = 350         # rough end of upper-trachea
N_PICK = 20
DARK_L_THRESH = 30.0
MIN_REAL_FILE_BYTES = 100_000  # below this we treat the file as a stub


def find_video(vid: str) -> Path | None:
    """Search First 15 Videos (uppercase-dash) first, then First 13 Videos
    Trimmed (lowercase-underscore). Treat sub-MB files as stubs."""
    candidates = []
    base15 = vid.replace("_", "-").upper()
    for ext in ("MP4", "mp4", "MOV", "mov"):
        candidates.append(VIDEOS_15 / f"{base15}.{ext}")
    for ext in ("mp4", "MP4", "mov", "MOV"):
        candidates.append(VIDEOS_TRIMMED / f"{vid}.{ext}")
    for p in candidates:
        if p.exists() and p.stat().st_size >= MIN_REAL_FILE_BYTES:
            return p
    return None


def load_intrinsics(vid: str):
    p = PROJECT / f"runs/barbour/{vid}/intrinsics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def detect_content_mask_from_intr(intr):
    Wi, Hi = intr["image_size"]
    cx, cy = intr["fov_center"]
    r = float(intr["fov_radius"]) - 4.0
    yy, xx = np.indices((Hi, Wi), dtype=np.float32)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def auto_pick_subcord(video_path: Path, mask: np.ndarray):
    """Two-pass:
    Pass 1: scan all frames sequentially; record L_roi for each, find the
            first sustained-lit run of >=5 consecutive frames with L_roi>=30.
            That run start = 'cords-passage'. The sub-cord window is at
            [run_start + SUB_CORD_REL_LO, run_start + SUB_CORD_REL_HI] but
            clipped to video length.
    Pass 2: sample N_PICK frames evenly in the window; keep only those with
            L_roi >= DARK_L_THRESH.
    """
    cap = cv2.VideoCapture(str(video_path))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if N <= 0:
        cap.release()
        return [], 0, None
    L_per = np.full(N, np.nan, dtype=np.float32)
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi >= N: break
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        L_per[fi] = float(L[mask].mean())
        fi += 1
    cap.release()
    N_read = fi
    lit = (L_per[:N_read] >= DARK_L_THRESH).astype(np.int32)
    # Find first run of >=5 consecutive lit
    run_start = None
    run = 0
    for i, v in enumerate(lit):
        if v: run += 1
        else: run = 0
        if run >= 5:
            run_start = i - 4; break
    if run_start is None:
        return [], N_read, None
    lo = min(run_start + SUB_CORD_REL_LO, N_read - 2)
    hi = min(run_start + SUB_CORD_REL_HI, N_read - 1)
    if hi <= lo + 5:
        return [], N_read, run_start

    targets = np.linspace(lo, hi, N_PICK).round().astype(int).tolist()
    target_set = set(targets)
    cap = cv2.VideoCapture(str(video_path))
    picked = []
    fi = 0
    while target_set:
        ok, fr = cap.read()
        if not ok or fi > max(target_set): break
        if fi in target_set and L_per[fi] >= DARK_L_THRESH:
            picked.append((fi, fr.copy(), float(L_per[fi])))
            target_set.discard(fi)
        elif fi in target_set:
            target_set.discard(fi)
        fi += 1
    cap.release()
    return picked, N_read, run_start


def preprocess(out_base: Path, picked, mask, vid: str):
    img_dir = out_base / "images"
    msk_dir = out_base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); msk_dir.mkdir(parents=True)
    extracted = []
    for fi, img, L_roi in picked:
        img2 = img.copy()
        img2[~mask] = 0
        name = f"{vid}_f{fi:06d}.png"
        cv2.imwrite(str(img_dir / name), img2,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (mask.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        extracted.append({"frame_idx": int(fi), "filename": name,
                           "L_roi": L_roi})
    return extracted


def cam_params_str(intr):
    params = [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
              intr["k1"], intr["k2"], 0.0, 0.0]
    return ",".join(f"{p:.10g}" for p in params)


def run_hloc(out_base, image_list):
    from hloc import extract_features, match_features, pairs_from_exhaustive
    images_dir = out_base / "images"
    pairs_path = out_base / "pairs-exhaustive.txt"
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    feats = extract_features.main(
        feature_conf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    matches = match_features.main(
        matcher_conf, pairs_path, feature_conf["output"], out_base)
    return feats, matches, pairs_path


def reconstruct_poses(out_base, image_list, intr, feats, matches, sfm_pairs):
    from hloc import reconstruction as hloc_recon
    sfm_dir = out_base / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_base / "images"
    cam_params = cam_params_str(intr)
    image_options = {"camera_model": "OPENCV", "camera_params": cam_params}
    mapper_options = {
        "min_num_matches": 8,
        "ba_local_max_num_iterations": 50,
        "ba_global_max_num_iterations": 100,
        "multiple_models": True, "max_num_models": 50, "min_model_size": 3,
        "ba_refine_focal_length": False,
        "ba_refine_extra_params": False,
        "ba_refine_principal_point": False,
        "mapper": {
            "init_min_tri_angle": 4.0,
            "init_max_error": 8.0,
            "init_min_num_inliers": 15,
            "init_max_forward_motion": 0.99,
            "abs_pose_max_error": 20.0,
            "abs_pose_min_num_inliers": 15,
            "abs_pose_min_inlier_ratio": 0.1,
            "filter_max_reproj_error": 8.0,
            "filter_min_tri_angle": 1.0,
        },
    }
    try:
        model = hloc_recon.main(
            sfm_dir, images_dir, sfm_pairs, feats, matches,
            image_list=image_list,
            camera_mode=hloc_recon.pycolmap.CameraMode.SINGLE,
            image_options=image_options,
            verbose=False, mapper_options=mapper_options)
    except Exception as e:
        print(f"  recon error: {e}")
        return None
    return model


def analyze_poses(sfm_dir: Path):
    sub = r"""
import pycolmap, json, sys, re, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
out = {"n_reg": int(rec.num_reg_images()),
        "n_sparse": int(rec.num_points3D())}
errs = [float(p.error) for p in rec.points3D.values()]
out["mean_reprojection_error_px"] = float(np.mean(errs)) if errs else None
imgs = []
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    cfw = img.cam_from_world()
    M = np.array(cfw.matrix())
    R = M[:3,:3]; t = M[:3,3]
    C = (-R.T @ t)
    a = R.T @ np.array([0,0,1.0])
    a = a / max(np.linalg.norm(a), 1e-9)
    imgs.append({"name": img.name, "frame_idx": fi,
                  "C": C.tolist(), "axis": a.tolist()})
imgs.sort(key=lambda r: r["frame_idx"])
out["registered"] = imgs
cam = next(iter(rec.cameras.values()))
out["camera_model"] = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
out["camera_params"] = list(cam.params)
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(sfm_dir)],
                          capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def cone_stats(centers, axes):
    A = np.array(axes); C = np.array(centers)
    if len(A) < 2:
        return None
    mean_v = A.sum(0)
    mean_v = mean_v / max(np.linalg.norm(mean_v), 1e-9)
    dots = np.clip(A @ A.T, -1.0, 1.0)
    np.fill_diagonal(dots, 1.0)
    ang_pp = np.degrees(np.arccos(dots))
    np.fill_diagonal(ang_pp, 0.0)
    cone_total = float(ang_pp.max())
    cone_half = float(np.degrees(np.arccos(np.clip(A @ mean_v, -1, 1))).max())
    bbox = (C.max(0) - C.min(0)).tolist()
    bbox_diag = float(np.linalg.norm(C.max(0) - C.min(0)))
    # lateral vs along-axis translation
    cc = C - C.mean(0)
    along = cc @ mean_v             # along mean optical axis
    lateral_vec = cc - np.outer(along, mean_v)
    lateral_mag = np.linalg.norm(lateral_vec, axis=1)
    along_span = float(along.max() - along.min())
    lateral_span = float(lateral_mag.max() - lateral_mag.min())
    lateral_full = float(2 * lateral_mag.max())  # diameter of lateral spread
    ratio = (lateral_full / along_span) if along_span > 1e-6 else float("inf")
    return {
        "n_cams": int(len(A)),
        "mean_axis_world": mean_v.tolist(),
        "cone_half_angle_deg": cone_half,
        "cone_total_extent_deg": cone_total,
        "position_bbox_xyz": bbox,
        "position_bbox_diag": bbox_diag,
        "along_axis_span": along_span,
        "lateral_diameter": lateral_full,
        "lateral_over_along_ratio": ratio,
    }


def triage_one(vid: str, results: dict):
    print(f"\n=== {vid} ===")
    intr = load_intrinsics(vid)
    if intr is None:
        print("  NO_CALIBRATION; skip")
        results[vid] = {"vid": vid, "status": "NO_CALIBRATION"}
        return
    if intr["rms_reproj_px"] >= 0.5:
        print(f"  calibration RMS {intr['rms_reproj_px']:.3f} >= 0.5; skip")
        results[vid] = {"vid": vid, "status": "BAD_CALIBRATION",
                         "rms": intr["rms_reproj_px"]}
        return
    vp = find_video(vid)
    if vp is None:
        print("  video file not found; skip")
        results[vid] = {"vid": vid, "status": "VIDEO_MISSING"}
        return

    out = OUT_ROOT / vid
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)

    mask = detect_content_mask_from_intr(intr)
    picked, n_total, run_start = auto_pick_subcord(vp, mask)
    rs_str = f"first-lit-run@{run_start}" if run_start is not None else "no-lit-run"
    print(f"  source: {vp.name}  N={n_total}  {rs_str}  "
          f"window [+{SUB_CORD_REL_LO},+{SUB_CORD_REL_HI}]  "
          f"picked {len(picked)} non-dark frames")
    if len(picked) < 6:
        print("  too few frames after dark-drop; skip")
        results[vid] = {"vid": vid, "status": "TOO_FEW_LIT_FRAMES",
                         "n_picked": len(picked)}
        return

    extracted = preprocess(out, picked, mask, vid)
    image_list = sorted([e["filename"] for e in extracted])
    feats, matches, pairs = run_hloc(out, image_list)
    model = reconstruct_poses(out, image_list, intr, feats, matches, pairs)
    n_reg = model.num_reg_images() if model else 0
    print(f"  registered: {n_reg}/{len(extracted)}")
    if n_reg < 4:
        results[vid] = {"vid": vid, "status": "NOT_REGISTERED",
                         "n_picked": len(extracted), "n_registered": n_reg,
                         "intrinsics_rms": intr["rms_reproj_px"]}
        return

    info = analyze_poses(out / "sfm")
    centers = [im["C"] for im in info["registered"]]
    axes = [im["axis"] for im in info["registered"]]
    cs = cone_stats(centers, axes)
    cone = cs["cone_total_extent_deg"]
    ratio = cs["lateral_over_along_ratio"]
    if cone >= 15.0 and ratio >= 0.30:
        status = "VIABLE"
    elif cone >= 5.0:
        status = "MARGINAL"
    else:
        status = "NOT_RECON"
    print(f"  cone total = {cone:.2f}°  bbox_diag = {cs['position_bbox_diag']:.3f}  "
          f"lateral/along = {ratio:.3f}  -> {status}")
    results[vid] = {
        "vid": vid, "status": status,
        "source_video": str(vp),
        "video_n_frames": n_total,
        "first_lit_run_start": (int(run_start) if run_start is not None else None),
        "n_picked": len(extracted),
        "n_registered": int(n_reg),
        "intrinsics_rms": intr["rms_reproj_px"],
        "mean_reprojection_error_px": info["mean_reprojection_error_px"],
        "cone_total_extent_deg": cone,
        "cone_half_angle_deg": cs["cone_half_angle_deg"],
        "position_bbox_xyz": cs["position_bbox_xyz"],
        "position_bbox_diag": cs["position_bbox_diag"],
        "along_axis_span": cs["along_axis_span"],
        "lateral_diameter": cs["lateral_diameter"],
        "lateral_over_along_ratio": ratio,
        "n_sparse_points": info["n_sparse"],
        "final_camera_params": info["camera_params"],
        "registered_frame_indices": [im["frame_idx"] for im in info["registered"]],
    }


def main():
    results = {}
    for vid in COHORT:
        try:
            triage_one(vid, results)
        except Exception as e:
            print(f"  ERROR on {vid}: {e}")
            results[vid] = {"vid": vid, "status": "ERROR", "error": str(e)}
    (OUT_ROOT / "triage_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_ROOT/'triage_results.json'}")
    # Print ranked table (by viability + cone)
    ranked = sorted(
        results.values(),
        key=lambda r: (
            {"VIABLE": 0, "MARGINAL": 1, "NOT_RECON": 2,
              "NOT_REGISTERED": 3, "TOO_FEW_LIT_FRAMES": 4,
              "BAD_CALIBRATION": 5, "NO_CALIBRATION": 6,
              "VIDEO_MISSING": 7, "ERROR": 8}.get(r.get("status"), 9),
            -r.get("cone_total_extent_deg", 0),
        ))
    print("\n=== RANKED ===")
    print(f"{'vid':<8}{'status':<14}{'n_reg':>6}{'cone°':>8}{'bbox_diag':>11}"
          f"{'lat/along':>11}{'reproj_px':>11}")
    for r in ranked:
        v = r["vid"]; st = r["status"]
        nr = r.get("n_registered", "")
        cone = r.get("cone_total_extent_deg")
        bbd = r.get("position_bbox_diag")
        lr = r.get("lateral_over_along_ratio")
        rp = r.get("mean_reprojection_error_px")
        def fmt(x, w, d=2):
            return f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f"{v:<8}{st:<14}{fmt(nr,6,0)}{fmt(cone,8)}{fmt(bbd,11,3)}"
              f"{fmt(lr,11,3)}{fmt(rp,11,3)}")


if __name__ == "__main__":
    main()
