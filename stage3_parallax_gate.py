"""STAGE 3 (gated plan): parallax gate for 15_v2 -- SfM POSES ONLY, no MVS.

The make-or-break pre-check. Runs SuperPoint+LightGlue + COLMAP incremental
mapper on the Stage-1 curated frames with PINNED intrinsics, then measures the
viewing-cone extent (max pairwise optical-axis angle) and lateral/along ratio.

GATE (matches stage4c triage thresholds so the number is comparable to 17.5):
  VIABLE     : cone > 15 deg AND lateral/along > 0.30
  MARGINAL   : 5 <= cone <= 15
  NOT_RECON  : cone < 5   -> STOP the whole plan (parallax-zero, like 2_v2's 2 deg)

Reports cone over (a) ALL registered cameras and (b) the SUB-CORD cameras, and
whether the insertion AND withdrawal sub-cord frames landed in ONE connected
model (the dual-pass fusion that gives the subglottic ring its parallax).

Run in the `depth-eval` env (hloc + pycolmap). No colmap binary needed.

Outputs (runs/gluemap_15v2/15_v2/stage3_colmap/):
  sfm/ ...                 COLMAP sparse model
  stage3_results.json      registered frames, cone stats, connectivity, verdict
  cameras_3view.png        camera centers + optical axes (3 projections)
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
RUN = PROJECT / "runs/gluemap_15v2/15_v2"
STAGE1 = RUN / "stage1"
STAGE2 = RUN / "stage2"
STAGE0 = RUN / "stage0"
OUT = RUN / "stage3_colmap"


def cam_params_str(intr):
    return ",".join(f"{p:.10g}" for p in
                    [intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                     intr["k1"], intr["k2"], 0.0, 0.0])


def preprocess(out_base, src_images, mask):
    img_dir = out_base / "images"
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)
    bezel = mask > 0
    names = []
    for p in sorted(src_images.glob("f*.png")):
        img = cv2.imread(str(p))
        img[~bezel] = 0
        cv2.imwrite(str(img_dir / p.name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        names.append(p.name)
    return img_dir, sorted(names)


def run_hloc(out_base, image_list):
    from hloc import extract_features, match_features, pairs_from_exhaustive
    images_dir = out_base / "images"
    pairs_path = out_base / "pairs-exhaustive.txt"
    fconf = extract_features.confs["superpoint_max"]
    mconf = match_features.confs["superpoint+lightglue"]
    print("  HLoc SuperPoint ...", flush=True)
    feats = extract_features.main(fconf, images_dir, out_base, image_list=image_list)
    pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    print("  HLoc LightGlue ...", flush=True)
    matches = match_features.main(mconf, pairs_path, fconf["output"], out_base)
    return feats, matches, pairs_path


def colmap_sfm(out_base, image_list, intr, feats, matches, pairs):
    from hloc import reconstruction as hr
    sfm_dir = out_base / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    image_options = {"camera_model": "OPENCV", "camera_params": cam_params_str(intr)}
    mapper_options = {
        "min_num_matches": 8, "multiple_models": True, "max_num_models": 50,
        "min_model_size": 3,
        "ba_refine_focal_length": False, "ba_refine_extra_params": False,
        "ba_refine_principal_point": False,
        "mapper": {"init_min_tri_angle": 4.0, "init_max_error": 8.0,
                   "init_min_num_inliers": 15, "init_max_forward_motion": 0.99,
                   "abs_pose_max_error": 20.0, "abs_pose_min_num_inliers": 15,
                   "abs_pose_min_inlier_ratio": 0.1,
                   "filter_max_reproj_error": 8.0, "filter_min_tri_angle": 1.0}}
    model = hr.main(sfm_dir, out_base / "images", pairs, feats, matches,
                    image_list=image_list,
                    camera_mode=hr.pycolmap.CameraMode.SINGLE,
                    image_options=image_options, verbose=False,
                    mapper_options=mapper_options)
    return model, sfm_dir


def cone_stats(centers, axes):
    centers = np.asarray(centers, float)
    A = np.asarray(axes, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    if len(A) < 2:
        return None
    cosM = np.clip(A @ A.T, -1, 1)
    ang_pp = np.degrees(np.arccos(cosM))
    cone_total = float(ang_pp.max())
    mean_v = A.mean(0)
    mean_v /= (np.linalg.norm(mean_v) + 1e-12)
    cone_half = float(np.degrees(np.arccos(np.clip(A @ mean_v, -1, 1))).max())
    cc = centers - centers.mean(0)
    along = cc @ mean_v
    lateral_mag = np.linalg.norm(cc - np.outer(along, mean_v), axis=1)
    along_span = float(along.max() - along.min())
    lateral_full = float(2 * lateral_mag.max())
    ratio = (lateral_full / along_span) if along_span > 1e-6 else float("inf")
    bbox = centers.max(0) - centers.min(0)
    return {"n": int(len(A)), "cone_total_extent_deg": cone_total,
            "cone_half_angle_deg": cone_half, "along_axis_span": along_span,
            "lateral_diameter": lateral_full, "lateral_over_along_ratio": ratio,
            "position_bbox_diag": float(np.linalg.norm(bbox))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    intr = json.loads((STAGE2 / "intrinsics_pinned.json").read_text())
    assert intr["rms_reproj_px"] < 0.5, "calibration RMS gate failed"
    sel = json.loads((STAGE1 / "stage1_selection.json").read_text())["selection"]
    seg_of = {s["frame_idx"]: s["segment"] for s in sel}
    mask = cv2.imread(str(STAGE0 / "content_mask.png"), cv2.IMREAD_GRAYSCALE)
    print(f"pinned: fx={intr['fx']:.2f} cx={intr['cx']:.1f} cy={intr['cy']:.1f} "
          f"k1={intr['k1']:.4f} RMS={intr['rms_reproj_px']:.4f}")

    img_dir, names = preprocess(OUT, STAGE1 / "images", mask)
    print(f"preprocessed {len(names)} curated frames -> {img_dir}")
    feats, matches, pairs = run_hloc(OUT, names)
    model, sfm_dir = colmap_sfm(OUT, names, intr, feats, matches, pairs)
    if model is None:
        print("=== STAGE 3 VERDICT: NOT_REGISTERED (no model) -> STOP ===")
        (OUT / "stage3_results.json").write_text(json.dumps(
            {"verdict": "NOT_REGISTERED", "n_registered": 0}, indent=2))
        return

    # ---- analyze registered cameras ----
    import pycolmap
    rec = pycolmap.Reconstruction(str(sfm_dir))
    reg = []
    for img in rec.images.values():
        m = re.search(r"f(\d+)\.png", img.name)
        fi = int(m.group(1)) if m else -1
        M = np.array(img.cam_from_world().matrix())
        R, t = M[:3, :3], M[:3, 3]
        C = (-R.T @ t)
        axis = R.T @ np.array([0, 0, 1.0])
        reg.append({"frame_idx": fi, "segment": seg_of.get(fi, "?"),
                    "C": C.tolist(), "axis": axis.tolist()})
    reg.sort(key=lambda r: r["frame_idx"])
    reg_idx = [r["frame_idx"] for r in reg]
    cam = next(iter(rec.cameras.values()))
    cam_params = list(cam.params)
    errs = [float(p.error) for p in rec.points3D.values()]
    mean_reproj = float(np.mean(errs)) if errs else None

    def subset(pred):
        rs = [r for r in reg if pred(r)]
        if len(rs) < 2:
            return None
        return cone_stats([r["C"] for r in rs], [r["axis"] for r in rs])

    cone_all = subset(lambda r: True)
    cone_sub = subset(lambda r: r["segment"].startswith("subglot"))
    cone_sub_in = subset(lambda r: r["segment"] == "subglot_in")
    cone_sub_out = subset(lambda r: r["segment"] == "subglot_out")

    sub_in_reg = [r["frame_idx"] for r in reg if r["segment"] == "subglot_in"]
    sub_out_reg = [r["frame_idx"] for r in reg if r["segment"] == "subglot_out"]
    both_passes = len(sub_in_reg) > 0 and len(sub_out_reg) > 0

    # verdict on the SUB-CORD cone (the ROI), fall back to all if too few subcord
    gate_cone_stats = cone_sub if cone_sub else cone_all
    gate_cone = gate_cone_stats["cone_total_extent_deg"] if gate_cone_stats else 0.0
    gate_ratio = gate_cone_stats["lateral_over_along_ratio"] if gate_cone_stats else 0.0
    if gate_cone >= 15.0 and gate_ratio >= 0.30:
        verdict = "VIABLE"
    elif gate_cone >= 5.0:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_RECON"

    # pin verification
    pin = {"model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
           "params": cam_params,
           "delta_fx": cam_params[0] - intr["fx"],
           "delta_fy": cam_params[1] - intr["fy"],
           "delta_k1": cam_params[4] - intr["k1"] if len(cam_params) > 4 else None}

    res = {
        "verdict": verdict,
        "n_curated": len(names), "n_registered": len(reg),
        "registered_frame_indices": reg_idx,
        "failed_frame_indices": sorted(set(seg_of) - set(reg_idx)),
        "mean_reprojection_error_px": mean_reproj,
        "cone_all_registered": cone_all,
        "cone_subcord": cone_sub,
        "cone_subcord_insertion": cone_sub_in,
        "cone_subcord_withdrawal": cone_sub_out,
        "subcord_insertion_registered": sub_in_reg,
        "subcord_withdrawal_registered": sub_out_reg,
        "dual_pass_in_one_model": both_passes,
        "gate_used": "subcord" if cone_sub else "all",
        "pin_verification": pin,
        "triage_reference_cone_deg": 17.51,
    }
    (OUT / "stage3_results.json").write_text(json.dumps(res, indent=2))

    print(f"\nregistered {len(reg)}/{len(names)}  mean_reproj={mean_reproj:.2f}px")
    if cone_all:
        print(f"cone ALL: {cone_all['cone_total_extent_deg']:.2f} deg  "
              f"lat/along={cone_all['lateral_over_along_ratio']:.2f}")
    if cone_sub:
        print(f"cone SUB-CORD: {cone_sub['cone_total_extent_deg']:.2f} deg  "
              f"lat/along={cone_sub['lateral_over_along_ratio']:.2f} "
              f"(n={cone_sub['n']})")
    print(f"sub-cord registered: insertion={sub_in_reg}  withdrawal={sub_out_reg}")
    print(f"dual-pass in one model: {both_passes}")
    print(f"pin delta: fx={pin['delta_fx']:.2e} fy={pin['delta_fy']:.2e}")
    print(f"\n=== STAGE 3 VERDICT: {verdict} "
          f"(gate cone={gate_cone:.2f} deg, ratio={gate_ratio:.2f}) ===")
    if verdict == "NOT_RECON":
        print("!!! cone < 5 deg -> STOP the plan (parallax-zero) !!!")


if __name__ == "__main__":
    main()
