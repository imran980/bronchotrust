"""STAGE 4 (rewrite): A/B reconstruction of 15_v2 -- COLMAP vs GlueMap.

Both arms consume IDENTICAL undistorted PINHOLE frames (undistort once with the
pinned OPENCV model). Only the SfM engine differs; MVS (COLMAP) is shared.

  --arm colmap   : run in `depth-eval` env  (hloc SuperPoint+LightGlue + pycolmap
                   incremental mapper, pinned PINHOLE, no refine) + MVS
  --arm gluemap  : run in `gluemap` env     (gluemap-demo: SALAD+Pi3 feedforward
                   global SfM, GT intrinsics pinned) + MVS

MARGINAL-PARALLAX EXPERIMENT (Stage-3 finding: ~4.5 deg median triangulation;
triage's 17.5 deg was an artifact). The deliverable is trusted ONLY where the
two arms agree (Stage 6). A clean GlueMap tube where COLMAP is noise = prior-
driven; flag, don't bank.

Outputs (runs/gluemap_15v2/15_v2/stage4/<arm>/):
  sfm/ (colmap) or gm/gluemap_aba/ (gluemap)   COLMAP-format sparse model
  dense/fused.ply                              dense MVS cloud (measured in St.5)
  stage4_results.json                          reg/sparse/dense, cone, tri-angles
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2

import s4_common as C

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
RUN = PROJECT / "runs/gluemap_15v2/15_v2"
GLUEMAP_SRC = Path("/home/mi3dr/projects/gluemap_src")


def get_inputs():
    intr = json.loads((RUN / "stage2/intrinsics_pinned.json").read_text())
    sel = json.loads((RUN / "stage1/stage1_selection.json").read_text())["selection"]
    seg_of = {s["frame_idx"]: s["segment"] for s in sel}
    mask = cv2.imread(str(RUN / "stage0/content_mask.png"), cv2.IMREAD_GRAYSCALE)
    return intr, seg_of, mask


def ensure_undist(intr, mask):
    undist = RUN / "stage4/undist"
    img_dir = undist / "images"
    have = sorted(img_dir.glob("f*.png")) if img_dir.exists() else []
    if len(have) >= 36 and (undist / "pinhole.json").exists():
        pinhole = json.loads((undist / "pinhole.json").read_text())
        return pinhole, [p.name for p in have], img_dir
    pinhole, names = C.undistort_curated(RUN / "stage1/images", intr, mask, img_dir)
    undist.mkdir(parents=True, exist_ok=True)
    (undist / "pinhole.json").write_text(json.dumps(pinhole, indent=2))
    print(f"  undistorted {len(names)} frames -> {img_dir}")
    return pinhole, names, img_dir


def colmap_arm(pinhole, names, img_dir, out):
    from hloc import (extract_features, match_features, pairs_from_exhaustive,
                      reconstruction as hr)
    sfm = out / "sfm"
    sfm.mkdir(parents=True, exist_ok=True)
    fconf = extract_features.confs["superpoint_max"]
    mconf = match_features.confs["superpoint+lightglue"]
    print("  hloc SuperPoint ...", flush=True)
    feats = extract_features.main(fconf, img_dir, out, image_list=names)
    pairs = out / "pairs.txt"
    pairs_from_exhaustive.main(pairs, image_list=names)
    print("  hloc LightGlue ...", flush=True)
    matches = match_features.main(mconf, pairs, fconf["output"], out)
    params = ",".join(f"{p:.10g}" for p in pinhole["params"])
    image_options = {"camera_model": "PINHOLE", "camera_params": params}
    mapper_options = {
        "min_num_matches": 8, "multiple_models": True, "max_num_models": 50,
        "min_model_size": 3, "ba_refine_focal_length": False,
        "ba_refine_extra_params": False, "ba_refine_principal_point": False,
        "mapper": {"init_min_tri_angle": 4.0, "init_max_error": 8.0,
                   "init_min_num_inliers": 15, "init_max_forward_motion": 0.99,
                   "abs_pose_max_error": 20.0, "abs_pose_min_num_inliers": 15,
                   "abs_pose_min_inlier_ratio": 0.1,
                   "filter_max_reproj_error": 8.0, "filter_min_tri_angle": 1.0}}
    print("  pycolmap incremental mapper (pinned PINHOLE) ...", flush=True)
    hr.main(sfm, img_dir, pairs, feats, matches, image_list=names,
            camera_mode=hr.pycolmap.CameraMode.SINGLE,
            image_options=image_options, verbose=False,
            mapper_options=mapper_options)
    return sfm


def gluemap_arm(pinhole, names, img_dir, out):
    gt = out / "gt_model"
    C.write_gt_model_text(gt, names, pinhole)
    gm_out = out / "gm"
    gm_out.mkdir(parents=True, exist_ok=True)
    cmd = ["gluemap-demo", "--config", str(GLUEMAP_SRC / "configs/example.yaml"),
           "--images_path", str(img_dir), "--write_path", str(gm_out),
           "--gt_intrinsics_path", str(gt), "--use_gt_intrinsics",
           "--camera_model", "SIMPLE_PINHOLE", "--intrinsics_mode", "SHARED"]
    print("  RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(GLUEMAP_SRC), capture_output=True, text=True)
    (out / "gluemap_stdout.log").write_text(r.stdout)
    (out / "gluemap_stderr.log").write_text(r.stderr)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR tail:", r.stderr[-2000:])
    model = gm_out / "gluemap_aba"
    if not ((model / "cameras.bin").exists() or (model / "cameras.txt").exists()):
        raise SystemExit(f"GlueMap output model not found at {model} "
                         f"(rc={r.returncode}); see gluemap_stderr.log")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["colmap", "gluemap"])
    args = ap.parse_args()
    intr, seg_of, mask = get_inputs()
    pinhole, names, img_dir = ensure_undist(intr, mask)
    out = RUN / f"stage4/{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"ARM={args.arm}  undist={len(names)} frames  "
          f"pinhole fx={pinhole['fx']:.2f} cx={pinhole['cx']:.1f}")

    sfm = colmap_arm(pinhole, names, img_dir, out) if args.arm == "colmap" \
        else gluemap_arm(pinhole, names, img_dir, out)

    print("  MVS (COLMAP patch_match_stereo) ...", flush=True)
    n_dense, ply, logs = C.run_mvs(sfm, img_dir, out / "dense")
    ana = C.analyze(sfm, seg_of)
    ana.update({"arm": args.arm, "n_curated": len(names), "n_dense": int(n_dense),
                "dense_ply": str(ply), "sfm_model_dir": str(sfm), "mvs_logs": logs})
    (out / "stage4_results.json").write_text(json.dumps(ana, indent=2))

    print(f"\n=== STAGE 4 [{args.arm}] ===")
    print(f"  registered : {ana['n_registered']}/{len(names)}")
    print(f"  sparse pts : {ana['n_sparse']}   dense pts: {n_dense}")
    print(f"  reproj px  : {ana['mean_reproj_px']}")
    print(f"  tri-angle  : med={ana['tri_angle_median']} p75={ana['tri_angle_p75']} max={ana['tri_angle_max']}")
    print(f"  cone all   : {ana['cone_all']}")
    print(f"  cone subcord: {ana['cone_subcord']}")
    print(f"  camera     : {ana['camera_model']} {ana['camera_params']}")


if __name__ == "__main__":
    main()
