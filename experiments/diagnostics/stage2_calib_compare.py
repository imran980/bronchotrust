"""STAGE 2 (gated plan): pinned intrinsics from the per-session calibration board.

Parametrized for any calibration video (default: 15_v2 calibration board).
Runs cv2.calibrateCamera (OPENCV k1,k2) AND cv2.fisheye.calibrate (k1..k4),
compares RMS / FOV, writes a side-by-side undistortion, and emits the canonical
intrinsics_pinned.json for the chosen model.

Locked decision #1 (CLAUDE.md): intrinsics are PINNED, never refined.
  - OPENCV model is the default recommendation; FISHEYE is chosen only if its
    RMS beats OPENCV by a clear margin AND its undistortion does not blow up.
  - Fisheye high-order k3/k4 with alternating large signs => reject (the eject
    criterion is the undistortion going blank/garbage).

Sampling uses POS_FRAMES seek across the board clip (fine: we only need diverse
board poses, not index integrity).

Outputs (--out, default runs/gluemap_15v2/15_v2/stage2/):
  calib_opencv.json / calib_fisheye.json / calib_summary.json
  calib_corners_overlay.png
  calib_undist_side_by_side.png
  intrinsics_pinned.json   <- canonical pinned intrinsics for stages 3-4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

DEF_VIDEO = ("/home/mi3dr/dataset/validation-videos/First 13 Calibration Videos/"
             "15_v2 Calibration Video.MP4")
DEF_OUT = "/home/mi3dr/projects/bronchotrust/runs/gluemap_15v2/15_v2/stage2"
EXPECTED_HFOV = 97.0


def fov_deg(f_px, side_px):
    return 2.0 * math.degrees(math.atan(side_px / (2.0 * f_px)))


def detect_corners(video: Path, board, square, sample_n):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"calib video: {N} frames, {W}x{H}")
    sample_ix = np.linspace(0, max(N - 1, 0), sample_n).astype(int)
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_pts, img_pts, frames = [], [], []
    overlay = None
    example = None
    for fi in sample_ix:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ok_cb, corners = cv2.findChessboardCornersSB(clahe.apply(gray), board, flags)
        if not ok_cb:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        obj_pts.append(objp.copy())
        img_pts.append(corners)
        frames.append(int(fi))
        if overlay is None:
            overlay = fr.copy()
            cv2.drawChessboardCorners(overlay, board, corners, True)
            example = (fr, int(fi))
        print(f"  fi={fi:4d}  detected ({len(obj_pts)})", flush=True)
    cap.release()
    return obj_pts, img_pts, frames, (W, H), overlay, example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEF_VIDEO)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--board", default="13,14", help="inner corners cols,rows")
    ap.add_argument("--square", type=float, default=1.0)
    ap.add_argument("--sample-n", type=int, default=80)
    args = ap.parse_args()

    board = tuple(int(x) for x in args.board.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    print(f"VIDEO: {video}\nOUT:   {out}\nboard: {board}\n")

    obj_pts, img_pts, frames, (W, H), overlay, example = detect_corners(
        video, board, args.square, args.sample_n)
    n_used = len(obj_pts)
    print(f"\ndetected on {n_used}/{args.sample_n} sampled frames")
    if n_used < 20:
        raise SystemExit(f"too few board detections ({n_used}); "
                         f"check --board (tried {board}) or calibration video")
    cv2.imwrite(str(out / "calib_corners_overlay.png"), overlay,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # ---- OPENCV (k1, k2) ----
    K0 = np.array([[1000., 0., W / 2.], [0., 1000., H / 2.], [0., 0., 1.]])
    flags_oc = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_USE_INTRINSIC_GUESS)
    rms_oc, K_oc, D_oc, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, (W, H), K0.copy(), None, flags=flags_oc)
    fx_o, fy_o, cx_o, cy_o = K_oc[0, 0], K_oc[1, 1], K_oc[0, 2], K_oc[1, 2]
    k1_o, k2_o, p1_o, p2_o, k3_o = D_oc.ravel()[:5]
    hfov_o, vfov_o = fov_deg(fx_o, W), fov_deg(fy_o, H)
    print(f"\nOPENCV  RMS={rms_oc:.4f}  fx={fx_o:.2f} fy={fy_o:.2f} "
          f"(fy/fx={fy_o/fx_o:.4f})  cx={cx_o:.1f} cy={cy_o:.1f}  "
          f"k1={k1_o:.4f} k2={k2_o:.4f}  HFOV={hfov_o:.2f} VFOV={vfov_o:.2f}")

    # ---- FISHEYE (k1..k4) ----
    obj_fe = [op.reshape(-1, 1, 3).astype(np.float64) for op in obj_pts]
    img_fe = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in img_pts]
    K_fe = K0.copy().astype(np.float64)
    D_fe = np.zeros((4, 1), np.float64)
    flags_fe = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_FIX_SKEW
                | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    try:
        rms_fe, K_fe, D_fe, _, _ = cv2.fisheye.calibrate(
            obj_fe, img_fe, (W, H), K_fe, D_fe, flags=flags_fe,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      120, 1e-6))
        fe_ok = np.isfinite(rms_fe)
    except cv2.error as e:
        print(f"FISHEYE calibrate raised: {e}")
        rms_fe, fe_ok = float("nan"), False
    if fe_ok:
        fx_f, fy_f = float(K_fe[0, 0]), float(K_fe[1, 1])
        cx_f, cy_f = float(K_fe[0, 2]), float(K_fe[1, 2])
        d_f = D_fe.ravel().tolist()
        print(f"FISHEYE RMS={rms_fe:.4f}  fx={fx_f:.2f} fy={fy_f:.2f}  "
              f"D={[round(x,4) for x in d_f]}  HFOV={fov_deg(fx_f,W):.2f}")
    else:
        fx_f = fy_f = cx_f = cy_f = float("nan")
        d_f = [float("nan")] * 4

    # ---- side-by-side undistortion ----
    ex_fr, ex_fi = example
    newK_o, _ = cv2.getOptimalNewCameraMatrix(K_oc, D_oc, (W, H), alpha=0.0)
    m1o, m2o = cv2.initUndistortRectifyMap(K_oc, D_oc, None, newK_o, (W, H),
                                           cv2.CV_16SC2)
    und_o = cv2.remap(ex_fr, m1o, m2o, cv2.INTER_LINEAR)
    if fe_ok:
        newK_f = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_fe, D_fe, (W, H), np.eye(3), balance=0.0)
        m1f, m2f = cv2.fisheye.initUndistortRectifyMap(
            K_fe, D_fe, np.eye(3), newK_f, (W, H), cv2.CV_16SC2)
        und_f = cv2.remap(ex_fr, m1f, m2f, cv2.INTER_LINEAR)
        # eject check: fraction of black pixels after undistort (blow-up -> mostly black)
        fe_black_frac = float((cv2.cvtColor(und_f, cv2.COLOR_BGR2GRAY) < 3).mean())
    else:
        und_f = np.zeros_like(ex_fr)
        fe_black_frac = 1.0
    sbs = np.zeros((H, 2 * W, 3), np.uint8)
    sbs[:, :W], sbs[:, W:] = und_o, und_f
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(sbs, (0, 0), (2 * W, 60), (30, 30, 30), -1)
    cv2.putText(sbs, f"OPENCV RMS={rms_oc:.3f}", (20, 42), font, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sbs, (f"FISHEYE RMS={rms_fe:.3f} blk={fe_black_frac:.2f}"
                      if fe_ok else "FISHEYE FAILED"),
                (W + 20, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "calib_undist_side_by_side.png"),
                cv2.resize(sbs, (W, H // 2), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # ---- recommendation (locked-rule #1 favors OPENCV stability) ----
    fisheye_blowup = fe_ok and fe_black_frac > 0.25
    recommend = "OPENCV"
    if fe_ok and (rms_fe < rms_oc - 0.05) and not fisheye_blowup:
        recommend = "FISHEYE"

    opencv = {"model": "OPENCV", "image_size": [W, H], "K": K_oc.tolist(),
              "fx": float(fx_o), "fy": float(fy_o), "cx": float(cx_o),
              "cy": float(cy_o), "k1": float(k1_o), "k2": float(k2_o),
              "p1": float(p1_o), "p2": float(p2_o), "k3": float(k3_o),
              "fy_over_fx": float(fy_o / fx_o), "rms_reproj_px": float(rms_oc),
              "hfov_deg": hfov_o, "vfov_deg": vfov_o,
              "n_frames_used": n_used, "example_frame_idx": ex_fi}
    fisheye = {"model": "FISHEYE", "image_size": [W, H],
               "K": K_fe.tolist() if fe_ok else None,
               "fx": fx_f, "fy": fy_f, "cx": cx_f, "cy": cy_f, "D": d_f,
               "rms_reproj_px": float(rms_fe) if fe_ok else None,
               "hfov_deg": fov_deg(fx_f, W) if fe_ok else None,
               "undistort_black_frac": fe_black_frac,
               "n_frames_used": n_used}
    (out / "calib_opencv.json").write_text(json.dumps(opencv, indent=2))
    (out / "calib_fisheye.json").write_text(json.dumps(fisheye, indent=2))

    summary = {"video": str(video), "board": list(board),
               "expected_hfov_deg": EXPECTED_HFOV, "opencv": opencv,
               "fisheye": fisheye,
               "comparison": {"rms_opencv": float(rms_oc),
                              "rms_fisheye": float(rms_fe) if fe_ok else None,
                              "fisheye_blowup": fisheye_blowup,
                              "recommendation": recommend}}
    (out / "calib_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- canonical pinned intrinsics for stages 3-4 ----
    if recommend == "OPENCV":
        pinned = {"model": "OPENCV", "width": W, "height": H,
                  "fx": float(fx_o), "fy": float(fy_o),
                  "cx": float(cx_o), "cy": float(cy_o),
                  "k1": float(k1_o), "k2": float(k2_o),
                  "p1": float(p1_o), "p2": float(p2_o),
                  "params_colmap": [float(fx_o), float(fy_o), float(cx_o),
                                    float(cy_o), float(k1_o), float(k2_o),
                                    float(p1_o), float(p2_o)],
                  "rms_reproj_px": float(rms_oc),
                  "hfov_deg": hfov_o, "vfov_deg": vfov_o,
                  "fy_over_fx": float(fy_o / fx_o),
                  "source_calib_video": str(video), "refine": False}
    else:
        pinned = {"model": "FISHEYE", "width": W, "height": H,
                  "fx": fx_f, "fy": fy_f, "cx": cx_f, "cy": cy_f, "D": d_f,
                  "rms_reproj_px": float(rms_fe),
                  "source_calib_video": str(video), "refine": False}
    (out / "intrinsics_pinned.json").write_text(json.dumps(pinned, indent=2))

    print(f"\n>>> RECOMMENDED: {recommend}")
    print(f"    OPENCV RMS={rms_oc:.4f} HFOV={hfov_o:.2f} fy/fx={fy_o/fx_o:.4f}")
    if fe_ok:
        print(f"    FISHEYE RMS={rms_fe:.4f} blowup={fisheye_blowup} "
              f"(black_frac={fe_black_frac:.2f})")
    print(f"saved intrinsics_pinned.json -> {out}")


if __name__ == "__main__":
    main()
