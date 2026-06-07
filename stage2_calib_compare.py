"""STAGE 2: OPENCV vs FISHEYE calibration on 2_v2 calibration video.

Board: 13x14 inner corners, 1.0 unit squares (pixel-scale, square_size cancels).
Sample ~80 frames across the calibration video, run findChessboardCornersSB,
sub-pixel refine, then calibrate with:
  (a) cv2.calibrateCamera  with CALIB_FIX_K3 | CALIB_FIX_TANGENT_DIST  -> OPENCV (k1,k2)
  (b) cv2.fisheye.calibrate                                            -> FISHEYE (k1..k4)
Compare RMS, FOV, and produce a side-by-side undistorted calibration frame.

Outputs (runs/gated2/2_v2/):
  calib_opencv.json
  calib_fisheye.json
  calib_compare.json
  calib_corners_overlay.png    (corners detected on one example frame)
  calib_undist_side_by_side.png (left=OPENCV undistorted, right=FISHEYE undistorted)
  calib_summary.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 13 Calibration Videos/"
             "2_v2 Calibration Video.MP4")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/2_v2")
OUT.mkdir(parents=True, exist_ok=True)

BOARD = (13, 14)  # inner corners (cols, rows)
SQUARE = 1.0
SAMPLE_N = 80


def fov_deg(f_px, side_px):
    return 2.0 * math.degrees(math.atan(side_px / (2.0 * f_px)))


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {N} frames @ {fps:.3f}, {W}x{H}")

    # Sample frame indices
    sample_ix = np.linspace(0, N - 1, SAMPLE_N).astype(int)
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE

    obj_pts = []
    img_pts = []
    detected_frames = []
    overlay_saved = False

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for fi in sample_ix:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gray_eq = clahe.apply(gray)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok_cb, corners = cv2.findChessboardCornersSB(gray_eq, BOARD, flags)
        if not ok_cb:
            continue
        # SB returns float corners; refine with cornerSubPix on plain gray
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        obj_pts.append(objp.copy())
        img_pts.append(corners)
        detected_frames.append(int(fi))
        if not overlay_saved:
            ov = fr.copy()
            cv2.drawChessboardCorners(ov, BOARD, corners, True)
            cv2.imwrite(str(OUT / "calib_corners_overlay.png"), ov,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
            overlay_saved = True
            example_frame = fr
            example_fi = int(fi)
        print(f"  fi={fi:4d}  OK ({len(obj_pts)} so far)")
    cap.release()
    n_used = len(obj_pts)
    print(f"\ndetected on {n_used}/{SAMPLE_N} sampled frames")
    assert n_used >= 20, "too few detections"

    # ---------------- (a) OPENCV (k1, k2) ----------------
    K0 = np.array([[1000., 0., W / 2.], [0., 1000., H / 2.], [0., 0., 1.]])
    flags_oc = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_USE_INTRINSIC_GUESS)
    rms_oc, K_oc, D_oc, rvecs_oc, tvecs_oc = cv2.calibrateCamera(
        obj_pts, img_pts, (W, H), K0.copy(), None, flags=flags_oc)
    fx_o, fy_o = K_oc[0, 0], K_oc[1, 1]
    cx_o, cy_o = K_oc[0, 2], K_oc[1, 2]
    k1_o, k2_o, p1_o, p2_o, k3_o = D_oc.ravel()[:5]
    print(f"\nOPENCV: RMS={rms_oc:.4f}  fx={fx_o:.3f} fy={fy_o:.3f}  "
          f"cx={cx_o:.3f} cy={cy_o:.3f}  k1={k1_o:.4f} k2={k2_o:.4f}")
    print(f"  HFOV={fov_deg(fx_o, W):.2f}  VFOV={fov_deg(fy_o, H):.2f}")

    # ---------------- (b) FISHEYE (k1..k4) ----------------
    obj_pts_fe = [op.reshape(-1, 1, 3).astype(np.float64) for op in obj_pts]
    img_pts_fe = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in img_pts]
    K_fe = K0.copy().astype(np.float64)
    D_fe = np.zeros((4, 1), dtype=np.float64)
    flags_fe = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_FIX_SKEW
                | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    try:
        rms_fe, K_fe, D_fe, _, _ = cv2.fisheye.calibrate(
            obj_pts_fe, img_pts_fe, (W, H), K_fe, D_fe, flags=flags_fe,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      120, 1e-6))
    except cv2.error as e:
        print(f"FISHEYE calibrate raised: {e}; retry without K_FIX")
        rms_fe = float("nan"); K_fe = None; D_fe = None
    fe_ok = (K_fe is not None) and np.isfinite(rms_fe)
    if fe_ok:
        fx_f, fy_f = float(K_fe[0, 0]), float(K_fe[1, 1])
        cx_f, cy_f = float(K_fe[0, 2]), float(K_fe[1, 2])
        d_f = D_fe.ravel().tolist()
        print(f"\nFISHEYE: RMS={rms_fe:.4f}  fx={fx_f:.3f} fy={fy_f:.3f}  "
              f"cx={cx_f:.3f} cy={cy_f:.3f}  D={d_f}")
        print(f"  HFOV={fov_deg(fx_f, W):.2f}  VFOV={fov_deg(fy_f, H):.2f}")
    else:
        fx_f = fy_f = cx_f = cy_f = float("nan")
        d_f = [float("nan")] * 4

    # ---------------- Side-by-side undistort ----------------
    # Use example_frame (a calibration board view)
    # OPENCV
    new_K_oc, _ = cv2.getOptimalNewCameraMatrix(K_oc, D_oc, (W, H),
                                                alpha=0.0)
    map1_o, map2_o = cv2.initUndistortRectifyMap(
        K_oc, D_oc, None, new_K_oc, (W, H), cv2.CV_16SC2)
    undist_oc = cv2.remap(example_frame, map1_o, map2_o, cv2.INTER_LINEAR)
    # FISHEYE
    if fe_ok:
        new_K_fe = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_fe, D_fe, (W, H), np.eye(3), balance=0.0)
        map1_f, map2_f = cv2.fisheye.initUndistortRectifyMap(
            K_fe, D_fe, np.eye(3), new_K_fe, (W, H), cv2.CV_16SC2)
        undist_fe = cv2.remap(example_frame, map1_f, map2_f, cv2.INTER_LINEAR)
    else:
        undist_fe = np.zeros_like(example_frame)

    # Label and assemble
    h2, w2 = H, W
    sbs = np.zeros((h2, 2 * w2, 3), dtype=np.uint8)
    sbs[:, :w2] = undist_oc
    sbs[:, w2:] = undist_fe
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(sbs, (0, 0), (w2, 60), (30, 30, 30), -1)
    cv2.rectangle(sbs, (w2, 0), (2 * w2, 60), (30, 30, 30), -1)
    cv2.putText(sbs, f"OPENCV (k1,k2)  RMS={rms_oc:.3f} px",
                (20, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sbs, f"FISHEYE (k1..k4)  RMS={rms_fe:.3f} px"
                if fe_ok else "FISHEYE: FAILED",
                (w2 + 20, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    # downscale for viewing
    out_sbs = cv2.resize(sbs, (sbs.shape[1] // 2, sbs.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "calib_undist_side_by_side.png"), out_sbs,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"saved side-by-side {out_sbs.shape}")

    # ---------------- Save ----------------
    opencv = {
        "model": "OPENCV (k1,k2,p1=0,p2=0,k3=0)",
        "image_size": [W, H],
        "K": K_oc.tolist(),
        "fx": float(fx_o), "fy": float(fy_o),
        "cx": float(cx_o), "cy": float(cy_o),
        "k1": float(k1_o), "k2": float(k2_o),
        "p1": float(p1_o), "p2": float(p2_o), "k3": float(k3_o),
        "rms_reproj_px": float(rms_oc),
        "n_frames_used": int(n_used),
        "n_frames_sampled": int(SAMPLE_N),
        "hfov_deg": fov_deg(fx_o, W),
        "vfov_deg": fov_deg(fy_o, H),
        "example_frame_idx": int(example_fi),
    }
    fisheye = {
        "model": "FISHEYE (Kannala-Brandt, k1..k4)",
        "image_size": [W, H],
        "K": K_fe.tolist() if fe_ok else None,
        "fx": fx_f, "fy": fy_f, "cx": cx_f, "cy": cy_f,
        "D": d_f,
        "rms_reproj_px": float(rms_fe) if fe_ok else None,
        "n_frames_used": int(n_used),
        "n_frames_sampled": int(SAMPLE_N),
        "hfov_deg": fov_deg(fx_f, W) if fe_ok else None,
        "vfov_deg": fov_deg(fy_f, H) if fe_ok else None,
    }
    (OUT / "calib_opencv.json").write_text(json.dumps(opencv, indent=2))
    (OUT / "calib_fisheye.json").write_text(json.dumps(fisheye, indent=2))

    expected_fov = 97.0
    opencv["hfov_off_from_expected"] = opencv["hfov_deg"] - expected_fov
    if fe_ok:
        fisheye["hfov_off_from_expected"] = fisheye["hfov_deg"] - expected_fov

    summary = {
        "video": str(VIDEO),
        "board_size": BOARD, "square_size": SQUARE,
        "expected_hfov_deg": expected_fov,
        "opencv": opencv,
        "fisheye": fisheye,
        "comparison": {
            "rms_opencv_px": float(rms_oc),
            "rms_fisheye_px": float(rms_fe) if fe_ok else None,
            "delta_rms": (float(rms_fe) - float(rms_oc)) if fe_ok else None,
            "recommendation": (
                "FISHEYE" if (fe_ok and float(rms_fe) < float(rms_oc) - 0.05)
                else "OPENCV"
            ),
        },
        "corners_overlay_path": str(OUT / "calib_corners_overlay.png"),
        "side_by_side_path": str(OUT / "calib_undist_side_by_side.png"),
    }
    (OUT / "calib_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved calib_summary.json")
    print(f"\n>>> RECOMMENDED MODEL: {summary['comparison']['recommendation']}")
    print(f"    (chooses FISHEYE only if its RMS beats OPENCV by >=0.05 px)")


if __name__ == "__main__":
    main()
