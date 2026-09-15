"""Per-session camera calibration from a checkerboard video -> pinned OPENCV intrinsics for recover_clip.py.

The board is FORCED (default 14x13 inner corners, the cohort board). Searching over patterns is not done: on a
soft or low-contrast board a spurious sub-grid (e.g. 7x5) can win with a huge RMS and a non-isotropic focal.
Detection is tried per frame at full resolution, then on a 0.5x downscaled + unsharp-masked copy (rescued a
defocused board), then with the classic detector + cornerSubPix. Before writing, the result must pass an isotropy
gate (fy/fx within 2%) and a plausible field of view (60-115 deg). RMS < 0.5 px is the strict pass used in the paper;
0.5-0.7 px is flagged marginal.

Usage:
  python pipeline/calibrate.py --video "<board.MP4>" --out calib/<session>_intrinsics.json [--board 14 13] [--step 6]
"""
import argparse, json, sys
import cv2, numpy as np

SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
CL = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def detect(gray, board):
    ok, cn = cv2.findChessboardCornersSB(gray, board, SB)
    if ok: return cn, "sb"
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sharp = cv2.addWeighted(small, 1.8, cv2.GaussianBlur(small, (0, 0), 2.0), -0.8, 0)
    ok, cn = cv2.findChessboardCornersSB(sharp, board, SB)
    if ok: return cn * 2.0, "sb-half-unsharp"
    ok, cn = cv2.findChessboardCorners(gray, board, CL)
    if ok:
        cn = cv2.cornerSubPix(gray, cn, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
        return cn, "classic"
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="checkerboard calibration video")
    ap.add_argument("--out", required=True, help="output intrinsics json (read by recover_clip.py --calib)")
    ap.add_argument("--board", type=int, nargs=2, default=(14, 13), metavar=("COLS", "ROWS"), help="inner corners")
    ap.add_argument("--step", type=int, default=6, help="use every STEP-th frame")
    ap.add_argument("--max-frames", type=int, default=400)
    a = ap.parse_args(); board = tuple(a.board)
    objp = np.zeros((board[0] * board[1], 3), np.float32); objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    cap = cv2.VideoCapture(a.video); obj, img, how, fi, w, h = [], [], {}, 0, None, None
    while len(obj) < a.max_frames:
        ok, f = cap.read()
        if not ok: break
        if fi % a.step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); h, w = g.shape; cn, mode = detect(g, board)
            if cn is not None: obj.append(objp); img.append(cn.astype(np.float32)); how[mode] = how.get(mode, 0) + 1
        fi += 1
    cap.release()
    if len(obj) < 10: sys.exit(f"FAIL: only {len(obj)} board detections of {board} in {a.video}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None, flags=cv2.CALIB_FIX_K3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; k1, k2, p1, p2 = dist.ravel()[:4]
    hf = 2 * np.degrees(np.arctan(w / (2 * fx))); vf = 2 * np.degrees(np.arctan(h / (2 * fy))); r = fy / fx
    if not (abs(r - 1) < 0.02 and 60 < hf < 115):
        sys.exit(f"FAIL: implausible geometry (fy/fx {r:.3f}, HFOV {hf:.1f} deg) -> not written. Check the board size.")
    tag = "PASS" if rms < 0.5 else ("MARGINAL" if rms <= 0.7 else "POOR")
    d = dict(model="OPENCV", width=w, height=h, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
             k1=float(k1), k2=float(k2), p1=float(p1), p2=float(p2), params_colmap=[float(v) for v in (fx, fy, cx, cy, k1, k2, p1, p2)],
             rms_reproj_px=float(rms), hfov_deg=float(hf), vfov_deg=float(vf), fy_over_fx=float(r), n_frames=len(obj),
             pattern=list(board), detection_modes=how, source_calib_video=a.video, refine=False, quality=tag)
    json.dump(d, open(a.out, "w"), indent=2)
    print(f"{a.out}: board {board} frames {len(obj)} {how} RMS {rms:.3f}px fx {fx:.1f} fy/fx {r:.4f} HFOV {hf:.1f} -> {tag}")


if __name__ == "__main__":
    main()
