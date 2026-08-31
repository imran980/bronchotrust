"""Pig 1 intrinsics — refined. Size FIXED at (13,12) (=14x13-square clinical board, confirmed). Denoise the
board crop (footage is very noisy) before subpixel corner refine, collect many views, then iterative
outlier rejection (drop high-reprojection-error views) to lower RMS. OPENCV model."""
import cv2, numpy as np, json, sys
from pathlib import Path

V = sys.argv[1] if len(sys.argv) > 1 else "/home/mi3dr/dataset/Porcine Airway Endoscopies/Pig Trachea 1 Calibration Video.mp4"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/own_data/porcine/pig1")
FLO, FHI = int(sys.argv[3]) if len(sys.argv) > 3 else 150, int(sys.argv[4]) if len(sys.argv) > 4 else 2400
SIZE = (13, 12)
OUT.mkdir(parents=True, exist_ok=True)
SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def locate(g):
    gg = g.astype(np.float32)
    var = cv2.blur(gg * gg, (15, 15)) - cv2.blur(gg, (15, 15)) ** 2
    mask = (var > np.percentile(var, 99.0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    if w < 90 or h < 90: return None
    m = 45
    return max(0, y - m), min(g.shape[0], y + h + m), max(0, x - m), min(g.shape[1], x + w + m)


cap = cv2.VideoCapture(V); W = int(cap.get(3)); H = int(cap.get(4))
objp = np.zeros((SIZE[0] * SIZE[1], 3), np.float32); objp[:, :2] = np.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)
objpts, imgpts = [], []
fi = 0
while fi <= FHI:
    if not cap.grab(): break
    if fi >= FLO and (fi - FLO) % 6 == 0:
        ok, fr = cap.retrieve()
        if ok:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(g, cv2.CV_64F).var() > 45:
                bb = locate(g)
                if bb:
                    y0, y1, x0, x1 = bb; crop = g[y0:y1, x0:x1]
                    dn = cv2.fastNlMeansDenoising(crop, None, 12, 7, 21)         # denoise the small crop
                    f, c = cv2.findChessboardCornersSB(dn, SIZE, flags=SB)
                    if f:
                        gden = g.copy(); gden[y0:y1, x0:x1] = dn
                        c = (c.reshape(-1, 2) + [x0, y0]).astype(np.float32).reshape(-1, 1, 2)
                        c = cv2.cornerSubPix(gden, c, (9, 9), (-1, -1), SUBPIX)
                        objpts.append(objp); imgpts.append(c)
    if len(imgpts) >= 70: break
    fi += 1
cap.release()
print(f"collected {len(imgpts)} denoised board views", flush=True)
if len(imgpts) < 12: print("too few"); sys.exit(1)

# calibrate + iterative outlier rejection
keep = list(range(len(imgpts)))
for it in range(4):
    op = [objpts[i] for i in keep]; ip = [imgpts[i] for i in keep]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(op, ip, (W, H), None, None)
    errs = []
    for j in range(len(keep)):
        proj, _ = cv2.projectPoints(op[j], rvecs[j], tvecs[j], K, dist)
        errs.append(np.sqrt(((proj.reshape(-1, 2) - ip[j].reshape(-1, 2)) ** 2).sum(1).mean()))
    errs = np.array(errs); med = np.median(errs)
    fx, fy = K[0, 0], K[1, 1]; fov = 2 * np.degrees(np.arctan(W / (2 * fx)))
    print(f"iter {it}: {len(keep)} views  RMS {rms:.3f}  med-view-err {med:.3f}  fx/fy {fx/fy:.4f}  FOV {fov:.1f}", flush=True)
    thr = max(0.8, 1.4 * med)
    newkeep = [keep[j] for j in range(len(keep)) if errs[j] < thr]
    if len(newkeep) < 12 or len(newkeep) == len(keep): break
    keep = newkeep

cx, cy = K[0, 2], K[1, 2]; fovv = 2 * np.degrees(np.arctan(H / (2 * fy)))
print("\n===== PIG 1 CALIBRATION (refined) =====")
print(f"board {SIZE} | {len(keep)} views")
print(f"RMS {rms:.4f} px   [gate <0.5, marginal <1.0]")
print(f"fx,fy {fx:.2f},{fy:.2f} (fx/fy {fx/fy:.4f})   cx,cy {cx:.1f},{cy:.1f} (center {W/2:.0f},{H/2:.0f})")
print(f"dist {np.round(dist.ravel(),4).tolist()}")
print(f"FOV H {fov:.1f}  V {fovv:.1f}")
print(f"GATE: {'PASS' if rms<0.5 else 'MARGINAL' if rms<1.0 else 'FAIL'}")
json.dump(dict(model="OPENCV", width=W, height=H, board=list(SIZE), n_views=len(keep), rms=float(rms),
               fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), dist=dist.ravel().tolist(),
               fov_h=float(fov), fov_v=float(fovv),
               params_colmap=[float(fx), float(fy), float(cx), float(cy)] + dist.ravel()[:4].tolist()),
          open(OUT / "intrinsics_pinned.json", "w"), indent=2)
print(f"saved {OUT/'intrinsics_pinned.json'}")
