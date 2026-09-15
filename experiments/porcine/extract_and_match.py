"""Diagnostic: run the EXACT recover_clip.py pipeline stages on pig1, instrumented. This does extract ->
feature_extractor -> exhaustive_matcher and reports per-stage stats (features/img, verified pairs, matches).
Mapper run separately with live output. Same commands/settings as recover_clip.py (SIFT 1600, exhaustive,
pinned OPENCV, CLAHE 3.0, stride 2 for 60fps)."""
import json, shutil, subprocess, os
from pathlib import Path
import numpy as np, cv2, sqlite3

import sys
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/home/mi3dr/dataset/Porcine Airway Endoscopies/Pig Trachea 1 Video.mp4"
CALIB = sys.argv[2] if len(sys.argv) > 2 else "runs/own_data/porcine/pig1/intrinsics_pinned.json"
LO = int(sys.argv[3]) if len(sys.argv) > 3 else 6850
HI = int(sys.argv[4]) if len(sys.argv) > 4 else 7250
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
WSNAME = sys.argv[6] if len(sys.argv) > 6 else "pig_diag"
SC = "/tmp/claude-100461304/-home-mi3dr-projects-bronchotrust/32888f63-b0c2-4d41-9ad1-91a5dfc651e6/scratchpad"
WS = Path(f"{SC}/{WSNAME}"); WS.mkdir(parents=True, exist_ok=True)
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
os.environ["LD_LIBRARY_PATH"] = "/home/mi3dr/.conda/envs/colmap-cuda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
DARK_L = 12.0


def bezel(video, n=60):
    cap = cv2.VideoCapture(video); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3)); cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release(); m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), 5) > 0


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", r.stderr[-300:], flush=True)
    return r


img = WS / "images"; msk = WS / "masks"
for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
mask = bezel(VIDEO); bez = mask.astype(np.uint8) * 255; clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cap = cv2.VideoCapture(VIDEO); fi = 0; n = 0
while True:
    ok, fr = cap.read()
    if not ok or fi > HI: break
    if LO <= fi <= HI and (fi - LO) % STRIDE == 0 and cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
        cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
    fi += 1
cap.release(); print(f"[1 extract] {n} frames f{LO}-{HI} stride{STRIDE} (CLAHE3.0)", flush=True)

db = WS / "db.db"; db.unlink(missing_ok=True)
pstr = ",".join(f"{p:.10g}" for p in json.loads(Path(CALIB).read_text())["params_colmap"])
sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
    "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
    "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192", "--SiftExtraction.peak_threshold", "0.005"])
con = sqlite3.connect(str(db)); cur = con.cursor()
kp = cur.execute("SELECT rows FROM keypoints").fetchall(); kp = [r[0] for r in kp]
print(f"[2 features] {len(kp)} imgs, features/img: median {int(np.median(kp))} min {min(kp)} max {max(kp)}", flush=True)

sh(["exhaustive_matcher", "--database_path", str(db)])
# raw matches
mr = cur.execute("SELECT rows FROM matches WHERE rows>0").fetchall(); mr = [r[0] for r in mr]
# geometrically-verified matches (two_view_geometries)
try:
    gv = cur.execute("SELECT rows FROM two_view_geometries WHERE rows>0").fetchall(); gv = [r[0] for r in gv]
except Exception:
    gv = []
con.close()
npairs = len(kp) * (len(kp) - 1) // 2
print(f"[3 matches] raw-match pairs {len(mr)}/{npairs} (median {int(np.median(mr)) if mr else 0} matches/pair)", flush=True)
print(f"[3 verified] geometrically-VERIFIED pairs {len(gv)}/{npairs} (median {int(np.median(gv)) if gv else 0} inliers/pair)", flush=True)
print(f"    -> verified/possible = {len(gv)/npairs:.1%}  (this drives connectivity; low = fragmentation)", flush=True)
print(f"WS={WS}", flush=True)
