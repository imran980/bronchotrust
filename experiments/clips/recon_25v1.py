"""25-V1 densify: reuse the validated batch4 sparse model (reproj 1.57, roundness 1.02), re-extract its
window frames with batch_dense preprocessing, undistort + 4-GPU patch_match + fuse -> dense cloud.
Usage: python recon_25v1.py  (then csa_run.py + render_airway.py on the fused.ply)."""
import shutil, subprocess
from pathlib import Path
import numpy as np, cv2

ROOT = Path("/home/mi3dr/projects/bronchotrust")
VIDEO = Path("/home/mi3dr/dataset/real-broncho/25-V1.MP4")
SPARSE = ROOT / "runs/batch4/25-V1/sparse/0"
OUT = ROOT / "runs/own_data/recon_25v1"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
LO, HI, DARK_L = 315, 620, 12.0


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4) > 0


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-200:], flush=True)
    return r


img = OUT / "images"
shutil.rmtree(img, ignore_errors=True); img.mkdir(parents=True)
mask = bezel(VIDEO); c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
cap = cv2.VideoCapture(str(VIDEO)); fi = 0; n = 0
while True:
    ok, fr = cap.read()
    if not ok or fi > HI: break
    if LO <= fi <= HI and cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
        cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
    fi += 1
cap.release(); print(f"extracted {n} CLAHE frames f{LO}-{HI}", flush=True)

sp = OUT / "sparse" / "0"; shutil.rmtree(OUT / "sparse", ignore_errors=True); sp.mkdir(parents=True)
for f in SPARSE.glob("*"): shutil.copy(str(f), str(sp / f.name))
dense = OUT / "dense0"; shutil.rmtree(dense, ignore_errors=True)
sh(["image_undistorter", "--image_path", str(img), "--input_path", str(sp), "--output_path", str(dense),
    "--output_type", "COLMAP", "--max_image_size", "1600"])
sh(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200",
    "--PatchMatchStereo.gpu_index", "0,1,2,3"])
fused = dense / "fused.ply"
r = sh(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--input_type", "geometric", "--StereoFusion.min_num_pixels", "3", "--output_path", str(fused)])
print([l for l in r.stdout.splitlines() if "fused points" in l][-1:], flush=True)
print(f"DENSE done: {fused}", flush=True)
