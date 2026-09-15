"""Pipeline-faithful SfM screen (recover_clip.py settings) for a pig window. extract (CLAHE3.0+bezel+stride)
-> SIFT(1600,8192,0.005) -> exhaustive -> mapper(pinned, init_min_tri 4) -> model_merger if fragmented.
Reports connectivity + median-tri. Sane window kept small so BA stays tractable on rich-texture pig data."""
import json, shutil, subprocess, os, re, sys, glob
from pathlib import Path
import numpy as np, cv2, pycolmap

VIDEO, CALIB = sys.argv[1], sys.argv[2]
LO, HI, STRIDE, WSNAME = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
MAXF = sys.argv[7] if len(sys.argv) > 7 else "8192"       # feature cap: lower (e.g. 4000) tames BA on rich-texture pig data
SC = "/tmp/claude-100461304/-home-mi3dr-projects-bronchotrust/32888f63-b0c2-4d41-9ad1-91a5dfc651e6/scratchpad"
WS = Path(f"{SC}/{WSNAME}"); WS.mkdir(parents=True, exist_ok=True)
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
os.environ["LD_LIBRARY_PATH"] = "/home/mi3dr/.conda/envs/colmap-cuda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
DARK_L = 12.0


def bezel(v, n=60):
    cap = cv2.VideoCapture(v); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3)); cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release(); m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), 5) > 0


def sh(c): return subprocess.run([COLMAP] + c, capture_output=True, text=True)


def tri(rec):
    cen = {i: np.asarray(im.projection_center()) for i, im in rec.images.items()}
    rng = np.random.default_rng(0); pts = list(rec.points3D.values())
    sel = pts if len(pts) <= 3000 else [pts[i] for i in rng.choice(len(pts), 3000, replace=False)]
    a = []
    for p in sel:
        ids = [e.image_id for e in p.track.elements if e.image_id in cen]
        if len(ids) < 2: continue
        if len(ids) > 12: ids = [ids[j] for j in rng.choice(len(ids), 12, replace=False)]
        X = np.asarray(p.xyz); v = np.array([cen[i] - X for i in ids]); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        a.append(np.degrees(np.arccos(np.clip(v @ v.T, -1, 1))).max())
    return np.median(a) if a else float("nan")


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
cap.release()
db = WS / "db.db"; db.unlink(missing_ok=True)
pstr = ",".join(f"{p:.10g}" for p in json.loads(Path(CALIB).read_text())["params_colmap"])
sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
    "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
    "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", MAXF, "--SiftExtraction.peak_threshold", "0.005"])
sh(["exhaustive_matcher", "--database_path", str(db)])
sp = WS / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
sh(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
    "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
mods = sorted([d for d in sp.iterdir() if d.is_dir()], key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
if not mods: print(f"{WSNAME}: extracted {n} -> NO MODEL (registration failed)", flush=True); sys.exit()
final = mods[0]
if len(mods) > 1:                                                    # model_merger auto-bridge (recover_clip logic)
    merged = sp / "merged"; shutil.rmtree(merged, ignore_errors=True); cur = str(mods[0])
    for nxt in mods[1:4]:
        sh(["model_merger", "--input_path1", cur, "--input_path2", str(nxt), "--output_path", str(merged)])
        if (merged / "images.bin").exists(): cur = str(merged)
    if (merged / "images.bin").exists(): final = merged
r = pycolmap.Reconstruction(str(final))
fr = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in r.images.values())
print(f"{WSNAME}: extracted {n} | {len(mods)} sub-model(s) | BEST {r.num_reg_images()}/{n} reg  span f{fr[0]}-{fr[-1]}  "
      f"{r.num_points3D()}pts  median-tri {tri(r):.2f}deg  reproj {r.compute_mean_reprojection_error():.2f}px", flush=True)
