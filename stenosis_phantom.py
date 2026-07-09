"""STENOSIS phantom: a textured tube with a known NORMAL radius Rn and a known THROAT radius Rt
(smooth Gaussian constriction). A camera flying BEHIND the throat images the narrowest ring as an
aperture that occludes the wall flaring back out behind it -> the throat is a genuine, fixed 3D
OCCLUDING CONTOUR of radius Rt (the clinical analog the uniform tube lacked). GT throat DCE=2*Rt.

Renders via first-forward-hit ray-march against the surface of revolution r(z), same OPENCV
intrinsics family as 7_v1 (scaled to render res). Saves per-frame texture PNGs, a GT
through-throat aperture MASK per frame (mask boundary = true throat contour), and GT poses.
Then runs the COLMAP baseline (SIFT sequential -> MVS) to provide realistic poses + a dense
cloud to slice at the throat. depth-eval env drives; colmap-cuda binary for SfM/MVS.
Usage: python stenosis_phantom.py [render|colmap|all]
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np, cv2

OUT = Path("/home/mi3dr/projects/bronchotrust/runs/phantom_sten")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
INTR = json.loads(Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib/7_v1/intrinsics_pinned.json").read_text())
SCALE = 2.0 / 3.0                                   # render at 1280x720 for tractable ray-march
W, H = int(INTR["width"] * SCALE), int(INTR["height"] * SCALE)
fx, fy = INTR["fx"] * SCALE, INTR["fy"] * SCALE
cx, cy = INTR["cx"] * SCALE, INTR["cy"] * SCALE
k1, k2 = INTR["k1"], INTR["k2"]                     # normalized-coord distortion: resolution-free
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); D = np.array([k1, k2, 0.0, 0.0])
PARAMS = [fx, fy, cx, cy, k1, k2, 0.0, 0.0]         # OPENCV camera params for COLMAP

RN = 5.0            # normal (wide) radius
RT = 3.0           # throat radius  -> GT throat DCE = 6.0
ZS = 42.0          # throat axial center (world z)
SIGMA = 4.0        # constriction width
TUBE_LEN = 150.0   # long tube: no near far-end -> the THROAT is the only dark occluding contour
                   # (a finite end would make a competing r=Rn escape-rim; real airways have none)
N_FRAMES = 220
OVERLAP = 30


def rprof(z):
    return RN - (RN - RT) * np.exp(-0.5 * ((z - ZS) / SIGMA) ** 2)


def make_texture(htex=1024, wtex=2048, seed=1):
    rng = np.random.default_rng(seed); base = np.zeros((htex, wtex, 3), np.float32)
    for sc in (32, 8, 2):
        n = rng.random((htex // sc, wtex // sc, 3)).astype(np.float32)
        base += cv2.resize(n, (wtex, htex), interpolation=cv2.INTER_LINEAR) / 3
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    tex = np.clip(0.35 + 0.65 * base, 0, 1) * np.array([0.55, 0.45, 0.85])
    return (tex * 255).astype(np.uint8)


def first_hit(C, d, nstep=130, nbis=10):
    """First forward wall hit against r(z). C:(3,) camera center; d:(N,3) unit world dirs.
    Returns t_hit (N,), valid (N,) bool.  Camera assumed inside (rho<r(z)).
    Incremental march (low memory): track the first inside->past-wall crossing per ray."""
    N = len(d); dz = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_to_end = np.where(dz > 1e-4, (TUBE_LEN - C[2]) / dz, np.inf)
    Tmax = np.clip(t_to_end, 1.0, 6.0 * TUBE_LEN)
    found = np.zeros(N, bool); t_lo = np.zeros(N); t_hi = np.zeros(N)
    g_prev = np.full(N, -1.0); t_prev = np.zeros(N)
    for fr in np.linspace(0.02, 1.0, nstep):
        t = fr * Tmax; X = C[None, :] + t[:, None] * d
        z = X[:, 2]; g = np.hypot(X[:, 0], X[:, 1]) - rprof(z)
        inside = (z >= 0) & (z <= TUBE_LEN); gv = np.where(inside, g, np.nan)
        cross = (~found) & (g_prev < 0) & (gv >= 0)
        t_lo = np.where(cross, t_prev, t_lo); t_hi = np.where(cross, t, t_hi); found |= cross
        okg = ~np.isnan(gv); g_prev = np.where(okg, gv, g_prev); t_prev = np.where(okg, t, t_prev)
    for _ in range(nbis):
        tm = 0.5 * (t_lo + t_hi); Xm = C[None, :] + tm[:, None] * d
        gm = np.hypot(Xm[:, 0], Xm[:, 1]) - rprof(Xm[:, 2])
        lo = gm < 0; t_lo = np.where(lo, tm, t_lo); t_hi = np.where(lo, t_hi, tm)
    return 0.5 * (t_lo + t_hi), found


def render():
    img_dir = OUT / "images"; msk_dir = OUT / "masks"
    for dd in (img_dir, msk_dir):
        dd.mkdir(parents=True, exist_ok=True)
        for p in dd.glob("*.png"): p.unlink()
    tex = make_texture(); htex, wtex = tex.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float32)[:, None, :]
    nrm = cv2.undistortPoints(pix, K, D).reshape(-1, 2)
    d_cam = np.concatenate([nrm, np.ones((len(nrm), 1))], 1); d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    gt = []; zc = np.linspace(6.0, ZS - 6.0, N_FRAMES)                    # stay BEHIND the throat
    for i in range(N_FRAMES):
        ph = 2 * np.pi * i / 33.0
        c = np.array([0.6 * np.sin(ph), 0.6 * np.cos(0.7 * ph), zc[i]])
        pitch = 0.05 * np.sin(0.5 * ph); yaw = 0.05 * np.cos(0.3 * ph)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R_wc = Rx @ Ry
        d_world = (R_wc.T @ d_cam.T).T
        t, valid = first_hit(c, d_world)
        hit = c[None, :] + t[:, None] * d_world
        zh = hit[:, 2]; ok = valid & (zh >= 0) & (zh <= TUBE_LEN)
        theta = np.arctan2(hit[:, 1], hit[:, 0])
        tu = np.clip(((theta + np.pi) / (2 * np.pi) * wtex).astype(int), 0, wtex - 1)
        tv = np.clip((zh / TUBE_LEN * htex).astype(int), 0, htex - 1)
        col = tex[tv, tu].astype(np.float32) * (1.0 / (1.0 + (t / 14.0) ** 2))[:, None]
        col[~ok] = 0
        img = col.reshape(H, W, 3).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_dir / f"f{i:05d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        # GT through-throat aperture mask: rays whose first hit is BEYOND the throat minimum
        # (z>ZS) -> boundary = throat's narrowest ring, radius exactly RT at z=ZS.
        aperture = (ok & (zh > ZS)).reshape(H, W).astype(np.uint8) * 255
        cv2.imwrite(str(msk_dir / f"f{i:05d}.png"), aperture, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        gt.append({"frame": i, "C": c.tolist(), "R_wc": R_wc.tolist()})
        if i % 40 == 0: print(f"  rendered {i}/{N_FRAMES}", flush=True)
    (OUT / "gt_poses.json").write_text(json.dumps(
        {"R_normal": RN, "R_throat": RT, "throat_z": ZS, "sigma": SIGMA, "tube_len": TUBE_LEN,
         "W": W, "H": H, "poses": gt}, indent=2))
    return sorted(p.name for p in img_dir.glob("f*.png"))


def sh(cmd): return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def ply_n(p):
    if not Path(p).exists(): return 0
    with open(p, "rb") as f:
        for _ in range(60):
            ln = f.readline().decode("ascii", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
            if ln.startswith("end_header"): return 0
    return 0


def colmap():
    import pycolmap, shutil
    img_dir = OUT / "images"; names = sorted(p.name for p in img_dir.glob("f*.png"))
    params = ",".join(f"{p:.10g}" for p in PARAMS); db = OUT / "database.db"
    if db.exists(): db.unlink()
    sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", params, "--SiftExtraction.max_image_size", "1600"])
    print("  features done", flush=True)
    sh(["sequential_matcher", "--database_path", str(db),
        "--SequentialMatching.overlap", str(OVERLAP), "--SequentialMatching.quadratic_overlap", "1"])
    print("  matching done", flush=True)
    sparse = OUT / "sparse"; sparse.mkdir(exist_ok=True)
    sh(["mapper", "--database_path", str(db), "--image_path", str(img_dir), "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_principal_point", "0"])
    models = sorted([d for d in sparse.iterdir() if d.is_dir()])
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rec = pycolmap.Reconstruction(str(best))
    print(f"  mapper: {len(models)} model(s); best {best.name} reg={rec.num_reg_images()}/{len(names)} sparse={rec.num_points3D()}", flush=True)
    if best.name != "0":
        (sparse / "0").mkdir(exist_ok=True)
        for f in best.glob("*"): shutil.copy(str(f), str(sparse / "0" / f.name))
    dense = OUT / "dense0"; dense.mkdir(exist_ok=True)
    sh(["image_undistorter", "--image_path", str(img_dir), "--input_path", str(sparse / "0"),
        "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    sh(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1100"])
    fused = dense / "fused.ply"
    sh(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--input_type", "geometric", "--output_path", str(fused)])
    summ = {"stenosis": True, "R_normal": RN, "R_throat": RT, "throat_z": ZS, "n_frames": len(names),
            "n_models": len(models), "n_registered": rec.num_reg_images(), "n_sparse": rec.num_points3D(),
            "n_dense": ply_n(fused), "dense_ply": str(fused)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== STENOSIS RECON ===")
    for k in ("n_frames", "n_registered", "n_models", "n_sparse", "n_dense"): print(f"  {k}: {summ[k]}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"STENOSIS phantom: Rn={RN} Rt={RT} (throat DCE={2*RT}) zs={ZS} sigma={SIGMA} @ {W}x{H}, {N_FRAMES} frames")
    if which in ("render", "all"):
        names = render(); print(f"rendered {len(names)} frames + masks")
    if which in ("colmap", "all"):
        colmap()


if __name__ == "__main__":
    main()
