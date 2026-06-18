"""DECISIVE PHANTOM TEST: can the Barbour-style pipeline reconstruct + close +
measure a KNOWN textured tube under perfect ground truth?

Synthetic textured cylinder, radius R_GT, forward passthrough with the SAME
intrinsics/resolution as 7_v1 (OPENCV 1920x1080). CLAHE -> COLMAP SIFT sequential
-> MVS -> Poisson, then (separately) the same lumen-axis + mesh-slice code.

Records GT camera poses + GT radius for absolute comparison (recon aligned to GT
by camera-center similarity).

Run in depth-eval env. Output runs/phantom/.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np, cv2

OUT = Path("/home/mi3dr/projects/bronchotrust/runs/phantom")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
INTR = json.loads(Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib/7_v1/intrinsics_pinned.json").read_text())
W, H = INTR["width"], INTR["height"]
K = np.array([[INTR["fx"], 0, INTR["cx"]], [0, INTR["fy"], INTR["cy"]], [0, 0, 1.0]])
D = np.array([INTR["k1"], INTR["k2"], 0.0, 0.0])
R_GT = 5.0           # known tube radius (scene units)
TUBE_LEN = 60.0
N_FRAMES = 350       # dense forward passthrough (7_v1 subglottis-model regime)
OVERLAP = 30


def make_texture(htex=1024, wtex=2048, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((htex, wtex, 3), np.float32)
    for sc in (32, 8, 2):  # multi-scale -> SIFT features at several scales
        n = rng.random((htex // sc, wtex // sc, 3)).astype(np.float32)
        base += cv2.resize(n, (wtex, htex), interpolation=cv2.INTER_LINEAR) / 3
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    # mucosa-ish tint for extra structure
    tint = np.array([0.55, 0.45, 0.85])  # BGR pinkish
    tex = np.clip(0.35 + 0.65 * base, 0, 1) * tint
    return (tex * 255).astype(np.uint8)


def render():
    img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.png"):
        p.unlink()
    tex = make_texture(); htex, wtex = tex.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    # per-pixel normalized camera rays (fixed) via OPENCV undistort
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float32)[:, None, :]
    nrm = cv2.undistortPoints(pix, K, D).reshape(-1, 2)
    d_cam = np.concatenate([nrm, np.ones((len(nrm), 1))], 1)        # (Npix,3)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    gt = []
    zc = np.linspace(3.0, TUBE_LEN * 0.72, N_FRAMES)
    for i in range(N_FRAMES):
        ph = 2 * np.pi * i / 35.0
        c = np.array([0.6 * np.sin(ph), 0.6 * np.cos(0.7 * ph), zc[i]])    # small lateral wobble
        # look ~+z with small tilt wobble
        pitch = 0.05 * np.sin(0.5 * ph); yaw = 0.05 * np.cos(0.3 * ph)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R_wc = (Rx @ Ry)            # world->camera (camera looks +z when R=I)
        d_world = (R_wc.T @ d_cam.T).T                                 # (Npix,3)
        ox, oy = c[0], c[1]
        a = d_world[:, 0] ** 2 + d_world[:, 1] ** 2
        b = 2 * (ox * d_world[:, 0] + oy * d_world[:, 1])
        cq = ox ** 2 + oy ** 2 - R_GT ** 2
        disc = b ** 2 - 4 * a * cq
        valid = disc > 0
        t = np.full(len(d_world), -1.0)
        sq = np.sqrt(np.maximum(disc, 0))
        t[valid] = (-b[valid] + sq[valid]) / (2 * a[valid] + 1e-12)     # forward inner-wall hit
        hit = c[None, :] + t[:, None] * d_world
        zc_hit = hit[:, 2]
        ok = valid & (t > 0) & (zc_hit >= 0) & (zc_hit <= TUBE_LEN)
        theta = np.arctan2(hit[:, 1], hit[:, 0])
        tu = np.clip(((theta + np.pi) / (2 * np.pi) * wtex).astype(int), 0, wtex - 1)
        tv = np.clip((zc_hit / TUBE_LEN * htex).astype(int), 0, htex - 1)
        col = tex[tv, tu].astype(np.float32)
        bright = 1.0 / (1.0 + (t / 14.0) ** 2)                          # scope-light falloff
        col *= bright[:, None]
        col[~ok] = 0
        img = col.reshape(H, W, 3).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_dir / f"f{i:05d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        gt.append({"frame": i, "C": c.tolist(), "R_wc": R_wc.tolist()})
        if i % 50 == 0:
            print(f"  rendered {i}/{N_FRAMES}", flush=True)
    (OUT / "gt_poses.json").write_text(json.dumps({"R_GT": R_GT, "tube_len": TUBE_LEN,
                                                   "W": W, "H": H, "poses": gt}, indent=2))
    return sorted(p.name for p in img_dir.glob("f*.png"))


def run(cmd):
    return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def ply_n(p):
    if not Path(p).exists():
        return 0
    with open(p, "rb") as f:
        for _ in range(60):
            ln = f.readline().decode("ascii", "ignore")
            if ln.startswith("element vertex"):
                return int(ln.split()[-1])
            if ln.startswith("end_header"):
                return 0
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"PHANTOM: tube R_GT={R_GT}, len={TUBE_LEN}, {N_FRAMES} frames @ {W}x{H}, OPENCV fx={INTR['fx']:.1f}")
    names = render()
    print(f"rendered {len(names)} frames")
    img_dir = OUT / "images"
    params = ",".join(f"{p:.10g}" for p in INTR["params_colmap"])
    db = OUT / "database.db"
    if db.exists():
        db.unlink()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", params, "--SiftExtraction.max_image_size", "1600"])
    print("  features done", flush=True)
    run(["sequential_matcher", "--database_path", str(db),
         "--SequentialMatching.overlap", str(OVERLAP), "--SequentialMatching.quadratic_overlap", "1"])
    print("  matching done", flush=True)
    sparse = OUT / "sparse"; sparse.mkdir(exist_ok=True)
    run(["mapper", "--database_path", str(db), "--image_path", str(img_dir), "--output_path", str(sparse),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0"])
    import pycolmap
    models = sorted([d for d in sparse.iterdir() if d.is_dir()])
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rec = pycolmap.Reconstruction(str(best))
    print(f"  mapper: {len(models)} model(s); largest {best.name} reg={rec.num_reg_images()}/{len(names)} sparse={rec.num_points3D()}", flush=True)
    # rename largest -> sparse/0 convention already; ensure dense0 from best
    dense = OUT / "dense0"; dense.mkdir(exist_ok=True)
    run(["image_undistorter", "--image_path", str(img_dir), "--input_path", str(best),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1100"])
    fused = dense / "fused.ply"
    run(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
    n_dense = ply_n(fused)
    run(["poisson_mesher", "--input_path", str(fused), "--output_path", str(dense / "meshed-poisson.ply"),
         "--PoissonMeshing.depth", "10", "--PoissonMeshing.trim", "5"])
    # write best model also as sparse/0 if not already
    if best.name != "0":
        import shutil
        (sparse / "0").mkdir(exist_ok=True)
        for f in best.glob("*"):
            shutil.copy(str(f), str(sparse / "0" / f.name))
    summ = {"phantom": True, "R_GT": R_GT, "tube_len": TUBE_LEN, "n_extracted": len(names),
            "n_models": len(models), "n_registered": rec.num_reg_images(), "n_sparse": rec.num_points3D(),
            "n_dense": n_dense, "n_mesh_verts": ply_n(dense / "meshed-poisson.ply"),
            "sfm_model_dir": str(sparse / "0"), "dense_ply": str(fused),
            "mesh_ply": str(dense / "meshed-poisson.ply"),
            "camera_params": list(next(iter(rec.cameras.values())).params)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== PHANTOM RECON ===")
    for k in ("n_extracted", "n_registered", "n_models", "n_sparse", "n_dense", "n_mesh_verts"):
        print(f"  {k}: {summ[k]}")
    print(f"saved {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
