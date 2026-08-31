"""Barbour-style DENSE reconstruction (the corrective pipeline).

Matches Barbour et al. 2025 (COLMAP, Storz 4.0mm scope): a DENSE continuous
passthrough (hundreds of frames), contrast enhancement (CLAHE), COLMAP SIFT +
SEQUENTIAL matching, incremental mapping with PINNED OPENCV intrinsics, MVS,
then Poisson mesh (the surface we will later slice -- NOT raw points).

This replaces our prior sparse 36-58 frame curation, which was the root cause of
the failed reconstructions.

Run in depth-eval env (cv2) ; uses the colmap-cuda binary for SfM/MVS.
Output: runs/barbour_dense/<stem>/
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import cv2, numpy as np

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
CALDIR = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/barbour_dense")
DARK_L = 12.0


def content_mask(video, n=60):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
                  np.ones((5, 5), np.uint8), iterations=4)
    return m > 0, (W, H)


def extract_dense(video, mask, lo, hi, stride, img_dir, msk_dir, clahe_on):
    """Sequential read; CLAHE-enhance; bezel; save every `stride`-th VALID frame in [lo,hi]."""
    img_dir.mkdir(parents=True, exist_ok=True); msk_dir.mkdir(parents=True, exist_ok=True)
    for p in img_dir.glob("*.png"):
        p.unlink()
    for p in msk_dir.glob("*.png"):
        p.unlink()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    mask_u8 = (mask.astype(np.uint8)) * 255
    cap = cv2.VideoCapture(str(video))
    fi = 0; kept = 0; vcount = 0; names = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if lo <= fi <= hi:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
            Lroi = float(lab[:, :, 0][mask].mean() * 100 / 255) if mask.sum() else 0.0
            if Lroi >= DARK_L:
                if vcount % stride == 0:
                    if clahe_on:
                        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                        fr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    fr[~mask] = 0
                    nm = f"f{fi:05d}.png"
                    cv2.imwrite(str(img_dir / nm), fr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    cv2.imwrite(str(msk_dir / (nm + ".png")), mask_u8, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    names.append(nm); kept += 1
                vcount += 1
        fi += 1
        if fi > hi:
            break
    cap.release()
    return names


def run(cmd, log):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    log.write_text((log.read_text() if log.exists() else "") + "\n$ colmap " + " ".join(cmd) +
                   f"\n[rc={r.returncode}]\n" + r.stdout[-1500:] + r.stderr[-1500:])
    return r


def ply_count(ply):
    if not ply.exists():
        return 0
    with open(ply, "rb") as fh:
        for _ in range(60):
            ln = fh.readline().decode("ascii", "ignore")
            if ln.startswith("element vertex"):
                return int(ln.split()[-1])
            if ln.startswith("end_header"):
                return 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=100000)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--no-clahe", action="store_true")
    args = ap.parse_args()

    video = DATA / args.video
    od = OUT / Path(args.video).stem
    od.mkdir(parents=True, exist_ok=True)
    log = od / "colmap.log"
    if log.exists():
        log.unlink()
    intr = json.loads((CALDIR / args.session / "intrinsics_pinned.json").read_text())
    params = f"{intr['fx']:.10g},{intr['fy']:.10g},{intr['cx']:.10g},{intr['cy']:.10g},{intr['k1']:.10g},{intr['k2']:.10g},0,0"
    print(f"DENSE recon {args.video}: passthrough f{args.lo}-{args.hi} stride {args.stride} "
          f"clahe={not args.no_clahe}  pinned OPENCV fx={intr['fx']:.2f}", flush=True)

    mask, (W, H) = content_mask(video)
    img_dir = od / "images"; msk_dir = od / "masks"
    names = extract_dense(video, mask, args.lo, args.hi, args.stride, img_dir, msk_dir, not args.no_clahe)
    print(f"extracted {len(names)} dense frames -> {img_dir}", flush=True)

    db = od / "database.db"
    if db.exists():
        db.unlink()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
         "--ImageReader.mask_path", str(msk_dir),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", params,
         "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.estimate_affine_shape", "1",
         "--SiftExtraction.domain_size_pooling", "1"], log)
    print("  features done", flush=True)
    run(["sequential_matcher", "--database_path", str(db),
         "--SequentialMatching.overlap", str(args.overlap),
         "--SequentialMatching.quadratic_overlap", "1"], log)
    print("  sequential matching done", flush=True)
    sparse = od / "sparse"; sparse.mkdir(exist_ok=True)
    run(["mapper", "--database_path", str(db), "--image_path", str(img_dir), "--output_path", str(sparse),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
         "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4"], log)
    models = sorted([d for d in sparse.iterdir() if d.is_dir()])
    if not models:
        print("  MAPPER PRODUCED NO MODEL"); return
    # pick largest model by #images (cameras.bin/images.bin)
    import pycolmap
    best = max(models, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
    rec = pycolmap.Reconstruction(str(best))
    n_comp = len(models)
    print(f"  mapper: {n_comp} model(s); largest {best.name} reg={rec.num_reg_images()}/{len(names)} sparse={rec.num_points3D()}", flush=True)

    dense = od / "dense"; dense.mkdir(exist_ok=True)
    run(["image_undistorter", "--image_path", str(img_dir), "--input_path", str(best),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"], log)
    run(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"], log)
    fused = dense / "fused.ply"
    run(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)], log)
    n_dense = ply_count(fused)
    print(f"  MVS dense: {n_dense}", flush=True)
    # Poisson mesh (Barbour slices the MESH, not raw points)
    mesh = dense / "meshed-poisson.ply"
    run(["poisson_mesher", "--input_path", str(fused), "--output_path", str(mesh),
         "--PoissonMeshing.depth", "10", "--PoissonMeshing.trim", "7"], log)
    n_mesh = ply_count(mesh)

    cam = next(iter(rec.cameras.values()))
    summary = {"video": args.video, "session": args.session, "approx": False,
               "passthrough": [args.lo, args.hi], "stride": args.stride, "clahe": not args.no_clahe,
               "n_extracted": len(names), "n_models": n_comp, "largest_model": best.name,
               "n_registered": rec.num_reg_images(), "n_sparse": rec.num_points3D(),
               "n_dense": n_dense, "n_mesh_verts": n_mesh,
               "camera_model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
               "camera_params": list(cam.params),
               "pin_delta": float(max(abs(a - b) for a, b in zip(
                   cam.params, [intr['fx'], intr['fy'], intr['cx'], intr['cy'], intr['k1'], intr['k2'], 0, 0]))),
               "sfm_model_dir": str(best), "dense_ply": str(fused), "mesh_ply": str(mesh)}
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== DENSE RECON {args.video} ===")
    for k in ("n_extracted", "n_registered", "n_sparse", "n_dense", "n_mesh_verts", "n_models", "pin_delta"):
        print(f"  {k}: {summary[k]}")
    print(f"saved {od/'summary.json'}")


if __name__ == "__main__":
    main()
