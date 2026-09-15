"""Systematic airway-recovery harness. One command per windowed clip. Encodes the proven recipe:
  window -> bezel + gentle CLAHE brighten -> pinned-intrinsics SfM (exhaustive, mild SIFT) ->
  auto-diagnose fragmentation -> model_merger bridge -> 4-GPU MVS -> partial-arc CSA -> render.
Reports each stage + a final verdict per clip. Does NOT auto-apply motion-cut (that's clip-specific).

Usage: python pipeline/recover_clip.py --video <clip> --calib <intrinsics.json> --lo L --hi H --out <dir>
                                       [--session NAME] [--gpus 0,1,2,3] [--clahe 3.0] [--target_fps 30]
  --calib   intrinsics json from pipeline/calibrate.py (pinned OPENCV; never refined in bundle adjustment)
  --lo/--hi frame window (pick it from pipeline/contact_sheet.py); a steady, centred, monotonic pass works best
Environment: BRONCHO_COLMAP (colmap binary, default "colmap"), BRONCHO_PY (interpreter for the post-steps,
default: this one). Outputs: <out>/sparse/0 (poses), <out>/dense0/fused.ply (dense cloud), CSA profile + render PNGs.
"""
import argparse, re, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2, pycolmap

ROOT = Path(__file__).resolve().parents[1]
COLMAP = os.environ.get("BRONCHO_COLMAP", "colmap")
PY = os.environ.get("BRONCHO_PY", sys.executable)
DARK_L = 12.0


def find_calib(session, explicit=None):
    """explicit --calib path first; otherwise the two legacy per-session locations under runs/."""
    cands = [Path(explicit)] if explicit else []
    cands += [ROOT / f"runs/retriage_first15/_calib/{session}/intrinsics_pinned.json", ROOT / f"runs/own_data/retry/{session}_intrinsics.json"]
    for p in cands:
        if p.exists():
            d = json.loads(p.read_text()); params = d.get("params_colmap") or d.get("params")
            return ",".join(f"{x:.10g}" for x in params), p
    return None, None


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), iterations=5) > 0


def sh(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-200:], flush=True)
    return r


def span(model):
    r = pycolmap.Reconstruction(str(model)); fr = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in r.images.values())
    return r.num_reg_images(), fr[0], fr[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--calib", default=None, help="intrinsics json (pipeline/calibrate.py)")
    ap.add_argument("--session", default=None, help="label; also used to locate a legacy calibration if --calib is omitted")
    ap.add_argument("--lo", type=int, required=True); ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--clahe", type=float, default=3.0); ap.add_argument("--mvs_size", default="1200")
    ap.add_argument("--target_fps", type=float, default=30.0)   # normalize input fps (inert on 30fps clips; stride 2 on 60fps)
    a = ap.parse_args()
    out = Path(a.out); img = out / "images"; msk = out / "masks"
    video = Path(a.video)
    if not video.exists(): print(f"video not found: {video}"); return
    if a.session is None: a.session = out.name
    pstr, cpath = find_calib(a.session, a.calib)
    print(f"=== recover {a.session} f{a.lo}-{a.hi} | calib {cpath} | gpus {a.gpus} ===", flush=True)
    if pstr is None: print("NO CALIB — pass --calib <intrinsics.json> (pipeline/calibrate.py); abort"); return
    out.mkdir(parents=True, exist_ok=True)

    for d in (img, msk): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = bezel(video); bez = mask.astype(np.uint8) * 255
    clahe = cv2.createCLAHE(clipLimit=a.clahe, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(video)); fi = 0; n = 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / a.target_fps))              # fps normalization: 30fps->1, 60fps->2
    print(f"[fps] source {src_fps:.1f} -> target {a.target_fps:.0f} -> stride {stride}", flush=True)
    while True:
        ok, fr = cap.read()
        if not ok or fi > a.hi: break
        if a.lo <= fi <= a.hi and (fi - a.lo) % stride == 0 and cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255 >= DARK_L:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
            cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release(); print(f"[extract] {n} CLAHE(clip{a.clahe}) frames", flush=True)

    db = out / "db.db"; db.unlink(missing_ok=True)
    sh(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
        "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", pstr,
        "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192", "--SiftExtraction.peak_threshold", "0.005"])
    sh(["exhaustive_matcher", "--database_path", str(db)])
    sp = out / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    sh(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_principal_point", "0", "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = sorted([d for d in sp.iterdir() if d.is_dir()], key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
    if not models: print("[SfM] NO MODEL -> registration FAILED (texture/calib limit)"); return
    regs = [span(m) for m in models]
    print(f"[SfM] {len(models)} model(s): " + " | ".join(f"{r}fr f{lo}-{hi}" for r, lo, hi in regs), flush=True)

    final = models[0]
    # auto-bridge: if fragmented, iteratively model_merger the largest into the rest
    if len(models) > 1:
        merged = sp / "merged"; shutil.rmtree(merged, ignore_errors=True)
        cur = str(models[0])
        for nxt in models[1:4]:
            sh(["model_merger", "--input_path1", cur, "--input_path2", str(nxt), "--output_path", str(merged)])
            if (merged / "images.bin").exists(): cur = str(merged)
        if (merged / "images.bin").exists():
            r, lo, hi = span(merged); print(f"[bridge] model_merger -> {r}fr f{lo}-{hi}", flush=True); final = merged

    r, lo, hi = span(final)
    rec = pycolmap.Reconstruction(str(final))
    try: mre = rec.compute_mean_reprojection_error()
    except: mre = float("nan")
    print(f"[model] {r} reg, span f{lo}-{hi}, reproj {mre:.2f}px", flush=True)
    keep = sp / "0";
    if str(final) != str(keep):
        shutil.rmtree(keep, ignore_errors=True); shutil.copytree(final, keep)

    dense = out / "dense0"; shutil.rmtree(dense, ignore_errors=True)
    sh(["image_undistorter", "--image_path", str(img), "--input_path", str(keep), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    sh(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", a.mvs_size, "--PatchMatchStereo.gpu_index", a.gpus])
    fused = dense / "fused.ply"
    sh(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--input_type", "geometric", "--StereoFusion.min_num_pixels", "3", "--output_path", str(fused)])
    if not fused.exists(): print("[MVS] fusion produced no cloud"); return
    lab = a.session.replace("_", "-").upper()
    HERE = Path(__file__).resolve().parent
    subprocess.run([PY, str(HERE / "csa_run.py"), str(fused), lab, str(out / f"{a.session}_csa_profile.png")])
    subprocess.run([PY, str(HERE / "render_airway.py"), str(fused), lab, str(out / f"{a.session}_airway_recon.png")])
    shutil.rmtree(dense / "stereo", ignore_errors=True); shutil.rmtree(dense / "images", ignore_errors=True); shutil.rmtree(img, ignore_errors=True); db.unlink(missing_ok=True)
    print(f"[DONE] {a.session}: fused.ply + CSA + airway render saved", flush=True)


if __name__ == "__main__":
    main()
