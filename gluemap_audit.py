"""GlueMap fair re-test (gluemap-audit branch).

Replace ONLY the sparse front-end with GlueMap; hold everything else identical to
the working COLMAP baseline (tag colmap-working-baseline):
  - same window / CLAHE+bezel frames           (runs/batch4/<v>/images, full window)
  - same pinned intrinsics                      (runs/retriage_first15/_calib/<s>/intrinsics_pinned.json)
  - same MVS settings                           (image_undistorter + patch_match geom + fusion, same flags as batch_dense.py)
  - same scaling assumption + landmark code     (batch_render.py UNCHANGED: k0=R_PHYS/Rmed, glottis/+5/+10mm)

Per video:
  1. undistort the FULL-window CLAHE frames with the pinned OPENCV intrinsics (cv2)
     -> pinhole frames + a PINHOLE GT model covering ALL frames, so GlueMap sees the
        SAME N frames the COLMAP front-end saw (not a pre-filtered registered subset).
  2. GlueMap front-end (pi3 + SALAD + Doppelgangers + augmented BA), pinned via
     --use_gt_intrinsics --gt_intrinsics_path <pinhole GT>.
  3. SAME COLMAP MVS on the GlueMap model.
  4. batch_render.py UNCHANGED -> A/B/C figures + landmarks + report.
  5. SfM metrics (focal before/after, reg, components, outlier cams, sparse, dense,
     reproj) + COLMAP-vs-GlueMap comparison -> runs/gluemap_audit/<out>/report.json

Idempotent: each heavy step is skipped if its output already exists.
Orchestrator runs in depth-eval (cv2/pycolmap/numpy); MVS via colmap-cuda binary;
front-end via the gluemap env's gluemap-demo subprocess."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, re
from pathlib import Path
import numpy as np, cv2, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
CALIB = ROOT / "runs/retriage_first15/_calib"
BATCH4 = ROOT / "runs/batch4"
AUDIT = ROOT / "runs/gluemap_audit"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
GLUEMAP = "/home/mi3dr/.conda/envs/gluemap/bin/gluemap-demo"
GM_SRC = Path("/home/mi3dr/projects/gluemap_src")
GM_CONFIG = GM_SRC / "configs/example.yaml"
DEPTH_PY = "/home/mi3dr/.conda/envs/depth-eval/bin/python"

VIDEOS = {
    "2_V2":  dict(src="2-V2",  session="2_v2",  glottis=875, calibnote="2_v2 own calibration · GlueMap front-end"),
    "25_V1": dict(src="25-V1", session="25_v1", glottis=335, calibnote="25_v1 own calibration · GlueMap front-end"),
}

# ---- metric helpers (identical definitions to fuse_32v2.py for a like-for-like A/B) ----

def centers(rec):
    out = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); C = -M[:3, :3].T @ M[:3, 3]
        out[int(re.search(r"f(\d+)", im.name).group(1))] = C
    return out


def outlier_ratio(C):
    fis = sorted(C); P = np.array([C[f] for f in fis])
    if len(P) < 5: return 1.0, np.zeros(len(P), bool)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-9
    gross = np.any(np.abs(P - med) > 6 * mad * 1.4826, axis=1)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    smed = np.median(step); jump = np.zeros(len(P), bool)
    big = step > 8 * smed
    jump[1:] |= big; jump[:-1] |= big
    out = gross | jump
    return float(out.mean()), out


def frame_runs(C):
    fis = sorted(C); runs = []; s = fis[0]; p = fis[0]
    for f in fis[1:]:
        if f - p > 8: runs.append((s, p)); s = f
        p = f
    runs.append((s, p)); return runs


def sfm_metrics(model_dir):
    rec = pycolmap.Reconstruction(str(model_dir))
    C = centers(rec); orat, _ = outlier_ratio(C)
    errs = np.array([p.error for p in rec.points3D.values()]) if rec.num_points3D() else np.array([0.0])
    cam = list(rec.cameras.values())[0]
    return {"reg": rec.num_reg_images(), "sparse": rec.num_points3D(),
            "reproj_px": round(float(errs.mean()), 3), "outlier_ratio": round(orat, 3),
            "frame_runs": frame_runs(C), "focal_fx": round(float(cam.params[0]), 3),
            "focal_fy": round(float(cam.params[1]), 3),
            "model": cam.model.name if hasattr(cam.model, "name") else str(cam.model)}


def ply_n(p):
    if not Path(p).exists(): return 0
    with open(p, "rb") as f:
        for _ in range(40):
            ln = f.readline().decode("latin1", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
    return 0


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd[:3]), "...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("  ERR", cmd[0], "rc=", r.returncode, "\n", (r.stderr or r.stdout)[-1500:], flush=True)
    return r


# ---- step 1: undistort full window + PINHOLE GT (text) covering ALL frames ----

def undistort_window(key, cfg, aud):
    src = BATCH4 / cfg["src"] / "images"
    und = aud / "undist_images"; gt = aud / "gt_pinhole"
    j = json.loads((CALIB / cfg["session"] / "intrinsics_pinned.json").read_text())
    W, H = int(j["width"]), int(j["height"])
    fx, fy, cx, cy, k1, k2, p1, p2 = j["params_colmap"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); dist = np.array([k1, k2, p1, p2])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0, (W, H))
    nfx, nfy, ncx, ncy = newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2]
    names = sorted(p.name for p in src.glob("f*.png"))
    if und.exists() and len(list(und.glob("f*.png"))) == len(names):
        print(f"[{key}] undistorted frames already present ({len(names)})", flush=True)
    else:
        shutil.rmtree(und, ignore_errors=True); und.mkdir(parents=True)
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_16SC2)
        for n in names:
            img = cv2.imread(str(src / n))
            cv2.imwrite(str(und / n), cv2.remap(img, m1, m2, cv2.INTER_LINEAR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f"[{key}] undistorted {len(names)} full-window frames (pinhole newK fx={nfx:.2f})", flush=True)
    # PINHOLE GT model (legacy text) for ALL frames -> GlueMap reads K by image name
    gt.mkdir(parents=True, exist_ok=True)
    (gt / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE %d %d %.8f %.8f %.8f %.8f\n" % (W, H, nfx, nfy, ncx, ncy))
    lines = ["# Image list"]
    for i, n in enumerate(names, 1):
        lines.append("%d 1 0 0 0 0 0 0 1 %s" % (i, n)); lines.append("")
    (gt / "images.txt").write_text("\n".join(lines) + "\n")
    (gt / "points3D.txt").write_text("# 3D point list\n")
    return und, gt, dict(W=W, H=H, newK=[float(nfx), float(nfy), float(ncx), float(ncy)],
                         pinned_opencv_fx=float(fx), n_frames=len(names))


# ---- step 2: GlueMap front-end ----

def run_gluemap(key, aud, und, gt):
    aba = aud / "gm" / "gluemap_aba"
    if (aba / "images.bin").exists() or (aba / "images.txt").exists():
        print(f"[{key}] GlueMap model already present -> {aba}", flush=True); return aba
    write = aud / "gm"; shutil.rmtree(write, ignore_errors=True); write.mkdir(parents=True)
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    r = run([GLUEMAP, "--config", str(GM_CONFIG), "--images_path", str(und),
             "--write_path", str(write), "--intrinsics_mode", "SHARED",
             "--gt_intrinsics_path", str(gt), "--use_gt_intrinsics"],
            cwd=str(GM_SRC), env=env)
    log = aud / "gm" / "gluemap_run.log"
    log.write_text((r.stdout or "") + "\n==STDERR==\n" + (r.stderr or ""))
    if not ((aba / "images.bin").exists() or (aba / "images.txt").exists()):
        raise RuntimeError(f"GlueMap produced no model at {aba}; see {log}")
    print(f"[{key}] GlueMap model written -> {aba}", flush=True)
    return aba


# ---- step 3: MVS on the GlueMap model (same flags as batch_dense.py) ----

def run_mvs(key, aud, und, aba):
    dense = aud / "gm_dense"; fused = dense / "fused.ply"
    if fused.exists() and ply_n(fused) > 0:
        print(f"[{key}] GlueMap dense already present ({ply_n(fused)} pts)", flush=True); return fused
    shutil.rmtree(dense, ignore_errors=True)
    model = aba
    r = run([COLMAP, "image_undistorter", "--image_path", str(und), "--input_path", str(model),
             "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    if not (dense / "sparse").exists():
        leg = aud / "gm" / "aba_legacy"; leg.mkdir(parents=True, exist_ok=True)
        pycolmap.Reconstruction(str(aba)).write(str(leg))
        run([COLMAP, "image_undistorter", "--image_path", str(und), "--input_path", str(leg),
             "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run([COLMAP, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"])
    run([COLMAP, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
    print(f"[{key}] GlueMap DENSE {ply_n(fused)} fused points", flush=True)
    return fused


# ---- step 4: measurement (UNCHANGED batch_render.py) ----

def run_render(key, cfg, aud, fused, aba):
    if (aud / "report.json").exists() and (aud / "A_organ_cloud.png").exists():
        print(f"[{key}] batch_render output already present", flush=True); return
    r = run([DEPTH_PY, str(ROOT / "batch_render.py"), "--dense", str(fused), "--sparse", str(aba),
             "--glottis", str(cfg["glottis"]), "--label", f"{cfg['src']} GlueMap",
             "--calibnote", cfg["calibnote"], "--out", str(aud)], cwd=str(ROOT))
    print((r.stdout or "")[-600:], flush=True)


# ---- step 5: comparison report ----

def build_report(key, cfg, aud, meta, gm_focal_given, fused):
    gm_sfm = sfm_metrics(aud / "gm" / "gluemap_aba")
    gm_land = json.loads((aud / "report.json").read_text())  # batch_render landmark report
    col_sfm = sfm_metrics(BATCH4 / cfg["src"] / "sparse/0")
    col_dsum = json.loads((BATCH4 / cfg["src"] / "dense_summary.json").read_text())
    col_land = json.loads((BATCH4 / cfg["src"] / "report.json").read_text())
    pinned = meta["pinned_opencv_fx"]

    def lm_row(d, k): r = d["landmarks"][k]; return {"coverage": r["coverage"], "r_std/r_med": r["r_std/r_med"], "closed": r["closed"]}

    gm_drift = round(gm_sfm["focal_fx"] / gm_focal_given - 1.0, 4)
    col_drift = round(col_sfm["focal_fx"] / pinned - 1.0, 4)
    out = {
        "video": key, "source": cfg["src"], "session": cfg["session"], "glottis_frame": cfg["glottis"],
        "n_frames_window": meta["n_frames"],
        "COLMAP": {
            "registered": col_dsum["reg"], "n_frames": col_dsum["n_frames"], "components": col_dsum["n_models"],
            "focal_pinned": round(pinned, 3), "focal_after": col_sfm["focal_fx"], "focal_drift": col_drift,
            "outlier_ratio": col_sfm["outlier_ratio"], "sparse": col_sfm["sparse"],
            "dense_pts": col_dsum["dense_pts"], "reproj_px": col_sfm["reproj_px"],
            "landmarks": {k: lm_row(col_land, k) for k in ("glottis", "+5mm", "+10mm")}},
        "GlueMap": {
            "registered": gm_sfm["reg"], "n_frames": meta["n_frames"], "components": 1,
            "frame_runs": gm_sfm["frame_runs"],
            "focal_given": round(gm_focal_given, 3), "focal_after": gm_sfm["focal_fx"], "focal_drift": gm_drift,
            "outlier_ratio": gm_sfm["outlier_ratio"], "sparse": gm_sfm["sparse"],
            "dense_pts": ply_n(fused), "reproj_px": gm_sfm["reproj_px"],
            "landmarks": {k: lm_row(gm_land, k) for k in ("glottis", "+5mm", "+10mm")}},
    }
    # pass criteria
    g = out["GlueMap"]; c = out["COLMAP"]
    no_drift = abs(g["focal_drift"]) <= 0.02
    no_outliers = g["outlier_ratio"] <= 0.05
    def closed(arm, k): return arm["landmarks"][k]["closed"]
    p5 = closed(g, "+5mm") and (not closed(c, "+5mm") or g["landmarks"]["+5mm"]["r_std/r_med"] <= c["landmarks"]["+5mm"]["r_std/r_med"] + 0.02)
    p10 = closed(g, "+10mm") and (not closed(c, "+10mm") or g["landmarks"]["+10mm"]["r_std/r_med"] <= c["landmarks"]["+10mm"]["r_std/r_med"] + 0.02)
    checks = {"no_focal_drift": bool(no_drift), "no_major_camera_outliers": bool(no_outliers),
              "+5mm_equal_or_better": bool(p5), "+10mm_equal_or_better": bool(p10)}
    out["pass_checks"] = checks
    out["verdict"] = "ACCEPT" if all(checks.values()) else "REJECT"
    (aud / "report.json").write_text(json.dumps(out, indent=2))
    return out


def process(key):
    cfg = VIDEOS[key]; aud = AUDIT / key; aud.mkdir(parents=True, exist_ok=True)
    print(f"\n========== {key} ({cfg['src']}) ==========", flush=True)
    und, gt, meta = undistort_window(key, cfg, aud)
    aba = run_gluemap(key, aud, und, gt)
    fused = run_mvs(key, aud, und, aba)
    run_render(key, cfg, aud, fused, aba)
    rep = build_report(key, cfg, aud, meta, meta["newK"][0], fused)
    # cleanup heavy regenerable MVS workspace, keep fused.ply + model + figures + report
    shutil.rmtree(aud / "gm_dense" / "stereo", ignore_errors=True)
    shutil.rmtree(aud / "gm_dense" / "images", ignore_errors=True)
    print(f"[{key}] VERDICT={rep['verdict']}  checks={rep['pass_checks']}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(VIDEOS), help="run one video")
    a = ap.parse_args()
    keys = [a.only] if a.only else list(VIDEOS)
    for k in keys:
        process(k)


if __name__ == "__main__":
    main()
