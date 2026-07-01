"""Smart local-batch COLMAP measurement pipeline (Barbour-style), with adaptive
fragmentation fallback. For each video x landmark:
  1. smart 30-frame selection (CLAHE; sharpness/glare/contrast gating; coverage-diversity
     farthest-point sampling) from a local pool around the landmark.
  2. COLMAP: pinned intrinsics, no refine, exhaustive matching inside the 30 frames.
  3. dense MVS -> ring cross-section CSA/DCE measured 3 ways (polar / ellipse / mesh).
  ADAPTIVE: start +/-30 pool; if the model FRAGMENTS or the ring is OPEN, also try +/-20
  (more frame overlap -> matchability/coverage) and +/-45 (more parallax). Prioritize ring
  COVERAGE + sharpness over parallax (wider-but-still-matchable wins). ACCEPT only:
  1 connected component AND closed ring (cov>=60% & r_std/r_med<=0.35) AND estimator
  spread <=1.3x. Pick the best pool per landmark.
Figures per accepted landmark: selected-frame contact sheet, organ cloud, centerline/slice,
landmark ring, reprojection-vs-real check. Final report table + verdict.
depth-eval env (colmap-cuda binary). Primary videos: 2-V2, 25-V1, 32-V2."""
from __future__ import annotations
import argparse, json, shutil, subprocess, re
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
import barbour30_smart as S
import barbour30_dense as D

OUT = S.OUT; COLMAP = S.COLMAP; DATA = S.DATA
POOLS = [30, 20, 45]               # try order: sweet spot, tighter (overlap/coverage), wider (parallax)
COV_MIN, RATIO_MAX, SPREAD_MAX = 0.60, 0.35, 1.30
VIDEOS = ["2-V2", "25-V1", "32-V2"]
LMS = ["glottis", "prox_subglottis", "dist_subglottis", "trachea_ref"]


def run_mvs_fast(bdir, max_size=1000):
    fused = bdir / "dense0/fused.ply"
    if fused.exists() and D.ply_n(fused) > 0: return fused
    dense = bdir / "dense0"; shutil.rmtree(dense, ignore_errors=True)
    for cmd in (
        ["image_undistorter", "--image_path", str(bdir / "images"), "--input_path", str(bdir / "sparse/0"),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"],
        ["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", str(max_size)],
        ["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)]):
        r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
        if r.returncode != 0: print("   MVS ERR", cmd[0], r.stderr[-180:], flush=True)
    return fused


def reproject_check(bdir, vk, landmark, center):
    """Synthesize the landmark view from the colored cloud (OPENCV model, z-buffer splat)
    vs the real CLAHE frame, for the camera nearest the landmark center."""
    try:
        rec = pycolmap.Reconstruction(str(bdir / "sparse/0")); cam = list(rec.cameras.values())[0]
        by = {int(re.search(r"f(\d+)", im.name).group(1)): im for im in rec.images.values()}
        f = min(by, key=lambda x: abs(x - center)); im = by[f]
        pc = o3d.io.read_point_cloud(str(bdir / "dense0/fused.ply"))
        Pw = np.asarray(pc.points); cols = np.asarray(pc.colors); ok = np.isfinite(Pw).all(1)
        Pw, cols = Pw[ok], cols[ok]
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]; t = M[:3, 3]
        fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
        Pc = (R @ Pw.T).T + t; front = Pc[:, 2] > 1e-3
        uv = cv2.projectPoints(Pw[front], cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
        z = Pc[front, 2]; c = (cols[front] * 255).astype(np.uint8)
        H, W = cam.height, cam.width
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H); u, v, z, c = u[inb], v[inb], z[inb], c[inb]
        o = np.argsort(-z); u, v, c = u[o], v[o], c[o]
        syn = np.zeros((H, W, 3), np.uint8)
        for du in (-2, -1, 0, 1, 2):
            for dv in (-2, -1, 0, 1, 2):
                syn[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = c
        # real frame
        cap = cv2.VideoCapture(str(DATA / S.LANDMARKS[vk]["video"])); cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        okr, fr = cap.read(); cap.release()
        mask = S.bezel(DATA / S.LANDMARKS[vk]["video"]); clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        real = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); real[~mask] = 0
        rm = cv2.resize(mask.astype(np.uint8), (W, H)) > 0; syn[~rm] = 0
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        ax[0].imshow(cv2.cvtColor(syn, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"f{f}: synthesized from cloud"); ax[0].axis("off")
        ax[1].imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB)); ax[1].set_title(f"f{f}: real (CLAHE)"); ax[1].axis("off")
        fig.suptitle(f"{vk} {landmark} — reprojection check", fontsize=12)
        fig.tight_layout(); fig.savefig(bdir / "reproject_check.png", dpi=110); plt.close(fig)
        return True
    except Exception as e:
        print("   reproject fail", str(e)[:90]); return False


def sparse_pool(vk, lm, pool):
    info, _ = S.select(vk, lm, pool)
    name = f"{vk}_{lm}_pipe_p{pool}"
    res = S.recon(vk, lm, info, name=name)
    res["pool"] = pool; res["frames"] = info["frames"]; res["selection"] = info
    return name, res


def measured_clean(m):
    return bool(m and m.get("measurable") and m.get("closed") and (m.get("DCE_estimator_spread") or 9) <= SPREAD_MAX)


def process_landmark(vk, lm):
    center = S.LANDMARKS[vk]["centers"][lm]
    print(f"\n=== {vk} / {lm} (center f{center}) ===", flush=True)
    attempts = {}      # pool -> {"name","sparse","measure"}
    for pool in POOLS:
        name, res = sparse_pool(vk, lm, pool)
        attempts[pool] = {"name": name, "sparse": res, "measure": None}
        matchable = res["n_models"] == 1 and res.get("reg", 0) >= 25 and res.get("reproj_px", 9) < 3.0
        print(f"   pool±{pool:<2} reg={res.get('reg',0)}/30 models={res['n_models']} cone={res.get('cone_deg','-')} "
              f"par={res.get('parallax_lat_along','-')} {'matchable' if matchable else 'FRAGMENTS/weak'}", flush=True)
        if matchable:
            bdir = OUT / "batches" / name
            run_mvs_fast(bdir)
            try:
                m = D.measure(name, center, do_mesh=True)
            except Exception as e:
                m = {"measurable": False, "error": str(e)[:150]}; print("   measure fail", str(e)[:90])
            attempts[pool]["measure"] = m
            shutil.rmtree(bdir / "dense0/stereo", ignore_errors=True); shutil.rmtree(bdir / "dense0/images", ignore_errors=True)
            print(f"            -> measurable={m.get('measurable')} cov={m.get('coverage','-')} "
                  f"ratio={m.get('r_std/r_med','-')} closed={m.get('closed','-')} spread={m.get('DCE_estimator_spread','-')}", flush=True)
            if measured_clean(m):  # accept first clean (pool order = 30,20,45)
                break
    # choose best attempt: clean first, then highest coverage, then lowest spread
    def score(a):
        m = a["measure"]
        if not (m and m.get("measurable")): return (-1, 0, 0)
        return (1 if measured_clean(m) else 0, m.get("coverage", 0), -(m.get("DCE_estimator_spread") or 9))
    best_pool = max(attempts, key=lambda p: score(attempts[p]))
    best = attempts[best_pool]; m = best["measure"]; sp = best["sparse"]
    # if nothing measurable, densify the largest-reg sparse for a figure
    if not (m and m.get("measurable")):
        cand = max(attempts, key=lambda p: attempts[p]["sparse"].get("reg", 0))
        best_pool = cand; best = attempts[cand]; sp = best["sparse"]
        bdir = OUT / "batches" / best["name"]
        if sp["n_models"] >= 1 and sp.get("reg", 0) >= 8:
            run_mvs_fast(bdir)
            try: m = D.measure(best["name"], center, do_mesh=True)
            except Exception: m = {"measurable": False}
            best["measure"] = m
            shutil.rmtree(bdir / "dense0/stereo", ignore_errors=True); shutil.rmtree(bdir / "dense0/images", ignore_errors=True)
    # verdict
    if sp["n_models"] != 1 or sp.get("reg", 0) < 25:
        verdict, why = "REJECT", "fragmented/under-registered"
    elif not (m and m.get("measurable")):
        verdict, why = "REJECT", "no measurable ring"
    elif (m.get("coverage", 0) < COV_MIN) or (m.get("r_std/r_med", 9) > RATIO_MAX):
        verdict, why = "REJECT", "open/low-coverage ring"
    elif (m.get("DCE_estimator_spread") or 9) > SPREAD_MAX:
        verdict, why = "REJECT", "estimator disagreement >1.3x"
    else:
        verdict, why = "ACCEPT", "Barbour-ready ring"
    # figures + contact sheet for the chosen pool
    bdir = OUT / "batches" / best["name"]
    S.montage(vk, lm, best["frames"] if "frames" in best else sp["frames"], f"pipe_{vk}_{lm}")
    if m and m.get("measurable"):
        reproject_check(bdir, vk, lm, center)
    dce = (m or {}).get("DCE_scene", {})
    frames = sp["frames"]
    row = {"video": vk, "landmark": lm, "chosen_pool": best_pool,
           "selected_frames": f"{frames[0]}-{frames[-1]} (n{len(frames)})", "frames_list": frames,
           "reg": f"{sp.get('reg',0)}/30", "components": sp["n_models"], "reproj": sp.get("reproj_px"),
           "cone_deg": sp.get("cone_deg"), "parallax": sp.get("parallax_lat_along"),
           "coverage": (m or {}).get("coverage"), "r_std/r_med": (m or {}).get("r_std/r_med"),
           "DCE_polar": dce.get("polar"), "DCE_ellipse": dce.get("ellipse"), "DCE_mesh": dce.get("mesh"),
           "spread": (m or {}).get("DCE_estimator_spread"), "closed": (m or {}).get("closed"),
           "verdict": verdict, "why": why, "dense_pts": (m or {}).get("n_dense")}
    (bdir / "pipeline_row.json").write_text(json.dumps(row, indent=2, default=str))
    print(f"   => chosen ±{best_pool}: {verdict} ({why})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(VIDEOS)); ap.add_argument("--landmarks", default=",".join(LMS))
    a = ap.parse_args()
    rows = []
    for vk in a.videos.split(","):
        for lm in a.landmarks.split(","):
            try:
                rows.append(process_landmark(vk, lm))
            except Exception as e:
                print(f"   {vk}/{lm} PIPELINE FAIL {str(e)[:150]}")
                rows.append({"video": vk, "landmark": lm, "verdict": "ERROR", "why": str(e)[:150]})
    (OUT / "pipeline_report.json").write_text(json.dumps(rows, indent=2, default=str))
    cols = ["video", "landmark", "selected_frames", "reg", "components", "reproj", "cone_deg", "coverage",
            "r_std/r_med", "DCE_polar", "DCE_ellipse", "DCE_mesh", "spread", "closed", "verdict"]
    print("\n\n================ BARBOUR LOCAL-BATCH PIPELINE — REPORT ================")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c)) for c in cols))
    acc = sum(1 for r in rows if r.get("verdict") == "ACCEPT")
    print(f"\nACCEPTED (Barbour-ready): {acc}/{len(rows)}")
    print(f"report -> {OUT}/pipeline_report.json")


if __name__ == "__main__":
    main()
