"""Fuse 32-V2 insertion descent (subglottis + upper trachea) into ONE clean model.
NO MVS. Borrowed 32_v1 calibration (PROVISIONAL). CLAHE + bezel mask, dense stride-1
frames 191-529, COLMAP SIFT + EXHAUSTIVE matching (the diag split shared frames
263-301 across 2 models, so matches cross the boundary -> exhaustive removes
matching as a variable). Mapper pinned (refine off). If still >1 model, try
model_merger on the shared frames + pinned BA. Reject fusion that only works by
creating outlier (flung) cameras.

Reports: reg/total, #models, largest size, per-model frame ranges, sparse, reproj,
camera-outlier ratio, trajectory plot. Run in depth-eval env."""
from __future__ import annotations
import json, sqlite3, shutil, subprocess, re
from pathlib import Path
import cv2, numpy as np, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/32-V2.MP4")
CALIB = json.loads(Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib/32_v1/intrinsics_pinned.json").read_text())
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2")
LO, HI, DARK_L = 191, 529, 12.0
PARAMS = ",".join(f"{p:.10g}" for p in CALIB["params_colmap"])


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4)
    return m > 0


def run(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  COLMAP ERR:", cmd[0], r.stderr[-400:])
    return r


def extract():
    img = OUT / "images"; msk = OUT / "masks"
    for d in (img, msk):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    mask = bezel(DATA); bez = mask.astype(np.uint8) * 255
    c = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(str(DATA)); fi = 0; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > HI: break
        if LO <= fi <= HI:
            L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255
            if L >= DARK_L:
                lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = c.apply(lab[:, :, 0])
                im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
                cv2.imwrite(str(img / f"f{fi:05d}.png"), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(str(msk / f"f{fi:05d}.png.png"), bez, [cv2.IMWRITE_PNG_COMPRESSION, 9]); n += 1
        fi += 1
    cap.release()
    return img, msk, n


def centers(rec):
    out = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); C = -M[:3, :3].T @ M[:3, 3]
        out[int(re.search(r"f(\d+)", im.name).group(1))] = C
    return out


def outlier_ratio(C):
    """fraction of cameras flung off the trajectory: dist-to-neighbor-median > 4*MAD,
    measured on frame-ordered consecutive steps + per-axis MAD gross check."""
    fis = sorted(C); P = np.array([C[f] for f in fis])
    if len(P) < 5: return 1.0, np.zeros(len(P), bool)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-9
    gross = np.any(np.abs(P - med) > 6 * mad * 1.4826, axis=1)         # flung far from cloud
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    smed = np.median(step); jump = np.zeros(len(P), bool)
    big = step > 8 * smed                                              # teleport between consecutive frames
    jump[1:] |= big; jump[:-1] |= big
    out = gross | jump
    return float(out.mean()), out


def evaluate(rec, tag):
    C = centers(rec); fis = sorted(C)
    errs = np.array([p.error for p in rec.points3D.values()]) if rec.num_points3D() else np.array([0.0])
    orat, omask = outlier_ratio(C)
    runs = []; s = fis[0]; p = fis[0]
    for f in fis[1:]:
        if f - p > 8: runs.append((s, p)); s = f
        p = f
    runs.append((s, p))
    return {"tag": tag, "reg": rec.num_reg_images(), "sparse": rec.num_points3D(),
            "reproj_px": round(float(errs.mean()), 3), "frame_min": fis[0], "frame_max": fis[-1],
            "runs": runs, "outlier_ratio": round(orat, 3), "C": C, "omask": omask, "fis": fis}


def trajectory_plot(ev, allpts, path):
    C = ev["C"]; fis = ev["fis"]; P = np.array([C[f] for f in fis]); om = ev["omask"]
    order = np.array(fis, float); order = (order - order.min()) / (np.ptp(order) + 1e-9)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (ix, iy, lx, ly) in zip(ax, [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        if allpts is not None and len(allpts):
            a.scatter(allpts[:, ix], allpts[:, iy], s=.3, c="0.8", alpha=.4, zorder=1)
        a.plot(P[:, ix], P[:, iy], "-", c="0.5", lw=.8, zorder=2)
        sc = a.scatter(P[:, ix], P[:, iy], c=order, cmap="viridis", s=18, zorder=3)
        if om.any():
            a.scatter(P[om, ix], P[om, iy], facecolors="none", edgecolors="red", s=90, lw=1.6, zorder=4, label="outlier cam")
        a.set_title(f"{lx}{ly}"); a.set_aspect("equal", "datalim"); a.grid(alpha=.3)
    if om.any(): ax[0].legend(fontsize=8)
    fig.suptitle(f"32-V2 fused [{ev['tag']}]  reg={ev['reg']}  sparse={ev['sparse']}  reproj={ev['reproj_px']}px  "
                 f"outlier_cam_ratio={ev['outlier_ratio']}  (calib=32_v1 BORROWED)\ncolor=frame order (subglottis->trachea)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def cloud(rec):
    return np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img, msk, n = extract()
    print(f"extracted {n} dense frames f{LO}-{HI} (stride1, CLAHE+bezel)\n")
    db = OUT / "db.db"
    if db.exists(): db.unlink()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(img), "--ImageReader.mask_path", str(msk),
         "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1", "--ImageReader.camera_params", PARAMS,
         "--SiftExtraction.max_image_size", "1600"])
    print("matching: EXHAUSTIVE (all pairs)")
    run(["exhaustive_matcher", "--database_path", str(db)])
    sp = OUT / "sparse"; shutil.rmtree(sp, ignore_errors=True); sp.mkdir()
    run(["mapper", "--database_path", str(db), "--image_path", str(img), "--output_path", str(sp),
         "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.init_min_tri_angle", "4", "--Mapper.min_num_matches", "8"])
    models = sorted([d for d in sp.iterdir() if d.is_dir()], key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
    recs = [pycolmap.Reconstruction(str(m)) for m in models]
    print(f"\nDIRECT MAPPER: {len(models)} model(s)")
    for m, r in zip(models, recs):
        ev = evaluate(r, f"direct/{m.name}")
        print(f"  {m.name}: reg={ev['reg']} sparse={ev['sparse']} reproj={ev['reproj_px']}px runs={ev['runs']} outlier={ev['outlier_ratio']}")

    result = {"n_frames": n, "calib": "32_v1 BORROWED/PROVISIONAL", "matching": "exhaustive"}
    if len(models) == 1:
        final = recs[0]; ftag = "direct-single"
        print("\n=> ONE MODEL from direct mapping.")
    else:
        # fuse the two largest on their shared frames
        print(f"\n>1 model -> model_merger on largest two ({models[0].name}+{models[1].name}) ...")
        mg = OUT / "merged"; shutil.rmtree(mg, ignore_errors=True); mg.mkdir()
        r = run(["model_merger", "--input_path1", str(models[0]), "--input_path2", str(models[1]),
                 "--output_path", str(mg), "--max_reproj_error", "8"])
        if not (mg / "images.bin").exists() and not (mg / "images.txt").exists():
            print("  merge produced no model; keeping largest direct model.")
            final = recs[0]; ftag = "direct-largest (merge failed)"
        else:
            # pinned bundle adjust to settle the seam
            ba = OUT / "merged_ba"; shutil.rmtree(ba, ignore_errors=True); ba.mkdir()
            run(["bundle_adjuster", "--input_path", str(mg), "--output_path", str(ba),
                 "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_extra_params", "0",
                 "--BundleAdjustment.refine_principal_point", "0"])
            final = pycolmap.Reconstruction(str(ba if (ba / "images.bin").exists() else mg)); ftag = "merged+BA"
            print(f"  merged: reg={final.num_reg_images()} sparse={final.num_points3D()}")

    ev = evaluate(final, ftag)
    trajectory_plot(ev, cloud(final), OUT / "trajectory.png")
    subg = sum(1 for f in ev["fis"] if f <= 305); trach = sum(1 for f in ev["fis"] if f > 305)
    verdict = "PASS" if (ev["outlier_ratio"] <= 0.05 and subg >= 10 and trach >= 10) else "REVIEW/FAIL"
    result.update({"final_tag": ftag, "reg": ev["reg"], "total_frames": n, "n_models_direct": len(models),
                   "largest_model_reg": recs[0].num_reg_images(), "sparse": ev["sparse"], "reproj_px": ev["reproj_px"],
                   "frame_range": [ev["frame_min"], ev["frame_max"]], "runs": ev["runs"],
                   "outlier_camera_ratio": ev["outlier_ratio"], "n_subglottis(<=305)": subg, "n_trachea(>305)": trach,
                   "verdict": verdict})
    result.pop("C", None)
    (OUT / "fuse_report.json").write_text(json.dumps(result, indent=2, default=str))
    print("\n================ 32-V2 FUSION REPORT ================")
    print(f" calibration         : 32_v1 BORROWED / PROVISIONAL")
    print(f" matching            : exhaustive")
    print(f" registered / total  : {ev['reg']} / {n}")
    print(f" # models (direct)   : {len(models)}   largest={recs[0].num_reg_images()}")
    print(f" final model         : {ftag}")
    print(f" frame range / runs  : {ev['frame_min']}..{ev['frame_max']}  runs={ev['runs']}")
    print(f" sparse points       : {ev['sparse']}")
    print(f" reproj error        : {ev['reproj_px']} px")
    print(f" outlier cam ratio   : {ev['outlier_ratio']}")
    print(f" subglottis / trachea: {subg} (<=305) / {trach} (>305) frames in final model")
    print(f" VERDICT             : {verdict}")
    print(f" trajectory plot     : {OUT/'trajectory.png'}")


if __name__ == "__main__":
    main()
