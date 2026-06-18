"""Generate requested artifacts for 7-V1 (phase3_7v1) and 32-V2 (robust/bridge40)."""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path
import numpy as np, cv2, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust/runs")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def load_cams(sfm):
    rec = pycolmap.Reconstruction(str(sfm))
    out = {}
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        out[fi] = -R.T @ t
    return rec, out


def traj(sfm, title, png, seg=None):
    rec, cc = load_cams(sfm)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    C = np.array(list(cc.values())); fis = list(cc)
    for axp, (ix, iy, lx, ly) in zip(ax, [(0, 2, "X", "Z"), (0, 1, "X", "Y")]):
        axp.scatter(C[:, ix], C[:, iy], c=[ (seg(f) if seg else 0) for f in fis] if seg else "tab:green",
                    s=30, edgecolors="k", linewidths=.3, cmap="viridis")
        axp.set_xlabel(lx); axp.set_ylabel(ly); axp.set_aspect("equal", "datalim"); axp.grid(alpha=.3); axp.set_title(f"{lx}{ly}")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)


def contact(img_dir, png, title):
    ps = sorted(img_dir.glob("f*.png"))
    cols = 10; rows = (len(ps) + cols - 1) // cols
    im0 = cv2.imread(str(ps[0])); tw = 200; th = int(round(tw * im0.shape[0] / im0.shape[1])); lh = 24
    canvas = np.full((rows * (th + lh), cols * tw, 3), 245, np.uint8); f = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(ps):
        r, c = divmod(i, cols); y0 = r * (th + lh); x0 = c * tw
        canvas[y0:y0 + th, x0:x0 + tw] = cv2.resize(cv2.imread(str(p)), (tw, th), interpolation=cv2.INTER_AREA)
        fi = re.search(r"f(\d+)", p.name).group(1)
        cv2.putText(canvas, fi, (x0 + 4, y0 + th + lh - 6), f, 0.55, (0, 0, 0), 3)
        cv2.putText(canvas, fi, (x0 + 4, y0 + th + lh - 6), f, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def dense_render(ply, cams, png, title):
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    if len(pts) > 80000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 80000, replace=False)]
    C = np.array(list(cams.values()))
    fig = plt.figure(figsize=(18, 6))
    for k, (ix, iy, lx, ly) in enumerate([(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(pts[:, ix], pts[:, iy], s=.3, c="lightgray", alpha=.5)
        ax.plot(C[:, ix], C[:, iy], "-", c="red", lw=.8)
        ax.scatter(C[:, ix], C[:, iy], s=10, c="red")
        ax.set_title(f"{lx}{ly}"); ax.set_aspect("equal", "datalim"); ax.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)


# ===================== 7-V1 =====================
def seg7(fi):
    return 1 if (130 <= fi <= 220 or 1846 <= fi <= 1992) else 0
P7 = ROOT / "phase3_7v1"
res7 = json.loads((P7 / "stage4_results.json").read_text())
intr7 = json.loads(ROOT / "retriage_first15/_calib/7_v1/intrinsics_pinned.json").read_text() if False else json.loads((ROOT / "retriage_first15/_calib/7_v1/intrinsics_pinned.json").read_text())
contact(P7 / "images", OUT / "7v1_selected_frames.png", "7-V1 selected recon frames")
rec7, cc7 = load_cams(P7 / "sfm")
traj(P7 / "sfm", f"7-V1 trajectory reg={len(cc7)}", OUT / "7v1_trajectory.png", seg=seg7)
dense_render(P7 / "dense/fused.ply", cc7, OUT / "7v1_dense_render.png", f"7-V1 MVS dense ({res7['n_dense']} pts) + cameras")
# model summary via model_converter
txt = OUT / "7v1_sfm_text"; txt.mkdir(exist_ok=True)
subprocess.run([COLMAP, "model_converter", "--input_path", str(P7 / "sfm"), "--output_path", str(txt), "--output_type", "TXT"], capture_output=True)
def count_txt(p):
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#"))
cam = next(iter(rec7.cameras.values())); cp = list(cam.params)
pin = [intr7["fx"], intr7["fy"], intr7["cx"], intr7["cy"], intr7["k1"], intr7["k2"], 0.0, 0.0]
art7 = {
    "selected_frames_png": str(OUT / "7v1_selected_frames.png"),
    "trajectory_png": str(OUT / "7v1_trajectory.png"),
    "dense_render_png": str(OUT / "7v1_dense_render.png"),
    "model_summary": {"n_cameras": rec7.num_cameras() if hasattr(rec7, "num_cameras") else len(rec7.cameras),
                      "cameras_txt_lines": count_txt(txt / "cameras.txt"),
                      "images_txt_registered": count_txt(txt / "images.txt") // 2,
                      "points3D_txt": count_txt(txt / "points3D.txt"),
                      "n_reg_images": rec7.num_reg_images(), "n_points3D": rec7.num_points3D()},
    "intrinsics_final_vs_pinned": {
        "camera_model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
        "final": [round(x, 6) for x in cp], "pinned": [round(x, 6) for x in pin],
        "max_abs_delta": float(max(abs(a - b) for a, b in zip(cp, pin)))},
    "mvs_stats": {"n_dense": res7["n_dense"], "reproj_px": res7["reproj_px"],
                  "n_registered": res7["n_reg"], "subcord_median_tri_deg": res7["subcord_median_tri_deg"],
                  "subcord_p95_tri_deg": res7["subcord_p95_tri_deg"], "subcord_lat_along": res7["subcord_lat_along"]},
}
(OUT / "7v1_artifacts.json").write_text(json.dumps(art7, indent=2))

# ===================== 32-V2 =====================
def comp_frames(models_dir):
    out = []
    if not models_dir.exists():
        return out
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            r = pycolmap.Reconstruction(str(d))
        except Exception:
            continue
        fis = sorted(int(re.search(r"f(\d+)", im.name).group(1)) for im in r.images.values())
        segc = lambda fi: ("sub_in" if 191 <= fi <= 305 else ("sub_out" if 2229 <= fi <= 2281 else "bridge"))
        out.append({"model": d.name, "n": len(fis), "frames": fis,
                    "n_sub_in": sum(1 for f in fis if segc(f) == "sub_in"),
                    "n_sub_out": sum(1 for f in fis if segc(f) == "sub_out"),
                    "n_bridge": sum(1 for f in fis if segc(f) == "bridge")})
    return out

robust = ROOT / "phase2_subcord/32-V2_robust"
bridge40 = ROOT / "phase2_subcord/32-V2_bridge40"
# outlier camera frame IDs in the loose bridge40 fused model
rec_b, cc_b = load_cams(bridge40 / "sfm")
Cb = np.array(list(cc_b.values())); fib = list(cc_b)
ctr = np.median(Cb, 0); db = np.linalg.norm(Cb - ctr, axis=1); medb = np.median(db)
outliers_b = sorted(int(fib[k]) for k in range(len(fib)) if db[k] > 10 * medb)
art32 = {
    "robust_traj_before_png": str(robust / "traj_before.png"),
    "robust_traj_after_png": str(robust / "traj_after.png"),
    "robust_report": json.loads((robust / "robust_report.json").read_text()) if (robust / "robust_report.json").exists() else None,
    "robust_component_frame_lists": comp_frames(robust / "sfm" / "models"),
    "bridge40_outlier_camera_frame_ids": outliers_b,
    "bridge40_max_dist_over_median": round(float(db.max() / max(medb, 1e-9)), 1),
    "calibration_assessment": {
        "has_real_32v2_calib": False,
        "used": "32_v1 Calibration Video (APPROX)",
        "is_wrong_calib_the_only_issue": "NO - two independent failure modes: (1) approx intrinsics likely "
            "degrade triangulation on the long low-texture bridge; (2) sparse 38/68-frame curation + forward-"
            "viewing means weak cross-pass linkage. Barbour-style DENSE passthrough (500-1200 frames) is the "
            "bigger lever. Real 32_v2 calibration would help but is not the sole fix.",
    },
}
(OUT / "32v2_artifacts.json").write_text(json.dumps(art32, indent=2))

print("=== 7-V1 ARTIFACTS ===")
print(json.dumps({k: v for k, v in art7.items() if k != "model_summary"}, indent=1)[:600])
print("model_summary:", art7["model_summary"])
print("intrinsics:", art7["intrinsics_final_vs_pinned"])
print("\n=== 32-V2 ARTIFACTS ===")
print("robust components:", [{k: c[k] for k in ("model", "n", "n_sub_in", "n_sub_out", "n_bridge")} for c in art32["robust_component_frame_lists"]])
print("bridge40 outlier camera frame IDs:", art32["bridge40_outlier_camera_frame_ids"])
print("bridge40 max/median dist:", art32["bridge40_max_dist_over_median"])
print(f"saved artifacts -> {OUT}")
