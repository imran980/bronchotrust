"""Real-video test of the multi-view contour-consistency throat fit (validated on phantoms in
phantom_contourfit.py). Uses the OCCLUSION-EDGE detector (brightness thresholding was disqualified
on the stenosis phantom). COLMAP poses + the exact extracted frames on disk (runs/batch4/<v>/images
by name -> NO POS_FRAMES drift). Scene units only (monocular gauge; no mm, no GT here).

This is a SANITY check vs the COLMAP dense-cloud slice at the same plane -- NOT validation. Report
per subglottic region: contour-fit throat DCE (scene), COLMAP slice DCE, their agreement, the
cross-view consistency spread (self-diagnostic), reprojection residual, and stratified subset
stability. High spread / disagreement = the method refusing, which is the honest outcome.
depth-eval env.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, cv2, pycolmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phantom_contourfit as cf   # reuse detect(edge)/rays_world/fit_circle/basis_from_traj/reproj_px

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit/realvideo_contourfit"

CONFIG = [
    dict(label="2_V2 prox-subglottis",  model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  imgs="runs/batch4/2-V2/images",  rng=(875, 945)),
    dict(label="25_V1 prox-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", imgs="runs/batch4/25-V1/images", rng=(335, 388)),
    dict(label="25_V1 dist-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", imgs="runs/batch4/25-V1/images", rng=(405, 452)),
]


def load_model(model):
    rec = pycolmap.Reconstruction(model); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    import re
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ M[:3, 3])
    return K, dist, poses


def run_region(cfg):
    K, dist, poses = load_model(cfg["model"]); img_dir = ROOT / cfg["imgs"]
    frames = [f for f in range(cfg["rng"][0], cfg["rng"][1] + 1) if f in poses]
    g = cf.gather(frames, poses, K, dist, "edge", img_dir, None, max_frames=200)
    if g is None or len(g["frames"]) < 6:
        return dict(label=cfg["label"], status="FAIL", reason=f"edge detected in <6 frames"), None
    used = [fm["frame"] for fm in g["frames"]]
    a0, n, e1, e2 = cf.basis_from_traj(poses, used)
    fit = cf.fit_circle(g["C"], g["U"], g["W"], a0, n, e1, e2, True)
    rp = cf.reproj_px(fit, g["frames"], K)
    # COLMAP dense slice at the fitted throat plane (same-plane sanity)
    col = colmap_slice(ROOT / cfg["cloud"], fit, n, a0)
    # stratified spread-gated stability
    order = np.array(used); subs, rej = [], 0
    for k in range(4):
        fl = list(order[np.arange(len(order)) % 4 == k])
        gs = cf.gather(fl, poses, K, dist, "edge", img_dir, None, max_frames=999)
        if gs is None or len(gs["frames"]) < 4: continue
        fb = cf.basis_from_traj(poses, [fm["frame"] for fm in gs["frames"]])
        fs = cf.fit_circle(gs["C"], gs["U"], gs["W"], *fb, True)
        if fs["spread"] <= 0.12: subs.append(2 * fs["R"])
        else: rej += 1
    stab = round(100 * (max(subs) - min(subs)) / max(np.median(subs), 1e-9), 1) if len(subs) >= 2 else None
    agree = round(100 * abs(2 * fit["R"] - col) / max(col, 1e-9), 1) if col else None
    r = dict(label=cfg["label"], status="OK", n_frames=len(g["frames"]),
             fit_DCE_scene=round(2 * fit["R"], 4), COLMAP_slice_DCE_scene=(round(col, 4) if col else None),
             agree_pct=agree, consistency_spread=round(fit["spread"], 4),
             reproj_px=(round(rp, 2) if rp else None), stab_pct=stab, n_subsets_used=len(subs), n_rejected=rej,
             z0=round(fit["z0"], 4))
    # honest sanity gate: converged (low spread) AND agrees with COLMAP AND stable
    r["sanity_ok"] = bool(fit["spread"] <= 0.10 and agree is not None and agree <= 25 and stab is not None and stab <= 25)
    return r, dict(g=g, fit=fit, K=K, n=n, a0=a0, img_dir=img_dir)


def colmap_slice(cloud, fit, n, a0):
    import open3d as o3d
    if not cloud.exists(): return None
    P = np.asarray(o3d.io.read_point_cloud(str(cloud)).points); P = P[np.isfinite(P).all(1)]
    if len(P) < 50: return None
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    slab = P[np.abs((P - a0) @ n - fit["z0"]) <= 0.5 * fit["R"]]
    if len(slab) < 30: return None
    rr = np.linalg.norm((slab - a0) - ((slab - a0) @ n)[:, None] * n, axis=1)
    return float(2 * np.median(rr))


def figure(rows, extras):
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [(r, e) for r, e in zip(rows, extras) if r["status"] == "OK" and e]
    if not ok: return
    fig, axes = plt.subplots(2, len(ok), figsize=(5 * len(ok), 8), squeeze=False)
    for j, (r, e) in enumerate(ok):
        g, fit, K = e["g"], e["fit"], e["K"]
        fm = g["frames"][len(g["frames"]) // 2]
        bgr = cv2.imread(str(e["img_dir"] / f"f{fm['frame']:05d}.png"))
        und = cv2.undistort(bgr, K, np.zeros(4))
        ax = axes[0][j]; ax.imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
        ax.plot(fm["pts"][:, 0], fm["pts"][:, 1], ".", ms=2.5, c="lime")
        q = cf.circle3d(fit); Xc = (fm["R"] @ (q - fm["C"]).T).T; fwd = Xc[:, 2] > 1e-6
        uv = (K @ Xc[fwd].T).T; uv = uv[:, :2] / uv[:, 2:3]
        ax.plot(uv[:, 0], uv[:, 1], "-", lw=1.6, c="red")
        ax.set_title(f"{r['label']}\nf{fm['frame']} green=edge red=fit", fontsize=9); ax.axis("off")
        ax = axes[1][j]
        nC = (g["C"] - fit["a0"]) @ fit["n"]; nU = g["U"] @ fit["n"]; m = np.abs(nU) > 0.05
        d = nC[m]; ext = d.max() - d.min() + 1e-9; zs = np.linspace(d.max() + 0.02 * ext, d.max() + 8 * ext, 200); sp = []
        for z0 in zs:
            s = (z0 - nC[m]) / nU[m]; mm = s > 1e-6
            X = g["C"][m][mm] + s[mm, None] * g["U"][m][mm]
            rad = np.linalg.norm((X - fit["a0"]) - z0 * fit["n"], axis=1); Rh = np.median(rad)
            sp.append(np.median(np.abs(rad - Rh)) / max(Rh, 1e-9))
        ax.plot(zs, sp); ax.axvline(fit["z0"], c="r", ls="--")
        ax.set_title(f"consistency  DCE={r['fit_DCE_scene']} (scene)\nCOLMAP={r['COLMAP_slice_DCE_scene']} agree {r['agree_pct']}% stab±{r['stab_pct']}%", fontsize=8)
        ax.set_xlabel("z0"); ax.set_ylabel("spread"); ax.grid(alpha=.3)
    fig.suptitle("Real-video contour-consistency throat fit (SANITY vs COLMAP; scene units, no mm)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "realvideo.png", dpi=120); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, extras = [], []
    for c in CONFIG:
        r, e = run_region(c); rows.append(r); extras.append(e)
        if r["status"] != "OK":
            print(f"[{r['label']}] FAIL: {r['reason']}", flush=True)
        else:
            print(f"[{r['label']}] fit_DCE={r['fit_DCE_scene']} COLMAP={r['COLMAP_slice_DCE_scene']} "
                  f"agree={r['agree_pct']}% spread={r['consistency_spread']} reproj={r['reproj_px']}px "
                  f"stab±{r['stab_pct']}% sanity_ok={r['sanity_ok']}", flush=True)
    figure(rows, extras)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n=== REAL-VIDEO SANITY (contour-consistency vs COLMAP; NOT validation) ===")
    print(f"{'region':26}{'fitDCE':>8}{'COLMAP':>8}{'agree%':>8}{'spread':>8}{'stab%':>7}  sanity")
    for r in rows:
        if r["status"] == "OK":
            print(f"{r['label']:26}{r['fit_DCE_scene']:>8}{str(r['COLMAP_slice_DCE_scene']):>8}{str(r['agree_pct']):>8}{r['consistency_spread']:>8}{str(r['stab_pct']):>7}  {r['sanity_ok']}")
        else:
            print(f"{r['label']:26}  FAIL: {r['reason']}")
    print(f"figure -> {OUT}/realvideo.png")


if __name__ == "__main__":
    main()
