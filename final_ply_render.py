"""Presentation-ready renders from each video's best final airway .ply. Consistent style
across videos: black background, no axes, RGB point colors where available (else depth-turbo),
canonical PCA orientation (tube long-axis = X). Per video: A main / B side / C oblique /
D inner(end-on) clean views (no trajectory), plus a main-view-with-trajectory where cameras
exist, a 2x2 labelled montage, and entries for a report + all-video summary sheet.
NO reconstruction recompute. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path("/home/mi3dr/projects/bronchotrust"); OUTD = ROOT / "runs/final_ply_renders"
BG = "black"; rng = np.random.default_rng(0)
CFG = [
    dict(v="2-V2",  ply="runs/batch4/2-V2/dense0/fused.ply",   sparse="runs/batch4/2-V2/sparse/0",
         note="COLMAP dense MVS, long-window (f870-1200)"),
    dict(v="25-V1", ply="runs/batch4/25-V1/dense0/fused.ply",  sparse="runs/batch4/25-V1/sparse/0",
         note="COLMAP dense MVS, long-window (f315-620)"),
    dict(v="32-V2", ply="runs/fuse_32v2/dense0/fused.ply",     sparse="runs/fuse_32v2/sparse/0",
         note="COLMAP dense MVS, fused 339 frames; borrowed 32_v1 calib (provisional)"),
    dict(v="7-V1",  ply="runs/batch4/7-V1/double_poisson.ply", sparse=None, mesh=True,
         note="dense cloud was freed; rendered from double-Poisson MESH vertices (no RGB -> depth-colored)"),
    dict(v="16_v1", ply="runs/ct_validation_16v1/recon_connect/dense_m0/fused.ply",
         sparse="runs/ct_validation_16v1/recon_connect/sparse_strict/0",
         note="NOISY / low-parallax (best m0 fragment, 66 frames); shown for completeness"),
]
VIEWS = [("A_main", 18, -62), ("B_side", 6, -90), ("C_oblique", 70, -58), ("D_inner", 2, 2)]
TITLES = {"A_main": "Main View", "B_side": "Side View", "C_oblique": "Oblique View", "D_inner": "Inner View"}


def load(cfg):
    if cfg.get("mesh"):
        m = o3d.io.read_triangle_mesh(cfg["ply"]); P = np.asarray(m.vertices)
        C = np.asarray(m.vertex_colors)
    else:
        pc = o3d.io.read_point_cloud(cfg["ply"]); P = np.asarray(pc.points); C = np.asarray(pc.colors)
    fin = np.isfinite(P).all(1); P = P[fin]; C = C[fin] if len(C) == len(fin) else np.zeros((0, 3))
    keep = np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4; P = P[keep]; C = C[keep] if len(C) else C
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-9
    inl = np.all(np.abs(P - med) < 8 * mad, axis=1); P = P[inl]; C = C[inl] if len(C) else C
    # extra statistical clean for a tidy render
    pc2 = o3d.geometry.PointCloud(); pc2.points = o3d.utility.Vector3dVector(P)
    if len(C) == len(P) and len(C): pc2.colors = o3d.utility.Vector3dVector(C)
    pc2, ind = pc2.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
    P = np.asarray(pc2.points); C = np.asarray(pc2.colors)
    has_rgb = len(C) == len(P) and len(C) > 0 and C.max() > 0
    return P, (C if has_rgb else None)


def cam_centers(sparse):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse); rows = []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix())
        rows.append((int(re.search(r"f(\d+)", im.name).group(1)), -M[:3, :3].T @ M[:3, 3]))
    rows.sort(); return np.array([r[1] for r in rows])


def canon(P, cams=None):
    center = P.mean(0); sub = P[rng.choice(len(P), min(len(P), 60000), replace=False)] - center
    _, _, Vt = np.linalg.svd(sub, full_matrices=False); V = Vt.T
    if np.linalg.det(V) < 0: V[:, 2] = -V[:, 2]
    Pc = (P - center) @ V
    Cc = (cams - center) @ V if cams is not None and len(cams) else None
    return Pc, Cc


def lims_of(Pc):
    lo = np.percentile(Pc, 0.5, 0); hi = np.percentile(Pc, 99.5, 0); c = (lo + hi) / 2; h = (hi - lo) / 2 * 1.10
    return [(c[i] - h[i], c[i] + h[i]) for i in range(3)]


def render(Pc, col, el, az, lims, path, cams=None, s=1.1, n=180000):
    fig = plt.figure(figsize=(8, 8), facecolor=BG); ax = fig.add_subplot(111, projection="3d")
    idx = rng.choice(len(Pc), min(len(Pc), n), replace=False)
    if col is not None:
        ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c=np.clip(col[idx], 0, 1), s=s, alpha=.6, linewidths=0)
    else:
        ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c=Pc[idx, 0], cmap="turbo", s=s, alpha=.6, linewidths=0)
    if cams is not None:
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], c="cyan", lw=1.8)
    ax.set_facecolor(BG); ax.set_axis_off()
    ext = np.array([lims[i][1] - lims[i][0] for i in range(3)])
    try: ax.set_box_aspect(tuple(np.clip(ext / ext.max(), 0.32, 1.0)))
    except Exception: pass
    for i, sl in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)): sl(*lims[i])
    ax.view_init(el, az)
    fig.savefig(path, dpi=145, facecolor=BG, bbox_inches="tight", pad_inches=0.05); plt.close(fig)


def montage(v, vdir, note):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.6), facecolor=BG)
    for ax, (key, *_ ) in zip(axes.ravel(), VIEWS):
        im = mpimg.imread(vdir / f"{key}.png"); ax.imshow(im); ax.axis("off")
        ax.set_title(TITLES[key], color="white", fontsize=15, pad=6)
    fig.suptitle(v, color="white", fontsize=24, fontweight="bold", y=0.99)
    fig.text(0.5, 0.012, note, color="0.7", fontsize=9, ha="center")
    fig.subplots_adjust(top=0.94, bottom=0.04, wspace=0.02, hspace=0.08)
    fig.savefig(vdir / f"montage_{v}.png", dpi=120, facecolor=BG); plt.close(fig)


def main():
    OUTD.mkdir(parents=True, exist_ok=True); report = []
    for cfg in CFG:
        v = cfg["v"]; vdir = OUTD / v; vdir.mkdir(parents=True, exist_ok=True)
        if not Path(cfg["ply"]).exists():
            report.append({"video": v, "ply": cfg["ply"], "status": "MISSING"}); print(f"{v}: MISSING"); continue
        P, col = load(cfg)
        cams = None
        if cfg.get("sparse") and Path(cfg["sparse"]).exists():
            try: cams = cam_centers(cfg["sparse"])
            except Exception as e: print(f"  {v} cams fail {str(e)[:60]}")
        Pc, Cc = canon(P, cams); lims = lims_of(Pc)
        for key, el, az in VIEWS:
            render(Pc, col, el, az, lims, vdir / f"{key}.png")
        if Cc is not None:
            render(Pc, col, 18, -62, lims, vdir / "A_main_trajectory.png", cams=Cc)
        montage(v, vdir, cfg["note"])
        report.append({"video": v, "ply_used": cfg["ply"], "n_points": int(len(P)),
                       "colored": bool(col is not None), "trajectory": bool(Cc is not None),
                       "render_folder": str(vdir.relative_to(ROOT)), "notes": cfg["note"]})
        print(f"{v}: {len(P)} pts colored={col is not None} traj={Cc is not None} -> {vdir}", flush=True)
    # all-video summary sheet (main view per video)
    ok = [r for r in report if r.get("n_points")]
    ncol = min(3, len(ok)); nrow = (len(ok) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.4 * nrow), facecolor=BG)
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, ok):
        im = mpimg.imread(OUTD / r["video"] / "A_main.png"); ax.imshow(im); ax.axis("off")
        ax.set_title(f"{r['video']}  ({r['n_points']//1000}k pts)", color="white", fontsize=14)
    for ax in axes[len(ok):]: ax.axis("off")
    fig.suptitle("Airway reconstructions — all videos (COLMAP dense .ply)", color="white", fontsize=18, y=0.995)
    fig.subplots_adjust(top=0.93, wspace=0.02, hspace=0.1)
    fig.savefig(OUTD / "ALL_VIDEOS_summary.png", dpi=120, facecolor=BG); plt.close(fig)
    (OUTD / "render_report.json").write_text(json.dumps(report, indent=2))
    print("\nSUMMARY:")
    print(f"{'video':8}{'pts':>10}{'colored':>9}{'traj':>6}  ply")
    for r in report:
        if r.get("n_points"):
            print(f"{r['video']:8}{r['n_points']:>10}{str(r['colored']):>9}{str(r['trajectory']):>6}  {r['ply_used']}")
    print(f"\nsummary sheet -> {OUTD}/ALL_VIDEOS_summary.png ; report -> {OUTD}/render_report.json")


if __name__ == "__main__":
    main()
