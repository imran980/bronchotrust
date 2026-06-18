"""32-V2 VISUAL-ONLY 3D render package — eyeball the whole reconstructed airway as
one organ. COLMAP fused 339-frame dense cloud + medial-axis centerline. Calibration
32_v1 BORROWED/PROVISIONAL. No tables, no measurements. Outputs 4 clean images:
  A render_A_organ_cloud.png   (HERO: organ-only, color by axial position, regions)
  B render_B_centerline.png    (organ + medial centerline + glottis/+5mm/+10mm planes)
  C render_C_multiview.png     (4 views, same scale, shape inspection)
  D render_D_cloud_vs_surface.png (dense cloud | Poisson surface, surface labeled)
Also prints a per-axial-bin clean/noisy summary for the report. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, re
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geometry_debug_32v2 as G

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
GLOTTIS_FRAME = 297; K0 = 1.694; CMAP = "turbo"
PLANE_COLS = {"glottis": "#ff3df0", "+5mm": "#39ff14", "+10mm": "#ff9b1a"}
rng = np.random.default_rng(0)


def style3d(ax, black=True):
    ax.set_axis_off()
    if black:
        ax.set_facecolor("black")
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass


def frame_center(rec, frame):
    for im in rec.images.values():
        if int(re.search(r"f(\d+)", im.name).group(1)) == frame:
            M = np.array(im.cam_from_world().matrix()); return -M[:3, :3].T @ M[:3, 3]
    return None


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); Cc = G.cam_centers(rec); cam_centroid = Cc.mean(0)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    scum = G.arclen(pf); L = scum[-1]
    ctree = cKDTree(pf); dpt, ipt = ctree.query(P); s_pt = scum[ipt]; r_pt = dpt   # axial pos + radial dist

    # canonical PCA frame: PC1 = tube long axis, oriented proximal(glottis)->distal
    center = P.mean(0)
    sub = P[rng.choice(len(P), min(len(P), 60000), replace=False)] - center
    _, _, Vt = np.linalg.svd(sub, full_matrices=False); V = Vt.T
    if (pf[0] - center) @ V[:, 0] > (pf[-1] - center) @ V[:, 0]: V[:, 0] = -V[:, 0]
    if np.linalg.det(V) < 0: V[:, 2] = -V[:, 2]
    rot = lambda X: (X - center) @ V
    rotd = lambda D: D @ V
    Pc = rot(P); pfc = rot(pf)
    span = np.percentile(Pc, 99, 0) - np.percentile(Pc, 1, 0)
    cen = (np.percentile(Pc, 99, 0) + np.percentile(Pc, 1, 0)) / 2
    half = 0.62 * max(span)
    lims = [(cen[i] - half, cen[i] + half) for i in range(3)]

    def draw_cloud(ax, n=160000, s=1.2, a=0.55):
        idx = rng.choice(len(Pc), min(len(Pc), n), replace=False)
        ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c=s_pt[idx] / L, cmap=CMAP, s=s, alpha=a, linewidths=0)
        for i, lim in enumerate(lims):
            (ax.set_xlim, ax.set_ylim, ax.set_zlim)[i](*lim)

    regions = [("GLOTTIS", 0.015), ("SUBGLOTTIS", 0.34), ("TRACHEA", 0.62), ("DEEP TRACHEA", 0.94)]
    def label_regions(ax, dz=1.0):
        for name, fr in regions:
            p = np.array(splev(np.interp(fr * L, scum, uf), tck)).T
            pcl = rot(p)
            ax.text(pcl[0], pcl[1], pcl[2] + dz * half * 0.5, name, color="white", fontsize=11,
                    ha="center", fontweight="bold")

    # ---------------- A: HERO organ cloud ----------------
    fig = plt.figure(figsize=(13, 11), facecolor="black"); ax = fig.add_subplot(111, projection="3d")
    draw_cloud(ax, n=190000, s=1.3, a=0.6); style3d(ax)
    ax.view_init(elev=18, azim=-62); label_regions(ax)
    ax.set_title("32-V2 reconstructed airway — dense cloud, colored proximal→distal\n"
                 "(COLMAP, 339/339 frames · calibration 32_v1 BORROWED/PROVISIONAL)",
                 color="white", fontsize=13, pad=2)
    fig.savefig(DIAG / "render_A_organ_cloud.png", dpi=140, facecolor="black", bbox_inches="tight"); plt.close(fig)

    # ---------------- B: organ + centerline + landmark planes ----------------
    cg = frame_center(rec, GLOTTIS_FRAME); s_g = scum[int(np.argmin(np.linalg.norm(pf - cg, axis=1)))]
    fig = plt.figure(figsize=(13, 11), facecolor="black"); ax = fig.add_subplot(111, projection="3d")
    idx = rng.choice(len(Pc), min(len(Pc), 110000), replace=False)
    ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c="0.55", s=1.0, alpha=0.18, linewidths=0)
    ax.plot(pfc[:, 0], pfc[:, 1], pfc[:, 2], c="cyan", lw=2.6)
    for lim, sl in zip(lims, (ax.set_xlim, ax.set_ylim, ax.set_zlim)): sl(*lim)
    for key, mm in [("glottis", 0.0), ("+5mm", 5.0), ("+10mm", 10.0)]:
        s = min(max(s_g + mm / K0, scum[1]), scum[-2])
        u = np.interp(s, scum, uf); p0 = np.array(splev(u, tck)).T
        t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
        e1, e2 = G.frame(t); p0c = rot(p0); e1c, e2c = rotd(e1), rotd(e2); hh = 1.8 * Rmed
        q = np.array([p0c + a*hh*e1c + b*hh*e2c for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        ax.add_collection3d(Poly3DCollection([q], alpha=.5, facecolor=PLANE_COLS[key], edgecolor=PLANE_COLS[key]))
        ax.text(p0c[0], p0c[1], p0c[2] + hh * 1.5, key, color=PLANE_COLS[key], fontsize=11, fontweight="bold")
    style3d(ax); ax.view_init(elev=18, azim=-62)
    ax.set_title("32-V2 airway + medial-axis centerline (cyan) + Barbour landmark planes\n"
                 "glottis(magenta) / +5mm(green) / +10mm(orange)  — 5/10mm placement PROVISIONAL",
                 color="white", fontsize=12, pad=2)
    fig.savefig(DIAG / "render_B_centerline.png", dpi=140, facecolor="black", bbox_inches="tight"); plt.close(fig)

    # ---------------- C: multi-view (same scale) ----------------
    views = [("side (length)", 8, -90), ("top (length)", 89, -90),
             ("end-on (down lumen)", 4, 0), ("oblique", 20, -58)]
    fig = plt.figure(figsize=(15, 14), facecolor="black")
    for i, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d"); draw_cloud(ax, n=80000, s=1.0, a=0.5)
        style3d(ax); ax.view_init(elev=el, azim=az)
        ax.set_title(name, color="white", fontsize=12)
    fig.suptitle("32-V2 reconstructed airway — 4 views, same scale (shape inspection only)", color="white", fontsize=14)
    fig.savefig(DIAG / "render_C_multiview.png", dpi=135, facecolor="black", bbox_inches="tight"); plt.close(fig)

    # ---------------- D: dense cloud vs Poisson surface ----------------
    Vm = np.asarray(o3d.io.read_triangle_mesh(str(DIAG / "double_poisson.ply")).vertices)
    Vm = Vm[np.isfinite(Vm).all(1)]; Vmc = rot(Vm)
    dV, iV = ctree.query(Vm); sV = scum[iV]
    fig = plt.figure(figsize=(17, 9), facecolor="black")
    ax1 = fig.add_subplot(121, projection="3d"); draw_cloud(ax1, n=150000, s=1.1, a=0.55)
    style3d(ax1); ax1.view_init(elev=18, azim=-62)
    ax1.set_title("DENSE CLOUD  (used for measurement)", color="white", fontsize=12)
    ax2 = fig.add_subplot(122, projection="3d")
    mi = rng.choice(len(Vmc), min(len(Vmc), 150000), replace=False)
    ax2.scatter(Vmc[mi, 0], Vmc[mi, 1], Vmc[mi, 2], c=sV[mi] / L, cmap=CMAP, s=1.1, alpha=0.5, linewidths=0)
    for lim, sl in zip(lims, (ax2.set_xlim, ax2.set_ylim, ax2.set_zlim)): sl(*lim)
    style3d(ax2); ax2.view_init(elev=18, azim=-62)
    ax2.set_title("POISSON SURFACE — view only / NOT used for measurement\n(over-extended / blobby)",
                  color="#ff6666", fontsize=12)
    fig.suptitle("32-V2: dense cloud vs Poisson surface (same scale & coloring)", color="white", fontsize=14)
    fig.savefig(DIAG / "render_D_cloud_vs_surface.png", dpi=135, facecolor="black", bbox_inches="tight"); plt.close(fig)

    # ---------------- report stats: per-axial-bin density + ring tightness ----------------
    nb = 10; edges = np.linspace(0, L, nb + 1); rep = []
    for i in range(nb):
        m = (s_pt >= edges[i]) & (s_pt < edges[i + 1])
        if m.sum() > 50:
            rr = r_pt[m]; rep.append((round(edges[i], 1), round(edges[i+1], 1), int(m.sum()),
                                      round(float(np.std(rr) / max(np.median(rr), 1e-6)), 2)))
        else:
            rep.append((round(edges[i], 1), round(edges[i+1], 1), int(m.sum()), None))
    (DIAG / "render_report.json").write_text(json.dumps(
        {"L_scene": round(float(L), 2), "axial_bins_[s0,s1,npts,rstd/rmed]": rep,
         "regions_s": {"glottis": 0.0, "subglottis": round(0.34*L, 1), "trachea": round(0.62*L, 1),
                       "deep_trachea": round(0.94*L, 1)}}, indent=2))
    print(f"L={L:.2f} scene units; cloud {len(P)} pts; Rmed={Rmed:.3f}")
    print("axial bin  [s0..s1]   n_pts   ring r_std/r_med (lower=cleaner)")
    for s0, s1, n, ratio in rep:
        print(f"  {s0:5.1f}..{s1:5.1f}   {n:>8}   {ratio}")
    print("\nsaved render_A_organ_cloud.png (HERO), render_B_centerline.png, "
          "render_C_multiview.png, render_D_cloud_vs_surface.png + render_report.json")


if __name__ == "__main__":
    main()
