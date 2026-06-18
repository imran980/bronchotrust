"""Generalized Barbour render+landmark package for ANY 32-V2-style recon.
Inputs: a dense cloud (fused.ply) + COLMAP sparse model + glottis frame.
Produces (into --out): double-Poisson mesh, medial-axis centerline, glottis/+5mm/
+10mm landmark slices, and the 3 figures A (organ cloud), B (centerline+planes),
C (multiview), plus a landmark-rings figure + report.json. Calibration provisional
where noted. No Myer-Cotton %, no final mm. depth-eval env.

Usage: batch_render.py --dense <fused.ply> --sparse <model_dir> --glottis <frame>
                       --label "<name>" --calibnote "<note>" --out <dir>"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, re
if not hasattr(np, "in1d"): np.in1d = np.isin
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geometry_debug_32v2 as G

CMAP = "turbo"; R_PHYS_MM = 2.0
PLANE = {"glottis": "#ff3df0", "+5mm": "#39ff14", "+10mm": "#ff9b1a"}
COV_MIN, RATIO_MAX = 0.60, 0.35
rng = np.random.default_rng(0)


def frame_center(rec, frame):
    best = None; bd = 1e18
    for im in rec.images.values():
        f = int(re.search(r"f(\d+)", im.name).group(1))
        M = np.array(im.cam_from_world().matrix()); c = -M[:3, :3].T @ M[:3, 3]
        if abs(f - frame) < bd: bd = abs(f - frame); best = c
    return best


def section(P, tree, p0, t, Rmed, slab=0.4, ball=6):
    rel = P[tree.query_ball_point(p0, ball * Rmed)] - p0
    ins = rel[np.abs(rel @ t) <= slab]
    e1, e2 = G.frame(t)
    ip = np.stack([ins @ e1, ins @ e2], 1) if len(ins) else np.zeros((0, 2))
    if len(ip):
        th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
        cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
        return ip, cov, float(np.median(r)), float(np.std(r)), e1, e2
    return ip, 0.0, 0.0, 0.0, e1, e2


def style3d(ax):
    ax.set_axis_off(); ax.set_facecolor("black")
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass


def double_poisson(P, out):
    if (out / "double_poisson.ply").exists():
        m = o3d.io.read_triangle_mesh(str(out / "double_poisson.ply"))
        return np.asarray(m.vertices)
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(k=20)
    m1, d1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    d1 = np.asarray(d1); m1.remove_vertices_by_index(np.where(d1 < np.quantile(d1, 0.05))[0].tolist())
    m1 = m1.filter_smooth_taubin(number_of_iterations=20); m1.compute_vertex_normals()
    samp = m1.sample_points_poisson_disk(number_of_points=120000)
    m2, d2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc + samp, depth=9)
    d2 = np.asarray(d2); m2.remove_vertices_by_index(np.where(d2 < np.quantile(d2, 0.03))[0].tolist())
    m2 = m2.filter_smooth_taubin(number_of_iterations=15)
    m2.remove_degenerate_triangles(); m2.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out / "double_poisson.ply"), m2)
    return np.asarray(m2.vertices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True); ap.add_argument("--sparse", required=True)
    ap.add_argument("--glottis", type=int, required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--calibnote", default="provisional"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    P = np.asarray(o3d.io.read_point_cloud(a.dense).points); P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(a.sparse); Cc = G.cam_centers(rec); cam_centroid = Cc.mean(0)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    scum = G.arclen(pf); L = scum[-1]; tree = cKDTree(P)
    ctree = cKDTree(pf); s_pt = scum[ctree.query(P)[1]]
    cg = frame_center(rec, a.glottis); s_g = scum[int(np.argmin(np.linalg.norm(pf - cg, axis=1)))]
    k0 = R_PHYS_MM / Rmed

    # landmarks
    def lm(s):
        s = min(max(s, scum[1]), scum[-2]); u = np.interp(s, scum, uf); p0 = np.array(splev(u, tck)).T
        t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
        if (p0 - cam_centroid) @ t < 0: t = -t
        ip, cov, rm, rs, e1, e2 = section(P, tree, p0, t, Rmed)
        return dict(s=float(s), p0=p0, t=t, e1=e1, e2=e2, ip=ip, cov=cov, r_med=rm,
                    ratio=rs / max(rm, 1e-6), closed=bool(cov >= COV_MIN and rs / max(rm, 1e-6) <= RATIO_MAX))
    cur = {"glottis": lm(s_g), "+5mm": lm(s_g + 5.0 / k0), "+10mm": lm(s_g + 10.0 / k0)}

    # canonical PCA frame
    center = P.mean(0); sub = P[rng.choice(len(P), min(len(P), 60000), replace=False)] - center
    _, _, Vt = np.linalg.svd(sub, full_matrices=False); V = Vt.T
    if (pf[0] - center) @ V[:, 0] > (pf[-1] - center) @ V[:, 0]: V[:, 0] = -V[:, 0]
    if np.linalg.det(V) < 0: V[:, 2] = -V[:, 2]
    rot = lambda X: (X - center) @ V; rotd = lambda D: D @ V
    Pc = rot(P); pfc = rot(pf)
    lo = np.percentile(Pc, 1, 0); hi = np.percentile(Pc, 99, 0); cen = (lo + hi) / 2; half = 0.62 * max(hi - lo)
    lims = [(cen[i] - half, cen[i] + half) for i in range(3)]

    def draw(ax, n, s=1.2, al=0.55):
        idx = rng.choice(len(Pc), min(len(Pc), n), replace=False)
        ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c=s_pt[idx] / L, cmap=CMAP, s=s, alpha=al, linewidths=0)
        for i, sl in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)): sl(*lims[i])

    regions = [("GLOTTIS", 0.015), ("SUBGLOTTIS", 0.34), ("TRACHEA", 0.62), ("DEEP TRACHEA", 0.94)]
    def labels(ax):
        for nm, fr in regions:
            p = rot(np.array(splev(np.interp(fr * L, scum, uf), tck)).T)
            ax.text(p[0], p[1], p[2] + half * 0.5, nm, color="white", fontsize=11, ha="center", fontweight="bold")

    # A
    fig = plt.figure(figsize=(13, 11), facecolor="black"); ax = fig.add_subplot(111, projection="3d")
    draw(ax, 190000, 1.3, 0.6); style3d(ax); ax.view_init(18, -62); labels(ax)
    ax.set_title(f"{a.label} reconstructed airway — dense cloud, proximal→distal\n(calibration {a.calibnote})",
                 color="white", fontsize=13)
    fig.savefig(out / "A_organ_cloud.png", dpi=140, facecolor="black", bbox_inches="tight"); plt.close(fig)
    # B
    fig = plt.figure(figsize=(13, 11), facecolor="black"); ax = fig.add_subplot(111, projection="3d")
    idx = rng.choice(len(Pc), min(len(Pc), 110000), replace=False)
    ax.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c="0.55", s=1.0, alpha=0.18, linewidths=0)
    ax.plot(pfc[:, 0], pfc[:, 1], pfc[:, 2], c="cyan", lw=2.6)
    for i, sl in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)): sl(*lims[i])
    for key in ["glottis", "+5mm", "+10mm"]:
        d = cur[key]; p0c = rot(d["p0"]); e1c, e2c = rotd(d["e1"]), rotd(d["e2"]); hh = 1.8 * Rmed
        q = np.array([p0c + x*hh*e1c + y*hh*e2c for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        ax.add_collection3d(Poly3DCollection([q], alpha=.5, facecolor=PLANE[key], edgecolor=PLANE[key]))
        ax.text(p0c[0], p0c[1], p0c[2] + hh * 1.5, key, color=PLANE[key], fontsize=11, fontweight="bold")
    style3d(ax); ax.view_init(18, -62)
    ax.set_title(f"{a.label} airway + medial centerline (cyan) + glottis/+5mm/+10mm planes\n"
                 "(+5/+10mm placement PROVISIONAL — blade-gated)", color="white", fontsize=12)
    fig.savefig(out / "B_centerline.png", dpi=140, facecolor="black", bbox_inches="tight"); plt.close(fig)
    # C
    views = [("side (length)", 8, -90), ("top (length)", 89, -90), ("end-on (down lumen)", 4, 0), ("oblique", 20, -58)]
    fig = plt.figure(figsize=(15, 14), facecolor="black")
    for i, (nm, el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d"); draw(ax, 80000, 1.0, 0.5)
        style3d(ax); ax.view_init(el, az); ax.set_title(nm, color="white", fontsize=12)
    fig.suptitle(f"{a.label} reconstructed airway — 4 views, same scale (shape inspection)", color="white", fontsize=14)
    fig.savefig(out / "C_multiview.png", dpi=135, facecolor="black", bbox_inches="tight"); plt.close(fig)
    # landmark rings
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for j, key in enumerate(["glottis", "+5mm", "+10mm"]):
        ax = axes[j]; d = cur[key]
        if len(d["ip"]): ax.scatter(d["ip"][:, 0], d["ip"][:, 1], s=2.5, c=PLANE[key], alpha=.55)
        ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{key}  cov={d['cov']*100:.0f}%  r_std/r_med={d['ratio']:.2f}\n"
                     f"{'CLOSED' if d['closed'] else 'open/aperture'}", fontsize=10,
                     color=("green" if d["closed"] else "crimson"))
    fig.suptitle(f"{a.label} Barbour landmark slices (glottis / +5mm / +10mm, provisional scale; scene units)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "landmarks.png", dpi=130); plt.close(fig)

    # double Poisson (mesh stat only, for record)
    try:
        Vm = double_poisson(P, out); nmesh = len(Vm)
    except Exception as e:
        nmesh = -1; print("double-Poisson failed:", str(e)[:120])

    rep = {"label": a.label, "calibnote": a.calibnote, "glottis_frame": a.glottis,
           "dense_pts": int(len(P)), "L_scene": round(float(L), 2), "Rmed_scene": round(Rmed, 3),
           "k0_mm_per_unit_PROVISIONAL": round(k0, 3), "double_poisson_verts": int(nmesh),
           "landmarks": {k: {"s": round(cur[k]["s"], 2), "coverage": round(cur[k]["cov"], 3),
                             "r_med_scene": round(cur[k]["r_med"], 3), "r_std/r_med": round(cur[k]["ratio"], 3),
                             "closed": cur[k]["closed"]} for k in cur}}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"[{a.label}] dense={len(P)} L={L:.2f} Rmed={Rmed:.3f} | "
          f"glottis cov{cur['glottis']['cov']*100:.0f}%/{cur['glottis']['ratio']:.2f} "
          f"+5mm cov{cur['+5mm']['cov']*100:.0f}%/{cur['+5mm']['ratio']:.2f}({'Y' if cur['+5mm']['closed'] else 'n'}) "
          f"+10mm cov{cur['+10mm']['cov']*100:.0f}%/{cur['+10mm']['ratio']:.2f}({'Y' if cur['+10mm']['closed'] else 'n'})")
    print(f"saved A/B/C + landmarks + report -> {out}")


if __name__ == "__main__":
    main()
