"""Geometry-debug visualization (NO SfM/MVS rerun). For 32-V2 nodes 0-3 at the
proximal subglottis: overlay dense cloud + centerline + slice plane + resulting
cross-section in 3D, to check whether the slice plane is truly PERPENDICULAR to
the lumen. Compares the current GLOBAL-PCA centerline (cloud PC1 + per-bin
centroid, anchored to one straight axis) against a LOCAL MEDIAL-AXIS centerline
(iteratively refined to the local perpendicular centroid -> follows curvature).
Reads runs/fuse_32v2/dense0/fused.ply + sparse/0 only. Run in depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2")
DIAG = OD / "diag"
SLAB = 0.4
rng = np.random.default_rng(0)


def cam_centers(rec):
    out = []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); out.append(-M[:3, :3].T @ M[:3, 3])
    return np.array(out)


def arclen(pf):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])


def pca_centerline(P, cam_centroid):
    Pc = P - P.mean(0)
    idx = rng.choice(len(Pc), min(len(Pc), 60000), replace=False)
    _, _, Vt = np.linalg.svd(Pc[idx], full_matrices=False); axis = Vt[0]
    if (P.mean(0) - cam_centroid) @ axis < 0: axis = -axis
    s = (P - P.mean(0)) @ axis
    edges = np.linspace(np.percentile(s, 1), np.percentile(s, 99), 41)
    cen = [P[(s >= edges[i]) & (s < edges[i + 1])].mean(0) for i in range(40)
           if ((s >= edges[i]) & (s < edges[i + 1])).sum() > 20]
    cen = np.array(cen)
    tck, _ = splprep(cen.T, s=0.3 * len(cen), k=3)
    uf = np.linspace(0, 1, 2000); pf = np.array(splev(uf, tck)).T
    if np.linalg.norm(pf[0] - cam_centroid) > np.linalg.norm(pf[-1] - cam_centroid):
        uf = uf[::-1]; pf = pf[::-1]
    return tck, uf, pf


def medial_centerline(P, pf_init, Rmed, cam_centroid):
    """Refine each centerline sample to the LOCAL perpendicular-plane centroid
    (medial center), recomputing the tangent from the refined track each pass so
    the centerline FOLLOWS curvature instead of staying on the global axis."""
    tree = cKDTree(P); ref = pf_init.copy()
    for _ in range(5):
        T = np.gradient(ref, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        new = ref.copy()
        for i, (p, t) in enumerate(zip(ref, T)):
            idx = tree.query_ball_point(p, 3.0 * Rmed)
            if len(idx) < 15: continue
            loc = P[idx] - p; perp = loc[np.abs(loc @ t) <= 0.5]
            if len(perp) < 10: continue
            c = perp.mean(0); c -= (c @ t) * t            # move only within the perpendicular plane
            new[i] = p + 0.6 * c
        ref = new
    tck, _ = splprep(ref.T, s=0.5 * len(ref), k=3)
    uf = np.linspace(0, 1, 2000); pf = np.array(splev(uf, tck)).T
    if np.linalg.norm(pf[0] - cam_centroid) > np.linalg.norm(pf[-1] - cam_centroid):
        uf = uf[::-1]; pf = pf[::-1]
    return tck, uf, pf


def frame(t):
    up = np.array([0, 0, 1.]) if abs(t[2]) < .95 else np.array([0, 1., 0])
    e1 = up - (up @ t) * t; e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(t, e1)
    return e1, e2


def nodes_from(tck, uf, P, cam_centroid, fracs):
    pf = np.array(splev(uf, tck)).T
    sc = arclen(pf); L = sc[-1]
    u_n = np.interp([f * L for f in fracs], sc, uf)
    Pn = np.array(splev(u_n, tck)).T
    Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
    # tangent points away from cameras (consistent)
    for i in range(len(Tn)):
        if (Pn[i] - cam_centroid) @ Tn[i] < 0: Tn[i] = -Tn[i]
    return Pn, Tn, pf


def section(P, tree, p0, t, Rmed):
    rel = P[tree.query_ball_point(p0, 6 * Rmed)] - p0
    ins3 = rel[np.abs(rel @ t) <= SLAB]
    e1, e2 = frame(t)
    if not len(ins3): return ins3 + p0, np.zeros((0, 2)), 0.0, 0.0, 0.0
    ip = np.stack([ins3 @ e1, ins3 @ e2], 1)
    th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
    cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
    return ins3 + p0, ip, cov, float(np.median(r)), float(np.std(r))


def plane_quad(p0, t, half):
    e1, e2 = frame(t)
    c = [p0 + a * half * e1 + b * half * e2 for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array(c)


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); C = cam_centers(rec); cam_centroid = C.mean(0)
    tree = cKDTree(P)
    tck_p, uf_p, pf_p = pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck_m, uf_m, pf_m = medial_centerline(P, pf_p, Rmed, cam_centroid)
    fracs = np.linspace(0.03, 0.97, 10)[:4]
    Pn_p, Tn_p, _ = nodes_from(tck_p, uf_p, P, cam_centroid, fracs)
    Pn_m, Tn_m, _ = nodes_from(tck_m, uf_m, P, cam_centroid, fracs)

    rows = []
    # ---------- 3D overlay: 2 rows (PCA, MEDIAL) x 4 nodes ----------
    fig = plt.figure(figsize=(22, 11))
    Psub = P[rng.choice(len(P), min(len(P), 120000), replace=False)]
    for col in range(4):
        for row, (name, pf, Pn, Tn, col3, oth_T) in enumerate([
                ("PCA", pf_p, Pn_p, Tn_p, "tab:blue", Tn_m),
                ("MEDIAL", pf_m, Pn_m, Tn_m, "tab:orange", Tn_p)]):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            p0 = Pn[col]; t = Tn[col]
            loc = Psub[np.linalg.norm(Psub - p0, axis=1) < 4.0 * Rmed]
            ax.scatter(loc[:, 0], loc[:, 1], loc[:, 2], s=1, c="0.7", alpha=.25)
            seg = pf[np.linalg.norm(pf - p0, axis=1) < 5 * Rmed]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=col3, lw=2.5, label=f"{name} centerline")
            ins3, ip, cov, rm, rs = section(P, tree, p0, t, Rmed)
            ax.scatter(ins3[:, 0], ins3[:, 1], ins3[:, 2], s=4, c="crimson", alpha=.6, label="slice points")
            q = plane_quad(p0, t, 1.6 * max(rm, 0.4))
            ax.add_collection3d(Poly3DCollection([q], alpha=.18, facecolor=col3, edgecolor=col3))
            ax.quiver(*p0, *(t * 1.8 * max(rm, .4)), color="k", lw=1.5, arrow_length_ratio=.18)
            ang = np.degrees(np.arccos(np.clip(abs(t @ oth_T[col]), -1, 1)))
            ax.set_title(f"{name}  node{col}\ncov={cov*100:.0f}%  r_std/r_med={rs/max(rm,1e-6):.2f}\n"
                         f"tangent vs other: {ang:.1f}°", fontsize=8.5, color=col3)
            r = 4 * Rmed
            ax.set_xlim(p0[0]-r, p0[0]+r); ax.set_ylim(p0[1]-r, p0[1]+r); ax.set_zlim(p0[2]-r, p0[2]+r)
            try: ax.set_box_aspect((1, 1, 1))
            except Exception: pass
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if row == 0:
                rows.append({"node": col, "pca_cov": round(cov*100, 1), "pca_ratio": round(rs/max(rm, 1e-6), 3),
                             "pca_rmed": round(rm, 3), "tangent_angle_deg": round(float(ang), 1)})
            else:
                rows[col].update({"medial_cov": round(cov*100, 1), "medial_ratio": round(rs/max(rm, 1e-6), 3),
                                  "medial_rmed": round(rm, 3)})
    fig.suptitle("32-V2 nodes 0-3 (proximal subglottis): dense cloud(gray) + centerline + slice plane(quad) + "
                 "slice points(red) + tangent(black arrow)\nTOP=current global-PCA centerline   BOTTOM=local medial-axis "
                 "centerline   (black arrow ⟂ plane should point DOWN the lumen)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(DIAG / "geomdebug_3d_nodes0-3.png", dpi=120); plt.close(fig)

    # ---------- 2D in-plane cross-sections: PCA vs MEDIAL ----------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col in range(4):
        for row, (name, Pn, Tn, c) in enumerate([("PCA", Pn_p, Tn_p, "tab:blue"),
                                                  ("MEDIAL", Pn_m, Tn_m, "tab:orange")]):
            ax = axes[row, col]; p0 = Pn[col]; t = Tn[col]
            _, ip, cov, rm, rs = section(P, tree, p0, t, Rmed)
            if len(ip): ax.scatter(ip[:, 0], ip[:, 1], s=2, c=c, alpha=.5)
            ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_title(f"{name} node{col}: cov={cov*100:.0f}% r_med={rm:.2f}\nr_std/r_med={rs/max(rm,1e-6):.2f}",
                         fontsize=9, color=c)
    fig.suptitle("32-V2 nodes 0-3 in-plane cross-section: global-PCA (top) vs local medial-axis (bottom). "
                 "A clean thin ring = perpendicular slice through a tube.", fontsize=11)
    fig.tight_layout(); fig.savefig(DIAG / "geomdebug_2d_sections0-3.png", dpi=130); plt.close(fig)

    (DIAG / "geomdebug.json").write_text(json.dumps(rows, indent=2))
    print("node  PCA(cov%/ratio)   MEDIAL(cov%/ratio)   tangentΔ(deg)   r_med PCA->MED")
    for r in rows:
        print(f"  {r['node']}    {r['pca_cov']:5.0f} / {r['pca_ratio']:.2f}     "
              f"{r['medial_cov']:5.0f} / {r['medial_ratio']:.2f}        {r['tangent_angle_deg']:5.1f}      "
              f"{r['pca_rmed']:.2f} -> {r['medial_rmed']:.2f}")
    print(f"\nsaved {DIAG}/geomdebug_3d_nodes0-3.png  +  geomdebug_2d_sections0-3.png")


if __name__ == "__main__":
    main()
