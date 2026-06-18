"""Subglottis cross-section VISUAL diagnostic on model-0 (7-V1, f110-411).

Shows, at ~10 centerline nodes (focused on subglottis f130-220):
  - the in-plane CLOUD scatter (the actual reconstructed wall: ring / arc / blob)
  - the watertight-MESH section contour overlaid (closed? encloses centerline?)
  - angular coverage %
Plus 3-view renders of the dense cloud and the mesh.

Judge VISUALLY. No obstruction %. Run in depth-eval env.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, trimesh, open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OD = Path("/home/mi3dr/projects/bronchotrust/runs/barbour_dense/7-V1")
SFM = OD / "sparse/0"
PLY = OD / "dense0/fused.ply"
DIAG = OD / "diag"; DIAG.mkdir(exist_ok=True)
SUBGLOTTIS = (130, 220)
N_NODES = 10


def main():
    rec = pycolmap.Reconstruction(str(SFM))
    cams = []
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        cams.append((fi, -R.T @ t))
    cams.sort()
    fis = np.array([c[0] for c in cams]); C = np.array([c[1] for c in cams])

    pcd = o3d.io.read_point_cloud(str(PLY))
    P = np.asarray(pcd.points)
    print(f"cameras={len(C)} f{fis.min()}-{fis.max()}  dense cloud={len(P)}")

    # centerline through cameras
    tck, u = splprep(C.T, s=0.5 * len(C), k=3)
    uf = np.linspace(0, 1, 4000); pf = np.array(splev(uf, tck)).T
    scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    L = scum[-1]
    cam_s = np.array([scum[cKDTree(pf).query(c)[1]] for c in C])
    # outlier filter cloud
    dcl, _ = cKDTree(pf).query(P, k=1); R = np.median(dcl)
    P = P[dcl <= 3 * R]
    print(f"centerline L={L:.2f}  lumen R~{R:.2f}  cloud(filtered)={len(P)}")

    # watertight-ish Poisson mesh (low trim + Taubin)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    dens = np.asarray(dens)
    m.remove_vertices_by_index(np.where(dens < np.quantile(dens, 0.005))[0].tolist())
    m = m.filter_smooth_taubin(number_of_iterations=20)
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(DIAG / "model0_mesh.ply"), m)
    mesh = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles), process=False)
    print(f"mesh verts={len(mesh.vertices)} tris={len(mesh.faces)} watertight={mesh.is_watertight}")

    # nodes: focus on subglottis arclength
    sub_mask = (fis >= SUBGLOTTIS[0]) & (fis <= SUBGLOTTIS[1])
    if sub_mask.sum() >= 2:
        s_lo, s_hi = cam_s[sub_mask].min(), cam_s[sub_mask].max()
        s_lo = max(0.02 * L, s_lo - 0.1 * (s_hi - s_lo)); s_hi = min(0.98 * L, s_hi + 0.1 * (s_hi - s_lo))
    else:
        s_lo, s_hi = 0.05 * L, 0.6 * L
    s_nodes = np.linspace(s_lo, s_hi, N_NODES)
    u_n = np.interp(s_nodes, scum, uf)
    Pn = np.array(splev(u_n, tck)).T
    Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9

    pts_tree = cKDTree(P)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    rep = []
    for k, (p0, tg, sv) in enumerate(zip(Pn, Tn, s_nodes)):
        ax = axes.flat[k]
        nf = int(fis[np.argmin(np.abs(cam_s - sv))])
        reg = "SUBGLOTTIS" if SUBGLOTTIS[0] <= nf <= SUBGLOTTIS[1] else ("glottis" if nf < SUBGLOTTIS[0] else "trachea")
        # cloud slab
        idx = pts_tree.query_ball_point(p0, 4 * R)
        rel = P[idx] - p0; da = rel @ tg
        slab = rel[np.abs(da) <= 0.10 * R]
        up = np.array([0, 0, 1.]) if abs(tg[2]) < .95 else np.array([0, 1., 0])
        e1 = up - (up @ tg) * tg; e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(tg, e1)
        ip = np.stack([slab @ e1, slab @ e2], 1) if len(slab) else np.zeros((0, 2))
        th = np.arctan2(ip[:, 1], ip[:, 0]) if len(ip) else np.array([])
        cov = (np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean() if len(ip) else 0
        if len(ip):
            ax.scatter(ip[:, 0], ip[:, 1], s=2, c="0.5", alpha=.5)
        # mesh contour
        status = ""
        try:
            sec = mesh.section(plane_origin=p0, plane_normal=tg)
            planar, to3d = sec.to_planar()
            o2 = (np.linalg.inv(to3d) @ np.array([*p0, 1.0]))[:2]
            for ent in planar.entities:
                v = planar.vertices[ent.points]
                ax.plot(v[:, 0] - o2[0], v[:, 1] - o2[1], "-", c="navy", lw=1.2)
            polys = planar.polygons_full or []
            enc = [pg for pg in polys if pg.contains(Point(o2))]
            if enc:
                pg = max(enc, key=lambda g: g.area)
                xs, ys = pg.exterior.xy
                ax.fill(np.array(xs) - o2[0], np.array(ys) - o2[1], color="lightgreen", alpha=.45)
                status = f"CLOSED A={pg.area:.2f}"
            else:
                status = f"open({len(polys)}loops)"
        except Exception:
            status = "no section"
        ax.plot(0, 0, "r+", ms=12, mew=2)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"~f{nf} [{reg}]  cov={cov*100:.0f}%\n{status}", fontsize=9)
        rep.append({"s": float(sv), "frame": nf, "region": reg, "coverage": float(cov),
                    "n_pts": int(len(ip)), "status": status})
    fig.suptitle("7-V1 SUBGLOTTIS (model 0) cross-sections: gray=dense-cloud wall, navy=mesh contour, "
                 "green=closed loop, red+=centerline", fontsize=12)
    fig.tight_layout(); fig.savefig(DIAG / "subglottis_slices.png", dpi=130); plt.close(fig)

    # cloud + mesh 3-view
    Pp = P[np.random.default_rng(0).choice(len(P), min(len(P), 90000), replace=False)]
    V = np.asarray(m.vertices)
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    for row, (pts, lab, col) in enumerate([(Pp, f"cloud {len(P)}", "0.6"), (V, f"mesh {len(V)}v", "steelblue")]):
        for a, (ix, iy, lx, ly) in zip(ax[row], [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
            a.scatter(pts[:, ix], pts[:, iy], s=.4, c=col, alpha=.5)
            a.plot(C[:, ix], C[:, iy], "-", c="red", lw=1.2)
            sub = sub_mask
            a.scatter(C[sub, ix], C[sub, iy], s=12, c="orange", label="subglottis cams")
            a.set_title(f"{lab} {lx}{ly}"); a.set_aspect("equal", "datalim"); a.grid(alpha=.3)
    fig.suptitle("7-V1 model 0 (subglottis): dense cloud (top) + mesh (bottom); orange=subglottis cameras")
    fig.tight_layout(); fig.savefig(DIAG / "subglottis_cloud_mesh.png", dpi=130); plt.close(fig)

    n_closed = sum(1 for r in rep if "CLOSED" in r["status"])
    n_sub_closed = sum(1 for r in rep if r["region"] == "SUBGLOTTIS" and "CLOSED" in r["status"])
    (DIAG / "subglottis_report.json").write_text(json.dumps(
        {"L": float(L), "R": float(R), "dense": len(P), "mesh_verts": len(V),
         "watertight": bool(mesh.is_watertight), "nodes": rep,
         "n_closed": n_closed, "n_subglottis_closed": n_sub_closed}, indent=2))
    print(f"\nnodes closed: {n_closed}/{N_NODES}  (subglottis-region closed: {n_sub_closed})")
    for r in rep:
        print(f"  ~f{r['frame']} [{r['region']}] cov={r['coverage']*100:.0f}% pts={r['n_pts']} {r['status']}")
    print(f"saved subglottis_slices.png + subglottis_cloud_mesh.png -> {DIAG}")


if __name__ == "__main__":
    main()
