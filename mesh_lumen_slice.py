"""The untested combination: slice a WATERTIGHT Poisson MESH SURFACE perpendicular
to the LUMEN axis (not the camera path, not the raw cloud). A Poisson surface is
a 2-manifold (~zero radial thickness) fit through the noisy point band, so its
perpendicular slice should be ONE clean closed contour -- the denoising step we
were missing. This tests the user's hypothesis directly.

Run in depth-eval env.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, trimesh, open3d as o3d
if not hasattr(np, "in1d"):      # numpy 2.x removed np.in1d; trimesh.section needs it
    np.in1d = np.isin
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OD = Path("/home/mi3dr/projects/bronchotrust/runs/barbour_dense/7-V1")
DIAG = OD / "diag"


def lumen_centerline(P, cam_centroid):
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc[np.random.default_rng(0).choice(len(Pc), min(len(Pc), 60000), replace=False)], full_matrices=False)
    axis = Vt[0]
    if (P.mean(0) - cam_centroid) @ axis < 0:
        axis = -axis
    s = (P - P.mean(0)) @ axis
    edges = np.linspace(np.percentile(s, 1), np.percentile(s, 99), 41)
    cen = [P[(s >= edges[i]) & (s < edges[i + 1])].mean(0) for i in range(40)
           if ((s >= edges[i]) & (s < edges[i + 1])).sum() > 20]
    cen = np.array(cen)
    tck, _ = splprep(cen.T, s=0.3 * len(cen), k=3)
    uf = np.linspace(0, 1, 2000); pf = np.array(splev(uf, tck)).T
    if np.linalg.norm(pf[0] - cam_centroid) > np.linalg.norm(pf[-1] - cam_centroid):
        uf = uf[::-1]
    return tck, uf


def main():
    global OD, DIAG
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--recon", default=str(OD)); a = ap.parse_args()
    OD = Path(a.recon); DIAG = OD / "diag"; DIAG.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(OD / "sparse/0"))
    C = np.array([(-np.array(im.cam_from_world().matrix())[:3, :3].T @ np.array(im.cam_from_world().matrix())[:3, 3])
                  for im in rec.images.values()])
    cam_centroid = C.mean(0)
    pcd = o3d.io.read_point_cloud(str(OD / "dense0/fused.ply"))
    P = np.asarray(pcd.points)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]

    # WATERTIGHT Poisson mesh (NO trim) -> closed 2-manifold surface
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    for trim, tag in [(0.0, "watertight"), (0.01, "trim1")]:
        m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        dens = np.asarray(dens)
        if trim > 0:
            m.remove_vertices_by_index(np.where(dens < np.quantile(dens, trim))[0].tolist())
        m = m.filter_smooth_taubin(number_of_iterations=20)
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        mesh = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles), process=False)
        print(f"[{tag}] mesh verts={len(mesh.vertices)} tris={len(mesh.faces)} watertight={mesh.is_watertight}")

        tck, uf = lumen_centerline(P, cam_centroid)
        pf = np.array(splev(uf, tck)).T
        scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
        L = scum[-1]
        s_nodes = np.linspace(0.05 * L, 0.95 * L, 10)
        u_n = np.interp(s_nodes, scum, uf)
        Pn = np.array(splev(u_n, tck)).T
        Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
        tp = cKDTree(P); Rm = np.median(cKDTree(pf).query(P)[0])

        fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
        rep = []
        for k, (p0, tg, sv) in enumerate(zip(Pn, Tn, s_nodes)):
            ax = axes.flat[k]
            # underlay the cloud band for context
            rel = P[tp.query_ball_point(p0, 6 * Rm)] - p0; da = rel @ tg; ins = rel[np.abs(da) <= 0.4]
            up = np.array([0, 0, 1.]) if abs(tg[2]) < .95 else np.array([0, 1., 0])
            e1 = up - (up @ tg) * tg; e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(tg, e1)
            if len(ins):
                ip = np.stack([ins @ e1, ins @ e2], 1)
                ax.scatter(ip[:, 0], ip[:, 1], s=1, c="0.75", alpha=.4)
            status = ""; area = None
            try:
                sec = mesh.section(plane_origin=p0, plane_normal=tg)
                planar, to3d = sec.to_planar()
                o2 = (np.linalg.inv(to3d) @ np.array([*p0, 1.0]))[:2]
                for ent in planar.entities:
                    v = planar.vertices[ent.points]
                    ax.plot(v[:, 0] - o2[0], v[:, 1] - o2[1], "-", c="navy", lw=1.5)
                polys = planar.polygons_full or []
                enc = [pg for pg in polys if pg.contains(Point(o2))]
                if enc:
                    pg = max(enc, key=lambda g: g.area); area = float(pg.area)
                    xs, ys = pg.exterior.xy
                    ax.fill(np.array(xs) - o2[0], np.array(ys) - o2[1], color="lightgreen", alpha=.5)
                    status = f"CLOSED A={area:.2f} Deq={2*np.sqrt(area/np.pi):.2f}"
                else:
                    status = f"open ({len(polys)} loops)"
            except Exception:
                status = "no section"
            ax.plot(0, 0, "r+", ms=10, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            end = "subglottis(near scope)" if k <= 3 else "deeper"
            ax.set_title(f"node{k} [{end}]\n{status}", fontsize=8)
            rep.append({"node": k, "area": area, "status": status})
        nclosed = sum(1 for r in rep if r["area"] is not None)
        fig.suptitle(f"7-V1 MESH({tag}, watertight={mesh.is_watertight}) sliced perpendicular to LUMEN axis — "
                     f"{nclosed}/10 closed contours (green). gray=cloud band", fontsize=11)
        fig.tight_layout(); fig.savefig(DIAG / f"mesh_lumen_slices_{tag}.png", dpi=130); plt.close(fig)
        areas = [r["area"] for r in rep if r["area"]]
        print(f"  [{tag}] {nclosed}/10 closed; areas={[round(a,2) for a in areas]}")
        if areas and len(areas) >= 3:
            amin, amax = min(areas), max(areas)
            print(f"    A_min={amin:.2f} A_max={amax:.2f}  %obstr(if measured)={(1-amin/amax)*100:.0f}%")
        (DIAG / f"mesh_lumen_report_{tag}.json").write_text(json.dumps({"tag": tag, "watertight": bool(mesh.is_watertight), "nodes": rep}, indent=2))
    print(f"saved mesh_lumen_slices_*.png -> {DIAG}")


if __name__ == "__main__":
    main()
