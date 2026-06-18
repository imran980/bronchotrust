"""Barbour-style cross-section: slice the POISSON MESH (not raw points) with planes
perpendicular to the camera-centerline, take the CLOSED contour enclosing the
centerline -> cross-sectional area -> D_eq = 2*sqrt(A/pi).

A closed contour = a real measurable ring (what point-slicing could never give).
Reports, per node: closed? area (3 ways: shapely polygon, convex hull, n_loops),
restricted to / tagged with the sub-cord region.

Run in depth-eval env (trimesh, shapely). Default: 7-V1 dense mesh.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pycolmap, trimesh
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, Polygon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF = "/home/mi3dr/projects/bronchotrust/runs/barbour_dense/7-V1"
WIN = {"sub_in": [130, 220], "sub_out": [1846, 1992]}  # 7-V1 (sub_out outside passthrough)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=DEF)
    ap.add_argument("--mesh", default=None, help="override mesh ply")
    ap.add_argument("--n-nodes", type=int, default=160)
    args = ap.parse_args()
    od = Path(args.recon)
    summ = json.loads((od / "summary.json").read_text())
    import open3d as o3d
    _m = o3d.io.read_triangle_mesh(args.mesh or summ["mesh_ply"])
    mesh = trimesh.Trimesh(vertices=np.asarray(_m.vertices),
                           faces=np.asarray(_m.triangles), process=False)
    print(f"mesh: {len(mesh.vertices)} verts {len(mesh.faces)} faces")
    rec = pycolmap.Reconstruction(summ["sfm_model_dir"])
    cams = []
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        cams.append((fi, -R.T @ t))
    cams.sort()
    fis = np.array([c[0] for c in cams]); C = np.array([c[1] for c in cams])

    tck, u = splprep(C.T, s=0.5 * len(C), k=3)
    uf = np.linspace(0, 1, 4000); pf = np.array(splev(uf, tck)).T
    scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, 0), axis=1))]) if False else \
        np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    L = scum[-1]
    # map camera frame -> arclength (nearest pf point)
    from scipy.spatial import cKDTree
    tree = cKDTree(pf)
    cam_s = np.array([scum[tree.query(c)[1]] for c in C])
    s_t = np.linspace(0, L, args.n_nodes)
    u_t = np.interp(s_t, scum, uf)
    P = np.array(splev(u_t, tck)).T
    Tg = np.array(splev(u_t, tck, der=1)).T
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9

    rows = []
    for k, (p0, tg, sv) in enumerate(zip(P, Tg, s_t)):
        try:
            sec = mesh.section(plane_origin=p0, plane_normal=tg)
        except Exception:
            sec = None
        if sec is None:
            rows.append({"s": float(sv), "closed": False, "n_loops": 0}); continue
        try:
            planar, to3d = sec.to_planar()
        except Exception:
            rows.append({"s": float(sv), "closed": False, "n_loops": 0}); continue
        # centerline point in planar coords
        inv = np.linalg.inv(to3d)
        o2 = (inv @ np.array([p0[0], p0[1], p0[2], 1.0]))[:2]
        polys = planar.polygons_full if planar.polygons_full is not None else []
        enclosing = [pg for pg in polys if pg.contains(Point(o2))]
        if enclosing:
            pg = max(enclosing, key=lambda g: g.area)
            area = float(pg.area)
            rows.append({"s": float(sv), "closed": True, "n_loops": len(polys),
                         "area": area, "Deq": float(2 * np.sqrt(area / np.pi)),
                         "perim": float(pg.length)})
        else:
            # not enclosed -> open arc / off-center; record largest loop area if any
            a = max((pg.area for pg in polys), default=0.0)
            rows.append({"s": float(sv), "closed": False, "n_loops": len(polys),
                         "largest_loop_area": float(a)})

    def seg(s):
        f = fis[np.argmin(np.abs(cam_s - s))]
        return "sub_in" if WIN["sub_in"][0] <= f <= WIN["sub_in"][1] else (
            "sub_out" if WIN["sub_out"][0] <= f <= WIN["sub_out"][1] else "other")
    for r in rows:
        r["seg"] = seg(r["s"])
    closed = [r for r in rows if r.get("closed")]
    closed_sub = [r for r in closed if r["seg"] == "sub_in"]
    print(f"centerline L={L:.2f}  nodes={len(rows)}")
    print(f"CLOSED rings (centerline-enclosing): {len(closed)}/{len(rows)}")
    print(f"  in sub_in region: {len(closed_sub)}")
    if closed:
        A = np.array([r["area"] for r in closed]); D = np.array([r["Deq"] for r in closed])
        print(f"  area range {A.min():.3f}-{A.max():.3f}  Deq range {D.min():.3f}-{D.max():.3f} (scene units)")
    out = {"recon": str(od), "mesh_verts": len(mesh.vertices), "centerline_L": float(L),
           "n_nodes": len(rows), "n_closed": len(closed), "n_closed_sub_in": len(closed_sub),
           "cam_frame_arclength": {int(f): float(s) for f, s in zip(fis, cam_s)},
           "nodes": rows}
    (od / "mesh_xsection.json").write_text(json.dumps(out, indent=2))

    # profile plot: Deq vs arclength, closed=green, open=red
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    sc = np.array([r["s"] for r in rows])
    deq = np.array([r.get("Deq", np.nan) for r in rows])
    cl = np.array([r.get("closed", False) for r in rows])
    ax[0].scatter(sc[cl], deq[cl], c="green", s=22, label="closed ring")
    ax[0].scatter(sc[~cl], np.zeros((~cl).sum()), c="red", s=10, marker="x", label="not closed")
    for f, s in zip(fis, cam_s):
        seg_c = {"sub_in": "#2ca02c", "sub_out": "#19c819", "other": "#cccccc"}[(
            "sub_in" if WIN["sub_in"][0] <= f <= WIN["sub_in"][1] else ("sub_out" if WIN["sub_out"][0] <= f <= WIN["sub_out"][1] else "other"))]
        ax[0].axvline(s, ymin=0.97, color=seg_c, lw=1.2, alpha=.7)
    ax[0].set_ylabel("Deq (scene units)"); ax[0].legend()
    ax[0].set_title(f"{od.name} MESH cross-sections: {len(closed)}/{len(rows)} closed rings "
                    f"(green ticks=sub_in cameras)")
    nl = np.array([r.get("n_loops", 0) for r in rows])
    ax[1].scatter(sc, nl, c=["green" if c else "red" for c in cl], s=14)
    ax[1].set_ylabel("# loops in slice"); ax[1].set_xlabel("arclength s")
    fig.tight_layout(); fig.savefig(od / "mesh_xsection_profile.png", dpi=130); plt.close(fig)
    print(f"saved {od/'mesh_xsection.json'} + profile")


if __name__ == "__main__":
    main()
