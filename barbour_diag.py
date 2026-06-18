"""Barbour-style VISUAL diagnostics for a dense recon (default 7-V1).

Produces (in <recon>/diag/):
  1 cloud.png            dense MVS cloud, 3 projections + camera path
  2 mesh.png             Poisson mesh (open3d, low-trim + Taubin), 3 projections
  3 slices.png           10 representative MESH cross-section contours (the
                         actual plane-mesh intersection) with closed/open status
  4 frame_distribution.png  extracted vs registered frames along the video
  5 region_counts.png    registered frames per airway region (+ printed)

We SLICE THE MESH (not raw points) and judge each slice VISUALLY (closed loop
enclosing the centerline = a real airway boundary). No obstruction % here.

Run in depth-eval env.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pycolmap, trimesh, open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF = "/home/mi3dr/projects/bronchotrust/runs/barbour_dense/7-V1"
# 7-V1 airway regions (frame index)
REGIONS = [("glottis", 110, 130), ("subglottis", 130, 220),
           ("upper_trachea", 220, 450), ("mid/lower_trachea", 450, 700),
           ("carina_region", 700, 900)]


def remesh(fused_ply, out_ply, trim=0.02):
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_index(np.where(dens < np.quantile(dens, trim))[0].tolist())
    mesh = mesh.filter_smooth_taubin(number_of_iterations=30)
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    return np.asarray(pcd.points), np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def proj3(ax_list, pts, cw, label, s=.3, c="lightgray"):
    for axp, (ix, iy, lx, ly) in zip(ax_list, [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        if len(pts):
            axp.scatter(pts[:, ix], pts[:, iy], s=s, c=c, alpha=.5)
        axp.plot(cw[:, ix], cw[:, iy], "-", c="red", lw=1.0)
        axp.set_title(f"{lx}{ly}"); axp.set_aspect("equal", "datalim"); axp.grid(alpha=.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=DEF)
    args = ap.parse_args()
    od = Path(args.recon)
    diag = od / "diag"; diag.mkdir(parents=True, exist_ok=True)
    summ = json.loads((od / "summary.json").read_text())
    rec = pycolmap.Reconstruction(summ["sfm_model_dir"])
    cams = []
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        cams.append((fi, -R.T @ t))
    cams.sort()
    fis = np.array([c[0] for c in cams]); C = np.array([c[1] for c in cams])
    reg_set = set(int(f) for f in fis)
    extracted = sorted(int(re.search(r"f(\d+)", p.name).group(1)) for p in (od / "images").glob("f*.png"))

    print(f"recon {od.name}: extracted={len(extracted)} registered={len(fis)} "
          f"frame range f{fis.min()}-{fis.max()}")

    # re-mesh
    P, V, F = remesh(od / "dense/fused.ply", diag / "mesh_taubin.ply")
    print(f"cloud {len(P)} pts; mesh {len(V)} verts {len(F)} tris")
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # centerline
    tck, u = splprep(C.T, s=0.5 * len(C), k=3)
    uf = np.linspace(0, 1, 4000); pf = np.array(splev(uf, tck)).T
    scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    L = scum[-1]
    cam_s = np.array([scum[cKDTree(pf).query(c)[1]] for c in C])

    # ---- 1 cloud ----
    Pp = P[np.random.default_rng(0).choice(len(P), min(len(P), 80000), replace=False)]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6)); proj3(ax, Pp, C, "cloud")
    fig.suptitle(f"{od.name} DENSE MVS cloud: {len(P)} pts + camera path ({len(C)} cams)")
    fig.tight_layout(); fig.savefig(diag / "1_cloud.png", dpi=130); plt.close(fig)

    # ---- 2 mesh ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    proj3(ax, V, C, "mesh", s=1.0, c="steelblue")
    fig.suptitle(f"{od.name} POISSON MESH (Taubin-smoothed): {len(V)} verts {len(F)} tris")
    fig.tight_layout(); fig.savefig(diag / "2_mesh.png", dpi=130); plt.close(fig)

    # ---- 3 ten slices ----
    s_nodes = np.linspace(0.04 * L, 0.96 * L, 10)
    u_n = np.interp(s_nodes, scum, uf)
    Pn = np.array(splev(u_n, tck)).T
    Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    slice_stats = []
    for k, (p0, tg, sv) in enumerate(zip(Pn, Tn, s_nodes)):
        ax = axes.flat[k]
        nearest_f = int(fis[np.argmin(np.abs(cam_s - sv))])
        reg = "glottis"
        for nm, lo, hi in REGIONS:
            if lo <= nearest_f < hi:
                reg = nm; break
        status = "no intersection"; closed = False; area = None; nloop = 0
        try:
            sec = mesh.section(plane_origin=p0, plane_normal=tg)
            planar, to3d = sec.to_planar()
            o2 = (np.linalg.inv(to3d) @ np.array([*p0, 1.0]))[:2]
            for ent in planar.entities:
                pts = planar.vertices[ent.points]
                ax.plot(pts[:, 0] - o2[0], pts[:, 1] - o2[1], "-", c="navy", lw=1.0)
            polys = planar.polygons_full or []
            nloop = len(polys)
            enc = [pg for pg in polys if pg.contains(Point(o2))]
            if enc:
                pg = max(enc, key=lambda g: g.area); area = float(pg.area); closed = True
                xs, ys = pg.exterior.xy
                ax.fill(np.array(xs) - o2[0], np.array(ys) - o2[1], color="lightgreen", alpha=.5)
                status = f"CLOSED A={area:.2f} Deq={2*np.sqrt(area/np.pi):.2f}"
            else:
                status = f"open/non-enclosing ({nloop} loops)"
        except Exception as e:
            status = f"section fail"
        ax.plot(0, 0, "r+", ms=12, mew=2)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"s={sv:.1f} ~f{nearest_f} [{reg}]\n{status}", fontsize=8)
        slice_stats.append({"s": float(sv), "nearest_frame": nearest_f, "region": reg,
                            "closed": closed, "n_loops": nloop, "area": area})
    fig.suptitle(f"{od.name} MESH cross-sections (10 nodes) — green=closed loop enclosing centerline(+)")
    fig.tight_layout(); fig.savefig(diag / "3_slices.png", dpi=130); plt.close(fig)

    # ---- 4 frame distribution ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.eventplot([extracted], colors="lightgray", lineoffsets=1, linelengths=0.8, label="extracted")
    ax.eventplot([sorted(reg_set)], colors="green", lineoffsets=2, linelengths=0.8, label="registered")
    for nm, lo, hi in REGIONS:
        ax.axvspan(lo, hi, alpha=0.08); ax.text((lo + hi) / 2, 2.7, nm, ha="center", fontsize=8, rotation=20)
    ax.set_yticks([1, 2]); ax.set_yticklabels(["extracted", "registered"])
    ax.set_xlabel("video frame index"); ax.set_title(
        f"{od.name}: {len(extracted)} extracted, {len(reg_set)} registered along the passthrough")
    fig.tight_layout(); fig.savefig(diag / "4_frame_distribution.png", dpi=130); plt.close(fig)

    # ---- 5 per-region counts ----
    counts = []
    for nm, lo, hi in REGIONS:
        ne = sum(1 for f in extracted if lo <= f < hi)
        nr = sum(1 for f in reg_set if lo <= f < hi)
        counts.append({"region": nm, "range": [lo, hi], "extracted": ne, "registered": nr,
                       "reg_frac": round(nr / max(ne, 1), 2)})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(counts))
    ax.bar(x - .2, [c["extracted"] for c in counts], .4, label="extracted", color="lightgray")
    ax.bar(x + .2, [c["registered"] for c in counts], .4, label="registered", color="green")
    for i, c in enumerate(counts):
        ax.text(i + .2, c["registered"], str(c["registered"]), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([c["region"] for c in counts], rotation=20)
    ax.set_ylabel("# frames"); ax.legend(); ax.set_title(f"{od.name} frames registered per airway region")
    fig.tight_layout(); fig.savefig(diag / "5_region_counts.png", dpi=130); plt.close(fig)

    rep = {"extracted": len(extracted), "registered": len(fis), "n_models": summ.get("n_models"),
           "centerline_L": float(L), "n_cloud": len(P), "n_mesh_verts": len(V),
           "slices": slice_stats, "n_slices_closed": sum(1 for s in slice_stats if s["closed"]),
           "region_counts": counts}
    (diag / "diag_report.json").write_text(json.dumps(rep, indent=2))
    print("\n=== REGION REGISTRATION ===")
    for c in counts:
        print(f"  {c['region']:<18} f{c['range'][0]}-{c['range'][1]}: registered {c['registered']}/{c['extracted']} ({c['reg_frac']})")
    print(f"\n10-slice mesh check: {rep['n_slices_closed']}/10 closed (enclosing centerline)")
    for s in slice_stats:
        print(f"  s={s['s']:.1f} ~f{s['nearest_frame']} [{s['region']}]: closed={s['closed']} loops={s['n_loops']} area={s['area']}")
    print(f"\nsaved diagnostics -> {diag}")


if __name__ == "__main__":
    main()
