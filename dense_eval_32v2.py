"""Strict Barbour-style dense geometry-quality eval of the FUSED 32-V2 model.
Starts from runs/fuse_32v2/sparse/0. SAME MVS settings as the successful 7-V1 run
(image_undistorter 1600 -> patch_match geom_consistency max1200 -> stereo_fusion
geometric -> Poisson depth10 trim7). Adds Barbour double-Poisson. Lumen-axis
cross-sections EXACTLY as phantom / corrected 7-V1 (cloud PC1 + per-bin centroid
centerline). Compares 32-V2 vs 7-V1 vs phantom. NO scale / CSA / %obstruction.
Run in depth-eval env. MVS skipped if dense0/fused.ply already present."""
from __future__ import annotations
import json, re, subprocess, shutil
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
if not hasattr(np, "in1d"): np.in1d = np.isin
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OD = ROOT / "runs/fuse_32v2"
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
DIAG = OD / "diag"; DIAG.mkdir(parents=True, exist_ok=True)
COV_MIN, RATIO_MAX = 0.60, 0.35     # closed-ring criterion


def run(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  COLMAP ERR", cmd[0], r.stderr[-300:])
    return r


def ply_counts(path):
    v = f = 0
    with open(path, "rb") as fh:
        for _ in range(60):
            ln = fh.readline().decode("latin1", "ignore")
            if ln.startswith("element vertex"): v = int(ln.split()[-1])
            elif ln.startswith("element face"): f = int(ln.split()[-1])
            elif ln.startswith("end_header"): break
    return v, f


def mvs():
    dense = OD / "dense0"; fused = dense / "fused.ply"
    if fused.exists():
        print(f"MVS: dense0/fused.ply exists ({ply_counts(fused)[0]} pts) -> skip"); return fused
    sparse0 = OD / "sparse/0"
    run(["image_undistorter", "--image_path", str(OD / "images"), "--input_path", str(sparse0),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"])
    run(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
    return fused


def meshes(fused):
    # single Poisson via COLMAP (apples-to-apples with 7-V1/phantom)
    sp = OD / "dense0/meshed-poisson.ply"
    if not sp.exists():
        run(["poisson_mesher", "--input_path", str(fused), "--output_path", str(sp),
             "--PoissonMeshing.depth", "10", "--PoissonMeshing.trim", "7"])
    sv, sf = ply_counts(sp) if sp.exists() else (0, 0)
    # Barbour double-Poisson (open3d): Poisson -> Taubin -> resample surface + filtered cloud -> Poisson
    pcd = o3d.io.read_point_cloud(str(fused))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    m1, d1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    d1 = np.asarray(d1); m1.remove_vertices_by_index(np.where(d1 < np.quantile(d1, 0.05))[0].tolist())
    m1 = m1.filter_smooth_taubin(number_of_iterations=20); m1.compute_vertex_normals()
    samp = m1.sample_points_poisson_disk(number_of_points=120000)
    combo = pcd + samp
    m2, d2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combo, depth=9)
    d2 = np.asarray(d2); m2.remove_vertices_by_index(np.where(d2 < np.quantile(d2, 0.03))[0].tolist())
    m2 = m2.filter_smooth_taubin(number_of_iterations=15)
    m2.remove_degenerate_triangles(); m2.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(DIAG / "double_poisson.ply"), m2)
    dv, df = len(m2.vertices), len(m2.triangles)
    return {"poisson_verts": sv, "poisson_faces": sf, "dpoisson_verts": dv, "dpoisson_faces": df,
            "dpoisson_watertight": bool(m2.is_watertight()), "mesh_obj": m2}


def cam_centers(rec):
    out = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix())
        out[int(re.search(r"f(\d+)", im.name).group(1))] = -M[:3, :3].T @ M[:3, 3]
    return out


def lumen_axis(P, cam_centroid):
    Pc = P - P.mean(0)
    idx = np.random.default_rng(0).choice(len(Pc), min(len(Pc), 60000), replace=False)
    _, _, Vt = np.linalg.svd(Pc[idx], full_matrices=False); axis = Vt[0]
    if (P.mean(0) - cam_centroid) @ axis < 0: axis = -axis
    s = (P - P.mean(0)) @ axis
    edges = np.linspace(np.percentile(s, 1), np.percentile(s, 99), 41)
    cen = [P[(s >= edges[i]) & (s < edges[i + 1])].mean(0) for i in range(40)
           if ((s >= edges[i]) & (s < edges[i + 1])).sum() > 20]
    cen = np.array(cen)
    tck, _ = splprep(cen.T, s=0.3 * len(cen), k=3)
    uf = np.linspace(0, 1, 2000); pf = np.array(splev(uf, tck)).T
    scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    if np.linalg.norm(pf[0] - cam_centroid) > np.linalg.norm(pf[-1] - cam_centroid):
        uf = uf[::-1]; pf = pf[::-1]; scum = scum[-1] - scum[::-1]
    return tck, uf, pf, scum


def lumen_sections(fused, rec):
    P = np.asarray(o3d.io.read_point_cloud(str(fused)).points)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    C = np.array(list(cam_centers(rec).values())); cam_centroid = C.mean(0)
    tck, uf, pf, scum = lumen_axis(P, cam_centroid); L = scum[-1]
    s_nodes = np.linspace(0.03 * L, 0.97 * L, 10)
    u_n = np.interp(s_nodes, scum, uf)
    Pn = np.array(splev(u_n, tck)).T
    Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
    tree = cKDTree(P); Rmed = np.median(cKDTree(pf).query(P)[0])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5)); rep = []
    for k, (p0, tg) in enumerate(zip(Pn, Tn)):
        ax = axes.flat[k]
        rel = P[tree.query_ball_point(p0, 6 * Rmed)] - p0
        ins = rel[np.abs(rel @ tg) <= 0.4]
        up = np.array([0, 0, 1.]) if abs(tg[2]) < .95 else np.array([0, 1., 0])
        e1 = up - (up @ tg) * tg; e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(tg, e1)
        if len(ins):
            ip = np.stack([ins @ e1, ins @ e2], 1)
            th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
            cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
            rmed = float(np.median(r)); rstd = float(np.std(r))
            ax.scatter(ip[:, 0], ip[:, 1], s=1.5, c="0.4", alpha=.5)
        else:
            ip = np.zeros((0, 2)); cov = rmed = rstd = 0.0
        ratio = rstd / max(rmed, 1e-6)
        closed = (cov >= COV_MIN) and (ratio <= RATIO_MAX)
        ax.plot(0, 0, "r+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
        end = "SUBGLOTTIS" if k <= 2 else ("TRACHEA" if k >= 7 else "mid")
        ax.set_title(f"node{k} [{end}] n={len(ip)}\ncov={cov*100:.0f}% r_med={rmed:.2f} "
                     f"r_std/r_med={ratio:.2f}\n{'CLOSED' if closed else 'open/noisy'}",
                     fontsize=8, color=("green" if closed else "crimson"))
        rep.append({"node": k, "n": len(ip), "coverage": cov, "r_median": rmed,
                    "r_std": rstd, "ratio": ratio, "closed": closed})
    fig.suptitle("32-V2 cross-sections PERPENDICULAR TO LUMEN AXIS (gray=wall pts, +=center)  "
                 "calib=32_v1 BORROWED\nnode0=subglottis end; CLOSED = cov>=60% & r_std/r_med<=0.35", fontsize=11)
    fig.tight_layout(); fig.savefig(DIAG / "lumen_slices.png", dpi=130); plt.close(fig)
    # centerline + dense cloud render
    Pp = P[np.random.default_rng(0).choice(len(P), min(len(P), 90000), replace=False)]
    fig, axx = plt.subplots(1, 3, figsize=(18, 6))
    for a, (ix, iy, lx, ly) in zip(axx, [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        a.scatter(Pp[:, ix], Pp[:, iy], s=.4, c="0.6", alpha=.5)
        a.plot(pf[:, ix], pf[:, iy], "-", c="blue", lw=2, label="lumen centerline")
        a.scatter(C[:, ix], C[:, iy], s=6, c="red", label="cameras")
        a.set_title(f"{lx}{ly}"); a.set_aspect("equal", "datalim"); a.grid(alpha=.3)
    axx[0].legend(fontsize=8)
    fig.suptitle("32-V2 dense cloud + LUMEN centerline (blue) vs cameras (red)")
    fig.tight_layout(); fig.savefig(DIAG / "dense_cloud.png", dpi=130); plt.close(fig)
    return {"L": float(L), "Rmed": float(Rmed), "nodes": rep}


def mesh_render(mesh):
    V = np.asarray(mesh.vertices)
    if not len(V): return
    fig, axx = plt.subplots(1, 3, figsize=(18, 6))
    for a, (ix, iy, lx, ly) in zip(axx, [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        a.scatter(V[:, ix], V[:, iy], s=.5, c=V[:, iy], cmap="viridis", alpha=.5)
        a.set_title(f"{lx}{ly}"); a.set_aspect("equal", "datalim"); a.grid(alpha=.3)
    fig.suptitle(f"32-V2 double-Poisson mesh ({len(V)} verts, watertight={mesh.is_watertight()})")
    fig.tight_layout(); fig.savefig(DIAG / "mesh_render.png", dpi=130); plt.close(fig)


def summarize(rep):
    nd = rep["nodes"]
    cov = np.array([n["coverage"] for n in nd])
    ratio = np.array([n.get("ratio", n["r_std"] / max(n["r_median"], 1e-6)) for n in nd])
    closed = sum(1 for n, rt in zip(nd, ratio) if (n["coverage"] >= COV_MIN and rt <= RATIO_MAX))
    return {"median_coverage": round(float(np.median(cov)) * 100, 1),
            "median_ratio": round(float(np.median(ratio)), 3),
            "closed_rings": int(closed)}


def load_compare(report_path, n_reg, fused_path, mesh_path):
    """build comparison row from an existing lumen_report.json (7-V1 / phantom)."""
    rep = json.loads(Path(report_path).read_text())
    s = summarize(rep)
    dp = ply_counts(fused_path)[0] if Path(fused_path).exists() else 0
    mv = ply_counts(mesh_path)[0] if Path(mesh_path).exists() else 0
    return {"reg": n_reg, "dense": dp, "mesh_verts": mv, **s}


def main():
    fused = mvs()
    dpts = ply_counts(fused)[0]
    print(f"\nDENSE: {dpts} fused points")
    mz = meshes(fused)
    print(f"MESH: single-Poisson {mz['poisson_verts']}v/{mz['poisson_faces']}f ; "
          f"double-Poisson {mz['dpoisson_verts']}v/{mz['dpoisson_faces']}f watertight={mz['dpoisson_watertight']}")
    rec = pycolmap.Reconstruction(str(OD / "sparse/0"))
    rep = lumen_sections(fused, rec); mesh_render(mz["mesh_obj"])
    s32 = summarize(rep)
    (DIAG / "lumen_report.json").write_text(json.dumps(rep, indent=2))

    row32 = {"reg": rec.num_reg_images(), "dense": dpts, "mesh_verts": mz["poisson_verts"], **s32}
    row7 = load_compare(ROOT / "runs/barbour_dense/7-V1/diag/lumen_report.json", 302,
                        ROOT / "runs/barbour_dense/7-V1/dense/fused.ply",
                        ROOT / "runs/barbour_dense/7-V1/dense/meshed-poisson.ply")
    rowp = load_compare(ROOT / "runs/phantom/diag/lumen_report.json", 350,
                        ROOT / "runs/phantom/dense0/fused.ply",
                        ROOT / "runs/phantom/dense0/meshed-poisson.ply")

    tbl = {"32-V2": row32, "7-V1": row7, "phantom": rowp,
           "double_poisson": {k: mz[k] for k in ["dpoisson_verts", "dpoisson_faces", "dpoisson_watertight"]}}
    (DIAG / "compare.json").write_text(json.dumps(tbl, indent=2))
    def fmt(r): return (f"{r['reg']:>10}{r['dense']:>13}{r['mesh_verts']:>12}"
                        f"{r['median_coverage']:>11}{r['median_ratio']:>11}{r['closed_rings']:>9}/10")
    print("\n==================== GEOMETRY-QUALITY COMPARISON ====================")
    print(f"{'metric':<20}{'reg':>10}{'dense':>13}{'mesh_verts':>12}{'med_cov%':>11}{'med_ratio':>11}{'closed':>11}")
    print(f"{'32-V2':<20}{fmt(row32)}")
    print(f"{'7-V1':<20}{fmt(row7)}")
    print(f"{'phantom':<20}{fmt(rowp)}")
    print(f"\n  (med_ratio = median r_std/r_med across 10 nodes; lower=thinner ring. phantom~0.07 ideal)")
    print(f"  double-Poisson 32-V2: {mz['dpoisson_verts']}v watertight={mz['dpoisson_watertight']}")
    print(f"\nfigures -> {DIAG}/ : lumen_slices.png dense_cloud.png mesh_render.png ; trajectory.png in {OD}")


if __name__ == "__main__":
    main()
