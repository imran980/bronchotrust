"""Robust raw-geometry view of a dense cloud vs its camera trajectory. Judges ring-vs-funnel.
Cross-sections are taken PERPENDICULAR to the camera-trajectory axis (PCA of camera centers),
NOT PCA-of-points (which lies for one-sided clouds). No coverage metrics. Everything guarded.
Args: dense.ply  sparse_model_dir  out.png  TAG"""
import sys, numpy as np, pycolmap, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PLY, SPARSE, OUT, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
r = pycolmap.Reconstruction(SPARSE)
C = np.array([np.asarray(im.projection_center()) for im in r.images.values()])
pcd = o3d.io.read_point_cloud(PLY); P0 = np.asarray(pcd.points)
print(f"{TAG}: {len(P0):,} dense pts, {len(C)} cams", flush=True)

# outlier trim around median
m = np.median(P0, 0); d = np.linalg.norm(P0 - m, axis=1)
keep = d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))
pc = pcd.select_by_index(np.where(keep)[0])
try:
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
except Exception as e:
    print("  stat-outlier skip", e)
P = np.asarray(pc.points)

# axis = principal direction of the CAMERA TRAJECTORY (not the points)
Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False); ax = Vt[0]
ref = np.eye(3)[np.argmin(np.abs(ax))]                 # robust perpendicular basis (no [0,0,1] degeneracy)
u = np.cross(ax, ref); u /= np.linalg.norm(u); v = np.cross(ax, u)
Pc = P - C.mean(0); pa = Pc @ ax; pu = Pc @ u; pv = Pc @ v
ca = Cc @ ax; cu = Cc @ u; cv = Cc @ v
travel = ca.max() - ca.min(); depth = np.median(np.linalg.norm(Pc - np.outer(pa, ax), axis=1))
print(f"  cam axial travel {travel:.2f}  median lumen radius {depth:.2f}  travel/radius {travel/max(depth,1e-6):.2f}", flush=True)

lo, hi = np.percentile(pa, 4), np.percentile(pa, 96); cuts = np.linspace(lo, hi, 7)[1:-1]; sw = (hi - lo) / 14
fig = plt.figure(figsize=(22, 9))
for k, cut in enumerate(cuts):
    sel = np.abs(pa - cut) < sw; a1 = fig.add_subplot(2, 5, k + 1)
    a1.scatter(pu[sel], pv[sel], s=3, c='steelblue')
    cs = np.abs(ca - cut) < sw * 2
    if cs.any(): a1.scatter(cu[cs], cv[cs], c='red', s=40, marker='+')
    a1.set_title(f"slice a={cut:.1f} ({sel.sum()}pts)", fontsize=9); a1.set_aspect("equal"); a1.grid(alpha=.2)
# side view (along axis) + end-on all points with camera path
a = fig.add_subplot(2, 5, 6); a.scatter(pa[::3], pu[::3], s=1, c='steelblue'); a.plot(ca, cu, 'r.-', ms=2, lw=.5)
a.set_title("SIDE (axis=x): tube=band, funnel=wedge", fontsize=9); a.set_aspect("equal"); a.grid(alpha=.2)
a = fig.add_subplot(2, 5, 7); a.scatter(pu[::3], pv[::3], s=1, c='steelblue'); a.plot(cu, cv, 'r+', ms=4)
a.set_title("END-ON all pts (+=cams)", fontsize=9); a.set_aspect("equal"); a.grid(alpha=.2)
try:
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)); pc.orient_normals_consistent_tangent_plane(20)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9); dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.1)); mesh.remove_unreferenced_vertices()
    V = (np.asarray(mesh.vertices) - C.mean(0)) @ Vt.T; T = np.asarray(mesh.triangles)
    a = fig.add_subplot(2, 5, 8); a.tripcolor(V[:, 1], V[:, 2], T, V[:, 0], cmap="viridis", shading="flat"); a.set_title("MESH end-on"); a.set_aspect("equal")
    a = fig.add_subplot(2, 5, 9, projection='3d'); a.plot_trisurf(V[:, 0], V[:, 1], T, V[:, 2], cmap="viridis", linewidth=0); a.set_title("MESH 3D"); a.view_init(elev=12, azim=-70)
    a = fig.add_subplot(2, 5, 10); a.tripcolor(V[:, 0], V[:, 1], T, V[:, 2], cmap="viridis", shading="flat"); a.set_title("MESH side"); a.set_aspect("equal")
except Exception as e:
    print("  mesh fail", e)
fig.suptitle(f"{TAG} — {len(P0):,} dense pts, {len(C)} cams | travel/radius {travel/max(depth,1e-6):.2f} "
             f"(>~1.7 = tube-like)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT, dpi=85); print(f"saved {OUT}", flush=True)
