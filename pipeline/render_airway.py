"""General airway render: fused.ply -> cleaned cloud (side/end-on/oblique) + Poisson mesh (side/end-on),
with post-clean shape metrics. Usage: python render_airway.py <fused.ply> <label> <out.png>"""
import sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FP, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
pcd = o3d.io.read_point_cloud(FP); P0 = np.asarray(pcd.points); n0 = len(P0)
m = np.median(P0, 0); d = np.linalg.norm(P0 - m, axis=1)
pcd = pcd.select_by_index(np.where(d < np.median(d) + 4 * np.median(np.abs(d - np.median(d))))[0])
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=np.median(d) * 0.03)
P = np.asarray(pcd.points); nk = len(P)
Pc = P - P.mean(0); _, sv, Vt = np.linalg.svd(Pc, full_matrices=False)
perp = Pc - np.outer(Pc @ Vt[0], Vt[0]); az = np.degrees(np.arctan2(perp @ Vt[2], perp @ Vt[1]))
b = np.histogram(az, bins=36, range=(-180, 180))[0]
elong, rnd, onesided = sv[0] / sv[1], sv[1] / sv[2], b[np.argsort(b)[::-1][:18]].sum() / b.sum()
print(f"{LABEL}: raw {n0} -> clean {nk} ({nk/n0:.0%}) | elong {elong:.2f} roundness {rnd:.2f} one-sided {onesided:.2f}")
Q = Pc @ Vt.T; sub = Q[np.random.default_rng(0).choice(len(Q), min(80000, len(Q)), replace=False)]

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)); pcd.orient_normals_consistent_tangent_plane(30)
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
dens = np.asarray(dens); mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06)); mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(OUT.replace(".png", "_mesh.ply"), mesh)
V = (np.asarray(mesh.vertices) - P.mean(0)) @ Vt.T; Tr = np.asarray(mesh.triangles)

fig = plt.figure(figsize=(16, 9))
def sc(ax, X, Y, t): ax.scatter(X, Y, s=1, c=X, cmap="viridis"); ax.set_title(t, fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
sc(fig.add_subplot(2, 3, 1), sub[:, 0], sub[:, 1], f"{LABEL} cloud — SIDE")
sc(fig.add_subplot(2, 3, 2), sub[:, 1], sub[:, 2], f"{LABEL} cloud — END-ON")
a3 = fig.add_subplot(2, 3, 3, projection="3d"); a3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c=sub[:, 0], cmap="viridis")
a3.set_title(f"{LABEL} — OBLIQUE 3D", fontsize=10); a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([]); a3.view_init(elev=18, azim=-60)
tx = V[Tr].mean(1)[:, 0]
a4 = fig.add_subplot(2, 3, 4); a4.tripcolor(V[:, 0], V[:, 1], Tr, tx, cmap="viridis", shading="flat"); a4.set_title("Poisson mesh — SIDE", fontsize=10); a4.set_aspect("equal"); a4.set_xticks([]); a4.set_yticks([])
a5 = fig.add_subplot(2, 3, 5); a5.tripcolor(V[:, 1], V[:, 2], Tr, tx, cmap="viridis", shading="flat"); a5.set_title("Poisson mesh — END-ON", fontsize=10); a5.set_aspect("equal"); a5.set_xticks([]); a5.set_yticks([])
a6 = fig.add_subplot(2, 3, 6); a6.axis("off")
a6.text(0.02, 0.5, f"{LABEL} cleaned recon\n\nraw {n0:,} pts\nclean {nk:,} ({nk/n0:.0%})\n\nelong s1/s2 = {elong:.2f}\nroundness s2/s3 = {rnd:.2f}\none-sided = {onesided:.2f}\n(tube: one-sided 0.5, round 1)", fontsize=11, va="center", family="monospace")
fig.tight_layout(); fig.savefig(OUT, dpi=110); print("saved", OUT)
