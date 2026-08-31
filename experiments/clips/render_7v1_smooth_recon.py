"""Standalone cleaned 3D recon of the 7-V1 smooth-run dense cloud. Strip outer splay noise (robust MAD
trim + statistical + radius outlier removal), report shape metrics on the CLEANED core (so the picture
isn't misleading), then render point cloud (side / end-on / oblique) + a Poisson-meshed surface."""
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FP = "runs/own_data/recon_7v1_smooth/dense0/fused.ply"
OUT = "runs/own_data/renders/7v1_smooth_recon.png"
MESH_OUT = "runs/own_data/recon_7v1_smooth/smooth_mesh.ply"

pcd = o3d.io.read_point_cloud(FP)
P0 = np.asarray(pcd.points); n0 = len(P0)

# 1) robust MAD trim for gross outliers
m = np.median(P0, 0); d = np.linalg.norm(P0 - m, axis=1)
keep = d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))
pcd = pcd.select_by_index(np.where(keep)[0])
# 2) statistical outlier removal (sparse splay) + 3) radius outlier removal
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=(np.median(d)) * 0.03)
P = np.asarray(pcd.points); nkept = len(P)


def metrics(Q):
    Qc = Q - Q.mean(0); _, sv, Vt = np.linalg.svd(Qc, full_matrices=False)
    perp = Qc - np.outer(Qc @ Vt[0], Vt[0]); az = np.degrees(np.arctan2(perp @ Vt[2], perp @ Vt[1]))
    b = np.histogram(az, bins=36, range=(-180, 180))[0]
    return sv[0] / sv[1], sv[1] / sv[2], b[np.argsort(b)[::-1][:18]].sum() / b.sum(), Vt


e0, r0, o0, _ = metrics(P0)
el, rr, oo, Vt = metrics(P)
print(f"raw: {n0} pts | elong {e0:.2f} roundness {r0:.2f} one-sided {o0:.2f}")
print(f"cleaned: {nkept} pts ({nkept/n0:.0%} kept) | elong {el:.2f} roundness {rr:.2f} one-sided {oo:.2f}")

# align to PCA frame (axis -> x)
Pc = (P - P.mean(0)) @ Vt.T
sub = Pc[np.random.default_rng(0).choice(len(Pc), min(80000, len(Pc)), replace=False)]

# ---- Poisson mesh of the cleaned cloud ----
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(30)
mesh, dens = o3d.pipelines.surface_reconstruction_poisson(pcd, depth=9) if hasattr(
    o3d.pipelines, "surface_reconstruction_poisson") else o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06))
mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(MESH_OUT, mesh)
V = (np.asarray(mesh.vertices) - P.mean(0)) @ Vt.T
Tr = np.asarray(mesh.triangles)
print(f"mesh: {len(V)} verts {len(Tr)} tris -> {MESH_OUT}")

# ---- figure: cloud (side / end-on / oblique) + mesh (side / end-on) ----
fig = plt.figure(figsize=(16, 9))
def scat(ax, X, Y, C, t):
    ax.scatter(X, Y, s=1, c=C, cmap="viridis"); ax.set_title(t, fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax1 = fig.add_subplot(2, 3, 1); scat(ax1, sub[:, 0], sub[:, 1], sub[:, 0], "cleaned cloud — SIDE")
ax2 = fig.add_subplot(2, 3, 2); scat(ax2, sub[:, 1], sub[:, 2], sub[:, 0], "cleaned cloud — END-ON (down tube)")
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c=sub[:, 0], cmap="viridis"); ax3.set_title("cleaned cloud — OBLIQUE 3D", fontsize=10)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.view_init(elev=18, azim=-60)
# mesh views
tri_x = V[Tr].mean(1)[:, 0]
ax4 = fig.add_subplot(2, 3, 4); ax4.tripcolor(V[:, 0], V[:, 1], Tr, tri_x, cmap="viridis", shading="flat")
ax4.set_title("Poisson mesh — SIDE", fontsize=10); ax4.set_aspect("equal"); ax4.set_xticks([]); ax4.set_yticks([])
ax5 = fig.add_subplot(2, 3, 5); ax5.tripcolor(V[:, 1], V[:, 2], Tr, tri_x, cmap="viridis", shading="flat")
ax5.set_title("Poisson mesh — END-ON", fontsize=10); ax5.set_aspect("equal"); ax5.set_xticks([]); ax5.set_yticks([])
ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
ax6.text(0.02, 0.5, f"7-V1 SMOOTH-RUN cleaned recon\n\nraw {n0:,} pts\ncleaned {nkept:,} pts ({nkept/n0:.0%})\n\n"
         f"elongation s1/s2 = {el:.2f}\nroundness s2/s3 = {rr:.2f}\none-sided(densest-180) = {oo:.2f}\n"
         f"(symmetric tube -> one-sided 0.5, roundness ~1)", fontsize=11, va="center", family="monospace")
fig.tight_layout(); fig.savefig(OUT, dpi=110); print("saved", OUT)
