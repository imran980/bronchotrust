"""Pipeline MVS (geometric/1200) on a workspace's largest connected model + raw geometry view (cross-
sections perpendicular to camera trajectory + Poisson mesh; NO coverage metrics). Args: wsname [outtag]."""
import sys, glob, os, subprocess, numpy as np, pycolmap, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
WSNAME = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else WSNAME
SC = "/tmp/claude-100461304/-home-mi3dr-projects-bronchotrust/32888f63-b0c2-4d41-9ad1-91a5dfc651e6/scratchpad"
WS = Path(f"{SC}/{WSNAME}"); CM = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
os.environ["LD_LIBRARY_PATH"] = "/home/mi3dr/.conda/envs/colmap-cuda/lib"
OUT = Path("runs/own_data/porcine"); OUT.mkdir(parents=True, exist_ok=True)


def sh(c): subprocess.run([CM] + c, capture_output=True, text=True)


best = max([d for d in glob.glob(f"{WS}/sparse/*") if os.path.exists(f"{d}/images.bin")],
           key=lambda d: pycolmap.Reconstruction(d).num_reg_images())
n = pycolmap.Reconstruction(best).num_reg_images(); print(f"MVS on {best} ({n} cams)", flush=True)
D = WS / "dense"; import shutil; shutil.rmtree(D, ignore_errors=True)
sh(["image_undistorter", "--image_path", str(WS / "images"), "--input_path", best, "--output_path", str(D), "--output_type", "COLMAP", "--max_image_size", "1600"])
sh(["patch_match_stereo", "--workspace_path", str(D), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200", "--PatchMatchStereo.gpu_index", "0,1,2,3"])
sh(["stereo_fusion", "--workspace_path", str(D), "--workspace_format", "COLMAP", "--input_type", "geometric", "--StereoFusion.min_num_pixels", "3", "--output_path", str(D / "fused.ply")])

r = pycolmap.Reconstruction(best); C = np.array([np.asarray(im.projection_center()) for im in r.images.values()])
pcd = o3d.io.read_point_cloud(str(D / "fused.ply")); P0 = np.asarray(pcd.points); print(f"DENSE {len(P0):,} pts", flush=True)
o3d.io.write_point_cloud(str(OUT / f"{TAG}_dense.ply"), pcd)
m = np.median(P0, 0); d = np.linalg.norm(P0 - m, axis=1); pc = pcd.select_by_index(np.where(d < np.median(d) + 4 * np.median(np.abs(d - np.median(d))))[0])
pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0); P = np.asarray(pc.points)
Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False); ax = Vt[0]
u = np.cross(ax, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
Pc = P - C.mean(0); pa = Pc @ ax; pu = Pc @ u; pv = Pc @ v; cu = Cc @ u; cv = Cc @ v; ca = Cc @ ax
lo, hi = np.percentile(pa, 5), np.percentile(pa, 95); cuts = np.linspace(lo, hi, 6)[1:-1]; sw = (hi - lo) / 12
fig = plt.figure(figsize=(22, 9))
for k, cut in enumerate(cuts):
    sel = np.abs(pa - cut) < sw; a1 = fig.add_subplot(2, 4, k + 1); a1.scatter(pu[sel], pv[sel], s=3, c='steelblue')
    cs = np.abs(ca - cut) < sw * 2
    if cs.any(): a1.scatter(cu[cs], cv[cs], c='red', s=30, marker='+')
    a1.set_title(f"slice {cut:.1f} ({sel.sum()}pts)"); a1.set_aspect("equal")
try:
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)); pc.orient_normals_consistent_tangent_plane(20)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9); dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.1)); mesh.remove_unreferenced_vertices()
    V = (np.asarray(mesh.vertices) - C.mean(0)) @ Vt.T; T = np.asarray(mesh.triangles)
    a = fig.add_subplot(2, 4, 5); a.tripcolor(V[:, 1], V[:, 2], T, V[:, 0], cmap="viridis", shading="flat"); a.set_title("MESH end-on"); a.set_aspect("equal")
    a = fig.add_subplot(2, 4, 6, projection='3d'); a.plot_trisurf(V[:, 0], V[:, 1], T, V[:, 2], cmap="viridis", linewidth=0); a.set_title("MESH 3D"); a.view_init(elev=12, azim=-70)
    a = fig.add_subplot(2, 4, 7); a.tripcolor(V[:, 0], V[:, 1], T, V[:, 2], cmap="viridis", shading="flat"); a.set_title("MESH side"); a.set_aspect("equal")
except Exception as e: print("mesh fail", e)
a = fig.add_subplot(2, 4, 8); a.scatter(pu[::3], pv[::3], s=1, c='steelblue'); a.plot(cu, cv, 'r+', ms=3); a.set_title("ALL pts end-on"); a.set_aspect("equal")
fig.suptitle(f"{TAG} PIPELINE recon ({n} cams) — {len(P0):,} dense pts | raw cross-sections + mesh", fontsize=13)
fig.tight_layout(); fig.savefig(str(OUT / f"{TAG}_render.png"), dpi=85); print(f"saved {TAG}_render.png + {TAG}_dense.ply", flush=True)
