"""General partial-arc CSA on a dense fused cloud. Usage: python csa_run.py <fused.ply> <LABEL> <out.png>"""
import sys, numpy as np, open3d as o3d
from csa_partialarc import profile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FP, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P = np.asarray(o3d.io.read_point_cloud(FP).points)
m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=np.median(d) * 0.03)
P = np.asarray(pcd.points)
rows, cl = profile(P, nb=60)
good = [r for r in rows if r["cov"] >= 0.55 and r["resid"] < 0.15]
print(f"{LABEL}: {len(P)} clean pts | {len(rows)} stations, {len(good)} valid")
if good:
    csa = np.array([r["csa_circ"] for r in good]); ce = np.array([r["csa_ell"] for r in good])
    agree = float(np.nanmedian(np.abs(ce - csa) / csa))
    cs = csa[np.argsort([r["t"] for r in good])]; imin = cs.argmin(); interior = 0 < imin < len(cs) - 1
    print(f"  circle-vs-ellipse {agree:.0%} | CSA min={csa.min():.3f} max={csa.max():.3f} ratio={csa.min()/csa.max():.3f} "
          f"| {'LOCAL MIN (candidate narrowing)' if interior else 'monotonic (funnel/normal caliber)'}")
zt = np.array([r["t"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows]); cov = np.array([r["cov"] for r in rows])
fig, ax = plt.subplots(figsize=(9, 4.5)); sc = ax.scatter(zt, cc, c=cov, cmap="viridis", vmin=.3, vmax=1, s=26)
ax.plot(zt, cc, "-", color=".6", lw=.7, zorder=0); fig.colorbar(sc, label="arc coverage")
ax.set_xlabel("along-axis station (scene units)"); ax.set_ylabel("CSA (scene units^2)"); ax.set_title(f"{LABEL} partial-arc CSA profile"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT, dpi=120); print("saved", OUT)
