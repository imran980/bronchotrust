"""Visualize the TSDF-fused mesh from 3 angles + cross-section slice."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

MESH = Path("runs/phase1_5V1/recon_dust3r_foe_tsdf/mesh.ply")
POSES = Path("runs/phase1_5V1/recon_dust3r_foe_tsdf/poses.npz")

ply = PlyData.read(str(MESH))
verts = np.stack([ply["vertex"]["x"], ply["vertex"]["y"],
                  ply["vertex"]["z"]], 1).astype(np.float32)
faces = np.stack([ply["face"]["vertex_indices"]]).flatten()
print(f"mesh: {len(verts)} verts, {len(faces)//3} tris")
print(f"bbox: x[{verts[:,0].min():.2f},{verts[:,0].max():.2f}] "
      f"y[{verts[:,1].min():.2f},{verts[:,1].max():.2f}] "
      f"z[{verts[:,2].min():.2f},{verts[:,2].max():.2f}]")

poses = np.load(POSES)["c2w"]
trans = poses[:, :3, 3]
nz = ~np.all(trans == 0, axis=1)

# Subsample verts for matplotlib (full triangles are too slow)
n_show = 25000
np.random.seed(0)
idx = np.random.choice(len(verts), min(n_show, len(verts)), replace=False)
v = verts[idx]

# Try to color by RGB if available
rgb = None
if "red" in ply["vertex"]._property_lookup:
    rgb = np.stack([ply["vertex"]["red"], ply["vertex"]["green"],
                    ply["vertex"]["blue"]], 1).astype(np.float32) / 255.0
    rgb = rgb[idx]

fig = plt.figure(figsize=(20, 7))
views = [("top-down (x,y)", (90, -90)),
         ("side (x,z)", (0, -90)),
         ("isometric", (30, 45))]
for k, (title, (elev, azim)) in enumerate(views):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    if rgb is not None:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=rgb, s=2, alpha=0.4)
    else:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2],
                   c=v[:, 2], s=2, cmap="viridis", alpha=0.4)
    # Camera trajectory
    tr = trans[nz]
    cn = np.arange(len(trans))[nz]
    for i in range(1, len(tr)):
        ax.plot([tr[i - 1, 0], tr[i, 0]],
                [tr[i - 1, 1], tr[i, 1]],
                [tr[i - 1, 2], tr[i, 2]],
                color=plt.cm.plasma(cn[i] / max(len(trans) - 1, 1)),
                lw=2.5, alpha=0.95)
    ax.scatter(*tr[0], c="lime", s=110, marker="o", edgecolors="black",
               linewidths=1.2, zorder=5, label="start")
    ax.scatter(*tr[-1], c="red", s=110, marker="X", edgecolors="black",
               linewidths=1.2, zorder=5, label="end")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if k == 0:
        ax.legend(loc="upper right", fontsize=8)
plt.suptitle("dust3r-foe + TSDF fusion on REAL 5-V1\n"
             f"({len(verts)} mesh verts, {len(faces)//3} tris; trajectory in plasma)",
             fontsize=13, y=1.02)
plt.tight_layout()
out = Path("runs/phase1_5V1/recon_dust3r_foe_tsdf_views.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
