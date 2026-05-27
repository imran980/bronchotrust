"""Visualize MAP surface alongside posterior std + residual proxy."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

FIT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 \
    else Path("runs/phase1_5V1/recon_dust3r_foe/ssm_fit_post")
RECON_DIR = FIT_DIR.parent
OUT = FIT_DIR.parent / f"{FIT_DIR.name}_views.png"

data = np.load(FIT_DIR / "ssm_fit.npz")
per_vert_std = data["per_vert_std"]
per_vert_dist = data["per_vert_dist"]
print(f"posterior std (obs units): median={np.median(per_vert_std):.5f}  "
      f"p95={np.percentile(per_vert_std, 95):.5f}  "
      f"max={per_vert_std.max():.5f}")
print(f"nearest-obs dist (obs units): median={np.median(per_vert_dist):.4f}  "
      f"p95={np.percentile(per_vert_dist, 95):.4f}")

# Load fitted surface vertices
ply = PlyData.read(str(FIT_DIR / "mesh_ssm.ply"))
v = np.stack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], 1)

# Subsample for matplotlib
np.random.seed(0)
sub = np.random.choice(len(v), 25000, replace=False)
v_s = v[sub]

# Observations
ply_o = PlyData.read(str(RECON_DIR / "points.ply"))
obs = np.stack([ply_o["vertex"]["x"], ply_o["vertex"]["y"],
                ply_o["vertex"]["z"]], 1)
poses = np.load(RECON_DIR / "poses.npz")["c2w"]
nz = ~np.all(poses[:, :3, 3] == 0, axis=1)
n_per_frame = max(1, len(obs) // len(poses))
per_pt_frame = np.minimum(np.arange(len(obs)) // n_per_frame, len(poses) - 1)
obs = obs[nz[per_pt_frame]]
sub_o = np.random.choice(len(obs), min(6000, len(obs)), replace=False)
obs_s = obs[sub_o]

fig = plt.figure(figsize=(20, 7))
views = [("top-down (x,y)", (90, -90)),
         ("side (x,z)", (0, -90)),
         ("isometric", (30, 45))]
for k, (title, (elev, azim)) in enumerate(views):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    # Color by per-vertex posterior std (proper Bayesian uncertainty)
    sc = ax.scatter(v_s[:, 0], v_s[:, 1], v_s[:, 2],
                    c=per_vert_std[sub], s=2, alpha=0.7, cmap="viridis",
                    vmin=0, vmax=float(np.percentile(per_vert_std, 95)))
    # Overlay observations
    ax.scatter(obs_s[:, 0], obs_s[:, 1], obs_s[:, 2],
               c="gold", s=2, alpha=0.4, label="dust3r-foe obs")
    ax.set_title(f"{title}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if k == 0:
        plt.colorbar(sc, ax=ax, label="posterior std (obs units)",
                     shrink=0.55, pad=0.1)
        ax.legend(fontsize=8, loc="upper left")
plt.suptitle("MAP surface colored by posterior std (N=100 alpha samples) "
             "+ obs (gold)\n"
             f"sigma^2_hat={float(data['sigma2_hat']):.4f}, "
             f"||alpha||={float(np.linalg.norm(data['alpha'])):.2f}",
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
