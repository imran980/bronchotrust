"""Visualize SSM-fit result alongside dust3r-foe observation patches."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

import sys
FIT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/phase1_5V1/recon_dust3r_foe/ssm_fit")
RECON_DIR = Path("runs/phase1_5V1/recon_dust3r_foe")
OUT = FIT_DIR.parent / f"{FIT_DIR.name}_views.png"

# Fitted SSM
m = PlyData.read(str(FIT_DIR / "mesh_ssm.ply"))
v = np.stack([m["vertex"]["x"], m["vertex"]["y"], m["vertex"]["z"]], 1)
vc = np.stack([m["vertex"]["red"], m["vertex"]["green"],
               m["vertex"]["blue"]], 1) / 255.0
# Subsample for display
np.random.seed(0)
sub_v = np.random.choice(len(v), 25000, replace=False)
v_s = v[sub_v]
vc_s = vc[sub_v]

# Observations
o = PlyData.read(str(RECON_DIR / "points.ply"))
obs = np.stack([o["vertex"]["x"], o["vertex"]["y"], o["vertex"]["z"]], 1)
# Drop dead-pose contributions
poses = np.load(RECON_DIR / "poses.npz")["c2w"]
nz = ~np.all(poses[:, :3, 3] == 0, axis=1)
n_per_frame = max(1, len(obs) // len(poses))
per_pt_frame = np.minimum(np.arange(len(obs)) // n_per_frame, len(poses) - 1)
obs = obs[nz[per_pt_frame]]
sub_o = np.random.choice(len(obs), min(8000, len(obs)), replace=False)
obs_s = obs[sub_o]

# Per-vertex residual stats
per_vert = np.load(FIT_DIR / "ssm_fit.npz")["per_vert_dist"]
print(f"per-vert dist: median={np.median(per_vert):.3f}  "
      f"p25={np.percentile(per_vert, 25):.3f}  "
      f"p75={np.percentile(per_vert, 75):.3f}  "
      f"p95={np.percentile(per_vert, 95):.3f}")

views = [("top-down (x,y)", (90, -90)),
         ("side (x,z)", (0, -90)),
         ("isometric", (30, 45))]
fig = plt.figure(figsize=(20, 13))
# Row 1: SSM only (colored by residual: red=poor coverage, green=well-observed)
for k, (title, (elev, azim)) in enumerate(views):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")
    ax.scatter(v_s[:, 0], v_s[:, 1], v_s[:, 2], c=vc_s, s=2, alpha=0.6)
    ax.set_title(f"SSM fitted surface — {title}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
# Row 2: SSM overlaid with observations (yellow obs on gray mesh)
for k, (title, (elev, azim)) in enumerate(views):
    ax = fig.add_subplot(2, 3, k + 4, projection="3d")
    ax.scatter(v_s[:, 0], v_s[:, 1], v_s[:, 2],
               c="lightgray", s=1, alpha=0.25)
    ax.scatter(obs_s[:, 0], obs_s[:, 1], obs_s[:, 2],
               c="gold", s=3, alpha=0.7, label="dust3r-foe obs")
    ax.set_title(f"SSM (gray) + obs (yellow) — {title}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if k == 0:
        ax.legend(fontsize=8)
plt.suptitle(
    f"SSM PCA fit to dust3r-foe observations (REAL 5-V1)\n"
    f"top row: SSM colored red=high residual (no obs nearby) / green=low (well-observed)\n"
    f"bottom row: yellow observations overlaid on gray SSM mesh",
    fontsize=11, y=1.0)
plt.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
