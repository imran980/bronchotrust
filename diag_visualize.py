"""
Big-image visualization of dust3r-foe reconstructions.

Produces:
  - real_5V1_views.png   - dust3r-foe on real video 5-V1 (where method works)
  - synth_v2_views.png   - dust3r-foe on textured synth_v2 (where it collapses)

Each shows the dense point cloud + camera trajectory from 3 angles
(top-down, side, isometric). Points colored by source-frame index so you
can trace where each frame's contribution landed.

Run from project root: python diag_visualize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def _load_recon(run_dir: Path):
    poses_npz = np.load(run_dir / "poses.npz")
    poses = poses_npz["c2w"]
    ply = PlyData.read(run_dir / "points.ply")
    pts = np.stack([ply["vertex"]["x"], ply["vertex"]["y"],
                    ply["vertex"]["z"]], 1).astype(np.float32)
    if "red" in ply["vertex"]._property_lookup:
        rgb = np.stack([ply["vertex"]["red"], ply["vertex"]["green"],
                        ply["vertex"]["blue"]], 1).astype(np.float32) / 255.0
    else:
        rgb = None
    return poses, pts, rgb


def _3view(fig_title, poses, pts, rgb, out_path,
           n_subsample=10000):
    """Render trajectory + colored point cloud from 3 angles.

    Drops point contributions from dead-pose frames (c2w == identity)
    since those would all project to the world origin and dominate."""
    trans = poses[:, :3, 3]
    nz = ~np.all(trans == 0, axis=1)
    N = len(poses)

    # Per-point source-frame index assuming the runner concatenated
    # pointmaps in order with roughly equal counts per frame.
    n_per_frame = max(1, len(pts) // N)
    per_pt_frame = np.minimum(np.arange(len(pts)) // n_per_frame, N - 1)
    valid_pts_mask = nz[per_pt_frame]
    n_dropped = int((~valid_pts_mask).sum())
    pts = pts[valid_pts_mask]
    if rgb is not None:
        rgb = rgb[valid_pts_mask]
    per_pt_frame = per_pt_frame[valid_pts_mask]
    if n_dropped > 0:
        print(f"  dropped {n_dropped} pts from dead-pose frames; "
              f"{len(pts)} remaining")

    if len(pts) > n_subsample:
        np.random.seed(0)
        idx = np.random.choice(len(pts), n_subsample, replace=False)
        pts_s = pts[idx]
        rgb_s = rgb[idx] if rgb is not None else None
        per_pt_frame_s = per_pt_frame[idx]
    else:
        pts_s = pts
        rgb_s = rgb
        per_pt_frame_s = per_pt_frame

    fig = plt.figure(figsize=(20, 7))
    views = [("top-down (x,y)", (90, -90)),
             ("side (x,z)", (0, -90)),
             ("isometric", (30, 45))]
    for k, (title, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        # Point cloud — RGB if available, else by frame index
        if rgb_s is not None:
            ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                       c=rgb_s, s=1, alpha=0.5)
        else:
            ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                       c=per_pt_frame_s, s=1, alpha=0.5, cmap="viridis")
        # Trajectory as a line through valid (nonzero) poses
        if nz.sum() >= 2:
            tr = trans[nz]
            cn = np.arange(N)[nz]
            for i in range(1, len(tr)):
                ax.plot([tr[i - 1, 0], tr[i, 0]],
                        [tr[i - 1, 1], tr[i, 1]],
                        [tr[i - 1, 2], tr[i, 2]],
                        color=plt.cm.plasma(cn[i] / max(N - 1, 1)),
                        lw=2.2, alpha=0.95)
            ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2],
                       c=cn, s=18, cmap="plasma",
                       edgecolors="black", linewidths=0.4)
            ax.scatter(*tr[0], c="lime", s=110, marker="o",
                       edgecolors="black", linewidths=1.2,
                       label="start", zorder=5)
            ax.scatter(*tr[-1], c="red", s=110, marker="X",
                       edgecolors="black", linewidths=1.2,
                       label="end", zorder=5)
        # Dead poses (at origin) — mark them
        if (~nz).any():
            ax.scatter(0, 0, 0, c="black", s=80, marker="s",
                       label=f"{(~nz).sum()} dead poses @ origin", zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.suptitle(fig_title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Real video 5-V1 (dust3r-foe worked here) ----

real = Path("runs/phase1_5V1/recon_dust3r_foe")
if real.exists():
    poses, pts, rgb = _load_recon(real)
    print(f"\n[real 5-V1] poses={poses.shape[0]}  pts={len(pts)}")
    nz = ~np.all(poses[:, :3, 3] == 0, axis=1)
    print(f"  valid poses: {nz.sum()}/{len(poses)}  "
          f"span={float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])):.3f}")
    _3view("dust3r-foe on REAL 5-V1 (Phase 1 win: 0 direction reversals, "
           "+0.66 mean step cosine)",
           poses, pts, rgb, Path("runs/phase1_5V1/recon_dust3r_foe_views.png"))

# ---- Synth_v2 (where it collapses) ----

synth = Path("runs/synth_v2/recon_dust3r_foe")
if synth.exists():
    poses, pts, rgb = _load_recon(synth)
    print(f"\n[synth_v2] poses={poses.shape[0]}  pts={len(pts)}")
    nz = ~np.all(poses[:, :3, 3] == 0, axis=1)
    print(f"  valid poses: {nz.sum()}/{len(poses)}  "
          f"span={float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])):.3f}")
    _3view("dust3r-foe on SYNTH_v2 (collapsed: 12/28 dead poses, cluster blob)",
           poses, pts, rgb, Path("runs/synth_v2/recon_dust3r_foe_views.png"))

# ---- Baseline dust3r on REAL 5-V1 for contrast ----

baseline = Path("runs/phase1_5V1/recon_dust3r")
if baseline.exists():
    poses, pts, rgb = _load_recon(baseline)
    print(f"\n[real 5-V1 baseline] poses={poses.shape[0]}  pts={len(pts)}")
    _3view("dust3r BASELINE on REAL 5-V1 (Phase 1: 19/48 reversals, zigzag 22x)",
           poses, pts, rgb, Path("runs/phase1_5V1/recon_dust3r_baseline_views.png"))

print("\nDone. View the *_views.png files.")
