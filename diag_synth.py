"""
Visual diagnostics for synth_v2 reconstructions.

Four PNGs:
  1. trajectory_overlay.png — all available backbones' camera trajectories
     overlaid on the GT scope path, each Umeyama-aligned to GT.
  2. cloud_vs_gtmesh.png — dust3r-foe point cloud rendered on top of the
     GT airway interior surface, aligned to GT.
  3. dead_keyframes.png — montage of the original synth color frames at
     the indices that produced (0,0,0) poses, to see if there's a
     visual pattern.
  4. worst_failure_frame.png — per-keyframe mean surface-error bar chart
     plus the synth color frame for the worst keyframe.

Run from project root: python diag_synth.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


RUN = Path("runs/synth_v2")
OUT = RUN / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
# Derive N_RECON from whatever the runner actually produced. With the
# tighter min_valid_pixels filter the input frame count can drop below 50.
_d = np.load(RUN / "recon_dust3r_foe" / "poses.npz")
N_RECON = int(_d["c2w"].shape[0])


def _umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    cs = src - mu_s; cd = dst - mu_d
    H = cs.T @ cd / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    var_s = (cs ** 2).sum() / len(src)
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src.T)).T + t
    return s, R, t, aligned


def _gt_poses():
    raw = np.loadtxt(RUN / "pose_gt.txt", delimiter=",")
    poses = raw.reshape(-1, 4, 4).transpose(0, 2, 1)
    sub = np.linspace(0, len(poses) - 1, N_RECON).round().astype(int)
    return poses, sub


def _load_recon(sub_dir):
    p = RUN / sub_dir / "poses.npz"
    if not p.exists():
        return None
    return np.load(p)["c2w"][:, :3, 3]


METHODS = [
    ("recon_mast3r", "mast3r-slam", "tab:gray"),
    ("recon_dust3r", "dust3r baseline", "tab:blue"),
    ("recon_dust3r_smooth", "dust3r-smooth", "tab:orange"),
    ("recon_dust3r_foe", "dust3r-foe (ours)", "tab:red"),
    ("recon_colmap", "colmap", "tab:green"),
]


def plot_trajectory_overlay():
    gt_poses, sub = _gt_poses()
    gt_trans = gt_poses[sub, :3, 3]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # GT first
    ax1.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2],
             "k-", lw=3, label="GT scope path", alpha=0.8)
    ax1.scatter(gt_trans[0, 0], gt_trans[0, 1], gt_trans[0, 2],
                c="k", s=80, marker="o", label="start")
    ax1.scatter(gt_trans[-1, 0], gt_trans[-1, 1], gt_trans[-1, 2],
                c="k", s=80, marker="X", label="end")

    for sub_dir, label, color in METHODS:
        rec = _load_recon(sub_dir)
        if rec is None:
            continue
        nz = ~np.all(rec == 0, axis=1)
        if nz.sum() < 3:
            continue
        s, R, t, aligned = _umeyama(rec[nz], gt_trans[nz])
        full_aligned = (s * (R @ rec.T)).T + t  # transform all incl. zeros
        ax1.plot(full_aligned[nz][:, 0], full_aligned[nz][:, 1],
                 full_aligned[nz][:, 2], "-", color=color,
                 label=f"{label}  ({nz.sum()}/{len(rec)})", lw=1.5, alpha=0.85)
    ax1.set_title("camera trajectories (Umeyama-aligned to GT)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Side-by-side: GT alone, then dust3r-foe alone, then a 2D top-down
    rec = _load_recon("recon_dust3r_foe")
    nz = ~np.all(rec == 0, axis=1)
    s, R, t, aligned = _umeyama(rec[nz], gt_trans[nz])
    full_aligned = (s * (R @ rec.T)).T + t
    ax2.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2],
             "k-", lw=3, label="GT", alpha=0.6)
    ax2.scatter(full_aligned[nz, 0], full_aligned[nz, 1], full_aligned[nz, 2],
                c="tab:red", s=20, label="dust3r-foe", alpha=0.85)
    # Connect aligned points by index order (color by index)
    for i in range(1, len(full_aligned)):
        if nz[i] and nz[i - 1]:
            ax2.plot([full_aligned[i - 1, 0], full_aligned[i, 0]],
                     [full_aligned[i - 1, 1], full_aligned[i, 1]],
                     [full_aligned[i - 1, 2], full_aligned[i, 2]],
                     color="tab:red", lw=0.8, alpha=0.5)
    ax2.set_title("dust3r-foe trajectory vs GT (close-up)")
    ax2.legend(); ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    p = OUT / "1_trajectory_overlay.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def plot_cloud_vs_gt_mesh():
    """Render dust3r-foe point cloud overlaid on the GT SSM-mean mesh,
    using matplotlib 3D. (Open3D's offscreen renderer is finicky; this
    keeps things simple.) Subsample the cloud to 8k points for speed."""
    with h5py.File("atm22_corpus.h5", "r") as f:
        gt_verts = f["metadata/ssm_surface/mean_shape"][:]

    gt_poses, sub = _gt_poses()
    gt_trans = gt_poses[sub, :3, 3]

    rec = _load_recon("recon_dust3r_foe")
    nz = ~np.all(rec == 0, axis=1)
    s, R, t, _ = _umeyama(rec[nz], gt_trans[nz])

    ply = PlyData.read(RUN / "recon_dust3r_foe" / "points.ply")
    pts = np.stack([ply["vertex"]["x"], ply["vertex"]["y"],
                    ply["vertex"]["z"]], 1)
    pts_aln = (s * (R @ pts.T)).T + t

    # Compute per-point error against GT mesh for color-coding
    tree = cKDTree(gt_verts)
    np.random.seed(0)
    idx = np.random.choice(len(pts_aln), min(8000, len(pts_aln)), replace=False)
    pts_sub = pts_aln[idx]
    dists, _ = tree.query(pts_sub, k=1)

    # GT mesh: subsample to 4k vertices
    gv_idx = np.random.choice(len(gt_verts),
                              min(4000, len(gt_verts)), replace=False)
    gv = gt_verts[gv_idx]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(gv[:, 0], gv[:, 1], gv[:, 2],
                c="lightgray", s=1, alpha=0.3, label="GT mesh (4 k)")
    sc = ax1.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                     c=dists, s=2, cmap="viridis", vmin=0, vmax=20,
                     label="dust3r-foe (8 k)")
    plt.colorbar(sc, ax=ax1, label="surface error (mm)", shrink=0.6)
    ax1.set_title("dust3r-foe point cloud vs GT airway mesh")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Same view but mesh-only + trajectory
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(gv[:, 0], gv[:, 1], gv[:, 2],
                c="lightgray", s=1, alpha=0.4)
    ax2.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2],
             "k-", lw=2, label="GT scope path")
    # Color points by their FRAME index, not by error — see if structure is recovered
    pts_aln_full = pts_aln
    per_pt_frame = np.repeat(np.arange(N_RECON), len(pts_aln_full) // N_RECON
                             + 1)[:len(pts_aln_full)]
    idx2 = np.random.choice(len(pts_aln_full),
                            min(8000, len(pts_aln_full)), replace=False)
    sc2 = ax2.scatter(pts_aln_full[idx2, 0], pts_aln_full[idx2, 1],
                      pts_aln_full[idx2, 2],
                      c=per_pt_frame[idx2], s=2, cmap="plasma",
                      label="dust3r-foe (by frame idx)")
    plt.colorbar(sc2, ax=ax2, label="frame index", shrink=0.6)
    ax2.set_title("dust3r-foe colored by source-frame index")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    p = OUT / "2_cloud_vs_gtmesh.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def plot_dead_keyframes():
    """Show synth color frames at the indices that returned (0,0,0)."""
    rec = _load_recon("recon_dust3r_foe")
    zero_idx = np.where(np.all(rec == 0, axis=1))[0]
    # Map subsampled index -> original frame number (1-indexed PNGs)
    n_total_frames = len(sorted((RUN / "color").glob("*.png")))
    full_idx = np.linspace(0, n_total_frames - 1, N_RECON).round().astype(int)
    orig_frame_nums = full_idx[zero_idx] + 1  # 1-indexed filenames

    if len(orig_frame_nums) == 0:
        print("  no dead keyframes — skipping")
        return

    n = len(orig_frame_nums)
    cols = min(7, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes).flatten()
    for k, fn in enumerate(orig_frame_nums):
        img = np.array(Image.open(RUN / "color" / f"{fn:06d}.png"))
        axes[k].imshow(img)
        axes[k].set_title(f"recon idx {zero_idx[k]}\norig frame {fn}",
                          fontsize=8)
        axes[k].axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    plt.suptitle(f"{n} dead keyframes (poses stayed at origin) — looking for "
                 f"low-texture / bifurcation pattern", fontsize=10)
    plt.tight_layout()
    p = OUT / "3_dead_keyframes.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}  ({n} dead frames)")
    print(f"    recon idxs: {zero_idx.tolist()}")
    print(f"    orig frame nums: {orig_frame_nums.tolist()}")


def plot_worst_failure_frame():
    """Per-keyframe mean surface error + show the worst keyframe's image
    and its contribution to the point cloud."""
    with h5py.File("atm22_corpus.h5", "r") as f:
        gt_verts = f["metadata/ssm_surface/mean_shape"][:]

    gt_poses, sub = _gt_poses()
    gt_trans = gt_poses[sub, :3, 3]

    rec = _load_recon("recon_dust3r_foe")
    nz = ~np.all(rec == 0, axis=1)
    s, R, t, _ = _umeyama(rec[nz], gt_trans[nz])

    ply = PlyData.read(RUN / "recon_dust3r_foe" / "points.ply")
    pts = np.stack([ply["vertex"]["x"], ply["vertex"]["y"],
                    ply["vertex"]["z"]], 1)
    pts_aln = (s * (R @ pts.T)).T + t

    tree = cKDTree(gt_verts)
    dists, _ = tree.query(pts_aln, k=1)

    # Per-keyframe contributions: assume points are concatenated in frame
    # order with equal counts (true of dust3r_foe_runner._save_ply).
    n_per_frame = len(pts_aln) // N_RECON
    per_frame_err = np.zeros(N_RECON)
    for i in range(N_RECON):
        a = i * n_per_frame
        b = (i + 1) * n_per_frame if i < N_RECON - 1 else len(pts_aln)
        per_frame_err[i] = dists[a:b].mean() if b > a else 0.0

    # Mark dead keyframes specially
    is_dead = np.all(rec == 0, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = ["lightcoral" if d else "tab:blue" for d in is_dead]
    ax.bar(np.arange(N_RECON), per_frame_err, color=colors)
    ax.set_xlabel("recon keyframe index")
    ax.set_ylabel("mean surface error (mm)")
    ax.set_title("per-keyframe mean surface error  "
                 "(red = dead pose)")

    # Pick the worst non-dead keyframe
    masked = per_frame_err.copy()
    masked[is_dead] = -np.inf
    worst = int(masked.argmax())
    worst_err = per_frame_err[worst]

    # Look up its original frame number
    n_total_frames = len(sorted((RUN / "color").glob("*.png")))
    full_idx = np.linspace(0, n_total_frames - 1, N_RECON).round().astype(int)
    orig_fn = int(full_idx[worst]) + 1
    img = np.array(Image.open(RUN / "color" / f"{orig_fn:06d}.png"))
    axes[1].imshow(img)
    axes[1].set_title(f"worst non-dead keyframe: recon idx {worst}, "
                      f"orig frame {orig_fn},\n"
                      f"mean surface err {worst_err:.1f} mm")
    axes[1].axis("off")

    plt.tight_layout()
    p = OUT / "4_worst_failure.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}  worst kf idx={worst} (orig frame {orig_fn}), "
          f"mean err {worst_err:.1f} mm")


if __name__ == "__main__":
    print("1/4 trajectory overlay...")
    plot_trajectory_overlay()
    print("2/4 cloud vs GT mesh...")
    plot_cloud_vs_gt_mesh()
    print("3/4 dead keyframes...")
    plot_dead_keyframes()
    print("4/4 worst failure frame...")
    plot_worst_failure_frame()
    print(f"\nAll diagnostics in {OUT}/")
