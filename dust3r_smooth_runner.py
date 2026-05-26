"""
DUSt3R-with-SE(3)-smoothness subprocess runner.

This is a separate algorithm from dust3r_runner.py — it shares the
per-pair AsymmetricCroCo3DStereo inference but replaces the global
aligner with a subclassed PointCloudOptimizer that adds a temporal
acceleration penalty between consecutive frame poses. Phase 1 verdict
showed DUSt3R's per-pair priors work but its global aligner has no
SE(3)-continuity prior, producing 15-27x zigzag trajectories — this
adds that prior without otherwise changing the optimization.

Penalty (per-frame second difference, SCALE-INVARIANT):
    dT_t      = T_t^{-1} @ T_{t+1}                  # relative pose between t and t+1
    ddT_t     = dT_{t-1}^{-1} @ dT_t                # second difference
    v_mean    = mean(|| dT_t.translation ||)        # avg step length
    rot_pen   = mean(1 - cos(angle(ddT_t)))         # SO(3) penalty
    trans_pen = mean(|| t_ddT_t ||^2 / v_mean^2)    # rel translation acceleration
    loss      = base_loss + lam_rot * rot_pen + lam_trans * trans_pen

The v_mean normalization is critical. DUSt3R has no metric-scale
constraint, so a naive ||t_ddT||^2 penalty is minimized by globally
shrinking the scene toward the origin (zero motion = zero acceleration).
Dividing by mean step length squared makes the penalty measure
"acceleration as a fraction of typical velocity squared" — a
dimensionless ratio that scale-collapse cannot reduce.

This encodes "constant-velocity scope motion" rather than "zero velocity"
— sensible for a slow scope advance through the airway. With lam_rot=0
and lam_trans=0, behavior is identical to dust3r_runner.py.

CLI (called by reconstruct.py):
    python dust3r_smooth_runner.py \\
        --frames_dir runs/X/frames \\
        --out_dir runs/X/recon_dust3r_smooth \\
        --ckpt /path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
        --lam_rot 1.0 --lam_trans 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

DUST3R_ROOT = Path("/home/mi3dr/external/MASt3R-SLAM/thirdparty/mast3r/dust3r")
sys.path.insert(0, str(DUST3R_ROOT))

from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.cloud_opt.optimizer import PointCloudOptimizer  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402


class SmoothPointCloudOptimizer(PointCloudOptimizer):
    """PointCloudOptimizer + SE(3) second-difference penalty on frame poses.

    Behaviour is identical to the base class when both lambdas are zero,
    so this is safe to use as a drop-in replacement."""

    def __init__(self, *args, lam_rot: float = 1.0, lam_trans: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_rot = float(lam_rot)
        self.lam_trans = float(lam_trans)

    def forward(self):
        base = super().forward()
        if (self.lam_rot == 0 and self.lam_trans == 0) or self.n_imgs < 3:
            return base
        T = self.get_im_poses()  # (N, 4, 4) c2w, differentiable
        T_inv = torch.linalg.inv(T)
        dT = T_inv[:-1] @ T[1:]                   # (N-1, 4, 4)
        ddT = torch.linalg.inv(dT[:-1]) @ dT[1:]  # (N-2, 4, 4)
        R = ddT[:, :3, :3]
        t_acc = ddT[:, :3, 3]
        # tr(R) = 1 + 2 cos(theta) for SO(3) → 1 - cos(theta) is non-negative,
        # zero at identity, peaks at 2 for 180-deg flip.
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
        rot_pen = (1.0 - cos_theta).mean()
        # Scale-invariant translation penalty: divide ||t_acc||^2 by mean
        # consecutive step length squared. Otherwise the optimizer
        # minimizes the penalty by globally shrinking the scene.
        step_lens = dT[:, :3, 3].norm(dim=1)              # (N-1,)
        v2 = (step_lens ** 2).mean().clamp(min=1e-9)
        trans_pen = ((t_acc ** 2).sum(-1) / v2).mean()
        return base + self.lam_rot * rot_pen + self.lam_trans * trans_pen


# ---------- shared helpers (mirroring dust3r_runner.py) ----------

def _select_frames(frames_dir: Path, max_frames: int) -> list:
    all_frames = sorted([
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if not all_frames:
        raise FileNotFoundError(f"No image frames in {frames_dir}")
    if len(all_frames) <= max_frames:
        return [str(p) for p in all_frames]
    idx = np.linspace(0, len(all_frames) - 1, max_frames).round().astype(int)
    return [str(all_frames[i]) for i in idx]


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement
    n = len(points)
    pcd = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    pcd["x"], pcd["y"], pcd["z"] = points[:, 0], points[:, 1], points[:, 2]
    pcd["red"], pcd["green"], pcd["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(pcd, "vertex")], text=False).write(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--scene_graph", default="swin-3")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--lam_rot", type=float, default=1.0)
    ap.add_argument("--lam_trans", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dust3r_smooth] device={device}  lam_rot={args.lam_rot}  "
          f"lam_trans={args.lam_trans}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = _select_frames(args.frames_dir, args.max_frames)
    print(f"[dust3r_smooth] selected {len(paths)} frames")

    t0 = time.time()
    model = AsymmetricCroCo3DStereo.from_pretrained(str(args.ckpt)).to(device)
    print(f"[dust3r_smooth] loaded model in {time.time() - t0:.1f}s")

    images = load_images(paths, size=args.image_size, verbose=False)
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None,
                       symmetrize=True)
    print(f"[dust3r_smooth] {len(pairs)} pairs ({args.scene_graph})")

    t0 = time.time()
    output = inference(pairs, model, device, batch_size=4, verbose=True)
    print(f"[dust3r_smooth] inference {time.time() - t0:.1f}s")

    # Instantiate the subclassed optimizer directly (skipping the factory
    # in dust3r.cloud_opt.__init__ since it doesn't know about our class).
    view1, view2, pred1, pred2 = [output[k] for k in "view1 view2 pred1 pred2".split()]
    scene = SmoothPointCloudOptimizer(
        view1, view2, pred1, pred2,
        lam_rot=args.lam_rot, lam_trans=args.lam_trans,
    ).to(device)

    t0 = time.time()
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)
    print(f"[dust3r_smooth] global alignment {time.time() - t0:.1f}s")

    # ----- map to six-file contract -----
    poses = scene.get_im_poses().detach().cpu().numpy()
    timestamps = np.arange(len(poses), dtype=np.float64) / max(len(poses), 1)
    np.savez(args.out_dir / "poses.npz", c2w=poses, timestamps=timestamps)

    try:
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
        H, W = scene.imgs[0].shape[:2]
        K_mean = float(focals.mean())
        (args.out_dir / "intrinsics.yaml").write_text(
            "# DUSt3R-smooth-estimated intrinsics (per-frame focals averaged).\n"
            f"width: {W}\nheight: {H}\n"
            f"calibration: [{K_mean}, {K_mean}, {W / 2.0}, {H / 2.0}]\n"
            f"per_frame_focals: {focals.tolist()}\n"
            f"smoothness: {{lam_rot: {args.lam_rot}, lam_trans: {args.lam_trans}}}\n"
        )
    except Exception as e:
        print(f"[dust3r_smooth] focal extraction failed: {e}")

    pts3d_list = scene.get_pts3d()
    conf_list = scene.im_conf
    rgb_list = scene.imgs
    masks = scene.get_masks()
    all_pts, all_rgb, all_conf = [], [], []
    for pts, rgb, conf, mask in zip(pts3d_list, rgb_list, conf_list, masks):
        m = mask.cpu().numpy().astype(bool).reshape(-1)
        all_pts.append(pts.detach().cpu().numpy().reshape(-1, 3)[m])
        all_rgb.append((rgb * 255).astype(np.uint8).reshape(-1, 3)[m])
        all_conf.append(conf.detach().cpu().numpy().reshape(-1)[m])
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_rgb, axis=0)
    conf = np.concatenate(all_conf, axis=0).astype(np.float32)

    print(f"[dust3r_smooth] {len(points)} dense points")
    _save_ply(args.out_dir / "points.ply", points, colors)
    np.save(args.out_dir / "confidence.npy", conf)

    print("[dust3r_smooth] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
