"""
DUSt3R subprocess runner — invoked by reconstruct.py's DUSt3RBackbone.run().

Runs in the mast3r-slam conda env (where DUSt3R lives under
~/external/MASt3R-SLAM/thirdparty/mast3r/dust3r, made importable via
PYTHONPATH). Loads N evenly-spaced frames (DUSt3R's global aligner is
roughly O(pairs) in memory; a complete graph on 352-frame video is
~61k pairs which OOMs immediately), runs the per-pair AsymmetricCroCo
inference, then the PointCloudOptimizer global alignment, and writes
our six-file contract.

CLI (not user-facing; called by reconstruct.py):
    python dust3r_runner.py \\
        --frames_dir runs/X/frames \\
        --out_dir runs/X/recon_dust3r \\
        --ckpt /path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
        --max_frames 50 --scene_graph swin-3 --image_size 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# DUSt3R lives under MASt3R-SLAM's thirdparty; expose it.
DUST3R_ROOT = Path("/home/mi3dr/external/MASt3R-SLAM/thirdparty/mast3r/dust3r")
sys.path.insert(0, str(DUST3R_ROOT))

from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402


def _select_frames(frames_dir: Path, max_frames: int) -> list:
    """Evenly subsample PNG/JPG frames from a directory."""
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
    """Binary little-endian PLY with xyz + rgb (uint8)."""
    from plyfile import PlyData, PlyElement  # supplied by mast3r-slam env
    n = len(points)
    pcd = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    pcd["x"] = points[:, 0]
    pcd["y"] = points[:, 1]
    pcd["z"] = points[:, 2]
    pcd["red"] = colors[:, 0]
    pcd["green"] = colors[:, 1]
    pcd["blue"] = colors[:, 2]
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dust3r_runner] device={device}  ckpt={args.ckpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = _select_frames(args.frames_dir, args.max_frames)
    print(f"[dust3r_runner] selected {len(paths)} of "
          f"{len(list(args.frames_dir.iterdir()))} frames")

    t0 = time.time()
    model = AsymmetricCroCo3DStereo.from_pretrained(str(args.ckpt)).to(device)
    print(f"[dust3r_runner] loaded model in {time.time() - t0:.1f}s")

    images = load_images(paths, size=args.image_size, verbose=False)
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None,
                       symmetrize=True)
    print(f"[dust3r_runner] {len(pairs)} pairs (scene_graph={args.scene_graph})")

    t0 = time.time()
    output = inference(pairs, model, device, batch_size=4, verbose=True)
    print(f"[dust3r_runner] inference {time.time() - t0:.1f}s")

    t0 = time.time()
    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)
    print(f"[dust3r_runner] global alignment {time.time() - t0:.1f}s")

    # ----- map to six-file contract -----

    # poses: N x 4 x 4 c2w
    poses = scene.get_im_poses().detach().cpu().numpy()  # already (N, 4, 4)
    timestamps = np.arange(len(poses), dtype=np.float64) / max(len(poses), 1)
    np.savez(args.out_dir / "poses.npz", c2w=poses, timestamps=timestamps)

    # per-frame focals (DUSt3R estimates them) → write an intrinsics_est yaml
    try:
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
        H, W = scene.imgs[0].shape[:2]
        K_mean = float(focals.mean())
        (args.out_dir / "intrinsics.yaml").write_text(
            "# DUSt3R-estimated intrinsics (per-frame focals averaged).\n"
            f"width: {W}\nheight: {H}\n"
            f"calibration: [{K_mean}, {K_mean}, {W / 2.0}, {H / 2.0}]\n"
            f"per_frame_focals: {focals.tolist()}\n"
        )
    except Exception as e:
        print(f"[dust3r_runner] focal extraction failed: {e}")

    # dense point cloud: stack per-frame pts3d + colors (only points the
    # confidence map agrees with — i.e. apply scene.get_masks()).
    pts3d_list = scene.get_pts3d()  # list of (H, W, 3) tensors
    conf_list = scene.im_conf  # list of (H, W) tensors
    rgb_list = scene.imgs  # list of (H, W, 3) numpy arrays in [0, 1]
    masks = scene.get_masks()  # list of (H, W) booleans

    all_pts = []
    all_rgb = []
    all_conf = []
    for pts, rgb, conf, mask in zip(pts3d_list, rgb_list, conf_list, masks):
        m = mask.cpu().numpy().astype(bool)
        all_pts.append(pts.detach().cpu().numpy().reshape(-1, 3)[m.reshape(-1)])
        all_rgb.append((rgb * 255).astype(np.uint8).reshape(-1, 3)[m.reshape(-1)])
        all_conf.append(conf.detach().cpu().numpy().reshape(-1)[m.reshape(-1)])
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_rgb, axis=0)
    conf = np.concatenate(all_conf, axis=0).astype(np.float32)

    print(f"[dust3r_runner] {len(points)} dense points (post-confidence mask)")
    _save_ply(args.out_dir / "points.ply", points, colors)
    np.save(args.out_dir / "confidence.npy", conf)

    print(f"[dust3r_runner] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
