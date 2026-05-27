"""
DUSt3R + optical-flow FoE sign-fix + (optional) SE(3) smoothness.

This is the "ours" combination: keep DUSt3R's per-pair geometric priors
(they work — Phase 1 confirmed dense pointmaps + consistent ~540 px
focal across videos), but inject the missing temporal-direction
information from optical-flow focus-of-expansion, plus a residual
SE(3) smoothness term to clean up high-frequency wiggle.

Algorithm:

  1. For each pair (i, j) in DUSt3R's scene graph: compute consecutive-
     frame RAFT flow + FoE, classify direction as {+1 forward, -1
     backward, 0 uncertain} with a confidence score. See flow_foe.py.

  2. DUSt3R inference as usual (per-pair AsymmetricCroCo3DStereo).

  3. Modified optimizer (FoESmoothPointCloudOptimizer):
       loss = base_loss
            + lam_sign * sum over confident pairs of hinge(t_ij_z, s_ij)
            + lam_rot   * SO(3) acceleration penalty
            + lam_trans * scale-inv translation acceleration penalty

     where t_ij_z is the z-component of the relative translation T_j - T_i
     expressed in camera-i's frame (OpenCV convention: +z = scope forward),
     and the hinge is  max(0, -s_ij * t_ij_z)  — zero when sign agrees,
     linear in magnitude when it disagrees.

  4. Same six-file contract output.

Defaults: lam_sign=10, lam_rot=1, lam_trans=1. Override via env vars.

CLI (called by reconstruct.py):
    python dust3r_foe_runner.py \\
        --frames_dir runs/X/frames --out_dir runs/X/recon_dust3r_foe \\
        --ckpt /path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
        --lam_sign 10 --lam_rot 1 --lam_trans 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

DUST3R_ROOT = Path("/home/mi3dr/external/MASt3R-SLAM/thirdparty/mast3r/dust3r")
sys.path.insert(0, str(DUST3R_ROOT))

from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.cloud_opt.optimizer import PointCloudOptimizer  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402

# local
import flow_foe  # noqa: E402


class FoESmoothPointCloudOptimizer(PointCloudOptimizer):
    """PointCloudOptimizer + per-pair FoE sign hinge + SE(3) smoothness."""

    def __init__(self, *args,
                 lam_sign: float = 10.0,
                 lam_rot: float = 1.0,
                 lam_trans: float = 1.0,
                 sign_table: Dict[Tuple[int, int], int] = None,
                 conf_table: Dict[Tuple[int, int], float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_sign = float(lam_sign)
        self.lam_rot = float(lam_rot)
        self.lam_trans = float(lam_trans)
        # Per-pair FoE classification: maps (i, j) -> int in {+1, -1, 0}.
        # Pairs with sign == 0 contribute nothing.
        self.sign_table = sign_table or {}
        self.conf_table = conf_table or {}

    # --- helpers ---

    def _sign_loss(self, T_world: torch.Tensor) -> torch.Tensor:
        """Hinge loss: for each pair (i, j) with FoE sign s_ij != 0,
        penalize max(0, -s_ij * z_ij) where z_ij is the relative-translation
        z-component in camera-i's frame.

        OpenCV convention: camera looks at +z, so forward motion (camera
        approaching the scene) implies T_j is at positive z in camera-i's
        frame (the new viewpoint is "ahead" of the old one's optical axis).
        """
        if not self.sign_table or self.lam_sign == 0:
            return T_world.new_zeros(())
        # T_world: (N, 4, 4) c2w. We want relative T_j in cam-i frame:
        #   T_ij = T_i^{-1} @ T_j ; its translation column gives the
        #   position of camera j in camera i's frame.
        T_inv = torch.linalg.inv(T_world)
        device = T_world.device
        # Vectorize over confident pairs.
        ii, jj, ss, ww = [], [], [], []
        for (i, j), s in self.sign_table.items():
            if s == 0:
                continue
            ii.append(i)
            jj.append(j)
            ss.append(float(s))
            ww.append(float(self.conf_table.get((i, j), 1.0)))
        if not ii:
            return T_world.new_zeros(())
        i_idx = torch.tensor(ii, device=device, dtype=torch.long)
        j_idx = torch.tensor(jj, device=device, dtype=torch.long)
        s_vec = torch.tensor(ss, device=device)
        w_vec = torch.tensor(ww, device=device)
        T_rel = T_inv[i_idx] @ T_world[j_idx]      # (P, 4, 4)
        z_ij = T_rel[:, 2, 3]                      # z component
        hinge = torch.clamp(-s_vec * z_ij, min=0.0)
        return (w_vec * hinge).mean()

    def _smooth_loss(self, T_world: torch.Tensor) -> torch.Tensor:
        if (self.lam_rot == 0 and self.lam_trans == 0) or self.n_imgs < 3:
            return T_world.new_zeros(())
        T_inv = torch.linalg.inv(T_world)
        dT = T_inv[:-1] @ T_world[1:]                   # (N-1, 4, 4)
        ddT = torch.linalg.inv(dT[:-1]) @ dT[1:]        # (N-2, 4, 4)
        R = ddT[:, :3, :3]
        t_acc = ddT[:, :3, 3]
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
        rot_pen = (1.0 - cos_theta).mean()
        step_lens = dT[:, :3, 3].norm(dim=1)
        v2 = (step_lens ** 2).mean().clamp(min=1e-9)
        trans_pen = ((t_acc ** 2).sum(-1) / v2).mean()
        return self.lam_rot * rot_pen + self.lam_trans * trans_pen

    def forward(self):
        base = super().forward()
        T_world = self.get_im_poses()
        return base + self.lam_sign * self._sign_loss(T_world) + self._smooth_loss(T_world)


# ---------- shared helpers (mirror dust3r_runner.py) ----------

def _select_frames(frames_dir: Path, max_frames: int):
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


def _build_sign_table(frame_paths, pairs, device, min_mag, conf_threshold):
    """Run RAFT + FoE on every pair in the DUSt3R scene graph.

    pairs is a list of (view1, view2) tuples whose 'idx' fields index into
    frame_paths. We compute sign for each (i, j) where i < j.
    """
    print(f"[dust3r_foe] loading RAFT...")
    raft = flow_foe.load_raft(device)
    # We classify each ordered pair (i, j) once; sign symmetrizes via the
    # optimizer (it uses the magnitude regardless).
    seen = set()
    sign_table = {}
    conf_table = {}
    n_fwd = n_bwd = n_unc = 0
    n_pairs = len(pairs)
    print(f"[dust3r_foe] classifying {n_pairs} pair signs via FoE...")
    for k, (v1, v2) in enumerate(pairs):
        i = int(v1["idx"]) if hasattr(v1, "__getitem__") else int(v1.idx)
        j = int(v2["idx"]) if hasattr(v2, "__getitem__") else int(v2.idx)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        img_a = np.array(Image.open(frame_paths[a]).convert("RGB"))
        img_b = np.array(Image.open(frame_paths[b]).convert("RGB"))
        r = flow_foe.pair_sign(raft, img_a, img_b,
                               min_mag=min_mag,
                               conf_threshold=conf_threshold)
        sign_table[(a, b)] = int(r["sign"])
        conf_table[(a, b)] = float(r["confidence"])
        if r["sign"] == +1: n_fwd += 1
        elif r["sign"] == -1: n_bwd += 1
        else: n_unc += 1
    del raft
    torch.cuda.empty_cache()
    print(f"[dust3r_foe] FoE: {n_fwd} fwd, {n_bwd} bwd, {n_unc} uncertain "
          f"(of {len(seen)} unique pairs)")
    return sign_table, conf_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--scene_graph", default="swin-3")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--lam_sign", type=float, default=10.0)
    ap.add_argument("--lam_rot", type=float, default=1.0)
    ap.add_argument("--lam_trans", type=float, default=1.0)
    ap.add_argument("--flow_min_mag", type=float, default=0.5)
    ap.add_argument("--foe_conf_threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dust3r_foe] device={device} lam_sign={args.lam_sign} "
          f"lam_rot={args.lam_rot} lam_trans={args.lam_trans}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = _select_frames(args.frames_dir, args.max_frames)
    print(f"[dust3r_foe] selected {len(paths)} frames")

    t0 = time.time()
    model = AsymmetricCroCo3DStereo.from_pretrained(str(args.ckpt)).to(device)
    print(f"[dust3r_foe] loaded model in {time.time() - t0:.1f}s")

    images = load_images(paths, size=args.image_size, verbose=False)
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None,
                       symmetrize=True)
    print(f"[dust3r_foe] {len(pairs)} pairs ({args.scene_graph})")

    # FoE sign classification BEFORE DUSt3R inference so we can free RAFT.
    sign_table, conf_table = _build_sign_table(
        paths, pairs, device,
        min_mag=args.flow_min_mag,
        conf_threshold=args.foe_conf_threshold,
    )

    t0 = time.time()
    output = inference(pairs, model, device, batch_size=4, verbose=True)
    print(f"[dust3r_foe] DUSt3R inference {time.time() - t0:.1f}s")

    view1, view2, pred1, pred2 = [output[k] for k in "view1 view2 pred1 pred2".split()]
    scene = FoESmoothPointCloudOptimizer(
        view1, view2, pred1, pred2,
        lam_sign=args.lam_sign,
        lam_rot=args.lam_rot, lam_trans=args.lam_trans,
        sign_table=sign_table, conf_table=conf_table,
    ).to(device)

    t0 = time.time()
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)
    print(f"[dust3r_foe] global alignment {time.time() - t0:.1f}s")

    # ----- six-file contract -----
    poses = scene.get_im_poses().detach().cpu().numpy()
    timestamps = np.arange(len(poses), dtype=np.float64) / max(len(poses), 1)
    np.savez(args.out_dir / "poses.npz", c2w=poses, timestamps=timestamps)

    try:
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
        H, W = scene.imgs[0].shape[:2]
        K_mean = float(focals.mean())
        (args.out_dir / "intrinsics.yaml").write_text(
            "# DUSt3R+FoE-estimated intrinsics (per-frame focals averaged).\n"
            f"width: {W}\nheight: {H}\n"
            f"calibration: [{K_mean}, {K_mean}, {W / 2.0}, {H / 2.0}]\n"
            f"per_frame_focals: {focals.tolist()}\n"
            f"foe: {{lam_sign: {args.lam_sign}, "
            f"lam_rot: {args.lam_rot}, lam_trans: {args.lam_trans}}}\n"
        )
    except Exception as e:
        print(f"[dust3r_foe] focal extraction failed: {e}")

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
    _save_ply(args.out_dir / "points.ply", points, colors)
    np.save(args.out_dir / "confidence.npy", conf)
    print(f"[dust3r_foe] {len(points)} dense points.")

    # ----- TSDF fusion -> mesh.ply -----
    # Fuses per-frame depth maps (DUSt3R-predicted) using the now-coherent
    # camera poses into a single signed-distance field; extracts a unified
    # surface mesh via marching cubes. This is what makes a "measurable
    # airway interior" possible — without it, the per-frame pointmaps
    # remain fragmented blobs that don't agree about absolute geometry.
    try:
        import open3d as o3d
        depths = scene.get_depthmaps()  # list of (H, W) tensors
        poses = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4) c2w
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
        pp_arr = scene.get_principal_points().detach().cpu().numpy()
        # Heuristic voxel size: 1/200 of the trajectory diagonal.
        nz_mask = ~np.all(poses[:, :3, 3] == 0, axis=1)
        if nz_mask.sum() >= 2:
            traj_diag = float(np.linalg.norm(
                poses[nz_mask][-1, :3, 3] - poses[nz_mask][0, :3, 3]))
        else:
            traj_diag = 1.0
        voxel_size = max(traj_diag / 200.0, 1e-4)
        sdf_trunc = 5 * voxel_size
        print(f"[dust3r_foe] TSDF: voxel={voxel_size:.4f} "
              f"sdf_trunc={sdf_trunc:.4f}  (traj_diag={traj_diag:.3f})")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        n_integrated = 0
        for i in range(len(rgb_list)):
            if not nz_mask[i]:
                continue
            d_np = depths[i].detach().cpu().numpy().astype(np.float32)
            H, W = d_np.shape
            rgb_u8 = (rgb_list[i] * 255).astype(np.uint8)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb_u8)),
                o3d.geometry.Image(np.ascontiguousarray(d_np)),
                depth_scale=1.0,
                depth_trunc=traj_diag * 10.0,
                convert_rgb_to_intensity=False,
            )
            K_o3d = o3d.camera.PinholeCameraIntrinsic(
                W, H, float(focals[i]), float(focals[i]),
                float(pp_arr[i, 0]), float(pp_arr[i, 1]),
            )
            T_w2c = np.linalg.inv(poses[i])
            volume.integrate(rgbd, K_o3d, T_w2c)
            n_integrated += 1
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        out_mesh = args.out_dir / "mesh.ply"
        o3d.io.write_triangle_mesh(str(out_mesh), mesh)
        print(f"[dust3r_foe] TSDF integrated {n_integrated} frames -> "
              f"{out_mesh}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")
    except Exception as e:
        print(f"[dust3r_foe] TSDF fusion skipped: {e}")

    print("[dust3r_foe] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
