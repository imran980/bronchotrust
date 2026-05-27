"""Render a synthetic bronchoscopy session from the SSM mean mesh.

Uses Open3D's tensor raycasting scene (GPU-accelerated, much faster
than trimesh's Python BVH on a 300k-face mesh). Virtual pinhole
camera placed at the SSM TRACHEA landmark, advanced along the
centerline path TRACHEA -> CARINA -> RMB_BIF -> BI_BIF. Per frame
we cast a ray per pixel against the SSM mean surface, take depth +
hit-triangle normals, shade salmon RGB by Lambertian, project the
named landmarks for visibility, and record the camera-to-world pose.

Outputs are shaped like a UAAL session so depth_eval_uaal,
build_endo2dtam_scene, and render_dashboard_v1 all run on it
unchanged. We also drop a params_gt.npz in Endo-2DTAM's format so
the dashboard can be tested with perfect tracking, isolating
downstream errors from SLAM drift.

Pre-req: pip install open3d

Outputs under <out_dir>/:
  color/NNNNNN.png         RGB render
  depth_gt/NNNNNN.npy      float32 ground-truth depth in mm
  pose_gt.txt              one row/frame, 16 floats (C3VD layout:
                           T_c2w flattened column-major so the
                           loader's reshape+transpose returns T_c2w)
  landmarks_gt.jsonl       per-frame visible landmarks
  manifest.jsonl           UAAL-shaped manifest with visible
                           landmarks as annotations
  params_gt.npz            Endo-2DTAM-format world-to-camera poses
  uaal.yaml                data config matching the synthetic camera

Usage:
  python render_synthetic_bronchoscopy.py \\
      --corpus atm22_corpus.h5 \\
      --out_dir runs/synthetic_session \\
      --n_frames 200 \\
      --image_size 400
"""
from __future__ import annotations

import argparse
import json
import os

import networkx as nx
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from atm22_corpus import Corpus

# Inlined from the deleted render_external_view.py.
CENTERLINE_PATH_LANDMARKS = ["TRACHEA", "CARINA", "RMB_BIF", "BI_BIF"]


def _build_graph(n_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(a), int(b)) for a, b in edges)
    return G


def _centerline_path(template):
    nodes = template.centerline_nodes
    edges = template.centerline_edges
    G = _build_graph(len(nodes), edges)
    name_to_idx = {n: int(i)
                   for i, n in zip(template.label_node_idx, template.label_name)}
    missing = [n for n in CENTERLINE_PATH_LANDMARKS if n not in name_to_idx]
    if missing:
        raise SystemExit(f"Template missing landmarks: {missing}")
    idx_path = []
    for a, b in zip(CENTERLINE_PATH_LANDMARKS, CENTERLINE_PATH_LANDMARKS[1:]):
        seg = nx.shortest_path(G, name_to_idx[a], name_to_idx[b])
        if idx_path:
            seg = seg[1:]
        idx_path.extend(seg)
    return nodes[idx_path]


def _resample_polyline(poly, n_samples):
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-9 or n_samples <= 0:
        return np.tile(poly[0], (max(n_samples, 0), 1))
    targets = np.linspace(0.0, total, n_samples)
    out = np.zeros((n_samples, 3), dtype=poly.dtype)
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cum, t) - 1)
        idx = max(0, min(idx, len(seg_len) - 1))
        f = (t - cum[idx]) / max(seg_len[idx], 1e-9)
        out[i] = poly[idx] + f * seg[idx]
    return out


LANDMARKS_TO_TRACK = ["TRACHEA", "CARINA", "RMB_BIF", "LMB_BIF", "BI_BIF"]

# Map SSM landmark identifiers to the anatomical class names the
# dashboard's DEPTH_FRAC table uses. Without this remapping the
# manifest annotations would be labeled "CARINA"/"RMB_BIF"/... which
# the dashboard treats as 'outside the airway' and stalls the dot.
SSM_TO_ANATOMICAL = {
    "TRACHEA":  "Trachea",
    "CARINA":   "Main carina",
    "RMB_BIF":  "Right main bronchus",
    "LMB_BIF":  "Left main bronchus",
    "BI_BIF":   "Intermediate bronchus",
}


def _camera_basis(forward, world_up=np.array([0.0, 0.0, 1.0])):
    """OpenCV camera basis: returns (right, down, forward) such that
    R_c2w with these as columns sends camera-frame +X to right,
    +Y to down, +Z to forward in world."""
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-9)
    return right, down, forward


def _pose_c2w(cam_pos, forward):
    right, down, fwd = _camera_basis(forward)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = fwd
    T[:3, 3] = cam_pos
    return T


def _rot_to_quat(R):
    """3x3 rotation matrix -> (w, x, y, z) unit quaternion."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / max(np.linalg.norm(q), 1e-9)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--image_size", type=int, default=400)
    ap.add_argument("--fov_deg", type=float, default=90.0,
                    help="Diagonal pinhole FOV in degrees.")
    ap.add_argument("--png_depth_scale", type=float, default=2.55,
                    help="Depth scaling for Endo-2DTAM yaml (must match "
                         "what build_endo2dtam_scene writes).")
    ap.add_argument("--fwd_smooth_sigma", type=float, default=2.5,
                    help="Gaussian sigma (frames) applied to cam_pos. "
                         "Smooths the trajectory through bifurcations "
                         "so cam_fwd is naturally continuous; 0 = raw "
                         "centerline polyline (snaps at branches).")
    ap.add_argument("--min_valid_pixels", type=int, default=100,
                    help="Skip frames where fewer than this many rays "
                         "hit the mesh. Filters out the carina-region "
                         "SSM-warp artifact.")
    args = ap.parse_args()

    color_dir = os.path.join(args.out_dir, "color")
    depth_dir = os.path.join(args.out_dir, "depth_gt")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    corpus = Corpus(args.corpus)
    ssm = corpus.load_ssm(surface=True)
    if ssm is None:
        raise SystemExit("No surface SSM in corpus. "
                         "Run fit_ssm.py --use_surface first.")
    template = corpus.load(corpus.template_subject_id)
    faces = corpus.surface_template_faces
    if faces is None:
        raise SystemExit("No surface_template_faces in corpus.")
    verts = ssm["mean_shape"].astype(np.float32)
    tris = faces.astype(np.uint32)
    print(f"Loaded SSM mean mesh: {len(verts)} verts, {len(tris)} faces")

    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(verts)
    mesh_t.triangle.indices = o3d.core.Tensor(tris)
    mesh_t.vertex.normals = o3d.core.Tensor(
        np.asarray(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(tris.astype(np.int32))
        ).compute_vertex_normals().vertex_normals).astype(np.float32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    landmark_positions = template.labeled_points()
    for n in LANDMARKS_TO_TRACK:
        if n not in landmark_positions:
            raise SystemExit(f"Template missing {n} landmark.")

    poly = _centerline_path(template)
    cam_pos_raw = _resample_polyline(poly, args.n_frames).astype(np.float64)
    # The centerline polyline has discrete kink-angles at bifurcations.
    # If we use it raw, cam_fwd snaps direction over a single frame at
    # the carina + lobar branches, producing rotation-dominated motion
    # that breaks optical-flow FoE. Gaussian-smooth the POSITIONS in
    # time (sigma ~2-3 frames) so the trajectory curves gently through
    # turns. cam_fwd is then the raw tangent of the smoothed path —
    # naturally continuous, and camera stays inside the airway because
    # the smoothing only deflects position by sub-mm at typical turns.
    if args.fwd_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        cam_pos = np.stack([
            gaussian_filter1d(cam_pos_raw[:, k],
                              sigma=args.fwd_smooth_sigma, mode="nearest")
            for k in range(3)
        ], axis=1)
    else:
        cam_pos = cam_pos_raw
    raw_fwd = np.zeros_like(cam_pos)
    raw_fwd[:-1] = cam_pos[1:] - cam_pos[:-1]
    raw_fwd[-1] = raw_fwd[-2] if args.n_frames > 1 else np.array([0, 0, -1])
    cam_fwd = raw_fwd / np.maximum(
        np.linalg.norm(raw_fwd, axis=1, keepdims=True), 1e-9)
    max_offset = float(np.linalg.norm(cam_pos - cam_pos_raw, axis=1).max())
    print(f"Sampled {args.n_frames} camera positions; "
          f"path Gaussian-smoothed (sigma={args.fwd_smooth_sigma}, "
          f"max deflection {max_offset:.2f} mm).")

    W = H = args.image_size
    # Convert diagonal FOV -> focal length.
    fov_rad = np.deg2rad(args.fov_deg)
    f = (np.sqrt(W * W + H * H) * 0.5) / np.tan(fov_rad * 0.5)
    fx = fy = f
    cx = W * 0.5
    cy = H * 0.5
    intrinsic_o3d = o3d.core.Tensor(
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))

    pose_rows = []
    cam_quats_w2c = []
    cam_trans_w2c = []
    landmark_jsonl = []
    manifest_rows = []
    n_skipped = 0
    out_idx = 0  # serial index of successfully-written frames

    salmon = np.array([220, 110, 95], dtype=np.float32)

    # Procedural texture volumes — sampled in world-space mm by trilinear
    # interp of pseudo-random scalar fields. Pre-generated once; tiled
    # over space via modular indexing. Three octaves of value-noise sum
    # into an fbm-style mucosa pattern; a fourth volume with one axis
    # stretched 8x gives vessel-like elongated darker streaks. Adds
    # roughly 1s/frame at 512x512.
    rng = np.random.default_rng(seed=0)
    NOISE_VOL = rng.random((64, 64, 64), dtype=np.float32)         # ~mm scale
    NOISE_VOL2 = rng.random((128, 128, 128), dtype=np.float32)     # fine detail
    VESSEL_VOL = rng.random((96, 96, 96), dtype=np.float32)        # streaks

    def _sample_volume(vol, p_xyz, scale):
        """Trilinear sample of `vol` at world coords `p_xyz` (..., 3),
        with `scale` mapping mm -> grid cells. Wraps modularly.
        Returns same leading shape as p_xyz, float32 in [0, 1]."""
        N = vol.shape[0]
        c = p_xyz * scale
        i0 = np.floor(c).astype(np.int64)
        f = (c - i0.astype(np.float32))
        f = f * f * (3.0 - 2.0 * f)  # smoothstep
        i0 = i0 % N
        i1 = (i0 + 1) % N
        x0, x1 = i0[..., 0], i1[..., 0]
        y0, y1 = i0[..., 1], i1[..., 1]
        z0, z1 = i0[..., 2], i1[..., 2]
        c000 = vol[x0, y0, z0]; c100 = vol[x1, y0, z0]
        c010 = vol[x0, y1, z0]; c110 = vol[x1, y1, z0]
        c001 = vol[x0, y0, z1]; c101 = vol[x1, y0, z1]
        c011 = vol[x0, y1, z1]; c111 = vol[x1, y1, z1]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        x00 = c000 * (1 - fx) + c100 * fx
        x01 = c001 * (1 - fx) + c101 * fx
        x10 = c010 * (1 - fx) + c110 * fx
        x11 = c011 * (1 - fx) + c111 * fx
        y0r = x00 * (1 - fy) + x10 * fy
        y1r = x01 * (1 - fy) + x11 * fy
        return y0r * (1 - fz) + y1r * fz

    for i in tqdm(range(args.n_frames), desc="render"):
        T_c2w = _pose_c2w(cam_pos[i], cam_fwd[i])
        R_c2w = T_c2w[:3, :3]
        T_w2c = np.linalg.inv(T_c2w)

        rays = scene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_o3d,
            extrinsic_matrix=o3d.core.Tensor(T_w2c.astype(np.float64)),
            width_px=W, height_px=H)
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()           # (H, W) ray param
        normals = ans["primitive_normals"].numpy()  # (H, W, 3)
        ray_dirs = rays.numpy()[..., 3:]       # (H, W, 3)

        valid = np.isfinite(t_hit)
        # Skip frames where rays miss the mesh almost entirely. This
        # happens around the carina node where the SSM mean mesh has a
        # known TPS-warp boundary artifact (see CLAUDE.md). Dropping
        # those frames gives a clean sequence; downstream tools see no
        # gap because we renumber.
        if valid.sum() < args.min_valid_pixels:
            n_skipped += 1
            continue
        out_idx += 1
        depth_img = np.where(valid, t_hit, 0.0).astype(np.float32)

        rgb_img = np.zeros((H, W, 3), dtype=np.float32)
        if valid.any():
            shading = np.abs((ray_dirs * normals).sum(axis=-1))
            shading = np.clip(0.25 + 0.75 * shading, 0.0, 1.0)
            d = depth_img.copy()
            d_max = float(d[valid].max())
            dist_norm = np.where(valid, d / max(d_max, 1e-9), 0.0)
            attenuation = 1.0 - 0.6 * dist_norm

            # World-space hit point per pixel: cam_pos + t_hit * ray_dir.
            hit_world = (cam_pos[i][None, None, :].astype(np.float32)
                         + depth_img[..., None] * ray_dirs).astype(np.float32)

            # fbm: 2 octaves of value noise -> mucosa surface variation.
            n_lo = _sample_volume(NOISE_VOL, hit_world, scale=0.7)
            n_hi = _sample_volume(NOISE_VOL2, hit_world, scale=2.5)
            mucosa = 0.65 * n_lo + 0.35 * n_hi          # [0, 1]
            mucosa_mod = 0.75 + 0.50 * mucosa            # [0.75, 1.25]

            # Vessel streaks: stretch y-axis 8x so noise produces
            # elongated features. Threshold the tail to make dark streaks.
            hw_stretch = hit_world.copy()
            hw_stretch[..., 1] *= 0.125
            v_raw = _sample_volume(VESSEL_VOL, hw_stretch, scale=1.2)
            vessels = np.clip((v_raw - 0.65) / 0.25, 0.0, 1.0)  # 0 mostly, 1 in tail
            vessel_mod = 1.0 - 0.35 * vessels             # darken by up to 35%

            # Mild specular highlights: bright on near-normal rays AND
            # local noise hot spots — mimics fluid sheen.
            spec_mask = (shading > 0.85).astype(np.float32)
            spec_noise = _sample_volume(NOISE_VOL2, hit_world, scale=4.0)
            specular = spec_mask * np.clip(spec_noise - 0.6, 0, 1) * 0.6

            texture = mucosa_mod * vessel_mod
            shaded = (texture * shading * attenuation + specular) \
                * valid.astype(np.float32)
            rgb_img = (salmon[None, None, :]
                       * shaded[..., None]).astype(np.float32)
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

        Image.fromarray(rgb_img).save(
            os.path.join(color_dir, f"{out_idx:06d}.png"))
        np.save(os.path.join(depth_dir, f"{out_idx:06d}.npy"), depth_img)

        # GT pose (T_w2c was computed above)
        pose_rows.append(",".join(f"{x:.6f}" for x in T_c2w.T.flatten()))
        cam_quats_w2c.append(_rot_to_quat(T_w2c[:3, :3]))
        cam_trans_w2c.append(T_w2c[:3, 3])

        # Landmark visibility (project to camera frame)
        visible = []
        for name in LANDMARKS_TO_TRACK:
            P = np.asarray(landmark_positions[name], dtype=np.float64)
            P_cam = R_c2w.T @ (P - cam_pos[i])
            z = P_cam[2]
            if z <= 1e-6:
                continue
            u = fx * P_cam[0] / z + cx
            v = fy * P_cam[1] / z + cy
            if not (0 <= u < W and 0 <= v < H):
                continue
            # Occlusion check via depth_img: if the rendered depth at
            # (u, v) is much smaller than z, something occludes the
            # landmark.
            ui = max(0, min(W - 1, int(round(u))))
            vi = max(0, min(H - 1, int(round(v))))
            d_here = depth_img[vi, ui]
            occluded = d_here > 0 and d_here < z - 1.0
            if occluded:
                continue
            visible.append({
                "name": name,
                "x_px": float(u), "y_px": float(v),
                "depth_mm": float(z),
                "world": P.tolist(),
            })
        landmark_jsonl.append({"frame": out_idx, "visible": visible})

        anns = []
        for vlm in visible:
            cx_p, cy_p = vlm["x_px"], vlm["y_px"]
            box_half = 12
            anns.append({
                "label": SSM_TO_ANATOMICAL.get(vlm["name"], vlm["name"]),
                "bbox": [cx_p - box_half, cy_p - box_half,
                         2 * box_half, 2 * box_half],
                "cx": cx_p, "cy": cy_p,
                "area": float((2 * box_half) ** 2),
                "segmentation": [],
            })
        manifest_rows.append({
            "subset": "synthetic",
            "split": "all",
            "session": "synth_v0",
            "frame": out_idx,
            "path": os.path.abspath(
                os.path.join(color_dir, f"{out_idx:06d}.png")),
            "width": W, "height": H,
            "annotations": anns,
        })

    with open(os.path.join(args.out_dir, "pose_gt.txt"), "w") as f:
        f.write("\n".join(pose_rows) + "\n")
    with open(os.path.join(args.out_dir, "landmarks_gt.jsonl"), "w") as f:
        for r in landmark_jsonl:
            f.write(json.dumps(r) + "\n")
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")

    cam_quats_w2c = np.asarray(cam_quats_w2c, dtype=np.float32)
    cam_trans_w2c = np.asarray(cam_trans_w2c, dtype=np.float32)
    np.savez(
        os.path.join(args.out_dir, "params_gt.npz"),
        cam_unnorm_rots=cam_quats_w2c.T[None, ...],
        cam_trans=cam_trans_w2c.T[None, ...],
        intrinsics=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                            dtype=np.float32),
        org_width=np.array(W), org_height=np.array(H),
    )

    yaml_path = os.path.join(args.out_dir, "uaal.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"dataset_name: 'c3vd'\n"
                f"camera_params:\n"
                f"  image_height: {H}\n  image_width: {W}\n"
                f"  fx: {fx:.4f}\n  fy: {fy:.4f}\n"
                f"  cx: {cx}\n  cy: {cy}\n"
                f"  png_depth_scale: {args.png_depth_scale}\n"
                f"  crop_edge: 0\n")

    n_visible = sum(len(r["visible"]) for r in landmark_jsonl)
    by_name = {}
    for r in landmark_jsonl:
        for v in r["visible"]:
            by_name[v["name"]] = by_name.get(v["name"], 0) + 1
    print(f"\nWrote synthetic session at {args.out_dir}/")
    print(f"  {out_idx} frames kept, {n_skipped} dropped (mesh-miss), "
          f"{n_visible} landmark detections total")
    for n in LANDMARKS_TO_TRACK:
        print(f"    {n:10s} visible in {by_name.get(n, 0)} frames")
    print()
    print("Quick test: render the dashboard against the GT pose")
    print("(skips SLAM entirely, isolates dashboard logic):")
    print(f"  python render_dashboard_v1.py \\")
    print(f"      --corpus {args.corpus} \\")
    print(f"      --rgb_dir {color_dir} \\")
    print(f"      --depth_dir {depth_dir} \\")
    print(f"      --params_npz {args.out_dir}/params_gt.npz \\")
    print(f"      --uaal_manifest {manifest_path} \\")
    print(f"      --uaal_subset synthetic --uaal_session synth_v0 \\")
    print(f"      --out_dir {args.out_dir}/dashboard_gt")


if __name__ == "__main__":
    main()
