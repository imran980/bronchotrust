"""Diagnostic ONLY: viewing-direction cone of the recon_airway cameras.

Question: is the f0..f552 cluster's near-zero arc length because the cameras
hardly moved AND look the same direction (forward-staring scope) — or just
clustered in position but with varied viewing directions?

Optical axis in world coordinates per camera:
  cam_from_world: x_c = R @ x_w + t  (COLMAP convention)
  camera z-axis points forward (into the scene)
  -> world-frame axis = R^T @ [0, 0, 1]
  (already stored as `axis` by stage3b extractor)
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
SFM = PROJECT / "runs/gated2/2_v2/recon_airway/sfm"


def load_cams():
    sub = r"""
import pycolmap, json, sys, re, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
out = []
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    cfw = img.cam_from_world()
    M = np.array(cfw.matrix())
    R = M[:3, :3]; t = M[:3, 3]
    C = (-R.T @ t)
    a = R.T @ np.array([0, 0, 1.0])
    a = a / max(np.linalg.norm(a), 1e-9)
    out.append({"name": img.name, "frame_idx": fi,
                 "C": C.tolist(), "axis": a.tolist()})
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(SFM)],
                          capture_output=True, text=True)
    return sorted(json.loads(res.stdout.strip().splitlines()[-1]),
                   key=lambda r: r["frame_idx"])


def stats(cams, label):
    A = np.array([c["axis"] for c in cams])
    C = np.array([c["C"] for c in cams])
    # Mean direction (normalized vector mean)
    mean_v = A.sum(0)
    mean_v = mean_v / max(np.linalg.norm(mean_v), 1e-9)
    # Angle of each cam axis from mean axis
    dots_mean = np.clip(A @ mean_v, -1.0, 1.0)
    ang_mean = np.degrees(np.arccos(dots_mean))
    # Pairwise angles
    dots_pp = np.clip(A @ A.T, -1.0, 1.0)
    ang_pp = np.degrees(np.arccos(dots_pp))
    # Mask diagonal
    np.fill_diagonal(ang_pp, 0.0)
    max_pp = float(ang_pp.max())
    # Cone half-angle (max ang_mean)
    cone_half = float(ang_mean.max())
    # Position bbox
    bbox = (C.max(0) - C.min(0)).tolist()
    bbox_diag = float(np.linalg.norm(C.max(0) - C.min(0)))
    print(f"\n=== {label}  (n={len(cams)}) ===")
    print(f"  mean axis (world frame): "
          f"[{mean_v[0]:+.4f}, {mean_v[1]:+.4f}, {mean_v[2]:+.4f}]")
    print(f"  per-camera angle from mean axis (deg):")
    for c, ang in zip(cams, ang_mean):
        print(f"    fi={c['frame_idx']:>4}  ang={ang:>6.2f}°  "
              f"axis=[{c['axis'][0]:+.4f},{c['axis'][1]:+.4f},{c['axis'][2]:+.4f}]")
    print(f"  cone half-angle (max from mean): {cone_half:.2f}°")
    print(f"  total angular extent (max pairwise angle): {max_pp:.2f}°")
    print(f"  position bbox (dx, dy, dz) scene units: "
          f"{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}  diag={bbox_diag:.3f}")
    return {
        "label": label, "n": len(cams),
        "mean_axis_world": mean_v.tolist(),
        "per_cam_angle_from_mean_deg": [
            {"frame_idx": c["frame_idx"], "angle_deg": float(a),
             "axis": c["axis"]}
            for c, a in zip(cams, ang_mean)
        ],
        "cone_half_angle_deg": cone_half,
        "total_angular_extent_deg": max_pp,
        "position_bbox_xyz": bbox,
        "position_bbox_diag": bbox_diag,
    }


def main():
    cams = load_cams()
    print(f"loaded {len(cams)} cams")
    cluster = [c for c in cams if c["frame_idx"] <= 552]
    subcord = [c for c in cams if 76 <= c["frame_idx"] <= 358]
    out = {
        "all_recon_airway": stats(cams, "ALL recon_airway cameras"),
        "cluster_f0_f552": stats(cluster, "f0..f552 cluster"),
        "subcord_f76_f358": stats(subcord, "f76..f358 sub-cord subset"),
    }
    diag_path = PROJECT / "runs/gated2/2_v2/recon_airway/view_cone.json"
    diag_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {diag_path}")


if __name__ == "__main__":
    main()
