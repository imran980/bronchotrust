"""Per-landmark lumen-plane validity tests on pinned & unpinned 2_v2 meshes.

For each provisional landmark frame L1=20, L2=140, L3=2250:
  1. find registered camera nearest by frame_idx
  2. camera center C and optical axis a (world frame, direction of view)
  3. cut mesh with plane through C, normal a; stitch contour polylines
  4. RANSAC 2D circle fit on contour points; record center, radius, RMS, n_inl
  5. validity:
       (a) cam_inside_lumen: in-plane |C - circle_center| < radius
       (b) plane_faces_lumen: angle(plane_normal, a) < 30 deg (true by const.)
       (c) closed_contour: angular coverage around fitted circle >= 60%
       (d) fit: RANSAC RMS < 1.0 (scene units, pre-scale)
  6. diameter = 2 * radius
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


PROJECT = Path("/home/mi3dr/projects/bronchotrust")
LANDMARK_TARGETS = {"L1 glottis": 20, "L2 prox subglottis": 140,
                     "L3 distal pre-carina": 2250}


def get_camera_pose(sfm_dir: Path, target_frame: int):
    sub = r"""
import pycolmap, json, sys, re, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
target = int(sys.argv[2])
best = None; best_d = 1e9
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    if not m: continue
    fi = int(m.group(1))
    d = abs(fi - target)
    if d < best_d:
        best_d = d
        cfw = img.cam_from_world()
        M = np.array(cfw.matrix())
        R = M[:3,:3]; t = M[:3,3]
        C = (-R.T @ t).tolist()
        a = (R.T @ np.array([0,0,1])).tolist()
        best = {"cam_frame": fi, "image_id": int(img.image_id),
                 "C": C, "axis": a}
print(json.dumps(best))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(sfm_dir),
                           str(target_frame)],
                          capture_output=True, text=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def stitch_to_2d(segments, plane_origin, plane_normal):
    if segments is None or len(segments) == 0:
        return [], np.empty((0, 2))
    up = np.array([0.0, 0.0, 1.0])
    if abs(plane_normal @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    e1 = up - (up @ plane_normal) * plane_normal
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(plane_normal, e1)
    seg_2d = np.zeros((len(segments), 2, 2))
    for i, seg in enumerate(segments):
        for j in range(2):
            rel = seg[j] - plane_origin
            seg_2d[i, j, 0] = rel @ e1
            seg_2d[i, j, 1] = rel @ e2
    n = len(seg_2d); used = np.zeros(n, dtype=bool); polylines = []
    for start in range(n):
        if used[start]:
            continue
        used[start] = True
        chain = [seg_2d[start, 0], seg_2d[start, 1]]
        ext = True
        while ext:
            ext = False
            for j in range(n):
                if used[j]:
                    continue
                if np.linalg.norm(seg_2d[j, 0] - chain[-1]) < 1e-3:
                    chain.append(seg_2d[j, 1]); used[j] = True
                    ext = True; break
                elif np.linalg.norm(seg_2d[j, 1] - chain[-1]) < 1e-3:
                    chain.append(seg_2d[j, 0]); used[j] = True
                    ext = True; break
        ext = True
        while ext:
            ext = False
            for j in range(n):
                if used[j]:
                    continue
                if np.linalg.norm(seg_2d[j, 1] - chain[0]) < 1e-3:
                    chain.insert(0, seg_2d[j, 0]); used[j] = True
                    ext = True; break
                elif np.linalg.norm(seg_2d[j, 0] - chain[0]) < 1e-3:
                    chain.insert(0, seg_2d[j, 1]); used[j] = True
                    ext = True; break
        polylines.append(np.array(chain))
    return polylines, np.vstack(polylines) if polylines else np.empty((0, 2))


def _circle_3pts(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx_, cy_ = p3
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax*ax+ay*ay)*(by-cy_)+(bx*bx+by*by)*(cy_-ay)+(cx_*cx_+cy_*cy_)*(ay-by))/d
    uy = ((ax*ax+ay*ay)*(cx_-bx)+(bx*bx+by*by)*(ax-cx_)+(cx_*cx_+cy_*cy_)*(bx-ax))/d
    r = np.hypot(ux-ax, uy-ay)
    return ux, uy, r


def _kasa(pts):
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x*x + y*y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = -sol[0]/2; cy = -sol[1]/2
    r2 = cx*cx + cy*cy - sol[2]
    if r2 <= 0:
        return None
    return cx, cy, np.sqrt(r2)


def ransac_circle(pts, thresh=1.5, max_iter=2000):
    n = len(pts)
    if n < 5:
        return None
    rng = np.random.default_rng(42)
    best = None
    for _ in range(max_iter):
        idx = rng.choice(n, size=3, replace=False)
        c = _circle_3pts(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if c is None:
            continue
        cx, cy, r = c
        if r > 200 or r < 0.5:
            continue
        d = np.linalg.norm(pts - [cx, cy], axis=1)
        inl = np.abs(d - r) < thresh
        if best is None or inl.sum() > best["n"]:
            best = {"cx": cx, "cy": cy, "r": r, "n": int(inl.sum()),
                     "inl": inl}
    if best is None:
        return None
    if best["inl"].sum() >= 5:
        refit = _kasa(pts[best["inl"]])
        if refit is not None:
            cx, cy, r = refit
            d = np.linalg.norm(pts - [cx, cy], axis=1)
            inl = np.abs(d - r) < thresh
            best.update({"cx": cx, "cy": cy, "r": r,
                          "n": int(inl.sum()), "inl": inl})
    d = np.linalg.norm(pts - [best["cx"], best["cy"]], axis=1)
    rms = float(np.sqrt(np.mean((d[best["inl"]] - best["r"]) ** 2))) \
          if best["inl"].any() else float("inf")
    return {"cx": float(best["cx"]), "cy": float(best["cy"]),
             "r": float(best["r"]), "n_inliers": int(best["n"]),
             "n_total": int(n), "rms": rms,
             "inlier_fraction": float(best["n"] / n)}


def angular_coverage(pts, cx, cy, n_bins=36):
    theta = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts = np.histogram(theta, bins=bins)[0]
    return float((counts > 0).sum() / n_bins)


def analyze(recon_dir: Path, mesh_path: Path, label: str):
    print(f"\n--- {label} ({recon_dir.name}) ---")
    if not mesh_path.exists():
        return {"label": label, "error": "mesh missing"}
    # Bypass trimesh's PLY loader (numpy 2.x incompat); use open3d to read
    # then feed vertex/face arrays into trimesh.Trimesh(...).
    import open3d as o3d
    m_o = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = trimesh.Trimesh(vertices=np.asarray(m_o.vertices),
                            faces=np.asarray(m_o.triangles),
                            process=False)
    print(f"  mesh: {len(mesh.vertices)} verts")
    out = {"label": label, "landmarks": {}}
    for name, target_f in LANDMARK_TARGETS.items():
        pose = get_camera_pose(recon_dir / "sfm", target_f)
        if pose is None:
            out["landmarks"][name] = {"error": "no camera"}
            continue
        C = np.array(pose["C"])
        a = np.array(pose["axis"])
        a = a / max(np.linalg.norm(a), 1e-9)
        cam_f = pose["cam_frame"]
        segments = trimesh.intersections.mesh_plane(
            mesh, plane_normal=a, plane_origin=C)
        polylines, contour_2d = stitch_to_2d(segments, C, a)
        if len(contour_2d) < 10:
            out["landmarks"][name] = {
                "target_frame": target_f, "cam_frame": cam_f,
                "C_world": C.tolist(), "axis_world": a.tolist(),
                "n_contour": int(len(contour_2d)), "circle": None,
                "validity": {"a_cam_inside_lumen": False,
                              "b_plane_faces_lumen": False,
                              "c_closed_contour": False,
                              "d_fit_ok": False},
                "PASS_ALL": False, "note": "insufficient contour"}
            continue
        fit = ransac_circle(contour_2d, thresh=1.5)
        cam_in_plane = np.array([0.0, 0.0])
        in_plane_d = float(np.linalg.norm(
            cam_in_plane - [fit["cx"], fit["cy"]]))
        a_cam_inside = in_plane_d < fit["r"]
        b_angle_deg = 0.0  # by construction plane_normal == a
        b_plane_faces = True
        ang_cov = angular_coverage(contour_2d, fit["cx"], fit["cy"])
        c_closed = ang_cov >= 0.60
        d_fit = fit["rms"] < 1.0
        PASS = a_cam_inside and b_plane_faces and c_closed and d_fit
        out["landmarks"][name] = {
            "target_frame": target_f, "cam_frame": cam_f,
            "C_world": C.tolist(), "axis_world": a.tolist(),
            "n_contour": int(len(contour_2d)),
            "n_polylines": int(len(polylines)),
            "circle": {
                "center_2d": [fit["cx"], fit["cy"]],
                "radius": fit["r"], "diameter": 2 * fit["r"],
                "n_inliers": fit["n_inliers"], "n_total": fit["n_total"],
                "inlier_fraction": fit["inlier_fraction"],
                "rms": fit["rms"],
            },
            "validity": {
                "a_cam_inside_lumen": bool(a_cam_inside),
                "b_plane_faces_lumen": bool(b_plane_faces),
                "c_closed_contour": bool(c_closed),
                "d_fit_ok": bool(d_fit),
            },
            "extra": {"in_plane_dist_cam_to_center": in_plane_d,
                       "b_angle_deg": b_angle_deg,
                       "angular_coverage": ang_cov},
            "PASS_ALL": bool(PASS),
        }
        print(f"  {name}: cam_f={cam_f}  r={fit['r']:.2f}  "
              f"D={2*fit['r']:.2f}  RMS={fit['rms']:.3f}  "
              f"inl={fit['n_inliers']}/{fit['n_total']}  "
              f"cov={ang_cov*100:.0f}%  PASS={PASS}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--recons", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()
    results = []
    for rd, lab in zip(args.recons, args.labels):
        r = analyze(Path(rd),
                     Path(rd) / "dense" / "mesh_poisson.ply", lab)
        results.append(r)
    Path(args.out_json).write_text(json.dumps(
        {"reconstructions": results,
         "landmark_targets": LANDMARK_TARGETS}, indent=2))
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
