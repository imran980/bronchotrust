"""Band-based blade detection on f=52, 58, 64, 70 (untrimmed 2_v2).

For each blade-visible frame, find the bright vertical bands of the
suspension laryngoscope blade in source-image pixel coordinates:
  left band : column centered around x ~= 575
  right band: column centered around x ~= 1390
Band width is auto-determined from per-frame bright-pixel column histogram.

3D-classify any reconstruction point whose source-frame projection
(for that frame, via cam_from_world) falls inside either band.

PCA on the resulting 3D point set; expected: PC1/PC2 > 5, PC1 ~30-50 u,
PC2 = blade inner aperture.

Inner aperture (BLADE_MM=11 reference): distance between the INNER edges
of the left and right bands -> map to 3D as the cluster's smaller-axis
extent (PC2 robust). Save the 3D classification.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path("/home/mi3dr/projects/bronchotrust")
OUT = PROJECT / "runs/untrimmed/2_v2"
SRC = Path("/home/mi3dr/dataset/validation-videos/First 13 Videos")
VID = "2_v2"
BLADE_FRAMES = [52, 58, 64, 70]
LEFT_BAND_X = 575
RIGHT_BAND_X = 1390
BLADE_MM = 11.0


def video_path():
    for ext in ("mp4", "MP4"):
        p = SRC / f"{VID}.{ext}"
        if p.exists(): return p
    raise FileNotFoundError(VID)


def detect_bright_bands(frame_bgr, center_x_left, center_x_right,
                          search_half=120):
    """Within +/- search_half pixels of each center_x, find the
    horizontal extent of bright (L>200) vertical pixels. Returns
    (left_x_inner, left_x_outer, right_x_inner, right_x_outer) — the
    inner edges face the lumen between bands."""
    L = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[..., 0]
    H, W = L.shape
    # Column histogram: fraction of pixels in column that are L > 200
    col_bright = (L > 200).mean(axis=0)
    # Smooth
    k = np.ones(5) / 5
    col_bright_s = np.convolve(col_bright, k, mode="same")
    # Left band: search [center_x_left - half, center_x_left + half]
    def find_band(center, half):
        lo = max(0, center - half); hi = min(W, center + half)
        sub = col_bright_s[lo:hi]
        # Find columns where fraction > 0.10 (some bright pixels)
        bright_cols = np.where(sub > 0.10)[0]
        if len(bright_cols) < 5:
            return None, None
        return int(lo + bright_cols.min()), int(lo + bright_cols.max())
    Lo_L, Hi_L = find_band(center_x_left, search_half)
    Lo_R, Hi_R = find_band(center_x_right, search_half)
    return Lo_L, Hi_L, Lo_R, Hi_R


def visualize_bands(frame_bgr, bands_per_frame, out_path):
    """Annotate the source frame with detected bands; save as PNG."""
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    Lo_L, Hi_L, Lo_R, Hi_R = bands_per_frame
    if Lo_L is not None:
        cv2.rectangle(img, (Lo_L, 0), (Hi_L, H), (0, 255, 0), 3)
        cv2.putText(img, f"L band [{Lo_L},{Hi_L}]", (Lo_L, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    if Lo_R is not None:
        cv2.rectangle(img, (Lo_R, 0), (Hi_R, H), (0, 200, 255), 3)
        cv2.putText(img, f"R band [{Lo_R},{Hi_R}]", (Lo_R - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    # Indicate inner edges
    if Hi_L is not None: cv2.line(img, (Hi_L, 0), (Hi_L, H), (255, 0, 255), 2)
    if Lo_R is not None: cv2.line(img, (Lo_R, 0), (Lo_R, H), (255, 0, 255), 2)
    if Hi_L is not None and Lo_R is not None:
        inner_dist = Lo_R - Hi_L
        cv2.putText(img, f"inner_dist={inner_dist}px",
                    (Hi_L + 20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 3)
    cv2.imwrite(str(out_path), img)


def main():
    # Step 1: detect bands per blade-visible frame from source video
    cap = cv2.VideoCapture(str(video_path()))
    frame_bands = {}
    for fi in BLADE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok: continue
        bands = detect_bright_bands(f, LEFT_BAND_X, RIGHT_BAND_X)
        frame_bands[fi] = bands
        out_viz = OUT / f"_band_detect_f{fi:03d}.png"
        visualize_bands(f, bands, out_viz)
        Lo_L, Hi_L, Lo_R, Hi_R = bands
        print(f"  f={fi}  left=[{Lo_L},{Hi_L}]  right=[{Lo_R},{Hi_R}]"
              + (f"  inner_dist={Lo_R - Hi_L}px" if Hi_L and Lo_R else ""))
    cap.release()

    # Step 2: project all reconstruction points onto each blade-visible
    # frame and classify them by whether their pixel falls inside a band.
    sub = r"""
import pycolmap, numpy as np, json, sys, re
from pathlib import Path
sfm = Path(sys.argv[1]) / 'sfm'
rec = pycolmap.Reconstruction(str(sfm))
# Frame -> image_id + projection function (cam_from_world)
frame_to_img = {}
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    if m:
        frame_to_img[int(m.group(1))] = img.image_id
# For each 3D point, project to each blade frame's image plane
bands = json.loads(sys.argv[2])  # {fi_str: [LoL, HiL, LoR, HiR]}
blade_pt_ids = set()
pts_in_band_by_frame = {int(fi): [] for fi in bands}
# Need image's cam_from_world and camera
for fi_str, b in bands.items():
    fi = int(fi_str); img_id = frame_to_img.get(fi)
    if img_id is None: continue
    img = rec.images[img_id]
    cfw = img.cam_from_world()
    R = cfw.matrix()[:3,:3]; t = cfw.matrix()[:3,3]
    cam = img.camera; params = cam.params
    # Camera-model-aware projection
    cmodel = cam.model.name if hasattr(cam.model, 'name') else str(cam.model)
    if cmodel == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = params
    elif cmodel == 'SIMPLE_RADIAL':
        fx = fy = params[0]; cx, cy = params[1], params[2]
        k1 = params[3]; k2 = p1 = p2 = 0.0
    else:
        fx = fy = params[0]; cx, cy = params[1], params[2]
        k1 = k2 = p1 = p2 = 0.0
    Lo_L, Hi_L, Lo_R, Hi_R = b
    for pt_id, pt in rec.points3D.items():
        P = np.array(pt.xyz)
        Pc = R @ P + t
        Z = Pc[2]
        if Z <= 0: continue
        x = Pc[0]/Z; y = Pc[1]/Z
        r2 = x*x + y*y
        rad = 1 + k1*r2 + k2*r2*r2
        xd = x*rad + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*rad + p1*(r2 + 2*y*y) + 2*p2*x*y
        u = fx*xd + cx; v = fy*yd + cy
        in_left = (Lo_L is not None and Hi_L is not None
                    and Lo_L <= u <= Hi_L)
        in_right = (Lo_R is not None and Hi_R is not None
                     and Lo_R <= u <= Hi_R)
        if in_left or in_right:
            pts_in_band_by_frame[fi].append([int(pt_id),
                                              'L' if in_left else 'R'])
            blade_pt_ids.add(int(pt_id))
print(json.dumps({'blade_pt_ids': list(blade_pt_ids),
                  'by_frame': pts_in_band_by_frame}))
"""
    res = subprocess.run(
        [sys.executable, "-c", sub, str(OUT), json.dumps(
            {str(k): list(v) if v is not None else [None]*4
             for k, v in frame_bands.items()})],
        capture_output=True, text=True)
    print("subprocess stderr (last 500):", res.stderr[-500:])
    classification = json.loads(res.stdout.strip().splitlines()[-1])
    blade_pt_ids = set(classification["blade_pt_ids"])
    print(f"  total blade-classified 3D points: {len(blade_pt_ids)}")
    for fi, lst in classification["by_frame"].items():
        n_L = sum(1 for x in lst if x[1] == 'L')
        n_R = sum(1 for x in lst if x[1] == 'R')
        print(f"    f={fi}: {len(lst)} pts in bands (L={n_L}, R={n_R})")

    if len(blade_pt_ids) < 5:
        print("  ERROR: too few blade points classified")
        return

    # Step 3: PCA on blade points (load via subprocess to avoid pycolmap
    # in this process)
    sub2 = """
import pycolmap, numpy as np, json, sys
from pathlib import Path
sfm = Path(sys.argv[1]) / 'sfm'
rec = pycolmap.Reconstruction(str(sfm))
ids = set(json.loads(sys.argv[2]))
pts = []
for pt_id, pt in rec.points3D.items():
    if int(pt_id) in ids:
        pts.append(list(pt.xyz))
pts = np.array(pts)
print(json.dumps({'pts': pts.tolist()}))
"""
    res2 = subprocess.run([sys.executable, "-c", sub2, str(OUT),
                            json.dumps(list(blade_pt_ids))],
                           capture_output=True, text=True)
    data = json.loads(res2.stdout.strip().splitlines()[-1])
    blade_pts = np.array(data["pts"])
    bcen = blade_pts.mean(0)
    Xb = blade_pts - bcen
    _, _, Vt = np.linalg.svd(Xb, full_matrices=False)
    proj = Xb @ Vt.T
    pc_robust = (np.percentile(proj, 97.5, axis=0)
                  - np.percentile(proj, 2.5, axis=0))
    pc_full = proj.max(0) - proj.min(0)
    flatness = pc_robust[0] / max(pc_robust[1], 1e-9)
    print(f"\n  3D blade PCA on band-classified points:")
    print(f"  blade pts: {len(blade_pts)}")
    print(f"  centroid (scene units): {bcen.round(2)}")
    print(f"  PC robust extents: PC1={pc_robust[0]:.2f}u  "
          f"PC2={pc_robust[1]:.2f}u  PC3={pc_robust[2]:.2f}u")
    print(f"  PC full extents:   PC1={pc_full[0]:.2f}u  "
          f"PC2={pc_full[1]:.2f}u  PC3={pc_full[2]:.2f}u")
    print(f"  PC1/PC2 (flatness): {flatness:.2f}")
    print(f"  -> PC1 (length): {pc_robust[0]:.2f}u  "
          f"(expected blade extent 30-50u)")
    print(f"  -> PC2 (aperture): {pc_robust[1]:.2f}u  "
          f"(BLADE_MM = {BLADE_MM} mm  =>  scale = "
          f"{BLADE_MM/pc_robust[1]:.4f} mm/u)")

    np.savez(OUT / "_blade_points_v2.npz",
              blade_pts=blade_pts, pc_axes=Vt, pc_robust=pc_robust)
    out_data = {
        "method": "band-based detection on f=52,58,64,70",
        "left_band_x_center": LEFT_BAND_X,
        "right_band_x_center": RIGHT_BAND_X,
        "per_frame_bands": {str(k): (list(v) if v else None)
                              for k, v in frame_bands.items()},
        "n_blade_points_3d": int(len(blade_pts)),
        "pc1_robust_units": float(pc_robust[0]),
        "pc2_robust_units": float(pc_robust[1]),
        "pc3_robust_units": float(pc_robust[2]),
        "flatness_pc1_pc2": float(flatness),
        "BLADE_MM": BLADE_MM,
        "scale_mm_per_unit": float(BLADE_MM / pc_robust[1]),
    }
    (OUT / "_blade_v2_summary.json").write_text(json.dumps(out_data, indent=2))
    print(f"\n  saved {OUT}/_blade_v2_summary.json")


if __name__ == "__main__":
    main()
