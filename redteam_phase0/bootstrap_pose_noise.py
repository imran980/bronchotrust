"""Estimate the RELATIVE pose noise on our ACTUAL 2_V2 data by bootstrap-resampling each distal
camera's 2D-3D correspondences and re-solving the pose (PnP). Reports per-camera translation std
(as a fraction of the camera-to-scene depth) and rotation std (deg) -- the data-driven noise level
to feed the sim re-test, instead of an assumed 1%. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, cv2, pycolmap

ROOT = Path("/home/mi3dr/projects/bronchotrust")
MODEL = ROOT / "runs/batch4/2-V2/sparse/0"
RNG = (985, 1045)                     # distal reference region
B = 60


def main():
    rec = pycolmap.Reconstruction(str(MODEL)); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    p3d = {pid: np.array(p.xyz) for pid, p in rec.points3D.items()}
    rng = np.random.default_rng(0)
    t_std_frac, rot_std_deg, depths = [], [], []
    for im in rec.images.values():
        f = int(re.search(r"f(\d+)", im.name).group(1))
        if not (RNG[0] <= f <= RNG[1]): continue
        pts2, pts3 = [], []
        for p2 in im.points2D:
            if p2.has_point3D() and p2.point3D_id in p3d:
                pts2.append(p2.xy); pts3.append(p3d[p2.point3D_id])
        if len(pts3) < 30: continue
        pts2 = np.array(pts2, float); pts3 = np.array(pts3, float)
        M = np.array(im.cam_from_world().matrix()); C0 = -M[:3, :3].T @ M[:3, 3]
        depth0 = np.median(np.linalg.norm(pts3 - C0, axis=1))
        Cs, angs = [], []
        for b in range(B):
            idx = rng.integers(0, len(pts3), len(pts3))          # bootstrap resample
            ok, rvec, tvec = cv2.solvePnP(pts3[idx].reshape(-1, 1, 3), pts2[idx].reshape(-1, 1, 2), K, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: continue
            Rm, _ = cv2.Rodrigues(rvec); C = (-Rm.T @ tvec).ravel(); Cs.append(C); angs.append(Rm)
        if len(Cs) < 10: continue
        Cs = np.array(Cs); t_std = np.linalg.norm(Cs.std(0))
        # rotation spread: mean geodesic angle from the mean rotation
        Rmean = angs[len(angs) // 2]
        rot_dev = [np.degrees(np.arccos(np.clip((np.trace(Ri.T @ Rmean) - 1) / 2, -1, 1))) for Ri in angs]
        t_std_frac.append(t_std / depth0); rot_std_deg.append(np.median(rot_dev)); depths.append(depth0)
    out = dict(n_cameras=len(t_std_frac),
               translation_std_frac_of_depth=dict(median=round(float(np.median(t_std_frac)), 4),
                                                   p90=round(float(np.percentile(t_std_frac, 90)), 4)),
               rotation_std_deg=dict(median=round(float(np.median(rot_std_deg)), 3),
                                     p90=round(float(np.percentile(rot_std_deg, 90)), 3)),
               note="bootstrap PnP resampling of each distal camera's 2D-3D correspondences (60 samples/cam)")
    Path("bootstrap_pose_noise.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
