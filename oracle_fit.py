"""Oracle-contour test: run the phantom-VALIDATED contour-consistency fit on HUMAN-annotated
throat contours (runs/.../oracle/contours_<tag>.json) + real COLMAP poses. This removes the
detector from the loop: if perfect contours converge (low spread, low reproj) the geometry
supports it -> build a learned segmenter; if they DON'T, the footage lacks a stable contour.
Scene units; sanity vs COLMAP slice, not validation. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, cv2, pycolmap
import phantom_contourfit as cf

ROOT = Path("/home/mi3dr/projects/bronchotrust")
ORA = ROOT / "runs/barbour30/airwayfit/oracle"
REGION = {
    "2v2prox":  dict(model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply"),
    "25v1prox": dict(model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply"),
    "25v1dist": dict(model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply"),
}


def load_model(model):
    rec = pycolmap.Reconstruction(str(ROOT / model)); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ M[:3, 3])
    return K, dist, poses


def build(frames_contours, poses, K, dist):
    C_all, U_all, W_all, per = [], [], [], []
    for fr, pts in frames_contours:
        if fr not in poses: continue
        pts = np.asarray(pts, float)
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(-1, 2)
        step = max(1, len(und) // 80); und = und[::step]
        R, C = poses[fr]; u = cf.rays_world(und, K, R, C)
        hole = float(np.median(np.linalg.norm(und - und.mean(0), axis=1)))
        C_all.append(np.repeat(C[None], len(u), 0)); U_all.append(u); W_all.append(np.full(len(u), hole))
        per.append(dict(frame=fr, C=C, R=R, pts=und))
    if not per: return None
    return dict(C=np.vstack(C_all), U=np.vstack(U_all), W=np.concatenate(W_all), frames=per)


def colmap_slice(cloud, fit, n, a0):
    import open3d as o3d
    p = ROOT / cloud
    if not p.exists(): return None
    P = np.asarray(o3d.io.read_point_cloud(str(p)).points); P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    slab = P[np.abs((P - a0) @ n - fit["z0"]) <= 0.5 * fit["R"]]
    if len(slab) < 30: return None
    rr = np.linalg.norm((slab - a0) - ((slab - a0) @ n)[:, None] * n, axis=1)
    return float(2 * np.median(rr))


def run(tag):
    cj = ORA / f"contours_{tag}.json"
    if not cj.exists(): return dict(region=tag, status="no_contours")
    data = {int(k): v for k, v in json.loads(cj.read_text()).items()}
    K, dist, poses = load_model(REGION[tag]["model"])
    items = sorted(data.items())
    g = build(items, poses, K, dist)
    if g is None or len(g["frames"]) < 5: return dict(region=tag, status="FAIL", reason="<5 frames w/ poses")
    used = [fm["frame"] for fm in g["frames"]]
    a0, n, e1, e2 = cf.basis_from_traj(poses, used)
    fit = cf.fit_circle(g["C"], g["U"], g["W"], a0, n, e1, e2, True)
    rp = cf.reproj_px(fit, g["frames"], K)
    col = colmap_slice(REGION[tag]["cloud"], fit, n, a0)
    order = np.array(used); subs, rej = [], 0
    for k in range(4):
        fl = set(order[np.arange(len(order)) % 4 == k])
        gs = build([(f, v) for f, v in items if f in fl], poses, K, dist)
        if gs is None or len(gs["frames"]) < 4: continue
        fb = cf.basis_from_traj(poses, [fm["frame"] for fm in gs["frames"]])
        fs = cf.fit_circle(gs["C"], gs["U"], gs["W"], *fb, True)
        if fs["spread"] <= 0.12: subs.append(2 * fs["R"])
        else: rej += 1
    stab = round(100 * (max(subs) - min(subs)) / max(np.median(subs), 1e-9), 1) if len(subs) >= 2 else None
    agree = round(100 * abs(2 * fit["R"] - col) / max(col, 1e-9), 1) if col else None
    r = dict(region=tag, status="OK", n_frames=len(g["frames"]), fit_DCE_scene=round(2 * fit["R"], 4),
             COLMAP_slice_DCE=(round(col, 4) if col else None), agree_pct=agree,
             consistency_spread=round(fit["spread"], 4), reproj_px=(round(rp, 2) if rp else None),
             stab_pct=stab, n_subsets_used=len(subs), n_rejected=rej)
    # PASS vs phantom thresholds: spread ~0.04, reproj 3-5px (allow generous real-video margins)
    r["converges"] = bool(fit["spread"] <= 0.08 and (rp is not None and rp <= 12) and (stab is not None and stab <= 20))
    return r


def main():
    rows = [run(t) for t in REGION]
    (ORA / "oracle_fit_report.json").write_text(json.dumps(rows, indent=2, default=str))
    print("=== ORACLE-CONTOUR FIT (human contours + real COLMAP poses) ===")
    print("phantom reference: spread ~0.04, reproj 3-5px")
    print(f"{'region':10}{'fitDCE':>8}{'COLMAP':>8}{'agree%':>8}{'spread':>8}{'reproj':>8}{'stab%':>7}  converges")
    for r in rows:
        if r["status"] == "OK":
            print(f"{r['region']:10}{r['fit_DCE_scene']:>8}{str(r['COLMAP_slice_DCE']):>8}{str(r['agree_pct']):>8}"
                  f"{r['consistency_spread']:>8}{str(r['reproj_px']):>8}{str(r['stab_pct']):>7}  {r['converges']}")
        else:
            print(f"{r['region']:10}  {r['status']} {r.get('reason','')}")


if __name__ == "__main__":
    main()
