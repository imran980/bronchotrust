"""Airway-Fit v0 prototype (2_V2 proximal subglottis). Throat-depth search:
cameras BEHIND the throat image the narrowest ring ahead; detect the lumen boundary, back-
project to world rays, and find the axial plane where the rays converge to a tight ring (the
throat). Fit an ellipse there -> CSA/DCE (scene units). Fair comparison: slice the COLMAP dense
cloud at the SAME plane. Report reprojection residual + stability across frame subsets.
COLMAP used for poses only. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust"); OUT = ROOT / "runs/barbour30/airwayfit"
VIDEO = "/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4"
MODEL = "runs/batch4/2-V2/sparse/0"; CLOUD = "runs/batch4/2-V2/dense0/fused.ply"
REF_LO, REF_HI, NREF = 875, 945, 18          # reference cameras BEHIND the throat
OUT.mkdir(parents=True, exist_ok=True)


def load_model():
    rec = pycolmap.Reconstruction(MODEL); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]; tv = M[:3, 3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ tv)   # (R, C)
    return K, dist, poses, cam.width, cam.height


def detect_lumen(bgr, K, dist):
    """Undistort -> mask specular -> darkest central connected region -> fitEllipse. Returns
    (ellipse, undistorted_bgr) or (None, und)."""
    und = cv2.undistort(bgr, K, dist)
    g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY); H, W = g.shape
    spec = (g > 230).astype(np.uint8)
    gm = cv2.inpaint(g, cv2.dilate(spec, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA)
    body = gm[g > 8]
    if body.size < 1000: return None, und
    thr = np.percentile(body, 18)
    dark = ((gm < thr) & (g > 4)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    best, bd = None, 1e9
    for i in range(1, n):
        if st[i, 4] < 400: continue
        d = np.hypot(cen[i, 0] - W / 2, cen[i, 1] - H / 2)
        if d < bd: bd, best = d, i
    if best is None: return None, und
    cnts, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 8: return None, und
    return cv2.fitEllipse(c), und


def ellipse_pts(e, n=72):
    (cx, cy), (MA, ma), ang = e; a = np.deg2rad(ang); th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + (MA / 2) * np.cos(th) * np.cos(a) - (ma / 2) * np.sin(th) * np.sin(a)
    y = cy + (MA / 2) * np.cos(th) * np.sin(a) + (ma / 2) * np.sin(th) * np.cos(a)
    return np.stack([x, y], 1)


def frame_of(fr):
    cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr)); ok, im = cap.read(); cap.release()
    return im if ok else None


def basis(t):
    t = t / (np.linalg.norm(t) + 1e-9); a = np.array([0, 0, 1.]) if abs(t[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(t, e1)
    return t, e1, e2


def fit_ellipse_2d(P2):
    """robust 2D ellipse via cv2.fitEllipse after radial-trim; returns (a,b,center,rmed,rstd,ratio)."""
    c0 = np.median(P2, 0); r = np.linalg.norm(P2 - c0, axis=1); m = np.median(r)
    keep = (r > 0.4 * m) & (r < 2.0 * m); Q = P2[keep]
    if len(Q) < 8: return None
    e = cv2.fitEllipse(Q.astype(np.float32)); (cx, cy), (MA, ma), ang = e
    a, b = MA / 2, ma / 2
    rr = np.linalg.norm(Q - np.array([cx, cy]), axis=1)
    return dict(a=float(a), b=float(b), center=np.array([cx, cy]), ang=ang,
                r_med=float(np.median(rr)), ratio=float(np.std(rr) / max(np.median(rr), 1e-9)),
                CSA=float(np.pi * a * b), DCE=float(2 * np.sqrt(a * b)))


def rays_for(refs, K, dist, poses):
    """Detect lumen in each ref frame, back-project boundary to world rays. Returns list of
    (frame, C, dirs[Nx3], ellipse2d, undist)."""
    out = []
    Kinv = np.linalg.inv(K)
    for fr in refs:
        im = frame_of(fr)
        if im is None or fr not in poses: continue
        e, und = detect_lumen(im, K, dist)
        if e is None: continue
        R, C = poses[fr]; pts = ellipse_pts(e)
        cam_dirs = (Kinv @ np.c_[pts, np.ones(len(pts))].T).T
        world_dirs = (R.T @ cam_dirs.T).T; world_dirs /= np.linalg.norm(world_dirs, axis=1, keepdims=True) + 1e-9
        out.append(dict(frame=fr, C=C, dirs=world_dirs, ell=e, und=und))
    return out


def intersect(rays, O, t):
    X = []
    for r in rays:
        C = r["C"]
        for d in r["dirs"]:
            dt = d @ t
            if abs(dt) < 1e-6: continue
            lam = ((O - C) @ t) / dt
            if lam > 0: X.append(C + lam * d)
    return np.array(X) if X else np.zeros((0, 3))


def ring3d_from(O, e1, e2, fe, n=72):
    ang = np.deg2rad(fe["ang"]); u1 = np.cos(ang) * e1 + np.sin(ang) * e2; u2 = -np.sin(ang) * e1 + np.cos(ang) * e2
    ctr = O + fe["center"][0] * e1 + fe["center"][1] * e2
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return ctr + fe["a"] * np.outer(np.cos(th), u1) + fe["b"] * np.outer(np.sin(th), u2)


def reproj_score(ring3d, rays, K, poses):
    from scipy.spatial import cKDTree
    res = []
    for r in rays:
        R, C = poses[r["frame"]]; Xc = (R @ (ring3d - C).T).T; fr = Xc[:, 2] > 1e-6
        if fr.sum() < 10: continue
        uv = (K @ Xc[fr].T).T; uv = uv[:, :2] / uv[:, 2:3]
        res.append(float(np.median(cKDTree(ellipse_pts(r["ell"], 200)).query(uv)[0])))
    return float(np.mean(res)) if res else 1e9


def throat_search(rays, O_ref, t, e1, e2, Lref, K, poses):
    # forward-only: plane must sit ahead of every camera (avoids the degenerate near-camera min)
    d_cam = max((r["C"] - O_ref) @ t for r in rays)
    ds = np.linspace(d_cam + 0.05 * Lref, d_cam + 12 * Lref, 200); best = None; curve = []
    for d in ds:
        O = O_ref + d * t; X = intersect(rays, O, t)
        if len(X) < 40: curve.append((d, None)); continue
        P2 = np.stack([(X - O) @ e1, (X - O) @ e2], 1); fe = fit_ellipse_2d(P2)
        if fe is None: curve.append((d, None)); continue
        sc = reproj_score(ring3d_from(O, e1, e2, fe), rays, K, poses)
        curve.append((d, sc))
        if best is None or sc < best["score"]:
            best = dict(d=float(d), O=O, score=float(sc), **fe, P2=P2)
    return best, curve


def reproj_residual(best, rays, K, poses):
    """Project the fitted 3D ellipse into each ref cam, measure mean distance to observed boundary."""
    O, t, e1, e2 = best["O"], best["t"], best["e1"], best["e2"]
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ring3d = O + best["a"] * np.outer(np.cos(th), e1) + best["b"] * np.outer(np.sin(th), e2)
    res = []
    for r in rays:
        R, C = poses[r["frame"]]; Xc = (R @ (ring3d - C).T).T
        front = Xc[:, 2] > 1e-6
        if front.sum() < 10: continue
        uv = (K @ Xc[front].T).T; uv = uv[:, :2] / uv[:, 2:3]
        obs = ellipse_pts(r["ell"], 200)
        from scipy.spatial import cKDTree
        dd = cKDTree(obs).query(uv)[0]
        res.append(float(np.median(dd)))
    return float(np.mean(res)) if res else None


def colmap_ring(O, t, e1, e2, Rmed_hint):
    P = np.asarray(o3d.io.read_point_cloud(CLOUD).points); P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rel = P - O; along = rel @ t; slab = P[np.abs(along) <= 0.6 * Rmed_hint]
    if len(slab) < 20: return None
    P2 = np.stack([(slab - O) @ e1, (slab - O) @ e2], 1)
    th = np.arctan2(P2[:, 1], P2[:, 0]); rr = np.linalg.norm(P2, axis=1)
    cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
    rm = float(np.median(rr))
    return dict(P2=P2, r_med=rm, ratio=float(np.std(rr) / max(rm, 1e-9)), cov=cov,
                CSA=float(np.pi * rm * rm), DCE=float(2 * rm))


def main():
    K, dist, poses, W, H = load_model()
    refs = [f for f in range(REF_LO, REF_HI + 1) if f in poses]
    refs = refs[:: max(1, len(refs) // NREF)][:NREF]
    rays = rays_for(refs, K, dist, poses)
    print(f"detected lumen boundary in {len(rays)}/{len(refs)} reference frames", flush=True)
    Cs = np.array([r["C"] for r in rays]); O_ref = Cs.mean(0)
    u, s, vt = np.linalg.svd(Cs - O_ref, full_matrices=False); t = vt[0]
    if (Cs[-1] - Cs[0]) @ t < 0: t = -t
    Lref = float(np.linalg.norm(Cs[-1] - Cs[0]) + 1e-6)
    t, e1, e2 = basis(t)
    best, curve = throat_search(rays, O_ref, t, e1, e2, Lref, K, poses)
    if best is None:
        print("FIT FAILED (no converged throat)"); return
    best.update(t=t, e1=e1, e2=e2)
    resid = best["score"]
    col = colmap_ring(best["O"], t, e1, e2, best["r_med"])
    # stability across 3 subsets
    stab = []
    idx = np.array_split(np.arange(len(rays)), 3)
    for gi in idx:
        b2, _ = throat_search([rays[i] for i in gi], O_ref, t, e1, e2, Lref, K, poses)
        if b2: stab.append(b2["DCE"])
    stab_band = (round(min(stab), 3), round(max(stab), 3)) if len(stab) >= 2 else None

    res = {"target": "2_V2 proximal subglottis", "n_ref_frames": len(rays), "throat_depth_d": round(best["d"], 3),
           "AirwayFit": {"CSA": round(best["CSA"], 3), "DCE": round(best["DCE"], 3),
                         "ellipse_axes": [round(best["a"], 3), round(best["b"], 3)],
                         "ring_ratio": round(best["ratio"], 3), "reproj_residual_px": round(resid, 1) if resid else None,
                         "stability_DCE_band": stab_band},
           "COLMAP_same_plane": (None if col is None else
                                 {"CSA": round(col["CSA"], 3), "DCE": round(col["DCE"], 3),
                                  "coverage": round(col["cov"], 2), "ring_ratio": round(col["ratio"], 3)})}
    (OUT / "airwayfit_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)

    # ---- figures ----
    # (1) overlays: observed boundary (green) + reprojected fitted throat (red)
    ring3d = ring3d_from(best["O"], e1, e2, best)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for k, r in enumerate(rays[:3]):
        vis = r["und"].copy(); cv2.ellipse(vis, r["ell"], (0, 255, 0), 3)
        R, C = poses[r["frame"]]; Xc = (R @ (ring3d - C).T).T; fr = Xc[:, 2] > 1e-6
        uv = (K @ Xc[fr].T).T; uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
        for p in uv: cv2.circle(vis, tuple(p), 2, (0, 0, 255), -1)
        s2 = min(vis.shape[:2]); vis = vis[(vis.shape[0]-s2)//2:(vis.shape[0]-s2)//2+s2, (vis.shape[1]-s2)//2:(vis.shape[1]-s2)//2+s2]
        ax[k].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[k].axis("off")
        ax[k].set_title(f"f{r['frame']}: observed lumen (green) vs Airway-Fit throat reproj (red)", fontsize=9)
    fig.suptitle(f"Airway-Fit — boundary reprojection (mean residual {resid:.1f} px)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "overlays.png", dpi=120); plt.close(fig)
    # (2) side-by-side rings
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].scatter(best["P2"][:, 0], best["P2"][:, 1], s=4, c="tab:red", alpha=.4)
    e = cv2.fitEllipse(best["P2"][(np.linalg.norm(best["P2"]-np.median(best["P2"],0),axis=1)<2*best["r_med"])].astype(np.float32))
    ax[0].plot(0, 0, "k+", ms=12, mew=2); ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Airway-Fit ring (boundary)\nDCE={best['DCE']:.2f} CSA={best['CSA']:.2f} ratio={best['ratio']:.2f}", fontsize=10, color="crimson")
    if col is not None:
        ax[1].scatter(col["P2"][:, 0], col["P2"][:, 1], s=4, c="tab:blue", alpha=.4)
        ax[1].plot(0, 0, "k+", ms=12, mew=2); ax[1].set_aspect("equal"); ax[1].grid(alpha=.3)
        ax[1].set_title(f"COLMAP ring (same plane)\nDCE={col['DCE']:.2f} CSA={col['CSA']:.2f} cov={col['cov']*100:.0f}% ratio={col['ratio']:.2f}", fontsize=10, color="navy")
    fig.suptitle("2_V2 proximal subglottis — Airway-Fit vs COLMAP (same plane, scene units)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "sidebyside_ring.png", dpi=120); plt.close(fig)
    # (3) throat-depth curve
    dd = [c[0] for c in curve if c[1] is not None]; rr = [c[1] for c in curve if c[1] is not None]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(dd, rr, "-"); ax.axvline(best["d"], color="r", ls="--", label=f"throat d={best['d']:.2f}")
    ax.set_xlabel("axial depth d ahead of cameras (scene units)"); ax.set_ylabel("reprojection residual (px, lower=better)")
    ax.set_title("Throat-depth search — min reprojection residual = the throat"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "throat_depth_curve.png", dpi=120); plt.close(fig)
    print(f"\nfigures -> {OUT}/ (overlays, sidebyside_ring, throat_depth_curve)")


if __name__ == "__main__":
    main()
