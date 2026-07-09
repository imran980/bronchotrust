"""Airway-Fit v0 prototype + stress test. Throat-depth search: cameras BEHIND the throat image
the narrowest ring ahead; detect the lumen boundary, back-project to world rays (COLMAP poses),
find the axial plane where the rays converge (min reprojection residual = throat), fit an ellipse
-> CSA/DCE (scene units). Fair comparison: slice the COLMAP dense cloud at the SAME plane.
Reports ring tightness (r_std/r_med), reprojection residual, and stability across frame subsets.
COLMAP used for poses only. Runs a CONFIG of regions and applies the go/redesign decision rule.
depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit"
DATA = "/home/mi3dr/dataset/validation-videos/First 15 Videos"
NREF = 18
# reference cameras = frames BEHIND the throat (they image the narrowest ring ahead)
CONFIG = [
    dict(label="2_V2 prox-subglottis", subglottic=True,  video="2-V2.MP4",  model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  ref=(875, 945)),
    dict(label="25_V1 prox-subglottis", subglottic=True, video="25-V1.MP4", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", ref=(335, 388)),
    dict(label="25_V1 dist-subglottis", subglottic=True, video="25-V1.MP4", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", ref=(405, 452)),
    dict(label="32_V2 prox-subglottis", subglottic=True, video="32-V2.MP4", model="runs/fuse_32v2/sparse/0",     cloud="runs/fuse_32v2/dense0/fused.ply",     ref=(300, 328)),
    dict(label="32_V2 dist-subglottis", subglottic=True, video="32-V2.MP4", model="runs/fuse_32v2/sparse/0",     cloud="runs/fuse_32v2/dense0/fused.ply",     ref=(330, 358)),
    dict(label="2_V2 trachea-ref",      subglottic=False, video="2-V2.MP4", model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  ref=(1060, 1118)),
]


def load_model(model):
    rec = pycolmap.Reconstruction(model); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ M[:3, 3])
    return K, dist, poses


def detect_lumen(bgr, K, dist):
    und = cv2.undistort(bgr, K, dist); g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY); H, W = g.shape
    spec = (g > 230).astype(np.uint8)
    gm = cv2.inpaint(g, cv2.dilate(spec, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA)
    body = gm[g > 8]
    if body.size < 1000: return None, und
    dark = ((gm < np.percentile(body, 18)) & (g > 4)).astype(np.uint8)
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
    return (cv2.fitEllipse(c), und) if len(c) >= 8 else (None, und)


def ellipse_pts(e, n=72):
    (cx, cy), (MA, ma), ang = e; a = np.deg2rad(ang); th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + (MA / 2) * np.cos(th) * np.cos(a) - (ma / 2) * np.sin(th) * np.sin(a)
    y = cy + (MA / 2) * np.cos(th) * np.sin(a) + (ma / 2) * np.sin(th) * np.cos(a)
    return np.stack([x, y], 1)


def basis(t):
    t = t / (np.linalg.norm(t) + 1e-9); a = np.array([0, 0, 1.]) if abs(t[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1) + 1e-9; return t, e1, np.cross(t, e1)


def fit_ellipse_2d(P2):
    c0 = np.median(P2, 0); r = np.linalg.norm(P2 - c0, axis=1); m = np.median(r)
    Q = P2[(r > 0.4 * m) & (r < 2.0 * m)]
    if len(Q) < 8: return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(Q.astype(np.float32)); a, b = MA / 2, ma / 2
    rr = np.linalg.norm(Q - [cx, cy], axis=1)
    return dict(a=float(a), b=float(b), center=np.array([cx, cy]), ang=ang, r_med=float(np.median(rr)),
                ratio=float(np.std(rr) / max(np.median(rr), 1e-9)), CSA=float(np.pi * a * b), DCE=float(2 * np.sqrt(a * b)))


def ring3d_from(O, e1, e2, fe, n=72):
    ang = np.deg2rad(fe["ang"]); u1 = np.cos(ang) * e1 + np.sin(ang) * e2; u2 = -np.sin(ang) * e1 + np.cos(ang) * e2
    ctr = O + fe["center"][0] * e1 + fe["center"][1] * e2; th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return ctr + fe["a"] * np.outer(np.cos(th), u1) + fe["b"] * np.outer(np.sin(th), u2)


def reproj_score(ring3d, rays, K, poses):
    res = []
    for r in rays:
        R, C = poses[r["frame"]]; Xc = (R @ (ring3d - C).T).T; fr = Xc[:, 2] > 1e-6
        if fr.sum() < 10: continue
        uv = (K @ Xc[fr].T).T; uv = uv[:, :2] / uv[:, 2:3]
        res.append(float(np.median(cKDTree(ellipse_pts(r["ell"], 200)).query(uv)[0])))
    return float(np.mean(res)) if res else 1e9


def intersect(rays, O, t):
    X = []
    for r in rays:
        for d in r["dirs"]:
            dt = d @ t
            if abs(dt) < 1e-6: continue
            lam = ((O - r["C"]) @ t) / dt
            if lam > 0: X.append(r["C"] + lam * d)
    return np.array(X) if X else np.zeros((0, 3))


def rays_for(refs, K, dist, poses, video):
    Kinv = np.linalg.inv(K); cap = cv2.VideoCapture(f"{DATA}/{video}"); out = []
    for fr in refs:
        if fr not in poses: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr)); ok, im = cap.read()
        if not ok: continue
        e, und = detect_lumen(im, K, dist)
        if e is None: continue
        R, C = poses[fr]; pts = ellipse_pts(e)
        wd = (R.T @ (Kinv @ np.c_[pts, np.ones(len(pts))].T)).T; wd /= np.linalg.norm(wd, axis=1, keepdims=True) + 1e-9
        out.append(dict(frame=fr, C=C, dirs=wd, ell=e, und=und))
    cap.release(); return out


def throat_search(rays, O_ref, t, e1, e2, Lref, K, poses):
    if not rays: return None, []
    d_cam = max((r["C"] - O_ref) @ t for r in rays)
    ds = np.linspace(d_cam + 0.05 * Lref, d_cam + 12 * Lref, 200); best = None; curve = []
    for d in ds:
        O = O_ref + d * t; X = intersect(rays, O, t)
        if len(X) < 40: curve.append((d, None)); continue
        fe = fit_ellipse_2d(np.stack([(X - O) @ e1, (X - O) @ e2], 1))
        if fe is None: curve.append((d, None)); continue
        sc = reproj_score(ring3d_from(O, e1, e2, fe), rays, K, poses); curve.append((d, sc))
        if best is None or sc < best["score"]:
            best = dict(d=float(d), O=O, score=float(sc), **fe, P2=np.stack([(X - O) @ e1, (X - O) @ e2], 1))
    return best, curve


def colmap_ring(cloud, O, t, e1, e2, Rmed):
    P = np.asarray(o3d.io.read_point_cloud(cloud).points); P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    slab = P[np.abs((P - O) @ t) <= 0.6 * Rmed]
    if len(slab) < 20: return None
    P2 = np.stack([(slab - O) @ e1, (slab - O) @ e2], 1)
    rr = np.linalg.norm(P2, axis=1); rm = float(np.median(rr))
    th = np.arctan2(P2[:, 1], P2[:, 0]); cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
    return dict(P2=P2, r_med=rm, ratio=float(np.std(rr) / max(rm, 1e-9)), cov=cov, CSA=float(np.pi * rm * rm), DCE=float(2 * rm))


def run_region(cfg):
    lab = cfg["label"]; rdir = OUT / lab.replace(" ", "_").replace("/", "_"); rdir.mkdir(parents=True, exist_ok=True)
    K, dist, poses = load_model(cfg["model"])
    refs = [f for f in range(cfg["ref"][0], cfg["ref"][1] + 1) if f in poses]
    if len(refs) > NREF: refs = refs[:: max(1, len(refs) // NREF)][:NREF]
    rays = rays_for(refs, K, dist, poses, cfg["video"])
    if len(rays) < 6:
        return dict(label=lab, subglottic=cfg["subglottic"], status="FAIL", reason=f"lumen segmented in only {len(rays)}/{len(refs)} frames")
    Cs = np.array([r["C"] for r in rays]); O_ref = Cs.mean(0)
    _, _, vt = np.linalg.svd(Cs - O_ref, full_matrices=False); t = vt[0]
    if (Cs[-1] - Cs[0]) @ t < 0: t = -t
    Lref = float(np.linalg.norm(Cs[-1] - Cs[0]) + 1e-6); t, e1, e2 = basis(t)
    best, curve = throat_search(rays, O_ref, t, e1, e2, Lref, K, poses)
    if best is None:
        return dict(label=lab, subglottic=cfg["subglottic"], status="FAIL", reason="no throat convergence")
    best.update(t=t, e1=e1, e2=e2)
    col = colmap_ring(cfg["cloud"], best["O"], t, e1, e2, best["r_med"])
    stab = []
    for gi in np.array_split(np.arange(len(rays)), 3):
        b2, _ = throat_search([rays[i] for i in gi], O_ref, t, e1, e2, Lref, K, poses)
        if b2: stab.append(b2["DCE"])
    band = (round(min(stab), 3), round(max(stab), 3)) if len(stab) >= 2 else None
    stab_pct = round(100 * (max(stab) - min(stab)) / max(np.median(stab), 1e-9), 1) if len(stab) >= 2 else None
    dce_disagree = round(100 * abs(best["DCE"] - col["DCE"]) / max(col["DCE"], 1e-9), 1) if col else None
    # figure: side-by-side rings
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    ax[0].scatter(best["P2"][:, 0], best["P2"][:, 1], s=4, c="tab:red", alpha=.4); ax[0].plot(0, 0, "k+", ms=11, mew=2)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Airway-Fit  DCE={best['DCE']:.2f} ratio={best['ratio']:.2f}\nreproj {best['score']:.1f}px  stab±{stab_pct}%", fontsize=10, color="crimson")
    if col:
        ax[1].scatter(col["P2"][:, 0], col["P2"][:, 1], s=4, c="tab:blue", alpha=.4); ax[1].plot(0, 0, "k+", ms=11, mew=2)
        ax[1].set_aspect("equal"); ax[1].grid(alpha=.3)
        ax[1].set_title(f"COLMAP same plane  DCE={col['DCE']:.2f} ratio={col['ratio']:.2f}\ncov={col['cov']*100:.0f}%", fontsize=10, color="navy")
    fig.suptitle(f"{lab} — Airway-Fit vs COLMAP (same plane)", fontsize=12); fig.tight_layout()
    fig.savefig(rdir / "sidebyside.png", dpi=115); plt.close(fig)
    r = dict(label=lab, subglottic=cfg["subglottic"], status="OK", n_ref=len(rays),
             AF_DCE=round(best["DCE"], 3), AF_CSA=round(best["CSA"], 3), AF_ratio=round(best["ratio"], 3),
             reproj_px=round(best["score"], 1), stab_band=band, stab_pct=stab_pct,
             COL_DCE=(round(col["DCE"], 3) if col else None), COL_CSA=(round(col["CSA"], 3) if col else None),
             COL_ratio=(round(col["ratio"], 3) if col else None), COL_cov=(round(col["cov"], 2) if col else None),
             DCE_disagree_pct=dce_disagree)
    # verdict per region: cleaner (tighter) AND sanity DCE agreement
    r["cleaner"] = bool(col and best["ratio"] < col["ratio"])
    r["dce_sanity_ok"] = bool(dce_disagree is not None and dce_disagree <= 25)
    r["improves"] = bool(r["cleaner"] and r["dce_sanity_ok"] and best["ratio"] <= 0.35)
    (rdir / "result.json").write_text(json.dumps(r, indent=2, default=str))
    return r


def main():
    rows = [run_region(c) for c in CONFIG]
    (OUT / "stress_report.json").write_text(json.dumps(rows, indent=2, default=str))
    cols = ["label", "status", "AF_DCE", "COL_DCE", "DCE_disagree_pct", "AF_ratio", "COL_ratio",
            "reproj_px", "stab_pct", "improves"]
    print("\n=== AIRWAY-FIT STRESS TEST (sanity agreement with COLMAP; not validation) ===")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "-")) for c in cols))
    sub = [r for r in rows if r.get("subglottic") and r["status"] == "OK"]
    n_improve = sum(1 for r in sub if r.get("improves"))
    n_sub_total = sum(1 for c in CONFIG if c["subglottic"])
    print(f"\nsubglottic tests improved (cleaner + DCE sanity): {n_improve}/{n_sub_total}")
    if n_improve >= 3:
        print("DECISION: >=3/4 -> PROCEED to full per-s generalized-cylinder fitting.")
    elif n_improve <= 1:
        print("DECISION: only ~1 works -> STOP and REDESIGN before investing further.")
    else:
        print(f"DECISION: {n_improve}/4 borderline -> review per-region failures before deciding.")


if __name__ == "__main__":
    main()
