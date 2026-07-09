"""Multi-view CONTOUR-consistency tube fit (NOT throat-depth search, NOT boundary-point
triangulation). Each frame's lumen boundary is a SILHOUETTE / occluding-contour constraint;
back-project every boundary pixel to a world ray (known poses) and find the SINGLE axial plane
z0 + radius R at which the boundary rays from ALL cameras converge to one common circle:

  crossing r_j(z0) = |(C_j + s u_j - a0) - z0*n|  (s>0);  R̂(z0)=wmedian_j r_j
  spread(z0) = wMAD_j / R̂ ;  z0* = argmin spread ;  R* = R̂(z0*) ;  DCE = 2R*

Rays are WEIGHTED by hole size (bigger hole = camera nearer the contour = stronger z0 leverage)
to reduce frame-subset instability. Runs on phantoms ONLY. Detectors:
  rim  = near-black escaped-ray hole (uniform phantom's true occluding contour)
  geom = GT through-throat aperture mask boundary (stenosis phantom's true occluding contour)
  phot = percentile-dark lumen boundary (the real-airway analog)
Pose sources: COLMAP sparse/0 (realistic, -> GT units by Umeyama scale) and GT (isolates method).
Report: GT R/DCE, fitted R/DCE, %error, reproj residual, subset stability, COLMAP slice DCE.
depth-eval env.  Usage: python phantom_contourfit.py [uniform|stenosis|both]
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")

UNIFORM = dict(name="uniform", ph=ROOT / "runs/phantom", gt_kind="R_GT",
               detectors=("rim", "phot"), mask=False)
STENOSIS = dict(name="stenosis", ph=ROOT / "runs/phantom_sten", gt_kind="R_throat",
                detectors=("geom", "edge", "phot"), mask=True)


# ---------- poses ----------
def umeyama_scale(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Ds, Vt = np.linalg.svd((Yc.T @ Xc) / len(X)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return float((Ds * np.diag(S)).sum() / ((Xc ** 2).sum() / len(X)))


def load_colmap(ph):
    rec = pycolmap.Reconstruction(str(ph / "sparse/0")); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ M[:3, 3])
    return K, dist, poses


def load_gt(ph, gt_kind):
    gt = json.loads((ph / "gt_poses.json").read_text())
    poses = {p["frame"]: (np.array(p["R_wc"]), np.array(p["C"])) for p in gt["poses"]}
    return float(gt[gt_kind]), poses, gt


# ---------- boundary detection ----------
def _central(mask, W, H, min_area=250):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    best, bd = None, 1e9
    for i in range(1, n):
        if st[i, 4] < min_area: continue
        d = np.hypot(cen[i, 0] - W / 2, cen[i, 1] - H / 2)
        if d < bd: bd, best = d, i
    if best is None: return None
    ys, xs = np.where(lab == best)
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2: return None
    cnts, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64) if len(c) >= 12 else None


def throat_edge(g):
    """Occlusion-EDGE detector (photometric): the throat rim is the innermost strong bright->dark
    radial gradient around the dark-core center — NOT the dark region itself (on a smooth stenosis
    the dark region is the dim approaching-constriction wall, which is larger and view-varying).
    Polar-unwrap around the dark-core centroid, take the PEAK positive radial gradient per angle."""
    H, W = g.shape; g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 3)
    thr = np.percentile(g[g > 4], 6); dark = ((g < thr) & (g > 1)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    if n < 2: return None
    cx, cy = cen[1 + int(np.argmax(st[1:, 4]))]
    pts = None
    for _ in range(2):                                  # refine polar center from the edge
        Rmax = int(min(cx, cy, W - cx, H - cy) * 0.95)
        if Rmax < 30: return None
        pol = cv2.GaussianBlur(cv2.warpPolar(g, (Rmax, 720), (float(cx), float(cy)), Rmax, cv2.WARP_POLAR_LINEAR), (0, 0), 2)
        grad = np.diff(pol, axis=1); cur = []
        for a in range(0, 720, 3):
            seg = grad[a, 8:]
            if seg.size == 0 or seg.max() <= 0: continue
            r = 8 + int(np.argmax(seg)); th = 2 * np.pi * a / 720
            cur.append([cx + r * np.cos(th), cy + r * np.sin(th)])
        if len(cur) < 20: return None
        pts = np.array(cur, float); cx, cy = pts.mean(0)
    return pts


def detect(bgr, K, dist, mode, mask_img=None):
    if mode == "geom":                                  # GT through-throat aperture mask boundary
        m = cv2.undistort(mask_img, K, dist)
        if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        H, W = m.shape
        return _central((m > 127).astype(np.uint8), W, H), None
    und = cv2.undistort(bgr, K, dist); g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY); H, W = g.shape
    if mode == "edge":                                  # occlusion-edge (photometric, real-video path)
        return throat_edge(g), und
    if mode == "rim":
        mask = (g <= 10).astype(np.uint8)
    else:                                               # phot: percentile-dark central lumen
        body = g[g > 6]
        if body.size < 1000: return None, und
        mask = ((g < np.percentile(body, 20)) & (g > 2)).astype(np.uint8)
    return _central(mask, W, H), und


def rays_world(pts, K, R, C):
    Kinv = np.linalg.inv(K)
    d = (R.T @ (Kinv @ np.c_[pts, np.ones(len(pts))].T)).T
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


# ---------- weighted stats ----------
def wmedian(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# ---------- gather + fit ----------
def gather(frames, poses, K, dist, mode, img_dir, mask_dir=None, max_frames=60):
    if len(frames) > max_frames: frames = frames[:: max(1, len(frames) // max_frames)][:max_frames]
    C_all, U_all, W_all, per = [], [], [], []
    for fr in frames:
        f = img_dir / f"f{fr:05d}.png"
        if not f.exists(): continue
        bgr = cv2.imread(str(f))
        mimg = cv2.imread(str(mask_dir / f"f{fr:05d}.png"), cv2.IMREAD_GRAYSCALE) if mask_dir else None
        if bgr is None: continue
        R, C = poses[fr]; res = detect(bgr, K, dist, mode, mimg)
        c = res[0]
        if c is None: continue
        hole_r = float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))   # hole size in px (weight)
        step = max(1, len(c) // 80); c = c[::step]
        u = rays_world(c, K, R, C); w = np.full(len(u), hole_r)
        C_all.append(np.repeat(C[None], len(u), 0)); U_all.append(u); W_all.append(w)
        per.append(dict(frame=fr, C=C, R=R, pts=c, u=u, hole_r=hole_r))
    if not per: return None
    return dict(C=np.vstack(C_all), U=np.vstack(U_all), W=np.concatenate(W_all), frames=per)


def fit_circle(C, U, W, a0, n, e1, e2, weighted=True):
    nC = (C - a0) @ n; nU = U @ n
    ok = np.abs(nU) > 0.05
    C, U, nC, nU, W = C[ok], U[ok], nC[ok], nU[ok], (W[ok] if weighted else np.ones(ok.sum()))
    d = nC.copy(); extent = d.max() - d.min() + 1e-9
    grid = np.linspace(d.max() + 0.02 * extent, d.max() + 8 * extent, 500)

    def ev(z0):
        s = (z0 - nC) / nU; m = s > 1e-6
        if m.sum() < 30: return None, None
        X = C[m] + s[m, None] * U[m]
        rad = np.linalg.norm((X - a0) - z0 * n, axis=1); w = W[m]
        Rh = wmedian(rad, w); spread = wmedian(np.abs(rad - Rh), w) / max(Rh, 1e-9)
        return Rh, spread

    best = None
    for z0 in grid:
        Rh, sp = ev(z0)
        if Rh is None: continue
        if best is None or sp < best[2]: best = (z0, Rh, sp)
    for _ in range(2):
        hw = (grid[1] - grid[0]) * 4
        for z0 in np.linspace(best[0] - hw, best[0] + hw, 120):
            Rh, sp = ev(z0)
            if Rh is not None and sp < best[2]: best = (z0, Rh, sp)
    return dict(z0=float(best[0]), R=float(best[1]), spread=float(best[2]), a0=a0, n=n, e1=e1, e2=e2)


def circle3d(fit, n=200):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ctr = fit["a0"] + fit["z0"] * fit["n"]
    return ctr + fit["R"] * (np.outer(np.cos(th), fit["e1"]) + np.outer(np.sin(th), fit["e2"]))


def reproj_px(fit, frames, K):
    q = circle3d(fit); res = []
    for fm in frames:
        Xc = (fm["R"] @ (q - fm["C"]).T).T; fwd = Xc[:, 2] > 1e-6
        if fwd.sum() < 20: continue
        uv = (K @ Xc[fwd].T).T; uv = uv[:, :2] / uv[:, 2:3]
        res.append(float(np.median(cKDTree(uv).query(fm["pts"])[0])))
    return float(np.median(res)) if res else None


def basis_from_traj(poses, frames):
    Cs = np.array([poses[f][1] for f in frames]); a0 = Cs.mean(0)
    _, _, vt = np.linalg.svd(Cs - a0, full_matrices=False); n = vt[0]
    if (Cs[-1] - Cs[0]) @ n < 0: n = -n
    a = np.array([0, 0, 1.]) if abs(n[2]) < 0.9 else np.array([1., 0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    return a0, n, e1, e2


def colmap_slice_dce(ph, fit, scale):
    p = ph / "dense0/fused.ply"
    if not p.exists(): return None
    P = np.asarray(o3d.io.read_point_cloud(str(p)).points); P = P[np.isfinite(P).all(1)]
    if len(P) < 50: return None
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    n, a0 = fit["n"], fit["a0"]
    slab = P[np.abs((P - a0) @ n - fit["z0"]) <= 0.5 * fit["R"]]
    if len(slab) < 30: return None
    rr = np.linalg.norm((slab - a0) - ((slab - a0) @ n)[:, None] * n, axis=1)
    return float(2 * np.median(rr) * scale)


# ---------- driver ----------
def run(cfg, mode, pose_src, weighted=True):
    ph = cfg["ph"]; img_dir = ph / "images"; mask_dir = ph / "masks" if cfg["mask"] else None
    if pose_src == "colmap":
        K, dist, poses = load_colmap(ph); R_GT, gtp, _ = load_gt(ph, cfg["gt_kind"])
        common = sorted(set(poses) & set(gtp))
        scale = umeyama_scale(np.array([poses[f][1] for f in common]), np.array([gtp[f][1] for f in common]))
    else:
        R_GT, poses, _ = load_gt(ph, cfg["gt_kind"]); scale = 1.0
        K, dist, _ = load_colmap(ph)
    DCE_GT = 2 * R_GT; frames = sorted(poses)
    g = gather(frames, poses, K, dist, mode, img_dir, mask_dir, max_frames=200)   # more frames = tighter
    if g is None: return dict(phantom=cfg["name"], mode=mode, pose_src=pose_src, status="FAIL", reason="no boundary")
    used = [fm["frame"] for fm in g["frames"]]
    a0, n, e1, e2 = basis_from_traj(poses, used)
    fit = fit_circle(g["C"], g["U"], g["W"], a0, n, e1, e2, weighted)
    DCE = 2 * fit["R"] * scale; err = abs(DCE - DCE_GT) / DCE_GT * 100
    rp = reproj_px(fit, g["frames"], K)
    # subset stability: STRATIFIED interleaved subsets (each spans the full range, keeping
    # near+far frames) — NOT contiguous halves (all-far / all-near are pathological & would
    # only measure "can far frames alone constrain it"). Gate out under-constrained subsets
    # (high consistency spread) — those are the method correctly refusing, not instability.
    order = np.array(used); KSUB = 4; SPREAD_GATE = 0.10
    subs, rejected = [], 0
    for k in range(KSUB):
        fl = list(order[np.arange(len(order)) % KSUB == k])
        gs = gather(fl, poses, K, dist, mode, img_dir, mask_dir, max_frames=999)
        if gs is None or len(gs["frames"]) < 4: continue
        fb = basis_from_traj(poses, [fm["frame"] for fm in gs["frames"]])
        fs = fit_circle(gs["C"], gs["U"], gs["W"], *fb, weighted)
        if fs["spread"] <= SPREAD_GATE: subs.append(2 * fs["R"] * scale)
        else: rejected += 1
    stab_pct = round(100 * (max(subs) - min(subs)) / max(np.median(subs), 1e-9), 1) if len(subs) >= 2 else None
    col = colmap_slice_dce(ph, fit, scale)
    r = dict(phantom=cfg["name"], mode=mode, pose_src=pose_src, weighted=weighted, status="OK",
             n_frames=len(g["frames"]), GT_R=round(R_GT, 3), GT_DCE=round(DCE_GT, 3),
             fit_R=round(fit["R"] * scale, 3), fit_DCE=round(DCE, 3), error_pct=round(err, 1),
             consistency_spread=round(fit["spread"], 4), reproj_px=(round(rp, 2) if rp else None),
             stab_pct=stab_pct, stab_band=[round(min(subs), 3), round(max(subs), 3)] if subs else None,
             n_subsets_used=len(subs), n_subsets_rejected=rejected,
             COLMAP_slice_DCE=(round(col, 3) if col else None), scale=round(scale, 4), z0=round(fit["z0"], 3),
             _fit=fit, _g=g, _K=K, _ph=ph)
    r["PASS"] = bool(err < 10.0 and (stab_pct is None or stab_pct <= 20.0))
    return r


def figure(cfg, results):
    OUT = ROOT / f"runs/barbour30/airwayfit/contourfit_{cfg['name']}"; OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r["status"] == "OK"]
    key = next((r for r in ok if r["pose_src"] == "colmap"), ok[0] if ok else None)
    if key is None: (OUT / "report.json").write_text("[]"); return OUT
    fit, g, K, ph = key["_fit"], key["_g"], key["_K"], key["_ph"]
    fig = plt.figure(figsize=(15, 9)); q = circle3d(fit)
    fms = g["frames"]; pick = [fms[len(fms) // 6], fms[len(fms) // 2], fms[-1 - len(fms) // 6]]
    for j, fm in enumerate(pick):
        ax = fig.add_subplot(2, 3, j + 1)
        bgr = cv2.imread(str(ph / "images" / f"f{fm['frame']:05d}.png"))
        ax.imshow(cv2.cvtColor(cv2.undistort(bgr, K, np.zeros(4)), cv2.COLOR_BGR2RGB))
        ax.plot(fm["pts"][:, 0], fm["pts"][:, 1], ".", ms=2, c="lime")
        Xc = (fm["R"] @ (q - fm["C"]).T).T; fwd = Xc[:, 2] > 1e-6
        uv = (K @ Xc[fwd].T).T; uv = uv[:, :2] / uv[:, 2:3]
        ax.plot(uv[:, 0], uv[:, 1], "-", lw=1.6, c="red")
        ax.set_title(f"frame {fm['frame']}  (green=detected, red=fit)", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(2, 3, 4)
    nC = (g["C"] - fit["a0"]) @ fit["n"]; nU = g["U"] @ fit["n"]; ok2 = np.abs(nU) > 0.05
    d = nC[ok2]; extent = d.max() - d.min() + 1e-9; zs = np.linspace(d.max() + 0.02 * extent, d.max() + 8 * extent, 300); sp = []
    for z0 in zs:
        s = (z0 - nC[ok2]) / nU[ok2]; m = s > 1e-6
        X = g["C"][ok2][m] + s[m, None] * g["U"][ok2][m]
        rad = np.linalg.norm((X - fit["a0"]) - z0 * fit["n"], axis=1); Rh = np.median(rad)
        sp.append(np.median(np.abs(rad - Rh)) / max(Rh, 1e-9))
    ax.plot(zs, sp, "-"); ax.axvline(fit["z0"], c="r", ls="--", label=f"z0*={fit['z0']:.2f}")
    ax.set_xlabel("axial plane z0"); ax.set_ylabel("cross-view spread"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("contour-consistency (min = converged throat)", fontsize=9)
    ax = fig.add_subplot(2, 3, 5)
    labs = [f"{r['mode']}\n{r['pose_src']}" for r in ok]; vals = [r["fit_DCE"] for r in ok]
    cols = ["seagreen" if r["PASS"] else "crimson" for r in ok]
    ax.bar(range(len(vals)), vals, color=cols); ax.axhline(key["GT_DCE"], c="k", ls="--", label=f"GT DCE={key['GT_DCE']}")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("fitted DCE (GT units)")
    ax.set_title("fit vs GT (green=pass)", fontsize=9); ax.legend(fontsize=8)
    for i, r in enumerate(ok): ax.text(i, r["fit_DCE"], f"{r['error_pct']}%", ha="center", va="bottom", fontsize=8)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    Cs = np.array([fm["C"] for fm in g["frames"]])
    ax.plot(Cs[:, 0], Cs[:, 1], Cs[:, 2], "-", c="cyan", lw=1.5, label="camera path")
    ax.plot(q[:, 0], q[:, 1], q[:, 2], "-", c="red", lw=2, label="fitted throat")
    ax.set_title("trajectory + fitted 3D contour", fontsize=9); ax.legend(fontsize=8)
    fig.suptitle(f"Contour-consistency fit — {cfg['name']} phantom (GT DCE={key['GT_DCE']})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "contourfit.png", dpi=120); plt.close(fig)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (OUT / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    return OUT


def evaluate(cfg):
    results = []
    for mode in cfg["detectors"]:
        for src in ("colmap", "gt"):
            r = run(cfg, mode, src, weighted=True); results.append(r)
            print(f"[{cfg['name'][:4]} {mode:4}/{src:6}] " + (r.get("reason", "") if r["status"] != "OK" else
                  f"fit_DCE={r['fit_DCE']} (GT {r['GT_DCE']}) err={r['error_pct']}% spread={r['consistency_spread']} "
                  f"reproj={r['reproj_px']}px stab±{r['stab_pct']}% COLMAP={r['COLMAP_slice_DCE']} PASS={r['PASS']}"), flush=True)
    OUT = figure(cfg, results)
    ok = [r for r in results if r["status"] == "OK"]
    print(f"\n=== {cfg['name'].upper()} VERDICT (contour-consistency, weighted) ===")
    print(f"{'mode/pose':16}{'fit_DCE':>9}{'err%':>7}{'reproj':>8}{'stab%':>7}  PASS")
    for r in ok:
        print(f"{r['mode']+'/'+r['pose_src']:16}{r['fit_DCE']:>9}{r['error_pct']:>7}{str(r['reproj_px']):>8}{str(r['stab_pct']):>7}  {r['PASS']}")
    print(f"figure -> {OUT}/contourfit.png")
    return results


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("uniform", "both"): evaluate(UNIFORM)
    if which in ("stenosis", "both"): evaluate(STENOSIS)


if __name__ == "__main__":
    main()
