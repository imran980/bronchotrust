"""CORRECTED method: multi-view CONTOUR-consistency tube fit (NOT throat-depth search,
NOT boundary-point triangulation).

Idea: the 2D lumen boundary in each frame is treated as a SILHOUETTE / occluding-contour
constraint, not as a set of physical points shared across views. We back-project every
boundary pixel to a 3D ray (known poses), then find the SINGLE global cylinder-end circle
(axial plane z0 along the tube axis + radius R) at which the boundary rays from ALL cameras
converge to one common radius. R is recovered by cross-view CONSISTENCY of a fixed 3D contour,
not by a per-region depth search and not by assuming pixel correspondence.

  crossing-radius of ray_j at plane z0  r_j(z0) = | (C_j + s u_j - a0) - z0*n |   (s>0)
  R_hat(z0) = median_j r_j ;  spread(z0) = MAD_j / R_hat        (cross-view disagreement)
  z0* = argmin spread ;  R* = R_hat(z0*)   -> DCE = 2R*

Runs on the phantom ONLY (R_GT=5, DCE_GT=10). Two boundary detectors:
  (A) rim  = genuine occluding contour (escaped-ray hole edge)  -> tests the geometry/math
  (B) phot = percentile-dark lumen boundary (what real airways force) -> tests generalization
Two pose sources: COLMAP sparse/0 (realistic, converted to GT units by Umeyama scale) and GT
poses (isolates method error from pose error).

Report: GT R/DCE, fitted R/DCE, error %, reprojection residual (px), stability across frame
subsets, and the COLMAP cloud-slice DCE for reference. Pass iff DCE error < 5-10%.
depth-eval env.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")
PH = ROOT / "runs/phantom"
IMG = PH / "images"
OUT = ROOT / "runs/barbour30/airwayfit/phantom_contourfit"


# ---------- poses / geometry ----------
def umeyama_scale(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, Ds, Vt = np.linalg.svd((Yc.T @ Xc) / len(X)); S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    return float((Ds * np.diag(S)).sum() / ((Xc ** 2).sum() / len(X)))


def load_colmap():
    rec = pycolmap.Reconstruction(str(PH / "sparse/0")); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); dist = np.array([k1, k2, p1, p2])
    poses = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]
        poses[int(re.search(r"f(\d+)", im.name).group(1))] = (R, -R.T @ M[:3, 3])
    return K, dist, poses


def load_gt():
    gt = json.loads((PH / "gt_poses.json").read_text())
    poses = {p["frame"]: (np.array(p["R_wc"]), np.array(p["C"])) for p in gt["poses"]}
    return gt["R_GT"], poses


# ---------- boundary detection ----------
def _central_component(mask, W, H, min_area=300):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    best, bd = None, 1e9
    for i in range(1, n):
        if st[i, 4] < min_area: continue
        d = np.hypot(cen[i, 0] - W / 2, cen[i, 1] - H / 2)
        if d < bd: bd, best = d, i
    if best is None: return None
    # reject if it touches the image border (open hole -> not the rim)
    ys, xs = np.where(lab == best)
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2: return None
    cnts, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64) if len(c) >= 12 else None


def detect(bgr, K, dist, mode):
    und = cv2.undistort(bgr, K, dist); g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY); H, W = g.shape
    if mode == "rim":                                   # genuine occluding contour: escaped-ray (near-black) hole
        mask = (g <= 10).astype(np.uint8)
    else:                                               # photometric: percentile-dark lumen (real-airway analog)
        body = g[g > 6]
        if body.size < 1000: return None, und
        mask = ((g < np.percentile(body, 18)) & (g > 3)).astype(np.uint8)
    c = _central_component(mask, W, H)
    return c, und


def rays_world(pts, K, R, C):                            # undistorted px -> unit world rays
    Kinv = np.linalg.inv(K)
    d_cam = (Kinv @ np.c_[pts, np.ones(len(pts))].T).T
    d = (R.T @ d_cam.T).T
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


# ---------- the global contour-consistency fit ----------
def gather(frames, poses, K, dist, mode, max_frames=60):
    if len(frames) > max_frames: frames = frames[:: max(1, len(frames) // max_frames)][:max_frames]
    C_all, U_all, per_frame = [], [], []
    for fr in frames:
        f = IMG / f"f{fr:05d}.png"
        if not f.exists(): continue
        bgr = cv2.imread(str(f))
        if bgr is None: continue
        R, C = poses[fr]; c, und = detect(bgr, K, dist, mode)
        if c is None: continue
        step = max(1, len(c) // 80); c = c[::step]        # thin the contour
        u = rays_world(c, K, R, C)
        C_all.append(np.repeat(C[None], len(u), 0)); U_all.append(u)
        per_frame.append(dict(frame=fr, C=C, R=R, pts=c, u=u))
    if not per_frame: return None
    return dict(C=np.vstack(C_all), U=np.vstack(U_all), frames=per_frame)


def fit_circle(C, U, a0, n, e1, e2):
    nC = (C - a0) @ n; nU = U @ n
    ok = np.abs(nU) > 0.05                                # drop near-axis-parallel rays
    C, U, nC, nU = C[ok], U[ok], nC[ok], nU[ok]
    d = nC.copy(); extent = d.max() - d.min() + 1e-9
    grid = np.linspace(d.max() + 0.02 * extent, d.max() + 8 * extent, 500)

    def eval_z0(z0):
        s = (z0 - nC) / nU
        m = s > 1e-6
        if m.sum() < 30: return None, None
        X = C[m] + s[m, None] * U[m]
        rad = np.linalg.norm((X - a0) - z0 * n, axis=1)
        Rh = np.median(rad); spread = np.median(np.abs(rad - Rh)) / max(Rh, 1e-9)
        return Rh, spread

    best = None
    for z0 in grid:
        Rh, sp = eval_z0(z0)
        if Rh is None: continue
        if best is None or sp < best[2]: best = (z0, Rh, sp)
    # refine around the minimum
    for _ in range(2):
        z0b = best[0]; hw = (grid[1] - grid[0]) * 4
        for z0 in np.linspace(z0b - hw, z0b + hw, 120):
            Rh, sp = eval_z0(z0)
            if Rh is None: continue
            if sp < best[2]: best = (z0, Rh, sp)
    z0, R, spread = best
    return dict(z0=float(z0), R=float(R), spread=float(spread), a0=a0, n=n, e1=e1, e2=e2)


def circle3d(fit, n=120):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ctr = fit["a0"] + fit["z0"] * fit["n"]
    return ctr + fit["R"] * (np.outer(np.cos(th), fit["e1"]) + np.outer(np.sin(th), fit["e2"]))


def reproj_px(fit, frames, K):
    q = circle3d(fit, 200); res = []
    for fm in frames:
        R, C = fm["R"], fm["C"]; Xc = (R @ (q - C).T).T
        fwd = Xc[:, 2] > 1e-6
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


def colmap_slice_dce(fit, scale):
    p = PH / "dense0/fused.ply"
    if not p.exists(): return None
    P = np.asarray(o3d.io.read_point_cloud(str(p)).points); P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    n, a0 = fit["n"], fit["a0"]
    slab = P[np.abs((P - a0) @ n - fit["z0"]) <= 0.6 * fit["R"]]
    if len(slab) < 30: return None
    rr = np.linalg.norm((slab - a0) - ((slab - a0) @ n)[:, None] * n, axis=1)
    return float(2 * np.median(rr) * scale)


# ---------- driver ----------
def run(mode, pose_src):
    if pose_src == "colmap":
        K, dist, poses = load_colmap(); R_GT, gtp = load_gt()
        common = sorted(set(poses) & set(gtp))
        scale = umeyama_scale(np.array([poses[f][1] for f in common]),
                              np.array([gtp[f][1] for f in common]))
    else:
        R_GT, poses = load_gt(); scale = 1.0
        K, dist, _ = load_colmap()                       # intrinsics identical (pinned)
    DCE_GT = 2 * R_GT
    frames = sorted(poses)
    g = gather(frames, poses, K, dist, mode)
    if g is None: return dict(mode=mode, pose_src=pose_src, status="FAIL", reason="no boundary detected")
    used = [fm["frame"] for fm in g["frames"]]
    a0, n, e1, e2 = basis_from_traj(poses, used)
    fit = fit_circle(g["C"], g["U"], a0, n, e1, e2)
    R_gt_units = fit["R"] * scale; DCE = 2 * R_gt_units
    err = abs(DCE - DCE_GT) / DCE_GT * 100
    rp = reproj_px(fit, g["frames"], K)
    # stability across frame subsets
    subs, order = [], np.array(used)
    for idx in (np.arange(len(order)) % 3 == 0, np.arange(len(order)) % 3 == 1,
                np.arange(len(order)) % 3 == 2, np.arange(len(order)) < len(order) // 2,
                np.arange(len(order)) >= len(order) // 2):
        fl = list(order[idx])
        gs = gather(fl, poses, K, dist, mode, max_frames=999)
        if gs is None or len(gs["frames"]) < 4: continue
        fa0, fn, fe1, fe2 = basis_from_traj(poses, [fm["frame"] for fm in gs["frames"]])
        fs = fit_circle(gs["C"], gs["U"], fa0, fn, fe1, fe2)
        subs.append(fs["R"] * scale)
    stab_pct = round(100 * (max(subs) - min(subs)) / max(np.median(subs), 1e-9), 1) if len(subs) >= 2 else None
    col_dce = colmap_slice_dce(fit, scale)
    r = dict(mode=mode, pose_src=pose_src, status="OK", n_frames=len(g["frames"]),
             GT_R=R_GT, GT_DCE=DCE_GT, fit_R=round(R_gt_units, 3), fit_DCE=round(DCE, 3),
             error_pct=round(err, 1), consistency_spread=round(fit["spread"], 4),
             reproj_px=(round(rp, 2) if rp else None), stab_pct=stab_pct,
             stab_band=[round(min(subs), 3), round(max(subs), 3)] if subs else None,
             COLMAP_slice_DCE=(round(col_dce, 3) if col_dce else None),
             scale=round(scale, 4), z0=round(fit["z0"], 3), _fit=fit, _g=g, _K=K)
    r["PASS"] = bool(err < 10.0)
    return r


def figure(results):
    OUT.mkdir(parents=True, exist_ok=True)
    key = next((r for r in results if r.get("mode") == "rim" and r.get("pose_src") == "colmap" and r["status"] == "OK"), None)
    if key is None: key = next((r for r in results if r["status"] == "OK"), None)
    if key is None: return
    fit, g, K = key["_fit"], key["_g"], key["_K"]
    fig = plt.figure(figsize=(15, 9))
    # top row: 3 sample frames with detected boundary (green) + reprojected fit circle (red)
    q = circle3d(fit, 200)
    fms = g["frames"]; pick = [fms[len(fms) // 6], fms[len(fms) // 2], fms[-1 - len(fms) // 6]]
    for j, fm in enumerate(pick):
        ax = fig.add_subplot(2, 3, j + 1)
        bgr = cv2.imread(str(IMG / f"f{fm['frame']:05d}.png"))
        und = cv2.undistort(bgr, K, np.zeros(4))  # already-pinned K; show raw for context
        ax.imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
        ax.plot(fm["pts"][:, 0], fm["pts"][:, 1], ".", ms=2, c="lime")
        R, C = fm["R"], fm["C"]; Xc = (R @ (q - C).T).T; fwd = Xc[:, 2] > 1e-6
        uv = (K @ Xc[fwd].T).T; uv = uv[:, :2] / uv[:, 2:3]
        ax.plot(uv[:, 0], uv[:, 1], "-", lw=1.6, c="red")
        ax.set_title(f"frame {fm['frame']}  (green=detected, red=fit)", fontsize=9); ax.axis("off")
    # consistency curve
    ax = fig.add_subplot(2, 3, 4)
    nC = (g["C"] - fit["a0"]) @ fit["n"]; nU = g["U"] @ fit["n"]; ok = np.abs(nU) > 0.05
    d = nC[ok]; extent = d.max() - d.min() + 1e-9
    zs = np.linspace(d.max() + 0.02 * extent, d.max() + 8 * extent, 300); sp = []
    for z0 in zs:
        s = (z0 - nC[ok]) / nU[ok]; m = s > 1e-6
        X = g["C"][ok][m] + s[m, None] * g["U"][ok][m]
        rad = np.linalg.norm((X - fit["a0"]) - z0 * fit["n"], axis=1); Rh = np.median(rad)
        sp.append(np.median(np.abs(rad - Rh)) / max(Rh, 1e-9))
    ax.plot(zs, sp, "-"); ax.axvline(fit["z0"], c="r", ls="--", label=f"z0*={fit['z0']:.2f}")
    ax.set_xlabel("axial plane z0 (recon units)"); ax.set_ylabel("cross-view spread (MAD/R)")
    ax.set_title("contour-consistency (min = converged rim)", fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # per-mode/pose bar of DCE vs GT
    ax = fig.add_subplot(2, 3, 5)
    ok_r = [r for r in results if r["status"] == "OK"]
    labs = [f"{r['mode']}\n{r['pose_src']}" for r in ok_r]; vals = [r["fit_DCE"] for r in ok_r]
    cols = ["seagreen" if r["PASS"] else "crimson" for r in ok_r]
    ax.bar(range(len(vals)), vals, color=cols); ax.axhline(key["GT_DCE"], c="k", ls="--", label=f"GT DCE={key['GT_DCE']}")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("fitted DCE (GT units)"); ax.set_title("fit vs GT (green=pass)", fontsize=9); ax.legend(fontsize=8)
    for i, r in enumerate(ok_r): ax.text(i, r["fit_DCE"], f"{r['error_pct']}%", ha="center", va="bottom", fontsize=8)
    # 3D: trajectory + fitted rim circle
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    Cs = np.array([fm["C"] for fm in g["frames"]])
    ax.plot(Cs[:, 0], Cs[:, 1], Cs[:, 2], "-", c="cyan", lw=1.5, label="camera path")
    ax.plot(q[:, 0], q[:, 1], q[:, 2], "-", c="red", lw=2, label="fitted rim")
    ax.set_title("trajectory + fitted 3D contour", fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("Multi-view contour-consistency tube fit — phantom (R_GT=5, DCE_GT=10)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "contourfit.png", dpi=120); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("rim", "phot"):
        for src in ("colmap", "gt"):
            r = run(mode, src); results.append(r)
            print(f"[{mode:4} / {src:6}] " + (r["reason"] if r["status"] != "OK" else
                  f"fit_DCE={r['fit_DCE']} (GT {r['GT_DCE']}) err={r['error_pct']}% "
                  f"spread={r['consistency_spread']} reproj={r['reproj_px']}px stab±{r['stab_pct']}% "
                  f"COLMAP_slice={r['COLMAP_slice_DCE']} PASS={r['PASS']}"), flush=True)
    figure(results)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (OUT / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    ok = [r for r in results if r["status"] == "OK"]
    rim_colmap = next((r for r in ok if r["mode"] == "rim" and r["pose_src"] == "colmap"), None)
    phot_colmap = next((r for r in ok if r["mode"] == "phot" and r["pose_src"] == "colmap"), None)
    print("\n=== VERDICT (phantom, contour-consistency fit) ===")
    print(f"{'mode/pose':16}{'fit_DCE':>9}{'err%':>7}{'reproj':>8}{'stab%':>7}  PASS")
    for r in ok:
        print(f"{r['mode']+'/'+r['pose_src']:16}{r['fit_DCE']:>9}{r['error_pct']:>7}{str(r['reproj_px']):>8}{str(r['stab_pct']):>7}  {r['PASS']}")
    print("\nPass condition: DCE error < 10% on the phantom.")
    if rim_colmap and rim_colmap["PASS"]:
        print("RIM (true occluding contour) PASSES -> the multi-view contour math is CORRECT.")
        if phot_colmap and not phot_colmap["PASS"]:
            print("BUT photometric-boundary detection FAILS -> a uniform tube's dark hole is NOT a fixed")
            print("3D contour; real airways (no rim) are the photometric case -> generalization caveat.")
    else:
        print("RIM contour FAILS the phantom -> per rule 7, ABANDON this contour method.")
    (OUT / "report.json")


if __name__ == "__main__":
    main()
