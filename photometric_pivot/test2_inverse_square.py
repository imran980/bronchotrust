"""TEST 2 -- simple hand-rolled inverse-square (near-light) depth baseline on 2_V2.
Assume corrected brightness ~ 1/d^2 (near-light) -> RELATIVE depth d ~ 1/sqrt(brightness) (NO mm,
NO absolute scale). Back-project the lumen-boundary ring to a RELATIVE 3D ring, fit an ellipse, and
compute RELATIVE CSA/DCE per frame. Compare proximal/narrowest vs distal reference: ordering
(proximal should be SMALLER), CSA/DCE ratio (COLMAP sanity ~0.68 CSA / ~0.82 DCE), and stability.
This is a go/no-go diagnostic only -- pretrained/GT not used; scene units. """
from __future__ import annotations
import json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C

K, DIST = C.intrinsics(); KINV = np.linalg.inv(K)


def lumen_boundary(g, fov):
    gg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 4); v = gg[fov]
    thr = np.percentile(v, 14); dark = ((gg < thr) & fov).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    if n < 2: return None
    H, W = g.shape; best, bd = None, 1e9
    for i in range(1, n):
        if st[i, 4] < 400: continue
        d = np.hypot(cen[i, 0] - W / 2, cen[i, 1] - H / 2)
        if d < bd: bd, best = d, i
    if best is None: return None
    cnts, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64) if len(c) >= 12 else None


def rel_depth(bgr, g, fov, gamma=0.7):
    spec = ((g >= 230) & fov)                              # only truly-bright specular (rare here)
    b = C.clean_brightness(bgr, g, fov, spec, gamma=gamma)
    b = np.clip(b, 0.02, 1.0)
    return 1.0 / np.sqrt(b)                                # relative depth (per-frame scale)


def measure(fr):
    bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g)
    bnd = lumen_boundary(g, fov)
    if bnd is None: return None
    d = rel_depth(bgr, g, fov)
    # relative depth AT the boundary (sample a few px inward = the visible wall lip)
    ctr = bnd.mean(0)
    inward = ctr + 0.90 * (bnd - ctr)                     # 10% inward toward center = wall lip
    ii = np.clip(inward[:, 1].astype(int), 0, g.shape[0] - 1); jj = np.clip(inward[:, 0].astype(int), 0, g.shape[1] - 1)
    db = d[ii, jj]
    # undistort boundary pixels -> rays -> relative 3D ring P = d * ray
    und = cv2.undistortPoints(bnd.reshape(-1, 1, 2).astype(np.float32), K, DIST, P=K).reshape(-1, 2)
    rays = (KINV @ np.c_[und, np.ones(len(und))].T).T; rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    P = rays * db[:, None]
    # fit ellipse in the plane perpendicular to the ring's mean normal (PCA)
    Pc = P - P.mean(0); _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]; x, y = Pc @ e1, Pc @ e2
    rr = np.hypot(x, y); rm = np.median(rr); keep = (rr > 0.4 * rm) & (rr < 2.0 * rm)
    if keep.sum() < 10: return None
    try:
        (_, _), (MA, ma), _ = cv2.fitEllipse(np.c_[x[keep], y[keep]].astype(np.float32))
        a, bax = MA / 2, ma / 2; csa = float(np.pi * a * bax); dce = float(2 * np.sqrt(a * bax))
    except Exception:
        return None
    return dict(frame=fr, CSA_rel=csa, DCE_rel=dce, r_med=float(rm), ratio=float(np.std(rr[keep]) / max(rm, 1e-9)),
                depth_boundary=float(np.median(db)), angular_r_px=float(np.median(np.hypot(bnd[:, 0] - ctr[0], bnd[:, 1] - ctr[1]))),
                bnd=bnd, ctr=ctr.tolist())


def group(frames):
    return [m for m in (measure(f) for f in frames) if m]


def main():
    P, D = group(C.PROX), group(C.DIST)
    def stat(rows, k): v = np.array([r[k] for r in rows]); return dict(median=float(np.median(v)), cov_pct=round(float(100 * np.std(v) / max(np.median(v), 1e-9)), 1))
    out = dict(test="inverse-square near-light baseline (relative depth; NO mm)",
               proximal={"n": len(P), "CSA_rel": stat(P, "CSA_rel"), "DCE_rel": stat(P, "DCE_rel"), "angular_r_px": stat(P, "angular_r_px")},
               distal={"n": len(D), "CSA_rel": stat(D, "CSA_rel"), "DCE_rel": stat(D, "DCE_rel"), "angular_r_px": stat(D, "angular_r_px")})
    csa_ratio = out["proximal"]["CSA_rel"]["median"] / out["distal"]["CSA_rel"]["median"]
    dce_ratio = out["proximal"]["DCE_rel"]["median"] / out["distal"]["DCE_rel"]["median"]
    out["CSA_ratio_prox_over_dist"] = round(csa_ratio, 3)
    out["DCE_ratio_prox_over_dist"] = round(dce_ratio, 3)
    out["implied_CSA_obstruction_pct"] = round(100 * (1 - csa_ratio), 1)
    out["COLMAP_sanity"] = {"CSA_ratio": 0.68, "DCE_ratio": 0.82, "obstruction_CSA_pct": 32.0}
    out["ordering_prox_lt_dist"] = bool(csa_ratio < 1.0)
    out["ratio_near_colmap"] = bool(0.5 <= csa_ratio <= 0.85)
    out["stable"] = bool(out["proximal"]["CSA_rel"]["cov_pct"] <= 25 and out["distal"]["CSA_rel"]["cov_pct"] <= 25)
    # figure
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.bar([0, 1], [out["proximal"]["CSA_rel"]["median"], out["distal"]["CSA_rel"]["median"]],
           yerr=[out["proximal"]["CSA_rel"]["median"] * out["proximal"]["CSA_rel"]["cov_pct"] / 100,
                 out["distal"]["CSA_rel"]["median"] * out["distal"]["CSA_rel"]["cov_pct"] / 100],
           color=["tab:red", "tab:blue"], capsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["proximal", "distal"]); ax.set_ylabel("relative CSA (photometric)")
    ax.set_title(f"relative CSA (prox/dist={csa_ratio:.2f})\nCOLMAP sanity ratio≈0.68", fontsize=10)
    for k, (rows, col, ttl) in enumerate([(P, "m", "proximal"), (D, "r", "distal")]):
        ax = fig.add_subplot(1, 3, 2 + k)
        fr = rows[len(rows) // 2]; bgr, g = C.load_gray(fr["frame"])
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.plot(fr["bnd"][:, 0], fr["bnd"][:, 1], ".", ms=2, c="lime"); ax.plot(fr["ctr"][0], fr["ctr"][1], "m+", ms=12)
        ax.set_title(f"{ttl} f{fr['frame']}  angular_r={fr['angular_r_px']:.0f}px\nrel CSA={fr['CSA_rel']:.3f} DCE={fr['DCE_rel']:.3f}", fontsize=9); ax.axis("off")
    fig.suptitle("TEST 2 — inverse-square photometric depth: relative CSA (scene units; NOT clinical)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(C.OUT / "test2_inverse_square.png", dpi=120); plt.close(fig)
    (C.OUT / "test2_inverse_square.json").write_text(json.dumps(out, indent=2))
    print("=== TEST 2 ==="); print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k in ("proximal", "distal", "COLMAP_sanity")}, indent=2))
    print(f"figure -> {C.OUT}/test2_inverse_square.png")


if __name__ == "__main__":
    main()
