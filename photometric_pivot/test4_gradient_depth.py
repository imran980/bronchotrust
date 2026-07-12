"""TEST 4 -- CORRECTED within-frame photometric depth test on 2_V2 proximal/narrowest frames.
Fixes the prior flaws: (a) use the RADIAL LOG-intensity gradient d/dr log(B) -- invariant to per-frame
exposure/gain (a multiplicative gain on B is an additive constant on log B -> 0 gradient); (b) work
WITHIN each frame only; (c) compare SCALE-FREE shape / reprojection consistency, not cross-frame CSA.

Physics: near-light + Lambertian, B ~ rho*(n.l)/d^2 -> log B = const - 2 log d (+ albedo/shading).
Relative depth (up to scale, within frame): d_photo ~ 1/sqrt(B). For a straight tube viewed down its
axis, the wall at angular offset alpha from the axis is at distance d_geom = R/sin(alpha) -> a
SCALE-FREE prediction d_geom(alpha) ~ 1/sin(alpha). We test whether the photometric depth follows
that shape (log-log slope ~1, high R^2), whether the radial gradient is ANGULARLY CONSISTENT
(geometry, not texture), and whether these are STABLE across neighboring frames.
NO mm, NO GT, NO cross-frame absolute CSA."""
from __future__ import annotations
import json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C

K, DISTC = C.intrinsics(); F = float((K[0, 0] + K[1, 1]) / 2)
FRAMES = [905, 910, 915, 920, 925, 930, 935, 940]        # proximal/narrowest window


def analyze(fr, NR=90, NPHI=180):
    bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g)
    spec = ((g >= 230) & fov)
    B = C.clean_brightness(bgr, g, fov, spec, gamma=1.0)   # corrected brightness in [0,1], NaN outside FOV
    ctr = C.lumen_center(g, fov)
    H, W = g.shape
    Rmax = int(min(ctr[0], ctr[1], W - ctr[0], H - ctr[1]) * 0.95)
    if Rmax < 60: return None
    Lpol = cv2.warpPolar(np.log(np.clip(B, 1e-3, 1.0)).astype(np.float32), (Rmax, NPHI),
                         (float(ctr[0]), float(ctr[1])), Rmax, cv2.WARP_POLAR_LINEAR)   # (NPHI, Rmax)
    fovpol = cv2.warpPolar(fov.astype(np.float32), (Rmax, NPHI), (float(ctr[0]), float(ctr[1])), Rmax, cv2.WARP_POLAR_LINEAR) > 0.5
    Lpol[~fovpol] = np.nan
    r = np.arange(Rmax)
    # radial log-gradient per angle; angular median + relative MAD (consistency = geometry vs texture)
    Ls = cv2.GaussianBlur(Lpol, (1, 15), 0)                # smooth along radius
    grad = np.gradient(Ls, axis=1)
    with np.errstate(all="ignore"):
        gmed = np.nanmedian(grad, axis=0); gmad = np.nanmedian(np.abs(grad - gmed[None, :]), axis=0)
        Bmed_r = np.nanmedian(np.exp(Lpol), axis=0)        # angular-median brightness vs radius
    # valid radial band: outside the dark lumen core, inside the bezel
    valid = np.isfinite(Bmed_r) & (r > 0.10 * Rmax) & (r < 0.92 * Rmax) & (Bmed_r > 0.03)
    if valid.sum() < 20: return None
    rv = r[valid]; Bv = Bmed_r[valid]
    d_photo = 1.0 / np.sqrt(Bv)                            # relative depth up to scale (within frame)
    alpha = np.arctan(rv / F)                              # ray angle from the (lumen-center) axis
    d_geom = 1.0 / np.sin(np.clip(alpha, 1e-3, None))      # tube prediction (scale-free shape)
    # scale-free SHAPE match: log d_photo vs log d_geom  (slope ~1, high R^2 = follows tube geometry)
    x = np.log(d_geom); y = np.log(d_photo); A = np.polyfit(x, y, 1); yhat = np.polyval(A, x)
    ss = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-9)
    # angular consistency: fraction of the valid band with positive median gradient AND low rel-MAD
    gm = gmed[valid]; gd = gmad[valid]; rel = gd / (np.abs(gm) + 1e-6)
    consistency = float(((gm > 0) & (rel < 1.0)).mean())
    return dict(frame=fr, geom_slope=float(A[0]), geom_R2=float(ss), consistency=consistency,
                rv=rv.tolist(), d_photo=(d_photo / d_photo[0]).tolist(), d_geom=(d_geom / d_geom[0]).tolist(),
                alpha=alpha.tolist(), ctr=[float(ctr[0]), float(ctr[1])], Rmax=Rmax)


def main():
    rows = [r for r in (analyze(f) for f in FRAMES) if r]
    def arr(k): return np.array([r[k] for r in rows])
    sl, r2, cons = arr("geom_slope"), arr("geom_R2"), arr("consistency")
    def stat(v): return dict(mean=round(float(np.mean(v)), 3), std=round(float(np.std(v)), 3),
                             cov_pct=round(float(100 * np.std(v) / max(abs(np.mean(v)), 1e-9)), 1))
    verdict = dict(n_frames=len(rows), geom_slope=stat(sl), geom_R2=stat(r2), consistency=stat(cons),
                   note="slope~1 & high R^2 => photometric relative depth follows the tube shape d~1/sin(alpha); "
                        "consistency = radial log-gradient positive & angularly consistent (geometry, not texture)")
    # a cue is USABLE within-frame if it matches tube geometry, is angularly consistent, and STABLE
    verdict["shape_match"] = bool(np.median(r2) >= 0.8 and 0.5 <= np.median(sl) <= 1.6)
    verdict["angularly_consistent"] = bool(np.median(cons) >= 0.6)
    verdict["stable_across_frames"] = bool(stat(r2)["cov_pct"] <= 20 and stat(sl)["cov_pct"] <= 25 and stat(cons)["cov_pct"] <= 30)
    verdict["cue_usable_within_frame"] = bool(verdict["shape_match"] and verdict["angularly_consistent"] and verdict["stable_across_frames"])
    # figure
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    for r in rows:
        ax.plot(np.degrees(r["alpha"]), r["d_photo"], "-", alpha=.6, color="tab:red")
    ax.plot(np.degrees(rows[0]["alpha"]), rows[0]["d_geom"], "k--", lw=2, label="tube prediction 1/sin(α)")
    ax.plot([], [], "tab:red", label="photometric 1/√B (per frame)")
    ax.set_xlabel("ray angle α from lumen axis (deg)"); ax.set_ylabel("relative depth (normalized)")
    ax.set_title("within-frame photometric depth vs tube geometry (scale-free)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 2)
    for r in rows: ax.plot(np.log(r["d_geom"]), np.log(r["d_photo"]), ".", ms=3, alpha=.4)
    lim = [min(np.log(rows[0]["d_geom"])), 0]; ax.plot(lim, lim, "k--", label="slope 1")
    ax.set_xlabel("log d_geom (1/sinα)"); ax.set_ylabel("log d_photo (1/√B)")
    ax.set_title(f"shape match: slope={verdict['geom_slope']['mean']}±{verdict['geom_slope']['std']}, "
                 f"R²={verdict['geom_R2']['mean']}", fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 3)
    ax.bar(range(len(rows)), cons, color="tab:purple"); ax.axhline(0.6, c="g", ls="--", label="usable 0.6")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["frame"] for r in rows], rotation=45, fontsize=7)
    ax.set_ylabel("angular consistency of ∇log B"); ax.set_title(f"consistency per frame (stable? CoV={verdict['consistency']['cov_pct']}%)", fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("TEST 4 — CORRECTED within-frame ∇log(B) depth cue (2_V2 proximal; scale-free; NOT clinical)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(C.OUT / "test4_gradient.png", dpi=120); plt.close(fig)
    (C.OUT / "test4_gradient.json").write_text(json.dumps({"per_frame": [{k: v for k, v in r.items() if k in ("frame", "geom_slope", "geom_R2", "consistency")} for r in rows], "verdict": verdict}, indent=2))
    print("=== TEST 4 verdict ==="); print(json.dumps(verdict, indent=2))
    print(f"figure -> {C.OUT}/test4_gradient.png")


if __name__ == "__main__":
    main()
