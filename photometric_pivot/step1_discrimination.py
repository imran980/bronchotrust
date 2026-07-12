"""STEP 1 -- does the angularly-AGGREGATED, calibrated photometric cue track the KNOWN 2_V2
proximal<distal narrowing? (COLMAP sanity: proximal DCE~1.27 < distal DCE~1.54, ratio~0.82.)

Physics note (honest): for an on-axis tube the depth SHAPE d(alpha)~R/sin(alpha) is scale-free in R,
so within-frame shape CANNOT recover the radius -- the only handle on R is ABSOLUTE wall brightness
at a fixed viewing angle (B(alpha0) ~ I/R^2), which needs the light/exposure gain I to be roughly
CONSTANT across frames. So this step (a) checks exposure consistency, then (b) forms a photometric
radius proxy R_photo ~ 1/sqrt(B at a fixed angular annulus), angularly aggregated, and (c) tests
whether it tracks the COLMAP DCE across a proximal->distal sweep (right ordering, monotonic,
correlated). Scale-free RATIOS only; NO mm; NO GT (COLMAP used as sanity reference)."""
from __future__ import annotations
import json
import numpy as np, cv2
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C

K, DISTC = C.intrinsics(); F = float((K[0, 0] + K[1, 1]) / 2)
SWEEP = list(range(900, 1041, 10))                       # proximal(narrow) -> distal(wide)
ANN_DEG = (7.0, 13.0)                                    # fixed mid-wall angular annulus
# COLMAP sanity anchors (scene units): proximal region DCE~1.27, distal region DCE~1.54
COLMAP_PROX_DCE, COLMAP_DIST_DCE = 1.27, 1.54
PROX_FR, DIST_FR = (900, 945), (1000, 1045)


def frame_metrics(fr, NPHI=180):
    bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g)
    spec = ((g >= 230) & fov)
    B = C.clean_brightness(bgr, g, fov, spec, gamma=1.0)
    ctr = C.lumen_center(g, fov); H, W = g.shape
    Rmax = int(min(ctr[0], ctr[1], W - ctr[0], H - ctr[1]) * 0.95)
    if Rmax < 60: return None
    Bpol = cv2.warpPolar(B.astype(np.float32), (Rmax, NPHI), (float(ctr[0]), float(ctr[1])), Rmax, cv2.WARP_POLAR_LINEAR)
    fovpol = cv2.warpPolar(fov.astype(np.float32), (Rmax, NPHI), (float(ctr[0]), float(ctr[1])), Rmax, cv2.WARP_POLAR_LINEAR) > 0.5
    Bpol[~fovpol] = np.nan
    r = np.arange(Rmax); alpha = np.degrees(np.arctan(r / F))
    # fixed angular annulus: median wall brightness over ALL angles (angular aggregation)
    band = (alpha >= ANN_DEG[0]) & (alpha <= ANN_DEG[1])
    with np.errstate(all="ignore"):
        Bband = np.nanmedian(Bpol[:, band])              # angularly + radially aggregated wall brightness
        wall_all = np.nanmedian(Bpol[:, (alpha >= 5) & (alpha <= 20)])
    if not np.isfinite(Bband) or Bband <= 0: return None
    R_photo = float(1.0 / np.sqrt(Bband))                # photometric radius proxy (exposure-scale k common if I const)
    return dict(frame=fr, B_annulus=float(Bband), wall_brightness=float(wall_all), R_photo=R_photo)


def main():
    rows = [m for m in (frame_metrics(f) for f in SWEEP) if m]
    fr = np.array([r["frame"] for r in rows]); Rp = np.array([r["R_photo"] for r in rows])
    Bw = np.array([r["wall_brightness"] for r in rows])
    # COLMAP reference DCE per frame: linear ramp proximal->distal across the imaged span
    ref = np.interp(fr, [np.mean(PROX_FR), np.mean(DIST_FR)], [COLMAP_PROX_DCE, COLMAP_DIST_DCE])
    # group medians
    def grp(lo, hi, key):
        v = [r[key] for r in rows if lo <= r["frame"] <= hi]; return float(np.median(v)) if v else np.nan
    Rp_prox, Rp_dist = grp(*PROX_FR, "R_photo"), grp(*DIST_FR, "R_photo")
    Bw_prox, Bw_dist = grp(*PROX_FR, "wall_brightness"), grp(*DIST_FR, "wall_brightness")
    rho, p = spearmanr(Rp, ref)                           # does R_photo track COLMAP DCE across the sweep?
    out = dict(test="step1 discrimination -- angularly-aggregated photometric radius proxy vs known narrowing",
               COLMAP_sanity={"prox_DCE": COLMAP_PROX_DCE, "dist_DCE": COLMAP_DIST_DCE, "DCE_ratio_prox/dist": round(COLMAP_PROX_DCE / COLMAP_DIST_DCE, 3)},
               exposure_check={"wall_brightness_prox": round(Bw_prox, 4), "wall_brightness_dist": round(Bw_dist, 4),
                               "ratio_prox/dist": round(Bw_prox / Bw_dist, 3),
                               "note": "if the light/exposure gain were constant AND physics held, a narrower proximal wall (R~0.82x) should be ~1/0.82^2=1.49x BRIGHTER; ~1.0 means auto-gain flattened it"},
               R_photo_prox=round(Rp_prox, 4), R_photo_dist=round(Rp_dist, 4),
               R_photo_ratio_prox_over_dist=round(Rp_prox / Rp_dist, 3),
               spearman_R_photo_vs_COLMAP_DCE=round(float(rho), 3), spearman_p=round(float(p), 4))
    out["ordering_correct_prox_lt_dist"] = bool(Rp_prox < Rp_dist)
    out["ratio_near_colmap_0p82"] = bool(0.65 <= (Rp_prox / Rp_dist) <= 0.95)
    out["tracks_narrowing"] = bool(out["ordering_correct_prox_lt_dist"] and rho >= 0.6 and p < 0.05)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(fr, Rp / np.median(Rp), "o-", color="tab:purple", label="photometric R proxy (norm)")
    ax[0].plot(fr, ref / np.median(ref), "s--", color="k", label="COLMAP DCE (norm, sanity)")
    ax[0].set_xlabel("frame (proximal -> distal)"); ax[0].set_ylabel("normalized (scale-free)")
    ax[0].set_title(f"does photometric R track the widening?\nSpearman ρ={rho:.2f} (p={p:.3f}); prox/dist={Rp_prox/Rp_dist:.2f} vs COLMAP 0.82"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(fr, Bw, "o-", color="tab:orange"); ax[1].set_xlabel("frame (proximal -> distal)")
    ax[1].set_ylabel("angular-median wall brightness"); ax[1].axhline(np.median(Bw), c="0.6", ls=":")
    ax[1].set_title(f"exposure check: wall brightness prox/dist={Bw_prox/Bw_dist:.2f}\n(≈1.0 => auto-gain flattened the R signal)"); ax[1].grid(alpha=.3)
    fig.suptitle("STEP 1 — does the aggregated/calibrated photometric cue track the KNOWN narrowing? (scale-free; NOT clinical)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(C.OUT / "step1_discrimination.png", dpi=120); plt.close(fig)
    out["verdict"] = ("TRACKS the narrowing -> photometric cue is discriminative; proceed to a calibrated photometric-geometric model"
                      if out["tracks_narrowing"] else
                      "does NOT track the narrowing (wrong ordering and/or no correlation) -- consistent with auto-gain flattening the only R handle (absolute brightness); photometry cannot resolve the narrowing on this footage without a scale reference")
    (C.OUT / "step1_discrimination.json").write_text(json.dumps({"per_frame": rows, **out}, indent=2))
    print("=== STEP 1 ==="); print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=2))
    print(f"figure -> {C.OUT}/step1_discrimination.png")


if __name__ == "__main__":
    main()
