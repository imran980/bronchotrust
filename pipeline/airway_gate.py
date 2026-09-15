"""AirwayGate — a universal validity + normality gate that runs on EVERY video.

SKELETON (v0, not yet threshold-calibrated). The cheap signals are implemented; a few
(periodicity, centerline-graph branching) are clearly stubbed. Thresholds below are
provisional and are meant to be calibrated ONCE from the cohort (e.g. the 20-V2 sweep +
rigid baselines like 21-V1) — never tuned per video.

Design philosophy — why this is still ONE universal method, not a bag of per-video tricks:
  * Every check is an automatic, video-agnostic measurement applied identically to all inputs,
    and is simply inert when not needed (a normal rigid tube passes straight through).
  * The model decides the route; a human never selects settings per case.
  * It is the same "reject rather than fake" philosophy the pipeline already uses
    (coverage >= 60%, 3-estimator agreement, one-connected-model) — extended from
    "is the reconstruction valid?" to "is this case IN SCOPE, and what shape is it?".

Two checks and the routing:

  A. RIGIDITY (pre-MVS, from frames + sparse SfM) — can static SfM even apply?
       RIGID                 : scene rigid enough -> proceed
       QUASI_STATIC_GATEABLE : independent motion is small/cyclic -> phase-gate frames, then proceed
       NON_RIGID             : continuous fast deformation -> OUT OF SCOPE for the static method

  B. SHAPE / TOPOLOGY (post-MVS, from dense cloud + centerline) — normal tube vs abnormal-static?
       NORMAL_TUBE           : single, roughly-circular lumen, smooth CSA
       ABNORMAL_STATIC       : flags in {noncircular, mass_stenosis, multilumen, branch}

  Route:
    RIGID + NORMAL_TUBE     -> standard measurement           (csa_run / centerline_csa)
    RIGID + ABNORMAL_STATIC -> general measurement            (per-lumen / branched centerline)  [IN SCOPE]
    QUASI_STATIC_GATEABLE   -> phase-gated selection -> measure                                   [IN SCOPE]
    NON_RIGID               -> flag out-of-scope; defer to the separate dynamic-airway method
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2

# --- provisional thresholds (CALIBRATE ONCE FROM COHORT DATA; do not tune per video) ---
RIGID_INLIER_MIN = 0.55      # median epipolar inlier ratio above this => rigid two-view geometry
NONRIGID_INLIER_MAX = 0.35   # below this, persistently => independent tissue motion dominates
CIRC_RSTD_MAX = 0.35         # r_std/r_med at or below => acceptably circular (already a pipeline gate)
ECCENTRICITY_MAX = 1.6       # ellipse major/minor above this => noncircular
CSA_JUMP_MAX = 0.6           # |d ln(CSA)| between adjacent slices above this => abrupt (mass/stenosis)


# =====================================================================================
# A. RIGIDITY
# =====================================================================================
def epipolar_inlier_ratio(video, lo, hi, step=3, max_feat=2000):
    """Fraction of matched features between consecutive sampled frames that fit a SINGLE rigid
    two-view geometry (fundamental matrix, RANSAC). Rigid scene -> high; independently moving
    tissue leaves a persistent outlier population -> lower. Returns (median_ratio, n_pairs)."""
    cap = cv2.VideoCapture(video); sift = cv2.SIFT_create(max_feat); bf = cv2.BFMatcher(cv2.NORM_L2)
    ratios = []; prev = None; fi = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > hi:
            break
        if lo <= fi <= hi and (fi - lo) % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(g, None)
            if prev and des is not None and prev[1] is not None and len(kp) > 20 and len(prev[0]) > 20:
                good = [a for a, b in bf.knnMatch(prev[1], des, k=2) if a.distance < 0.8 * b.distance]
                if len(good) >= 15:
                    p0 = np.float32([prev[0][x.queryIdx].pt for x in good])
                    p1 = np.float32([kp[x.trainIdx].pt for x in good])
                    _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 2.0, 0.99)
                    if mask is not None:
                        ratios.append(float(mask.sum()) / len(good))
            prev = (kp, des)
        fi += 1
    cap.release()
    return (float(np.median(ratios)), len(ratios)) if ratios else (float("nan"), 0)


def motion_periodicity(video, lo, hi):
    """STUB: is the independent-motion signal cyclic (pulsation / breathing)? If so the case is
    phase-gateable. Plan: build a per-frame independent-motion magnitude (flow minus the
    ego-motion-consistent component), autocorrelate it, and report the dominant period + peak
    strength. A strong periodic peak => QUASI_STATIC_GATEABLE with a known phase to gate on."""
    return {"periodic": None, "period_frames": None, "peak": None, "implemented": False}


def sparse_fragmentation(num_reg, num_extracted, mean_reproj):
    """Registration completeness + reprojection tail as a secondary non-rigidity signal:
    non-rigid content tends to drop registration and inflate residuals despite good overlap."""
    reg_frac = num_reg / max(num_extracted, 1)
    return {"reg_frac": reg_frac, "mean_reproj": mean_reproj}


def assess_rigidity(inlier_ratio, frag, periodicity):
    """Combine the rigidity signals into a route label. Provisional logic."""
    if not np.isnan(inlier_ratio) and inlier_ratio >= RIGID_INLIER_MIN and frag["reg_frac"] >= 0.8:
        return "RIGID", {"inlier_ratio": inlier_ratio, **frag}
    if not np.isnan(inlier_ratio) and inlier_ratio <= NONRIGID_INLIER_MAX:
        # low rigid consistency: gateable only if the motion is cyclic
        if periodicity.get("periodic"):
            return "QUASI_STATIC_GATEABLE", {"inlier_ratio": inlier_ratio, **frag, **periodicity}
        return "NON_RIGID", {"inlier_ratio": inlier_ratio, **frag, **periodicity}
    # middle band: fragmented but not clearly non-rigid -> likely a coverage/window issue, retry
    return "INDETERMINATE", {"inlier_ratio": inlier_ratio, **frag}


# =====================================================================================
# B. SHAPE / TOPOLOGY  (operates on one cross-section's 2-D ring points, in the slice plane)
# =====================================================================================
def slice_lumen_components(ring_xy, gap_deg=40.0):
    """Count distinct lumen boundaries in one slice. Points are the ring's 2-D coords in the
    slice plane (centered on the centerline point). A single lumen fills all angular sectors;
    a second opening / branch shows as an extra, angularly-separated boundary cluster.
    v0 heuristic: cluster by angular occupancy gaps; return the component count."""
    if len(ring_xy) < 12:
        return 0
    ang = np.sort(np.degrees(np.arctan2(ring_xy[:, 1], ring_xy[:, 0])) % 360)
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    # a well-formed single ring has no large angular void; large voids split components
    return max(1, int((gaps > gap_deg).sum()))


def slice_shape_metrics(ring_xy):
    """Circularity (r_std/r_med — already a pipeline gate) and ellipse eccentricity for one slice."""
    r = np.linalg.norm(ring_xy, axis=1)
    r_med = float(np.median(r)); r_std = float(np.std(r))
    ecc = float("nan")
    if len(ring_xy) >= 5:
        try:
            (_, (MA, ma), _) = cv2.fitEllipse(ring_xy.astype(np.float32))
            ecc = max(MA, ma) / max(min(MA, ma), 1e-6)
        except cv2.error:
            pass
    return {"r_med": r_med, "r_std_over_med": (r_std / r_med if r_med else float("nan")), "eccentricity": ecc}


def csa_abruptness(csa_profile):
    """Largest |Δ ln(CSA)| between adjacent valid slices — a mass/stenosis makes a sharp step."""
    a = np.asarray([c for c in csa_profile if c and c > 0], float)
    if len(a) < 3:
        return float("nan")
    return float(np.max(np.abs(np.diff(np.log(a)))))


def centerline_branching(centerline_graph):
    """STUB: does the medial axis split (a node of degree > 2 = branch / fistula / second lumen
    over multiple slices)? Plan: build the centerline as a graph from the dense cloud's medial
    axis and report nodes with degree > 2. Pairs with slice_lumen_components >= 2 to confirm."""
    return {"branches": None, "implemented": False}


def assess_shape(per_slice_metrics, per_slice_components, csa_profile):
    """Fold the per-slice shape/topology signals into a NORMAL_TUBE vs ABNORMAL_STATIC label."""
    flags = []
    if any(c and c >= 2 for c in per_slice_components):
        flags.append("multilumen")
    eccs = [m["eccentricity"] for m in per_slice_metrics if not np.isnan(m["eccentricity"])]
    if eccs and np.median(eccs) > ECCENTRICITY_MAX:
        flags.append("noncircular")
    jump = csa_abruptness(csa_profile)
    if not np.isnan(jump) and jump > CSA_JUMP_MAX:
        flags.append("mass_stenosis")
    return ("ABNORMAL_STATIC" if flags else "NORMAL_TUBE"), flags


# =====================================================================================
# Top-level orchestration
# =====================================================================================
@dataclass
class GateResult:
    rigidity: str                       # RIGID | QUASI_STATIC_GATEABLE | NON_RIGID | INDETERMINATE
    shape: str = ""                     # NORMAL_TUBE | ABNORMAL_STATIC  (set only if rigidity allows)
    flags: list = field(default_factory=list)
    route: str = ""
    detail: dict = field(default_factory=dict)


def route(rigidity, shape, flags):
    if rigidity == "NON_RIGID":
        return "OUT_OF_SCOPE: defer to dynamic-airway (non-rigid) method"
    if rigidity == "QUASI_STATIC_GATEABLE":
        return "PHASE_GATE then measure"
    if rigidity == "INDETERMINATE":
        return "RETRY_WINDOW (fragmented, not clearly non-rigid)"
    # RIGID
    return "MEASURE_STANDARD" if shape == "NORMAL_TUBE" else "MEASURE_GENERAL (per-lumen / branched)"


def airway_gate(video, lo, hi, rec=None, per_slice_ring_xy=None, csa_profile=None):
    """Run the full gate. `rec` is a pycolmap Reconstruction (for reg_frac/reproj);
    `per_slice_ring_xy` is a list of Nx2 arrays (post-MVS slices); `csa_profile` the CSA-vs-arclength.
    Shape is assessed only when rigidity does not already put the case out of scope."""
    ir, npairs = epipolar_inlier_ratio(video, lo, hi)
    frag = sparse_fragmentation(rec.num_reg_images() if rec else 0,
                                (hi - lo) if rec is None else rec.num_reg_images(),
                                rec.compute_mean_reprojection_error() if rec else float("nan"))
    per = motion_periodicity(video, lo, hi)
    rigidity, rdetail = assess_rigidity(ir, frag, per)

    shape, flags = "", []
    if rigidity in ("RIGID",) and per_slice_ring_xy:
        metrics = [slice_shape_metrics(s) for s in per_slice_ring_xy if len(s)]
        comps = [slice_lumen_components(s) for s in per_slice_ring_xy if len(s)]
        shape, flags = assess_shape(metrics, comps, csa_profile or [])

    return GateResult(rigidity=rigidity, shape=shape, flags=flags,
                      route=route(rigidity, shape, flags),
                      detail={"epipolar_inlier_ratio": ir, "n_pairs": npairs, **rdetail})


if __name__ == "__main__":
    import sys
    v, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print(airway_gate(v, lo, hi))
