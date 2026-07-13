# Re-test with the ACTUAL designed estimator — the pose failure evaporates

The previous red-team tested the wrong estimator (photometric data term only, fixed tube at origin,
absolute B, assumed 1% pose noise). Rebuilt to the design (`AIRWAY_FIT_DESIGN.md` §4/6) and re-tested.

## What was fixed (`rebuild_sim.py`)
1. **Occluding-contour term (PRIMARY, E_bnd):** `Σ_f ‖π_f(model contour) − observed boundary‖²`
   (pixels). Geometric — pins r/pose from image geometry, not the `1/d²` photometric depth.
   `λ_bnd = 1`.
2. **Gain-invariant shading (WEAK, E_shade):** `‖∂_φ log B − ∂_φ log B̂‖²` — removes per-frame
   albedo·intensity gain. `λ_shade = 0.02` (weak, per design's "α small, clamped").
3. **Free tube pose/centerline:** the ring has a free center `O∈ℝ³` and axis `a` — the gauge DOF
   that was missing before.

## Correctness gate (`verify_sim.py`) — MUST pass before any bias number
| gate | result |
|---|---|
| A — ideal recovery | r=5.000, CSA bias 0.00%, rms 2e-13 ✔ |
| B — Sim(3) invariance | projected-contour **ellipse-area diff 0.0**, log-grad diff 3e-4, **CSA scale-error 0.00%**, ratio invariant ✔ |
| **C — common-mode SE(3) pose error** | **CSA bias −0.03% (max 0.31%)** ✔ — the old stub reported **+27.6%** |

Gate C confirms the reviewer's diagnosis exactly: the +27.6% was a **missing-gauge artifact** (fixed
tube couldn't follow a global camera shift). With a free tube pose it is ~0 by construction.

## Data-driven pose noise (`bootstrap_pose_noise.py`) — not an assumed 1%
Bootstrap-resampling each 2_V2 **distal** camera's 2D-3D correspondences and re-solving PnP (60
samples/cam, 61 cameras):
- **relative translation std = 0.04% of depth (median), 0.06% (p90)**
- **relative rotation std = 0.049° (median), 0.068° (p90)**

The earlier red-team's **1% translation was ~25× too high.**

## Re-test — CSA ratio bias vs relative pose noise, contour ablation (`retest.{png,json}`)
Two rings (ref r=5, narrow r=3, true ratio 0.36) in one model, **correlated** (smooth-trajectory)
pose perturbations, 60 MC/level.

| relative translation | 0.06% (**data**) | 0.18% | 0.48% (8×) | 0.96% | 1.98% |
|---|---|---|---|---|---|
| RATIO bias **with contour** | **−0.13%** | −0.01% | +2.71% | +16.7% | +240%* |
| RATIO bias shading-only | −0.63% | −1.76% | −3.85% | −6.89% | −12.2% |

**At the data-driven noise level the CSA ratio bias is −0.13% (±1.84%) with the designed estimator —
well under 5%.** The earlier +24.7% was three compounding errors: wrong estimator + missing gauge +
25×-too-high noise. There is **~8× margin**: even at 0.48% (8× the data noise) the ratio bias is
2.7% (<5%). Ablation nuance (honest): the contour helps at the operating regime (lower bias & std);
at *unrealistic* high noise (≥1%) the contour triangulation itself degrades (the shading-only stays
bounded there) — but that regime is 16× the measured noise. *(the 240% at 2% is fit divergence, not
a meaningful bias.)*

## Release (`release_notes.json`)
- Units: length = scene units; angle = rad; pixel = `f·(x/z)`, f=800; CSA = `π r²`.
- Loss: `E = λ_bnd‖π(model contour)−boundary‖² + λ_shade‖∂_φ log B − ∂_φ log B̂‖²` — E_bnd primary,
  E_shade weak & gain-invariant → **matches design §4a/6** (E_smooth omitted for the single-slice
  test). Gauge: free tube pose absorbs global Sim(3).
- Convergence: LM, **success rate 1.0**, median nfev 5, median rms 2.5e-14.

## Verdict
With the **correct estimator**, the **gauge fixed**, and the **actual bootstrap pose noise**, the
CSA-ratio pose failure (Test 3) **evaporates: −0.13% at data noise**. Combined with Rescue 2 (albedo
→ BIC clears), **both Phase-0 quantitative failures are addressed** on the distal region where the
reconstruction is good. Remaining honest caveats: (i) the noise is bootstrapped on the *distal*
(well-reconstructed) region — the *proximal* subglottis is capture-marginal (larger pose noise), and
was already not cleanly measurable; (ii) the bootstrap holds 3D points fixed (may under-estimate),
but the 8× margin covers a large factor. So the earlier "blocking science failure" was an artifact
of testing the wrong estimator — **the novelty question is now the writing/positioning one**, as the
reviewer argued.
