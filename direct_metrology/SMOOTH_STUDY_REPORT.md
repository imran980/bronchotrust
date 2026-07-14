# Smooth-Throat Study — Honest Re-test of the Direct-Metrology Method

**Verdict: FAIL. Do NOT proceed to real video.**

This reverts the crisp-diaphragm shortcut and re-tests the method on the smooth, anatomically
realistic Gaussian stenosis throat with its **true viewpoint-dependent occluding contour**. It does
what the crisp phantom deliberately avoided: it lets the silhouette be real, measures and models the
resulting bias, adds realistic pose noise, adds non-circular cross-sections and nonuniform albedo, and
asks the one question that decides whether the method is worth taking to real data — *does the
photometric term actually beat contour-only?* It does not.

Reproduce: `OMP_NUM_THREADS=1 PYTHONPATH=. python -m direct_metrology.smooth_study`
(outputs `outputs_smooth/results_smooth.json`, `fig_bias_vs_depth.png`, `fig_contour_vs_joint.png`).
All CSA/DCE metrics are computed through the frozen `evaluation/` framework. Nothing was tuned to pass.

---

## 1. The smooth-throat contour bias is real, and it is a geometric silhouette effect

Contour-only circle fit on the **circular** smooth throat, vs camera depth behind the throat:

| depth (mm) | r_fit | CSA err | DCE err |
|---|---|---|---|
| 8  | 3.007 | **+0.46 %** | +0.23 % |
| 10 | 3.021 | **+1.44 %** | +0.72 % |
| 12 | 3.045 | **+3.06 %** | +1.52 % |
| 14 | 3.072 | **+4.84 %** | +2.39 % |

The fit **systematically over-estimates** the throat, and the bias **grows monotonically with depth**.
This is the occluding contour of a smooth surface of revolution: for a converging→diverging profile
`r(z) = r_t + ½κ(z−z_t)²`, the apparent contour forms on the **diverging** side (tangent-from-camera
condition `r = r'(z*)·(z*−z_t+D)`), at a radius `r(z*) > r_t` that increases with depth `D`. The
on-axis tangent-ring model (`tangent_model_r` in the JSON) reproduces the **sign and the monotone
trend** (its magnitude is an upper bound — the multi-view circle fit + oblique cameras recover less
than the full on-axis ring).

**Model (empirical):** `r_fit − r_throat = a·D² + c`, `a = 4.98e-4`, `R² = 0.997`, with local flare
`κ = (R_ref−r_throat)/σ² = 0.1875 /mm`.

**Why this matters:** correcting the bias needs **both** the camera depth `D` **and** the local axial
flare `κ`. The single-cross-section circle fit estimates **neither** (it fits one ring; it does not see
the axial profile). So on the method as designed, **the smooth-throat bias is not resolved** — it can
be *characterised* (as above) but not *removed* without external geometry the method does not have.

## 2. The photometric term does not beat contour-only — anywhere

Best joint (over `λ_photo ∈ {0.02, 0.1, 0.5}`) vs contour-only, per cross-section / albedo case, at
depth 12 mm, with and without realistic COLMAP-bootstrap pose noise (±0.01 mm, ±0.06°):

| case | pose noise | contour-only CSA err | best-joint CSA err | photometry gain | joint < 10 %? |
|---|---|---|---|---|---|
| circular | no | +3.06 % | +3.06 % | −0.00 % | ✅ |
| circular | yes | +3.61 % | +3.61 % | +0.00 % | ✅ |
| ellipse ecc 0.2 | no | **+58.0 %** | +57.8 % | +0.19 % | ❌ |
| ellipse ecc 0.2 | yes | **+59.1 %** | +59.1 % | −0.01 % | ❌ |
| trilobe 0.15 | no | +4.68 % | +4.67 % | +0.01 % | ✅ |
| trilobe 0.15 | yes | +5.32 % | +5.31 % | +0.01 % | ✅ |
| albedo var 0.3 | no | +7.05 % | +7.05 % | +0.01 % | ✅ |
| albedo var 0.3 | yes | +7.64 % | +7.64 % | +0.00 % | ✅ |
| **realistic** (ellipse+lobe+albedo) | no | **+45.1 %** | +45.0 % | +0.08 % | ❌ |
| **realistic** | yes | **+46.1 %** | +46.1 % | +0.04 % | ❌ |

Two independent, decisive failures:

1. **Photometry adds nothing.** The largest gain anywhere is **+0.19 %** — on a **58 %** error. Even
   raising `λ_photo` 25× (to 0.5) moves the radius in the 4th significant figure. The weak
   **gain-invariant angular log-gradient** term carries essentially no throat-radius signal: per-frame
   auto-gain normalisation strips the absolute `1/d²` depth cue, and the residual angular log-gradient
   at a fixed ring is nearly flat in `r`, so there is no gradient to pull the radius toward truth. This
   is consistent with the earlier real-data finding that the ∇log-B cue is stable but does not track
   the narrowing.

2. **The circle cross-section model fails on realistic throats.** An elliptical throat (ecc 0.2) gives
   **+58 % CSA** and the realistic ellipse+lobe+albedo throat gives **+45 %** — the circle latches onto
   the larger apparent dimension of the multi-view silhouettes. Nonuniform vascular albedo alone adds
   **+7 %** (the stripes shift the detected contour) and the photometric term does not correct it.

Realistic pose noise (bootstrap-calibrated) contributes only ~0.5 % — confirming pose uncertainty is
**not** the binding problem here; cross-section shape and the silhouette bias are.

## 3. Pass gate

> **Pass only if the joint method clearly beats contour-only AND stays < 10 % CSA error.**

| requirement | result |
|---|---|
| photometry clearly beats contour-only in every case (≥ 0.5 % CSA gain) | **False** (max +0.19 %) |
| photometry ever decisive (any case) | **False** |
| all cases < 10 % CSA error | **False** (ellipse +58 %, realistic +45 %) |
| smooth-throat bias resolved by the method | **False** |

> **Do not proceed to real video if contour-only still equals joint or if the smooth-throat bias
> remains unresolved.** — Both conditions hold. **DO NOT PROCEED.**

## 4. What this rules in / out (honest)

- **The photometric term, as designed, is not a bias-reducing mechanism.** On synthetic data with a
  *perfectly matched* near-light shading model — the best case it will ever see — it still contributes
  nothing over contour reprojection. It should not be carried to real video as the thing that resolves
  the geometry.
- **The binding problems are geometric, not photometric:** (a) the smooth-throat silhouette bias
  (needs the local axial flare κ → a silhouette-aware / short-profile model, i.e. more than one ring),
  and (b) the circular cross-section assumption (needs an elliptical/lobed model). Neither is the
  photometric term.
- A defensible next method would therefore drop the photometric term, fit a **non-circular** cross
  section, and fit a **short axial segment** (so κ, hence the silhouette, is identified) — but that is
  no longer the "single cross-section, circle, photometric" prototype, and it must clear this same gate
  before any real-video claim.

**No novelty or clinical claim. No real bronchoscopy was used. The gate failed; per instruction the
result is reported as the exact failure rather than tuned until it disappears.**
