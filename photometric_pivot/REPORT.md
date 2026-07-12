# Photometric-depth pivot audit — 2_V2 (go/no-go)

**Scope:** a go/no-go experiment only — *not* a clinical result. No mm scale, no Myer-Cotton grade,
no validation. Pretrained depth was **not** treated as ground truth. All comparisons are directional
sanity checks against our accepted **same-model COLMAP/centerline** result for 2_V2:
distal ring CSA ≈ 1.87 / DCE ≈ 1.54, proximal/narrowest CSA ≈ 1.27 / DCE ≈ 1.27, obstruction
≈ 32 % CSA / 17.5 % diameter (lower bound), **scene units**. The COLMAP result says the proximal
ring is **smaller** than the distal (ratio ≈ 0.68 CSA). Frames tested: proximal 900–940, distal
1000–1040 (the accepted narrowest vs distal-reference regions). Scripts + figures + JSON/CSV in this
folder. The locked COLMAP baseline was **not** modified.

## 1. What was tested
- **Test 1 — photometric cue diagnostic** (`test1_cue_diagnostic.py`): specular + black-bezel
  masking, lumen-center finding, radial brightness profile from center outward; is the near-light
  falloff consistent enough to imply depth ordering?
- **Test 2 — inverse-square near-light baseline** (`test2_inverse_square.py`): relative depth
  `d ~ 1/sqrt(corrected brightness)`, back-project the lumen boundary to a relative 3-D ring, fit an
  ellipse → relative CSA/DCE; compare proximal vs distal ordering, ratio, stability.
- **Test 3 — pretrained depth quick probe** (`test3_pretrained.py`, timeboxed): the cleanly
  runnable option was **Depth-Anything-V2-Small**, a **general, NON-endoscopic** monocular model
  (endoscopy-specific PPSNet/LightDepth need repo clone + weights — deferred). Same back-projection.

## 2. Does photometry carry useful depth signal?
**A raw near-light cue exists, but it is not usable for the measurement.**
- Test 1: the radial brightness rises monotonically from the dark lumen center to the bright wall
  (median monotonicity **0.83**) — so brightness *does* encode near/far. **But it does not
  discriminate the narrowing:** the proximal and distal profiles overlap almost completely and the
  60 %-span radius is essentially equal (prox 313 px vs dist 306 px).
- Test 2: the inverse-square baseline is **stable but gives the WRONG ordering** — relative CSA
  ratio prox/dist = **1.28** (implying **−27.5 %** obstruction, i.e. proximal *wider*), the opposite
  of COLMAP's proximal-narrower ≈ 32 %. The dark-hole **angular sizes are nearly identical**
  (prox 200 px vs dist 209 px) → there is **no size signal**, and the brightness→depth weighting
  pushes it the wrong way.

## 3. Does pretrained depth help?
**No (with the model we could run).** The general Depth-Anything-V2 model gives **wrong ordering**
(ratio **2.08**, proximal 2× *wider*) and is **unstable** (distal CoV **63 %**). Its depth maps
respond to the **bright circular FOV vs dark bezel**, not the airway tube geometry — it is
out-of-distribution on endoscopy. **Caveat:** an *endoscopy-trained* model (PPSNet/LightDepth) was
**not** tested within the timebox; the general-model failure is expected and does not by itself
close that door — but the physics baseline already failing tempers expectations.

## 4. Exact failure modes
1. **No size signal** — proximal and distal lumen apertures have ~equal angular size (200 vs 209 px);
   the narrowing is not distinguishable from lumen appearance alone.
2. **Wrong ordering** — both the inverse-square physics and the general learned model rank the
   proximal ring as *wider* than the distal (COLMAP: narrower).
3. **Irregular lumen** — the proximal dark region is a fold/tail, not a clean aperture, so any
   boundary-based radius is unreliable.
4. **Auto-exposure / gain** — wall brightness is normalized across regions (mean 45 vs 44), which
   flattens the absolute brightness→depth differences the inverse-square model needs.
5. **Out-of-distribution learned depth** — a general monocular model keys on the FOV disk/bezel,
   not the tube; no per-frame photometric calibration (light-falloff model, vignetting, gamma,
   fixed exposure) is available in this footage.

## 4b. Corrected within-frame test (Test 4) — changes the conclusion
`test4_gradient_depth.py`. The prior tests had two real flaws: raw brightness (exposure-dependent)
and cross-frame absolute CSA. The corrected test uses the **radial log-intensity gradient**
`d/dr log(B)` (a multiplicative exposure/gain on B is an additive constant on log B → 0 gradient →
**exposure-invariant**), works **within each frame**, and compares only the **scale-free shape** of
the recovered relative depth against the tube-geometry prediction `d(α) ∝ 1/sin(α)` (α = ray angle
from the lumen axis), across 8 proximal frames (905–940).

**Result — the corrected test does NOT fail:**
- **A stable, tube-geometry-consistent depth cue exists.** The angularly-averaged photometric depth
  follows the tube shape with **R² = 0.885**, and it is remarkably **reproducible across the 8
  frames** (slope CoV **3.4 %**, R² CoV **2.9 %**; the curves overlap). Tests 1–3 missed this
  because raw brightness + cross-frame CSA discard it.
- **But it is aggregate-only and biased.** Per-direction angular consistency is only **0.34** —
  individual radial lines are texture/albedo-dominated, so the cue needs **angular averaging**. And
  the falloff **slope is 0.75, not 1**, with flattening at the wall (oblique-view shading +
  vignetting) — a *biased* version of the ideal tube that would need calibration to be metric.

## 5. Next recommendation
**REVISED: NOT a clean NO-GO — CONDITIONAL. Photometry is not "dead" on this footage, but it is not
directly usable yet.** The corrected within-frame `d/dr log(B)` cue carries a **stable,
tube-geometry-consistent** depth signal (R² 0.885, slope CoV 3 %), reproducible across neighboring
frames — a genuine signal the earlier tests hid. However it is **aggregate-only** (per-direction
texture noise) and **biased** (slope 0.75, wall flattening), and it has **not** been shown to
resolve the actual narrowing.

Concretely, before building any photometric-geometric method:
1. **One more targeted test:** apply the *angularly-aggregated, calibrated* cue and check whether it
   tracks the **known proximal < distal narrowing** (the clinically relevant discrimination this
   within-frame test deliberately did not address). Only a *yes* justifies a full method.
2. **Calibrate the falloff** — model the shading (n·l), vignetting, and the non-unit slope — so the
   biased cue becomes metric/relative-consistent.
3. Keep it as an **auxiliary term to parallax**, not a replacement: COLMAP/parallax remains the
   reliable geometry where parallax is adequate (2_V2 distal; the ~32 % lower-bound obstruction
   stands). Do **not** invest in a full bundle-adjustment photometric method before step 1 passes.

_(The earlier NO-GO in §5 applied to the naive raw-brightness / cross-frame CSA formulation and the
general pretrained model; the corrected within-frame gradient test above supersedes that verdict.)_
