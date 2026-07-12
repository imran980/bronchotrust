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

## 5. Next recommendation
**NO-GO. Do not build a full photometric-geometric airway model on this footage now.** Per the
decision rule (wrong ordering + a general learned model unstable ⇒ photometry not reliable without
calibration). Concretely:
- Photometry-**alone** is not a substitute for parallax here; the COLMAP/parallax path remains the
  reliable one where parallax is adequate (2_V2 distal; the ~32 % lower-bound obstruction stands).
- If photometry is revisited later (low priority), the *minimum* prerequisites are (a) a real
  **endoscopy-trained** depth model (PPSNet/LightDepth) tested properly, and (b) **photometric
  calibration** — a measured light-falloff model, vignetting correction, and fixed/known exposure —
  none of which this footage provides. Do **not** invest in a full bundle-adjustment photometric
  method before those are in hand.
