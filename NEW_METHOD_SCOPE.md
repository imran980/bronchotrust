# COLMAP baseline — close-out summary + next-method scope

_Branch: `barbour-30frame-audit`. Companion to the memory notes (barbour-30frame-audit,
barbour-pct-obstruction, barbour-scale-not-recoverable, 16v1-ct-validation, gluemap-*-reject).
COLMAP tuning is STOPPED as of this document._

---

## Part 1 — COLMAP baseline: what works, what fails, why it can't reach Barbour-level

### What works
- **The pipeline is correct.** Phantom test recovers a known tube radius to **1.8%** → errors on real
  video are data-limited, not algorithmic.
- **Poses are reliable.** Pinned OPENCV intrinsics (no refine) + CLAHE + exhaustive matching + MVS
  gives coherent airway tubes in the trachea, and — with smart frame selection — in the subglottis.
- **Frame SELECTION is the decisive lever.** Naive contiguous 30-frame windows fail the subglottis;
  SMART selection (CLAHE + sharpness/glare gating + coverage-diversity FPS + adaptive ±30/±20/±45)
  closes clean subglottic ring **shapes**. Pipeline result: **8/12 landmarks accepted**,
  `prox_subglottis` **3/3 reproducible** across videos.
- **One connected global model = one common scale** → a scale-free % obstruction is obtainable where
  the rings are clean. **2-V2: ~33% area (band 24–38%), clean monotonic narrowing** (trustworthy).
- CLAHE is required (raw frames fail to register on the low-texture subglottis).

### What fails
- **Absolute mm: not recoverable.** No static, non-specular, known-size object sits in a connected
  model with the airway. The 4 mm Hopkins shaft is a *moving, specular* instrument (SfM can't
  reconstruct it); tube bores/rims are *specular/overexposed* (scatter, no fittable cylinder/circle);
  no laryngoscope slot is reconstructable; the telescope can't self-image. → **DCE stays scene units.**
- **Ring-radius scale doesn't transfer across local batches.** The camera Sim(3) bridge is *stable*
  (CoV 0.008–0.095), but low-parallax local batches distort ring **radius** vs camera baseline, so a
  stable camera alignment does **not** carry ring size (local-bridge % disagreed with global-direct:
  −31% vs +50%). Separate 30-frame gauges cannot be metrically combined.
- **Low parallax / dwelling scope collapses.** 16_v1 fragments/collapses (301/437 coincident cameras
  under a lenient mapper) → its CT validation **could not be completed** (region imaged, reconstruction
  is the limit).
- **Subglottis is the hard region.** Low texture + low parallax + partial angular coverage. Even the
  *global-model* subglottic rings are frequently noisy: 25-V1 caliber bounces (non-monotonic),
  **32-V2 subglottic rings all invalid** (r_std/r_med 0.43–0.91) → % not obtainable.
- **Scale-free % obstruction is reliable on only 1 of 3 videos** (2-V2). 25-V1 weak/noisy; 32-V2 fails.
- GlueMap does not help (rejected on 3 videos: unpinnable focal, worse reproj, outlier cams, prior-
  flattened rings).

### Why it cannot reach Barbour-level clinical validation on our data
Barbour-level = absolute-mm CSA/DCE at glottis/subglottis → Myer–Cotton grade, validated vs CT/op-note.
- **No metric scale** in a connected model → no absolute mm; the blade-gated mm plan cannot proceed
  because the blade is off-screen / not reconstructable (specular, moving, or absent).
- **The achievable fallback (scale-free % obstruction) is itself thin**: wide bands (24–38%), the
  reference is the *patient's own trachea* (not a normative airway), the true-narrowest subglottis is
  often partially imaged (<100% coverage) or noisy, and only 1/3 videos give a trustworthy result.
- **Root cause is the DATA, not the algorithm** (phantom proves the pipeline). Existing clinical clips
  are captured for viewing, not photogrammetry: (1) textureless, specular, mucus-coated walls →
  feature-SfM MVS is noisy/fails on the surface; (2) low-parallax dwelling/forward scope → depth
  under-constrained; (3) no in-frame static known-size object → no metric scale; (4) partial angular
  coverage of the narrowest rings. **Feature-SfM + MVS is the wrong tool for textureless/specular
  surfaces** — it nails the *poses* but the dense *wall* is the weak link.

**Verdict:** COLMAP is a solid, correct **baseline** (reliable poses, clean ring shapes with smart
selection, a scale-free % on the best video), but it **cannot deliver Barbour-level absolute-mm grades**
on this data. Stop tuning it.

---

## Part 2 — Next method: smart selection + silhouette/shading lumen fitting

### Core idea
**Decouple POSE (COLMAP is good) from SURFACE (COLMAP is poor on textureless walls).** Use COLMAP only
for **camera poses** and **smart frame selection** as *initialization*; recover the lumen surface from
image cues that do **not** need texture:
- **Silhouette** — the endoluminal occluding contour (the boundary of the dark lumen ahead / where the
  wall curves out of view) per frame.
- **Shading** — endoscope light falloff (near = bright, far = dark, ∝ 1/d²) → a dense depth cue on
  textureless walls.

Fit a **parametric airway lumen model** (centerline + per-cross-section radius/ellipse) that best
explains the multi-view silhouettes + shading, regularized to a smooth tube. Output **CSA/DCE per
cross-section** (scene units; mm still needs a scale anchor — unchanged).

### Why this can beat COLMAP MVS here
- Feature-SfM/MVS needs texture; the wall is textureless/specular → noisy/fails. **Silhouette + shading
  are texture-free.**
- **Poses reconstruct fine even when the wall doesn't** → fix them as input.
- A **tube prior** regularizes the ill-posed textureless problem (the airway *is* a smooth tube),
  giving robust CSA/DCE even from partial coverage — exactly where MVS rings went noisy (25-V1, 32-V2).

### Components
1. **Frame selection** — reuse `barbour30_smart` (quality + coverage-diversity). No change.
2. **Poses** — from the COLMAP smart-batch / global model; fixed initialization.
3. **Lumen-contour detection (silhouette)** — per frame, segment the dark lumen-ahead region and the
   wall occluding contour; mask specular highlights and mucus.
4. **Photometric / shading model** — endoscope ≈ a point light co-located with the camera;
   radiance ≈ albedo·cosθ / d². Shape-from-shading → per-pixel wall distance (up to albedo). The
   co-located light + inverse-square falloff is the classic, comparatively well-posed endoscopy SfS
   setup (near-bright/far-dark is a strong monotonic depth cue).
5. **Lumen surface model** — generalized cylinder: centerline curve + cross-section (radius or ellipse)
   as a function of arc-length; smoothness + near-circularity priors.
6. **Multi-view fit** — optimize the tube params to match (a) projected occluding contours and
   (b) shading-derived depths across the selected frames, with the priors, poses fixed.
7. **Output** — `CSA(s) = π r(s)²` (or ellipse area), `DCE(s) = 2√(CSA/π)`, per arc-length. Scale-free.

### Key challenges / open questions
- **Endoluminal silhouette ≠ classic silhouette.** You are *inside* the tube; the apparent contour is
  where the concave wall self-occludes, not an external object outline. Need occluding-contour geometry
  for endoluminal views (the "hole ahead" boundary = projection of the far-visible cross-section).
- **SfS is ill-posed**, mitigated by co-located light + 1/d² falloff + the tube prior; but specular
  highlights, mucus, and inter-reflections need masking + robust losses. Non-Lambertian mucosa is a risk.
- **Lighting/albedo calibration** — recoverable up to a global scale (still gauge-free); relative depth
  suffices for scale-free CSA *ratios*.
- **Scale is unchanged** — the new method improves SURFACE robustness (denser, cleaner CSA), NOT metric
  scale. Absolute mm stays capture-protocol-gated (a matte static known-size object imaged with the
  airway) — a separate track.
- **Validation** — the phantom (known radius) is the go/no-go gate, exactly like the COLMAP phantom test.

### Prototype plan (first steps, in order)
1. **Phantom feasibility** — on the known-radius phantom: detect lumen contour + shading depth, fit the
   tube model, recover the radius. **Gate: <5% radius error** (vs COLMAP's 1.8%).
2. **Real prototype on 2-V2 subglottis** (best case) with COLMAP poses — compare the shading/silhouette
   CSA profile to the COLMAP MVS one.
3. **Stress test on 25-V1 / 32-V2 subglottis** (where MVS rings were noisy/invalid) — does the tube-
   model fit give a clean CSA where MVS could not? This is the real value test.
4. If it clears 1–3, build the full pipeline, phantom-validate, and re-run % obstruction on all videos.

### What COLMAP hands off to the new method
- Reliable **camera poses** (fixed input) and the **smart frame selector** (already built).
- The **scale-free % baseline** (2-V2 ~33%) to beat.
- The **phantom harness** to validate the new method.
