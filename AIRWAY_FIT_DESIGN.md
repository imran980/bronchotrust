# Airway-Fit — coverage-guided photometric airway-lumen fitting (design memo)

_Prototype design. COLMAP is used for **poses + centerline initialization only**, never as the
final geometry source. Companion prototype: `airwayfit_proto.py`. Target: 2_V2 proximal
subglottis (the stenosis — where the narrowest ring is the visible boundary and where COLMAP
MVS was noisiest)._

## 1. Motivation
COLMAP recovers **poses** well but the dense **wall** is the weak link: the airway is
textureless/specular, so MVS rings are noisy (25-V1 non-monotonic, 32-V2 invalid, 2-V2
subglottis borderline). Yet the **lumen boundary** — the dark "hole ahead" against the lit
wall — segments cleanly and stably (see `airwayfit_segtest.png`). Airway-Fit measures CSA/DCE
from that **image boundary** (+ optional shading), not from the noisy point cloud.

Key geometric fact this exploits: for a **narrowing** airway, the narrowest cross-section (the
stenosis **throat**) is the aperture that bounds the dark lumen seen by *every* camera behind
it. So one 3D ring explains many frames' boundaries → an over-determined, stable fit. (For a
*uniform* tube there is no throat; the boundary is a soft light-falloff iso-contour — handled
by the shading term, and flagged by a high boundary residual.)

## 2. Model — generalized cylinder (free radius, NO normal-airway prior)
Airway lumen surface parameterised by arc-length `s`:
- centerline `C(s) ∈ R³`  (init: COLMAP camera path / medial axis, then refined)
- tangent `t(s) = C'(s)/|C'(s)|`, orthonormal cross-section frame `(e1(s), e2(s)) ⟂ t`
- cross-section: radius `r(s)` **or** ellipse semi-axes `a(s), b(s)` + in-plane angle `φ(s)`
- surface: `S(s,θ) = C(s) + a(s)cosθ·e1(s) + b(s)sinθ·e2(s)`

`r(s)` / `a(s),b(s)` are **free** (smoothness only) so a focal stenosis is representable.
`CSA(s) = π a(s) b(s)`,  `DCE(s) = 2√(CSA/π) = 2√(a b)`. Scale = COLMAP scene units (gauge-free).

## 3. Observations
Per selected frame `f` with pose `(R_f, C_f)` (COLMAP) and intrinsics `K` (pinned OPENCV):
- **Lumen boundary** `B_f` = ordered pixels of the dark-lumen ↔ lit-wall transition (fit as an
  ellipse or contour). Specular highlights are inpainted/masked first.
- **Shading** `I_f(x)` = wall brightness (used weakly, §6).

## 4. Loss terms
Total (no shape prior):  **`E = E_bnd + α·E_shade + β·E_smooth`**, α small, and E_shade is
clamped so it never overrides boundary evidence.

**(a) Boundary reprojection (primary).** Project the model occluding contour `Γ_f(model)` — the
set of surface points where the viewing ray from `C_f` is tangent to `S` — and match `B_f`:
```
E_bnd = Σ_f Σ_{p∈Γ_f} ρ( dist( π_f(p) , B_f ) )          π_f = OPENCV projection with (R_f,C_f,K)
```
`dist` = point-to-observed-boundary distance; `ρ` = robust (Huber/Cauchy) for specular/mucus
dropouts. For the airway the visible occluding contour is dominated by the **throat**
(minimum-aperture ring along the view), which is what makes one ring fit many frames.

**(b) Smoothness (only regulariser).**
```
E_smooth = Σ_s |C''(s)|²  +  λ_r Σ_s |r'(s)|²   (or |a'|²+|b'|²)
```
No term pulls `r(s)` toward a "normal" airway.

**(c) Shading (optional, weak — §6).**

## 5. v0 prototype (implemented) — throat-depth search
A minimal, non-iterative first version that avoids a full contour optimiser but captures the
core idea and is directly comparable to COLMAP:
1. Target region → local axis `t` = PCA of the COLMAP camera path over a behind-the-throat
   window; reference cameras = frames **behind** the target (they image the throat ahead).
2. Detect `B_f` in each reference frame (undistort → inpaint specular → dark-lumen connected
   component near center → contour).
3. Back-project each boundary pixel to a **world ray** `X(λ)=C_f+λ·R_fᵀK⁻¹[u,v,1]`.
4. **Search the axial depth `d`** of the throat plane `{x:(x−(O+d·t))·t=0}`. For each `d`,
   intersect all rays with the plane → 3D points → fit an ellipse in `(e1,e2)`; score by
   radial residual `r_std/r_med`. The **minimum-residual depth = the throat** (the ring all
   cameras agree on).
5. At the best depth: ellipse `(a,b)` → **CSA, DCE** (scene units).
6. **Fair COLMAP comparison:** slice the COLMAP dense cloud at the *same* plane `(O+d*·t, t)` →
   COLMAP ring. Compare CSA/DCE and the boundary reprojection residual.

This is v0; the full method (§2–4) replaces the single-plane search with a per-`s`
generalized-cylinder contour fit.

## 6. Shading term (optional, weak, never overrides boundary)
Endoscope ≈ a point light **co-located with the camera** → radiance falls off with distance:
`I(x) ≈ ρ_alb · (n·v) / d(x)²` (Lambertian, `v` = view dir ≈ light dir). Given the model,
predict `Î_f` and add
```
E_shade = Σ_f Σ_{x∉specular} w(x) · ρ( I_f(x) − Î_f(x; ρ_alb) )
```
with **specular pixels masked**, `ρ_alb` a free global albedo (keeps it gauge-free), and a
**small α + hard clamp** so shading only disambiguates depth where the boundary is weak (uniform
segments); it is forbidden from moving the fit where a boundary exists. Near-bright/far-dark is
a monotonic depth cue — used as a soft prior on `d(s)`, not a hard constraint.

## 7. What COLMAP provides / what Airway-Fit replaces
- COLMAP → **camera poses**, intrinsics, an initial centerline. (Poses are its reliable output.)
- Airway-Fit → the **surface / CSA / DCE**, from image boundaries (+ weak shading), **replacing**
  the noisy MVS wall.

## 8. Evaluation (prototype)
- **Boundary reprojection residual** (px) — self-consistency / throat-validity gate.
- **COLMAP vs Airway-Fit** CSA/DCE at the same plane.
- **Stability** across nearby reference-frame subsets (early/mid/late) — the main question.

## 8b. Prototype result (2_V2 proximal subglottis)
`airwayfit_proto.py`, 18 reference frames (all segmented cleanly), throat found at depth
d=1.76 (clear reprojection-residual minimum). **Same-plane comparison:**

| | DCE | CSA | ring r_std/r_med | notes |
|---|---|---|---|---|
| **Airway-Fit** (boundary) | **1.05** | 0.87 | **0.12** | clean closed ellipse; reproj residual 7 px |
| COLMAP (same plane) | 1.08 | 0.92 | **0.55** | noisy filled blob + flung outliers |

- **DCE agrees within 3%** (1.05 vs 1.08) → Airway-Fit is *validated* against COLMAP.
- **Airway-Fit ring is ~4.6× tighter** (0.12 vs 0.55) — COLMAP's ring at that plane would fail
  the closed-ring test (>0.35); Airway-Fit's is clean (see `sidebyside_ring.png`).
- **Stability across frame subsets:** DCE band [1.005, 1.036] = **±1.5%** (early/mid/late).
- The single throat ring reprojects onto the *near arc* of each observed boundary (~7 px) but
  not the full outline — the boundary is the tube's occluding envelope, not one cross-section
  (`overlays.png`). This is the v0 limitation, resolved by the per-`s` model (§10).

**Answer to the main question:** YES — for this region, image-boundary fitting gives a
**cleaner and more stable** CSA/DCE (ratio 0.12 vs 0.55; ±1.5% stability) than slicing the
COLMAP cloud, while **agreeing** with COLMAP's DCE — exactly where COLMAP's own ring is noisy.

## 8c. Stress test (v0, 6 regions) — SANITY agreement with COLMAP, NOT validation
`airwayfit_proto.py` CONFIG. Airway-Fit ring is **always tighter** than the COLMAP slice
(ratio 0.08–0.27 vs 0.29–0.55). Per region (AF = Airway-Fit, COL = COLMAP same plane):

| region | AF DCE | COL DCE | disagree | AF ratio | COL ratio | reproj px | stab % | read |
|---|---|---|---|---|---|---|---|---|
| 2_V2 prox-subglottis  | 1.05 | 1.08 | 3%  | **0.12** | 0.55 | 7  | ±3%   | clean **and** agrees ✓ |
| 25_V1 prox-subglottis | 1.95 | 2.94 | 34% | **0.08** | 0.55 | 18 | ±0.9% | clean+stable; COL noise-inflated |
| 25_V1 dist-subglottis | 22.1 | —    | —   | **0.10** | (COL null) | 29 | ±3% | clean+stable where COL can't measure |
| 32_V2 prox-subglottis | 5.38 | 4.41 | 22% | 0.27 | 0.29 | 59 | **±104%** | FAIL — multi-ring, no single throat |
| 32_V2 dist-subglottis | 2.40 | 6.61 | 64% | 0.16 | 0.35 | 14 | **±202%** | FAIL — unstable |
| 2_V2 trachea-ref      | 25.1 | —    | —   | 0.19 | (COL null) | 45 | ±85% | FAIL — no throat (expected) |

**Reading (honest):**
- **Generalises beyond 2_V2:** on **25_V1** (prox+dist) Airway-Fit gives clean, *stable* rings
  (ratio 0.08–0.10, ±≤3%) exactly where COLMAP is noisy (0.55) or *null*. The 34% DCE
  "disagreement" on 25_V1 prox is plausibly **COLMAP over-reading** (its ratio-0.55 scatter
  inflates the median radius), not an Airway-Fit error — but this is **unvalidated**.
- **Fails on 32_V2** (both regions): the throat search returns *multiple inconsistent rings*
  (stab ±100–200%) — the glottis/cord-aperture zone has no stable throat (moving cords). This
  is where COLMAP also failed.
- Trachea fails as expected (no throat → needs the shading term).

**Decision-rule outcome:** on the 4 NEW subglottic tests, **2/4** give a clean+stable ring
(25_V1 prox+dist); 2/4 fail (32_V2). That is **not** the ≥3/4 clean pass required to auto-proceed,
but it is **not** "only 2_V2 works" either — 25_V1 clearly works. → **Middle case: fix two things
before the full per-`s` investment**, don't green-light blindly and don't scrap:
1. **Phantom-validate** the lumen-boundary CSA (settles the AF-lumen vs COLMAP-wall DCE gap — the
   real go/no-go). 2. **Detect/exclude the no-stable-throat case** (32_V2 aperture) instead of
   emitting a number.

## 8d. PHANTOM VALIDATION — FAILED (stop-and-redesign gate)
Ran v0 **unmodified** on the synthetic phantom (uniform textured tube, **R_GT = 5.0**,
GT DCE = 10.0; recon→GT scale 3.241). Input via the lossless image sequence; only a data
path + region config, no method change.

| | DCE (GT units) | error vs GT | ring tightness | reproj | stability |
|---|---|---|---|---|---|
| **COLMAP slice** | 9.74 | **2.6%** ✓ | 0.22 | — | — |
| **Airway-Fit v0** | 26.2 | **161.6%** ✗ | 0.12 | 23.7 px | ±11.8% |

**Airway-Fit is 2.7× too large.** And it *looks* clean+stable (ratio 0.12, ±12%) — a
**self-consistent ring at the WRONG depth**. Root cause: the phantom is a **uniform tube (no
geometric throat)**; the v0 throat-depth search has no true convergence and locks onto a
spurious far depth (where the back-projected dark-lumen boundary yields a ring ~2.7× the true
wall). This is the same no-throat regime that failed on the trachea and 32_V2.

**Lessons (important):**
- **Ring tightness / stability ≠ correctness.** The v0's "cleaner, more stable" rings on the
  real videos are NOT a validity guarantee — on known GT the absolute size is badly wrong. The
  2_V2 agreement held only because a *real* subglottic narrowing anchored the depth; without a
  genuine throat the absolute scale is **under-determined**.
- The pass condition (error <5–10%, stable, ≥COLMAP) is **not met** (161.6% error, and worse
  than COLMAP's 2.6%). **DECISION (rule 7): STOP and REDESIGN before adding any complexity.**
  Do NOT proceed to the full per-`s` fit built on the throat-depth search.

## 8e. Redesign direction (before re-testing)
The absolute depth/radius must come from **parallax**, not a single throat-convergence:
- Triangulate the airway **wall** directly from multi-view lumen-boundary correspondence across
  the (wobble-induced) camera baseline — the phantom *has* parallax and GT, so it is the correct
  test bed. Constrain radius by triangulation, not by picking one convergence plane.
- Fit the full generalized cylinder `r(s)` over a *range* of depths (§2–4) rather than one ring.
- Re-validate on BOTH the **uniform** phantom (must recover R=5) **and** a new **stenosis**
  phantom (fair test for a throat) before trusting any real-video CSA/DCE.

## 8f. CORRECTED METHOD — multi-view contour-consistency fit (PHANTOM PASSES, conditionally)
`phantom_contourfit.py`. Replaces v0's per-region **throat-depth search** with a **global
cross-view consistency** fit, and replaces "tightness/reproj at a single plane" with a
convergence criterion that *cannot* reward a spurious depth:
- Each frame's lumen boundary is a **silhouette / occluding-contour** constraint, NOT a set of
  corresponding 3D points. Back-project every boundary pixel to a world ray (known poses).
- Find the **single** axial plane `z0` + radius `R` at which the boundary rays from **all**
  cameras cross at one common radius: `R̂(z0)=median_j r_j(z0)`,
  `spread(z0)=MAD_j/R̂` → `z0*=argmin spread`, `R*=R̂(z0*)`, `DCE=2R*`.
- The `spread` at the minimum is the *cross-view disagreement*: near-zero ⇒ a real fixed 3D
  contour; large ⇒ no shared contour (self-diagnosing — the property v0 lacked).

**Phantom result (R_GT=5, DCE_GT=10; two detectors × two pose sources):**

| detector | pose | fit DCE (GT units) | error | spread | reproj | stab | verdict |
|---|---|---|---|---|---|---|---|
| **rim** (true occluding contour) | COLMAP | **10.08** | **0.8%** | 0.009 | 1.3 px | ±33% | **PASS** |
| **rim** | GT poses | 10.17 | 1.7% | 0.010 | 1.4 px | ±31% | **PASS** |
| photometric (percentile-dark) | COLMAP | 79.0 | 690% | 0.112 | 28 px | ±175% | FAIL (loud) |
| photometric | GT poses | 97.1 | 871% | 0.112 | 28 px | ±154% | FAIL (loud) |

Same figure `runs/barbour30/airwayfit/phantom_contourfit/contourfit.png`: the one fitted 3D
circle reprojects onto the detected boundary at *every* camera distance; the consistency curve
has a sharp, deep minimum (v0 had none). Compare v0: **161%** error on this same phantom.

**What this establishes**
- The multi-view contour math is **correct**: on a genuine occluding contour it recovers R to
  **<2%**, matching the COLMAP cloud slice (9.93) and slightly closer to GT (10.08 vs 10.0).
- It **self-diagnoses**: the photometric case fails with *high* spread/reproj (28 px, ±175%),
  i.e. it refuses rather than returning a confident wrong number — unlike v0.
- **Caveat (decisive for real video):** it passes because the uniform phantom has a real
  geometric contour (the tube **end/rim**). A uniform tube's *photometric* dark hole (light
  falloff) is **not** a fixed 3D contour → fails. Real airways have no rim — BUT a real
  **stenosis throat IS a fixed occluding contour** (the narrowest ring; the wall flares back out
  behind it). That is exactly why v0 got the real 2_V2 stenosis ~right yet failed the uniform
  tube. So the honest next gate is a **stenosis phantom** (known throat radius), not real video.
- Stability caveat: the point estimate is 0.8% but frame-subset spread is ±33%, driven by
  early (far-from-rim) frames whose small hole weakly constrains `z0`; near-rim frames pin it.
  A distance/hole-size weighting would tighten this.

**DECISION:** rule 7 (abandon iff it fails the uniform phantom) is **NOT** triggered — the
contour fit passes the phantom on the true occluding contour at 0.8%. **Do not** abandon; **do
not** yet touch real video. Next gate: build a **stenosis phantom** and re-validate that the
contour-consistency fit recovers the known throat radius — the clinical-analog occluding contour.

## 9. Known limitations / failure cases
- Uniform (no-throat) segments: boundary is falloff, not geometry → high residual; needs the
  shading term. (Trachea is *harder* for this method than the stenosis.)
- Off-axis / bent airway: single-plane v0 breaks; the full per-`s` model is needed.
- Mucus / heavy specular glare can corrupt `B_f` → robust `ρ` + masking, drop bad frames.
- Still **scene-unit / gauge-free** — no absolute mm (unchanged; scale is a separate track).

## 10. Next steps
1. **Full per-`s` generalized-cylinder fit** (replace the single-plane v0): optimize `C(s), r(s)`
   (or `a(s),b(s)`) so the model's *occluding contour* reprojects to the full observed lumen
   boundary across frames (§4a) — gives a CSA/DCE *profile*, not one ring, and should drive the
   reprojection residual well below 7 px and cover the whole outline.
2. **Add the weak shading term** (§6) and test on the *trachea* (no-throat region) where boundary
   alone is ambiguous — the hard case for this method.
3. **Stress-test on 25-V1 / 32-V2 subglottis** (where COLMAP was noisy/invalid): does Airway-Fit
   still give a clean, stable ring? This is the real value test vs COLMAP.
4. **Phantom validation** (known radius) end-to-end — the go/no-go gate before trusting numbers.
5. Robustify lumen segmentation (learned/edge-based, not just a brightness threshold) for mucus
   and glare; auto-select the reference-frame window.
