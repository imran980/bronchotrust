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

## 8g. STENOSIS PHANTOM — contour-consistency fit PASSES (the clinical-analog gate)
`stenosis_phantom.py` (render + COLMAP) + `phantom_contourfit.py stenosis`. A textured tube with
a known **normal** radius Rn=5 and a known **throat** radius Rt=3 (smooth Gaussian constriction at
z=42; GT throat DCE=6.0). Long tube (len 150) so the throat is the *only* dark occluding contour
(a finite end would add a competing r=Rn escape-rim; real airways have none). Camera flies behind
the throat → the narrowest ring is a fixed 3D occluding contour. COLMAP: 220/220 registered, one
model, 2.4M dense. Requirement-4 weighting (rays weighted by hole size) + **stratified,
spread-gated** subset stability (interleaved subsets, not contiguous halves — all-far/all-near
halves are pathological and self-flag with high spread).

| detector | pose | fit DCE (GT=6.0) | error | spread | reproj | stab | verdict |
|---|---|---|---|---|---|---|---|
| **geom** (true throat occluding contour) | COLMAP | 6.32 | **5.3%** | 0.042 | 3.3 px | ±0.8% | **PASS** |
| **geom** | GT poses | 6.32 | 5.3% | 0.042 | 3.2 px | ±1.3% | **PASS** |
| **edge** (occlusion-edge, PHOTOMETRIC — real-video path) | COLMAP | 6.50 | **8.3%** | 0.045 | 4.0 px | ±1.2% | **PASS** |
| **edge** | GT poses | 6.47 | 7.8% | 0.044 | 3.9 px | ±0.6% | **PASS** |
| phot (brightness threshold) | either | ~149 | ~2380% | 0.23 | 39 px | — | FAIL (loud) |
| COLMAP dense slice @ throat | — | 6.16–6.37 | 2.6–6.1% | — | — | — | (baseline agrees) |

Figure `runs/barbour30/airwayfit/contourfit_stenosis/contourfit.png`: one global throat circle
reprojects onto the detected boundary at every distance; sharp consistency minimum at z0*.

**Decisive lessons**
- **The fit recovers a known internal throat to 5–8%, stable to ±1%** — validated on the clinical
  analog, not just a tube end. Meets rule 5 (error <5–10% AND stability acceptable). **PASS.**
- **Detection is the make-or-break, and it is NOT brightness.** On a *smooth* stenosis the dark
  region is the **dim approaching-constriction wall** (larger, view-varying, not a fixed contour)
  → every brightness/percentile/near-black detector fails at 200–2400% (and self-flags: spread
  0.13–0.23). The throat must be found as the **occlusion EDGE** — the innermost strong bright→dark
  radial gradient (`throat_edge`, polar-unwrap + peak radial gradient) — which passes at 8% from
  images alone. This is the single most important transferable finding for real video.
- Weighting + stratified spread-gated subsets turned the earlier ±327% (contiguous halves) into
  ±1%. The all-far / all-near halves genuinely can't constrain the throat and correctly self-flag.

**DECISION:** stenosis phantom **PASSES** → cleared (per the plan) to test on real 2_V2 / 25_V1,
using the **occlusion-edge** detector (brightness thresholding is disqualified). Real-video is the
true stress of the edge detector under mucus/glare/partial throats — report as *sanity agreement
with COLMAP*, not validation.

## 8h. REAL VIDEO (2_V2 / 25_V1) — sanity test: does NOT transfer yet (self-diagnosed)
`realvideo_contourfit.py`. Ran the phantom-validated fit + occlusion-edge detector on real
subglottis regions, using the **exact extracted frames on disk** (`runs/batch4/<v>/images` by
name → no POS_FRAMES drift) with the baseline COLMAP poses. Scene units; SANITY vs the COLMAP
dense slice — **not validation** (no GT here).

| region | fit DCE (scene) | COLMAP slice | agree | spread | reproj | sanity |
|---|---|---|---|---|---|---|
| 2_V2 prox-subglottis | 0.10 (degenerate) | — (no slice) | — | **0.34** | 122 px | ✗ |
| 25_V1 prox-subglottis | 0.99 | 1.59 | 38% | **0.33** | 126 px | ✗ |
| 25_V1 dist-subglottis | 3.76 | 2.14 | 76% | **0.19** | 40 px | ✗ |

Figure `runs/barbour30/airwayfit/realvideo_contourfit/realvideo.png`. Compare the phantom:
spread **0.04**, reproj **3–4 px**. On real video spread is **5–9×** worse and the consistency
curves have **no clean minimum** → the per-frame detected edges do **not** back-project to one
fixed 3D circle. The method **refuses** (high spread/reproj) rather than returning a confident
wrong number — the designed-in self-diagnosis (unlike v0).

**Why (visible in the figure):** the real subglottic lumen boundary is irregular / non-circular,
often multi-lobed, and the dark region wanders frame-to-frame (mucus, glare, folds, oblique
scope). The occlusion-edge detector traces *a* boundary each frame, but not the *same* fixed 3D
throat ring across frames — so there is nothing consistent to converge to. This is exactly the
bottleneck the stenosis phantom flagged: the fit is only as good as a **consistent** occluding
contour, and real anatomy/imaging does not hand one to a brightness-gradient detector.

**Status / decision.** The multi-view contour-consistency **fit is validated on known geometry**
(uniform rim 0.8%, stenosis throat 5–8%), but it **does not yet transfer to real subglottis
video** — and it says so honestly. This is a *detection/consistency* gap, not a fit-math gap.
Next options (do NOT overclaim before one works): (a) a **learned/temporally-consistent** throat
segmenter that tracks one anatomical ring across frames; (b) restrict to clips with genuine
behind-a-constriction geometry; (c) accept that these particular low-texture, irregular
subglottis clips are data-limited (consistent with the whole project's finding). Not a redesign
of the fit.

## 8i. ORACLE-CONTOUR TEST — the real-video failure is GEOMETRY, not detection (decisive)
`annotate_oracle.py` + `oracle_fit.py`. Before building any learned segmenter, gave the validated
fit the best-possible contours: **human-annotated** the same narrowest-lumen ring in 10 frames per
region (2_V2 prox, 25_V1 prox/dist), verified every overlay (`runs/.../oracle/verify_*.png`), then
ran the fit with the real COLMAP poses.

| region | fit DCE | COLMAP slice | spread | reproj | vs phantom (0.04 / 3–5px) |
|---|---|---|---|---|---|
| 2_V2 prox  | 0.55 | 1.03 | **0.30** | 57 px | ✗ no convergence |
| 25_V1 prox | 0.93 | — | **0.25** | 114 px | ✗ no convergence |
| 25_V1 dist | 5.19 | 2.92 | **0.15** | 39 px | ✗ no convergence |

**Even perfect contours do NOT converge**, and the oracle 2_V2 spread (0.30) is **no better than
the auto-detector's (0.34)** → the bottleneck is *not* detection. The consistency curves are flat
with **no minimum** (phantom dipped sharply to 0.04); the fitted circles are wildly off the (good)
oracle contours (`runs/.../oracle/oracle_diagnostic.png`).

**Root cause = near-zero parallax.** Max optical-axis (viewing-cone) angle over the annotated
camera sets: **2_V2 prox 0.4°, 25_V1 prox 1.2°, 25_V1 dist 2.9°** — the optical axes are nearly
parallel. With ~0° parallax the back-projected contour rays are near-parallel, so there is no
well-conditioned axial plane where they cross at a common radius — regardless of contour quality.
This is information-theoretic (CLAUDE.md locked decision #6: no method overcomes zero parallax) and
matches the project's core finding that these dwelling subglottis clips are low-parallax.

**DECISION (per the oracle rule):** oracle/manual contours **FAIL** → **do NOT build a learned
segmenter.** The footage lacks a stable contour/geometry (near-zero parallax), not a detector.
The contour-consistency fit remains **validated on known geometry** (uniform rim 0.8%, stenosis
throat 5–8%) and is ready for any *adequately-parallaxed* clip; it correctly refuses these.

## 8j. GEOMETRY-GAP AUDIT — corrects §8i: it is NOT a blanket parallax wall (region-specific)
`geometry_gap_audit.py`. §8i concluded "near-zero parallax" from the optical-axis **cone** angle
(0.4–2.9°). That was the wrong metric: parallel-axis lateral translation gives a ~0° cone yet real
triangulation at a near target. Computing the TRUE target-aware triangulation angle (max pairwise
angle at each co-observed 3D point, restricted to the ring wall), track support, and dense-ring
quality per region:

| region | cone° | **tri median°** (ring) | tri p90° | n_pts | views | ring tight | cov | verdict |
|---|---|---|---|---|---|---|---|---|
| 2_V2 prox  | 0.71 | 2.48 | 4.6 | 1222 | 5 | 0.55 | 1.0 | marginal capture |
| 2_V2 **dist** | 1.52 | **3.71** | 8.6 | 1363 | 4 | 0.26 | 1.0 | **measurement gap** |
| 25_V1 prox | 1.28 | 2.80 | 4.8 | 789 | 6 | 0.39 | 1.0 | marginal capture |
| 25_V1 dist | 2.95 | **3.86** | 8.1 | 1071 | 6 | 0.36 | 1.0 | recipe gap |
| 32_V2 prox | 0.74 | 1.49 | 2.7 | 2961 | 4 | 0.50 | 1.0 | capture gap |
| 32_V2 **dist** | 0.44 | **4.27** | 7.6 | 6388 | 5 | 0.24 | 1.0 | **measurement gap** |

**Key correction.** The optical-axis cone badly *understates* parallax — 32_V2 dist has cone 0.44°
but true ring triangulation **median 4.27°, p90 7.6°**. Track support is **good everywhere**
(789–6388 co-observed pts, 4–6 views/pt, median track len ≥3), and dense rings have **100% angular
coverage**. So the earlier flat "footage lacks geometry / impossible" was **too strong**.

**Region-specific truth:**
- **Proximal subglottis** (2_V2, 25_V1, 32_V2 prox): triangulation **1.5–2.8°** — genuinely
  capture/parallax-**marginal** (below the pipeline's `init_min_tri_angle=4`). "Marginal," not
  "impossible."
- **Distal subglottis** (2_V2, 25_V1, 32_V2 dist): triangulation **3.7–4.3°** (at/above 4°) with a
  reconstructed **100%-coverage** dense wall ring (tightness 0.24–0.36) → **NOT parallax-limited**;
  the gap is **measurement/recipe** (extract a clean CSA from the existing ring).

**Why the contour-fit still failed here** (reconciles §8h/§8i): it needs a *fixed occluding
contour* (a throat aperture). An open subglottic lumen doesn't present one — the 2D lumen edge is a
view-dependent silhouette of the receding wall — so its rays don't converge *regardless of
parallax*. That is a **method-assumption mismatch**, not proof the footage is unusable. The dense
**wall ring** is reconstructed (100% cov); the COLMAP dense-**slice** baseline is the right tool for
it and already produced scene-unit % on 2_V2 (and is the path for the distal rings).

Caveats (do not over-read): the ring slice uses the camera-trajectory PCA as the axis, a weak proxy
for the true airway centerline when the scope dwells (small camera cluster → noisy PCA); the
per-point local baseline is what the tri angle captures, so the traj-level b/d ratio is only an
upper bound. Verdict thresholds: tri usable ≥3°, poor <1.5°; tracks ≥150 pts & ≥3 views; ring tight
≤0.35 & cov ≥0.6. Figure `runs/barbour30/airwayfit/geomgap/geomgap_summary.png`.

**Net decision.** Not a pure capture wall. Two actionable tracks: (a) the **distal** rings have
usable geometry + a reconstructed surface → invest in **measurement** (robust airway-centerline
slicing + CSA agreement) on the dense cloud, not a segmenter and not the aperture contour-fit;
(b) the **proximal** subglottis is parallax-marginal → only a capture-protocol change (advancing,
wider-baseline pass) reliably improves it. "Impossible" is retracted; "marginal proximally,
measurement-limited distally" is the supported statement.

## 8k. CENTERLINE-SLICE CSA on the distal rings — confirms §8j (2_V2 dist is measurable)
`centerline_csa.py`. Acting on §8j (distal subglottis has usable geometry + a reconstructed ring):
measured CSA on the DISTAL dense rings using the airway **medial-axis centerline from the point
cloud** (slab-centroid path + smoothing spline, NOT camera-PCA), slices **perpendicular to the
local centerline tangent**, CSA three ways (polar-median polygon, ellipse fit, convex-hull
free-contour). Accept iff coverage ≥0.75 AND estimators agree ≤1.3× AND r_std/r_med ≤0.35.

| region | centerline | cov | r_std/r_med | CSA | DCE | est. spread | stability | verdict |
|---|---|---|---|---|---|---|---|---|
| **2_V2 dist** | cloud medial-axis | **1.00** | **0.096** | 1.84 | **1.53** | **1.09** | ±2.5% | **ACCEPT** |
| 32_V2 dist | cloud medial-axis | 0.56* | 0.219* | (4.0*) | — | 1.17* | — | reject: coverage-limited |
| 25_V1 dist | cloud medial-axis | 0.61* | 0.34* | (3.9*) | — | 1.08* | — | reject: coverage-limited |
\* = best-attempt slice (not accepted). Scene units only (no mm).

**Findings**
- **2_V2 dist ACCEPTS cleanly**: a full ring (cov 100%, r_std/r_med 0.096), three estimators within
  **1.09×**, DCE **1.53** (scene units) stable to **±2.5%** across nearby slices; CSA tapers smoothly
  and monotonically along the centerline. This **confirms §8j** — the distal geometry is real and,
  with a proper cloud medial-axis slice, a clean CSA comes out. The measurement gap is *closable*
  where the reconstruction is clean.
- **32_V2 / 25_V1 dist fail on COVERAGE, not noise**: at their best slice the three estimators
  actually **agree** (spread 1.17 / 1.08) but only **56% / 61%** of the ring circumference is
  reconstructed → a **partial wall = reconstruction-completeness / recipe gap**, not estimator chaos
  and not absent geometry. A fuller reconstruction (more/better frames, Barbour's recipe) could
  complete the ring. (32_V2 also uses borrowed calibration — provisional.)

**Net.** The centerline-slice measurement *works* on the one clean distal reconstruction (2_V2
dist) and **correctly refuses** the partial ones rather than fabricating. Ordering of the gap by
region is now concrete: 2_V2 dist = measurable; 32_V2/25_V1 dist = complete-the-ring (recipe);
proximal subglottis = capture-marginal (§8j). Figure
`runs/barbour30/airwayfit/centerline_csa/centerline_csa.png`.

## 8l. 2_V2 DELIVERABLE — scale-free % obstruction from ONE global model (32% CSA, lower bound)
`obstruction_2v2.py`. One continuous cloud medial-axis centerline through the 2_V2 subglottis
(global dense model `runs/batch4/2-V2/dense0/fused.ply`, frames 875-1045); both the distal
reference ring and the proximal narrowest ring measured on this ONE centerline → one scene-unit
scale (no local-batch DCE, no mixed scales). CSA three ways with a uniform robust radial-outlier
trim; accept iff coverage ≥0.75 AND estimators agree ≤1.3× AND r_std/r_med ≤0.35.

| ring | u | coverage | r_std/r_med | CSA | DCE | est. spread (polar/ell/hull) | stability |
|---|---|---|---|---|---|---|---|
| distal reference | 5.2 | 100% | 0.12 | 1.87 | **1.54** | 1.14 (1.86/1.87/2.13) | ±1.3% |
| proximal narrowest | 2.25 | 100% | 0.10 | 1.27 | **1.27** | 1.29 (1.21/1.27/1.56) | ±6.2% |

- **CSA obstruction = 1 − 1.27/1.87 = 32%** (band 21.5–32.5%)
- **diameter narrowing = 1 − 1.27/1.54 = 17.5%**

Scene units only; scale-free, within-video; **NOT absolute mm; NOT a final Myer-Cotton grade.**

**This is a LOWER BOUND.** The accepted CSA profile is **monotonic** (no interior local minimum):
polar & ellipse (the two outlier-robust estimators) agree to ~5% everywhere and keep dropping below
the accepted band (to DCE ~1.13 at u≈0.2), but those proximal-most slices fail the strict 3-way
gate (the convex hull inflates on stray points) and the *true* subglottic throat is even more
proximal (frames ~875-945, dwelling cameras) and **not densely reconstructed** (centerline can't
form there). So the real obstruction is **≥32%**.

**Cross-validation.** 32% CSA (17.5% diameter) closely matches the independent `barbour30_band`
result (~33% area, band 24–38%) — two different methods on the same video converge. The convex
hull is a known outlier-sensitive upper bound; polar & ellipse are the load-bearing estimators
here (agree ~5%), with the hull kept only as the third gate. Figures
`runs/barbour30/airwayfit/obstruction_2v2/obstruction_{rings,profile}.png`.

**Status.** 2_V2 has a defensible, same-model, scale-free % obstruction: **~32% CSA / ~17.5%
diameter (lower bound)**. This is the first deliverable-grade number produced end-to-end by the
validated centerline-slice method.

## 8m. RING COMPLETENESS on 25_V1 / 32_V2 distal — coverage improved but strict gate not met
`ring_completeness.py`. MVS depth maps were freed (no cheap re-fusion; 25_V1 undistorted images
gone too), so **legitimate levers only** on the existing global cloud: (4) glare/dark + statistical
+ radial-MAD declutter, (5) a **refined** cloud medial-axis centerline (iterate: perp-slice →
recompute medial centroid → re-spline), (6) re-slice perpendicular, and a slab-thickness sweep
(real points from nearby axial offsets, guarded by the ratio + estimator gates). Polar / ellipse /
hull reported separately.

| video | coverage before → after | robust DCE (polar≈ell) | ratio | 3-way spread | median pe | accepted | reason |
|---|---|---|---|---|---|---|---|
| 25_V1 dist | 0.56 → **0.78** (thin, refined) / 1.0 (thick) | ~2.35 | 0.37 | 1.36 | 1.14 (agree) | **NO** | near-miss |
| 32_V2 dist | 0.56 → 1.0 (nominal) | unstable | — | >2 | 1.71 (disagree) | **NO** | genuine fail |

**Two different outcomes (figure `runs/.../ring_completeness/ring_completeness.png`):**
- **25_V1 dist = NEAR-MISS.** The refined centerline + declutter *alone* lifted thin-slab coverage
  0.56→0.78 (≥0.75), and thickening reaches 1.0. It's a **real ring** — polar≈ellipse agree
  typically (median pe 1.14, robust DCE ~2.35) — with **one residual arc gap** (lower-right). But
  no *thin clean* full ring exists, so full coverage needs a thick slab → ratio 0.37 (loose) and the
  outlier-sensitive convex hull inflates the 3-way spread to 1.36 (>1.3). Strict gate narrowly
  missed. A **denser reconstruction** could close the arc gap; slicing alone cannot certify it.
- **32_V2 dist = GENUINE FAILURE.** The distal cloud is a **radial-streak MVS scatter, not a
  coherent wall** (classic low-parallax/bad-depth artifact; borrowed calibration). Coverage is
  nominally reachable but polar vs ellipse *typically* disagree (median pe 1.71). Needs
  re-capture/re-reconstruction, not slicing.

**Honest verdict.** The coverage *number* is reachable (25_V1 genuinely, 32_V2 only nominally), but
neither meets the strict simultaneous gate (cov ≥0.75 AND 3-way spread ≤1.3 AND ratio ≤0.35 AND
stable). Per the rules I did **not** force a number, hallucinate fill, mix scales, or claim mm /
Myer-Cotton. 25_V1 is one denser-reconstruction away; 32_V2 is a real capture/recon-quality gap.
This is consistent with §8j (32_V2 dist was borrowed-calib/provisional; the audit's thick-slab
"tightness 0.24" was a union artifact of the streak scatter). Only **2_V2** (§8l) yields a
deliverable-grade ring today.

## 8n. 25_V1 DISTAL RE-RECONSTRUCTION — near-miss → ACCEPTED (arc gap closed)
`reconstruct_25v1_distal.py` + `measure_25v1_distal.py`. Targeted re-reconstruction of the 25_V1
distal window (frames on disk, no POS_FRAMES drift): 79 denser + sharpness-filtered frames
(390–470, dropped 2 blurry), CLAHE, **pinned** OPENCV intrinsics, **exhaustive** matching (1884
geometry-verified pairs), fresh dense MVS (geometric consistency), **both** geometric + photometric
fusion. Then refined cloud medial-axis centerline, perpendicular THIN-slab (0.30·r_med) slicing,
CSA 3 ways + a legitimate 2D **statistical-outlier declutter** (removes the scattered stray MVS
points that had inflated the convex hull — the ring itself is fully covered, so this is lever-4
decluttering, not hallucinated fill).

| fusion | thin-slab coverage before→after | CSA (pol/ell/hull) | DCE | 3-way spread | ratio | stability | accepted |
|---|---|---|---|---|---|---|---|
| **geometric** | ~56% (gap) → **100%** | 17.9 / 17.5 / 21.5 | **4.60** | **1.26** | 0.23 | ±9.1% | **YES** |
| photometric | → 100% | 14.0 / 14.2 / 16.5 | 4.23 | 1.21 | 0.21 | ±9.8% | YES |

**Result: 25_V1 distal converts from near-miss to ACCEPTED.** The denser exhaustive-matched
re-reconstruction **closed the residual arc gap** (thin-slab coverage 56%→100%, no thick-slab
faking), polar≈ellipse agree (pe 1.03), and after decluttering the stray points the 3-way spread
falls to 1.26 (≤1.3) with a tight ratio (0.23) and stable nearby slices (±9%). The two fusions
agree to ~8% (geometric is the cleaner, reported).

**Honest scope.** DCE 4.60 is in the **new distal reconstruction's own scene units** — scale-free
within this connected model only; **NOT** comparable to 2_V2 or the old 25_V1 global model, **NOT**
mm, **NOT** a Myer-Cotton grade. Stability ±9% is looser than 2_V2's ±1–2%. What made it work:
denser frames + exhaustive matching + fresh MVS (the depth maps the old cloud had lost), plus the
refined medial-axis and declutter. This confirms §8m's diagnosis — 25_V1 distal was one
denser-reconstruction away, and it got there.

## 8o. 25_V1 SAME-MODEL % OBSTRUCTION — connected model built, but not measurable (proximal marginal)
`reconstruct_25v1_subglottis.py` + `obstruction_25v1.py`. Built ONE connected 25_V1 subglottic model
spanning proximal narrowest + bridge + distal reference (frames 340–470, 128 sharpness-filtered +
CLAHE frames, pinned intrinsics, standard SIFT, plain exhaustive, fresh dense MVS + dual fusion).

**Requirement 1 met:** the model IS connected — **128/128 frames registered, span 340–470, 48
proximal (≤388) + 65 distal (≥405) in ONE model**, 3539 sparse / 427k dense points.

**But the rings in it are NOT cleanly measurable → NO % obstruction** (per the fallback rule):
- The airway is *curved* over this span, so the single-axis projection centerline **tangles**
  (its u-range 2.5–14 didn't even match the cameras −4.5…7.4). A camera-trajectory-backbone
  medial-axis centerline handles the curve, but along it the CSA profile **bounces** (DCE 4→31,
  ellipse estimator spikes to 2·10⁴) and **0 slices** pass accept+stability.
- Best ring attempt: cov 100% but **ratio 0.36, spread 1.75** (hull-inflated even after declutter),
  a noisy scatter — far from the distal-only re-recon's clean 1.26/0.23.

**Root cause:** including the **capture-marginal proximal frames** (25_V1 prox triangulation
2.35–2.8°, §8j) in the bundle adjustment **degrades the whole connected reconstruction** — the
distal region that was a clean accepted ring on its own (§8n, DCE 4.60, cov 100%, spread 1.26)
becomes noisy once the marginal proximal frames are added. There is no way around this with the
current data: a same-model % *requires* both rings in one model, but the proximal simply can't be
cleanly reconstructed, and its inclusion poisons the model.

**Verdict:** report proximal ring quality only (best attempt cov 100%, ratio 0.36, spread 1.75, DCE
4.76 scene units — REJECTED), **% obstruction NOT computed**. Honoring the rules: no
cross-reconstruction comparison, no Sim3, no mixing the clean distal-only DCE with a proximal DCE.
This is consistent with the whole project: **only 2_V2 yields a defensible same-model % obstruction**
(§8l, ~32% CSA lower bound); 25_V1's proximal subglottis is capture-marginal, so its within-video %
is not defensible — the distal *ring* is measurable (§8n) but the *narrowing* is not.
Figure `runs/.../recon_25v1_subglottis/obstruction_25v1.png`.

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
