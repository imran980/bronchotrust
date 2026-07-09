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
