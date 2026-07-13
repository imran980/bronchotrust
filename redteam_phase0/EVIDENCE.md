# Phase-0 evidence (not conclusions) — four requested artifacts

All from the simulated forward model + estimator in `sim.py` / `test3_pose.py`. Figures + JSON are
in this folder. Numbers, spectra, basis functions and equations — not prose verdicts.

## 1. CSA bias vs pose noise (sweep) — `evidence1_pose_sweep.{png,json}`
Monte-Carlo (50 runs/level), 2.5° parallax, `R_true=5`.

**Translation (0 → 3% of depth), rotation = 0:**

| transl % | 0 | 0.3 | 0.6 | **0.9** | 1.2 | 1.5 | 1.8 | 2.1 | 2.4 | 2.7 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSA bias % | 0.0 | 0.5 | 4.8 | **11.4** | 22.7 | 36.1 | 48.4 | 63.2 | 75.3 | 88.1 | 99.4 |

**Rotation (0 → 1°), translation = 0:** CSA bias stays −1 to −3% throughout (variance grows, bias does not).

- The translation curve is **super-linear (∝ noise²)** — the `1/d²` term rectifies symmetric pose
  noise into a positive area bias. It crosses **+10% at ~0.9% translation** and reaches **+99% at 3%**.
- The 2-D grid's 10% contour is a **horizontal line at ~0.8% translation**, nearly independent of
  rotation → **translation is the bias driver, rotation is not.**

## 2. The actual Jacobian & SVD — `evidence2_jacobian_svd.{png,json}`
Not "rank full" as an assertion — the spectra:

| model | #params | σ_max | σ_min | **σ_min/σ_max** | condition |
|---|---|---|---|---|---|
| ideal (r0, A) | 2 | 0.259 | 5.9e-3 | 0.0228 | 44 |
| geom4 + albedo 9 DOF | 18 | 0.209 | 4.1e-3 | 0.0196 | 51 |
| geom4 + albedo 13 DOF | 22 | 0.258 | 3.4e-3 | 0.0132 | 76 |

- Every spectrum **plateaus above the 1e-3 rank tolerance** — no singular value collapses to 0, so
  the fit is full-rank (condition 44–76), even at 22 parameters.
- The **smallest-σ direction** loads **0.99 on the geometry block and 0.15 on albedo** — i.e. the
  closest-to-null mode is a (weakly-constrained, high-frequency) *geometry* mode, **not** a
  geometry↔albedo mixing. The normalized `JᵀJ` shows moderate cross-correlation but a
  diagonal-dominant, non-degenerate structure.

## 3. Albedo basis and why 13 DOF fixes the bias — `evidence3_albedo_basis.{png,json}`
- The vascular stripe is `sign(cos 6θ)` — a **square wave whose Fourier energy sits at k = 6, 18, 30**
  (fundamental **k = 6**), shown in the spectrum (bars at 6 and 18).
- A **4-harmonic** albedo (9 DOF) spans only k ≤ 4 — its reconstruction of the stripe is essentially
  **flat** (misses k = 6). A **6-harmonic** albedo (13 DOF) spans k ≤ 6 and captures it.
- **CSA bias vs albedo DOF is a cliff, not a ramp:**

| albedo DOF | 3 | 5 | 7 | 9 | 11 | **13** | 15 | 17 | 19 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|
| CSA bias % | 77 | 44 | 44 | 45 | 44 | **−0.3** | −0.2 | −0.1 | 0 | 0 |

- **Mechanism (panel e):** with 4-harmonic albedo the fit **recruits the geometry params to absorb the
  unmodeled k = 6 residual**, so the fitted `r(θ)` develops a spurious wobble (radius swings 2 → 11)
  → the area integral `∮r²/2` picks up **+44.6%**. With 6-harmonic albedo the k = 6 term is fit *by
  albedo*, geometry stays flat at the true r = 5, bias → 0. The cliff is exactly at the **stripe
  fundamental** — it is basis *coverage of a discrete frequency*, not conditioning.

## 4. Objective vs NFL-BA, side by side — `evidence4_nflba_equations.png`
```
ours:     E_ours(r(·), ρ, I)      = Σ_i Σ_{p∈Ω_i} || B_i(p) − ρ(x)·max(0, n(x)·l_ip)/d_ip² · I ||²
                                     variables: r(·), ρ, I ;  FIXED: poses {T_i}, intrinsics K
NFL-BA:   E_NFLBA({T_i},{X_j},Θ,ρ) = Σ_i Σ_j w_ij || B_i(π(T_i,X_j)) − ρ_j·Φ(n_j·l_ij, d_ij; Θ) ||²
                                     variables: poses {T_i}, geometry {X_j}, light Θ, albedo ρ_j
```
Reduction:  **E_ours(r,ρ,I) = E_NFLBA({T_i},{X_j},Θ,ρ) |₍ Tᵢ fixed, Xⱼ∈S(r), Φ = I/d² ₎**
- (1) poses moved from VARIABLES → CONSTANTS; (2) geometry restricted to a surface of revolution
  `S(r(·))`; (3) light `Φ(n·l,d;Θ)` specialized to `(n·l)·I/d²`.
- Same data term, same unknown *types* (geometry/albedo/light). The only structural change is
  **dropping poses from the variable set** (+ geometry/light specialization) → ours is a
  **restriction** of the NFL-BA objective, not a distinct inverse problem. Test 1's translation bias
  is precisely the price of restriction (1): fixed noisy poses inject a bias a pose-refining BA would
  partly absorb. _(NFL-BA notation paraphrased; variable/objective structure faithful.)_
