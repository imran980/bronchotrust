# Red-Team Audit — "true params 0.70 mm, wrong params 0.06 mm"

**Outcome: (A). The true parameters ARE the global minimum. The previous contradiction was an
implementation error — a −90° azimuth-convention mismatch between the generator and the optimizer's
cross-section basis, which I fed into the GT-verification harness (and into the fit's shape init).**

The forward models are mathematically identical (same shape family, same tangency/occlusion condition).
No redesign, no new parameters, no impossibility claim — just the located bug.

Reproduce: `direct_metrology` scratch scripts `audit.py` / `audit2.py` (numbers below).

---

## The bug (root cause of both FAIL items)

Generator azimuth (`phantom._first_hit_smooth`): `theta = arctan2(X[1], X[0])` — world azimuth about z,
measured from world +x. Shape applied as `1 + ecc·cos2(θ−th0) + lobe·cos·k(θ−th0)`.

Optimizer azimuth (`segment._tangency_f`): `th = arctan2(radv·e2, radv·e1)` with the basis from
`segment._basis([0,0,1])` → **`e1=[0,1,0]`, `e2=[−1,0,0]`** (world +y and −x). Shape applied as
`1 + e·cos2(th−phi) + lobe·cos·k(th−phi)`.

Numerically, `th = world_φ − 90.000°` (constant), verified:

| world φ | generator θ | optimizer th | offset |
|---|---|---|---|
| 0° | 0.00° | −90.00° | −90° |
| 30° | 30.00° | −60.00° | −90° |
| 90° | 90.00° | −0.00° | −90° |

So the optimizer's `phi` parameter is **not** the generator's `th0`; the true value is
`phi = th0 − 90°`. A *single* harmonic can absorb a −90° rotation through its own free phase (2·Δ or
k·Δ), which is exactly why ellipse-only and lobe-only "passed". The *combined* shape needs the one
correct phi — and my GT-verification set `phi = th0`, evaluating a shape rotated 90° from the truth.

## Resolving the contradiction (numerical evidence)

Ellipse+lobe phantom (ecc 0.15, lobe 0.1, th0 20°), residual = RMS of the tangency residual (mm):

| parameter set | residual |
|---|---|
| "GT" at `phi = th0` (what the previous audit used) | **0.7024 mm** ← the false "0.70" |
| GT at `phi = th0 − 90°` (basis-corrected) | **0.0207 mm** |
| phi-scan global min over [0,2π) | **0.0207 mm at phi = 290° = th0 − 90°** |
| a wrong-shape set (e=−.068, lobe=−.065, r_t=3.36, κ=11.6) | 0.41 mm |
| the fit's converged local min (from the study) | ~0.057 mm |

Corrected-GT (0.0207 mm) is **below every wrong set** → the true parameters are the global minimum.
Starting the optimizer *at* the correctly-labelled GT, it **stays**: r_t 3.041, e 0.148 (true 0.15),
lobe 0.099 (true 0.1), phi −69.9° (= th0−90°), κ 0.187 (true 0.1875), residual **0.0081 mm**, CSA
+2.7 %. GT is a stable minimum; it was previously unreachable only because the data-driven shape init
seeded phi in the wrong −90°-offset basin, so the fit fell into spurious local minima (the +23 %
"failures").

## Per-item audit (PASS / FAIL + evidence)

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | generator == optimizer forward model | **PASS** | circular GT residual 0.0225 mm; corrected ellipse+lobe GT 0.008–0.021 mm. Axial Gaussian (gen) vs parabola+quartic (opt) differ by only ~0.02 mm — negligible. |
| 2 | coordinate frames identical | **FAIL** | optimizer cross-section basis e1=[0,1,0], e2=[−1,0,0] is rotated −90° about the axis vs world (x,y). `opt_th = world_φ − 90.00°` exactly. **This is the bug.** |
| 3 | ray parameterization identical | **PASS** | generator marches fractional t → first crossing; optimizer parametrizes by axial ζ. Same rays; if this differed materially the circular case would be off, but it is 0.02 mm. |
| 4 | tangent/silhouette condition identical | **PASS** | corrected-GT residual 0.008–0.021 mm across circular / ellipse / lobe / combined. |
| 5 | visibility / max operator identical | **PASS** | first-crossing (gen) vs `max_{ζ≤s_t}(hyp−r_model)` (opt) agree at the boundary; same 0.02 mm floor. |
| 6 | ellipse/lobe parameterization identical | **FAIL** | same functional form (cos2, cos·k) but evaluated at azimuth offset −90°, so phi_opt = th0 − 90° ≠ th0. Manifestation of #2. phi-scan min at 290° = th0−90°. |
| 7 | normals identical | **PASS** | generator normals affect only brightness (shading); optimizer residual is purely geometric; detection is a brightness edge. Combined effect = the 0.02 mm floor (circular 0.0225 mm). |
| 8 | optimizer minimizes the same residual that generated the data | **PASS (within 0.02 mm)** | data = brightness render + threshold/half-max detection; optimizer = geometric tangency. Discrepancy = 0.02 mm floor, not the 0.70 mm. |
| 9 | interpolation/discretization/sampling bias | **PASS (small)** | the 0.02 mm floor is discretization/detection; it produces the minor ~3 % circular CSA bias (r_t 3.041 vs 3.000 via the r_t–κ near-degeneracy), but is NOT the contradiction. |
| 10 | any numerical approximation making inverse ≠ forward | **PASS** | the only non-identity is the exact −90° convention (item 2/6), a discrete labeling bug — not a numerical approximation. With it corrected, GT is the global minimum. |

## Conclusion

**FAIL at items 2 and 6 — one root cause: a −90° cross-section-basis convention mismatch.** It is a
parameter-frame *labeling* error, not a forward-model non-identity. My GT-verification harness set the
optimizer's `phi` equal to the generator's world-frame `th0`, so it scored a shape rotated 90° from the
truth (0.70 mm) — the source of the impossible "true > wrong" residual. With the correct label
(`phi = th0 − 90°`) the true parameters are the verified global minimum (0.008–0.021 mm, below all wrong
sets), and the optimizer started there recovers the true shape.

The one-line correction (recorded, not applied here to avoid redesign): align `segment._basis` so the
in-plane axes match world (x,y) — e.g. build `e1` from the world-x projection — or equivalently
interpret/compare `phi` as `th0 − 90°`. The earlier "abandon" conclusion rested on this bug in the
verification; that conclusion is therefore not supported by this experiment and should be revisited only
after the convention is fixed.
