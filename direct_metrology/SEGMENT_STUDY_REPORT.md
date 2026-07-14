# Short-Axial-Segment Direct Metrology — Gate Test

**Verdict: FAIL. Abandon the direct-metrology direction.**

Photometric term **dropped** (the prior study showed it adds nothing). This tests the strongest
surviving idea: instead of a single ring, fit a short **generalized-cylinder segment** jointly —
centerline (axis + bend), a **non-circular** cross-section (ellipse + lobe), a local **flare/curvature**
profile `r0(s)`, and the CSA/DCE profile directly — from the multi-frame occluding contours, using an
**exact tangency forward model** (a viewing ray is on the silhouette iff it is tangent to the near wall
up to the throat). Gate: the segment must **beat the single-circle fit AND keep CSA error < 10 %** on a
smooth elliptical/lobed stenosis phantom, robustly over inits. It does not.

Reproduce: `OMP_NUM_THREADS=1 PYTHONPATH=. python -m direct_metrology.segment_study`
(`outputs_segment/results_segment.json`, `fig_segment_vs_circle.png`). Metrics via the frozen
`evaluation/` framework. Data-driven init + 4 random inits per case; median is the robust score.
Nothing tuned to pass.

---

## Results

| case | single-circle CSA err | segment best | segment **median** | frac inits < 10 % | GT-param fwd resid | beats & <10 %? |
|---|---|---|---|---|---|---|
| circular | +3.16 % | +3.24 % | **+3.24 %** | 1.00 | 0.02 mm | **FAIL** (no gain) |
| ellipse 0.2 | +32.5 % | +2.25 % | **+2.25 %** | 1.00 | 0.84 mm | PASS |
| trilobe 0.15 | +5.01 % | +4.52 % | **+4.52 %** | 1.00 | 0.45 mm | PASS |
| ellipse+lobe (clean) | +50.8 % | +2.25 % | **+8.24 %** | 0.60 | 0.70 mm | pass-but-fragile |
| **realistic** (ellipse+lobe+albedo) | +55.7 % | +5.98 % | **+23.2 %** | **0.20** | 0.70 mm | **FAIL** |

## What is genuinely better

The segment / non-circular model **fixes the catastrophic cross-section-shape error** that killed the
single-circle fit: ellipse 32 %→2 %, ellipse+lobe 51 %→2 % (best), realistic 56 %→6 % (best). The
model recovers ellipticity and lobe amplitude correctly for **isolated** shape perturbations, and it
reproduces the circular silhouette to 0.02 mm. This is a real, large improvement and confirms the
diagnosis that cross-section shape (not photometry) was the dominant failure.

## Why it still FAILS the gate

1. **No gain on the circular throat.** Segment +3.24 % vs circle +3.16 % — a tie. The ~3 % is the
   smooth-throat **silhouette-bias floor** driven by an `r_t`–`κ` (throat-radius vs flare) degeneracy:
   at this acquisition the multi-view contours do not separately pin the throat radius and the axial
   flare, so a slightly-too-large throat with different flare fits the silhouettes equally well. The
   segment cannot reduce this below the single-circle value — where the throat is already circular the
   extra machinery buys nothing.

2. **The realistic combined throat fails and is unreliable.** Median **+23 %** CSA, and only **20 %**
   of inits land < 10 %. Even the clean ellipse+lobe case converges < 10 % in only **60 %** of inits.
   The cause is not detection (nonuniform albedo shifts the contour by only ~1.3 px) and not the init
   (a good data-driven init still diverges): the forward model has a **~0.7 mm mismatch at the true
   parameters** for combined non-circular shapes, and the wrong-shape minimum has *lower* residual
   (~0.06 mm) than the truth (~0.70 mm). With ellipse **and** lobe **and** flare free, the optimizer
   exploits that mismatch — it drives `κ` to ~11 and collapses the ellipticity (e → −0.06 instead of
   +0.15) to reach a spurious lower-residual minimum. More model freedom made it *worse*, not better.

The failure is therefore **fundamental identifiability**, not a tuning knob: the combined non-circular
cross-section + axial flare is only weakly constrained by occluding contours at a realistic
bronchoscopic viewpoint, so a small model/observation mismatch is amplified into a wrong-shape CSA.

## Decision

> Gate: beat the single-circle fit AND keep CSA error < 10 % on the elliptical/lobed phantom. If it
> fails, abandon this direct-metrology direction.

It fails: no gain on the circular throat, and the realistic combined throat lands at a wrong-shape
minimum in 80 % of inits (median +23 %). **Abandon the direct-metrology (occluding-contour tube-fit)
direction.** The evidence across the whole arc is consistent:

- photometric term — adds nothing (prior study);
- single circle — catastrophic on non-circular throats (+45–58 %);
- short non-circular segment — fixes isolated shapes but is not identifiable on the realistic combined
  throat and cannot beat the circle where the throat is already circular.

Occluding contours from a low-diversity monocular bronchoscopic viewpoint do not contain enough
information to pin a combined non-circular cross-section with axial flare to < 10 % CSA. A method that
would clear this bar needs genuinely more information (e.g. dense multi-view depth / true parallax, or
a learned shape prior), not a better fit of the same contours.

**No real bronchoscopy was used. No novelty or clinical claim. The gate failed; per instruction the
result is reported as the exact failure and the direction is abandoned rather than tuned until it
passes.**
