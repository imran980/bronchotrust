# Rescue experiments for the two Phase-0 failures + the novelty decision

Two targeted rescues were run to see whether the blocking failures (Test 2 albedo, Test 3 pose)
evaporate under a fairer formulation. Evidence in this folder (`rescue1_*`, `rescue2_*`).

## Rescue 2 — Test 2 with Huber loss + BIC-adaptive albedo DOF — **CLEARS**
`rescue2_huber_bic.{png,json}`. CSA bias per albedo field, four configs:

| albedo | fixed 9DOF + L2 | fixed 9DOF + Huber | **BIC-adaptive + L2** (DOF) | BIC-adaptive + Huber (DOF) |
|---|---|---|---|---|
| gradient | 0.0% | 0.0% | 0.0% (3) | 0.0% (3) |
| **stripes (vascular)** | **44.6%** | **52.2%** | **−0.3% (13)** | −0.6% (13) |
| low-freq | 0.1% | 0.1% | 0.1% (5) | 0.1% (5) |
| high-freq | 0.0% | 0.0% | 0.0% (3) | 0.0% (3) |

- **BIC-adaptive DOF clears it:** it selects **13 DOF (6 harmonics)** for the stripe — exactly the k=6
  coverage — dropping the bias from 44.6% to **−0.3%**. Worst BIC-adaptive bias across all fields:
  **0.3% < 10%.**
- **Huber alone does NOT help — it makes it worse (52.2%).** The unmodeled-albedo residual is a
  *systematic, distributed* error, not sparse outliers, so a robust loss is the wrong tool. The fix
  is the model-selection (BIC picks enough DOF), consistent with Evidence 2 (no degeneracy — full
  rank). **Test 2's failure was a modeling choice, not a science problem. ✔ cleared.**

## Rescue 1 — Test 3 on the CSA RATIO with relative pose noise — **DOES NOT CLEAR**
`rescue1_ratio_pose.{png,json}`. Two cross-sections in one model: reference r=5, narrow r=3, true
ratio 9/25 = 0.36. 100 MC, realistic pose noise (1% translation, 0.3°):

| quantity | bias |
|---|---|
| ABS CSA (reference r=5) | +15.3% |
| ABS CSA (narrow r=3) | **+41.6%** |
| **RATIO (narrow / ref)** | **+24.7% (±30%)** |
| RATIO under *common-mode* (global) pose error | +27.6% |

- **The hypothesis (systematic bias cancels in the ratio) is FALSE here.** The `1/d²` pose bias is
  **radius-dependent**: the narrower ring's wall is *closer*, so a translation error is a larger
  *fractional* depth error → the narrow ring is biased **~2.7× more** (41.6% vs 15.3%). The ratio is
  the ratio of exactly these two unequal biases, so it does **not** cancel — it sits at **+24.7%**,
  above the 5% target.
- **Even a common-mode (global) pose error does not cancel (+27.6%)** — two different radii respond
  differently to the same shift. Only at ~0% pose noise is the ratio bias <5% (sweep: 4.8% at ~0,
  rising to 24.8% at 1%, 206% at 3%).
- **Worse for the clinical case:** the *stenosis* (narrow ring) is the **most** pose-sensitive
  measurement, so a scale-free % obstruction is *not* protected by being a ratio. **✘ not cleared.**

## Decision on novelty
The user's condition was: *if (1) and (2) clear, the method works and novelty is only a
writing/positioning problem.* **(2) clears; (1) does not.** Therefore the pose failure does **not**
evaporate — it is a **science problem**, not a writing problem:
- At realistic COLMAP pose noise the scale-free obstruction ratio is biased **~25%** because the
  stenosis wall is the closest surface and thus the most `1/d²`-sensitive.
- The only ways to reduce it are to **refine poses inside the loop** (which is exactly NFL-BA — you
  give up the "fixed-pose" simplicity that distinguished the proposal) or to have **lower pose
  uncertainty** (i.e. more parallax — the same wall the geometric baseline hits on our data).

**Conclusion:** the novelty question is moot for now. With Rescue 2 the albedo objection is dead, but
Rescue 1 leaves a **blocking, quantified science failure** (radius-dependent pose bias → ~25% ratio
error). So this is **not yet a working method whose only weakness is being a combination** — it has a
real accuracy failure at realistic pose noise, concentrated exactly on the stenosis. Recommendation
unchanged: do not implement as a fixed-pose estimator; if pursued, it must jointly refine poses
(= NFL-BA) and is still capped by the parallax available on the footage.
