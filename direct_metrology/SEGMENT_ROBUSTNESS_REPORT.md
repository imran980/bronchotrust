# Segment Method — Optimization Robustness (realistic phantom)

**Goal:** raise realistic-phantom convergence from 60% to ≥95% while keeping CSA error <10%, changing
**only** the optimization (initialization + solution selection). The model, loss, priors, and evaluator
are byte-for-byte unchanged (`residuals`, `_r0`, `_shape`, `_tangency_f` untouched; `evaluation/`
untouched).

**Result: the ≥95% convergence bar is MET via multistart + residual rejection — K≥6 → 96.2%, K=8 →
98.5% — with the selected solution at <10% CSA.** Per-init robustness itself was *not* improved; the gain
is entirely from restarts + residual-based selection.

Reproduce: `OMP_NUM_THREADS=1 PYTHONPATH=. python -m direct_metrology.segment_robust_study`
(`outputs_segment/results_robustness.json`, `fig_robustness.png`). 105 random initializations on the
realistic phantom; convergence-vs-K estimated by bootstrap over those fits. Deterministic (seeded).

---

## Exact optimization changes

| technique | outcome | used? |
|---|---|---|
| **Multistart initialization** — many data-driven (`init_from_contours`) + jittered inits, cycling **diverse κ seeds** {0.1, 0.2, 0.35} and full-range random φ. No GT. | the run pool from which the method selects | **YES** |
| **Reject local minima by final residual** — the method returns the **lowest-residual** solution among its K starts. | the decisive lever (low residual ⇒ low CSA, verified) | **YES** |
| **Trust-region `max_nfev` cap (250)** — bounds runaway fits. | modest speedup; no accuracy loss | YES |
| Explicit per-parameter `x_scale` | did not help | tried, **not used** |
| **Staged fitting circle→ellipse→lobe** | **HURT** — 0/8 reached the global; the circle stage lets the r_t–κ degeneracy set a bad basin (κ→10–14), which the later stages can't escape | tried, **not used** |

No priors were added, no bounds tightened, and nothing was tuned on the CSA answer. Selection uses only
the observed residual.

## Convergence

| case | per-init <10% | multistart K=3 / 5 / 8 | min K for ≥95% | lowest-residual CSA err |
|---|---|---|---|---|
| circular | 100% | 100 / 100 / 100 % | 1 | +3.24% |
| ellipse 0.2 | 91.7% | 100 / 100 / 100 % | 2 | +2.21% |
| trilobe 0.15 | 100% | 100 / 100 / 100 % | 1 | +4.75% |
| ellipse+lobe | 50.0% | 90.5 / 99.4 / 100 % | 4 | +2.72% |
| **realistic** | **51.4%** | **79.7 / 93.3 / 98.7 %** | **6** | **+7.79%** |

Realistic per-init is **51.4%** (lower than the earlier 60% because this init set is deliberately harder
— full-range random φ + diverse κ). Multistart lifts it to **96.2% at K=6** and **98.5% at K=8**.

## CSA / DCE / residual distributions (realistic, 105 inits)

| metric | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| CSA err % | 2.57 | 5.96 | 8.78 | 38.4 | 319 |
| DCE err % | 1.27 | 2.93 | 4.30 | 17.7 | 105 |
| residual mm | 0.0362 | 0.0368 | 0.0612 | 0.116 | 0.68 |

The distribution is **bimodal**: a low-CSA basin (~2.6–9%, low residual ~0.036–0.045) and wrong-shape
minima (15–45%+, higher residual). Because residual and CSA error are correlated at the bottom
(`fig_robustness.png`, right), selecting the lowest-residual solution lands in the low-CSA basin.

## Failure modes (the wrong minima that residual selection rejects)

| count | r_t | e | lobe | κ | CSA err | residual |
|---|---|---|---|---|---|---|
| 7 | 3.4 | −0.1 | +0.1 | ~11 | ~30% | ~0.061 |
| 5 | 3.2 | −0.1 | +0.1 | ~0 | ~15% | ~0.102 |
| 4 | 3.5 | −0.1 | −0.1 | ~0 | ~35% | ~0.088 |
| 4 | 3.3 | −0.1 | +0.1 | ~13 | ~23% | ~0.067 |
| 3 | 3.6 | 0.0 | +0.1 | ~11 | ~39% | ~0.082 |

All share the same signature: **ellipticity collapses** (e → −0.1 instead of +0.15) and/or **κ runs
away** (→11–13), inflating r_t. Every wrong cluster has **residual ≥ the global (~0.037)**, which is why
residual-based rejection works. The residual gap is small at the very bottom (best wrong minimum
~0.037 vs best global ~0.036), so convergence asymptotes at **~98.5%, not 100%** — a few K-subsets miss
the tight global cluster and a near-tied wrong minimum wins.

## Verdict

- **Convergence ≥95%: MET.** Multistart K≥6 → 96.2%, K=8 → 98.5% on the realistic phantom (recommend
  **K=8** for margin). All other cases reach ≥95% at K≤4.
- **CSA <10% maintained:** the selected (lowest-residual) solution is 7.79% CSA on realistic; the
  low-CSA basin is 2.6–9%. DCE of the basin is 1.3–4.3%.
- **Honest limits:** per-init robustness was *not* improved (staged/scaling hurt); the ≥95% comes purely
  from restarts + residual selection, and it caps near 98.5% because a small fraction of wrong minima
  have residuals essentially tied with the global. Everything here is **synthetic**.

Per the stated rule ("proceed to real video only if convergence ≥95%"), the robustness gate **clears**
with K≥6 multistart. This is a convergence result on the synthetic phantom only — not a clinical or
real-video validation, and no novelty claim.
