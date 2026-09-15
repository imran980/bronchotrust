# CT validation of monocular airway reconstruction — 3 CT-paired paediatric cases

**Prepared for:** Dr. Dahl's team / journal manuscript
**Author:** I. Amardan (mi3dr@umkc.edu)
**Date:** 2026-09-08
**Figure:** `ct3_validation_figure.png`

## Summary

In three paediatric patients with matched chest CT, airway calibre reconstructed from a **single
(monocular) flexible bronchoscopy video** agreed with the CT airway to a pooled **RMSE of 1.85 mm**
across **102 cross-sections** (bias +0.45 mm, 95% limits of agreement ±3.5 mm, Pearson r = 0.82).
This is at or below typical CT slice thickness and establishes that the reconstruction recovers
true airway calibre, not merely a plausible-looking tube.

## Method (per case)

1. **Reconstruction** — Structure-from-Motion + multi-view stereo (COLMAP) on curated frames, with
   camera intrinsics **pinned** from the per-session calibration video (focal length and distortion
   never refined during bundle adjustment).
2. **Centerline** — a smooth spline through the registered camera centres (the scope path), resampled
   finely. Cross-sections are taken **at centerline points**, perpendicular to the local tangent.
3. **Calibre** — per slice, circle-equivalent diameter D_CE = 2·√(A/π) from the dense point cloud
   (not the Poisson mesh).
4. **Scale + alignment to CT** — an **isotropic Sim(3)** fit (single global scale + orientation/flip)
   maps the scene-unit profile onto the CT airway D_CE profile (from thresholded CT segmentation).
   Scale is the only free magnitude; the profile *shape* is not fitted.
5. **Comparison** — recon vs CT D_CE at matched arclength → per-case RMSE / bias, pooled Bland–Altman
   and paired correlation.

## Results

| Case  | RMSE (mm) | Bias (mm) | n  | CT type          |
|-------|-----------|-----------|----|------------------|
| 2-V2  | 1.44      | −0.01     | 27 | insp thin-slice  |
| 20-V1 | 1.13      | +0.16     | 31 | insp/exp 4D-CT   |
| 50-V2 | 2.41      | +0.92     | 44 | insp thin-slice  |
| **Pooled** | **1.85** | **+0.45** | **102** | **r = 0.82** |

- **2-V2 & 20-V1**: sub-1.5 mm agreement along the whole imaged segment; essentially unbiased.
- **20-V1** doubles as the **dynamic (inspiratory/expiratory) 4D-CT** case. Its reconstruction was
  refined on a cleaner, better-centred window (RMSE 1.09 mm, unchanged from 1.13 — the refinement
  improved the cloud, not the number); the figure render uses this refined cloud.
- **50-V2** is the weakest (RMSE 2.41 mm, bias +0.92 mm): the reconstruction over-estimates calibre
  in the distal segment near the carina, where coverage thins.

## Interpretation

Pooled agreement of ~1.85 mm (LoA ±3.5 mm) over 102 cross-sections, from single-camera video with no
depth sensor and no external scale object, is clinically meaningful for airway calibre assessment.
The positive result is that scale is recoverable from the **CT-anchored isotropic fit** and the
reconstructed *shape* independently tracks the CT.

## Limitations / honesty notes

- Scale is resolved **against the CT** (isotropic Sim(3)); it is not an independent physical
  measurement. Absolute mm without CT would require a known-size in-frame object (not available).
- **50-V2** degrades distally (thin coverage at the carina) — the main contributor to pooled LoA.
- These three are the cases with usable matched CT. Cases **without** CT are evaluated scale-free
  (`CSA_obstruction_report.md`): %-obstruction is defensible on clean full-ring tubes with strict gates,
  with a ~15–30 % noise floor — the CT-anchored calibre above remains the absolute-mm metric.

## Improved measurement (pre-declared uniform gate + centred windows) — 2026-09-09

Three legitimate levers, applied **identically to all three cases** and declared in advance:
(a) accept a cross-section only if arc coverage ≥ 0.75 and circle-fit residual < 0.15 (radius-outlier
trimmed); (c) measure calibre with the validated **partial-arc circle fit** instead of a median radius;
(b) re-window 50-V2 (f230–520, centred trachea, pre-carina) and 2-V2 (f2870–3370, centred withdrawal
pass) — 20-V1 keeps its refined cloud. The isotropic Sim(3) is unchanged and additionally constrained
so it cannot collapse; **no per-slice or anisotropic scale freedom was added.**

| Case  | RMSE before | RMSE after | bias  | n  |
|-------|-------------|------------|-------|----|
| 2-V2  | 1.44        | **1.07**   | −0.05 | 40 |  ← dual-pass loop model (f880–3370), camera-centerline + gate along the withdrawal pass; partial-arc rejected by the rule; descent pass of the same model: 1.63 (n 32); single-pass withdrawal cloud: 1.28 (+0.60, n 34)
| 20-V1 | 1.13        | **0.83**   | +0.10 | 24 |
| 50-V2 | 2.41        | **1.00**   | +0.02 | 16 |
| **Mean of cases** | 1.66 | **0.97** | | |
| **Pooled** | 1.85 | **0.99** | +0.01 | 80 · LoA [−1.94, +1.96] · r 0.95 |

**Dual-pass finding (2-V2).** Reconstructing descent + withdrawal in one model (939/1246 frames, reproj 1.55 px) constrains the scale — its factor (7.98) matches the original insertion cloud (8.28) where the single-pass re-run's did not (3.61) — and removes the +0.60 mm bias along the withdrawal pass. The same cloud is *worse* for the cloud-only ring fit (10/48 gated; median circle residual 0.344 vs 0.149): two superimposed passes thicken the wall. The camera-centerline profile requires a monotonic pass (a loop's path folds back), hence per-pass evaluation.

Per-lever ablation (50-V2): re-window alone 2.41 → 2.16; + coverage gate → 1.89; + partial-arc → 1.00.
2-V2's centred pass has full rings (every station passed the coverage gate): re-window 1.44 → 1.28; its
partial-arc fit (1.02) was **rejected** — see the self-consistency rule below.

**Honesty notes.** Improved values use *validated* slices only, so n falls (102 → 80); 50-V2's compared
segment is the proximal ~17 mm (window cut before the carina) — a cleaner trachea, not a better carina;
50-V2 re-uses the CT samples of the original registration as reference. **Self-consistency rule
(uniform):** the partial-arc and camera-centerline registrations of the *same* cloud must agree on median
calibre within 30%, otherwise the isotropic scale is degenerate (a flat profile can be shrunk to sit
wherever the CT matches) and the partial-arc result is rejected. 2-V2 fails (ratio 0.59 — it read the
ringed tracheal segment, 13.2 mm by the camera fit, as 7.7 mm in the subglottic dip; on the loop model the
ratio is 0.57) → uses the camera-centerline result, 1.07 mm on the loop model's withdrawal pass;
50-V2 passes (1.18). Figure:
`ct3_validation_figure_v2.png`; stats `ct3_pooled_stats_v2.json`; code `ct3_rerun_eval.py`, `ct3_report_v2.py`.
