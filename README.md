# bronchotrust

**An uncertainty-aware monocular bronchoscopy reconstruction pipeline using
MASt3R-based two-view geometric priors with learned + anatomy-anchored scale
recovery, evaluated quantitatively on synthetic and phantom data and
demonstrated as a clinical feasibility study for pediatric subglottic-stenosis
sizing on four paired CT–bronchoscopy cases.**

## Status

"Ours v1" reconstruction working end-to-end on real video. Pipeline:
**dust3r-foe** (DUSt3R per-pair priors + RAFT focus-of-expansion
sign-fix + SE(3) smoothness) → **SSM-fit** (PCA shape model from
ATM22 surface SSM as prior, alternating-ICP with α + similarity, 37
modes) → **per-vertex posterior sampling** (proper Bayesian
N(α_MAP, σ²(AᵀA+λI)⁻¹), 100 samples). Tested on 3 real bronchoscopy
videos with consistent metrics (median \|α\|≈1σ, σ̂²≈7-11 mm² in SSM
frame, anatomically-valid tube fits with low-residual trunk +
high-residual lobar branches).

Phase 1 closed earlier (3 baselines compared: MASt3R-SLAM fails
universally on bronchoscopy, DUSt3R baseline has 16-26× trajectory
zigzag, COLMAP starves on textureless mucosa).

Calibration (Phase 3), absolute-coverage calibration of the
posterior STD via [conformal.py](conformal.py) on the Barbour
paired-CT cases (Phase 7), and the new dashboard are not built yet.

## Pipeline (target architecture)

```
video.mp4 (uncalibrated)
   │
   ├─ ingest        ffmpeg → frames
   │
   ├─ calibrate     learned (AnyCalib-FT on textured synthetic)
   │                + anatomical anchor (tracheal-diameter prior)
   │                → intrinsics + metric scale
   │
   ├─ reconstruct   MASt3R two-view geometric priors
   │                → dense surface mesh
   │
   ├─ quantify      deep ensemble (5×) or MC-dropout
   │                → per-vertex uncertainty + extrapolated-region flags
   │
   ├─ prior-match   ATM22 PCA shape model (similarity fit, surface-to-surface)
   │                → patient-frame airway prior
   │
   └─ output        uncertainty-coloured reconstruction + SSM overlay
                    (no camera trail, no depth panel)
```

## Roadmap

| # | Phase | Gate | Status |
|---|---|---|---|
| 1 | Baseline recon on 3 real videos (MASt3R-SLAM, DUSt3R, COLMAP) | ≥1 method coherent on 2/3 videos | **closed — gate fails 4/4** (see Phase 1 verdict below); DUSt3R per-pair priors are the salvageable signal |
| 2 | Synthetic validation set, 20 textured sequences | mean surface error ≤ 1.0 mm vs GT mesh | renderer textured (synth_v1: +40% gradient mag, lifts COLMAP 3.5× + DUSt3R 5× points); 20-seq generator + GT-error eval not run |
| 3 | Calibration module (AnyCalib FT + anatomical anchor) | predicted focal within 3%, distortion within 5%; recon ≤ 1.2 mm | not started |
| 4 | Uncertainty (ensemble or MC-dropout) | ECE ≤ 0.1; Spearman ρ(σ, \|err\|) ≥ 0.6 on synthetic | [conformal.py](conformal.py) ready, ensembles not built |
| 5 | Real-video eval on 10+ videos | 9/10 end-to-end + qualitative coherence | not started |
| 6 | Integration with SSM (surface-to-surface registration) | unified pipeline runs both outputs without regression | SSM build done, matching not wired |
| 7 | Clinical feasibility on 4 paired Barbour CT–bronchoscopy cases | mean surface ≤ 1.0 mm; stenosis diameter error ≤ 5% | not started |
| 8 | Ablation (calibration on/off, uncertainty on/off, MASt3R vs DUSt3R vs COLMAP) + write-up | submission-ready draft | not started |

## What runs today

| Capability | Entry point |
|---|---|
| ATM22 → SSM corpus (centerline + surface PCA) | [preprocess_atm22.py](preprocess_atm22.py) → [build_atm22_corpus.py](build_atm22_corpus.py) → [label_bifurcations.py](label_bifurcations.py) → [build_correspondence.py](build_correspondence.py) → [register_surface.py](register_surface.py) → [fit_ssm.py](fit_ssm.py) |
| SSM corpus reader | [atm22_corpus.py](atm22_corpus.py) |
| Textured synthetic renderer (color + GT depth + GT pose; **depth scale 2.55**) | [render_synthetic_bronchoscopy.py](render_synthetic_bronchoscopy.py) |
| Procedural bronchial-tree atlas (fallback when no patient CT) | [bronchus_atlas.py](bronchus_atlas.py) |
| Video → frames + manifest + intrinsics yaml | [process_video_input.py](process_video_input.py) |
| Split-conformal calibration math (tested) | [conformal.py](conformal.py), [tests/test_conformal.py](tests/test_conformal.py) |
| Reconstruction driver (5 backends wired: mast3r-slam, dust3r, dust3r-smooth, dust3r-foe, colmap) | [reconstruct.py](reconstruct.py); `python reconstruct.py --check-installed` |
| DUSt3R subprocess runner (per-pair + global aligner) | [dust3r_runner.py](dust3r_runner.py) |
| DUSt3R + SE(3) smoothness runner (ablation row) | [dust3r_smooth_runner.py](dust3r_smooth_runner.py) |
| DUSt3R + RAFT-FoE sign-fix + smoothness ("ours" v1) | [dust3r_foe_runner.py](dust3r_foe_runner.py), [flow_foe.py](flow_foe.py) |
| COLMAP subprocess runner (CLAHE + bezel-mask + SfM) | [colmap_runner.py](colmap_runner.py) |
| SSM-fit: PCA shape model to dust3r-foe observations + posterior sampling | [ssm_fit.py](ssm_fit.py) |
| Visual diagnostics (trajectory overlay, cloud-vs-GT, dead-frame audit, posterior std) | [diag_synth.py](diag_synth.py), [diag_visualize.py](diag_visualize.py), [diag_ssm.py](diag_ssm.py), [diag_ssm_posterior.py](diag_ssm_posterior.py) |

5-class anatomical landmark scheme: `vocal_cord`, `trachea`, `main_carina`,
`rmb`, `lmb`. Used by [label_bifurcations.py](label_bifurcations.py) and the
synthetic renderer.

## Data assets

- **ATM22** — population CT airway segmentations; basis of the SSM prior.
- **Textured synthetic renderer** — flies a virtual camera through the SSM
  mesh; emits GT depth (scale 2.55), pose, and landmark visibility.
- **Phantom** — TBD for Phase 7 quantitative eval.
- **Barbour cohort** — 4 paired pediatric bronchoscopy + CT cases for the
  clinical feasibility demonstration.

Not in scope: C3VD (colonoscopy), UAAL (intubation), BM-BronchoLC
(landmark-detector training set — detector path retired).

## Environment

Single conda env (the Endo-2DTAM env split is gone):

```bash
conda create -n bronchotrust python=3.11 -y
conda activate bronchotrust
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-atm22.txt
```

## Hard rules

Lifted from [CLAUDE.md](CLAUDE.md):

- Endo-2DTAM is shelved permanently.
- No colonoscopy pivot.
- No uninvited preprocessing — match training-time normalization exactly.
- Don't rebuild the SSM correspondence pipeline (known TPS issue, accept as-is).
- Document depth scale at every interface (past trap: C3VD raw=655.35 vs preprocessed=2.55).
- Phase gates are written. Don't skip them. One change at a time.

## Phase 1 verdict (3 methods × 3 real videos + 2 synthetic)

Subsampled to 50 frames for DUSt3R (global aligner is O(N²) in memory).
Span = first-last translation, pathlen = sum of consecutive step norms,
zigzag = pathlen / span (1.0 = perfectly direct; >5 = wandering).

| Source | Method | Registered | Span | Pathlen | Zigzag | Points |
|---|---|---:|---:|---:|---:|---:|
| 5-V1 (352) | MASt3R-SLAM | 3 / 352 | 0.31 | 0.31 | 1.0 | 106 k |
| 5-V1 | DUSt3R | 50 / 50 | 0.43 | 9.44 | **22×** | 1.8 M |
| 5-V1 | COLMAP+CLAHE | 10 / 352 | 10.62 | 11.19 | 1.1 | 150 |
| 13-V2 (501) | MASt3R-SLAM | 2 / 501 | 0.25 | 0.25 | 1.0 | 140 k |
| 13-V2 | DUSt3R | 50 / 50 | 0.42 | 6.92 | **16×** | 2.0 M |
| 13-V2 | COLMAP+CLAHE | 10 / 501 | 10.02 | 28.49 | 2.8 | 150 |
| 16-V1 (572) | MASt3R-SLAM | 36 / 572 | 1.12 | 1.90 | 1.7 | 2.2 M |
| 16-V1 | DUSt3R | 50 / 50 | 0.35 | 9.28 | **27×** | 1.9 M |
| 16-V1 | COLMAP+CLAHE | 68 / 572 | 8.23 | 73.11 | 8.9 | 510 |
| synth_v0 (untextured) | MASt3R-SLAM | 1 / 200 | – | – | – | – |
| synth_v0 | DUSt3R | 50 / 50 | 0.70 | 17.85 | 25.4 | 179 k |
| synth_v0 | COLMAP+CLAHE | 24 / 200 | 13.65 | 14.59 | 1.1 | 714 |
| **synth_v1 (textured)** | MASt3R-SLAM | 1 / 200 | – | – | – | – |
| **synth_v1** | DUSt3R | 50 / 50 | 0.89 | 13.35 | **15.0** | **871 k** |
| **synth_v1** | COLMAP+CLAHE | **84 / 200** | 13.76 | 16.64 | 1.2 | **3 860** |

Three structural findings:

1. **MASt3R-SLAM's retrieval database collapses on endoluminal frames.**
   The "Failed to relocalize" loop fires on every video, including
   textured synthetic. Texture quality is not the bottleneck — its
   per-frame retrieval module is.
2. **DUSt3R's per-pair priors work; its global aligner doesn't.**
   Consistent ~535-555 px focal estimate across videos, 1.8-2 M dense
   points, but zigzag 16-27× means the trajectory wanders. The aligner
   has no SE(3) temporal-continuity prior. This is the actionable
   engineering lever for the next phase.
3. **The textured synthetic must out-feature real video for Phase 2.**
   synth_v0 (Lambertian salmon) was *worse* than real for all methods.
   Adding 3D value-noise + vessel streaks + mild specular (synth_v1) put
   COLMAP at 42% registration (vs 12% on real best) and dropped DUSt3R
   zigzag by 40%.

## dust3r-foe ("ours" v1) result

| Video | Method | Span | Zigzag | Reversals | Mean cos |
|---|---|---:|---:|---:|---:|
| 5-V1 | dust3r baseline | 0.43 | 22.0 | 19/48 | −0.23 |
| 5-V1 | **dust3r-foe** | **1.39** | **8.8** | **0/48** | **+0.66** |
| 13-V2 | dust3r baseline | 0.42 | 16.4 | 21/48 | −0.13 |
| 13-V2 | **dust3r-foe** | **0.80** | **7.1** | **0/48** | **+0.77** |
| 16-V1 | dust3r baseline | 0.35 | 26.5 | 16/48 | −0.08 |
| 16-V1 | **dust3r-foe** | **1.07** | **13.2** | **3/48** | **+0.58** |

Direction reversals (consecutive step-direction cosine < −0.5) went
from 16-21 per video to 0-3. Mean step-to-step cosine flipped from
anti-correlated to strongly positive (forward-flowing trajectory).
Dense point count and per-pair geometry unchanged from baseline —
only the trajectory got fixed.

How it works:

1. **Optical flow.** Torchvision RAFT-Large between every pair in
   DUSt3R's swin-3 scene graph. Survives on textureless mucosa
   because it uses correlation volumes on local intensity gradients,
   not feature descriptors.
2. **Focus-of-expansion classifier.** LSQ-fit the FoE from flow lines,
   then weighted-vote sign(flow · (pixel − FoE)) → {+1 forward,
   −1 backward, 0 uncertain} with confidence.
3. **Per-pair sign hinge** added to DUSt3R's optimizer:
   `max(0, −s_ij · z_ij)` on the z-component of each pair's relative
   translation in camera-i's frame. Zero penalty when the sign agrees,
   linear penalty when it disagrees.
4. SE(3) second-difference (acceleration) smoothness on top, cleans up
   residual high-freq wiggle. Scale-invariant via mean-step-length
   normalisation; alone it was insufficient (Phase 1 ablation: zigzag
   stayed at 15-19), but with FoE supplying the missing direction
   information it does its intended job.

The diagnostic from Phase 1 was that DUSt3R's per-pair priors are
sign-ambiguous on bronchoscopy because the textureless mucosa
underconstrains the matching. Optical flow fills exactly that gap.

## SSM-fit + posterior result (3 real videos)

After dust3r-foe gives coherent poses + per-frame point patches, we fit
the ATM22 PCA surface SSM (37 modes, 147 k vertices) treating the
patches as noisy observations of a tube whose shape lives in the PCA
span. Alternating ICP between (α PCA coefficients) and (similarity
transform R, t, s) with λ=10000 prior weight. Then sample N=100 α from
the Gaussian posterior Σ_α = σ̂²·(AᵀA + λI)⁻¹ and propagate per-vertex
world-space std.

| Video | ‖α‖ | median \|α\| (σ) | σ̂² (mm²) | per-vert STD med | per-vert STD p95 | per-vert dist med | dist p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-V1 | 14.89 | 1.07 | 8.35 | 0.00033 | 0.00091 | 0.167 | 0.640 |
| 13-V2 | 9.85 | 0.86 | 10.58 | 0.00023 | 0.00058 | 0.069 | 0.183 |
| 16-V1 | 10.82 | 1.08 | 7.65 | 0.00030 | 0.00060 | 0.104 | 0.376 |

- **‖α‖** small (≤15 in z-score units across 37 modes), median \|α\|≈1σ
  → fits stay inside the PCA training distribution.
- **σ̂² ≈ 8-11 mm²** (RMSE ≈ 2.9-3.3 mm in SSM frame) consistent across
  videos.
- **Per-vertex posterior STD spatial structure**: low at central
  trunk (well-observed), higher at lobar branches (extrapolated from
  prior alone). Absolute magnitudes overconfident (model misspec —
  residuals are systematic bias, not i.i.d. noise); will be calibrated
  via split-conformal on the Barbour paired-CT cohort.

The "uncertainty-aware" claim in the project one-liner is now
mathematically defensible: real Bayesian posterior, not heuristic
proxy. Absolute coverage will be a Phase 7 (Barbour) calibration step
using [conformal.py](conformal.py).

## References

- ATM22: https://atm22.grand-challenge.org/
- MASt3R / MASt3R-SLAM: https://github.com/naver/mast3r
- DUSt3R: https://github.com/naver/dust3r
- AnyCalib (learned single-image calibration)
- Split conformal: Lei et al., JASA 2018; Vovk et al., 2005
- Barbour pediatric subglottic-stenosis cohort (paired CT + bronchoscopy)
