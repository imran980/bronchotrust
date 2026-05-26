# bronchotrust

**An uncertainty-aware monocular bronchoscopy reconstruction pipeline using
MASt3R-based two-view geometric priors with learned + anatomy-anchored scale
recovery, evaluated quantitatively on synthetic and phantom data and
demonstrated as a clinical feasibility study for pediatric subglottic-stenosis
sizing on four paired CT–bronchoscopy cases.**

## Status

Phase 1 baseline comparison closed (see verdict tables below). Three
reconstruction backbones — MASt3R-SLAM, DUSt3R, COLMAP — installed,
wired through a uniform [reconstruct.py](reconstruct.py) driver, and
benchmarked on 3 real bronchoscopy videos + 2 synthetic clips. Headline:
**MASt3R-SLAM fails everywhere** (retrieval-database collapse on
endoluminal frames), **DUSt3R produces dense output but a non-coherent
trajectory** (no SE(3)-smoothness in its global aligner), **COLMAP +
CLAHE + bezel-mask** works on textured synthetic but starves on real
mucosa. The calibration module, uncertainty module, and new dashboard
are not built yet — see roadmap.

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
| Reconstruction driver (3 backends wired: mast3r-slam, dust3r, colmap) | [reconstruct.py](reconstruct.py); `python reconstruct.py --check-installed` |
| DUSt3R subprocess runner (per-pair + global aligner) | [dust3r_runner.py](dust3r_runner.py) |
| COLMAP subprocess runner (CLAHE + bezel-mask + SfM) | [colmap_runner.py](colmap_runner.py) |

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

## References

- ATM22: https://atm22.grand-challenge.org/
- MASt3R / MASt3R-SLAM: https://github.com/naver/mast3r
- DUSt3R: https://github.com/naver/dust3r
- AnyCalib (learned single-image calibration)
- Split conformal: Lei et al., JASA 2018; Vovk et al., 2005
- Barbour pediatric subglottic-stenosis cohort (paired CT + bronchoscopy)
