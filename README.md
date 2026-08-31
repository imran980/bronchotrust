# bronchotrust — Barbour-style COLMAP airway reconstruction (baseline)

Monocular pediatric-airway (bronchoscopy/laryngoscopy) 3D reconstruction and
**scale-free % obstruction** measurement from clinical video, using COLMAP
Structure-from-Motion + Multi-View Stereo with **smart local-batch frame selection**.

This tag (`barbour-style-colmap-baseline`) is the frozen, working COLMAP baseline. It is a
research feasibility pipeline, **not** a validated clinical tool — see
[Known limitations](#known-limitations).

> The next method (texture-free silhouette/shading lumen fitting, using COLMAP poses only as
> initialization) is scoped separately in [`NEW_METHOD_SCOPE.md`](NEW_METHOD_SCOPE.md).

---

## Pipeline overview

```
video ─► smart frame selection ─► COLMAP SfM (pinned intrinsics, exhaustive)
      ─► dense MVS ─► airway point cloud ─► medial-axis centerline
      ─► perpendicular cross-sections ─► CSA / DCE per ring
      ─► (one connected model) scale-free % obstruction
```

Two things learned the hard way, baked into the design:

- **Frame SELECTION is the decisive lever.** Naive contiguous 30-frame windows fail the
  low-texture subglottis; *smart* selection — CLAHE + sharpness/glare/contrast gating +
  viewpoint-coverage-diversity sampling + adaptive pool (±30 → ±20 → ±45) — closes clean
  subglottic ring **shapes** where naive windows cannot (8/12 landmarks accepted;
  proximal subglottis 3/3 reproducible across videos).
- **COLMAP nails poses; the dense wall is the weak link.** Feature-SfM/MVS needs texture, and
  the airway wall is textureless/specular. Rings are measured on the **dense MVS cloud**
  (not the Poisson mesh), sliced **at centerline points** (a forward scope images the wall
  *ahead* of itself), with three CSA estimators that must agree.

### Locked design decisions
Intrinsics are **pinned, never refined** (OPENCV model, `ba_refine_*=0`); calibration gives
intrinsics **not** transferable scale; measure on the **dense cloud**; slice at **centerline
points**; **% obstruction = area ratio** (scale-invariant — the primary deliverable); no SfM
matcher overcomes low parallax (GlueMap was A/B-tested and rejected). See `CLAUDE.md`.

---

## Repository structure

```
pipeline/          the validated recon + measurement recipe (one recipe for every video)
  recover_clip.py      windowed harness: select → SfM (pinned) → merge → MVS → CSA → render
  batch_dense.py       long-window GLOBAL reconstruction (one connected model = one scale)
  barbour_dense_recon.py  Barbour-style dense reconstruction (CLAHE + COLMAP + MVS + Poisson)
  calibrate_pig1.py    robust intrinsics calibration from a checkerboard clip (pinned output)
  csa_run.py           partial-arc CSA / DCE on a dense fused cloud
  centerline_csa.py    cloud medial-axis centerline + perpendicular cross-sections, 3 estimators
  render_airway.py     presentation render of a fused airway cloud

experiments/
  porcine/         porcine cohort: SfM screen, MVS, ring-vs-funnel & cone-vs-tube geometry,
                   video-quality comparison (recon_pig_sfm, mvs_and_view, screen_rings,
                   render_dense_geom, cone_vs_tube, quality_compare, scan_quality, …)
  clips/           per-clip reconstruction / measurement (recon_25v1, recon_31v1_bridge,
                   fuse_32v2, obstruction_2v2, obstruction_25v1, measure_25v1_distal, …)
  diagnostics/     targeted probes (flow_looming, csa_partialarc, ring_completeness,
                   stage2_calib_compare)

evaluation/        CT-ground-truth validation harness (unified ReconstructionResult API)
direct_metrology/  short-axial-segment direct-metrology prototype
redteam_phase0/    near-light photometric estimator red-team (NO-GO record)
photometric_pivot/ photometric depth-cue feasibility probes (NO-GO record)
docs/              lightweight example figures + reports (the only tracked outputs)
```

Older exploratory scripts are preserved in `_archive/` (git-ignored, on disk).

---

## Environment

Two conda envs (heavy, external to this repo):

- **`depth-eval`** — Python driver: `cv2`, `pycolmap` (4.0.4), `open3d`, `numpy` 2.x, `scipy`,
  `matplotlib`. All `barbour30_*.py` scripts run here.
- **`colmap-cuda`** — the `colmap` binary (CUDA MVS), invoked via subprocess.

```bash
pip install -r requirements.txt          # into the depth-eval env
# colmap binary: /home/<you>/.conda/envs/colmap-cuda/bin/colmap
```

Input videos and per-session calibration live **outside** the repo (git-ignored):
`dataset/…` (videos) and `runs/retriage_first15/_calib/<session>/intrinsics_pinned.json`.

---

## How to run

**One command per windowed clip** — the validated harness selects frames, runs pinned-intrinsics
SfM (exhaustive), auto-bridges fragments (`model_merger`), runs 4-GPU dense MVS, then measures
and renders. `--target_fps` normalizes frame rate (inert on 30 fps clips; stride 2 on 60 fps):

```bash
# human 30 fps clip (subglottis window)
python pipeline/recover_clip.py --video 2-V2.MP4 --session 2_v2 --lo 900 --hi 1100 \
       --out runs/own_data/recon_2v2

# 60 fps porcine clip (fps-normalized to 30)
python pipeline/recover_clip.py --video "Pig Trachea 1 Video.mp4" --session pig1 \
       --lo 6736 --hi 7141 --out runs/own_data/porcine/pig1 --target_fps 30
```

**Global one-model reconstruction** (both rings in one scale, for a scale-free % obstruction):
```bash
python pipeline/batch_dense.py --video 2-V2.MP4 --session 2_v2 --lo 870 --hi 1200 --out runs/batch4/2-V2
```

**Measure / render** a fused cloud directly:
```bash
python pipeline/csa_run.py       runs/.../fused.ply  2-V2  runs/.../csa_profile.png   # CSA / DCE
python pipeline/centerline_csa.py runs/.../fused.ply                                   # centerline + 3-estimator rings
python pipeline/render_airway.py  runs/.../fused.ply  2-V2  runs/.../airway.png         # presentation render
```

**Calibration** (per-scope intrinsics from a checkerboard clip → pinned `intrinsics_pinned.json`):
```bash
python pipeline/calibrate_pig1.py    # robust board detection + iterative outlier rejection
```

**Porcine geometry checks** (ring-vs-funnel, cone-vs-tube, video-quality comparison):
```bash
python experiments/porcine/screen_rings.py  s_p5_a:"Pig Trachea 5.mp4":5800:6200 ...   # chunk ring/funnel map
python experiments/porcine/cone_vs_tube.py   fused.ply sparse_dir TAG                    # radius profile (tube vs cone)
python experiments/porcine/quality_compare.py tag:video:lo:hi:stride ...                 # sharpness/brightness/texture
```

---

## Current results summary

**Reconstruction (smart pipeline):** 8/12 landmarks **accepted**; **proximal subglottis 3/3**
reproducible across `2-V2`, `25-V1`, `32-V2`. The pipeline **rejects** rather than fakes the
hard cases (fragmentation / open rings / estimator disagreement).

**Scale-free % obstruction** (area ratio, scene units, **no mm**, within-video only):

| video | result | quality |
|---|---|---|
| **2-V2**  | **~33% area (band 24–38%)**, ~18% diameter | clean **monotonic** narrowing — trustworthy |
| **25-V1** | ~24% area (band 21–28%) | **non-monotonic / noisy** — low confidence |
| **32-V2** | not obtainable | global subglottic rings all noisy (r_std/r_med 0.43–0.91) |

So a trustworthy scale-free % is reliable on **1 of 3 videos** today.

**Porcine cohort** (controlled study, 4D-CT pending): the *same* pipeline reconstructs pig airways
wherever the scope runs a steady, centered, advancing pass — pig1 (normal trachea) and pig5
(continuous clean tube, f5800–6900). The pig footage is ~2× blurrier and ~30% dimmer than the best
human clip (60 fps → half exposure), which roughens the *dense wall* but **not** the *poses*
(pig1 sparse recon = 203/203, reproj 1.59 ≈ 2-V2's 201/201, 1.52). Absolute caliber is gated on the
incoming 4D-CT.

**Phantom validation:** the pipeline recovers a known tube radius to **1.8%** — errors on real
video are **data-limited, not algorithmic**.

Example figures (in [`docs/figures/`](docs/figures)):
`reconstructions_all_videos.png`, `landmark_rings_by_video.png`, `obstruction_2-V2.png`,
`obstruction_band_2-V2.png`; example reports in [`docs/results/`](docs/results).

![all-video reconstructions](docs/figures/reconstructions_all_videos.png)

---

## Known limitations

- **No absolute mm scale.** Monocular SfM is gauge-free, and no static, non-specular,
  known-size object appears in a connected model with the airway (the Hopkins shaft is a
  *moving, specular* instrument; tube bores/rims are *specular/overexposed* and don't
  reconstruct; the laryngoscope slot is off-screen; the telescope can't self-image). All
  CSA/DCE are **scene units only**. Absolute mm remains **capture-protocol-gated**.
- **No CT-paired validation completed.** The one CT-paired video (`16_v1`) reaches the carina
  and mainstems *visually*, but its slow, dwelling (low-parallax) scope fragments/collapses in
  SfM — no connected model, no recoverable scale — so the CT DCE comparison could not be done.
- **Scale-free % is reliable on only 1/3 videos.** The subglottis (the clinical target) is
  data-limited: low texture, low parallax, and partial angular coverage of the narrowest ring;
  the reference is the *patient's own trachea*, not a normative airway — this is **not** a
  final Myer–Cotton grade.
- **COLMAP MVS is texture-hungry**; clean *pose* recovery does not imply a clean *wall*
  surface. This motivates the next method (`NEW_METHOD_SCOPE.md`).
- Some videos use **borrowed calibration** (e.g., `32-V2` borrows `32_v1`) — provisional.

---

## Required future data (to reach clinical validation)

1. **Same-day paired video + CT** of the same airway, where the bronchoscopy **reaches and
   reconstructs** the CT-measured region (distal trachea / mainstems), so reconstructed DCE can
   be checked against a CT ground truth.
2. **A valid in-frame scale anchor** — a **matte, non-specular, known-size** object (a marked
   probe/ruler, or the laryngoscope aperture deliberately kept in view) held **static** in the
   **same continuous pass** as the airway, so a metric scale can be recovered in one connected
   model.
3. **Per-video camera calibration** — a checkerboard clip from the *same scope and session*
   (several videos currently borrow calibration).
4. **A capture protocol built for photogrammetry** — a steady, **advancing (non-dwelling)**
   scope for parallax; minimize overexposure, specular glare, and mucus at the target region.

---

## Repository notes

Generated outputs (`runs/`), videos, COLMAP databases (`*.db`), MVS workspaces, raw point
clouds (`*.ply`), and model weights are **git-ignored** and regenerable. Only source scripts,
docs, and the lightweight example figures/reports under `docs/` are tracked. See `.gitignore`.
