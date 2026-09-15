# Running the pipeline on a new video

Everything below uses the tools in `pipeline/`. Paths are yours; nothing is hard-coded. Set once:

```bash
export BRONCHO_COLMAP=/path/to/colmap        # COLMAP 3.13 built with CUDA
export BRONCHO_PY=$(which python)            # interpreter for post-steps (the env with requirements.txt)
# only if you will use CT ground truth:
export TOTALSEG_BIN=~/.venvs/totalseg/bin/TotalSegmentator
export TOTALSEG_HOME_DIR=~/.venvs/totalseg/home
```

## 1. Calibrate the session

Every examination needs the checkerboard clip recorded with the **same scope in the same session**.
The board has 14×13 inner corners.

```bash
python pipeline/calibrate.py --video "19-V1 Calibration Video.MP4" --out calib/19-V1_intrinsics.json
# -> board (14, 13) frames 297 RMS 0.158px fx 731.3 fy/fx 0.9995 HFOV 105.4 -> PASS
```

- **PASS** is RMS < 0.5 px; 0.5–0.7 px is marginal and flagged; the tool refuses to write anything
  with fy/fx off by more than 2 % or an implausible field of view (that is how a spurious sub-grid
  detection shows up).
- The cohort has three optics: HFOV ≈ 97–99° (fx ≈ 820–855), ≈ 105–107° (fx ≈ 717–734) and one 78°
  scope (fx ≈ 1193). If a session has no board video, do **not** guess: measure the endoscope image
  circle diameter in the video, match it to a calibrated session of the same scope (105° family ≈
  855–866 px, 98° family ≈ 1048–1069 px at 1080p) and record the borrow in the json
  (`source_calib_note`). A circle that matches nothing means no calibration, and no metric result.
- Intrinsics are **pinned**: `recover_clip.py` never refines focal length, principal point or distortion.

## 2. Choose the window from a contact sheet

```bash
python pipeline/contact_sheet.py --video 19-V1.MP4 --out sheets/19-V1.png
```

Open the sheet. Frame indices under each thumbnail are the true sequential indices (the decoder never
seeks). You want a **steady, centred, monotonic pass** through the segment of interest — the lumen dark
and roughly central, rings visible, no instrument, no dwelling, no reversal. Read off `--lo/--hi`.

Things that look fine but fail: pale washed-out wall-facing mucosa (no texture), footage inside an
endotracheal tube (printed markings reconstruct instead of the airway), stretches where the posterior
wall bulges or collapses (the scene is not rigid), instrument in the field. Blur alone is **not**
disqualifying. If a video has a descent and a withdrawal, try each pass separately; the withdrawal
is often steadier (it rescued two cases). A second video of the same patient may be the better one.

## 3. Reconstruct

```bash
python pipeline/recover_clip.py --video 19-V1.MP4 --calib calib/19-V1_intrinsics.json \
       --lo 1040 --hi 1320 --out runs/19-V1_w1 --gpus 0,1
```

What it does: sequential decode of the window, bezel mask, CLAHE (clip 3.0), dark-frame gate, SIFT +
exhaustive matching, mapper with pinned intrinsics, automatic bridging of fragments with
`model_merger`, undistortion, patch-match stereo on the listed GPUs, fusion. Outputs in `--out`:
`sparse/0` (poses), `dense0/fused.ply` (the cloud), a CSA profile PNG and a render, and `run.log` with
a line like `[model] 134 reg, span f1094-1227, reproj 1.52px`.

- 60 fps footage is normalised to 30 fps automatically (`--target_fps 30` → stride 2).
- If the log shows several models (`[SfM] 3 model(s): ...`) the pass broke; the largest is kept. A
  break that survives `model_merger` is usually a real event in the video (wall collapse, glare
  flash, reversal). Re-window around it rather than fighting it.
- `--clahe 4.5` is a legitimate lever for dim clips; record it if you use it.
- Windows of 200–600 frames are the sweet spot. MVS scratch is deleted after fusion; a finished
  workspace is 50–300 MB.

## 4. Is it measurable?

```bash
python pipeline/eval_recon.py runs/19-V1_w1
# -> registered 134 frames f1094-f1227 | reproj 1.52 px | clean pts 215,595 | gated 36/48 | median cov 0.89 -> measurable tube
```

48 interior stations along the lumen axis; a station is *gated* if arc coverage ≥ 0.75 and the circle
residual < 0.15. Tiers: **≥ 20 gated = measurable tube**, 10–19 partially measurable, < 10 not
measurable. Written to `eval.json` with a two-view render.

## 5. Scale-free calibre and %obstruction

```bash
python pipeline/measure_csa.py runs/19-V1_w1/dense0/fused.ply --out runs/19-V1_w1/csa.json --plot runs/19-V1_w1/csa.png
```

Adds the circle/ellipse agreement gate (< 30 %), excludes the end 10 % of stations, median-filters
single-station spikes, and reports `pct_obstruction = 1 − A_min/A_ref` (A_ref = 90th percentile of
accepted CSA) plus the profile CV. Interpretation rules from the cohort: values **≤ 30 % are within the
noise floor** (a clinically normal airway read 29 %), and **CV > 0.15 means unreliable geometry**, not a
stenosis. This is a ratio; no scale is needed and none is implied.

## 6. Absolute calibre against CT (optional)

You need a thin-slice (≤ 1.5 mm) axial series; 2.5 mm is too coarse for a paediatric trachea.

```bash
python pipeline/ct_ground_truth.py --case 19-V1 --zip 19V1.zip --out-dir ct_gt
# -> series SER00005 (319 slices, 0.6 mm) ... profile: arclength 47 mm, D_CE median 5.7 mm ... saved ct_gt/gt_19-V1.npz
python pipeline/ct_score.py 19-V1 runs/19-V1_w1 --gt ct_gt/gt_19-V1.npz
# -> M1 cam-centerline + cov>=0.75  RMSE 0.72  bias -0.14  n 16
#    !! M2 partial-arc gated: mapped length 10.0 mm = 22% of the 47 mm CT segment -> COLLAPSED FIT
#    REPORTED: M1 ... RMSE 0.72 mm, bias -0.14, n 16, spans 16/47 mm
```

`ct_ground_truth.py` picks the thinnest inspiratory axial series, runs the **full** TotalSegmentator
task (the ROI shortcut under-segments), walks the trachea mask from the subglottis to the carina
bridging closed slices (a severe stenosis closes the lumen; those are real zero-area stations), and
writes arclength / CSA / D_CE. Validate a new machine once with `--validate <known_gt.npz>`.

`ct_score.py` registers with a constrained isotropic Sim(3) and applies two pre-declared validity
rules. **Self-consistency:** the partial-arc (M2) and camera-centreline (M1) registrations of the same
cloud must agree on median calibre within 30 %, otherwise the partial-arc scale is degenerate and M1
is reported. **Span guard:** the mapped physical length must be ≥ 10 mm and ≥ 25 % of the CT segment,
otherwise the fit has collapsed onto a sliver and is not a measurement — this is what stops a
near-occlusive stenosis from producing a fake 0.2 mm result. If the reconstruction contains both a
descent and a withdrawal, score each pass with `--lo/--hi`; the camera path of a loop folds back.

## 7. Many windows unattended

```
# jobs.txt  —  TAG|VIDEO|CALIB_JSON|LO|HI|EXTRA_ARGS
19-V1_w1|/data/19-V1.MP4|calib/19-V1_intrinsics.json|1040|1320|
19-V1_w2|/data/19-V1.MP4|calib/19-V1_intrinsics.json|1700|2450|
46-V2_hi|/data/46-V2.MP4|calib/46-V2_intrinsics.json|540|780|--clahe 5.0
```
```bash
pipeline/run_queue.sh jobs.txt runs/queue A 0,1 &
pipeline/run_queue.sh jobs.txt runs/queue B 2,3 &
tail -f runs/queue/runner.log
```

Each worker claims a job atomically, runs `recover_clip.py`, then `eval_recon.py`, and logs one
`DONE`/`FAIL` line per window.

## What "rescued" looked like in practice

- **19-V1** (0 registrations in every earlier attempt): the board video calibrates at 0.158 px and
  reveals a 105° scope, not the assumed 98°; the contact sheet put the trachea at f1104–1840 while the
  earlier window sat inside a dark stretch. Result: measurable tube, 0.72 mm against CT.
- **26-V2** (0/48 on the whole traversal): a transient posterior-wall collapse at f1908–1921 breaks
  SfM into two models; the slow centred pullback before it gives 48/48.
- **17-V1** (9/48 on the whole traversal): the pullback alone gives 28/48; the full loop gives the
  carina and both bronchi in one model.
- **46-V2** (five windows, two with boosted contrast): never above 39 frames. Pale, wall-facing,
  texture-poor mucosa. Some videos cannot be reconstructed; say so.
