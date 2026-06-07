# 15_v2 — From-Scratch Gated Plan + Agent-Verification Guide

**Goal:** a defensible **scale-free Myer–Cotton % obstruction** on 15_v2 (the one parallax-viable video). Absolute mm in diameter is deferred to a blade-gated stage at the end, because it needs an external fact (Barbour's blade spec) you don't have yet — the *grade* does not.

**How to use this:** run ONE stage, the agent STOPs, then **you run the VERIFY block before approving.** If a VERIFY check fails, do not proceed — paste the failure back and we fix it. The agent's summary is *not* evidence; the cross-checks are.

---

## Locked decisions (baked in — do NOT let the agent relitigate these)

These were each established with evidence. If the agent re-proposes any of them as a "new idea," that's a red flag that it has lost the context.

1. **Intrinsics are pinned**, never refined. OPENCV model; `ba_refine_focal_length=0`, `ba_refine_extra_params=0`. Fisheye was rejected (its high-order coefficients explode and undistort to a blank frame).
2. **Calibration video gives intrinsics, NOT transferable scale.** A board clip scales only its own reconstruction, not the bronchoscopy clip. Do not attempt checkerboard scale transfer.
3. **Measure on the dense MVS cloud, not the Poisson mesh** (Poisson extent was 24–69% non-reproducible).
4. **Slice at centerline points, NOT at camera positions** (the forward scope images the wall ahead of itself; the ring at point *p* comes from the camera that sat behind *p*).
5. **Scale-free % obstruction = area ratio**, which is scale-invariant. This is the deliverable. No scale anchor is required for the grade.
6. **No SfM method or fit function overcomes zero parallax.** Don't chase GLUEMAP / different matchers / cylinder-vs-circle to fix a low-parallax video — it's information-theoretic, not algorithmic.

## Known agent failure signatures (watch for these every stage)

- Uses `cap.set(CAP_PROP_POS_FRAMES)` for extraction → frame-index drift on H.264. **Demand sequential `cap.read()` + a hash check.**
- Calls a video "corrupted" when the ffmpeg errors are all *non-monotonic DTS muxer warnings* (cosmetic) → not corruption.
- Reports a frame "failed to register" when it actually registered in a *different sub-model* → connectivity confusion.
- Saves the wrong (smaller) sub-model as the result → check it kept the largest one covering the target region.
- Reports a **plausible-looking number with no validity check** (the 4 mm / 30-frame trap). Plausible ≠ correct.
- Reports a % obstruction from slices at a *bend* or with area estimators that disagree 10–100× → garbage.

---

## STAGE 0 — Integrity + sequential frame extraction
**Prompt to agent:**
```
15_v2 integrity + extraction. STOP at end. NO reconstruction.
1. ffmpeg full decode: `ffmpeg -v error -i 15_v2.mp4 -f null -`. Classify each error line:
   DTS/non-monotonic = cosmetic; "corrupt"/"concealing"/"error while decoding" = real.
2. Decode ALL frames SEQUENTIALLY with cv2.read() (NOT cap.set(POS_FRAMES)). Report decodable
   count vs container-claimed, # black frames, # frozen runs.
3. Extract all frames sequentially to disk; save a frame_stats.json (idx, L_mean, valid).
STOP. Report: decodable count, real-vs-cosmetic error split, black/frozen counts.
```
**VERIFY (catch the agent):**
- Real errors should be **0 or only a few tail frames** (15_v2 is intact per your integrity check). If it reports "corrupted," confirm the errors are DTS-only — if so, it's wrong.
- It must say it used **sequential read**, not POS_FRAMES. If it used POS_FRAMES, reject — indices will be wrong downstream.
- Frozen-run count should be ~0 in the working range. A long frozen run = a real defect; investigate.

## STAGE 1 — Full-video contact sheet → you label → curate frames
**Prompt to agent:**
```
15_v2 frame triage. STOP at end.
1. Contact sheet of the FULL video (every ~15th valid frame), each thumb labeled with its
   sequential frame index. No classifier.
STOP. I will mark ranges: glottis / subglottis / trachea / any instrument.
```
*(After you label, second prompt:)*
```
Select 35-40 curated frames spanning glottis -> subglottis -> upper trachea, with DENSE
sampling through the sub-cord descent (that's the clinical region). Save a labeled 5xN contact
sheet, each thumb = my segment + sequential index. Hash-verify each saved frame matches the
canonical sequential frame at that index (report matches/total).
STOP.
```
**VERIFY:**
- **Eyeball the contact sheet yourself.** Confirm the sub-cord descent is densely sampled, frames are in focus (not mucus/blur), and labels match what you see.
- Hash-verify result must be **N/N matches**. Anything less = index drift; reject.
- Confirm frames look like 15_v2's anatomy (sanity that the right file was used).

## STAGE 2 — Pinned intrinsics + distortion model
**Prompt to agent:**
```
15_v2 intrinsics. STOP at end.
Load 15_v2's per-session calibration video. Run cv2.calibrateCamera (OPENCV) AND
cv2.fisheye.calibrate. Report for each: RMS, fx, fy, cx, cy, distortion, implied H/V FOV.
Save an undistortion side-by-side (OPENCV | FISHEYE) on one board frame.
Recommend a model and write intrinsics_pinned.json.
STOP.
```
**VERIFY:**
- RMS **< 0.5 px** (you have a strict-pass threshold). If the only calibration video is marginal, flag it.
- **fx/fy ratio ≈ 1.000–1.002** (isotropic). A ratio like 1.2 means BA contamination — should not happen if pinned.
- **FOV ≈ 96–98°** for this scope. ~50° would be the old focal-length bug.
- **Expected: OPENCV wins on stability even if fisheye RMS is marginally lower.** Fisheye's k3/k4 should be large with alternating signs (that's *why* we reject it). If the agent recommends fisheye and its coefficients are small/stable, double-check — but the eject criterion is the undistortion blowing up.

## STAGE 3 — Parallax gate (the make-or-break pre-check, cheap)
**Prompt to agent:**
```
15_v2 parallax gate. Diagnostic only - SfM POSES, no MVS, no mesh. STOP at end.
Lightweight recon on the Stage-1 curated frames: SuperPoint+LightGlue + COLMAP mapper, pinned
intrinsics, init_min_tri_angle=4. Report: # registered; viewing-cone total angular extent
(max pairwise optical-axis angle) over the SUB-CORD cameras; camera position bbox; lateral/along
translation ratio.
STOP.
```
**VERIFY (this gate decides everything):**
- Sub-cord viewing-cone extent should be **~17°** (triage said 17.51°) and **lateral/along > 0.30** (triage: 7.15).
- **If the cone comes back < 5°, STOP the whole plan** — 15_v2 is NOT viable after all, the triage was on different/garbage frames, and reconstruction cannot work. (This is exactly the 2.1° failure that killed 2_v2.)
- Cross-check: this number should *roughly match the triage's 17.5°*. A big mismatch means the triage and this run used different frames — find out which is right before spending MVS.
- **Honest caveat:** passing this gate is necessary, not sufficient. Coverage at Stage 5 can still fail.

## STAGE 4 — Full reconstruction + connectivity + eyeball
**Prompt to agent:**
```
15_v2 reconstruction. STOP at end.
SuperPoint+LightGlue, exhaustive matcher, COLMAP mapper init_min_tri_angle=4, min_num_matches=8,
intrinsics PINNED from intrinsics_pinned.json (ba_refine_focal_length=0, ba_refine_extra_params=0).
Manual init pair from a sub-cord frame. MVS dense. (Poisson for VIEWING ONLY.)
Report: registered/N, sparse, dense, mean reproj. List ALL sub-models with their point counts
AND frame lists. State which one you kept and WHY (must be the largest covering the sub-cord frames).
Verify final camera params == pinned (machine epsilon).
Save a 3-view render of the DENSE MVS cloud + camera trajectory, colored along the trajectory.
STOP.
```
**VERIFY:**
- **Final intrinsics == pinned** (delta ~1e-7). If they moved, pinning failed; reject.
- **ONE connected model covering the sub-cord frames.** If multiple sub-models, confirm it kept the **largest one that contains your sub-cord frames** — not a bigger sub-model elsewhere (the selector bug). Make it list point counts + frames for ALL sub-models.
- **Eyeball the render:** does it look like a tube, with the trajectory running down the middle? A collapsed spike/ball = parallax problem (shouldn't happen if Stage 3 passed).
- mean reproj err should be **~1–3 px**. Much higher = bad reconstruction.

## STAGE 5 — Centerline cross-section profile (scene units) + coverage gate
**Prompt to agent:**
```
15_v2 cross-section profile on the MVS cloud (NOT Poisson). Scene units. STOP at end.
1. Centerline = smooth spline through registered camera centers, ordered by frame; resample finely.
2. Outlier-filter the cloud (drop points far from any camera).
3. At each centerline sample p (tangent t): slab of points around p projected perpendicular to t.
   SLICE AT CENTERLINE POINTS, NOT CAMERAS. Coverage over 36 angular bins; keep slices >=60%.
4. Per kept slice compute area THREE ways - alpha-shape, convex hull, polar-median-r polygon -
   and Deq=2*sqrt(A/pi). Report all three per slice.
5. Output Area- and Deq-vs-arclength profiles colored by coverage%, + a 3D render of kept rings.
STOP. Report: # measurable slices, profile, and per-slice the three area values.
```
**VERIFY (this is where garbage hides):**
- **The three area estimators must AGREE within ~2×.** If alpha vs hull vs polar differ by **10–100×** (like 2_v2's 0.27 vs 26 vs bouncing), these are NOT closed rings — reject the slice; the "profile" is noise.
- **Measurable slices must be in the sub-cord region**, not clustered at a single bend (2_v2's fake 68% came entirely from a bend).
- Each kept slice: the **centerline point should fall inside the ring**.
- Coverage on kept slices **≥60%**. If <60% everywhere → parallax/coverage failure (contradicts Stage 3 → investigate).

## STAGE 6 — Scale-free % obstruction + clinician landmarks
**Prompt to agent:**
```
15_v2 % obstruction. STOP at end.
From the Stage-5 VALID slices only (>=60% coverage AND area estimators agreeing):
- A_min = narrowest valid slice in the sub-cord region; A_ref = widest valid adjacent normal slice.
- %obstruction = (1 - A_min/A_ref)*100. Report A_min, A_ref (scene units), the % , and which
  arc-length positions / frames they correspond to.
STOP.
```
**VERIFY:**
- A_min and A_ref must both come from **validated** slices (coverage + estimator agreement). If either fails validity, the % is meaningless.
- A_min must be **in the sub-cord region** you care about, not a bend artifact.
- **Cross-check against ground truth:** the % should be consistent with the **surgeon's intraoperative grade** for 15_v2 (15_v2 is NOT CT-paired — 16_v1 was — so the op note / clinical grade is your ground truth). A big mismatch = the reconstruction isn't capturing the real stenosis.
- Have the clinician confirm the landmark frames map to the right anatomy.

## STAGE 7 — Absolute mm (GATED on Barbour's blade spec — do not start without it)
Only when Barbour provides: (a) which blade dimension he uses, (b) its mm value, (c) how he locates the two endpoints. Then derive scale from the in-model blade and convert the profile to mm.
**VERIFY:** sub-cord Deq must land in **pediatric range (single-digit mm)**. A 20 mm subglottis = wrong scale. If you ever also get a CT-paired video with a real *subglottic* (not intubated/tracheal) measurement, cross-check the blade-derived scale against it.

---

### The one rule that ties it together
For every stage, the agent must hand you a number **plus** the independent check that makes it trustworthy (a hash match, an estimator agreement, a cross-check against the triage or the surgeon's grade, an in-range sanity). A number with no check is not a result — it's a claim. Send the checks back here and I'll tell you if they actually clear.