# AirwayGate — universal validity + normality gate

**Status:** v0 skeleton (`airway_gate.py`). Cheap signals implemented; periodicity and
centerline-graph branching stubbed. Thresholds provisional, to be calibrated once from cohort data.

## Why this keeps the method universal (the publication argument)

The gate is **one automatic stage applied identically to every video**. It never uses per-video
tuning — the model decides the route, the operator does not pick settings per case. That is the
line between "one universal method" and "a bag of per-video tricks," and everything here stays on
the universal side:

- A normal, rigid, round airway passes straight through (the gate is *inert*).
- The same machinery describes abnormal-but-static airways (mass, non-round, second lumen) — a
  **generalization of the measurement**, not a special case bolted on.
- It is the same **"reject rather than fake"** philosophy the pipeline already ships (coverage
  ≥ 60%, 3-estimator agreement, one-connected-model), extended from *"is the reconstruction
  valid?"* to *"is this case in scope, and what shape is it?"*.

A validity-aware method that states exactly when it works and when it does not is **more**
defensible in a medical-imaging venue than one that claims to always work.

## Scope split (settles the "one paper or two?" question)

| condition | handled by | in this paper? |
|---|---|---|
| normal rigid tube | standard measurement | ✅ |
| abnormal **shape**, static (mass, non-round, stenosis, static 2nd lumen/branch) | general measurement (per-lumen / branched centerline) | ✅ |
| **quasi-static / cyclic** wall motion | phase-gated frame selection, then measure | ✅ |
| continuous **non-rigid** deformation | flagged OUT OF SCOPE | ❌ → separate dynamic-airway paper |

Static SfM's one hard assumption is a **rigid scene**. Abnormal *shape* does not violate it (SfM
reconstructs whatever geometry is there). A fast-**deforming wall** does violate it — that case is
a genuinely different algorithm (deformable SLAM / NR-SfM) and is deferred to its own paper. Forcing
the static pipeline onto it would be exactly the per-video trickery we refuse to publish.

## What the gate measures

**A. Rigidity (pre-MVS, from frames + sparse SfM)**
- *Epipolar inlier ratio* between consecutive frames — fraction of matches consistent with a single
  rigid two-view geometry. Rigid → high; independently-moving tissue leaves a persistent outlier
  population → low. (implemented)
- *Registration completeness + reprojection tail* — non-rigid content drops registration and
  inflates residuals despite good overlap. (implemented)
- *Motion periodicity* — is the independent-motion signal cyclic (pulsation/breathing)? A strong
  periodic peak ⇒ phase-gateable. (stub)
- → `RIGID` / `QUASI_STATIC_GATEABLE` / `NON_RIGID` / `INDETERMINATE`.

**B. Shape / topology (post-MVS, from dense cloud + centerline)**
- *Lumen component count* per slice (angular-gap clustering) — ≥ 2 = extra opening / branch / fistula.
- *Circularity* (`r_std/r_med`, already a gate) and *ellipse eccentricity* per slice — non-round.
- *CSA abruptness* — a sharp step in CSA-vs-arclength = mass / focal stenosis.
- *Centerline branching* — a medial-axis node of degree > 2. (stub; pairs with lumen-count ≥ 2)
- → `NORMAL_TUBE` / `ABNORMAL_STATIC` with flags `{noncircular, mass_stenosis, multilumen, branch}`.

## Calibration plan (do once, from data — not per video)

1. **Rigid baseline** from the clean human successes (e.g. 21-V1, 12-V1): establish the RIGID band
   for the epipolar inlier ratio and registration completeness.
2. **Non-rigid probe** from 20-V2 (the moving-wall/abnormal case): the short-window sweep tells us
   whether short spans are quasi-static (⇒ `QUASI_STATIC_GATEABLE`, set the periodicity path) or fail
   even short (⇒ `NON_RIGID`). This fixes the `NONRIGID_INLIER_MAX` threshold and the gateable/hopeless
   boundary.
3. **Shape thresholds** (`ECCENTRICITY_MAX`, `CSA_JUMP_MAX`, gap angle) from phantom + known-normal
   human tubes vs the abnormal case.

Thresholds live at the top of `airway_gate.py` and are frozen after this one calibration.

## Open items
- Implement `motion_periodicity` (independent-motion magnitude → autocorrelation → dominant period).
- Implement `centerline_branching` on the medial-axis graph.
- General/branched measurement path (per-lumen CSA) in the measurement layer.
- Phase-gated frame selection as an automatic, inert-when-rigid criterion in `recover_clip.py`.
