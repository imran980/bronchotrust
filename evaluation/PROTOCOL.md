# Ground-truth evaluation protocol (immutable)

This is exactly how **every** future reconstruction method is evaluated against the same-day 4-D CT.
Once ground truth is built it is **frozen and content-hashed**; nothing downstream may edit it.
There is **no manual tuning** after this framework is written — every method runs the identical path.

## 0. Contract
Every method exposes the **same API** (`evaluation/interfaces/method_interface.py`):
```
run(CalibratedVideo) -> ReconstructionResult(centerline, arclength, CSA(s), DCE(s), covariance, confidence, units)
```
`units` is `"mm"` (metric) or `"scene"` (scale-free). A method never sees the CT. The evaluator never
runs reconstruction — it only consumes a `ReconstructionResult` (a method or a read-only adapter
produces it).

## 1. Ground truth (built once, frozen)
- **Full-profile CT** → `ct_processing.process_ct_mask(mask, spacing)` → CT **centerline**, **CSA(s)**,
  **DCE(s)**, **stenosis** landmark. Immutable `GroundTruth` (mode=`profile`).
- **Dynamic 4-D CT** → `ct_processing.ground_truth_from_bands(ct_reference.json)` → per-landmark **DCE/CSA
  bands** across ventilation phases (mode=`landmark_bands`). The airway *moves*, so truth is a **range**.
- `GroundTruth.assert_immutable()` must pass before any evaluation.

## 2. Registration (CT ↔ bronchoscopy)
- `registration.register(method_centerline, ct_centerline)` → **Sim(3)** (scale s, R, t) by
  arclength-init + ICP-with-scale, both endpoint orientations tried, lower residual kept. Named
  anatomical landmarks anchor it when available (≥3).
- A **scene-units** method's absolute scale is **recovered here** as `s` (the CT is the only metric
  anchor). A **mm** method uses `s=1`.
- **Slice correspondence:** each method slice → nearest CT centerline arclength.
- **Uncertainty propagation:** CSA→mm as `s²·CSA`, `Var = s⁴·Var(CSA) + (2 s CSA)²·Var(s)` (delta
  method; `Var(s)` from a correspondence bootstrap). DCE variance via `Var(CSA)/(π·CSA)`.

## 3. Metrics reported for EVERY method (identical, algorithm-agnostic)
| metric | definition |
|---|---|
| **CSA error** | per-slice `CSA_method(mm²) − CSA_CT`, aggregated (RMSE / MAE / bias / p50 / p90), abs & % |
| **DCE error** | per-slice `DCE_method(mm) − DCE_CT`, same aggregation, abs & % |
| **centerline error** | symmetric point distance (mean, p90, Hausdorff) between registered method & CT centerlines (mm) |
| **stenosis-location error** | arclength distance between method's narrowest slice and CT's narrowest slice (mm) |
| **% obstruction error** | \|`1−CSA_min/CSA_ref` (method) − same (CT)\|, scale-free (units cancel) |
| **uncertainty calibration** | `z = error/σ`; empirical 1σ/2σ coverage vs 68/95, `std(z)` (≈1 ideal; >1 overconfident), ECE |
| **runtime** | wall-clock seconds the method reported |
| *(band mode)* **within-band fraction** | fraction of landmarks whose method DCE falls inside the CT dynamic band; edge distance otherwise |

## 4. Acceptance / interpretation
- A metric is only reported where it is defined (e.g. absolute CSA/DCE needs a resolved scale;
  scale-free % obstruction is always reportable).
- **Calibration** is judged on `std(z)`: 0.5–2.0 with ECE<0.25 ⇒ *calibrated*; `std(z)>1` ⇒
  overconfident (errors exceed claimed σ), `<1` ⇒ underconfident.
- Nothing in the report depends on which algorithm produced the numbers.

## 5. Reproducibility
- `python -m evaluation.run_selftest` builds a synthetic CT with known answers and asserts the whole
  chain (a perfect method scores ~0; the recovered scale matches; calibration flags over/under-confidence).
- The report (`report.py`) writes `report.{json,csv,png}`; the JSON records the **ground-truth
  checksum** so results are traceable to a specific frozen GT.
