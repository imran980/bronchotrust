# `evaluation/` — airway reconstruction evaluation framework

Build-once, reuse-forever framework that scores **any** reconstruction method against the same-day
4-D CT with **identical, algorithm-agnostic metrics**. It **never runs or modifies reconstruction**
(the locked COLMAP pipeline is untouched) — it only consumes a method's standardized output.

## Folders
```
evaluation/
├── interfaces/       method API contract (method_interface.py) + read-only adapters (adapters.py)
├── ct_processing/    CT → centerline / CSA / DCE / stenosis landmarks  (ct_pipeline.py) + phantom.py
├── ground_truth/     immutable, content-hashed GroundTruth (profile OR dynamic landmark-bands)
├── registration/     CT↔bronchoscopy Sim(3) align, slice correspondence, uncertainty propagation
├── metrics/          all error metrics + uncertainty calibration + runtime
├── evaluator.py      the harness: ReconstructionResult + GroundTruth → EvaluationResult
├── report.py         one comparison report (CT vs A/B/C), identical metrics
├── PROTOCOL.md       the exact, frozen evaluation protocol every method follows
└── tests/            end-to-end self-test on a synthetic CT with known answers
```

## The one interface every method implements
```
Input :  CalibratedVideo         (calibrated bronchoscopy video)
Output:  ReconstructionResult(centerline, arclength, CSA(s), DCE(s), covariance, confidence, units)
```
`units` = `"mm"` or `"scene"` (scale-free — the CT resolves the scale during registration). Barbour,
COLMAP, a future optimizer, LightNeuS, ... all plug into the *same* evaluator via this contract or a
thin read-only adapter over their saved output.

## Metrics (reported for every method — see PROTOCOL.md)
CSA error · DCE error · centerline error · stenosis-location error · % obstruction error ·
uncertainty calibration · runtime · (dynamic-CT) within-band fraction.

## Run the self-test (proves the framework is internally consistent)
```
python -m evaluation.run_selftest          # synthetic CT: perfect method ≈0, scale recovered, calibration flagged
pytest evaluation/tests/test_framework.py
```

## Evaluate real methods (when a method's output is ready)
```python
from evaluation import evaluate, make_report
from evaluation.interfaces import adapters
from evaluation.ct_processing import ct_pipeline

gt   = ct_pipeline.ground_truth_from_bands("ct16v1", "runs/ct_validation_16v1/ct_reference.json")   # dynamic 4-D CT
res  = adapters.from_profile_arrays("COLMAP", centerline_xyz, csa_profile, units="scene")           # read-only wrap
make_report(gt, [evaluate(res, gt)], "evaluation/tests/_report_out/")
```
Nothing above depends on which algorithm produced `centerline_xyz`/`csa_profile`.
