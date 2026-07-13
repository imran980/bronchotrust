"""pytest wrapper for the framework self-test. Runs the full synthetic-CT chain and unit checks.
    pytest evaluation/tests/test_framework.py     (or)     python -m evaluation.run_selftest
"""
import numpy as np
from evaluation.ct_processing import phantom, ct_pipeline
from evaluation.interfaces import adapters
from evaluation import evaluate, GroundTruth
from evaluation import run_selftest


def test_end_to_end():
    run_selftest.main()                       # asserts internally (perfect≈0, scale recovered, calibration flagged)


def test_ct_pipeline_recovers_profile():
    mask, sp, prof = phantom.stenotic_tube(R_mm=5, throat_mm=3, spacing_mm=0.5)
    gt = ct_pipeline.process_ct_mask("t", mask, sp)
    assert abs(gt.dce_mm.min() - 2 * 3) / 6 < 0.10        # throat DCE ~6mm
    assert abs(gt.dce_mm.max() - 2 * 5) / 10 < 0.10       # reference DCE ~10mm


def test_immutability_enforced():
    mask, sp, _ = phantom.stenotic_tube(spacing_mm=0.6)
    gt = ct_pipeline.process_ct_mask("t", mask, sp); gt.assert_immutable()
    gt.csa_mm2[0] += 1.0
    try:
        gt.assert_immutable(); raise AssertionError("mutation not caught")
    except AssertionError as e:
        assert "MUTATED" in str(e)


def test_perfect_method_scores_zero():
    mask, sp, _ = phantom.stenotic_tube(spacing_mm=0.5)
    gt = ct_pipeline.process_ct_mask("t", mask, sp)
    res = adapters.from_profile_arrays("perfect", gt.centerline_mm.copy(), gt.csa_mm2.copy(), units="mm")
    ev = evaluate(res, gt)
    assert ev.DCE_error["DCE_error_pct"]["p50"] < 1.0
    assert ev.centerline_error["mean_mm"] < 0.5
    assert ev.obstruction_error["abs_error_pct"] < 0.5


def test_result_validation_rejects_bad_dce():
    import pytest
    from evaluation import ReconstructionResult
    with pytest.raises(AssertionError):
        ReconstructionResult("bad", np.zeros((4, 3)), np.arange(4.), np.ones(4), np.ones(4) * 99,  # DCE!=2sqrt(CSA/pi)
                             np.ones(4), np.ones(4), units="mm")
