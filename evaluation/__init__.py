"""Airway reconstruction EVALUATION framework.

Build once, reuse for every reconstruction method (Barbour, COLMAP, a future optimizer, LightNeuS,
...). This package NEVER runs or modifies reconstruction -- it only evaluates a method's output
against a frozen CT ground truth. See PROTOCOL.md.

    from evaluation import evaluate, make_report, GroundTruth, ReconstructionResult
    from evaluation.interfaces import adapters
    from evaluation.ct_processing import ct_pipeline

    gt   = ct_pipeline.process_ct_mask("ct_id", mask, spacing)          # or ground_truth_from_bands(...)
    res  = adapters.from_profile_arrays("myMethod", centerline, csa, units="scene")
    ev   = evaluate(res, gt)                                            # identical for every method
    make_report(gt, [ev_A, ev_B, ev_C], "out/")
"""
from .interfaces.method_interface import ReconstructionResult, ReconstructionMethod, CalibratedVideo
from .ground_truth.ground_truth import GroundTruth, Landmark
from .evaluator import evaluate
from .report import make_report, print_table

__all__ = ["ReconstructionResult", "ReconstructionMethod", "CalibratedVideo",
           "GroundTruth", "Landmark", "evaluate", "make_report", "print_table"]
