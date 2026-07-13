"""Bridge the single-cross-section direct-metrology fit into the FROZEN evaluation/ framework.

The prototype measures ONE ring (the stenosis throat). The evaluator is profile-based, so we present
the single throat measurement as a short straight profile segment of constant CSA along the fitted
axis, and compare it to a GT profile segment of constant GT-throat CSA. The prototype is METRIC
(poses + intrinsics are in mm) -> units="mm" -> the Sim(3) scale is fixed to 1 (no CT scale recovery).
Reported: CSA/DCE absolute + % error, covariance-based calibration, obstruction (0 for a single ring).
The evaluation code itself is NOT modified."""
from __future__ import annotations
import numpy as np
from evaluation.interfaces.method_interface import ReconstructionResult
from evaluation.ground_truth.ground_truth import GroundTruth
from evaluation.ground_truth.ground_truth import Landmark
from evaluation import evaluator as EV

_N = 9                     # samples along the short throat segment (single-ring method -> constant CSA)
_SPAN = 6.0                # mm span of the reported segment


def result_from_fit(fr, method="direct_metrology", runtime_s=float("nan"), confidence=0.8):
    """FitResult -> ReconstructionResult (a short straight segment at the fitted throat, constant CSA)."""
    s = np.linspace(-_SPAN / 2, _SPAN / 2, _N)
    C = fr.O[None, :] + s[:, None] * (fr.axis / np.linalg.norm(fr.axis))[None, :]
    arc = s - s.min()
    return ReconstructionResult(
        method=method, centerline=C, arclength=arc,
        csa=np.full(_N, fr.csa), dce=np.full(_N, fr.dce),
        covariance=np.full(_N, max(fr.csa_var, 1e-12)),
        confidence=np.full(_N, float(np.clip(confidence, 0, 1))),
        units="mm", frame="phantom", runtime_s=runtime_s,
        meta=dict(contour_rms_px=fr.contour_rms_px, photo_rms=fr.photo_rms, nfev=fr.nfev))


def gt_from_phantom(gt: dict, ct_id="phantom_throat") -> GroundTruth:
    """phantom GT dict -> immutable profile GroundTruth (constant GT-throat CSA over the same segment)."""
    s = np.linspace(-_SPAN / 2, _SPAN / 2, _N); arc = s - s.min()
    z = gt["stenosis_z_mm"]
    C = np.c_[np.zeros(_N), np.zeros(_N), z + s]
    csa = np.full(_N, gt["csa_mm2"])
    lm = (Landmark("stenosis", arclength_mm=float(arc[_N // 2]),
                   csa_band_mm2=(gt["csa_mm2"], gt["csa_mm2"]), dce_band_mm=(gt["dce_mm"], gt["dce_mm"])),)
    return GroundTruth.from_profile(ct_id, C, arc, csa, landmarks=lm,
                                    provenance=dict(source="synthetic stenosis-throat phantom",
                                                    throat_radius_mm=gt["throat_radius_mm"]))


def evaluate_fit(fr, gt: dict, runtime_s=float("nan"), confidence=0.8):
    """Run the fit through the frozen evaluator; return (EvaluationResult, ReconstructionResult, GroundTruth)."""
    res = result_from_fit(fr, runtime_s=runtime_s, confidence=confidence)
    g = gt_from_phantom(gt)
    ev = EV.evaluate(res, g)
    return ev, res, g
