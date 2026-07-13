"""The evaluator harness: ReconstructionResult + GroundTruth -> EvaluationResult, via
register -> unit-resolve (Sim3 scale) -> slice-correspond -> metrics. Identical for every method."""
from __future__ import annotations
import numpy as np
from .interfaces.method_interface import ReconstructionResult
from .ground_truth.ground_truth import GroundTruth
from .registration import register as reg
from .metrics import metrics as M


def evaluate(result: ReconstructionResult, gt: GroundTruth) -> M.EvaluationResult:
    result.validate(); gt.assert_immutable()
    out = M.EvaluationResult(method=result.method, ct_id=gt.ct_id, mode=gt.mode, runtime_s=result.runtime_s)

    if gt.mode == "profile":
        with_scale = (result.units == "scene")         # scene-units -> recover scale from CT; mm -> scale fixed 1
        T = reg.register(result.centerline, gt.centerline_mm, with_scale=with_scale)
        out.scale_recovered = T.s; out.registration_rmse_mm = T.rmse_mm
        mC_reg = T.apply(result.centerline)
        csa_mm, csa_var_mm, dce_mm, _ = reg.propagate_csa(result.csa, result.csa_std() ** 2, T)
        ct_arc, _ = reg.slice_correspondence(mC_reg, result.arclength, gt.centerline_mm, gt.arclength_mm)
        pe = M.profile_errors(mC_reg, result.arclength, csa_mm, dce_mm, gt.arclength_mm, gt.csa_mm2, gt.dce_mm, gt.centerline_mm, ct_arc)
        out.CSA_error = {k: v for k, v in pe.items() if k.startswith("CSA")}
        out.DCE_error = {k: v for k, v in pe.items() if k.startswith("DCE")}
        out.centerline_error = M.centerline_error(mC_reg, gt.centerline_mm)
        out.stenosis_location_error = M.stenosis_location_error(result.arclength, result.csa, ct_arc, gt.arclength_mm, gt.csa_mm2)
        out.obstruction_error = M.obstruction_error(result.csa, gt.csa_mm2)   # scale-free (ratio) -> units cancel
        # calibration: per-slice CSA error (mm^2) vs propagated sigma (mm^2)
        gt_csa_i = np.interp(ct_arc, gt.arclength_mm, gt.csa_mm2)
        out.uncertainty_calibration = M.uncertainty_calibration(csa_mm - gt_csa_i, np.sqrt(np.clip(csa_var_mm, 0, None)))
        out.detail = dict(per_slice=pe["per_slice"], sim3=dict(s=T.s, rmse_mm=T.rmse_mm, scale_var=T.scale_var))

    elif gt.mode == "landmark_bands":
        # No CT centerline to register to. Compare the method's DCE AT each named landmark against the
        # CT dynamic band. The method supplies meta['landmark_dce_mm'] (absolute mm) when it has a
        # scale; a scale-free method can still be checked on the ratio-based obstruction if it labels
        # a stenosis + reference landmark.
        ldce = result.meta.get("landmark_dce_mm", {})
        if ldce:
            out.band_errors = M.band_landmark_errors(ldce, gt)
        # scale-free obstruction stays comparable if the method exposes a full profile
        if len(result.csa) >= 5 and gt.landmark("stenosis") is not None:
            st = gt.landmark("stenosis")
            if st.csa_band_mm2 and result.meta.get("reference_landmark") and st.dce_band_mm:
                pass  # (obstruction across landmarks handled by band_errors above)
        out.detail = dict(note="landmark-band mode: DCE-within-band per landmark; absolute mm requires a scaled method")
    else:
        raise ValueError(gt.mode)
    return out
