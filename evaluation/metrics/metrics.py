"""All evaluation metrics, computed identically for every method. Outputs are algorithm-agnostic.

Required per method:  CSA error, DCE error, centerline error, stenosis-location error,
                      % obstruction error, uncertainty calibration, runtime.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.spatial import cKDTree


def _agg(err):
    err = np.asarray(err, float); err = err[np.isfinite(err)]
    if not len(err): return dict(rmse=np.nan, mae=np.nan, bias=np.nan, p50=np.nan, p90=np.nan)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                bias=float(np.mean(err)), p50=float(np.median(np.abs(err))), p90=float(np.percentile(np.abs(err), 90)))


def profile_errors(mC_reg, m_arclen, m_csa_mm, m_dce_mm, gt_arclen, gt_csa, gt_dce, gt_C, ct_arc_at_slice):
    """per-slice CSA & DCE error at CT arclength correspondence + relative % errors."""
    gt_csa_i = np.interp(ct_arc_at_slice, gt_arclen, gt_csa)
    gt_dce_i = np.interp(ct_arc_at_slice, gt_arclen, gt_dce)
    csa_abs = m_csa_mm - gt_csa_i; dce_abs = m_dce_mm - gt_dce_i
    csa_rel = 100 * csa_abs / np.maximum(gt_csa_i, 1e-9); dce_rel = 100 * dce_abs / np.maximum(gt_dce_i, 1e-9)
    return dict(CSA_error_mm2=_agg(csa_abs), CSA_error_pct=_agg(csa_rel),
                DCE_error_mm=_agg(dce_abs), DCE_error_pct=_agg(dce_rel),
                per_slice=dict(ct_arclength_mm=ct_arc_at_slice.tolist(), csa_err_mm2=csa_abs.tolist(),
                               dce_err_mm=dce_abs.tolist(), gt_csa=gt_csa_i.tolist(), gt_dce=gt_dce_i.tolist()))


def centerline_error(mC_reg, gt_C):
    """symmetric distance between the registered method centerline and the CT centerline (mm)."""
    d1 = cKDTree(gt_C).query(mC_reg)[0]; d2 = cKDTree(mC_reg).query(gt_C)[0]
    return dict(mean_mm=float(np.mean(np.r_[d1, d2])), hausdorff_mm=float(max(d1.max(), d2.max())),
                p90_mm=float(np.percentile(np.r_[d1, d2], 90)))


def stenosis_location_error(m_arclen, m_csa, ct_arc_at_slice, gt_arclen, gt_csa):
    """arclength distance (mm) between the method's narrowest slice and the CT's narrowest slice."""
    i_m = int(np.argmin(m_csa)); ct_pos_of_m_min = float(ct_arc_at_slice[i_m])
    ct_min = float(gt_arclen[int(np.argmin(gt_csa))])
    return dict(error_mm=abs(ct_pos_of_m_min - ct_min), method_min_ct_arclength_mm=ct_pos_of_m_min, ct_min_arclength_mm=ct_min)


def obstruction(csa):
    """scale-free %% obstruction = 1 - CSA_min / CSA_ref(=95th pct wide slice)."""
    csa = np.asarray(csa, float); return float(100 * (1 - csa.min() / np.percentile(csa, 95)))


def obstruction_error(m_csa, gt_csa):
    om, og = obstruction(m_csa), obstruction(gt_csa)
    return dict(method_pct=round(om, 2), ct_pct=round(og, 2), abs_error_pct=round(abs(om - og), 2))


def uncertainty_calibration(err, sigma):
    """are the reported per-slice sigmas calibrated? z=err/sigma should be ~N(0,1).
       report empirical coverage vs nominal (68/95), calibration factor std(z), and an ECE."""
    err = np.asarray(err, float); sigma = np.asarray(sigma, float)
    m = np.isfinite(err) & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < 3: return dict(n=int(m.sum()), calibrated=None, note="insufficient sigmas")
    z = err[m] / sigma[m]
    cov1 = float(np.mean(np.abs(z) <= 1.0)); cov2 = float(np.mean(np.abs(z) <= 2.0))
    ece = abs(cov1 - 0.6827) + abs(cov2 - 0.9545)
    return dict(n=int(m.sum()), coverage_1sigma=cov1, coverage_2sigma=cov2, expected=(0.6827, 0.9545),
                calibration_factor_std_z=float(np.std(z)), ECE=float(ece),
                calibrated=bool(0.5 <= np.std(z) <= 2.0 and ece < 0.25),
                note="calibration_factor std(z)≈1 is ideal; >1 OVERconfident (errors exceed claimed σ), <1 underconfident")


def band_landmark_errors(method_dce_by_landmark, gt):
    """LANDMARK-BAND mode: does the method's DCE at each landmark fall within the CT dynamic band?
    error = distance to the nearest band edge (0 if inside)."""
    rows = {}
    for lm in gt.landmarks:
        if lm.name not in method_dce_by_landmark or not lm.dce_band_mm: continue
        d = method_dce_by_landmark[lm.name]; lo, hi = lm.dce_band_mm
        inside = lo <= d <= hi; err = 0.0 if inside else (lo - d if d < lo else d - hi)
        rows[lm.name] = dict(method_dce_mm=round(float(d), 3), band_mm=[round(lo, 3), round(hi, 3)],
                             inside_band=bool(inside), edge_error_mm=round(float(err), 3))
    inside_frac = float(np.mean([r["inside_band"] for r in rows.values()])) if rows else np.nan
    return dict(per_landmark=rows, fraction_within_band=inside_frac,
                mean_edge_error_mm=float(np.mean([r["edge_error_mm"] for r in rows.values()])) if rows else np.nan)


@dataclass
class EvaluationResult:
    method: str
    ct_id: str
    mode: str
    scale_recovered: float = float("nan")          # Sim(3) scale (scene->mm) or 1.0 if method is mm
    registration_rmse_mm: float = float("nan")
    CSA_error: dict = field(default_factory=dict)
    DCE_error: dict = field(default_factory=dict)
    centerline_error: dict = field(default_factory=dict)
    stenosis_location_error: dict = field(default_factory=dict)
    obstruction_error: dict = field(default_factory=dict)
    uncertainty_calibration: dict = field(default_factory=dict)
    band_errors: dict = field(default_factory=dict)
    runtime_s: float = float("nan")
    detail: dict = field(default_factory=dict)

    def summary_row(self):
        """the algorithm-agnostic one-line summary the report tabulates."""
        g = lambda d, *k: (d.get(k[0], {}).get(k[1]) if len(k) == 2 else d.get(k[0])) if d else None
        return dict(method=self.method,
                    CSA_err_pct_p50=g(self.CSA_error, "CSA_error_pct", "p50"),
                    DCE_err_pct_p50=g(self.DCE_error, "DCE_error_pct", "p50") if self.DCE_error else g(self.CSA_error, "DCE_error_pct", "p50"),
                    centerline_mm=self.centerline_error.get("mean_mm"),
                    stenosis_loc_mm=self.stenosis_location_error.get("error_mm"),
                    obstruction_err_pct=self.obstruction_error.get("abs_error_pct"),
                    calib_factor=self.uncertainty_calibration.get("calibration_factor_std_z"),
                    within_band=self.band_errors.get("fraction_within_band"),
                    runtime_s=self.runtime_s, scale=self.scale_recovered)

    def to_dict(self): return asdict(self)
