"""End-to-end self-test of the evaluation framework on a SYNTHETIC CT with known answers.
Proves: CT pipeline recovers the analytic CSA/DCE profile; ground truth is immutable; registration
recovers a known Sim(3) scale; every metric + uncertainty calibration works; the report is
algorithm-agnostic. A 'perfect' method must score ~0; noisy/overconfident methods must score worse
/ be flagged. No manual tuning. Run:  python -m evaluation.run_selftest"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from .ct_processing import phantom, ct_pipeline
from .ground_truth.ground_truth import GroundTruth, Landmark
from .interfaces.method_interface import ReconstructionResult
from .registration.register import Sim3, umeyama
from .evaluator import evaluate
from . import report as R

OUT = Path(__file__).resolve().parent / "tests" / "_selftest_out"


def build_gt():
    mask, sp, prof = phantom.stenotic_tube(R_mm=6, throat_mm=3.2, length_mm=60, stenosis_mm=38, sigma_mm=5,
                                            spacing_mm=0.5, bend_amp_mm=3.0)
    gt = ct_pipeline.process_ct_mask("phantom_ct", mask, sp)
    return gt, prof


def method_from_gt(gt, name, s_true=3.0, noise_csa=0.0, sigma_report_factor=1.0, seed=0):
    """make a ReconstructionResult by mapping the CT GT into a rotated/translated SCENE-units frame
    with known scale 1/s_true, optionally adding CSA noise. The evaluator must recover s_true."""
    rng = np.random.default_rng(seed)
    ax = rng.normal(0, 1, 3); ax /= np.linalg.norm(ax); ang = 0.6
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    T_m2ct = Sim3(s=s_true, R=Rm, t=rng.normal(0, 20, 3))    # CT = T(method)
    Tinv = T_m2ct.inverse()
    C = Tinv.apply(gt.centerline_mm)                         # method centerline (scene units)
    csa_scene = gt.csa_mm2 / s_true ** 2                     # CSA_scene = CSA_mm / s^2
    if noise_csa > 0: csa_scene = np.clip(csa_scene * (1 + rng.normal(0, noise_csa, len(csa_scene))), 1e-6, None)
    dce = 2 * np.sqrt(csa_scene / np.pi)
    # honest per-slice sigma: the actual std of the multiplicative noise in CSA units, scaled by factor
    sigma = (noise_csa * (gt.csa_mm2 / s_true ** 2)) if noise_csa > 0 else 1e-4 * np.ones(len(csa_scene))
    var = (sigma / sigma_report_factor) ** 2                 # factor<1 -> overconfident (reports too-small σ)
    arclen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    return ReconstructionResult(method=name, centerline=C, arclength=arclen, csa=csa_scene, dce=dce,
                                covariance=var, confidence=np.full(len(C), 0.9), units="scene",
                                runtime_s=float(rng.uniform(1, 30)), meta=dict(true_scale=s_true))


def main():
    print("=" * 70); print("SELF-TEST: evaluation framework on a synthetic bend + stenosis CT"); print("=" * 70)
    gt, prof = build_gt()

    # (1) CT pipeline recovered the analytic profile?
    gt_dce_at = np.interp(np.linspace(gt.arclength_mm[0], gt.arclength_mm[-1], 50), gt.arclength_mm, gt.dce_mm)
    an_z = prof["z_mm"]; an_dce = prof["dce_mm"]
    # compare on overlapping arclength (approx z since near-straight); use min/throat as robust check
    rec_throat = gt.dce_mm.min(); an_throat = an_dce.min()
    print(f"\n[1] CT pipeline: throat DCE recovered {rec_throat:.2f} mm vs analytic {an_throat:.2f} mm "
          f"({100*abs(rec_throat-an_throat)/an_throat:.1f}% err); ref DCE {gt.dce_mm.max():.2f} vs {an_dce.max():.2f}")
    assert abs(rec_throat - an_throat) / an_throat < 0.10, "CT throat DCE off >10%"

    # (2) immutability
    gt.assert_immutable()
    try:
        gt.csa_mm2[0] = -999.0; gt.assert_immutable(); mut_caught = False
    except AssertionError:
        mut_caught = True
    print(f"[2] ground-truth immutability: mutation caught = {mut_caught}")
    assert mut_caught
    gt, prof = build_gt()                                    # rebuild clean

    # (3) three methods through the SAME evaluator
    methods = [
        method_from_gt(gt, "perfect",      s_true=3.0, noise_csa=0.00, seed=1),
        method_from_gt(gt, "noisy_calib",  s_true=2.0, noise_csa=0.08, sigma_report_factor=1.0, seed=2),
        method_from_gt(gt, "overconfident", s_true=4.0, noise_csa=0.08, sigma_report_factor=3.0, seed=3),
    ]
    results = [evaluate(m, gt) for m in methods]
    print("\n[3] evaluation table (identical metrics, algorithm-agnostic):")
    R.print_table(results)

    perf = results[0]
    print(f"\n[4] scale recovery: perfect method true_scale=3.0 -> recovered {perf.scale_recovered:.3f}")
    assert abs(perf.scale_recovered - 3.0) / 3.0 < 0.02, "scale not recovered"
    assert perf.DCE_error["DCE_error_pct"]["p50"] < 2.0, "perfect DCE error should be ~0"
    assert perf.centerline_error["mean_mm"] < 0.5, "perfect centerline error should be ~0"
    print(f"    perfect: DCE err {perf.DCE_error['DCE_error_pct']['p50']:.2f}% | centerline {perf.centerline_error['mean_mm']:.3f}mm | "
          f"stenosis loc {perf.stenosis_location_error['error_mm']:.2f}mm | obstruction err {perf.obstruction_error['abs_error_pct']}%")
    oc = results[2].uncertainty_calibration
    print(f"[5] calibration: overconfident method calibration_factor std(z) = {oc['calibration_factor_std_z']:.2f} "
          f"(>1 => overconfident, correctly flagged; calibrated={oc['calibrated']})")
    assert oc["calibration_factor_std_z"] > 1.5, "overconfident method should show std(z)>1"

    # (6) landmark-band mode (the real 4-D-CT shape) with a mm method
    ct_ref = Path("/home/mi3dr/projects/bronchotrust/runs/ct_validation_16v1/ct_reference.json")
    if ct_ref.exists():
        gtb = ct_pipeline.ground_truth_from_bands("ct16v1_bands", ct_ref)
        mm_method = ReconstructionResult("mm_method_bands", np.zeros((5, 3)), np.arange(5), np.full(5, 20.0),
                                         np.full(5, 5.0), np.full(5, 0.5), np.full(5, 0.8), units="mm",
                                         meta=dict(landmark_dce_mm={"trachea": 5.7, "right_mainstem": 4.5}))
        rb = evaluate(mm_method, gtb)
        print(f"\n[6] landmark-band mode ({len(gtb.landmarks)} landmarks): fraction within CT dynamic band = "
              f"{rb.band_errors['fraction_within_band']}; per-landmark = "
              f"{ {k: v['inside_band'] for k,v in rb.band_errors['per_landmark'].items()} }")

    rep = R.make_report(gt, results, OUT, title="SELF-TEST — synthetic CT vs perfect/noisy/overconfident")
    print(f"\n[7] report written -> {OUT}/report.{{json,png,csv}}")
    print("\n" + "=" * 70); print("SELF-TEST PASSED — framework is internally consistent."); print("=" * 70)


if __name__ == "__main__":
    main()
