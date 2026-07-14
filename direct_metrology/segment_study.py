"""SHORT-AXIAL-SEGMENT direct metrology — the strongest surviving idea, tested honestly.

Photometric term DROPPED. Instead of a single ring, fit a short generalized-cylinder SEGMENT jointly:
centerline (axis + bend), a non-circular cross-section (ellipse + lobe), a local flare/curvature profile
r0(s), and the CSA/DCE profile directly -- from the multi-frame occluding contours (exact tangency
forward model). Gate: it must BEAT the single-circle contour fit AND keep CSA error < 10% on a smooth
elliptical/lobed stenosis phantom. If it fails -> abandon the direct-metrology direction.

Reports, per case: single-circle CSA err; segment CSA err from a data-driven init + random inits
(best / median / convergence); the GT-parameter forward-model residual (the model mismatch that can
displace the optimum). All metrics via the frozen evaluation/ framework. Nothing tuned to pass.
Usage: python -m direct_metrology.segment_study
"""
from __future__ import annotations
import sys, json, copy, time
from pathlib import Path
import numpy as np
import yaml

from . import phantom as PH
from . import detect as DT
from . import estimator as ES
from . import segment as SG
from . import eval_adapter as EA

HERE = Path(__file__).parent
OUT = HERE / "outputs_segment"

CASES = [
    dict(name="circular",        ecc=0.0,  lobe=0.0,  albedo_var=0.0),
    dict(name="ellipse_0.2",     ecc=0.2,  lobe=0.0,  albedo_var=0.0),
    dict(name="trilobe_0.15",    ecc=0.0,  lobe=0.15, albedo_var=0.0),
    dict(name="ellipse+lobe",    ecc=0.15, lobe=0.1,  albedo_var=0.0),   # combined shape, CLEAN
    dict(name="realistic",       ecc=0.15, lobe=0.1,  albedo_var=0.3),   # combined + vascular albedo
]
N_RANDOM_INIT = 4
K4_GAUSS = -3 * 3.0 / 4.0 ** 4       # quartic term matching a Gaussian wall (B=3, w=4)


def _base_cfg():
    return dict(phantom=dict(aperture_model="smooth", R_ref_mm=6.0, r_throat_mm=3.0, length_mm=40.0,
                             stenosis_z_mm=25.0, sigma_mm=4.0, n_cameras=16, cam_depth_mm=12.0,
                             cam_depth_spread_mm=8.0, cam_cone_deg=8.0, albedo=1.0, seed=0,
                             ecc=0.0, lobe=0.0, lobe_k=3, theta0_deg=20.0, albedo_var=0.0, albedo_k=6,
                             intrinsics=dict(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)),
                optimizer=dict(lambda_contour=1.0, lambda_photo=0.02, max_nfev=4000, wall_sample_offset=1.15))


class _FitShim:
    """Adapt a SegFit (throat measurement) to the FitResult shape the frozen eval_adapter expects."""
    def __init__(self, fs):
        a = fs.p.a / np.linalg.norm(fs.p.a)
        self.O = fs.p.O + fs.s_throat * a          # throat centre on the segment
        self.axis = a
        self.r = float(np.sqrt(fs.csa / np.pi))    # equivalent-area radius of the non-circular throat
        self.csa = float(fs.csa); self.dce = float(fs.dce)
        self.csa_var = float((0.05 * fs.csa) ** 2)  # nominal (calibration not used in this study)
        self.dce_var = float((0.05 * fs.dce) ** 2)
        self.contour_rms_px = float(fs.resid_rms_mm); self.photo_rms = float("nan"); self.nfev = int(fs.nfev)


def _seg_err(fs, g):
    return 100 * abs(fs.csa - g["csa_mm2"]) / g["csa_mm2"], 100 * abs(fs.dce - g["dce_mm"]) / g["dce_mm"]


def run_case(case):
    cfg = _base_cfg(); cfg["phantom"].update({k: case[k] for k in ("ecc", "lobe", "albedo_var")})
    P = PH.render(cfg); g = P.gt; cons = DT.detect_all(P.images); Kinv = np.linalg.inv(P.K)
    ndet = sum(c is not None for c in cons)

    # single-circle baseline (through the frozen framework)
    thc = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    frc = ES.fit(cons, P.cameras, P.K, P.images, thc, lam_c=1.0, lam_p=0.0, use_photo=False)
    ev_c, _, _ = EA.evaluate_fit(frc, g)
    circle_csa = ev_c.CSA_error["CSA_error_pct"]["rmse"]

    # forward-model mismatch at GT parameters (0 => exact; large => optimum can be displaced)
    tgt = SG.theta_from(g["throat_center_mm"], [0, 0, 1.0], r_t=g["throat_radius_mm"], s_t=0.0,
                        kappa=(g["R_ref_mm"] - g["throat_radius_mm"]) / cfg["phantom"]["sigma_mm"] ** 2,
                        kappa4=K4_GAUSS, e=case["ecc"], phi=np.radians(cfg["phantom"]["theta0_deg"]),
                        lobe=case["lobe"])
    rr = SG.residuals(tgt, cons, P.cameras, Kinv, 3.0, 3)
    gt_resid_mm = float(np.sqrt(np.mean(rr[:-2] ** 2)))

    # segment: data-driven init + random inits
    th0 = SG.init_from_contours(cons, P.cameras, P.K, g["stenosis_z_mm"], kappa=0.15, kappa4=K4_GAUSS)
    inits = [("data", th0)]
    rng = np.random.default_rng(7)
    for k in range(N_RANDOM_INIT):
        j = th0.copy()
        j[7] += rng.normal(0, 0.3)                       # r_t
        j[11] += rng.normal(0, 0.1)                      # e
        j[12] += rng.normal(0, 0.6)                      # phi
        j[13] += rng.normal(0, 0.1)                      # lobe
        inits.append((f"rand{k}", j))
    fits = []
    for tag, ti in inits:
        fs = SG.fit(cons, P.cameras, P.K, ti, half_len=3.0, max_nfev=600)
        ev, _, _ = EA.evaluate_fit(_FitShim(fs), g)
        csa_e = ev.CSA_error["CSA_error_pct"]["rmse"]; dce_e = ev.DCE_error["DCE_error_pct"]["rmse"]
        fits.append(dict(init=tag, csa_err_pct=float(csa_e), dce_err_pct=float(dce_e),
                         r_t=float(fs.p.r_t), e=float(fs.p.e), lobe=float(fs.p.lobe),
                         kappa=float(fs.p.kappa), resid_rms_mm=float(fs.resid_rms_mm)))
    csa_errs = np.array([f["csa_err_pct"] for f in fits])
    best = float(csa_errs.min()); med = float(np.median(csa_errs))
    conv = float(np.mean(csa_errs < 10.0))
    return dict(case=case["name"], gt_csa_mm2=g["csa_mm2"], n_detected=ndet,
                circle_csa_err_pct=float(circle_csa), gt_forward_resid_mm=gt_resid_mm,
                seg_csa_err_best_pct=best, seg_csa_err_median_pct=med, seg_frac_below_10pct=conv,
                data_init_csa_err_pct=fits[0]["csa_err_pct"], fits=fits)


def _figure(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [r["case"] for r in rows]; x = np.arange(len(rows))
    fig, a = plt.subplots(figsize=(9, 3.8))
    a.bar(x - 0.25, [r["circle_csa_err_pct"] for r in rows], 0.25, label="single-circle", color="silver")
    a.bar(x, [r["seg_csa_err_best_pct"] for r in rows], 0.25, label="segment (best init)", color="seagreen")
    a.bar(x + 0.25, [r["seg_csa_err_median_pct"] for r in rows], 0.25, label="segment (median init)", color="indianred")
    a.axhline(10, color="k", ls=":", label="pass gate 10% CSA")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8, rotation=8); a.set_ylabel("CSA error %")
    a.set_title("Short-segment vs single-circle: fixes isolated shapes, FAILS the combined realistic throat")
    a.legend(fontsize=8); a.set_ylim(0, max(60, max(r['circle_csa_err_pct'] for r in rows) * 1.1))
    fig.tight_layout(); fig.savefig(OUT / "fig_segment_vs_circle.png", dpi=120); plt.close(fig)
    return ["fig_segment_vs_circle.png"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [run_case(c) for c in CASES]
    # GATE: for EVERY case, segment must beat single-circle AND stay < 10% CSA (use median init = robust)
    per = []
    for r in rows:
        beats = r["seg_csa_err_median_pct"] < r["circle_csa_err_pct"]
        below = r["seg_csa_err_median_pct"] < 10.0
        per.append(dict(case=r["case"], beats_circle=bool(beats), below_10pct=bool(below),
                        seg_median=r["seg_csa_err_median_pct"], circle=r["circle_csa_err_pct"]))
    overall = bool(all(p["beats_circle"] and p["below_10pct"] for p in per))
    figs = _figure(rows)
    summary = dict(verdict=("PASS" if overall else "FAIL"),
                   proceed_to_real_video=bool(overall),
                   decision=("test real video" if overall else "ABANDON the direct-metrology direction"),
                   gate="segment beats single-circle AND <10% CSA (median over inits) in EVERY case",
                   per_case_gate=per, cases=rows, runtime_s=time.time() - t0, figures=figs,
                   note=("Photometric term dropped. Short generalized-cylinder segment (centerline, "
                         "non-circular cross-section, local flare, CSA/DCE profile) vs single-circle. "
                         "Smooth elliptical/lobed phantom. Synthetic only; no novelty/clinical claim."))
    (OUT / "results_segment.json").write_text(json.dumps(summary, indent=2,
                                              default=lambda o: float(o) if isinstance(o, np.floating) else o))
    _print(summary)
    return summary


def _print(s):
    print("=" * 92)
    print("SHORT-AXIAL-SEGMENT DIRECT METROLOGY  (photometry dropped; honest gate)")
    print("=" * 92)
    print("  case             det  circle_CSA%   seg_best%  seg_median%  frac<10%  GT_fwd_resid(mm)  beats&<10%")
    for r, p in zip(s["cases"], s["per_case_gate"]):
        ok = "PASS" if (p["beats_circle"] and p["below_10pct"]) else "FAIL"
        print("  %-14s  %2d   %+8.2f    %+7.2f    %+8.2f     %.2f      %.3f            %s" % (
            r["case"], r["n_detected"], r["circle_csa_err_pct"], r["seg_csa_err_best_pct"],
            r["seg_csa_err_median_pct"], r["seg_frac_below_10pct"], r["gt_forward_resid_mm"], ok))
    print("-" * 92)
    print(f"  VERDICT: {s['verdict']}   ->   {s['decision']}")
    print("=" * 92)


if __name__ == "__main__":
    main()
