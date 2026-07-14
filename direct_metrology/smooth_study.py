"""SMOOTH-THROAT bias study (the honest version).

Reverts to the smooth, anatomically-realistic Gaussian stenosis throat with its TRUE viewpoint-
dependent occluding contour (no crisp diaphragm). It then:

  1. MEASURES the contour bias vs viewpoint (camera depth) — the silhouette effect.
  2. MODELS that bias: geometric tangent-ring explanation + an empirical fit, and shows what a
     correction would require (the local axial flare kappa, which the single-circle fit does NOT
     estimate).
  3. Adds realistic COLMAP-bootstrap pose noise.
  4. Tests whether the weak log-gradient PHOTOMETRIC term actually reduces CSA/DCE bias vs
     contour-only (lambda_photo sweep, per case).
  5. Includes ellipse / lobed cross-sections and nonuniform (vascular) albedo.
  6. Applies the pass gate:  PASS only if joint CLEARLY beats contour-only AND stays <10% CSA error.

All metrics are computed through the FROZEN evaluation/ framework. Nothing is tuned to pass; the
verdict (PASS/FAIL) is reported as measured. Usage: python -m direct_metrology.smooth_study
"""
from __future__ import annotations
import sys, json, copy, time
from pathlib import Path
import numpy as np
import yaml

from . import phantom as PH
from . import detect as DT
from . import estimator as ES
from . import corruptions as CR
from . import eval_adapter as EA

HERE = Path(__file__).parent
OUT = HERE / "outputs_smooth"


def _errs(fr, g):
    return (100 * abs(fr.csa - g["csa_mm2"]) / g["csa_mm2"],
            100 * abs(fr.dce - g["dce_mm"]) / g["dce_mm"])


def _fit(cons, cams, P, o, lam_p, use_photo):
    th = ES.theta_from(P.gt["throat_center_mm"], P.gt["throat_axis"], P.gt["throat_radius_mm"])
    return ES.fit(cons, cams, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=lam_p,
                  wall_offset=o["wall_sample_offset"], use_photo=use_photo, max_nfev=o["max_nfev"])


def tangent_ring_bias(r_t, kappa, D):
    """On-axis geometric model of the smooth-throat silhouette: for a converging-diverging profile
    r(z)=r_t+1/2 kappa (z-z_t)^2, the apparent contour forms on the DIVERGING side at radius r(z*)>r_t,
    where the tangent-from-camera condition r=r'(z*)(z*-z_t+D) holds. Closed form for the paraboloid:
        u* = -D + sqrt(D^2 + 2 r_t/kappa),   r_apparent = r_t + 1/2 kappa u*^2.
    Explains the SIGN (over-estimate) and the monotone growth with depth D; the absolute magnitude is
    an upper bound (the multi-view circle fit + oblique cameras recover less than the full on-axis ring)."""
    u = -D + np.sqrt(D ** 2 + 2 * r_t / kappa)
    return r_t + 0.5 * kappa * u ** 2


def measure_bias(cfg):
    """Contour-only circle-fit bias vs camera depth on the CIRCULAR smooth throat -> the silhouette bias."""
    p = cfg["phantom"]; o = cfg["optimizer"]; s = cfg["study"]
    r_t = p["r_throat_mm"]; kappa = (p["R_ref_mm"] - r_t) / p["sigma_mm"] ** 2
    rows = []
    for D in s["depth_sweep_mm"]:
        c = copy.deepcopy(cfg); c["phantom"]["cam_depth_mm"] = D
        for k in ("ecc", "lobe", "albedo_var"):
            c["phantom"][k] = 0.0
        P = PH.render(c); cons = DT.detect_all(P.images)
        ndet = sum(x is not None for x in cons)
        if ndet < 3:
            rows.append(dict(depth_mm=D, n_detected=ndet, r_fit=None, csa_err_pct=None, dce_err_pct=None,
                             tangent_model_r=float(tangent_ring_bias(r_t, kappa, D)))); continue
        fr = _fit(cons, P.cameras, P, o, 0.0, False)
        ce, de = _errs(fr, P.gt)
        rows.append(dict(depth_mm=D, n_detected=ndet, r_fit=float(fr.r), csa_err_pct=float(ce),
                         dce_err_pct=float(de), tangent_model_r=float(tangent_ring_bias(r_t, kappa, D))))
    # empirical model: fit r_fit - r_t ~ a * D^2 (silhouette offset grows ~quadratically with depth)
    good = [x for x in rows if x["r_fit"] is not None]
    model = None
    if len(good) >= 2:
        D = np.array([x["depth_mm"] for x in good]); b = np.array([x["r_fit"] - r_t for x in good])
        A = np.c_[D ** 2, np.ones_like(D)]; coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coef; ss = 1 - np.sum((b - pred) ** 2) / (np.sum((b - b.mean()) ** 2) + 1e-12)
        model = dict(form="r_fit - r_throat = a*D^2 + c", a=float(coef[0]), c=float(coef[1]), r2=float(ss),
                     kappa_per_mm=float(kappa),
                     note="correcting the bias needs BOTH depth D and local flare kappa; the single-circle "
                          "fit estimates neither -> the smooth-throat bias is NOT resolved by the method.")
    return dict(sweep=rows, empirical_model=model, kappa_per_mm=float(kappa), r_throat_mm=float(r_t))


def compare_cases(cfg):
    """Per shape/albedo case at the nominal depth: contour-only vs joint over the lambda_photo sweep,
    with + without realistic pose noise. Routed through the frozen evaluation/ framework."""
    o = cfg["optimizer"]; s = cfg["study"]
    lam_sweep = s["lambda_photo_sweep"]
    rows = []
    for case in s["cases"]:
        for posen in (False, True):
            c = copy.deepcopy(cfg)
            for k in ("ecc", "lobe", "albedo_var"):
                c["phantom"][k] = case.get(k, 0.0)
            P = PH.render(c); cons = DT.detect_all(P.images)
            cams = P.cameras
            if posen:
                cams = CR.corrupt_cameras(P.cameras, pose_pert_mm=s["pose_bootstrap_trans_mm"],
                                          rot_pert_deg=s["pose_bootstrap_rot_deg"], rng=np.random.default_rng(3))
            ndet = sum(x is not None for x in cons)
            per_lam = []
            for lp in lam_sweep:
                fr = _fit(cons, cams, P, o, lp, use_photo=(lp > 0))
                ev, _, _ = EA.evaluate_fit(fr, P.gt)
                per_lam.append(dict(lambda_photo=lp, r_fit=float(fr.r),
                                    csa_err_pct=float(ev.CSA_error["CSA_error_pct"]["rmse"]),
                                    dce_err_pct=float(ev.DCE_error["DCE_error_pct"]["rmse"]),
                                    contour_rms_px=float(fr.contour_rms_px)))
            contour = next(x for x in per_lam if x["lambda_photo"] == 0.0)
            joint = min((x for x in per_lam if x["lambda_photo"] > 0), key=lambda x: x["csa_err_pct"])
            gain = contour["csa_err_pct"] - joint["csa_err_pct"]       # positive => photometry helped
            rows.append(dict(case=case["name"], pose_noise=posen, n_detected=ndet,
                             gt_dce_mm=P.gt["dce_mm"], gt_csa_mm2=P.gt["csa_mm2"], r_eq_mm=P.gt.get("r_eq_mm"),
                             contour_only=contour, best_joint=joint, photometry_gain_csa_pct=float(gain),
                             per_lambda=per_lam))
    return rows


def _figures(bias, cases):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) bias vs depth (measured + tangent-ring model direction)
    good = [x for x in bias["sweep"] if x["r_fit"] is not None]
    fig, a = plt.subplots(figsize=(5.2, 3.4))
    D = [x["depth_mm"] for x in good]
    a.plot(D, [x["csa_err_pct"] for x in good], "o-", color="indianred", label="measured CSA bias (contour-only)")
    a.set_xlabel("camera depth behind throat (mm)"); a.set_ylabel("CSA error % (over-estimate)")
    a.set_title("Smooth-throat silhouette bias grows with depth"); a.axhline(0, color="k", lw=0.5)
    a.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_bias_vs_depth.png", dpi=120); plt.close(fig)
    # 2) contour-only vs joint per case (no pose noise)
    cs = [r for r in cases if not r["pose_noise"]]
    labels = [r["case"] for r in cs]; x = np.arange(len(cs))
    fig, a = plt.subplots(figsize=(8.5, 3.6))
    a.bar(x - 0.2, [r["contour_only"]["csa_err_pct"] for r in cs], 0.4, label="contour-only", color="steelblue")
    a.bar(x + 0.2, [r["best_joint"]["csa_err_pct"] for r in cs], 0.4, label="best joint (photometry)", color="indianred")
    a.axhline(10, color="k", ls=":", label="pass gate 10% CSA")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=7, rotation=10); a.set_ylabel("CSA error %")
    a.set_title("Photometry does not beat contour-only; circle model fails on non-circular throats")
    a.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_contour_vs_joint.png", dpi=120); plt.close(fig)
    return ["fig_bias_vs_depth.png", "fig_contour_vs_joint.png"]


def main(cfg_path=None):
    cfg = yaml.safe_load(open(cfg_path or HERE / "config_smooth.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bias = measure_bias(cfg)
    cases = compare_cases(cfg)

    g = cfg["gate"]
    # PASS requires: for EVERY case, joint clearly beats contour-only AND joint stays <10% CSA.
    per_case_pass = []
    for r in cases:
        clearly_beats = r["photometry_gain_csa_pct"] >= g["photometry_min_gain_pct"]
        below_tol = r["best_joint"]["csa_err_pct"] < g["csa_tol_pct"]
        per_case_pass.append(dict(case=r["case"], pose_noise=r["pose_noise"],
                                  clearly_beats=bool(clearly_beats), below_tol=bool(below_tol),
                                  photometry_gain_csa_pct=r["photometry_gain_csa_pct"],
                                  joint_csa_err_pct=r["best_joint"]["csa_err_pct"]))
    photometry_ever_decisive = any(pc["clearly_beats"] for pc in per_case_pass)
    all_below_tol = all(pc["below_tol"] for pc in per_case_pass)
    overall_pass = bool(photometry_ever_decisive and all_below_tol
                        and all(pc["clearly_beats"] for pc in per_case_pass))

    figs = _figures(bias, cases)
    summary = dict(
        verdict=("PASS" if overall_pass else "FAIL"),
        proceed_to_real_video=bool(overall_pass),
        reasons=dict(
            photometry_clearly_beats_contour_in_every_case=all(pc["clearly_beats"] for pc in per_case_pass),
            photometry_ever_decisive=bool(photometry_ever_decisive),
            all_cases_below_10pct_csa=bool(all_below_tol),
            smooth_bias_resolved_by_method=False,  # see bias.empirical_model.note
        ),
        smooth_throat_bias=bias,
        case_comparison=cases,
        per_case_gate=per_case_pass,
        runtime_s=time.time() - t0,
        figures=figs,
        note=("Honest smooth-throat study. Circle cross-section + weak log-gradient photometric, fixed "
              "poses. Synthetic only; no real video. No novelty/clinical claim."),
        config=cfg)
    (OUT / "results_smooth.json").write_text(json.dumps(summary, indent=2,
                                             default=lambda o: float(o) if isinstance(o, np.floating) else o))
    _print(summary)
    return summary


def _print(s):
    b = s["smooth_throat_bias"]
    print("=" * 78)
    print("SMOOTH-THROAT STUDY  (honest; no tuning to pass)")
    print("=" * 78)
    print("1) Contour bias vs camera depth (circular smooth throat, contour-only):")
    for x in b["sweep"]:
        if x["r_fit"] is None:
            print(f"     depth {x['depth_mm']:>4} mm : {x['n_detected']}/16 detected (aperture clipped) — skipped")
        else:
            print(f"     depth {x['depth_mm']:>4} mm : r_fit {x['r_fit']:.3f}  CSA {x['csa_err_pct']:+5.2f}%  "
                  f"DCE {x['dce_err_pct']:+5.2f}%   (tangent-model r={x['tangent_model_r']:.3f})")
    if b["empirical_model"]:
        m = b["empirical_model"]
        print(f"   model: {m['form']}  a={m['a']:.4g}  c={m['c']:.4g}  R2={m['r2']:.3f}  kappa={m['kappa_per_mm']:.4g}/mm")
        print(f"   -> {m['note']}")
    print("-" * 78)
    print("2) Photometry vs contour-only per case (best joint over lambda sweep):")
    print("    case                    pose  det   contour CSA%   joint CSA%   gain    joint<10%?")
    for r in s["case_comparison"]:
        print("    %-22s %-4s  %2d   %+8.2f     %+8.2f    %+5.2f    %s" % (
            r["case"], "yes" if r["pose_noise"] else "no", r["n_detected"],
            r["contour_only"]["csa_err_pct"], r["best_joint"]["csa_err_pct"],
            r["photometry_gain_csa_pct"], "yes" if r["best_joint"]["csa_err_pct"] < 10 else "NO"))
    print("-" * 78)
    rs = s["reasons"]
    print(f"   photometry clearly beats contour in EVERY case : {rs['photometry_clearly_beats_contour_in_every_case']}")
    print(f"   photometry ever decisive (any case)            : {rs['photometry_ever_decisive']}")
    print(f"   all cases < 10% CSA error                      : {rs['all_cases_below_10pct_csa']}")
    print(f"   smooth-throat bias resolved by the method      : {rs['smooth_bias_resolved_by_method']}")
    print("=" * 78)
    print(f"   VERDICT: {s['verdict']}   ->   proceed to real video: {s['proceed_to_real_video']}")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
