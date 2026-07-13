"""Deterministic driver for the smallest direct-metrology prototype.

Pipeline (NO manual editing, NO hidden state): render synthetic stenosis-throat phantom -> detect
lumen occluding contours -> fit the throat circle (O, axis, r) by contour reprojection + weak
gain-invariant log-gradient photometric -> evaluate through the FROZEN evaluation/ framework.

Runs the 6 correctness gates + corruption sensitivity + writes JSON / ablation table / figures.
Usage:  python -m direct_metrology.run [config.yaml]

Every number is paired with the check that makes it trustworthy (per CLAUDE.md's one rule).
Nothing here tunes to pass; if a gate fails it is reported as FAIL."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import yaml

from . import phantom as PH
from . import detect as DT
from . import estimator as ES
from . import corruptions as CR
from . import eval_adapter as EA

HERE = Path(__file__).parent
OUT = HERE / "outputs"


# --------------------------------------------------------------------------- helpers
def _pct(a, b):
    return 100.0 * abs(a - b) / abs(b)


def _fit_pipeline(P, cons, cams, theta0, cfg):
    o = cfg["optimizer"]
    return ES.fit(cons, cams, P.K, P.images, theta0,
                  lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                  wall_offset=o["wall_sample_offset"], max_nfev=o["max_nfev"])


def _random_theta(gt, spread, rng):
    O = np.array(gt["throat_center_mm"], float) + rng.normal(0, spread["center_mm"], 3)
    r = gt["throat_radius_mm"] * (1 + rng.uniform(-spread["radius_frac"], spread["radius_frac"]))
    ang = np.radians(spread["axis_deg"])
    ax = np.array([rng.uniform(-1, 1) * np.tan(ang), rng.uniform(-1, 1) * np.tan(ang), 1.0])
    return ES.theta_from(O, ax, max(0.3, r))


def _exact_contours(P, gt):
    """Project the GT throat circle through each camera -> noise-free contour observations.
    Isolates the ESTIMATOR from the detector (Gate 1a)."""
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    O = np.array(gt["throat_center_mm"]); r = gt["throat_radius_mm"]
    circ = O[None, :] + r * np.c_[np.cos(th), np.sin(th), np.zeros(len(th))]
    cons = []
    for cam in P.cameras:
        uv, z = ES._project(circ, cam.R, cam.C, P.K)
        cons.append(uv[z > 0])
    return cons


def _sim3_cams(cams, s, Rg, tg):
    """Transform cameras by a Sim(3) so the SAME images are consistent with a world scaled by s:
    a fit against the same contours must recover r' = s*r (scale-free ratio invariant)."""
    out = []
    for cam in cams:
        Cp = s * (Rg @ cam.C) + tg
        Rp = cam.R @ Rg.T
        out.append(PH.Camera(R=Rp, C=Cp))
    return out


def _rot(ax, deg):
    w = np.asarray(ax, float); w = w / (np.linalg.norm(w) + 1e-12) * np.radians(deg)
    th = np.linalg.norm(w) + 1e-12; k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


# --------------------------------------------------------------------------- gates
def gate1_gt_init(P, cfg):
    """GT-init recovers GT: (a) estimator on EXACT projected contours -> ~machine-exact;
       (b) full pipeline (rendered+detected) GT-init -> within clean tol."""
    gt = P.gt; th_gt = ES.theta_from(gt["throat_center_mm"], gt["throat_axis"], gt["throat_radius_mm"])
    # (a) estimator self-consistency on exact contours
    exact = _exact_contours(P, gt)
    fa = _fit_pipeline(P, exact, P.cameras, th_gt, cfg)
    err_a = _pct(fa.dce, gt["dce_mm"])
    # (b) full pipeline
    cons = DT.detect_all(P.images)
    ndet = sum(c is not None for c in cons)
    fb = _fit_pipeline(P, cons, P.cameras, th_gt, cfg)
    err_b_dce = _pct(fb.dce, gt["dce_mm"]); err_b_csa = _pct(fb.csa, gt["csa_mm2"])
    tol = cfg["gates"]["clean_tol_pct"]
    ok = (err_a < 0.5) and (err_b_dce < tol) and (err_b_csa < tol)
    return dict(name="gate1_gt_init", passed=bool(ok),
                exact_contour_dce_err_pct=err_a,
                pipeline_dce_err_pct=err_b_dce, pipeline_csa_err_pct=err_b_csa,
                n_detected=ndet, n_views=len(P.images),
                note="(a) estimator on exact contours ~0; (b) rendered+detected within clean tol"), cons, fb


def gate2_random_inits(P, cons, ref, cfg):
    """>= n_random_init random inits converge to the same optimum (basin of the GT-init fit)."""
    g = cfg["gates"]; rng = np.random.default_rng(P.gt.get("stenosis_z_mm", 1) and 12345)
    n = g["n_random_init"]; dces = []; succ = 0
    for _ in range(n):
        th = _random_theta(P.gt, g["init_spread"], rng)
        fr = _fit_pipeline(P, cons, P.cameras, th, cfg)
        dces.append(fr.dce)
        if fr.success and _pct(fr.dce, ref.dce) < 5.0:
            succ += 1
    rate = succ / n
    return dict(name="gate2_random_inits", passed=bool(rate >= g["convergence_min"]),
                n=n, convergence_rate=rate, min_required=g["convergence_min"],
                dce_mean=float(np.mean(dces)), dce_std=float(np.std(dces)),
                note="converged := success AND |DCE-DCE_ref|/DCE_ref < 5%"), dces


def gate3_sim3_invariance(P, cons, ref, cfg):
    """Apply random Sim(3) to cameras; recovered DCE must scale by exactly s (scale-free ratio invariant).
       Also a pure SE(3) (s=1) must leave metric DCE identical."""
    rng = np.random.default_rng(7)
    checks = []
    for s in (1.0, 0.5, 2.0, 3.3):
        Rg = _rot(rng.uniform(-1, 1, 3), rng.uniform(5, 40)); tg = rng.uniform(-20, 20, 3)
        cams2 = _sim3_cams(P.cameras, s, Rg, tg)
        th0 = ES.theta_from(s * (Rg @ np.array(P.gt["throat_center_mm"])) + tg, Rg @ np.array(P.gt["throat_axis"]),
                            s * P.gt["throat_radius_mm"])
        fr = _fit_pipeline(P, cons, cams2, th0, cfg)
        ratio = fr.dce / (s * ref.dce)                     # should be 1.0
        checks.append(dict(s=s, dce=fr.dce, ratio_to_expected=float(ratio),
                           ratio_err_pct=_pct(ratio, 1.0)))
    max_err = max(c["ratio_err_pct"] for c in checks)
    return dict(name="gate3_sim3_invariance", passed=bool(max_err < 1.0),
                max_ratio_err_pct=max_err, checks=checks,
                note="recovered DCE / (s*DCE_ref) must equal 1 for every Sim(3) -> scale-free ratio invariant")


def gate4_ablations(P, cons, cfg):
    """contour-only vs photo-only vs joint. Expected: contour~joint (accurate); photo alone underdetermined."""
    gt = P.gt; th = ES.theta_from(gt["throat_center_mm"], gt["throat_axis"], gt["throat_radius_mm"])
    o = cfg["optimizer"]
    def run(uc, up):
        fr = ES.fit(cons, P.cameras, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                    wall_offset=o["wall_sample_offset"], use_contour=uc, use_photo=up, max_nfev=o["max_nfev"])
        return dict(dce=fr.dce, csa=fr.csa, dce_err_pct=_pct(fr.dce, gt["dce_mm"]),
                    csa_err_pct=_pct(fr.csa, gt["csa_mm2"]), contour_rms_px=fr.contour_rms_px)
    joint = run(True, True); contour = run(True, False); photo = run(False, True)
    ok = (joint["dce_err_pct"] < cfg["gates"]["clean_tol_pct"] and
          contour["dce_err_pct"] < cfg["gates"]["clean_tol_pct"] and
          photo["dce_err_pct"] > 5 * joint["dce_err_pct"])     # photo-only must be clearly worse
    return dict(name="gate4_ablations", passed=bool(ok), joint=joint, contour_only=contour, photo_only=photo,
                note="joint & contour-only accurate; photo-only alone cannot recover radius (weak term, by design)")


def gate5_jacobian_cov(P, cons, cfg):
    """Jacobian / SVD / covariance reproducible + full-rank (identifiable)."""
    gt = P.gt; th = ES.theta_from(gt["throat_center_mm"], gt["throat_axis"], gt["throat_radius_mm"])
    def one():
        fr = _fit_pipeline(P, cons, P.cameras, th, cfg)
        return fr
    f1 = one(); f2 = one()
    # rebuild J at solution
    o = cfg["optimizer"]; args = (cons, P.cameras, P.K, P.images, o["lambda_contour"], o["lambda_photo"],
                                  o["wall_sample_offset"], True, True)
    from scipy.optimize import least_squares
    th1 = ES.theta_from(f1.O, f1.axis, f1.r)
    sol = least_squares(ES.residuals, th1, args=args, method="trf", max_nfev=1)
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    cond = float(sv[0] / sv[-1]); rank = int(np.sum(sv > 1e-9 * sv[0]))
    repro = (abs(f1.r - f2.r) < 1e-9 and abs(f1.r_var - f2.r_var) < 1e-12)
    ok = repro and rank == 6 and np.isfinite(f1.r_var)
    return dict(name="gate5_jacobian_cov", passed=bool(ok),
                reproducible=bool(repro), jac_rank=rank, jac_cond=cond,
                singular_values=sv.tolist(), r_var=f1.r_var, csa_var=f1.csa_var, dce_var=f1.dce_var,
                note="two runs identical to 1e-9; J full-rank(6) -> throat identifiable; cov finite")


def gate6_determinism(P, cons, cfg):
    """Whole pipeline re-run from the same config is bit-identical (no hidden state / manual editing)."""
    P2 = PH.render(cfg); cons2 = DT.detect_all(P2.images)
    gt = P.gt; th = ES.theta_from(gt["throat_center_mm"], gt["throat_axis"], gt["throat_radius_mm"])
    a = _fit_pipeline(P, cons, P.cameras, th, cfg); b = _fit_pipeline(P2, cons2, P2.cameras, th, cfg)
    img_same = all(np.array_equal(x, y) for x, y in zip(P.images, P2.images))
    fit_same = abs(a.r - b.r) < 1e-12
    ok = img_same and fit_same
    return dict(name="gate6_determinism", passed=bool(ok), images_identical=bool(img_same),
                fit_identical=bool(fit_same), note="render+detect+fit reproduce bit-for-bit")


# --------------------------------------------------------------------------- corruption sensitivity
def sensitivity(P, cfg):
    """One corruption axis at a time (others at 0) + a worst-case all-on. Report CSA/DCE err via the
    frozen evaluator, convergence over random inits, and the reported uncertainty (must grow)."""
    c = cfg["corruptions"]; g = cfg["gates"]
    axes = []
    for lv in c["contour_noise_px"]: axes.append(("contour_noise_px", lv))
    for lv in c["specular_frac"]:
        if lv > 0: axes.append(("specular_frac", lv))
    for lv in c["albedo_var"]:
        if lv > 0: axes.append(("albedo_var", lv))
    for lv in c["pose_pert_mm"]:
        if lv > 0: axes.append(("pose_pert_mm", lv))
    axes.append(("ALL", None))
    rows = []
    for name, lv in axes:
        rng = np.random.default_rng(2024)
        spec = c["specular_frac"][-1] if name in ("specular_frac", "ALL") else 0.0
        alb = c["albedo_var"][-1] if name in ("albedo_var", "ALL") else 0.0
        cn = (lv if name == "contour_noise_px" else (c["contour_noise_px"][-1] if name == "ALL" else 0.0))
        pp = (lv if name == "pose_pert_mm" else (c["pose_pert_mm"][-1] if name == "ALL" else 0.0))
        if name == "specular_frac": spec = lv
        if name == "albedo_var": alb = lv
        imgs = CR.corrupt_images(P.images, specular_frac=spec, albedo_var=alb, rng=rng)
        Pc = PH.Phantom(images=imgs, cameras=P.cameras, K=P.K, width=P.width, height=P.height, gt=P.gt)
        cons = DT.detect_all(imgs)
        cons = CR.corrupt_contours(cons, noise_px=cn, rng=rng)
        cams = CR.corrupt_cameras(P.cameras, pose_pert_mm=pp, rot_pert_deg=0.0, rng=rng)
        ndet = sum(x is not None for x in cons)
        th = ES.theta_from(P.gt["throat_center_mm"], P.gt["throat_axis"], P.gt["throat_radius_mm"])
        fr = _fit_pipeline(Pc, cons, cams, th, cfg)
        ev, _, _ = EA.evaluate_fit(fr, P.gt)
        csa_pct = ev.CSA_error["CSA_error_pct"]["rmse"]; dce_pct = ev.DCE_error["DCE_error_pct"]["rmse"]
        # convergence over random inits under this corruption
        rng2 = np.random.default_rng(99); succ = 0; ninit = 20
        for _ in range(ninit):
            thr = _random_theta(P.gt, g["init_spread"], rng2)
            frr = _fit_pipeline(Pc, cons, cams, thr, cfg)
            if frr.success and _pct(frr.dce, fr.dce) < 5.0: succ += 1
        rows.append(dict(axis=name, level=lv, n_detected=ndet, dce_err_pct=dce_pct, csa_err_pct=csa_pct,
                         dce_sigma=float(np.sqrt(max(fr.dce_var, 0))), contour_rms_px=fr.contour_rms_px,
                         convergence=succ / ninit))
    return rows


# --------------------------------------------------------------------------- figures
def _figures(P, cons, fit_ref, dces, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) detection + fit reprojection on 3 views
    th = np.linspace(0, 2 * np.pi, 200); e1, e2 = ES._basis(fit_ref.axis)
    circ = fit_ref.O[None, :] + fit_ref.r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    idx = [i for i, c in enumerate(cons) if c is not None][:3]
    fig, ax = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
    if len(idx) == 1: ax = [ax]
    for k, i in enumerate(idx):
        ax[k].imshow(P.images[i], cmap="gray", vmin=0, vmax=1)
        ax[k].plot(cons[i][:, 0], cons[i][:, 1], ".", ms=1.5, color="lime", label="detected")
        uv, z = ES._project(circ, P.cameras[i].R, P.cameras[i].C, P.K)
        ax[k].plot(uv[:, 0], uv[:, 1], "-", lw=1.2, color="red", label="fit reproj")
        ax[k].set_title(f"view {i}"); ax[k].axis("off")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Detected throat contour (green) vs fitted-circle reprojection (red)")
    fig.tight_layout(); fig.savefig(OUT / "fig_detection.png", dpi=110); plt.close(fig)
    # 2) convergence histogram over random inits
    fig, a = plt.subplots(figsize=(5, 3.2))
    a.hist(dces, bins=20, color="steelblue", edgecolor="k")
    a.axvline(P.gt["dce_mm"], color="red", ls="--", label=f"GT DCE={P.gt['dce_mm']:.2f}")
    a.set_xlabel("recovered DCE (mm)"); a.set_ylabel("count"); a.set_title("Random-init convergence"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_convergence.png", dpi=110); plt.close(fig)
    # 3) sensitivity: error + uncertainty vs corruption
    labels = [f"{r['axis']}\n{r['level']}" for r in sens]
    fig, a = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(sens))
    a.bar(x - 0.2, [r["dce_err_pct"] for r in sens], 0.4, label="DCE err %", color="indianred")
    a.bar(x + 0.2, [r["csa_err_pct"] for r in sens], 0.4, label="CSA err %", color="steelblue")
    a.axhline(10, color="k", ls=":", label="corrupt tol 10%")
    a2 = a.twinx(); a2.plot(x, [r["dce_sigma"] for r in sens], "o-", color="green", label="DCE σ (mm)")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=6); a.set_ylabel("error %"); a2.set_ylabel("DCE σ (mm)")
    a.set_title("Corruption sensitivity: error (bars) + reported uncertainty (green)"); a.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_sensitivity.png", dpi=110); plt.close(fig)
    return ["fig_detection.png", "fig_convergence.png", "fig_sensitivity.png"]


# --------------------------------------------------------------------------- main
def main(cfg_path=None):
    cfg = yaml.safe_load(open(cfg_path or HERE / "config.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = PH.render(cfg)
    gt = P.gt

    g1, cons, fit_ref = gate1_gt_init(P, cfg)
    g2, dces = gate2_random_inits(P, cons, fit_ref, cfg)
    g3 = gate3_sim3_invariance(P, cons, fit_ref, cfg)
    g4 = gate4_ablations(P, cons, cfg)
    g5 = gate5_jacobian_cov(P, cons, cfg)
    g6 = gate6_determinism(P, cons, cfg)
    gates = [g1, g2, g3, g4, g5, g6]

    # clean evaluation through the frozen framework
    ev, res, gtruth = EA.evaluate_fit(fit_ref, gt, runtime_s=time.time() - t0,
                                      confidence=float(np.clip(1 / (1 + fit_ref.contour_rms_px), 0, 1)))
    clean = dict(csa_gt_mm2=gt["csa_mm2"], csa_pred_mm2=fit_ref.csa,
                 dce_gt_mm=gt["dce_mm"], dce_pred_mm=fit_ref.dce,
                 csa_abs_err_mm2=abs(fit_ref.csa - gt["csa_mm2"]), dce_abs_err_mm=abs(fit_ref.dce - gt["dce_mm"]),
                 csa_err_pct=ev.CSA_error["CSA_error_pct"]["rmse"], dce_err_pct=ev.DCE_error["DCE_error_pct"]["rmse"],
                 csa_var=fit_ref.csa_var, dce_var=fit_ref.dce_var,
                 contour_rms_px=fit_ref.contour_rms_px, photo_rms=fit_ref.photo_rms,
                 obstruction_pct=ev.obstruction_error, scale_recovered=ev.scale_recovered,
                 gt_checksum=gtruth._checksum)

    sens = sensitivity(P, cfg)

    # ---- overall pass gate ----
    clean_ok = clean["csa_err_pct"] < cfg["gates"]["clean_tol_pct"] and clean["dce_err_pct"] < cfg["gates"]["clean_tol_pct"]
    corrupt_ok = all(r["dce_err_pct"] < cfg["gates"]["corrupt_tol_pct"] and
                     r["csa_err_pct"] < cfg["gates"]["corrupt_tol_pct"] for r in sens)
    conv_ok = g2["convergence_rate"] >= cfg["gates"]["convergence_min"] and \
              all(r["convergence"] >= cfg["gates"]["convergence_min"] for r in sens)
    clean_sigma = np.sqrt(max(clean["dce_var"], 0))
    worst_sigma = max(r["dce_sigma"] for r in sens)
    unc_ok = worst_sigma >= clean_sigma            # uncertainty grows under corruption
    gates_ok = all(x["passed"] for x in gates)
    overall = bool(clean_ok and corrupt_ok and conv_ok and unc_ok and gates_ok)

    figs = _figures(P, cons, fit_ref, dces, sens)

    summary = dict(
        overall_pass=overall,
        criteria=dict(clean_below_5pct=bool(clean_ok), corrupt_below_10pct=bool(corrupt_ok),
                      convergence_ge_95pct=bool(conv_ok), uncertainty_increases=bool(unc_ok),
                      all_correctness_gates=bool(gates_ok),
                      clean_dce_sigma=float(clean_sigma), worst_dce_sigma=float(worst_sigma)),
        gates=gates, clean_evaluation=clean, sensitivity=sens, figures=figs,
        config=cfg, note="direct-metrology smallest prototype; synthetic phantom only; no real video; no novelty/clinical claim")

    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o))
    _write_ablation_csv(gates, sens, clean)
    _print_report(summary)
    return summary


def _write_ablation_csv(gates, sens, clean):
    lines = ["stage,setting,dce_err_pct,csa_err_pct,convergence,dce_sigma_mm,n_detected"]
    lines.append(f"clean,joint,{clean['dce_err_pct']:.3f},{clean['csa_err_pct']:.3f},,{np.sqrt(max(clean['dce_var'],0)):.4f},")
    g4 = next(g for g in gates if g["name"] == "gate4_ablations")
    for k in ("joint", "contour_only", "photo_only"):
        lines.append(f"ablation,{k},{g4[k]['dce_err_pct']:.3f},{g4[k]['csa_err_pct']:.3f},,,")
    for r in sens:
        lines.append(f"corruption,{r['axis']}={r['level']},{r['dce_err_pct']:.3f},{r['csa_err_pct']:.3f},"
                     f"{r['convergence']:.2f},{r['dce_sigma']:.4f},{r['n_detected']}")
    (OUT / "ablation.csv").write_text("\n".join(lines) + "\n")


def _print_report(s):
    print("=" * 72)
    print("DIRECT-METROLOGY PROTOTYPE  —  gate report")
    print("=" * 72)
    for g in s["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']}")
    c = s["clean_evaluation"]
    print("-" * 72)
    print(f"  CLEAN:  DCE {c['dce_pred_mm']:.3f} vs GT {c['dce_gt_mm']:.3f}  ({c['dce_err_pct']:.2f}%)   "
          f"CSA {c['csa_pred_mm2']:.2f} vs {c['csa_gt_mm2']:.2f} ({c['csa_err_pct']:.2f}%)")
    print(f"          contour_rms={c['contour_rms_px']:.2f}px  DCE σ={np.sqrt(max(c['dce_var'],0)):.4f}mm")
    print("-" * 72)
    print("  SENSITIVITY:")
    for r in s["sensitivity"]:
        print(f"    {r['axis']:>16}={str(r['level']):<5}  DCE {r['dce_err_pct']:5.2f}%  CSA {r['csa_err_pct']:5.2f}%  "
              f"conv {r['convergence']:.2f}  σ={r['dce_sigma']:.4f}  det {r['n_detected']}")
    print("-" * 72)
    cr = s["criteria"]
    for k in ("clean_below_5pct", "corrupt_below_10pct", "convergence_ge_95pct", "uncertainty_increases", "all_correctness_gates"):
        print(f"  {'PASS' if cr[k] else 'FAIL'}  {k}")
    print("=" * 72)
    print(f"  OVERALL: {'PASS' if s['overall_pass'] else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
