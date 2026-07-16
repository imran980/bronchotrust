"""Optimization-ROBUSTNESS study for the short-segment method. Model / loss / priors / evaluator are
UNCHANGED; this only changes initialization and solution selection, and measures convergence.

Approach (all techniques the user allowed):
  - multistart initialization: many data-driven + jittered inits, with DIVERSE kappa seeds (no GT);
  - reject local minima by FINAL RESIDUAL: the multistart method returns the lowest-residual solution;
  - (staged fitting circle->ellipse->lobe and explicit x_scale were also tried; see the report — staged
    HURT because the circle stage lets the r_t-kappa degeneracy set a bad basin, so the deployed method
    is plain-fit multistart with residual selection).

Runs >=100 random initializations on the realistic phantom, reports convergence / CSA-DCE / residual
distributions / failure modes, and estimates the multistart method's convergence by bootstrap grouping.
Deterministic. Usage: python -m direct_metrology.segment_robust_study
"""
from __future__ import annotations
import json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from . import phantom as PH
from . import detect as DT
from . import segment as SG

HERE = Path(__file__).parent
OUT = HERE / "outputs_segment"
K4_GAUSS = -3 * 3.0 / 4.0 ** 4
N_INIT_REALISTIC = 105                   # >= 100 random initializations as required
N_INIT_OTHER = 12
KAPPA_SEEDS = [0.1, 0.2, 0.35]           # diverse flare seeds (the runaway parameter) -- NOT GT
MAXNFEV = 250                            # trust-region iteration cap (LM/TRF setting; caps runaway fits)
WORKERS = 24
CASES = [
    dict(name="circular",     ecc=0.0,  lobe=0.0,  albedo_var=0.0),
    dict(name="ellipse_0.2",  ecc=0.2,  lobe=0.0,  albedo_var=0.0),
    dict(name="trilobe_0.15", ecc=0.0,  lobe=0.15, albedo_var=0.0),
    dict(name="ellipse+lobe", ecc=0.15, lobe=0.1,  albedo_var=0.0),
    dict(name="realistic",    ecc=0.15, lobe=0.1,  albedo_var=0.3),
]


def _cfg(case):
    return dict(phantom=dict(aperture_model="smooth", R_ref_mm=6.0, r_throat_mm=3.0, length_mm=40.0,
                             stenosis_z_mm=25.0, sigma_mm=4.0, n_cameras=16, cam_depth_mm=12.0,
                             cam_depth_spread_mm=8.0, cam_cone_deg=8.0, albedo=1.0, seed=0,
                             ecc=case["ecc"], lobe=case["lobe"], lobe_k=3, theta0_deg=20.0,
                             albedo_var=case["albedo_var"], albedo_k=6,
                             intrinsics=dict(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)))


def _make_inits(cons, cams, K, z_guess, n, seed):
    """n diverse inits from the DATA-driven init + jitter, cycling through diverse kappa seeds. No GT."""
    base = SG.init_from_contours(cons, cams, K, z_guess, kappa=0.15, kappa4=K4_GAUSS)
    rng = np.random.default_rng(seed)
    inits = []
    for i in range(n):
        j = base.copy()
        kap = KAPPA_SEEDS[i % len(KAPPA_SEEDS)]
        j[9] = np.log(np.expm1(max(kap, 1e-3)))          # kappa seed (softplus-inverted)
        j[7] += rng.normal(0, 0.4)                       # r_t
        j[8] += rng.normal(0, 0.6)                       # s_t
        j[11] += rng.normal(0, 0.15)                     # e
        j[12] += rng.uniform(-np.pi, np.pi)              # phi (full range -> genuinely random)
        j[13] += rng.normal(0, 0.15)                     # lobe
        inits.append(j)
    return inits


# ---- parallel worker (plain fit; single-threaded BLAS per worker) ----
_W = {}
def _winit(cons, cams, Kmat):
    _W["cons"], _W["cams"], _W["K"] = cons, cams, Kmat


def _worker(task):
    idx, init = task
    fs = SG.fit(_W["cons"], _W["cams"], _W["K"], np.asarray(init), max_nfev=MAXNFEV)
    return dict(idx=idx, csa=fs.csa, dce=fs.dce, resid=fs.resid_rms_mm, r_t=fs.p.r_t,
                e=fs.p.e, lobe=fs.p.lobe, kappa=fs.p.kappa, csa_var=fs.csa_var)


def _run_parallel(cons, cams, K, inits, workers=WORKERS):
    with ProcessPoolExecutor(max_workers=workers, initializer=_winit, initargs=(cons, cams, K)) as ex:
        return list(ex.map(_worker, list(enumerate(inits))))


def _multistart_bootstrap(res, gt_csa, Ks=(3, 5, 8), n_trials=5000, seed=0):
    """Estimate the multistart method's convergence: sample K fits, keep the LOWEST-residual one,
    check its CSA error < 10%. Uses only observed residual (no GT in the selection)."""
    rng = np.random.default_rng(seed)
    resid = np.array([r["resid"] for r in res]); csa = np.array([r["csa"] for r in res])
    csa_err = 100 * np.abs(csa - gt_csa) / gt_csa
    out = {}
    for K in Ks:
        ok = 0
        for _ in range(n_trials):
            idx = rng.choice(len(res), size=K, replace=False)
            pick = idx[np.argmin(resid[idx])]                 # reject local minima by residual
            ok += (csa_err[pick] < 10.0)
        out[K] = ok / n_trials
    return out


def _dist(vals):
    v = np.asarray(vals, float)
    return dict(min=float(v.min()), p10=float(np.percentile(v, 10)), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(v.max()), mean=float(v.mean()))


def _failure_modes(res, gt_csa):
    csa_err = np.array([100 * abs(r["csa"] - gt_csa) / gt_csa for r in res])
    fails = [r for r, e in zip(res, csa_err) if e >= 10.0]
    clusters = {}
    for r in fails:
        key = (round(r["r_t"], 1), round(r["e"], 1), round(r["lobe"], 1), round(min(r["kappa"], 20), 0))
        c = clusters.setdefault(key, dict(count=0, csa_err=[], resid=[]))
        c["count"] += 1; c["csa_err"].append(100 * abs(r["csa"] - gt_csa) / gt_csa); c["resid"].append(r["resid"])
    return [dict(r_t=k[0], e=k[1], lobe=k[2], kappa=k[3], count=c["count"],
                 csa_err_mean=float(np.mean(c["csa_err"])), resid_mean=float(np.mean(c["resid"])))
            for k, c in sorted(clusters.items(), key=lambda kv: -kv[1]["count"])]


_CASE_SEED = {"circular": 11, "ellipse_0.2": 22, "trilobe_0.15": 33, "ellipse+lobe": 44, "realistic": 55}


def run_case(case, n_init):
    cfg = _cfg(case); P = PH.render(cfg); g = P.gt; cons = DT.detect_all(P.images)
    inits = _make_inits(cons, P.cameras, P.K, g["stenosis_z_mm"], n_init, seed=_CASE_SEED[case["name"]])
    res = _run_parallel(cons, P.cameras, P.K, inits)
    csa_err = np.array([100 * abs(r["csa"] - g["csa_mm2"]) / g["csa_mm2"] for r in res])
    dce_err = np.array([100 * abs(r["dce"] - g["dce_mm"]) / g["dce_mm"] for r in res])
    resid = np.array([r["resid"] for r in res])
    per_init_conv = float(np.mean(csa_err < 10.0))
    # residual-selection validity: does the lowest-residual fit have <10% CSA?
    ibest = int(np.argmin(resid)); best_is_ok = bool(csa_err[ibest] < 10.0)
    boot = _multistart_bootstrap(res, g["csa_mm2"], n_trials=(5000 if case["name"] == "realistic" else 2000))
    return dict(case=case["name"], n_init=n_init, gt_csa_mm2=g["csa_mm2"], gt_dce_mm=g["dce_mm"],
                per_init_convergence=per_init_conv,
                lowest_residual_csa_err_pct=float(csa_err[ibest]), lowest_residual_is_below10=best_is_ok,
                multistart_convergence=boot,
                csa_err_dist=_dist(csa_err), dce_err_dist=_dist(dce_err), resid_dist=_dist(resid),
                failure_modes=_failure_modes(res, g["csa_mm2"]),
                all_csa_err=[float(x) for x in csa_err], all_resid=[float(x) for x in resid])


def _kcurve(csa_err, resid, Kmax=12, n_trials=8000, seed=0):
    rng = np.random.default_rng(seed); csa_err = np.asarray(csa_err); resid = np.asarray(resid)
    curve = {}
    for K in range(1, Kmax + 1):
        ok = 0
        for _ in range(n_trials):
            idx = rng.choice(len(csa_err), size=min(K, len(csa_err)), replace=False)
            ok += csa_err[idx[np.argmin(resid[idx])]] < 10.0
        curve[K] = ok / n_trials
    return curve


def _figure(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in rows:
        ax[0].plot(list(r["kcurve"].keys()), [100 * v for v in r["kcurve"].values()], "o-", ms=3, label=r["case"])
    ax[0].axhline(95, color="k", ls=":", label="95% target"); ax[0].set_xlabel("multistart K")
    ax[0].set_ylabel("convergence %"); ax[0].set_title("Multistart convergence vs K"); ax[0].legend(fontsize=7)
    ax[0].set_ylim(60, 101)
    real = next(r for r in rows if r["case"] == "realistic")
    ce = np.array(real["all_csa_err"]); rz = np.array(real["all_resid"])
    ax[1].hist(ce, bins=np.linspace(0, 60, 31), color="indianred", edgecolor="k")
    ax[1].axvline(10, color="k", ls=":"); ax[1].set_xlabel("CSA error %"); ax[1].set_ylabel("count")
    ax[1].set_title(f"Realistic per-init CSA err ({real['n_init']} inits)")
    ax[2].scatter(rz, ce, s=14, c=np.where(ce < 10, "seagreen", "indianred")); ax[2].axhline(10, color="k", ls=":")
    ax[2].set_xlabel("final residual (mm)"); ax[2].set_ylabel("CSA error %"); ax[2].set_xlim(0.03, 0.2)
    ax[2].set_ylim(0, 60); ax[2].set_title("Residual selects the low-CSA basin")
    fig.tight_layout(); fig.savefig(OUT / "fig_robustness.png", dpi=110); plt.close(fig)
    return "fig_robustness.png"


def _summarize(rows, runtime_s):
    for r in rows:
        r["kcurve"] = _kcurve(r["all_csa_err"], r["all_resid"])
        r["min_K_for_95pct"] = next((K for K, v in r["kcurve"].items() if v >= 0.95), None)
    real = next(r for r in rows if r["case"] == "realistic")
    mink = real["min_K_for_95pct"]
    proceed = bool(mink is not None and real["lowest_residual_is_below10"])
    fig = _figure(rows)
    summary = dict(
        realistic_min_K_for_95pct=mink,
        realistic_convergence_at_min_K=(real["kcurve"][mink] if mink else None),
        realistic_convergence_at_K8=real["kcurve"][8],
        proceed_to_real_video=proceed,
        decision=(f"multistart K>={mink} reaches >=95% convergence -> convergence bar MET (still synthetic)"
                  if proceed else "convergence <95% for all K tried -> DO NOT proceed"),
        runtime_s=runtime_s, figures=[fig], cases=rows,
        optimization_changes=[
            "multistart initialization (data-driven + jitter, DIVERSE kappa seeds; no GT, no answer-tuning)",
            "reject local minima by FINAL RESIDUAL (multistart returns the lowest-residual solution)",
            "explicit per-parameter x_scale + staged circle->ellipse->lobe were TRIED but HURT and are NOT used",
            "trust-region max_nfev cap (250) to bound runaway fits"],
        note="Model/loss/priors/evaluator UNCHANGED; optimization robustness only.")
    (OUT / "results_robustness.json").write_text(json.dumps(summary, indent=2,
                                                 default=lambda o: float(o) if isinstance(o, np.floating) else o))
    _print(summary)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    rows = [run_case(c, N_INIT_REALISTIC if c["name"] == "realistic" else N_INIT_OTHER) for c in CASES]
    return _summarize(rows, time.time() - t0)


def finalize_from_json():
    """Recompute the K-curve / decision / figure from a prior run's saved per-fit data (no re-fitting)."""
    prev = json.loads((OUT / "results_robustness.json").read_text())
    rows = prev["cases"]
    for r in rows:                                   # strip stale derived fields; fix JSON string keys
        r.pop("kcurve", None); r.pop("min_K_for_95pct", None)
        r["multistart_convergence"] = {int(k): v for k, v in r["multistart_convergence"].items()}
    return _summarize(rows, prev.get("runtime_s", float("nan")))


def _print(s):
    print("=" * 100)
    print("SEGMENT OPTIMIZATION-ROBUSTNESS  (model/loss/evaluator unchanged; multistart + residual reject)")
    print("=" * 100)
    print("  case           n_init  per-init<10%   multistart<10% (K=3/5/8)   minK95  lowest-resid CSA%")
    for r in s["cases"]:
        m = r["multistart_convergence"]
        print("  %-13s   %3d      %5.1f%%       %5.1f / %5.1f / %5.1f %%      %3s      %+6.2f%%" % (
            r["case"], r["n_init"], 100 * r["per_init_convergence"], 100 * m[3], 100 * m[5], 100 * m[8],
            str(r.get("min_K_for_95pct")), r["lowest_residual_csa_err_pct"]))
    print("-" * 100)
    r = next(c for c in s["cases"] if c["case"] == "realistic")
    print("  REALISTIC (%d inits) distributions:" % r["n_init"])
    print("    CSA err %%:  %s" % {k: round(v, 2) for k, v in r["csa_err_dist"].items()})
    print("    DCE err %%:  %s" % {k: round(v, 2) for k, v in r["dce_err_dist"].items()})
    print("    residual mm:%s" % {k: round(v, 4) for k, v in r["resid_dist"].items()})
    print("    failure modes (wrong minima; note resid >= global):")
    for fm in r["failure_modes"][:5]:
        print("      count %2d  r_t=%.1f e=%+.1f lobe=%+.1f kappa~%.0f  CSA_err~%.0f%%  resid~%.3f" % (
            fm["count"], fm["r_t"], fm["e"], fm["lobe"], fm["kappa"], fm["csa_err_mean"], fm["resid_mean"]))
    print("-" * 100)
    print("  realistic: per-init %.1f%% -> multistart minK=%s reaches %.1f%% (K=8: %.1f%%)" % (
        100 * r["per_init_convergence"], s["realistic_min_K_for_95pct"],
        100 * (s["realistic_convergence_at_min_K"] or 0), 100 * s["realistic_convergence_at_K8"]))
    print(f"  DECISION: {s['decision']}")
    print("=" * 100)


if __name__ == "__main__":
    import sys
    finalize_from_json() if "--finalize" in sys.argv else main()
