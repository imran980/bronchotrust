"""RESCUE 2 -- Test 2 with a Huber loss + adaptive albedo DOF chosen by BIC. If the stripe CSA bias
drops, the failure was a modeling CHOICE (basis coverage), not a degeneracy (Evidence 2 already
showed full rank). Compare four configs per albedo field: fixed-9DOF+L2 (the original 44.6% case),
fixed-9DOF+Huber, BIC-adaptive-DOF+L2, BIC-adaptive-DOF+Huber. depth-eval env."""
from __future__ import annotations
import json
import numpy as np, sim
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

THETA = np.linspace(0, 2 * np.pi, 120, endpoint=False)
C = sim.make_cameras(N=14, depth=8.0, cone_deg=2.5)
ALBEDOS = ["const", "gradient", "stripes", "lowfreq", "highfreq"]
NFR_GEOM = 4


def fit_cfg(B_obs, vis, nfr_alb, loss):
    p0 = np.array([5.0] + [0.0] * (2 * NFR_GEOM) + [1.0] + [0.0] * (2 * nfr_alb))
    m = vis
    def resid(p):
        Bp, _, _ = sim.model_B(p, THETA, C, NFR_GEOM, nfr_alb, False)
        return (Bp - B_obs)[m]
    if loss == "l2":
        sol = least_squares(resid, p0, method="trf", max_nfev=4000)
    else:
        r0 = resid(p0); sc = 1.5 * np.median(np.abs(r0 - np.median(r0))) + 1e-9
        sol = least_squares(resid, p0, method="trf", loss="huber", f_scale=sc, max_nfev=4000)
    rss = float(np.sum(sol.fun ** 2)); n = int(m.sum()); nparam = len(p0)
    bic = n * np.log(rss / n + 1e-30) + nparam * np.log(n)
    r0, gk, A0, ak, cx, cy = sim.unpack(sol.x, NFR_GEOM, nfr_alb)
    csa, _ = sim.csa_dce(sim.fourier_eval(r0, gk, THETA), THETA)
    return csa, bic


def run(albedo):
    r_true = sim.boundary(THETA, "circle", r0=5.0); A_true = sim.albedo_field(THETA, albedo, A0=1.0)
    B, vis = sim.forward(C, r_true, A_true, THETA); B = B + np.random.default_rng(1).normal(0, 0.003 * B[vis].std(), B.shape)
    csa_true, _ = sim.csa_dce(r_true, THETA)
    def bias(csa): return round(100 * (csa - csa_true) / csa_true, 1)
    # fixed 9 DOF (nfr_alb=4)
    fix_l2, _ = fit_cfg(B, vis, 4, "l2"); fix_h, _ = fit_cfg(B, vis, 4, "huber")
    # BIC-adaptive DOF over nfr_alb = 1..8 (DOF = 1+2*nfr_alb = 3..17)
    def bic_pick(loss):
        best = None
        for na in range(1, 9):
            csa, b = fit_cfg(B, vis, na, loss)
            if best is None or b < best[1]: best = (csa, b, 1 + 2 * na)
        return best
    bic_l2 = bic_pick("l2"); bic_h = bic_pick("huber")
    return dict(albedo=albedo, fixed9_L2=bias(fix_l2), fixed9_Huber=bias(fix_h),
                BIC_L2_bias=bias(bic_l2[0]), BIC_L2_DOF=bic_l2[2],
                BIC_Huber_bias=bias(bic_h[0]), BIC_Huber_DOF=bic_h[2])


def main():
    rows = [run(a) for a in ALBEDOS]
    worst_fixed = max(abs(r["fixed9_L2"]) for r in rows)
    worst_bic = max(abs(r["BIC_L2_bias"]) for r in rows)
    verdict = dict(worst_fixed9_L2=worst_fixed, worst_BIC_L2=worst_bic,
                   BIC_clears_10pct=bool(worst_bic < 10.0),
                   note="if BIC-adaptive bias < 10% while fixed-9DOF was 44.6%, the failure was albedo-DOF choice, not degeneracy")
    open("rescue2_huber_bic.json", "w").write(json.dumps({"verdict": verdict, "rows": rows}, indent=2))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(rows)); w = 0.2
    ax.bar(x - 1.5 * w, [r["fixed9_L2"] for r in rows], w, label="fixed 9DOF + L2 (original)", color="crimson")
    ax.bar(x - 0.5 * w, [r["fixed9_Huber"] for r in rows], w, label="fixed 9DOF + Huber", color="salmon")
    ax.bar(x + 0.5 * w, [r["BIC_L2_bias"] for r in rows], w, label="BIC-adaptive DOF + L2", color="seagreen")
    ax.bar(x + 1.5 * w, [r["BIC_Huber_bias"] for r in rows], w, label="BIC-adaptive DOF + Huber", color="mediumseagreen")
    ax.axhline(10, c="k", ls="--"); ax.axhline(-10, c="k", ls="--")
    ax.set_xticks(x); ax.set_xticklabels([r["albedo"] for r in rows]); ax.set_ylabel("CSA bias %")
    ax.set_title(f"RESCUE 2 — Huber + BIC-adaptive albedo DOF (worst BIC-L2 |bias|={worst_bic}%)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("rescue2_huber_bic.png", dpi=120)
    print("=== RESCUE 2 ===")
    print(f"{'albedo':10}{'fix9_L2':>9}{'fix9_Hub':>9}{'BIC_L2':>8}{'(DOF)':>7}{'BIC_Hub':>9}{'(DOF)':>7}")
    for r in rows: print(f"{r['albedo']:10}{r['fixed9_L2']:>9}{r['fixed9_Huber']:>9}{r['BIC_L2_bias']:>8}{r['BIC_L2_DOF']:>7}{r['BIC_Huber_bias']:>9}{r['BIC_Huber_DOF']:>7}")
    print(f"\nworst fixed-9 L2 = {worst_fixed}% ; worst BIC-adaptive L2 = {worst_bic}% -> BIC clears 10%: {verdict['BIC_clears_10pct']}")


if __name__ == "__main__":
    main()
