"""TEST 1 -- model mismatch. Feed the CIRCULAR surface-of-revolution estimator progressively harder
cross-sections (SoR = axisymmetric = circular by assumption). Measure CSA bias, DCE bias, Jacobian
rank, condition number. PASS iff CSA bias < 10% across realistic deviations. depth-eval env."""
from __future__ import annotations
import json
import numpy as np, sim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

THETA = np.linspace(0, 2 * np.pi, 96, endpoint=False)
CONES = {"realistic_2_5deg": 2.5, "ideal_15deg": 15.0}
PHANTOMS = [
    ("ideal_circle", dict(kind="circle", r0=5.0), (0, 0)),
    ("ellipse_b/a=0.7", dict(kind="ellipse", r0=5.0, ecc=0.7), (0, 0)),
    ("ellipse_b/a=0.5", dict(kind="ellipse", r0=5.0, ecc=0.5), (0, 0)),
    ("lobed_3x25%", dict(kind="lobed", r0=5.0, k=3, amp=0.25), (0, 0)),
    ("asym_scar_35%", dict(kind="asym_scar", r0=5.0, depth=0.35, phi=1.0), (0, 0)),
    ("noncircular", dict(kind="noncirc", r0=5.0), (0, 0)),
    ("offcenter_0.20R", dict(kind="circle", r0=5.0), (1.0, 0.0)),
    ("offcenter_0.40R", dict(kind="circle", r0=5.0), (2.0, 0.0)),
]


def run(cone):
    C = sim.make_cameras(N=14, depth=8.0, cone_deg=cone)
    rows = []
    for name, kw, ctr in PHANTOMS:
        r_true = sim.boundary(THETA, **kw); A_true = sim.albedo_field(THETA, "const", A0=1.0)
        B, vis = sim.forward(C, r_true, A_true, THETA, center=ctr)
        # true CSA/DCE: area enclosed by the (possibly off-center) boundary
        px = ctr[0] + r_true * np.cos(THETA); py = ctr[1] + r_true * np.sin(THETA)
        csa_true = 0.5 * abs(np.sum(px * np.roll(py, -1) - py * np.roll(px, -1)))
        dce_true = 2 * np.sqrt(csa_true / np.pi)
        sol = sim.fit(B, vis, THETA, C, nfr_geom=0, nfr_alb=0)     # CIRCULAR SoR estimator
        r0, gk, A0, ak, cx, cy = sim.unpack(sol.x, 0, 0)
        csa_fit = np.pi * r0 ** 2; dce_fit = 2 * r0
        J = sim.jacobian(sol.x, THETA, C, 0, 0, False, vis); rank, condn, _ = sim.rank_cond(J)
        rows.append(dict(phantom=name, csa_true=round(csa_true, 3), csa_fit=round(csa_fit, 3),
                         CSA_bias_pct=round(100 * (csa_fit - csa_true) / csa_true, 1),
                         DCE_bias_pct=round(100 * (dce_fit - dce_true) / dce_true, 1),
                         J_rank=rank, J_cond=round(condn, 1), resid_rms=float(np.sqrt(np.mean(sol.fun ** 2)))))
    return rows


def main():
    out = {c: run(v) for c, v in CONES.items()}
    real = out["realistic_2_5deg"]
    worst = max(abs(r["CSA_bias_pct"]) for r in real if r["phantom"] != "ideal_circle")
    verdict = dict(pass_criterion="CSA bias < 10% across realistic deviations",
                   worst_CSA_bias_pct_realistic=worst, PASS=bool(worst < 10.0))
    (open("test1_model_mismatch.json", "w")).write(json.dumps({"verdict": verdict, **out}, indent=2))
    # figure
    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = [r["phantom"] for r in real]; bias = [r["CSA_bias_pct"] for r in real]
    cols = ["seagreen" if abs(b) < 10 else "crimson" for b in bias]
    ax.bar(range(len(names)), bias, color=cols); ax.axhline(10, c="k", ls="--"); ax.axhline(-10, c="k", ls="--")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CSA bias % (circular SoR fit)"); ax.set_title(f"TEST 1 model mismatch @ 2.5° parallax — PASS={verdict['PASS']} (worst |bias|={worst:.1f}%)")
    fig.tight_layout(); fig.savefig("test1_model_mismatch.png", dpi=120)
    print("=== TEST 1 (realistic 2.5deg) ===")
    print(f"{'phantom':18}{'CSA_bias%':>10}{'DCE_bias%':>10}{'J_rank':>8}{'J_cond':>9}")
    for r in real: print(f"{r['phantom']:18}{r['CSA_bias_pct']:>10}{r['DCE_bias_pct']:>10}{r['J_rank']:>8}{r['J_cond']:>9}")
    print(f"\nVERDICT: worst |CSA bias| (realistic) = {worst:.1f}%  -> PASS={verdict['PASS']} (criterion <10%)")


if __name__ == "__main__":
    main()
