"""TEST 2 -- spatially varying albedo. Phantom = circular geometry with a non-constant albedo field;
the estimator makes albedo FREE (Fourier, >=5-10 DOF) and estimates geometry. Report Jacobian rank,
null space, CSA bias, and whether geometry & albedo become coupled (the shape-from-shading ambiguity).
Two estimator variants: (a) SoR-strict circular geometry r0 + albedo(theta); (b) non-circular r(theta)
+ albedo(theta). Primary regime = realistic 2.5 deg parallax (our 2_V2 data). depth-eval env."""
from __future__ import annotations
import json
import numpy as np, sim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

THETA = np.linspace(0, 2 * np.pi, 120, endpoint=False)
ALBEDOS = ["const", "gradient", "stripes", "lowfreq", "highfreq"]
CONES = {"realistic_2_5deg": 2.5, "ideal_15deg": 15.0}
NFR_ALB = 4                        # albedo DOF = 1 + 2*4 = 9


def coupling(J, p_geom_idx, p_alb_idx, tol_ratio=1e-3):
    """rank deficiency + how much the null space mixes geometry & albedo blocks."""
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    smax = s[0] if len(s) else 0.0
    rank = int((s > tol_ratio * smax).sum())
    ncol = Vt.shape[0]; null = Vt[rank:]                     # null-space directions (rows of Vt)
    mixes = []
    for v in null:
        g = np.linalg.norm(v[p_geom_idx]); a = np.linalg.norm(v[p_alb_idx])
        if g + a > 1e-9: mixes.append(min(g, a) / (g + a))   # ~0.5 => strongly mixes geom & albedo
    cond = float(smax / s[s > 1e-12][-1]) if (s > 1e-12).any() else np.inf
    return rank, ncol, cond, (float(np.max(mixes)) if mixes else 0.0), len(null)


def run(cone, geom_mode):
    C = sim.make_cameras(N=14, depth=8.0, cone_deg=cone)
    nfr_geom = 4 if geom_mode == "noncirc" else 0
    rows = []
    for alb in ALBEDOS:
        r_true = sim.boundary(THETA, "circle", r0=5.0)
        A_true = sim.albedo_field(THETA, alb, A0=1.0)
        B, vis = sim.forward(C, r_true, A_true, THETA)
        B = B + np.random.default_rng(1).normal(0, 0.003 * B[vis].std(), B.shape)
        sol = sim.fit(B, vis, THETA, C, nfr_geom=nfr_geom, nfr_alb=NFR_ALB)
        r0, gk, A0, ak, cx, cy = sim.unpack(sol.x, nfr_geom, NFR_ALB)
        r_fit = sim.fourier_eval(r0, gk, THETA); csa_fit, _ = sim.csa_dce(r_fit, THETA)
        csa_true, _ = sim.csa_dce(r_true, THETA)
        # param index blocks: geom = [0 .. 2*nfr_geom], albedo = [after A0 .. ]
        ngeom = 1 + 2 * nfr_geom; p_geom = list(range(0, ngeom)); p_alb = list(range(ngeom + 1, ngeom + 1 + 2 * NFR_ALB))
        J = sim.jacobian(sol.x, THETA, C, nfr_geom, NFR_ALB, False, vis)
        rank, ncol, condn, maxmix, nnull = coupling(J, p_geom, p_alb)
        rows.append(dict(albedo=alb, geom_mode=geom_mode, CSA_bias_pct=round(100 * (csa_fit - csa_true) / csa_true, 1),
                         J_rank=rank, n_params=ncol, J_cond=round(condn, 1), n_nullspace=nnull,
                         max_geom_albedo_mix=round(maxmix, 3), coupled=bool(maxmix > 0.25 or rank < ncol)))
    return rows


def main():
    out = {}
    for cname, cone in CONES.items():
        for gm in ("circle", "noncirc"):
            out[f"{cname}__{gm}"] = run(cone, gm)
    real = out["realistic_2_5deg__noncirc"]                 # the fair coupling test (geometry free too)
    worst_bias = max(abs(r["CSA_bias_pct"]) for r in real if r["albedo"] != "const")
    any_coupled = any(r["coupled"] for r in real if r["albedo"] != "const")
    verdict = dict(worst_CSA_bias_pct=worst_bias, any_geometry_albedo_coupling=any_coupled,
                   PASS=bool(worst_bias < 10.0 and not any_coupled))
    open("test2_albedo.json", "w").write(json.dumps({"verdict": verdict, **out}, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, gm in enumerate(("circle", "noncirc")):
        r = out[f"realistic_2_5deg__{gm}"]; names = [x["albedo"] for x in r]; bias = [x["CSA_bias_pct"] for x in r]
        cols = ["crimson" if x["coupled"] else "seagreen" for x in r]
        ax[k].bar(range(len(names)), bias, color=cols); ax[k].axhline(10, c="k", ls="--"); ax[k].axhline(-10, c="k", ls="--")
        ax[k].set_xticks(range(len(names))); ax[k].set_xticklabels(names, rotation=25, fontsize=8)
        ax[k].set_ylabel("CSA bias %"); ax[k].set_title(f"geometry={gm} @2.5° (red=geom/albedo coupled)")
    fig.suptitle(f"TEST 2 spatially-varying albedo @2.5° — PASS={verdict['PASS']}", fontsize=12)
    fig.tight_layout(); fig.savefig("test2_albedo.png", dpi=120)
    print("=== TEST 2 (realistic 2.5deg, geometry free r(theta) + albedo free) ===")
    print(f"{'albedo':10}{'CSA_bias%':>10}{'J_rank':>8}{'/params':>8}{'cond':>10}{'geom~alb_mix':>13}{'coupled':>9}")
    for x in real: print(f"{x['albedo']:10}{x['CSA_bias_pct']:>10}{x['J_rank']:>8}{x['n_params']:>8}{x['J_cond']:>10}{x['max_geom_albedo_mix']:>13}{str(x['coupled']):>9}")
    print(f"\nVERDICT: worst |CSA bias|={worst_bias:.1f}%, coupling={any_coupled} -> PASS={verdict['PASS']}")


if __name__ == "__main__":
    main()
