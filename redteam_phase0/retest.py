"""RE-TEST the CORRECT estimator (rebuild_sim, gate-verified) on the CSA RATIO under CORRELATED
relative pose noise at the DATA-DRIVEN level from bootstrap_pose_noise.json. Ablation: contour term
ON (E_bnd primary) vs OFF (shading-only). Correlated (smooth trajectory) perturbations, not i.i.d.
depth-eval env."""
from __future__ import annotations
import json
import numpy as np, rebuild_sim as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PHI = np.linspace(0, 2 * np.pi, 72, endpoint=False)
A0 = np.array([0.0, 0.0, 1.0])
O_REF, R_REF = np.array([0., 0., 0.]), 5.0
O_NAR, R_NAR = np.array([0., 0., 0.]), 3.0
RATIO_TRUE = (R_NAR / R_REF) ** 2                     # 0.36
boot = json.loads(open("bootstrap_pose_noise.json").read())
SIG_T = boot["translation_std_frac_of_depth"]["p90"]  # 0.0006 (fraction of depth)
SIG_ROT = boot["rotation_std_deg"]["p90"]             # 0.068 deg
DEPTH = 8.0


def rotm(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def smooth_drift(N, sigma, seed, ndim):
    """correlated (low-frequency) per-camera drift, normalized to std=sigma per dim."""
    rng = np.random.default_rng(seed); t = np.linspace(0, 1, N)
    out = np.zeros((N, ndim))
    for d in range(ndim):
        c = rng.normal(0, 1, 3)
        s = c[0] * np.sin(np.pi * t) + c[1] * np.sin(2 * np.pi * t) + c[2] * (t - 0.5)
        s = s / (s.std() + 1e-9) * sigma; out[:, d] = s
    return out


def perturb_traj(cams, sig_t_frac, sig_rot_deg, seed):
    N = len(cams); dT = smooth_drift(N, sig_t_frac * DEPTH, seed, 3)
    dR = smooth_drift(N, np.radians(sig_rot_deg), seed + 1, 3)
    out = []
    for i, (R, C) in enumerate(cams):
        ax = dR[i]; ang = np.linalg.norm(ax); Rp = rotm(ax, ang) @ R if ang > 1e-12 else R
        out.append((Rp, C + dT[i]))
    return out


def fit_ring(O, r, cams_true, cams_est, use_contour):
    obs = S.observe(O, A0, r, cams_true, S.F0, 1.0, PHI)   # true observations
    th0 = S.theta_from(O, A0, r * 1.05, 1.0)
    fit = S.fit(obs, cams_est, S.F0, PHI, th0, use_contour=use_contour)
    return fit["CSA"]


def mc(sig_t_frac, sig_rot_deg, use_contour, n=60):
    cams_ref = S.cameras(N=14, depth=DEPTH, cone_deg=2.5, seed=1, O=O_REF)
    cams_nar = S.cameras(N=14, depth=DEPTH, cone_deg=2.5, seed=2, O=O_NAR)
    csr, csn, rat = [], [], []
    for k in range(n):
        cr = perturb_traj(cams_ref, sig_t_frac, sig_rot_deg, 100 + k)
        cn = perturb_traj(cams_nar, sig_t_frac, sig_rot_deg, 500 + k)
        try:
            a = fit_ring(O_REF, R_REF, cams_ref, cr, use_contour)
            b = fit_ring(O_NAR, R_NAR, cams_nar, cn, use_contour)
        except Exception:
            continue
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0): continue
        csr.append(a); csn.append(b); rat.append(b / a)
    csr, csn, rat = map(np.array, (csr, csn, rat))
    return dict(ref_CSA_bias=round(100 * (csr.mean() - np.pi * 25) / (np.pi * 25), 2),
                narrow_CSA_bias=round(100 * (csn.mean() - np.pi * 9) / (np.pi * 9), 2),
                RATIO_bias=round(100 * (rat.mean() - RATIO_TRUE) / RATIO_TRUE, 2),
                RATIO_std=round(100 * rat.std() / RATIO_TRUE, 2), n=len(rat))


def main():
    # at the actual data-driven level, with vs without contour
    data_c = mc(SIG_T, SIG_ROT, True); data_s = mc(SIG_T, SIG_ROT, False)
    # sweep relative translation noise (multiples of the bootstrap level up to 1%)
    mults = [0, 1, 3, 8, 16, 33]                          # x SIG_T ; 33x ~= 1% (old assumption)
    sweep_c = [mc(m * SIG_T, m * SIG_ROT, True) for m in mults]
    sweep_s = [mc(m * SIG_T, m * SIG_ROT, False) for m in mults]
    out = dict(data_driven_noise=dict(sigma_t_frac_depth=SIG_T, sigma_rot_deg=SIG_ROT),
               at_data_level=dict(with_contour=data_c, shading_only=data_s),
               sweep_translation_pct=[round(m * SIG_T * 100, 3) for m in mults],
               sweep_ratio_bias_with_contour=[s["RATIO_bias"] for s in sweep_c],
               sweep_ratio_bias_shading_only=[s["RATIO_bias"] for s in sweep_s])
    out["ratio_bias_below_5pct_at_data_level_with_contour"] = bool(abs(data_c["RATIO_bias"]) < 5.0)
    open("retest.json", "w").write(json.dumps(out, indent=2))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xt = [m * SIG_T * 100 for m in mults]
    ax.plot(xt, out["sweep_ratio_bias_with_contour"], "o-", color="seagreen", lw=2, label="RATIO bias — WITH contour (designed)")
    ax.plot(xt, out["sweep_ratio_bias_shading_only"], "s--", color="crimson", label="RATIO bias — shading-only (ablation)")
    ax.axvline(SIG_T * 100, color="tab:blue", ls=":", label=f"data-driven noise ({SIG_T*100:.2f}%)")
    ax.axvline(1.0, color="0.5", ls=":", label="old assumed 1%")
    ax.axhline(5, c="k", ls="--"); ax.axhline(-5, c="k", ls="--"); ax.axhline(0, c="0.8", lw=.7)
    ax.set_xlabel("relative translation noise (% depth, correlated)"); ax.set_ylabel("CSA RATIO bias %")
    ax.set_title(f"RE-TEST — CSA ratio bias vs relative pose noise (contour ablation)\n"
                 f"@data level: with contour {data_c['RATIO_bias']}% | shading-only {data_s['RATIO_bias']}%"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("retest.png", dpi=120)
    print("=== RE-TEST (correct estimator, gate-verified) ===")
    print(f"data-driven relative noise: translation {SIG_T*100:.3f}% depth, rotation {SIG_ROT}deg")
    print(f"@data level  WITH contour : ref bias {data_c['ref_CSA_bias']}%  narrow {data_c['narrow_CSA_bias']}%  -> RATIO {data_c['RATIO_bias']}% (±{data_c['RATIO_std']}%)")
    print(f"@data level  shading-only : ref bias {data_s['ref_CSA_bias']}%  narrow {data_s['narrow_CSA_bias']}%  -> RATIO {data_s['RATIO_bias']}% (±{data_s['RATIO_std']}%)")
    print(f"sweep transl %: {out['sweep_translation_pct']}")
    print(f"  RATIO bias WITH contour : {out['sweep_ratio_bias_with_contour']}")
    print(f"  RATIO bias shading-only : {out['sweep_ratio_bias_shading_only']}")
    print(f"\nRATIO bias < 5% at data level WITH contour: {out['ratio_bias_below_5pct_at_data_level_with_contour']}")


if __name__ == "__main__":
    main()
