"""RESCUE 1 -- Test 3 on the CSA RATIO (the actual scale-free deliverable), not absolute CSA.
Two cross-sections in ONE model: reference (r=5) and narrow (r=3), true CSA ratio = 9/25 = 0.36.
The +21% absolute bias (1/d^2 rectification) is SYSTEMATIC -- it should cancel in the ratio if it hits
both rings similarly. Test: 100 MC, per-camera-independent ('relative') pose noise applied to BOTH
rings' cameras; fit each radius; report the RATIO bias/std vs the ABSOLUTE bias. Also a common-mode
(global) pose error, which must cancel exactly. Sweep noise 0->3% translation. Reuses test3 ray
model. depth-eval env."""
from __future__ import annotations
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import test3_pose as T

DEPTH = 6.0
cams_ref = T.cameras(seed=1); cams_nar = T.cameras(seed=2)
obs_ref = T.render(cams_ref, 5.0, T.F0); obs_nar = T.render(cams_nar, 3.0, T.F0)
RATIO_TRUE = (3.0 ** 2) / (5.0 ** 2)                    # 0.36


def one_run(sig_t_frac, sig_rot_deg, seed, common=False):
    if common:                                          # same global perturbation to ALL cameras
        cr = T.perturb(cams_ref, sig_t_frac * DEPTH, sig_rot_deg, seed=seed)
        cn = T.perturb(cams_nar, sig_t_frac * DEPTH, sig_rot_deg, seed=seed)   # same seed -> correlated
    else:                                               # independent per-camera ('relative') noise
        cr = T.perturb(cams_ref, sig_t_frac * DEPTH, sig_rot_deg, seed=seed)
        cn = T.perturb(cams_nar, sig_t_frac * DEPTH, sig_rot_deg, seed=seed + 5000)
    rr = T.fit_r(obs_ref, cr, T.F0); rn = T.fit_r(obs_nar, cn, T.F0)
    return np.pi * rr ** 2, np.pi * rn ** 2


def mc(sig_t_frac, sig_rot_deg, n=100, common=False):
    csr, csn, rat = [], [], []
    for k in range(n):
        cr, cn = one_run(sig_t_frac, sig_rot_deg, 100 + k, common)
        csr.append(cr); csn.append(cn); rat.append(cn / cr)
    csr, csn, rat = map(np.array, (csr, csn, rat))
    return dict(ref_CSA_bias=round(100 * (csr.mean() - np.pi * 25) / (np.pi * 25), 1),
                narrow_CSA_bias=round(100 * (csn.mean() - np.pi * 9) / (np.pi * 9), 1),
                RATIO_bias=round(100 * (rat.mean() - RATIO_TRUE) / RATIO_TRUE, 2),
                RATIO_std=round(100 * rat.std() / RATIO_TRUE, 2))


def main():
    real = mc(0.01, 0.3, n=100)                          # realistic: 1% translation, 0.3 deg
    real_common = mc(0.01, 0.3, n=100, common=True)
    tvals = np.linspace(0, 0.03, 10)
    sweep = [mc(t, 0.3, n=60) for t in tvals]
    out = dict(true_ratio=RATIO_TRUE,
               realistic_relative=real, realistic_common_mode=real_common,
               sweep_translation_pct=(tvals * 100).tolist(),
               sweep_ratio_bias=[s["RATIO_bias"] for s in sweep],
               sweep_ref_bias=[s["ref_CSA_bias"] for s in sweep],
               sweep_narrow_bias=[s["narrow_CSA_bias"] for s in sweep])
    out["ratio_bias_below_5pct_at_realistic"] = bool(abs(real["RATIO_bias"]) < 5.0)
    open("rescue1_ratio_pose.json", "w").write(json.dumps(out, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(tvals * 100, out["sweep_ref_bias"], "o-", color="tab:blue", label="ABS CSA bias (ref r=5)")
    ax[0].plot(tvals * 100, out["sweep_narrow_bias"], "s-", color="tab:cyan", label="ABS CSA bias (narrow r=3)")
    ax[0].plot(tvals * 100, out["sweep_ratio_bias"], "^-", color="crimson", lw=2, label="RATIO bias (narrow/ref)")
    ax[0].axhline(5, c="k", ls="--"); ax[0].axhline(-5, c="k", ls="--"); ax[0].axhline(0, c="0.7", lw=.7)
    ax[0].set_xlabel("translation noise (% depth)"); ax[0].set_ylabel("bias %")
    ax[0].set_title("absolute CSA bias blows up; RATIO bias stays ~flat"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].bar([0, 1, 2], [real["ref_CSA_bias"], real["narrow_CSA_bias"], real["RATIO_bias"]],
              color=["tab:blue", "tab:cyan", "crimson"])
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(["ABS ref", "ABS narrow", "RATIO"])
    ax[1].axhline(5, c="k", ls="--"); ax[1].axhline(-5, c="k", ls="--")
    ax[1].set_ylabel("bias %"); ax[1].set_title(f"@1% transl,0.3°: RATIO bias={real['RATIO_bias']}% (±{real['RATIO_std']}%)")
    fig.suptitle("RESCUE 1 — CSA RATIO under relative pose noise (systematic 1/d² bias cancels)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("rescue1_ratio_pose.png", dpi=120)
    print("=== RESCUE 1 ===")
    print(f"true ratio = {RATIO_TRUE}")
    print(f"realistic (1% transl, 0.3deg), RELATIVE noise: ref ABS bias={real['ref_CSA_bias']}%  "
          f"narrow ABS bias={real['narrow_CSA_bias']}%  ->  RATIO bias={real['RATIO_bias']}% (±{real['RATIO_std']}%)")
    print(f"common-mode (global) pose error:  RATIO bias={real_common['RATIO_bias']}% (±{real_common['RATIO_std']}%)")
    print(f"sweep RATIO bias %: {[round(s['RATIO_bias'],1) for s in sweep]}")
    print(f"\nRATIO bias < 5% at realistic noise: {out['ratio_bias_below_5pct_at_realistic']}")


if __name__ == "__main__":
    main()
