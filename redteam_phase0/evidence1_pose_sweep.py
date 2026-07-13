"""EVIDENCE 1 -- CSA bias vs pose noise sweep (not a single point).
Translation 0 -> 3% of depth, rotation 0 -> 1 deg, focal fixed. Monte-Carlo per level -> mean bias
+/- std. Also a 2-D (translation x rotation) bias grid. Reuses the vectorized ray model in
test3_pose.py. depth-eval env."""
from __future__ import annotations
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import test3_pose as T

DEPTH = 6.0; CSA_TRUE = np.pi * T.R_TRUE ** 2
cams = T.cameras(); obs = T.render(cams, T.R_TRUE, T.F0)


def mc_bias(sig_t_frac, sig_rot_deg, n=50, seed0=0):
    csa = []
    for k in range(n):
        cp = T.perturb(cams, sig_t_frac * DEPTH, sig_rot_deg, seed=seed0 + k)
        r = T.fit_r(obs, cp, T.F0); csa.append(np.pi * r ** 2)
    csa = np.array(csa)
    return 100 * (csa.mean() - CSA_TRUE) / CSA_TRUE, 100 * csa.std() / CSA_TRUE


def main():
    tvals = np.linspace(0, 0.03, 11)                 # 0 -> 3% translation
    rvals = np.linspace(0, 1.0, 11)                  # 0 -> 1 deg rotation
    t_bias = np.array([mc_bias(t, 0.0) for t in tvals])   # (bias, std)
    r_bias = np.array([mc_bias(0.0, rr) for rr in rvals])
    # 2D grid (coarser, MC=35)
    tg = np.linspace(0, 0.03, 7); rg = np.linspace(0, 1.0, 7)
    grid = np.array([[mc_bias(t, rr, n=35)[0] for rr in rg] for t in tg])
    out = dict(translation_pct=tvals.tolist(), translation_bias=t_bias[:, 0].tolist(), translation_std=t_bias[:, 1].tolist(),
               rotation_deg=rvals.tolist(), rotation_bias=r_bias[:, 0].tolist(), rotation_std=r_bias[:, 1].tolist())
    open("evidence1_pose_sweep.json", "w").write(json.dumps(out, indent=2))
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.errorbar(tvals * 100, t_bias[:, 0], yerr=t_bias[:, 1], fmt="o-", color="tab:red", capsize=3)
    ax.axhline(10, c="k", ls="--"); ax.axhline(-10, c="k", ls="--"); ax.axhline(0, c="0.7", lw=.8)
    ax.set_xlabel("translation noise (% of depth)"); ax.set_ylabel("CSA bias % (mean ± std, MC=50)")
    ax.set_title("CSA bias vs TRANSLATION noise"); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 2)
    ax.errorbar(rvals, r_bias[:, 0], yerr=r_bias[:, 1], fmt="o-", color="tab:blue", capsize=3)
    ax.axhline(10, c="k", ls="--"); ax.axhline(-10, c="k", ls="--"); ax.axhline(0, c="0.7", lw=.8)
    ax.set_xlabel("rotation noise (deg)"); ax.set_ylabel("CSA bias %")
    ax.set_title("CSA bias vs ROTATION noise"); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
                   extent=[rg[0], rg[-1], tg[0] * 100, tg[-1] * 100])
    ax.set_xlabel("rotation (deg)"); ax.set_ylabel("translation (% depth)"); ax.set_title("CSA bias % (2-D)")
    cs = ax.contour(rg, tg * 100, grid, levels=[-10, 10], colors="k"); ax.clabel(cs, fmt="%d%%")
    fig.colorbar(im, ax=ax, label="CSA bias %")
    fig.suptitle("EVIDENCE 1 — CSA bias vs pose noise (near-light photometric SoR fit; simulation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("evidence1_pose_sweep.png", dpi=120)
    print("translation %:", [round(x, 2) for x in tvals * 100])
    print("  CSA bias %:", [round(b, 1) for b in t_bias[:, 0]])
    print("rotation deg:", [round(x, 2) for x in rvals])
    print("  CSA bias %:", [round(b, 1) for b in r_bias[:, 0]])
    # where does |bias| cross 10%?
    tt = tvals[np.argmax(np.abs(t_bias[:, 0]) > 10)] * 100 if (np.abs(t_bias[:, 0]) > 10).any() else None
    rr = rvals[np.argmax(np.abs(r_bias[:, 0]) > 10)] if (np.abs(r_bias[:, 0]) > 10).any() else None
    print(f"|bias|>10% crossed at: translation ~{tt}% depth, rotation ~{rr} deg")


if __name__ == "__main__":
    main()
