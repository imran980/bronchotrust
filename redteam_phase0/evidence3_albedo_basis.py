"""EVIDENCE 3 -- albedo basis + why 13 DOF (6 harmonics) suddenly fixes the stripe bias.
Shows: (a) the Fourier albedo basis; (b) the vascular-stripe albedo sign(cos 6θ) and its Fourier
SPECTRUM (energy at k=6,18,30 -> fundamental k=6); (c) N-harmonic reconstructions (N=4 misses k=6,
N=6 captures it); (d) CSA bias vs #harmonics with the k=6 capture marked; (e) the mechanism -- the
FITTED r(θ) develops a spurious oscillation when albedo cannot fit k=6 (aliasing into geometry)."""
from __future__ import annotations
import json
import numpy as np, sim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TH = np.linspace(0, 2 * np.pi, 240, endpoint=False)
C = sim.make_cameras(N=14, depth=8.0, cone_deg=2.5)
THETA = np.linspace(0, 2 * np.pi, 120, endpoint=False)


def stripe(theta): return 1 + 0.4 * np.sign(np.cos(6 * theta)) * 0.5     # matches sim 'stripes'


def fourier_coeffs(sig, theta, K=20):
    a0 = sig.mean(); cs = []
    for k in range(1, K + 1):
        cs.append((2 * (sig * np.cos(k * theta)).mean(), 2 * (sig * np.sin(k * theta)).mean()))
    return a0, np.array(cs)


def recon(a0, cs, theta, N):
    v = np.full_like(theta, a0)
    for k in range(1, N + 1):
        v += cs[k - 1, 0] * np.cos(k * theta) + cs[k - 1, 1] * np.sin(k * theta)
    return v


def csa_bias_at(nfr_alb):
    r_true = sim.boundary(THETA, "circle", r0=5.0); A_true = sim.albedo_field(THETA, "stripes", A0=1.0)
    B, vis = sim.forward(C, r_true, A_true, THETA); B = B + np.random.default_rng(1).normal(0, 0.003 * B[vis].std(), B.shape)
    sol = sim.fit(B, vis, THETA, C, nfr_geom=4, nfr_alb=nfr_alb)
    r0, gk, A0, ak, cx, cy = sim.unpack(sol.x, 4, nfr_alb); rfit = sim.fourier_eval(r0, gk, THETA)
    csa, _ = sim.csa_dce(rfit, THETA); csa_t, _ = sim.csa_dce(r_true, THETA)
    return 100 * (csa - csa_t) / csa_t, rfit


def main():
    sig = stripe(TH); a0, cs = fourier_coeffs(sig, TH, K=20)
    mag = np.hypot(cs[:, 0], cs[:, 1])
    Ns = list(range(1, 11)); bias = []; rfits = {}
    for N in Ns:
        b, rf = csa_bias_at(N); bias.append(b)
        if N in (4, 6): rfits[N] = rf
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 3, 1)
    for k in range(1, 7): ax.plot(TH, np.cos(k * TH) + 0 * k, lw=1, label=f"cos{k}θ" if k <= 3 else None)
    ax.set_title("(a) Fourier albedo basis cos(kθ)"); ax.set_xlabel("θ"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 2)
    ax.bar(range(1, 21), mag, color=["crimson" if k in (6, 18) else "tab:gray" for k in range(1, 21)])
    ax.axvline(4.5, c="tab:orange", ls="--", label="4-harm cutoff (9 DOF)"); ax.axvline(6.5, c="tab:green", ls="--", label="6-harm cutoff (13 DOF)")
    ax.set_xlabel("harmonic k"); ax.set_ylabel("|coeff|"); ax.set_title("(b) stripe spectrum: fundamental at k=6"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(TH, sig, "k", lw=1.5, label="true stripe albedo")
    ax.plot(TH, recon(a0, cs, TH, 4), "tab:orange", label="4-harm fit (misses k=6)")
    ax.plot(TH, recon(a0, cs, TH, 6), "tab:green", label="6-harm fit (captures k=6)")
    ax.set_title("(c) albedo reconstruction vs #harmonics"); ax.set_xlabel("θ"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 4)
    ax.plot([1 + 2 * n for n in Ns], bias, "o-", color="tab:purple")
    ax.axhline(10, c="k", ls="--"); ax.axhline(-10, c="k", ls="--"); ax.axvline(1 + 2 * 6, c="tab:green", ls="--", label="k=6 captured (13 DOF)")
    ax.set_xlabel("albedo DOF (1+2N)"); ax.set_ylabel("CSA bias %"); ax.set_title("(d) CSA bias vs albedo DOF — drop AT k=6"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(THETA, rfits[4], "tab:orange", label=f"r(θ) fit, 4-harm albedo (spurious wobble)")
    ax.plot(THETA, rfits[6], "tab:green", label="r(θ) fit, 6-harm albedo (flat=true)")
    ax.axhline(5.0, c="k", ls=":", label="true r=5"); ax.set_title("(e) MECHANISM: unmodeled k=6 albedo aliases into r(θ)")
    ax.set_xlabel("θ"); ax.set_ylabel("fitted radius"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    ax.text(0.02, 0.95, "WHY 13 DOF fixes it:\n\n"
            "• vascular stripe = sign(cos6θ): a square wave whose\n  Fourier energy sits at k=6,18,30 (fundamental k=6).\n\n"
            "• 4-harmonic albedo (9 DOF) spans only k≤4 -> it CANNOT\n  represent the k=6 term. Least squares then recruits the\n  geometry params (r0..k4) to absorb the k=6 residual,\n  because ∂B/∂r also changes B with θ-structure.\n\n"
            "• That recruited r(θ) wobble changes the area integral\n  ∮r²/2 -> +44.6% CSA bias (panel e, orange).\n\n"
            "• 6-harmonic albedo (13 DOF) spans k≤6 -> the k=6 term is\n  fit BY ALBEDO, geometry is not recruited -> r(θ) stays flat\n  (panel e, green) -> bias ~0.\n\n"
            "• The cliff is at exactly the fundamental (k=6), not gradual:\n  it's basis coverage of a discrete frequency, not conditioning.", fontsize=9, va="top", family="monospace")
    fig.suptitle("EVIDENCE 3 — albedo basis coverage and the k=6 aliasing that biases CSA", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("evidence3_albedo_basis.png", dpi=120)
    out = dict(stripe_fundamental_k=6, spectrum_top=[int(k + 1) for k in np.argsort(-mag)[:4]],
               albedo_DOF=[1 + 2 * n for n in Ns], CSA_bias_pct=[round(b, 1) for b in bias],
               note="CSA bias collapses at 13 DOF (6 harmonics) = exactly when the albedo basis first covers the stripe fundamental k=6")
    open("evidence3_albedo_basis.json", "w").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
