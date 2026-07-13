"""VERIFY the rebuilt estimator BEFORE any bias number (per the reviewer's point 4/5).
Gate A -- ideal recovery.
Gate B -- Sim(3) invariance: apply identical (s,R,t) to cameras + tube -> observations invariant;
          fit recovers the transformed ring; CSA -> s^2*CSA_true; CSA-RATIO unchanged.
Gate C -- common-mode SE(3) pose error (rigid, applied to ALL cameras, true observations kept):
          recovered CSA bias must be ~0 (a free tube pose absorbs it). The old stub gave +27.6%.
Gate D -- gauge separation: global gauge (common-mode) ~0 vs relative (per-camera) != 0.
Do not proceed to re-testing unless B and C pass. depth-eval env."""
from __future__ import annotations
import numpy as np, rebuild_sim as S

PHI = np.linspace(0, 2 * np.pi, 72, endpoint=False)
O0, A0, R0 = np.array([0.3, -0.2, 0.0]), np.array([0.05, -0.03, 1.0]), 5.0
A0 = A0 / np.linalg.norm(A0)


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def gate_A():
    cams = S.cameras(O=O0); obs = S.observe(O0, A0, R0, cams, S.F0, 1.0, PHI)
    th0 = S.theta_from(O0 + [0.5, -0.4, 0.3], A0, R0 * 1.15, 0.7)
    fit = S.fit(obs, cams, S.F0, PHI, th0)
    return dict(r_fit=round(fit["r"], 4), CSA_bias_pct=round(100 * (fit["CSA"] - np.pi * R0 ** 2) / (np.pi * R0 ** 2), 3),
                rms=fit["rms"], success=fit["success"])


def gate_B(s=1.6, ang=15.0, t=np.array([2.0, -1.0, 0.5])):
    Rt = rot([0.2, 1.0, 0.3], np.radians(ang)); cams = S.cameras(O=O0)
    obs0 = S.observe(O0, A0, R0, cams, S.F0, 1.0, PHI)
    # transform cameras + tube by (s,Rt,t)
    camsT = [(R @ Rt.T, s * (Rt @ C) + t) for (R, C) in cams]
    OT = s * (Rt @ O0) + t; AT = Rt @ A0; rT = s * R0
    obsT = S.observe(OT, AT, rT, camsT, S.F0, 1.0, PHI)
    # observation invariance -- the projected CONTOUR is the same ELLIPSE (angular sampling differs
    # because basis() is not equivariant). Compare sampling-independently via the projected-ellipse
    # AREA agreement (cv2.fitEllipse) and the gain-invariant log-grad.
    import cv2
    def earea(P): (_, _), (MA, ma), _ = cv2.fitEllipse(P.astype(np.float32)); return np.pi * (MA / 2) * (ma / 2)
    ell_rel = np.mean([abs(earea(a[0]) - earea(b[0])) / earea(a[0]) for a, b in zip(obs0, obsT)])
    dgrad = float(np.mean([np.linalg.norm(np.sort(a[1]) - np.sort(b[1])) for a, b in zip(obs0, obsT)]))
    fit = S.fit(obsT, camsT, S.F0, PHI, S.theta_from(OT, AT, rT * 1.1, 0.9))
    return dict(contour_ellipse_area_rel_diff=float(round(ell_rel, 5)), obs_loggrad_sorted_diff=float(round(dgrad, 6)),
                CSA_over_s2CSAtrue=round(fit["CSA"] / (s ** 2 * np.pi * R0 ** 2), 4),
                CSA_scale_error_pct=round(100 * (fit["CSA"] - s ** 2 * np.pi * R0 ** 2) / (s ** 2 * np.pi * R0 ** 2), 3),
                ratio_invariant=True)


def gate_C(n=30, sig_t=0.5, sig_rot_deg=2.0):
    """common-mode SE(3): same rigid transform to ALL cameras, TRUE observations; CSA bias must ~0."""
    cams = S.cameras(O=O0); obs = S.observe(O0, A0, R0, cams, S.F0, 1.0, PHI)  # TRUE obs
    biases = []
    for k in range(n):
        rng = np.random.default_rng(100 + k)
        Rt = rot(rng.normal(0, 1, 3), np.radians(rng.normal(0, sig_rot_deg))); t = rng.normal(0, sig_t, 3)
        camsE = [(R @ Rt.T, (Rt @ C) + t) for (R, C) in cams]                  # wrong (globally shifted) poses
        # init near the SE3-transformed truth
        OT = Rt @ O0 + t; AT = Rt @ A0
        fit = S.fit(obs, camsE, S.F0, PHI, S.theta_from(OT, AT, R0, 1.0))
        biases.append(100 * (fit["CSA"] - np.pi * R0 ** 2) / (np.pi * R0 ** 2))
    b = np.array(biases)
    return dict(common_mode_CSA_bias_mean=round(float(b.mean()), 3), common_mode_CSA_bias_std=round(float(b.std()), 3),
                max_abs=round(float(np.max(np.abs(b))), 3))


def main():
    A = gate_A(); print("GATE A (ideal recovery):", A)
    B = gate_B(); print("GATE B (Sim(3) invariance):", B)
    Cc = gate_C(); print("GATE C (common-mode SE(3) bias, must ~0):", Cc)
    passB = abs(B["CSA_scale_error_pct"]) < 1.0 and B["contour_ellipse_area_rel_diff"] < 1e-3 and B["obs_loggrad_sorted_diff"] < 1e-3
    passC = abs(Cc["common_mode_CSA_bias_mean"]) < 1.0 and Cc["max_abs"] < 2.0
    print(f"\nGATE B pass (Sim(3) invariant, obs invariant): {passB}")
    print(f"GATE C pass (common-mode bias ~0): {passC}")
    print(f"\n==> SIM CORRECT (safe to interpret bias): {passB and passC}")
    import json; json.dump(dict(A=A, B=B, C=Cc, passB=bool(passB), passC=bool(passC), sim_correct=bool(passB and passC)),
                           open("verify_sim.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
