"""TEST 3 -- pose uncertainty (Monte-Carlo). RAY-BASED model so translation, rotation AND focal all
matter: a cylinder wall (radius r) imaged by cameras with known poses; each pixel back-projects to a
ray (via pose+focal), hits the wall -> near-light Lambertian B. Data is rendered with TRUE poses; the
estimator fits r using PERTURBED poses (as COLMAP would give). 100 MC runs per noise setting.
Report CSA variance, CSA bias, and a sensitivity ranking (translation vs rotation vs focal).
depth-eval env."""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
R_TRUE = 5.0; F0 = 800.0; NPIX = 26; CONE = 2.5


def look_at_pose(C):
    """camera at C looking toward +z (down the cylinder axis)."""
    fwd = np.array([0, 0, 1.0]); up = np.array([0, 1.0, 0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right); up2 = np.cross(fwd, right)
    R = np.stack([right, up2, fwd])                    # world->cam rows
    return R


def cameras(N=10, depth=6.0, cone=CONE, seed=1):
    r = np.random.default_rng(seed); lat = depth * np.tan(np.radians(cone))
    C = np.c_[r.uniform(-1, 1, (N, 2)) * lat, -depth + r.uniform(-0.15, 0.15, N) * depth]
    return [(look_at_pose(c), c) for c in C]


def cyl_hit_batch(C, DW, r, A=1.0):
    """vectorized: forward intersection of rays (C, DW[M,3]) with cylinder x^2+y^2=r^2 -> B/A per ray."""
    a = DW[:, 0] ** 2 + DW[:, 1] ** 2
    b = 2 * (C[0] * DW[:, 0] + C[1] * DW[:, 1]); c = C[0] ** 2 + C[1] ** 2 - r ** 2
    disc = b * b - 4 * a * c
    ok = (a > 1e-9) & (disc >= 0)
    t = np.where(ok, (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a + 1e-12), -1.0)
    P = C[None, :] + t[:, None] * DW
    ok = ok & (t > 1e-6) & (P[:, 2] > -25) & (P[:, 2] < 25)
    n = np.c_[-P[:, 0], -P[:, 1], np.zeros(len(P))] / r
    ndl = np.sum(n * (-DW), axis=1)
    ok = ok & (ndl > 1e-3)
    B = np.where(ok, A * ndl / (t ** 2 + 1e-12), 0.0)
    return ok, B


def pixel_dirs(R, f):
    us = np.linspace(-0.5, 0.5, NPIX)
    U, V = np.meshgrid(us, us)
    DC = np.c_[U.ravel() * F0, V.ravel() * F0, np.full(U.size, f)]
    DC /= np.linalg.norm(DC, axis=1, keepdims=True)
    return (R.T @ DC.T).T                              # world ray dirs (M,3)


def render(cams, r, f, A=1.0):
    """observed pixel-index + B per camera (rendered with TRUE geometry/pose)."""
    obs = []
    for (R, C) in cams:
        DW = pixel_dirs(R, f); ok, B = cyl_hit_batch(C, DW, r, A)
        idx = np.where(ok)[0]
        obs.append((C, idx, B[idx]))
    return obs


def predict_resid(obs, cams_est, f_est, r_est, A_est=1.0):
    """residual using ESTIMATED poses/focal at the SAME pixel indices."""
    res = []
    for (C, idx, Bobs), (Re, Ce) in zip(obs, cams_est):
        DW = pixel_dirs(Re, f_est)[idx]; ok, Bp = cyl_hit_batch(Ce, DW, r_est, A_est)
        res.append(Bp - Bobs)
    return np.concatenate(res) if res else np.zeros(1)


def fit_r(obs, cams_est, f_est):
    from scipy.optimize import least_squares
    A0 = 1.0
    def resid(x): return predict_resid(obs, cams_est, f_est, x[0], x[1])
    sol = least_squares(resid, [5.0, 1.0], method="lm", max_nfev=800)
    return sol.x[0]


def perturb(cams, sigma_t, sigma_rot_deg, seed):
    r = np.random.default_rng(seed); out = []
    for (R, C) in cams:
        Cp = C + r.normal(0, sigma_t, 3)
        ax = r.normal(0, 1, 3); ax /= np.linalg.norm(ax); ang = np.radians(r.normal(0, sigma_rot_deg))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        out.append((dR @ R, Cp))
    return out


def mc(cams, obs, setting, n=100):
    csa = []
    for k in range(n):
        cp = perturb(cams, setting["t"], setting["rot"], seed=100 + k)
        fe = F0 * (1 + np.random.default_rng(200 + k).normal(0, setting["f"]))
        r = fit_r(obs, cp, fe); csa.append(np.pi * r ** 2)
    csa = np.array(csa); csa_true = np.pi * R_TRUE ** 2
    return dict(CSA_mean=float(csa.mean()), CSA_bias_pct=round(100 * (csa.mean() - csa_true) / csa_true, 2),
               CSA_std_pct=round(100 * csa.std() / csa_true, 2), CSA_p5_p95_pct=[round(100 * (np.percentile(csa, 5) - csa_true) / csa_true, 1), round(100 * (np.percentile(csa, 95) - csa_true) / csa_true, 1)])


def main():
    cams = cameras(); obs = render(cams, R_TRUE, F0)
    # realistic COLMAP-ish noise; then isolate each source for sensitivity ranking
    SET = {
        "combined": {"t": 0.06, "rot": 0.3, "f": 0.01},         # ~1% transl of depth, 0.3deg, 1% focal
        "translation_only": {"t": 0.06, "rot": 0.0, "f": 0.0},
        "rotation_only": {"t": 0.0, "rot": 0.3, "f": 0.0},
        "focal_only": {"t": 0.0, "rot": 0.0, "f": 0.01},
    }
    res = {k: mc(cams, obs, v, n=100) for k, v in SET.items()}
    rank = sorted([("translation", res["translation_only"]["CSA_std_pct"]),
                   ("rotation", res["rotation_only"]["CSA_std_pct"]),
                   ("focal", res["focal_only"]["CSA_std_pct"])], key=lambda x: -x[1])
    verdict = dict(combined_CSA_std_pct=res["combined"]["CSA_std_pct"], combined_CSA_bias_pct=res["combined"]["CSA_bias_pct"],
                   sensitivity_ranking=[f"{n}:{v}%" for n, v in rank],
                   PASS=bool(res["combined"]["CSA_std_pct"] < 10.0 and abs(res["combined"]["CSA_bias_pct"]) < 10.0))
    open("test3_pose.json", "w").write(json.dumps({"verdict": verdict, "settings": SET, "results": res}, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    keys = list(res); std = [res[k]["CSA_std_pct"] for k in keys]; bias = [res[k]["CSA_bias_pct"] for k in keys]
    x = np.arange(len(keys)); ax.bar(x - 0.2, std, 0.4, label="CSA std %", color="tab:blue")
    ax.bar(x + 0.2, np.abs(bias), 0.4, label="|CSA bias| %", color="tab:orange"); ax.axhline(10, c="k", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15); ax.set_ylabel("% of true CSA"); ax.legend()
    ax.set_title(f"TEST 3 pose uncertainty (100 MC) @2.5° — PASS={verdict['PASS']}"); fig.tight_layout(); fig.savefig("test3_pose.png", dpi=120)
    print("=== TEST 3 (100 MC each) ===")
    for k in keys: print(f"{k:18} CSA std={res[k]['CSA_std_pct']:>6}%  bias={res[k]['CSA_bias_pct']:>6}%  p5/p95={res[k]['CSA_p5_p95_pct']}")
    print(f"\nsensitivity: {verdict['sensitivity_ranking']}")
    print(f"VERDICT: combined std={verdict['combined_CSA_std_pct']}%, bias={verdict['combined_CSA_bias_pct']}% -> PASS={verdict['PASS']}")


if __name__ == "__main__":
    main()
