"""REBUILD -- the ACTUAL designed estimator (per AIRWAY_FIT_DESIGN.md sec.4/6), not the photometry-
only stub. Total loss  E = E_bnd + alpha*E_shade + beta*E_smooth  with:
  E_bnd  (PRIMARY): occluding-contour reprojection -- project the model ring's contour and match the
                    observed lumen boundary (pixels). Geometric; pins r/pose from image geometry.
  E_shade (WEAK):   near-light shading via the GAIN-INVARIANT log-gradient residual
                    || d_phi log B - d_phi log B_hat ||^2 (removes exposure/albedo gain).
  free POSE/centerline: the tube ring has a free center O and axis a (the gauge DOF that was MISSING
                    before -- a global Sim(3) on cameras+tube is absorbed by (O,a,r)).
Cross-section = one ring (O in R^3, unit axis a, radius r); CSA = pi r^2. Cameras have KNOWN poses
(R_i, C_i, focal f). Units: length = scene units, angles = rad, pixels = f * (x/z). depth-eval env."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

F0 = 800.0
LAM_BND = 1.0            # primary (pixel^2)
LAM_SHADE = 0.02         # weak, per design (alpha small); acts on the gain-invariant log-gradient


def basis(a):
    a = a / (np.linalg.norm(a) + 1e-12)
    t = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(a, t); e1 /= np.linalg.norm(e1) + 1e-12; e2 = np.cross(a, e1)
    return e1, e2


def ring_pts(O, a, r, phi):
    e1, e2 = basis(a)
    return O[None, :] + r * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])


def project(X, R, C, f):
    Xc = (R @ (X - C[None, :]).T).T; z = Xc[:, 2]
    return f * Xc[:, :2] / z[:, None], z


def shading(X, O, a, R, C, AI):
    rel = X - O[None, :]; ax = rel @ a; radial = rel - ax[:, None] * a[None, :]
    n = -radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12)
    v = C[None, :] - X; d = np.linalg.norm(v, axis=1); vh = v / d[:, None]
    ndl = np.sum(n * vh, axis=1)
    return AI * np.maximum(1e-4, ndl) / d ** 2


def dphi(v):                       # circular angular derivative (gain-invariant when applied to log B)
    return np.roll(v, -1) - np.roll(v, 1)


def observe(O, a, r, cams, f, AI, phi):
    """observed contour pixels + log-gradient shading per camera (from the TRUE scene)."""
    X = ring_pts(O, a, r, phi); obs = []
    for (R, C) in cams:
        p, z = project(X, R, C, f); B = shading(X, O, a, R, C, AI)
        obs.append((p, dphi(np.log(np.clip(B, 1e-6, None)))))
    return obs


def _unpack(theta):
    O = theta[:3]; a = np.array([theta[3], theta[4], 1.0]); r = theta[5]; AI = np.exp(theta[6])
    return O, a, r, AI


def residual(theta, obs, cams, f, phi, use_contour=True):
    O, a, r, AI = _unpack(theta); X = ring_pts(O, a, r, phi); res = []
    for (p_obs, g_obs), (R, C) in zip(obs, cams):
        p, z = project(X, R, C, f); B = shading(X, O, a, R, C, AI)
        g = dphi(np.log(np.clip(B, 1e-6, None)))
        if use_contour:
            res.append(np.sqrt(LAM_BND) * (p - p_obs).ravel())
        res.append(np.sqrt(LAM_SHADE) * (g - g_obs))
    return np.concatenate(res)


def fit(obs, cams, f, phi, theta0, use_contour=True):
    sol = least_squares(lambda th: residual(th, obs, cams, f, phi, use_contour),
                        theta0, method="lm", max_nfev=6000)
    O, a, r, AI = _unpack(sol.x)
    return dict(O=O, a=a / np.linalg.norm(a), r=abs(r), CSA=np.pi * r ** 2, sol=sol,
                rms=float(np.sqrt(np.mean(sol.fun ** 2))), nfev=sol.nfev, success=sol.success)


def cameras(N=14, depth=8.0, cone_deg=2.5, seed=1, O=np.zeros(3)):
    """cameras behind the ring (offset along -a=-z from O) looking +z, lateral spread = cone."""
    rng = np.random.default_rng(seed); lat = depth * np.tan(np.radians(cone_deg))
    C = O[None, :] + np.c_[rng.uniform(-1, 1, (N, 2)) * lat, -depth + rng.uniform(-0.15, 0.15, N) * depth]
    R = np.stack([np.eye(3)] * N)      # looking +z (world==cam axes)
    return [(R[i], C[i]) for i in range(N)]


def theta_from(O, a, r, AI=1.0):
    a = a / np.linalg.norm(a)
    return np.array([O[0], O[1], O[2], a[0] / a[2], a[1] / a[2], r, np.log(AI)])
