"""Phase-0 red-team simulation core: forward model + estimator for the PROPOSED estimator --
a near-light (point light at camera) Lambertian shading fit of a surface-of-revolution airway
cross-section, using known (COLMAP) poses. We try to FALSIFY it; we do NOT improve it.

Forward model (proposed estimator's own assumptions):
  wall point p(theta) = (r(theta) cos, r(theta) sin, 0), inward normal n = -(cos,sin,0)
  camera i center C_i (known pose), light AT camera, Lambertian:
      B_ij = A(theta_j) * max(0, n_j . vhat_ij) / d_ij^2 ,  vhat = (C_i - p_j)/d ,  A = albedo*I
Cross-section area CSA = area enclosed by the boundary r(theta); DCE = 2*sqrt(CSA/pi).

The estimator fits geometry (circular r0, or Fourier r(theta)) + albedo (constant A, or Fourier
A(theta)) to the observed B by least squares. Cameras carry a controllable viewing-cone (parallax);
'realistic 2_V2' ~ 2.5 deg (from the geometry-gap audit), 'ideal' ~ 15 deg. depth-eval env."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

rng0 = np.random.default_rng(0)


def make_cameras(N=14, depth=8.0, cone_deg=2.5, seed=1):
    """N cameras behind the ring (z<0) looking +z, lateral spread set by the viewing cone."""
    rng = np.random.default_rng(seed)
    lat = depth * np.tan(np.radians(cone_deg))            # lateral spread ~ cone * depth
    z = -depth + rng.uniform(-0.15, 0.15, N) * depth      # small axial spread (dwelling)
    xy = rng.uniform(-1, 1, (N, 2)) * lat
    C = np.c_[xy, z]
    return C


def boundary(theta, kind="circle", r0=5.0, **kw):
    """true cross-section boundary r(theta) for the phantoms."""
    if kind == "circle":
        return np.full_like(theta, r0)
    if kind == "ellipse":                                 # axis ratio kw['ecc'] (b/a)
        a = r0; b = r0 * kw.get("ecc", 0.7)
        return a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    if kind == "lobed":                                   # k-lobed
        return r0 * (1 + kw.get("amp", 0.25) * np.cos(kw.get("k", 3) * theta))
    if kind == "asym_scar":                               # localized inward scar over an arc
        d = np.cos(theta - kw.get("phi", 1.0)); s = np.clip(d, 0, 1) ** 4
        return r0 * (1 - kw.get("depth", 0.35) * s)
    if kind == "noncirc":                                 # generic smooth non-circular (few Fourier)
        return r0 * (1 + 0.18 * np.cos(2 * theta + 0.5) + 0.10 * np.cos(3 * theta) + 0.06 * np.sin(5 * theta))
    raise ValueError(kind)


def albedo_field(theta, kind="const", A0=1.0, **kw):
    if kind == "const":
        return np.full_like(theta, A0)
    if kind == "gradient":
        return A0 * (1 + kw.get("amp", 0.5) * np.cos(theta - kw.get("phi", 0.7)))
    if kind == "stripes":                                 # vascular stripes
        return A0 * (1 + kw.get("amp", 0.4) * np.sign(np.cos(kw.get("k", 6) * theta)) * 0.5)
    if kind == "lowfreq":
        r = np.random.default_rng(kw.get("seed", 3))
        c = r.normal(0, 1, 4)
        return A0 * (1 + kw.get("amp", 0.4) * (c[0] * np.cos(theta) + c[1] * np.sin(theta) + c[2] * np.cos(2 * theta) + c[3] * np.sin(2 * theta)) / 2)
    if kind == "highfreq":
        return A0 * (1 + kw.get("amp", 0.4) * np.cos(kw.get("k", 12) * theta + 0.3))
    raise ValueError(kind)


def forward(C, r_theta, A_theta, theta, center=(0.0, 0.0), z0=0.0):
    """B[i,j] over cameras i, wall points j; returns B and a visibility mask."""
    cx, cy = center
    P = np.c_[cx + r_theta * np.cos(theta), cy + r_theta * np.sin(theta), np.full_like(theta, z0)]
    n = np.c_[-np.cos(theta), -np.sin(theta), np.zeros_like(theta)]   # inward normal (local cylinder)
    B = np.zeros((len(C), len(theta))); vis = np.zeros_like(B, bool)
    for i, Ci in enumerate(C):
        v = Ci[None, :] - P; d = np.linalg.norm(v, axis=1); vh = v / d[:, None]
        ndl = np.sum(n * vh, axis=1)
        lit = ndl > 1e-3
        B[i, lit] = A_theta[lit] * ndl[lit] / d[lit] ** 2
        vis[i, lit] = True
    return B, vis


# ---------- estimator parametrizations ----------
def unpack(p, nfr_geom, nfr_alb):
    """p = [r0, geomFourier(2*nfr_geom), A0, albFourier(2*nfr_alb), (cx,cy)]"""
    i = 0; r0 = p[i]; i += 1
    gk = p[i:i + 2 * nfr_geom]; i += 2 * nfr_geom
    A0 = p[i]; i += 1
    ak = p[i:i + 2 * nfr_alb]; i += 2 * nfr_alb
    cx, cy = (p[i], p[i + 1]) if len(p) > i + 1 else (0.0, 0.0)
    return r0, gk, A0, ak, cx, cy


def fourier_eval(base, coeffs, theta):
    v = np.full_like(theta, base)
    for k in range(len(coeffs) // 2):
        v = v + coeffs[2 * k] * np.cos((k + 1) * theta) + coeffs[2 * k + 1] * np.sin((k + 1) * theta)
    return v


def model_B(p, theta, C, nfr_geom, nfr_alb, fit_center):
    r0, gk, A0, ak, cx, cy = unpack(p, nfr_geom, nfr_alb)
    r = fourier_eval(r0, gk, theta); A = fourier_eval(A0, ak, theta)
    ctr = (cx, cy) if fit_center else (0.0, 0.0)
    B, vis = forward(C, r, A, theta, center=ctr)
    return B, vis, r


def fit(B_obs, vis_obs, theta, C, nfr_geom=0, nfr_alb=0, fit_center=False, r_init=5.0, A_init=1.0):
    p0 = [r_init] + [0.0] * (2 * nfr_geom) + [A_init] + [0.0] * (2 * nfr_alb) + ([0.0, 0.0] if fit_center else [])
    p0 = np.array(p0, float)
    m = vis_obs

    def resid(p):
        B, vis, _ = model_B(p, theta, C, nfr_geom, nfr_alb, fit_center)
        return (B - B_obs)[m]
    sol = least_squares(resid, p0, method="lm", max_nfev=4000)
    return sol


def jacobian(p, theta, C, nfr_geom, nfr_alb, fit_center, mask, eps=1e-5):
    """finite-difference Jacobian of the residual wrt params at p (over visible obs)."""
    base, _, _ = model_B(p, theta, C, nfr_geom, nfr_alb, fit_center)
    base = base[mask]; J = np.zeros((base.size, len(p)))
    for k in range(len(p)):
        pp = p.copy(); h = eps * max(1.0, abs(p[k])); pp[k] += h
        Bk, _, _ = model_B(pp, theta, C, nfr_geom, nfr_alb, fit_center)
        J[:, k] = (Bk[mask] - base) / h
    return J


def csa_dce(r_theta, theta):
    """area enclosed by polar boundary r(theta); DCE = 2 sqrt(A/pi)."""
    A = 0.5 * np.trapz(r_theta ** 2, theta) if hasattr(np, "trapz") else 0.5 * np.trapezoid(r_theta ** 2, theta)
    return float(A), float(2 * np.sqrt(A / np.pi))


def rank_cond(J, tol_ratio=1e-3):
    s = np.linalg.svd(J, compute_uv=False)
    smax = s[0] if len(s) else 0.0
    rank = int((s > tol_ratio * smax).sum())
    cond = float(smax / s[s > 1e-12][-1]) if (s > 1e-12).any() else np.inf
    return rank, cond, s
