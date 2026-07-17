"""Direct geometric metrology from a SHORT AXIAL SEGMENT (not a single ring). Photometric term DROPPED.

The airway throat region is modelled as a short generalized cylinder:
  - centerline  C(s) = O + s*a + 1/2 s^2 * b        (a: axis unit, b: bend vector)
  - NON-CIRCULAR cross-section  shape(theta) = 1 + e*cos(2(theta-phi)) + lobe*cos(k(theta-phi))
  - axial radius profile  r0(s) = r_t + 1/2 * kappa * (s - s_t)^2   (throat radius r_t, throat location
    s_t, local flare/curvature kappa).
  Surface: X(s,theta) = C(s) + r0(s)*shape(theta)*(cos theta * e1 + sin theta * e2).
  Throat CSA = pi * r_t^2 * (1 + 1/2 e^2 + 1/2 lobe^2);  DCE = 2 sqrt(CSA/pi).

Forward model = the TRUE occluding contour (silhouette), enforced IMPLICITLY and consistently with the
phantom's first-hit/occlusion definition: a viewing ray belongs to the silhouette iff it is TANGENT to
the near wall (up to the throat plane). For a ray, define
    f(ray) = max over zeta<=s_t of ( ray_radius(zeta) - r_model(zeta, theta(zeta)) ).
f<0 => ray stays inside (through the aperture / dark);  f>0 => ray exits the wall before the throat
(lit);  f=0 => tangent = ON the silhouette. Each detected contour pixel is a silhouette ray, so its
residual is f(ray) (mm), zero at the true geometry. This is exact by construction (no contour-prediction
approximation) and is the honest replacement for the single-circle projection-chamfer fit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

N_ZETA = 60                      # axial march samples per ray for the tangency (silhouette) test


def _basis(a):
    """In-plane cross-section basis with e1 aligned to the world-x projection, so the azimuth
    th = atan2(radv·e2, radv·e1) MATCHES the generator's world azimuth atan2(X1, X0) (for axis a=+z,
    e1=[1,0,0], e2=[0,1,0]). This is the SAME convention as phantom._shape, so the optimizer's `phi`
    corresponds directly to the generator's th0 (fixes the prior -90° basis-convention bug)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - (ref @ a) * a; e1 /= np.linalg.norm(e1) + 1e-12
    return e1, np.cross(a, e1)


def _shape(theta, e, lobe, lobe_k, phi):
    return 1.0 + e * np.cos(2 * (theta - phi)) + lobe * np.cos(lobe_k * (theta - phi))


def _r0(s, r_t, s_t, kappa, kappa4=0.0):
    u = s - s_t
    return r_t + 0.5 * kappa * u ** 2 + (1.0 / 24.0) * kappa4 * u ** 4   # quartic ~ matches a Gaussian wall


@dataclass
class SegParams:
    O: np.ndarray; a: np.ndarray; b: np.ndarray
    r_t: float; s_t: float; kappa: float; kappa4: float
    e: float; phi: float; lobe: float
    lobe_k: int = 3
    half_len: float = 3.0


def throat_csa_dce(p: SegParams):
    """Throat = the narrowest cross-section = the profile vertex (r0 minimum at s_t). CSA at the vertex,
    NOT a min over the finite segment (which is degenerate when kappa~0)."""
    shape_factor = 1 + 0.5 * p.e ** 2 + 0.5 * p.lobe ** 2
    csa = float(np.pi * p.r_t ** 2 * shape_factor)
    dce = float(2 * np.sqrt(max(csa, 0) / np.pi))
    s_th = float(np.clip(p.s_t, -p.half_len, p.half_len))
    return csa, dce, s_th


def csa_dce_profile(p: SegParams, ns=41):
    s = np.linspace(-p.half_len, p.half_len, ns)
    r0 = _r0(s, p.r_t, p.s_t, p.kappa, p.kappa4)
    shape_factor = 1 + 0.5 * p.e ** 2 + 0.5 * p.lobe ** 2
    csa = np.pi * r0 ** 2 * shape_factor
    return s, csa, 2 * np.sqrt(np.clip(csa, 0, None) / np.pi)


def _unpack(theta, half_len, lobe_k):
    O = theta[0:3]; a = np.array([theta[3], theta[4], 1.0]); b = np.array([theta[5], theta[6], 0.0])
    r_t, s_t, kappa, kappa4, e, phi, lobe = theta[7:14]
    kappa = np.log1p(np.exp(min(kappa, 30.0)))            # softplus -> kappa>=0 (throat is a MINIMUM)
    return SegParams(O=O, a=a, b=b, r_t=abs(r_t), s_t=s_t, kappa=float(kappa), kappa4=float(kappa4),
                     e=e, phi=phi, lobe=lobe, lobe_k=lobe_k, half_len=half_len)


def _tangency_core(p: SegParams, rays_o, rays_d):
    """For each ray compute the tangency margin fmax = max over the FORWARD (t>0) axial support of
    (ray_radius - r_model), and a per-ray validity flag. A ray is VALID iff the candidate segment has a
    forward intersection/tangency in front of the camera within the axial support; otherwise the segment
    is entirely behind the camera / outside support -> has_valid=False (fmax=-inf). NO sentinel value is
    baked into the residual here (that is applied by the caller)."""
    a = p.a / (np.linalg.norm(p.a) + 1e-12); e1, e2 = _basis(a)
    zeta = np.linspace(-p.half_len, p.s_t, N_ZETA)        # near wall up to the throat plane
    a_dot_oO = (rays_o - p.O[None, :]) @ a                # (N,)
    a_dot_d = rays_d @ a                                  # (N,)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (zeta[None, :] - a_dot_oO[:, None]) / (a_dot_d[:, None] + 1e-12)   # (N,Z)
    X = rays_o[:, None, :] + t[:, :, None] * rays_d[:, None, :]                # (N,Z,3)
    rel = X - p.O[None, None, :]
    zc = rel @ a                                          # (N,Z) actual axial coord (~zeta)
    Ccl = p.O[None, None, :] + zc[:, :, None] * a[None, None, :] + 0.5 * (zc ** 2)[:, :, None] * p.b[None, None, :]
    radv = X - Ccl
    radv = radv - (np.einsum("nzk,k->nz", radv, a)[:, :, None]) * a[None, None, :]
    hyp = np.linalg.norm(radv, axis=2)                    # (N,Z) ray radius about the centerline
    th = np.arctan2(radv @ e2, radv @ e1)
    rmod = _r0(zc, p.r_t, p.s_t, p.kappa, p.kappa4) * _shape(th, p.e, p.lobe, p.lobe_k, p.phi)
    valid_s = t > 0                                       # forward samples only
    gv = np.where(valid_s, hyp - rmod, -np.inf)
    fmax = gv.max(axis=1)                                 # (N,) tangency margin; -inf if no forward sample
    has_valid = np.isfinite(fmax)
    return fmax, has_valid


def _tangency_f(p: SegParams, rays_o, rays_d, pen):
    """Optimized per-ray residual: the tangency margin for VALID rays; a moderate constant PENALTY `pen`
    (NOT zero, NOT the old -1e9 sentinel) for rays whose segment is behind the camera / outside support,
    so invalid rays are penalized (the optimizer cannot evade disagreement by hiding the ring behind
    cameras) without a huge constant swamping the loss."""
    fmax, hv = _tangency_core(p, rays_o, rays_d)
    return np.where(hv, fmax, pen)


def _rays(det, cam, Kinv):
    uv1 = np.c_[det, np.ones(len(det))]
    dcam = (Kinv @ uv1.T).T
    d = (cam.R.T @ dcam.T).T
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    o = np.repeat(cam.C[None, :], len(det), axis=0)
    return o, d


def residuals(theta, contours, cams, Kinv, half_len, lobe_k, pen=0.5):
    p = _unpack(theta, half_len, lobe_k)
    res = []
    for det, cam in zip(contours, cams):
        if det is None:
            continue
        o, d = _rays(det, cam, Kinv)
        res.append(_tangency_f(p, o, d, pen))            # tangency for valid rays; `pen` for invalid
    r = np.concatenate(res) if res else np.zeros(1)
    reg = np.array([1.0 * theta[5], 1.0 * theta[6]])     # keep bend small (weak prior)
    return np.concatenate([r, reg])


def frame_diagnostics(theta, contours, cams, K, half_len=3.0, lobe_k=3, ray_frac_valid=0.5):
    """Per-frame validity + tangency quality (REPORTING only; not part of the optimized loss).
    Returns (per_frame list, n_valid_frames, valid_ray_tangency_rms). A frame is 'valid' if at least
    ray_frac_valid of its rays have a forward tangency (segment visible in front of that camera)."""
    Kinv = np.linalg.inv(K); p = _unpack(theta, half_len, lobe_k)
    per = []; all_valid_f = []
    for det, cam in zip(contours, cams):
        if det is None:
            per.append(dict(valid_frac=0.0, tang_rms=float("nan"), n_rays=0, frame_valid=False)); continue
        o, d = _rays(det, cam, Kinv)
        fmax, hv = _tangency_core(p, o, d)
        vf = float(hv.mean())
        trms = float(np.sqrt(np.mean(fmax[hv] ** 2))) if hv.any() else float("nan")
        if hv.any():
            all_valid_f.append(fmax[hv])
        per.append(dict(valid_frac=vf, tang_rms=trms, n_rays=int(len(det)), frame_valid=bool(vf >= ray_frac_valid)))
    nvf = int(sum(x["frame_valid"] for x in per))
    vrms = float(np.sqrt(np.mean(np.concatenate(all_valid_f) ** 2))) if all_valid_f else float("nan")
    return per, nvf, vrms


_N_REG = 2                                                # number of regularisation residuals appended


@dataclass
class SegFit:
    p: SegParams
    csa: float; dce: float; s_throat: float
    resid_rms_mm: float; success: bool; nfev: int; cost: float
    jac_rank: int = -1; jac_cond: float = float("nan")
    singular_values: tuple = ()
    csa_var: float = float("nan"); dce_var: float = float("nan")


def _cov_from_jac(sol, p):
    """Gauss-Newton covariance from the solution Jacobian (verification only; does NOT change the loss).
    Propagate to Var(CSA), Var(DCE) via the delta method. CSA = pi r_t^2 (1 + e^2/2 + lobe^2/2)."""
    J = sol.jac; m, n = J.shape
    sv = np.linalg.svd(J, compute_uv=False)
    tol = 1e-6 * sv[0]
    rank = int(np.sum(sv > tol))
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else float("inf")   # over the identifiable subspace
    dof = max(1, m - n); sigma2 = 2.0 * sol.cost / dof
    r_t, e, lobe = abs(sol.x[7]), sol.x[11], sol.x[13]
    sf = 1 + 0.5 * e ** 2 + 0.5 * lobe ** 2
    grad = np.zeros(n)
    grad[7] = 2 * np.pi * r_t * sf * np.sign(sol.x[7] if sol.x[7] != 0 else 1.0)
    grad[11] = np.pi * r_t ** 2 * e
    grad[13] = np.pi * r_t ** 2 * lobe
    # pseudo-inverse over the identifiable subspace: the segment has ~2 unconstrained gauge directions
    # (bend b for a straight throat) so J^T J is rank-deficient; pinv gives finite variance for the
    # well-constrained CSA directions (r_t, e, lobe). Verification only; does not change the loss.
    cov = sigma2 * np.linalg.pinv(J.T @ J, rcond=1e-8)
    csa_var = float(grad @ cov @ grad)
    if not np.isfinite(csa_var) or csa_var < 0:
        csa_var = float("nan")
    csa = np.pi * r_t ** 2 * sf
    dce_var = float(csa_var / (np.pi * csa)) if (np.isfinite(csa_var) and csa > 1e-9) else float("nan")
    return rank, cond, tuple(float(s) for s in sv), csa_var, dce_var


def fit(contours, cams, K, theta0, half_len=3.0, lobe_k=3, max_nfev=400, pen=0.5):
    Kinv = np.linalg.inv(K)
    sol = least_squares(residuals, theta0, args=(contours, cams, Kinv, half_len, lobe_k, pen),
                        method="trf", max_nfev=max_nfev, x_scale="jac")
    p = _unpack(sol.x, half_len, lobe_k)
    csa, dce, s_th = throat_csa_dce(p)
    r = residuals(sol.x, contours, cams, Kinv, half_len, lobe_k, pen)
    rms = float(np.sqrt(np.mean(r[:-2] ** 2))) if len(r) > 3 else float("nan")
    rank, cond, sv, csa_var, dce_var = _cov_from_jac(sol, p)
    return SegFit(p=p, csa=csa, dce=dce, s_throat=s_th, resid_rms_mm=rms,
                  success=bool(sol.success), nfev=int(sol.nfev), cost=float(sol.cost),
                  jac_rank=rank, jac_cond=cond, singular_values=sv, csa_var=csa_var, dce_var=dce_var)


# =========================================================================================
# OPTIMIZATION-ROBUSTNESS ONLY (staged fitting / scaling / multistart). The MODEL and the
# valid-ray tangency loss (_r0, _shape, _tangency_core) are unchanged; these only decide the
# initialization path and solver settings, and select among solutions by FINAL RESIDUAL.
# (Invalid-ray handling in _tangency_f was changed from a -1e9 sentinel to a moderate penalty;
#  this is identity on the phantom, where every ray has a forward tangency.)
# =========================================================================================

# characteristic per-parameter scales for the trust-region (better conditioning than 'jac' here):
# O(mm), a(2), b(2), r_t, s_t, kappa_raw, kappa4, e, phi, lobe
_XSCALE = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 0.05, 0.2, 1.0, 0.2])

# continuation stages: fit a circular SoR first, then release ellipticity, then the lobe.
_STAGE_FREE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],            # circle / surface-of-revolution (e=lobe=0 fixed)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],    # + ellipse (e, phi)
    list(range(14)),                               # + lobe (full parameter set)
]


def _masked_residuals(free_vals, base, free_idx, contours, cams, Kinv, half_len, lobe_k, pen):
    th = base.copy(); th[free_idx] = free_vals
    return residuals(th, contours, cams, Kinv, half_len, lobe_k, pen)


def fit_staged(contours, cams, K, theta0, half_len=3.0, lobe_k=3, max_nfev=200, use_xscale=True, pen=0.5):
    """Same loss/model as fit(); solved by continuation (circle -> ellipse -> lobe) with explicit
    parameter scaling. The final stage optimizes ALL 14 parameters with the identical residual, so the
    objective is unchanged — only the initialization path and x_scale differ."""
    Kinv = np.linalg.inv(K)
    th = np.array(theta0, float).copy(); th[11] = 0.0; th[13] = 0.0     # start from a circular section
    xs = _XSCALE if use_xscale else None
    sol = None
    for si, free in enumerate(_STAGE_FREE):
        base = th.copy()
        if si == 0: base[11] = 0.0; base[13] = 0.0                     # e=lobe=0 during the circle stage
        if si == 1: base[13] = 0.0                                     # lobe=0 during the ellipse stage
        free = np.asarray(free)
        xsub = xs[free] if xs is not None else "jac"
        sol = least_squares(_masked_residuals, base[free],
                            args=(base, free, contours, cams, Kinv, half_len, lobe_k, pen),
                            method="trf", max_nfev=max_nfev, x_scale=xsub)
        th = base.copy(); th[free] = sol.x
    p = _unpack(th, half_len, lobe_k)
    csa, dce, s_th = throat_csa_dce(p)
    r = residuals(th, contours, cams, Kinv, half_len, lobe_k, pen)
    rms = float(np.sqrt(np.mean(r[:-2] ** 2))) if len(r) > 3 else float("nan")
    rank, cond, sv, csa_var, dce_var = _cov_from_jac(sol, p)           # stage-3 sol.x/jac is the full set
    return SegFit(p=p, csa=csa, dce=dce, s_throat=s_th, resid_rms_mm=rms,
                  success=bool(sol.success), nfev=int(sol.nfev), cost=float(sol.cost),
                  jac_rank=rank, jac_cond=cond, singular_values=sv, csa_var=csa_var, dce_var=dce_var)


def fit_multistart(contours, cams, K, inits, half_len=3.0, lobe_k=3, max_nfev=200, pen=0.5):
    """Run the staged fit from several inits and RETURN THE LOWEST-FINAL-RESIDUAL solution (rejecting
    higher-residual local minima). Selection uses only the observed residual — no ground truth. Invalid
    rays cost `pen`, so a candidate that explains only a few frames has a high residual and loses."""
    fits = [fit_staged(contours, cams, K, ti, half_len, lobe_k, max_nfev, pen=pen) for ti in inits]
    best = min(fits, key=lambda f: f.resid_rms_mm)
    return best, fits


def predicted_contour(p: SegParams, cam, K, ndir=120, rmax_px=260):
    """Silhouette contour for FIGURES: from the projected throat centre, march each image radial outward
    and find the tangency (f=0) crossing. Uses the tangency margin (VALID rays only); rays whose segment
    is behind the camera / outside support are treated as interior so the crossing marks the silhouette."""
    Kinv = np.linalg.inv(K)
    c0, z0 = _project_pt(p.O + p.s_t * (p.a / np.linalg.norm(p.a)), cam, K)
    if z0 <= 0:
        c0 = np.array([K[0, 2], K[1, 2]])
    ang = np.linspace(-np.pi, np.pi, ndir, endpoint=False)
    tvals = np.linspace(2.0, rmax_px, 80)
    out = []
    for al in ang:
        px = c0[None, :] + tvals[:, None] * np.array([np.cos(al), np.sin(al)])[None, :]
        o, d = _rays(px, cam, Kinv)
        fmax, hv = _tangency_core(p, o, d)
        f = np.where(hv, fmax, -1e3)                     # invalid ray -> interior (very negative)
        s = np.sign(f)
        cr = np.where(np.diff(s) != 0)[0]
        if len(cr):
            out.append(px[cr[0]])
    return np.array(out) if len(out) >= 8 else None


def _project_pt(X, cam, K):
    Xc = cam.R @ (X - cam.C); z = max(Xc[2], 1e-6)
    uv = K @ (Xc / z)
    return uv[:2], Xc[2]


def init_from_contours(contours, cams, K, z_guess, kappa=0.15, kappa4=0.0):
    """Data-driven init: O from back-projected contour centroids; r_t from median angular radius*depth;
    ellipticity (e, phi) from the aperture second moments (averaged, de-projected roughly). lobe=0."""
    Kinv = np.linalg.inv(K); Os = []; rs = []; ecs = []; phis = []
    for det, cam in zip(contours, cams):
        if det is None or len(det) < 8:
            continue
        c = det.mean(0)
        ray = Kinv @ np.array([c[0], c[1], 1.0]); ray /= np.linalg.norm(ray)
        dw = cam.R.T @ ray
        if abs(dw[2]) < 1e-6:
            continue
        t = (z_guess - cam.C[2]) / dw[2]
        Os.append(cam.C + t * dw)
        d = det - c; rpx = np.median(np.linalg.norm(d, axis=1))
        rs.append(rpx / K[0, 0] * t)
        # 2nd-moment ellipse of the aperture contour -> ellipticity amplitude + orientation
        cov = np.cov(d.T); w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-9, None); ratio = np.sqrt(w[1] / w[0])          # major/minor
        ecs.append((ratio - 1) / (ratio + 1))                            # ~ e amplitude
        vmaj = V[:, 1]; phis.append(np.arctan2(vmaj[1], vmaj[0]))
    O = np.mean(Os, 0) if Os else np.array([0, 0, z_guess])
    r_t = float(np.median(rs)) if rs else 3.0
    e0 = float(np.median(ecs)) if ecs else 0.0
    phi0 = float(np.median(np.unwrap(2 * np.array(phis)) / 2)) if phis else 0.0
    return theta_from(O, [0, 0, 1.0], r_t=r_t * 0.95, s_t=0.0, kappa=kappa, kappa4=kappa4,
                      e=e0, phi=phi0, lobe=0.0)


def theta_from(O, a, r_t, s_t=0.0, kappa=0.15, kappa4=0.0, e=0.0, phi=0.0, lobe=0.0, b=(0.0, 0.0)):
    a = np.asarray(a, float); a = a / a[2] if abs(a[2]) > 1e-6 else a
    kraw = np.log(np.expm1(max(kappa, 1e-3)))            # invert softplus for the init
    return np.array([O[0], O[1], O[2], a[0], a[1], b[0], b[1], r_t, s_t, kraw, kappa4, e, phi, lobe])
