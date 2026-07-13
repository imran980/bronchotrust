"""Direct-metrology estimator for ONE airway cross-section (the stenosis throat).
Circular cross-section model only. Fits the throat circle (centre O, plane axis a, radius r) to
multi-view lumen-contour observations, with a weak gain-invariant log-gradient photometric term.
Known/fixed camera poses + intrinsics. World units = mm (poses metric) -> output is metric.

Variables (6): O in R^3 (throat centre), a (plane axis, 2 params: a=normalize([ax,ay,1])), r.
Loss:  lambda_c * ||contour reprojection (chamfer)||^2   (PRIMARY)
     + lambda_p * ||d_phi log B_obs - d_phi log B_hat||^2  (WEAK, gain-invariant)
Covariance from the Gauss-Newton Jacobian at the solution. Reproducible (seeded, deterministic).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

M_MODEL = 200                                  # model-contour angular samples
PHI = np.linspace(0, 2 * np.pi, 64, endpoint=False)   # photometric wall samples


def _basis(a):
    a = a / (np.linalg.norm(a) + 1e-12)
    t = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1., 0, 0])
    e1 = np.cross(a, t); e1 /= np.linalg.norm(e1) + 1e-12
    return e1, np.cross(a, e1)


def _project(X, R, C, K):
    Xc = (R @ (X - C[None, :]).T).T; z = np.clip(Xc[:, 2], 1e-6, None)
    uv = (K @ (Xc / z[:, None]).T).T
    return uv[:, :2], Xc[:, 2]


def _unpack(theta):
    return theta[:3], np.array([theta[3], theta[4], 1.0]), theta[5]


def _sample(img, uv):
    v = uv[:, 1].astype(np.float32); u = uv[:, 0].astype(np.float32)
    return cv2.remap(img, u.reshape(-1, 1), v.reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).ravel()


def _dphi(x): return np.roll(x, -1) - np.roll(x, 1)


def residuals(theta, contours, cams, K, images, lam_c, lam_p, wall_offset, use_contour, use_photo):
    O, a, r = _unpack(theta); r = abs(r); e1, e2 = _basis(a)
    Xc = O[None, :] + r * (np.cos(np.linspace(0, 2 * np.pi, M_MODEL, endpoint=False))[:, None] * e1[None, :]
                           + np.sin(np.linspace(0, 2 * np.pi, M_MODEL, endpoint=False))[:, None] * e2[None, :])
    res = []
    for det, cam, img in zip(contours, cams, images):
        if det is None:
            continue
        if use_contour:
            P, z = _project(Xc, cam.R, cam.C, K)
            d = cKDTree(P).query(det)[0]                 # each detected pt -> nearest model-contour pt
            res.append(np.sqrt(lam_c) * d)
        if use_photo:
            Xw = O[None, :] + (wall_offset * r) * (np.cos(PHI)[:, None] * e1[None, :] + np.sin(PHI)[:, None] * e2[None, :])
            uv, z = _project(Xw, cam.R, cam.C, K)
            B_obs = np.clip(_sample(img, uv), 1e-3, None)
            n = -(Xw - O[None, :]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9  # inward wall normal (any axis)
            vv = cam.C[None, :] - Xw; dd = np.linalg.norm(vv, axis=1); vh = vv / dd[:, None]
            B_hat = np.clip(np.sum(n * vh, axis=1), 1e-3, None) / dd ** 2
            res.append(np.sqrt(lam_p) * (_dphi(np.log(B_obs)) - _dphi(np.log(B_hat))))
    return np.concatenate(res) if res else np.zeros(1)


@dataclass
class FitResult:
    O: np.ndarray; axis: np.ndarray; r: float
    csa: float; dce: float
    csa_var: float; dce_var: float; r_var: float
    contour_rms_px: float; photo_rms: float
    success: bool; nfev: int; cost: float


def _cov_r(sol, theta, args):
    """Var(r) from the Gauss-Newton normal matrix at the solution; delta-method to CSA/DCE."""
    J = sol.jac; m, npar = J.shape
    dof = max(1, m - npar); sigma2 = 2 * sol.cost / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        rv = float(abs(cov[5, 5]))
    except np.linalg.LinAlgError:
        rv = float("nan")
    r = abs(theta[5])
    return rv, (2 * np.pi * r) ** 2 * rv, 4 * rv          # Var(r), Var(CSA)=(2 pi r)^2 Var r, Var(DCE)=4 Var r


def fit(contours, cams, K, images, theta0, lam_c=1.0, lam_p=0.02, wall_offset=1.15,
        use_contour=True, use_photo=True, max_nfev=4000):
    args = (contours, cams, K, images, lam_c, lam_p, wall_offset, use_contour, use_photo)
    sol = least_squares(residuals, theta0, args=args, method="trf", max_nfev=max_nfev, x_scale="jac")
    O, a, r = _unpack(sol.x); r = abs(r); a = a / np.linalg.norm(a)
    rv, cv_, dv = _cov_r(sol, sol.x, args)
    # report per-term RMS at the solution
    cr = residuals(sol.x, *(contours, cams, K, images, 1.0, 0.0, wall_offset, True, False))
    pr = residuals(sol.x, *(contours, cams, K, images, 0.0, 1.0, wall_offset, False, True))
    return FitResult(O=O, axis=a, r=float(r), csa=float(np.pi * r ** 2), dce=float(2 * r),
                     csa_var=cv_, dce_var=dv, r_var=rv,
                     contour_rms_px=float(np.sqrt(np.mean(cr ** 2))) if len(cr) > 1 else float("nan"),
                     photo_rms=float(np.sqrt(np.mean(pr ** 2))) if len(pr) > 1 else float("nan"),
                     success=bool(sol.success), nfev=int(sol.nfev), cost=float(sol.cost))


def theta_from(O, axis, r):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.array([O[0], O[1], O[2], axis[0] / axis[2], axis[1] / axis[2], r])


def init_from_contours(contours, cams, K, gt_z_guess):
    """cheap data-driven init: back-project each contour's centroid to the z=gt_z_guess plane,
    average -> O; axis=+z; r from the median contour angular size * depth."""
    Kinv = np.linalg.inv(K); Os = []; rs = []
    for det, cam in zip(contours, cams):
        if det is None: continue
        c = det.mean(0)
        ray = Kinv @ np.array([c[0], c[1], 1.0]); ray /= np.linalg.norm(ray)
        dw = cam.R.T @ ray
        if abs(dw[2]) < 1e-6: continue
        t = (gt_z_guess - cam.C[2]) / dw[2]; Os.append(cam.C + t * dw)
        rpx = np.median(np.linalg.norm(det - c, axis=1))
        rs.append(rpx / K[0, 0] * (t))                    # angular radius * depth ~ metric radius
    O = np.mean(Os, 0) if Os else np.array([0, 0, gt_z_guess])
    return theta_from(O, [0, 0, 1.0], float(np.median(rs)) if rs else 3.0)
