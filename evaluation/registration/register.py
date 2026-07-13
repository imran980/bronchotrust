"""CT <-> bronchoscopy registration. A method's reconstruction (centerline in its own frame, mm OR
scale-free scene units) is aligned to the CT ground truth by a Sim(3) (scale s, rotation R, transl t).
The CT is the ONLY metric anchor: a scene-units method's absolute scale is recovered here as `s`.
Also: slice correspondence (method arclength -> CT arclength) and first-order uncertainty propagation
of CSA/DCE through the Sim(3). Automatic (no manual tuning): centerline ICP-with-scale, plus optional
named-landmark anchoring."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev


@dataclass
class Sim3:
    s: float
    R: np.ndarray                      # (3,3)
    t: np.ndarray                      # (3,)
    scale_var: float = 0.0             # variance of the estimated scale s (registration uncertainty)
    rmse_mm: float = float("nan")      # residual after alignment

    def apply(self, X):
        X = np.atleast_2d(X); return (self.s * (self.R @ X.T).T) + self.t[None, :]

    def inverse(self):
        Ri = self.R.T; return Sim3(1.0 / self.s, Ri, -(1.0 / self.s) * (Ri @ self.t))


def umeyama(src, dst, with_scale=True):
    """least-squares similarity mapping src->dst (Umeyama 1991)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0); Sc, Dc = src - mu_s, dst - mu_d
    Sigma = (Dc.T @ Sc) / len(src); U, Dvals, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = (Dvals * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    resid = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
    return Sim3(float(s), R, t, rmse_mm=float(np.sqrt(np.mean(resid ** 2))))


def _resample(C, n):
    k = min(3, len(C) - 1); tck, _ = splprep(C.T, s=0, k=k); u = np.linspace(0, 1, n)
    return np.array(splev(u, tck)).T


def register(source_C, target_C, source_landmarks=None, target_landmarks=None, with_scale=True, n=120, iters=6):
    """align source (method) centerline to target (CT) centerline.
    If named landmark correspondences are given, anchor on them; otherwise arclength-resample both and
    ICP-with-scale (trying both endpoint orientations, keeping the lower residual)."""
    if source_landmarks and target_landmarks:
        keys = [k for k in source_landmarks if k in target_landmarks]
        if len(keys) >= 3:
            return umeyama(np.array([source_landmarks[k] for k in keys]),
                           np.array([target_landmarks[k] for k in keys]), with_scale)
    best = None
    for flip in (False, True):
        S0 = _resample(source_C[::-1] if flip else source_C, n); D = _resample(target_C, n)
        T = umeyama(S0, D, with_scale)                 # arclength-corresponded init
        for _ in range(iters):                         # ICP refine on nearest points
            Sw = T.apply(S0); j = cKDTree(D).query(Sw)[1]; T = umeyama(S0, D[j], with_scale)
        if best is None or T.rmse_mm < best.rmse_mm: best = T
    # scale uncertainty: bootstrap the correspondence
    S0 = _resample(source_C, n); D = _resample(target_C, n); rng = np.random.default_rng(0); ss = []
    Sw = best.apply(S0); j = cKDTree(D).query(Sw)[1]
    for _ in range(30):
        idx = rng.integers(0, n, n); ss.append(umeyama(S0[idx], D[j][idx], with_scale).s)
    best.scale_var = float(np.var(ss))
    return best


def slice_correspondence(source_C_registered, source_arclen, target_C, target_arclen):
    """for each registered source slice, the CT arclength of the nearest CT centerline point."""
    j = cKDTree(target_C).query(source_C_registered)[1]
    return target_arclen[j], j


def propagate_csa(csa_scene, csa_var_scene, T: Sim3):
    """CSA/DCE from method units to mm through the Sim(3) scale, with first-order variance.
       CSA_mm = s^2 CSA ;  Var = s^4 Var(CSA) + (2 s CSA)^2 Var(s)   (delta method)."""
    s = T.s; csa_mm = s ** 2 * csa_scene
    var_mm = (s ** 4) * np.asarray(csa_var_scene) + (2 * s * np.asarray(csa_scene)) ** 2 * T.scale_var
    dce_mm = 2 * np.sqrt(np.clip(csa_mm, 0, None) / np.pi)
    # Var(DCE): DCE = 2 sqrt(CSA/pi) -> dDCE/dCSA = 1/sqrt(pi CSA); Var(DCE)=Var(CSA)/(pi CSA)
    with np.errstate(divide="ignore", invalid="ignore"):
        dce_var = np.where(csa_mm > 1e-9, var_mm / (np.pi * csa_mm), np.nan)
    return csa_mm, var_mm, dce_mm, dce_var
