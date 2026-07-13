"""Synthetic stenosis-THROAT phantom RENDERER. Ray-casts a surface-of-revolution airway with a
Gaussian throat, from known/fixed camera poses with known intrinsics, producing images in which the
throat aperture is a clean occluding contour (dark hole = through-throat) surrounded by a near-light-
shaded wall. Known GT: throat radius / CSA / DCE. Deterministic (fixed seeds). World units = mm.

The throat is a fixed 3-D circle (radius r_throat at z=stenosis_z). Cameras sit BEHIND it and image
it; the boundary of the dark through-throat region IS the throat circle's projection -> the occluding
contour the estimator fits.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    R: np.ndarray            # world->cam (3,3)
    C: np.ndarray            # camera centre (3,), mm


@dataclass
class Phantom:
    images: list             # list of HxW float32 images (0..1)
    cameras: list            # list[Camera]
    K: np.ndarray            # (3,3) intrinsics
    width: int
    height: int
    gt: dict                 # {throat_radius_mm, throat_center_mm, throat_axis, csa_mm2, dce_mm, ...}


def _render_cam(C, d, R_ref, r_throat, z_s, albedo):
    """Crisp aperture: cylinder wall (radius R_ref, z<z_s) + a diaphragm at z=z_s with a circular hole
    of radius r_throat. The occluding contour = the hole rim = EXACTLY the throat circle (radius
    r_throat at z_s), removing the smooth-silhouette ambiguity so the GT is unambiguous. Lit surfaces
    (cylinder wall / diaphragm face) are near-light shaded; through-hole rays are dark."""
    dz = d[:, 2]; fwd = dz > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        t_s = np.where(fwd, (z_s - C[2]) / dz, np.inf)
    Xs = C[None, :] + t_s[:, None] * d; rho_s = np.hypot(Xs[:, 0], Xs[:, 1])
    a = d[:, 0] ** 2 + d[:, 1] ** 2; b = 2 * (C[0] * d[:, 0] + C[1] * d[:, 1]); c = C[0] ** 2 + C[1] ** 2 - R_ref ** 2
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_cyl = np.where(disc > 0, (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a + 1e-12), np.inf)
    z_cyl = C[2] + t_cyl * dz
    cyl_before = fwd & (disc > 0) & (t_cyl > 1e-6) & (t_cyl < t_s) & (z_cyl >= 0) & (z_cyl <= z_s)
    through = fwd & (rho_s < r_throat) & (~cyl_before)
    lit = fwd & (~through)
    t_hit = np.where(cyl_before, t_cyl, t_s); X = C[None, :] + t_hit[:, None] * d
    n = np.where(cyl_before[:, None], -np.c_[X[:, 0], X[:, 1], np.zeros(len(X))] / R_ref,
                 np.tile([0, 0, -1.0], (len(X), 1)))
    v = C[None, :] - X; dd = np.linalg.norm(v, axis=1) + 1e-9; vh = v / dd[:, None]
    ndl = np.sum(n * vh, axis=1)
    return np.where(lit, albedo * np.maximum(0, ndl) / dd ** 2, 0.0)


def render(cfg, rng=None):
    p = cfg["phantom"]; K = np.array([[p["intrinsics"]["fx"], 0, p["intrinsics"]["cx"]],
                                      [0, p["intrinsics"]["fy"], p["intrinsics"]["cy"]], [0, 0, 1.0]])
    W, H = p["intrinsics"]["width"], p["intrinsics"]["height"]; Kinv = np.linalg.inv(K)
    rng = rng or np.random.default_rng(p["seed"])
    z_s = p["stenosis_z_mm"]; R_ref, r_th, sigma, L = p["R_ref_mm"], p["r_throat_mm"], p["sigma_mm"], p["length_mm"]
    # pixel ray directions in camera frame (fixed)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dc = (Kinv @ np.c_[uu.ravel(), vv.ravel(), np.ones(uu.size)].T).T
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    lat = p["cam_depth_mm"] * np.tan(np.radians(p["cam_cone_deg"]))
    cams, images = [], []
    for i in range(p["n_cameras"]):
        ph = 2 * np.pi * i / max(1, p["n_cameras"])
        C = np.array([lat * np.sin(ph) * rng.uniform(0.6, 1.0), lat * np.cos(0.7 * ph) * rng.uniform(0.6, 1.0),
                      z_s - p["cam_depth_mm"] + rng.uniform(-0.3, 0.3)])
        pitch = 0.03 * np.sin(ph); yaw = 0.03 * np.cos(ph)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R = Rx @ Ry                                       # world->cam (looks +z when R=I)
        d_world = (R.T @ dc.T).T
        B = _render_cam(C, d_world, R_ref, r_th, z_s, p["albedo"])
        lit = B > 0
        B = B / (np.nanpercentile(B[lit], 99) + 1e-9)         # normalize per-frame (mimics auto-gain)
        img = np.clip(B, 0, 1).reshape(H, W).astype(np.float32)
        cams.append(Camera(R=R, C=C)); images.append(img)
    gt = dict(throat_radius_mm=float(r_th), throat_center_mm=[0.0, 0.0, float(z_s)],
              throat_axis=[0.0, 0.0, 1.0], csa_mm2=float(np.pi * r_th ** 2), dce_mm=float(2 * r_th),
              R_ref_mm=float(R_ref), stenosis_z_mm=float(z_s))
    return Phantom(images=images, cameras=cams, K=K, width=W, height=H, gt=gt)
