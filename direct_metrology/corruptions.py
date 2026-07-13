"""Controlled corruptions for stress-testing the direct-metrology prototype. Each acts at the correct
stage so it probes a distinct failure mode:

  specular_frac  : add bright specular blobs to IMAGES  (before detection)  -> tests specular inpaint
  albedo_var     : multiply IMAGES by a spatially-varying vascular texture   -> tests contour+photo robustness
  contour_noise  : jitter DETECTED contour points (px)                       -> tests observation noise
  pose_pert_mm   : perturb the CAMERAS handed to the estimator (not render)  -> tests pose uncertainty

Deterministic given a seeded rng. None of these touch the ground truth."""
from __future__ import annotations
import numpy as np
from .phantom import Camera


def corrupt_images(images, specular_frac=0.0, albedo_var=0.0, rng=None):
    """Return new images with optional specular highlights + spatially-varying albedo texture."""
    rng = rng or np.random.default_rng(0)
    out = []
    H, W = images[0].shape
    yy, xx = np.mgrid[0:H, 0:W]
    for im in images:
        g = im.copy()
        if albedo_var > 0:
            # smooth low-freq multiplicative texture + high-freq vascular stripes (Fourier energy off-axis)
            fx, fy = rng.uniform(0.5, 2.0, 2) * 2 * np.pi / W
            stripes = 1.0 + albedo_var * (0.5 * np.sin(fx * xx + fy * yy + rng.uniform(0, 6.28))
                                          + 0.5 * np.sin(2.3 * fx * xx - 1.7 * fy * yy))
            g = np.clip(g * stripes, 0, 1)
        if specular_frac > 0:
            nblob = max(1, int(specular_frac * 20))
            lit = g > 0.05
            ys, xs = np.where(lit)
            if len(xs):
                for _ in range(nblob):
                    k = rng.integers(len(xs)); cx, cy = xs[k], ys[k]
                    rad = rng.integers(3, 8)
                    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    g = np.maximum(g, np.exp(-d2 / (2 * rad ** 2)).astype(np.float32))
        out.append(np.clip(g, 0, 1).astype(np.float32))
    return out


def corrupt_contours(contours, noise_px=0.0, rng=None):
    """Add isotropic Gaussian jitter to each detected contour point."""
    if noise_px <= 0:
        return contours
    rng = rng or np.random.default_rng(0)
    out = []
    for c in contours:
        if c is None:
            out.append(None)
        else:
            out.append(c + rng.normal(0, noise_px, c.shape))
    return out


def corrupt_cameras(cams, pose_pert_mm=0.0, rot_pert_deg=0.0, rng=None):
    """Perturb the camera centres (and optionally rotation) HANDED TO THE ESTIMATOR, while the images
    were rendered from the true poses -> simulates pose uncertainty the estimator must absorb."""
    if pose_pert_mm <= 0 and rot_pert_deg <= 0:
        return cams
    rng = rng or np.random.default_rng(0)
    out = []
    for cam in cams:
        C = cam.C + rng.normal(0, pose_pert_mm, 3)
        R = cam.R
        if rot_pert_deg > 0:
            w = rng.normal(0, np.radians(rot_pert_deg), 3); th = np.linalg.norm(w) + 1e-12
            k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
            R = dR @ cam.R
        out.append(Camera(R=R, C=C))
    return out
