"""Synthetic CT airway phantoms with KNOWN centerline/CSA/DCE/stenosis -- used to validate the CT
pipeline and the whole evaluation framework (a 'perfect' method must score ~0 error on these).
Voxelized so it exercises the real segmentation -> skeleton -> cross-section path."""
from __future__ import annotations
import numpy as np


def stenotic_tube(R_mm=6.0, throat_mm=3.0, length_mm=60.0, stenosis_mm=40.0, sigma_mm=5.0,
                  spacing_mm=0.5, bend_amp_mm=0.0, pad_mm=6.0):
    """A tube along +z with a Gaussian stenosis; radius r(z)=R-(R-throat)exp(-((z-zs)/sigma)^2/2).
    Optional lateral bend (bend_amp) to exercise a curved centerline. Returns:
      mask (uint8 volume, 1=airway), spacing (3,), and the analytic profile dict."""
    sp = np.array([spacing_mm] * 3)
    W = int((2 * (R_mm + pad_mm)) / sp[0]); H = W; D = int((length_mm + 2 * pad_mm) / sp[2])
    zc = (np.arange(D) * sp[2])                        # z in mm from 0
    z0 = pad_mm
    zz = zc - z0                                       # tube-local z (0..length)
    r_of_z = R_mm - (R_mm - throat_mm) * np.exp(-0.5 * ((zz - stenosis_mm) / sigma_mm) ** 2)
    cx = (W * sp[0]) / 2 + bend_amp_mm * np.sin(np.pi * np.clip(zz, 0, length_mm) / length_mm)
    cy = (H * sp[1]) / 2
    mask = np.zeros((W, H, D), np.uint8)
    xs = (np.arange(W) * sp[0])[:, None]; ys = (np.arange(H) * sp[1])[None, :]
    for k in range(D):
        if zz[k] < 0 or zz[k] > length_mm: continue
        rr = np.hypot(xs - cx[k], ys - cy)
        mask[:, :, k] = (rr <= r_of_z[k]).astype(np.uint8)
    inside = (zz >= 0) & (zz <= length_mm)
    prof = dict(z_mm=zc[inside], r_mm=r_of_z[inside], csa_mm2=np.pi * r_of_z[inside] ** 2,
                dce_mm=2 * r_of_z[inside], stenosis_z_mm=z0 + stenosis_mm, throat_dce_mm=2 * throat_mm,
                ref_dce_mm=2 * R_mm, centerline_mm=np.c_[cx[inside], np.full(inside.sum(), cy), zc[inside]])
    return mask, sp, prof
