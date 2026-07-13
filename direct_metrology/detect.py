"""Lumen occluding-contour detection from a rendered phantom image. The throat aperture is the dark
central region; its external boundary is the throat circle's projection. Returns 2-D contour points.
Robust to specular masks (bright spots inpainted) and additive contour noise (added in corruptions)."""
from __future__ import annotations
import numpy as np, cv2


def _subpixel_refine(g, pts, ctr):
    """Refine each coarse boundary point to the HALF-MAX intensity crossing along the outward radial.
    For a symmetric blur the 50%-crossing is the unbiased edge location, removing the systematic
    fixed-threshold bias (a low threshold sits inside the true rim). Returns (Kx2) float points."""
    out = []
    for x, y in pts:
        dvec = np.array([x - ctr[0], y - ctr[1]], float); L = np.hypot(*dvec)
        if L < 1e-6:
            out.append([x, y]); continue
        dvec /= L
        ss = np.linspace(-3.5, 3.5, 29)
        xs = (x + dvec[0] * ss).astype(np.float32).reshape(-1, 1)
        ys = (y + dvec[1] * ss).astype(np.float32).reshape(-1, 1)
        vals = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        lo = float(vals[:6].min()); hi = float(vals[-6:].max()); mid = 0.5 * (lo + hi)
        if hi - lo < 1e-3:
            out.append([x, y]); continue
        above = vals >= mid
        if not above.any() or above.all():
            out.append([x, y]); continue
        idx = int(np.argmax(above))
        if idx == 0:
            out.append([x, y]); continue
        v0, v1 = vals[idx - 1], vals[idx]; frac = (mid - v0) / (v1 - v0 + 1e-9)
        s = ss[idx - 1] + frac * (ss[idx] - ss[idx - 1])
        out.append([x + dvec[0] * s, y + dvec[1] * s])
    return np.asarray(out, float)


def detect_contour(img01, center_hint=None, dark_thresh=0.06, min_area_frac=0.005, subpixel=True):
    """img01: HxW float in [0,1]. Returns (Kx2) contour points (x,y) of the central dark aperture,
    or None if not found."""
    H, W = img01.shape
    g = cv2.GaussianBlur(img01.astype(np.float32), (0, 0), 1.5)
    # inpaint saturated specular pixels so they don't corrupt the dark-region detection
    spec = (g > 0.98).astype(np.uint8)
    if spec.any():
        g = cv2.inpaint((g * 255).astype(np.uint8), cv2.dilate(spec, np.ones((5, 5), np.uint8)), 4, cv2.INPAINT_TELEA).astype(np.float32) / 255
    dark = (g < dark_thresh).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    if n < 2:
        return None
    cx0, cy0 = center_hint if center_hint is not None else (W / 2, H / 2)
    best, bd = None, 1e18
    for i in range(1, n):
        if st[i, 4] < min_area_frac * H * W:
            continue
        d = np.hypot(cen[i, 0] - cx0, cen[i, 1] - cy0)
        if d < bd:
            bd, best = d, i
    if best is None:
        return None
    m = (lab == best).astype(np.uint8)
    # reject apertures touching the image border (open / clipped)
    ys, xs = np.where(m)
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None
    pts = c[:, 0, :].astype(np.float64)
    if subpixel:
        pts = _subpixel_refine(g, pts, pts.mean(0))
    return pts


def detect_all(images, center_hint=None):
    return [detect_contour(im, center_hint) for im in images]
