"""Shared helpers for the 2_V2 photometric-depth pivot audit. Frame sets around the accepted
proximal/narrowest and distal reference regions; specular + bezel masking; lumen-center finding;
pinned 2_V2 OPENCV intrinsics for back-projection. Scene/relative units only -- NO mm, NO GT."""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2

ROOT = Path("/home/mi3dr/projects/bronchotrust")
IMG = ROOT / "runs/batch4/2-V2/images"
OUT = ROOT / "photometric_pivot"
# proximal/narrowest (~875-945) vs distal reference (~985-1045) subglottic frame sets
PROX = [900, 910, 920, 930, 940]
DIST = [1000, 1010, 1020, 1030, 1040]
# pinned 2_V2 OPENCV intrinsics (from runs/batch4/2-V2/sparse/0)
import json, pycolmap
def intrinsics():
    cam = list(pycolmap.Reconstruction(str(ROOT / "runs/batch4/2-V2/sparse/0")).cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array([k1, k2, p1, p2]))


def load_gray(fr):
    bgr = cv2.imread(str(IMG / f"f{fr:05d}.png")); return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def fov_mask(g):
    """circular endoscope FOV (exclude the black bezel). Largest bright-ish connected region."""
    m = (g > 6).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return m.astype(bool), None
    i = 1 + int(np.argmax(st[1:, 4])); return (lab == i), cen[i]


def specular_mask(g, fov):
    """specular highlights: locally very bright pixels inside the FOV."""
    thr = max(200, np.percentile(g[fov], 99.3))
    sp = ((g >= thr) & fov).astype(np.uint8)
    return cv2.dilate(sp, np.ones((7, 7), np.uint8)).astype(bool)


def clean_brightness(bgr, g, fov, spec, gamma=1.0):
    """inpaint specular, zero the bezel, optional gamma; return float brightness in [0,1] within FOV."""
    gm = cv2.inpaint(g, spec.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
    gm = cv2.GaussianBlur(gm, (0, 0), 2)
    b = np.clip(gm / 255.0, 0, 1)
    if gamma != 1.0: b = np.power(b, gamma)
    b[~fov] = np.nan
    return b


def lumen_center(g, fov):
    """darkest interior blob centroid = lumen center (deepest visible)."""
    gg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 5); v = gg[fov]
    thr = np.percentile(v, 6); dark = ((gg < thr) & fov).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    if n < 2:
        ys, xs = np.where(fov); return np.array([xs.mean(), ys.mean()])
    i = 1 + int(np.argmax(st[1:, 4])); return cen[i]
