"""CT -> immutable ground truth. AUTOMATIC (no manual tuning): read a CT airway mask (or segment
one), extract the centerline, sample perpendicular cross-sections for CSA(s)/DCE(s), detect stenosis
landmarks. Also a loader for the 4-D dynamic-CT landmark *bands* (ct_reference.json).

Readers: NIfTI (nibabel) or DICOM series (pydicom) or a numpy mask. Segmentation: air-HU region-grow
from a seed (only when given a raw volume; a provided lumen mask is used directly)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from ..ground_truth.ground_truth import GroundTruth, Landmark


# ---------------- readers ----------------
def load_volume(path):
    """-> (volume float32 HU, spacing (3,) mm). Supports .nii/.nii.gz, a DICOM dir, or .npz(mask,spacing)."""
    path = Path(path)
    if path.suffix in (".nii", ".gz") or ".nii" in path.suffixes:
        import nibabel as nib
        img = nib.load(str(path)); vol = np.asarray(img.dataobj, float)
        sp = np.abs(np.diag(img.affine)[:3]); return vol, sp
    if path.is_dir():
        import pydicom
        files = sorted(path.glob("*.dcm"))
        sl = [pydicom.dcmread(str(f)) for f in files]
        sl.sort(key=lambda d: float(getattr(d, "ImagePositionPatient", [0, 0, 0])[2]))
        vol = np.stack([s.pixel_array.astype(float) * float(getattr(s, "RescaleSlope", 1)) +
                        float(getattr(s, "RescaleIntercept", 0)) for s in sl], axis=-1)
        ps = [float(x) for x in sl[0].PixelSpacing]; dz = abs(float(sl[1].ImagePositionPatient[2]) - float(sl[0].ImagePositionPatient[2])) if len(sl) > 1 else 1.0
        return vol, np.array([ps[0], ps[1], dz])
    if path.suffix == ".npz":
        z = np.load(path); return z["mask"].astype(float), z["spacing"]
    raise ValueError(f"unsupported CT input: {path}")


def segment_airway(volume, spacing, seed_ijk, hu_air=-500.0):
    """air-HU (< hu_air) connected component containing the seed voxel = airway lumen mask."""
    air = volume < hu_air
    lab, n = ndimage.label(air)
    s = lab[tuple(int(v) for v in seed_ijk)]
    assert s != 0, "seed is not in air; check seed_ijk / hu threshold"
    return (lab == s).astype(np.uint8)


# ---------------- centerline ----------------
def _order_skeleton(skel_vox):
    """order a (single-branch) skeleton's voxels into a path: find an endpoint, walk nearest unused."""
    pts = np.argwhere(skel_vox > 0)
    if len(pts) < 4: return pts
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    deg = np.array([len(tree.query_ball_point(p, np.sqrt(3) + 1e-6)) - 1 for p in pts])
    ends = np.where(deg <= 1)[0]
    start = ends[0] if len(ends) else 0
    order = [start]; used = {start}; cur = start
    for _ in range(len(pts) - 1):
        nb = tree.query_ball_point(pts[cur], np.sqrt(3) + 1e-6)
        nxt = [j for j in nb if j not in used]
        if not nxt: break
        cur = min(nxt, key=lambda j: np.linalg.norm(pts[j] - pts[cur])); order.append(cur); used.add(cur)
    return pts[order]


def centerline_from_mask(mask, spacing, n_samples=200, smooth=None):
    """skeletonize -> order -> smoothing spline -> resample; returns centerline in mm + tangents."""
    skel = skeletonize(mask > 0)
    order_vox = _order_skeleton(skel)
    pts_mm = order_vox * spacing[None, :]
    if len(pts_mm) < 6:                                 # fallback: slab-centroid along principal axis
        pts_mm = _slab_centroid_axis(mask, spacing)
    k = min(3, len(pts_mm) - 1); s = (len(pts_mm) * 0.5) if smooth is None else smooth
    tck, _ = splprep(pts_mm.T, s=s, k=k)
    u = np.linspace(0, 1, n_samples); C = np.array(splev(u, tck)).T
    T = np.array(splev(u, tck, der=1)).T; T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    arclen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    return C, T, arclen


def _slab_centroid_axis(mask, spacing):
    P = np.argwhere(mask > 0) * spacing[None, :]
    c = P.mean(0); _, _, Vt = np.linalg.svd(P - c, full_matrices=False); a = Vt[0]
    u = (P - c) @ a; edges = np.linspace(u.min(), u.max(), 40); out = []
    for i in range(len(edges) - 1):
        m = (u >= edges[i]) & (u < edges[i + 1])
        if m.sum() > 5: out.append(P[m].mean(0))
    return np.array(out)


# ---------------- cross-sections ----------------
def cross_section_profile(mask, spacing, C, T, half_extent_mm=None, grid_mm=0.25):
    """at each centerline sample, sample the perpendicular plane, interpolate the mask, area =
    in-lumen fraction x cell area. Returns CSA(s) (mm^2) and DCE(s) (mm)."""
    R_guess = np.sqrt((mask.sum() * np.prod(spacing)) / (np.pi * max(1e-6, np.ptp((np.argwhere(mask > 0) * spacing[None, :])[:, 2]))))
    ext = half_extent_mm if half_extent_mm else max(4.0, 3 * R_guess)
    g = np.arange(-ext, ext + grid_mm, grid_mm); GX, GY = np.meshgrid(g, g); cell = grid_mm ** 2
    inv = 1.0 / spacing
    csa = np.zeros(len(C))
    for i, (p, t) in enumerate(zip(C, T)):
        e1 = np.cross(t, [0, 0, 1.] if abs(t[2]) < 0.9 else [1., 0, 0]); e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(t, e1)
        X = p[None, None, :] + GX[..., None] * e1 + GY[..., None] * e2       # (G,G,3) mm
        vox = X * inv[None, None, :]
        val = ndimage.map_coordinates(mask.astype(float), [vox[..., 0].ravel(), vox[..., 1].ravel(), vox[..., 2].ravel()],
                                      order=1, mode="constant").reshape(GX.shape)
        # keep only the connected in-lumen blob around the centre (avoid neighbouring lumens)
        binm = val > 0.5
        lab, n = ndimage.label(binm); cc = lab[GX.shape[0] // 2, GX.shape[1] // 2]
        area = (lab == cc).sum() * cell if cc != 0 else binm.sum() * cell
        csa[i] = area
    dce = 2 * np.sqrt(np.clip(csa, 0, None) / np.pi)
    return csa, dce


# ---------------- landmarks ----------------
def detect_stenosis(arclen, csa, ref_frac=0.9):
    """stenosis = global min CSA; reference = a wide slice (percentile) for % obstruction."""
    i = int(np.argmin(csa)); ref = float(np.percentile(csa, 100 * ref_frac))
    return Landmark("stenosis", arclength_mm=float(arclen[i]),
                    csa_band_mm2=(float(csa[i]), float(csa[i])), dce_band_mm=(float(2 * np.sqrt(csa[i] / np.pi)),) * 2), ref


# ---------------- top-level ----------------
def process_ct_mask(ct_id, mask, spacing, provenance=None):
    """AUTOMATIC full-profile ground truth from a lumen mask."""
    C, T, arclen = centerline_from_mask(mask, spacing)
    csa, dce = cross_section_profile(mask, spacing, C, T)
    lm, ref_csa = detect_stenosis(arclen, csa)
    gt = GroundTruth.from_profile(ct_id, C, arclen, csa, landmarks=(lm,), voxel_spacing_mm=tuple(spacing),
                                  provenance=dict(source="process_ct_mask", ref_csa_mm2=ref_csa, **(provenance or {})))
    return gt


def ground_truth_from_bands(ct_id, ct_reference_json, provenance=None):
    """LANDMARK-BAND ground truth from a 4-D dynamic-CT reference (per-landmark DCE bands across
    PEEP phases), e.g. runs/ct_validation_16v1/ct_reference.json."""
    ref = json.loads(Path(ct_reference_json).read_text()) if isinstance(ct_reference_json, (str, Path)) else ct_reference_json
    lms = []
    for name, d in ref.items():
        if not isinstance(d, dict) or "overall_range" not in d: continue
        lo, hi = d["overall_range"]; band = d.get("band_mm", 0.0)
        lms.append(Landmark(name, dce_band_mm=(lo - band, hi + band),
                            csa_band_mm2=(np.pi * (lo / 2) ** 2, np.pi * (hi / 2) ** 2),
                            band_source=f"dynamic 4D CT phases; band ±{band}mm"))
    return GroundTruth.from_landmark_bands(ct_id, tuple(lms), provenance=dict(source=str(ct_reference_json), **(provenance or {})))
