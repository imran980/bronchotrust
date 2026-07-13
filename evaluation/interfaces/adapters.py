"""Read-only adapters that wrap an EXISTING method's saved output into a ReconstructionResult.
These do NOT run or modify any reconstruction pipeline -- they only READ artifacts a method already
produced (a per-slice CSA/DCE profile, or a point cloud + a provided centerline). This is how
Barbour / COLMAP / a future optimizer / LightNeuS plug into the same evaluator without being touched.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from .method_interface import ReconstructionResult


def from_profile_arrays(method, centerline, csa, units, arclength=None, covariance=None,
                        confidence=None, runtime_s=float("nan"), frame="method", meta=None):
    """Universal adapter: any method that can emit (centerline, CSA) plugs in here. DCE is derived."""
    centerline = np.asarray(centerline, float); csa = np.asarray(csa, float)
    N = len(centerline)
    if arclength is None:
        arclength = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))])
    dce = 2 * np.sqrt(np.clip(csa, 0, None) / np.pi)
    if covariance is None: covariance = np.full(N, np.nan)
    if confidence is None: confidence = np.ones(N)
    return ReconstructionResult(method=method, centerline=centerline, arclength=np.asarray(arclength, float),
                                csa=csa, dce=dce, covariance=np.asarray(covariance, float),
                                confidence=np.asarray(confidence, float), units=units, frame=frame,
                                runtime_s=runtime_s, meta=meta or {})


def from_saved_profile_json(method, path, units, keys=None):
    """Read a saved per-slice profile JSON (the shape our COLMAP centerline-slice / obstruction runs
    emit) and wrap it read-only. `keys` maps json field names; defaults try common ones. The point
    cloud / reconstruction is NOT touched -- only its measured profile is read."""
    d = json.loads(Path(path).read_text())
    keys = keys or {}
    def pick(*names):
        for n in names:
            if n in d: return d[n]
        return None
    centerline = pick(keys.get("centerline", "centerline"), "centerline_xyz")
    csa = pick(keys.get("csa", "csa"), "CSA", "csa_profile")
    conf = pick(keys.get("confidence", "coverage"), "confidence")
    cov = pick(keys.get("covariance", "csa_var"), "variance")
    assert centerline is not None and csa is not None, f"profile json missing centerline/csa: {list(d)[:8]}"
    return from_profile_arrays(method, np.asarray(centerline, float), np.asarray(csa, float), units,
                               covariance=(np.asarray(cov, float) if cov is not None else None),
                               confidence=(np.asarray(conf, float) if conf is not None else None),
                               runtime_s=float(d.get("runtime_s", float("nan"))),
                               meta=dict(source=str(path)))


def landmark_band_method(method, landmark_dce_mm, units="mm", meta=None):
    """A minimal result for LANDMARK-BAND mode: the method only needs to report DCE at named CT
    landmarks (e.g. {'trachea':5.7,'right_mainstem':4.5}). Placeholder profile arrays keep the API
    uniform; the evaluator uses meta['landmark_dce_mm'] in band mode."""
    m = dict(landmark_dce_mm=dict(landmark_dce_mm)); m.update(meta or {})
    n = max(2, len(landmark_dce_mm))
    dce = np.array(list(landmark_dce_mm.values()) + [0.0] * (n - len(landmark_dce_mm)))[:n]
    csa = np.pi * (dce / 2) ** 2
    return ReconstructionResult(method=method, centerline=np.zeros((n, 3)), arclength=np.arange(n, dtype=float),
                                csa=csa, dce=dce, covariance=np.full(n, np.nan), confidence=np.ones(n),
                                units=units, meta=m)
