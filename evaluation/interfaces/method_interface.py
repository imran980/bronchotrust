"""THE method interface -- every reconstruction method (Barbour, COLMAP, a future optimizer,
LightNeuS, ...) exposes exactly this API so they all plug into the same evaluator.

    Input :  CalibratedVideo   (calibrated bronchoscopy video)
    Output:  ReconstructionResult(centerline, arclength, CSA(s), DCE(s), covariance, confidence)

The evaluation framework NEVER runs reconstruction. A method (or a thin read-only adapter over an
existing pipeline's saved outputs) produces a ReconstructionResult; the evaluator consumes it.
Units may be "mm" (metric) or "scene" (scale-free); registration to CT resolves the scale.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

VALID_UNITS = ("mm", "scene")


@dataclass(frozen=True)
class CalibratedVideo:
    """The ONLY input a method receives. Frames + calibration; poses optional (a method may or may
    not use externally-provided poses -- that is the method's choice, evaluation is agnostic)."""
    video_id: str
    frames_dir: Path                       # directory of extracted, calibrated frames
    intrinsics: dict                       # {model, width, height, params:[...]} e.g. OPENCV
    calib_id: str                          # which calibration session produced `intrinsics`
    poses: Optional[dict] = None           # optional {frame_index: 4x4 world<-cam} if the method wants them
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert Path(self.frames_dir).exists() or self.meta.get("synthetic"), f"frames_dir missing: {self.frames_dir}"
        assert "params" in self.intrinsics, "intrinsics must carry a params list"


@dataclass
class ReconstructionResult:
    """The ONLY output the evaluator consumes. Arrays are per-centerline-sample (length N).

    centerline : (N,3)  ordered 3-D points in the method's own frame/units
    arclength  : (N,)   monotonic arclength along the centerline (same units as centerline)
    csa        : (N,)   cross-sectional area per sample (units**2)
    dce        : (N,)   diameter of a circle of equal area = 2*sqrt(CSA/pi) (units)
    covariance : (N,)   variance of CSA per sample (units**4), OR (N,2,2) cov of [CSA, DCE]
    confidence : (N,)   per-sample confidence in [0,1] (e.g. coverage/estimator-agreement)
    units      : "mm" | "scene"      (scene = scale-free; registration recovers the scale)
    frame      : free string id of the coordinate frame
    runtime_s  : wall-clock seconds the method took (for the runtime metric)
    """
    method: str
    centerline: np.ndarray
    arclength: np.ndarray
    csa: np.ndarray
    dce: np.ndarray
    covariance: np.ndarray
    confidence: np.ndarray
    units: str
    frame: str = "method"
    runtime_s: float = float("nan")
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.centerline = np.asarray(self.centerline, float)
        for a in ("arclength", "csa", "dce", "covariance", "confidence"):
            setattr(self, a, np.asarray(getattr(self, a), float))
        self.validate()

    def validate(self):
        N = len(self.centerline)
        assert self.centerline.shape == (N, 3), "centerline must be (N,3)"
        assert self.units in VALID_UNITS, f"units must be one of {VALID_UNITS}"
        for a in ("arclength", "csa", "dce", "confidence"):
            assert getattr(self, a).shape[0] == N, f"{a} length must match centerline"
        assert self.covariance.shape[0] == N, "covariance length must match centerline"
        assert np.all(self.csa >= 0), "CSA must be non-negative"
        # DCE consistency (2*sqrt(CSA/pi)) within 1% where CSA>0
        good = self.csa > 1e-9
        if good.any():
            dce_pred = 2 * np.sqrt(self.csa[good] / np.pi)
            rel = np.abs(self.dce[good] - dce_pred) / np.maximum(dce_pred, 1e-9)
            assert np.nanmedian(rel) < 0.02, "DCE must equal 2*sqrt(CSA/pi) (median rel err <2%)"
        assert np.all((self.confidence >= 0) & (self.confidence <= 1)), "confidence in [0,1]"

    # convenience
    @property
    def n(self) -> int: return len(self.centerline)

    def csa_std(self) -> np.ndarray:
        c = self.covariance
        return np.sqrt(c if c.ndim == 1 else c[:, 0, 0])


class ReconstructionMethod(ABC):
    """Subclass this (or write an adapter) so a method plugs into the evaluator unchanged."""
    name: str = "unnamed"

    @abstractmethod
    def run(self, video: CalibratedVideo) -> ReconstructionResult:
        """Reconstruct + measure. MUST return a valid ReconstructionResult. The framework times it
        if the method does not set runtime_s itself."""
        raise NotImplementedError
