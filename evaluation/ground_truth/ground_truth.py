"""Immutable CT-derived ground truth. Two representations, both supported because our data has both:

  (A) FULL PROFILE  -- CT centerline + CSA(s) + DCE(s) along arclength (when a full airway
                       segmentation is available). Landmarks reference arclength positions.
  (B) LANDMARK BANDS -- per anatomical landmark, a DCE/CSA *band* [lo,hi] across the 4-D (dynamic)
                       CT phases (PEEP states). This is what the same-day 4-D CT actually gives
                       (see runs/ct_validation_16v1/ct_reference.json): the airway MOVES with
                       breathing, so ground truth is a RANGE, not a point.

Once built, a GroundTruth is FROZEN and content-hashed. `assert_immutable()` fails if anyone edits
the arrays after creation -- "these become immutable ground truth."
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib, json
import numpy as np


@dataclass(frozen=True)
class Landmark:
    name: str                       # 'carina', 'cricoid', 'stenosis', 'trachea', 'right_mainstem', ...
    arclength_mm: float = float("nan")     # position along the CT centerline (profile mode)
    position_mm: tuple = ()                # 3-D CT-frame coordinate, if known
    dce_band_mm: tuple = ()                # (lo,hi) diameter band across 4-D phases (band mode)
    csa_band_mm2: tuple = ()               # (lo,hi) CSA band across 4-D phases
    band_source: str = ""                  # e.g. 'PEEP5/7/8 dynamic 4D CT'


@dataclass(frozen=True)
class GroundTruth:
    ct_id: str
    mode: str                              # 'profile' | 'landmark_bands'
    units: str = "mm"
    # profile mode
    centerline_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    arclength_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    csa_mm2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dce_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    csa_band_mm2: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # per-sample (lo,hi) if dynamic
    voxel_spacing_mm: tuple = ()
    # both modes
    landmarks: tuple = ()                   # tuple[Landmark]
    provenance: dict = field(default_factory=dict)
    _checksum: str = ""

    # ---- construction helpers (compute checksum, then freeze) ----
    @staticmethod
    def from_profile(ct_id, centerline_mm, arclength_mm, csa_mm2, landmarks=(), csa_band_mm2=None,
                     voxel_spacing_mm=(), provenance=None):
        centerline_mm = np.asarray(centerline_mm, float); arclength_mm = np.asarray(arclength_mm, float)
        csa_mm2 = np.asarray(csa_mm2, float); dce_mm = 2 * np.sqrt(np.clip(csa_mm2, 0, None) / np.pi)
        if csa_band_mm2 is None: csa_band_mm2 = np.zeros((0, 2))
        gt = GroundTruth(ct_id=ct_id, mode="profile", centerline_mm=centerline_mm, arclength_mm=arclength_mm,
                         csa_mm2=csa_mm2, dce_mm=dce_mm, csa_band_mm2=np.asarray(csa_band_mm2, float),
                         voxel_spacing_mm=tuple(voxel_spacing_mm), landmarks=tuple(landmarks),
                         provenance=provenance or {})
        object.__setattr__(gt, "_checksum", gt._compute_checksum())
        return gt

    @staticmethod
    def from_landmark_bands(ct_id, landmarks, provenance=None):
        gt = GroundTruth(ct_id=ct_id, mode="landmark_bands", landmarks=tuple(landmarks), provenance=provenance or {})
        object.__setattr__(gt, "_checksum", gt._compute_checksum())
        return gt

    # ---- immutability ----
    def _compute_checksum(self) -> str:
        h = hashlib.sha256()
        h.update(self.ct_id.encode()); h.update(self.mode.encode())
        for a in (self.centerline_mm, self.arclength_mm, self.csa_mm2, self.dce_mm, self.csa_band_mm2):
            h.update(np.ascontiguousarray(a, float).tobytes())
        h.update(json.dumps([asdict(l) for l in self.landmarks], sort_keys=True, default=list).encode())
        return h.hexdigest()[:16]

    def assert_immutable(self):
        assert self._checksum and self._checksum == self._compute_checksum(), \
            "GROUND TRUTH MUTATED after freezing -- this must never happen."

    # ---- lookups used by the evaluator ----
    def landmark(self, name):
        for l in self.landmarks:
            if l.name == name: return l
        return None

    def stenosis(self):
        """the reference stenosis landmark (min-CSA), or the explicit 'stenosis' landmark."""
        s = self.landmark("stenosis")
        if s is not None: return s
        if self.mode == "profile" and len(self.csa_mm2):
            i = int(np.argmin(self.csa_mm2))
            return Landmark("stenosis", arclength_mm=float(self.arclength_mm[i]),
                            csa_band_mm2=(float(self.csa_mm2[i]), float(self.csa_mm2[i])),
                            dce_band_mm=(float(self.dce_mm[i]), float(self.dce_mm[i])))
        return None

    # ---- IO (JSON + npz) ----
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        meta = dict(ct_id=self.ct_id, mode=self.mode, units=self.units, voxel_spacing_mm=list(self.voxel_spacing_mm),
                    landmarks=[asdict(l) for l in self.landmarks], provenance=self.provenance, checksum=self._checksum)
        (path / "ground_truth.json").write_text(json.dumps(meta, indent=2, default=list))
        np.savez(path / "ground_truth.npz", centerline_mm=self.centerline_mm, arclength_mm=self.arclength_mm,
                 csa_mm2=self.csa_mm2, dce_mm=self.dce_mm, csa_band_mm2=self.csa_band_mm2)
        return path

    @staticmethod
    def load(path):
        path = Path(path); meta = json.loads((path / "ground_truth.json").read_text())
        z = np.load(path / "ground_truth.npz"); lms = tuple(Landmark(**{k: (tuple(v) if isinstance(v, list) else v)
                                                                        for k, v in l.items()}) for l in meta["landmarks"])
        gt = GroundTruth(ct_id=meta["ct_id"], mode=meta["mode"], units=meta["units"],
                         centerline_mm=z["centerline_mm"], arclength_mm=z["arclength_mm"], csa_mm2=z["csa_mm2"],
                         dce_mm=z["dce_mm"], csa_band_mm2=z["csa_band_mm2"], voxel_spacing_mm=tuple(meta["voxel_spacing_mm"]),
                         landmarks=lms, provenance=meta["provenance"], _checksum=meta["checksum"])
        gt.assert_immutable()
        return gt
