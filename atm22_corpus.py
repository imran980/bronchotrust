"""Read-only loader for the packed ATM22 corpus (atm22_corpus.h5).

All downstream pipeline steps (correspondence, PCA / SSM fitting,
graph matching, etc.) should depend only on this module, not on the
raw ATM22 dataset or the per-subject preprocessing intermediates.

Typical use:
    from atm22_corpus import Corpus
    corpus = Corpus("atm22_corpus.h5")
    print(corpus.subject_ids)
    s = corpus.load("ATM_001")
    s.surface_verts        # (V, 3) float32 in mm
    s.centerline_nodes     # (N, 3) float32 in mm
    s.is_bifurcation       # (N,) bool
    bif_pts = s.bifurcation_points()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import h5py
import numpy as np


@dataclass
class Subject:
    subject_id: str
    surface_verts: np.ndarray       # (V, 3) float32, mm
    surface_faces: np.ndarray       # (F, 3) int32
    centerline_nodes: np.ndarray    # (N, 3) float32, mm
    centerline_edges: np.ndarray    # (E, 2) int32
    centerline_radii: np.ndarray    # (N,)   float32, mm
    is_bifurcation: np.ndarray      # (N,)   bool
    affine: np.ndarray              # (4, 4) float64
    voxel_spacing_mm: np.ndarray    # (3,)   float32
    label_node_idx: Optional[np.ndarray] = None   # (L,) int32
    label_name: Optional[List[str]] = None        # length L
    qa_flags: List[str] = field(default_factory=list)
    corresponded_points: Optional[np.ndarray] = None   # (N, 3) float32
    correspondence_residual_mm: Optional[float] = None
    corresponded_surface_verts: Optional[np.ndarray] = None  # (V, 3)
    surface_registration_residual_mm: Optional[float] = None

    def bifurcation_points(self) -> np.ndarray:
        return self.centerline_nodes[self.is_bifurcation]

    def endpoint_indices(self) -> np.ndarray:
        deg = np.bincount(self.centerline_edges.ravel(),
                          minlength=len(self.centerline_nodes))
        return np.where(deg == 1)[0]

    def labeled_points(self) -> dict:
        """{label_name: (3,) mm coord} for whatever labels exist."""
        if self.label_node_idx is None or self.label_name is None:
            return {}
        return {n: self.centerline_nodes[int(i)]
                for i, n in zip(self.label_node_idx, self.label_name)}


class Corpus:
    """Lazy, read-only access to atm22_corpus.h5."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as h5:
            ids = h5["metadata/subject_ids"][:]
            self.subject_ids: List[str] = [
                s.decode() if isinstance(s, bytes) else str(s) for s in ids]
            self.metadata = dict(h5["metadata"].attrs)

    def __len__(self):
        return len(self.subject_ids)

    def __contains__(self, sid):
        return sid in self.subject_ids

    @property
    def template_subject_id(self) -> Optional[str]:
        with h5py.File(self.h5_path, "r") as h5:
            if "template_subject_id" not in h5["metadata"]:
                return None
            v = h5["metadata/template_subject_id"][()]
            return v.decode() if isinstance(v, bytes) else str(v)

    def load(self, subject_id: str) -> Subject:
        def _strs(ds):
            return [s.decode() if isinstance(s, bytes) else str(s)
                    for s in ds[:]]
        with h5py.File(self.h5_path, "r") as h5:
            g = h5[f"subjects/{subject_id}"]
            label_idx = g["label_node_idx"][:] if "label_node_idx" in g else None
            label_name = _strs(g["label_name"]) if "label_name" in g else None
            qa = _strs(g["qa_flags"]) if "qa_flags" in g else []
            cp = g["corresponded_points"][:] if "corresponded_points" in g else None
            cr = (float(g["correspondence_residual_mm"][()])
                  if "correspondence_residual_mm" in g else None)
            csv = (g["corresponded_surface_verts"][:]
                   if "corresponded_surface_verts" in g else None)
            srr = (float(g["surface_registration_residual_mm"][()])
                   if "surface_registration_residual_mm" in g else None)
            return Subject(
                subject_id=subject_id,
                surface_verts=g["surface_verts"][:],
                surface_faces=g["surface_faces"][:],
                centerline_nodes=g["centerline_nodes"][:],
                centerline_edges=g["centerline_edges"][:],
                centerline_radii=g["centerline_radii"][:],
                is_bifurcation=g["is_bifurcation"][:],
                affine=g["affine"][:],
                voxel_spacing_mm=g["voxel_spacing_mm"][:],
                label_node_idx=label_idx,
                label_name=label_name,
                qa_flags=qa,
                corresponded_points=cp,
                correspondence_residual_mm=cr,
                corresponded_surface_verts=csv,
                surface_registration_residual_mm=srr,
            )

    @property
    def surface_template_faces(self) -> Optional[np.ndarray]:
        """Triangle faces of the registered surface (same connectivity
        for every subject, since the template's topology is reused)."""
        with h5py.File(self.h5_path, "r") as h5:
            if "surface_template_faces" not in h5["metadata"]:
                return None
            return h5["metadata/surface_template_faces"][:]

    @property
    def point_labels(self) -> Optional[List[str]]:
        with h5py.File(self.h5_path, "r") as h5:
            if "point_labels" not in h5["metadata"]:
                return None
            return [s.decode() if isinstance(s, bytes) else str(s)
                    for s in h5["metadata/point_labels"][:]]

    def load_ssm(self, surface: bool = False) -> Optional[dict]:
        """Return the fitted SSM artifacts as a dict, or None if the
        chosen variant hasn't been fit yet.

        surface=False -> centerline SSM (~85 pts; landmarks + along-
        branch samples). surface=True -> surface SSM (~30k pts;
        warped template mesh).
        """
        group = "ssm_surface" if surface else "ssm"
        with h5py.File(self.h5_path, "r") as h5:
            if group not in h5["metadata"]:
                return None
            ssm = h5[f"metadata/{group}"]
            return {
                "mean_shape": ssm["mean_shape"][:],
                "modes": ssm["modes"][:],
                "eigenvalues": ssm["eigenvalues"][:],
                "cumulative_var": ssm["cumulative_var"][:],
                "n_subjects": int(ssm.attrs["n_subjects"]),
            }

    def iter_subjects(self):
        for sid in self.subject_ids:
            yield self.load(sid)
