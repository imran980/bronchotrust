"""Fit a statistical shape model (PCA) on the ATM22 correspondence.

Two flavors, selected by --use_surface:

  Centerline SSM (default): stacks each subject's (N, 3)
    corresponded_points (5 named landmarks + 4*K along-branch
    samples; produced by build_correspondence.py). Compact (~85 pts
    per subject), captures upper-airway shape.

  Surface SSM (--use_surface): stacks each subject's
    (V_template, 3) corresponded_surface_verts (TPS-warped template
    mesh in template frame; produced by register_surface.py). High-
    resolution (~30k pts per subject), captures full airway shape
    including distal segments.

Stores results under /metadata/ssm/ (centerline) or
/metadata/ssm_surface/ (surface). A runtime app loads only those
arrays — never the per-subject data.

Runtime use:
  shape = mean_shape + sum_k alpha_k * modes[k]
  alpha_k typically bounded by ±3 * sqrt(eigenvalues[k]).

Usage:
  python fit_ssm.py --corpus atm22_corpus.h5 --n_modes 20
  python fit_ssm.py --corpus atm22_corpus.h5 --var_target 0.95
  python fit_ssm.py --corpus atm22_corpus.h5 --use_surface --var_target 0.95
  python fit_ssm.py --corpus atm22_corpus.h5 --exclude_residual_above 20
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np

from atm22_corpus import Corpus


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n_modes", type=int, default=20)
    ap.add_argument("--var_target", type=float, default=None,
                    help="If set, pick n_modes to reach this cumulative "
                         "variance fraction (e.g. 0.95). Overrides "
                         "--n_modes.")
    ap.add_argument("--use_surface", action="store_true",
                    help="Fit on corresponded_surface_verts (V ~ 30k) "
                         "instead of corresponded_points (N ~ 85). "
                         "Requires register_surface.py has been run.")
    ap.add_argument("--exclude_residual_above", type=float, default=None,
                    help="Drop subjects whose correspondence residual "
                         "exceeds this many mm. With --use_surface, "
                         "this checks surface_registration_residual_mm; "
                         "without, it checks correspondence_residual_mm.")
    args = ap.parse_args()

    corpus = Corpus(args.corpus)
    pts, used = [], []
    for sid in corpus.subject_ids:
        s = corpus.load(sid)
        if args.use_surface:
            arr = s.corresponded_surface_verts
            resid = s.surface_registration_residual_mm
        else:
            arr = s.corresponded_points
            resid = s.correspondence_residual_mm
        if arr is None:
            continue
        if (args.exclude_residual_above is not None
                and resid is not None
                and resid > args.exclude_residual_above):
            continue
        pts.append(arr)
        used.append(sid)
    if not pts:
        raise SystemExit(
            "No subjects with the required correspondence. "
            "Run build_correspondence.py (or register_surface.py if "
            "--use_surface) first.")

    pts = np.stack(pts)            # (M, N, 3)
    M, N, _ = pts.shape
    print(f"PCA on {M} subjects, {N} points each "
          f"(state-vector dim = {N * 3})")

    X = pts.reshape(M, -1).astype(np.float64)
    mean = X.mean(axis=0)
    centered = X - mean

    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = S ** 2 / max(M - 1, 1)
    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var if total_var > 0 \
        else np.zeros_like(eigenvalues)

    if args.var_target is not None:
        n_modes = int(np.searchsorted(cum_var, args.var_target) + 1)
        n_modes = max(1, min(n_modes, len(S)))
    else:
        n_modes = min(args.n_modes, len(S))

    modes = Vt[:n_modes].reshape(n_modes, N, 3)
    eigenvalues = eigenvalues[:n_modes]
    cum_var = cum_var[:n_modes]
    mean_shape = mean.reshape(N, 3)

    print(f"\nKeeping {n_modes} modes; cumulative variance "
          f"{cum_var[-1] * 100:.1f}%")
    print(f"{'mode':>4s} {'var (mm^2)':>12s} {'std (mm)':>10s} "
          f"{'cum_var':>9s}")
    for i, (ev, cv) in enumerate(zip(eigenvalues, cum_var)):
        print(f"{i:>4d} {ev:>12.2f} {np.sqrt(ev):>10.2f} "
              f"{cv * 100:>8.1f}%")

    group_name = "ssm_surface" if args.use_surface else "ssm"
    with h5py.File(args.corpus, "a") as h5:
        if group_name in h5["metadata"]:
            del h5[f"metadata/{group_name}"]
        ssm = h5["metadata"].create_group(group_name)
        ssm.create_dataset("mean_shape",
                           data=mean_shape.astype(np.float32),
                           compression="gzip", compression_opts=6)
        ssm.create_dataset("modes", data=modes.astype(np.float32),
                           compression="gzip", compression_opts=6)
        ssm.create_dataset("eigenvalues",
                           data=eigenvalues.astype(np.float32))
        ssm.create_dataset("cumulative_var",
                           data=cum_var.astype(np.float32))
        ssm.attrs["n_subjects"] = M
        ssm.attrs["n_modes"] = n_modes
        ssm.attrs["n_points"] = N

    print(f"\nWrote /metadata/{group_name}/ to {args.corpus}")


if __name__ == "__main__":
    main()
