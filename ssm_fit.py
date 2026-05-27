"""
Fit the ATM22 PCA surface SSM to a noisy point cloud of fragmented
per-frame DUSt3R-foe observations.

Why this exists: dust3r-foe gives coherent camera poses but fragmented
per-frame pointmaps that disagree on absolute geometry voxel-by-voxel
(TSDF fusion confirmed they don't average to anything coherent). The
underlying anatomy IS a tube with shape that lives in a low-dim PCA
span (ATM22 surface SSM, 37 modes, 147 k vertices). So instead of
trying to fuse the noisy patches into a surface, we constrain the
surface to the PCA prior and let the patches act as observations:

    minimize  ||obs - (s * R @ S(alpha) + t)||^2 + lambda * ||alpha / sigma||^2
        over  alpha  (PCA coefficients, shape (K,))
              R, t, s  (similarity transform, world frame)

    where  S(alpha) = mean_shape + sum_k alpha_k * mode_k   (V x 3)

Output is a single, anatomically valid, metric tube. Regions where
observations strongly disagree with the SSM fit are exactly where
clinical anomalies live (severe stenosis = anatomy outside the PCA
span), and the per-vertex residual gives the uncertainty signal.

Solved by alternating updates:
    1. correspondence: nearest SSM vertex per observation (KDTree)
    2. similarity (R, t, s) update via Umeyama on corresponded pairs
    3. alpha update: linear LSQ in {R, t, s} fixed
Repeat until convergence (residual change < tol).

CLI:
    python ssm_fit.py \\
        --corpus atm22_corpus.h5 \\
        --recon_dir runs/phase1_5V1/recon_dust3r_foe \\
        --out_dir  runs/phase1_5V1/recon_dust3r_foe/ssm_fit \\
        --max_iter 20 --lambda_alpha 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Closed-form similarity (s, R, t) so that dst ~ s * R @ src + t."""
    mu_s = src.mean(0); mu_d = dst.mean(0)
    cs = src - mu_s; cd = dst - mu_d
    H = cs.T @ cd / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    var_s = (cs ** 2).sum() / len(src)
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def fit_alpha(q_obs: np.ndarray, mean: np.ndarray, modes: np.ndarray,
              eigenvalues: np.ndarray, idx: np.ndarray,
              lambda_alpha: float = 1.0) -> np.ndarray:
    """Solve for PCA coefficients in the "world->SSM-frame" residuals.

    q_obs:       (N, 3) observations in the SSM frame (after inverse
                 similarity).
    mean:        (V, 3) mean shape.
    modes:       (K, V, 3) PCA modes (deformation directions at each vertex).
    eigenvalues: (K,) PCA variances; alpha is regularised by 1/sqrt(lambda).
    idx:         (N,) per-obs index into SSM vertices (the current
                 correspondence).
    lambda_alpha: global prior weight.

    Solves the dense K x K normal equations.
        E(alpha) = sum_n || q_obs_n - (mean[idx_n] + sum_k a_k mode_k[idx_n]) ||^2
                  + lambda_alpha * sum_k (a_k)^2 / eigenvalue_k
    Stack A (3N x K) where each row of M_n = mode_k[idx_n] becomes a row
    block in A; b is the 3N flattened residual mean[idx_n] - q_obs_n.
    """
    N = len(q_obs)
    K = len(modes)
    # A: (3N, K). For each obs n, A[3n:3n+3, k] = modes[k, idx[n], :].
    A = modes[:, idx, :].transpose(1, 2, 0).reshape(N * 3, K)
    b = (q_obs - mean[idx]).reshape(-1)
    # Ridge regularization. Penalize a_k^2 / eigval_k -> Tikhonov with
    # diag(lambda_alpha / eigvals).
    reg = lambda_alpha * np.diag(1.0 / np.maximum(eigenvalues, 1e-9))
    AtA = A.T @ A + reg
    Atb = A.T @ b
    return np.linalg.solve(AtA, Atb)


def posterior_sample_alpha(A: np.ndarray, b: np.ndarray,
                           alpha_map: np.ndarray, lambda_alpha: float,
                           n_samples: int = 100, seed: int = 0):
    """Sample alpha from the Gaussian linearization of the posterior.

    The MAP minimises ||A α - b||^2 + lambda * ||α||^2.  Under the
    proper Bayesian model with observation noise variance sigma^2 and
    prior precision Lambda = (lambda / sigma^2) * I, the posterior
    covariance is:

        Sigma_alpha = sigma^2 * (A^T A + lambda * I)^{-1}

    With sigma^2 estimated from MAP residuals:
        sigma^2_hat = ||A α_MAP - b||^2 / (3N - K)
    """
    n_rows = len(b)
    K = len(alpha_map)
    residuals = A @ alpha_map - b
    dof = max(n_rows - K, 1)
    sigma2_hat = float((residuals ** 2).sum()) / dof
    AtA = A.T @ A
    G = AtA + lambda_alpha * np.eye(K)
    Sigma_alpha = sigma2_hat * np.linalg.inv(G)
    # Symmetrize for numerical safety, add small jitter to Cholesky.
    Sigma_alpha = 0.5 * (Sigma_alpha + Sigma_alpha.T)
    L = np.linalg.cholesky(Sigma_alpha + 1e-12 * np.eye(K))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((K, n_samples))
    samples = alpha_map[None, :] + (L @ z).T  # (n_samples, K)
    return samples, sigma2_hat, Sigma_alpha


def per_vertex_world_uncertainty(samples, mean, modes, R, t, s):
    """For each SSM vertex, compute per-coordinate std across alpha samples
    after the similarity transform.  Returns (V,) magnitude of std vector
    per vertex."""
    n_samples = len(samples)
    V = len(mean)
    # Each sample: surface = mean + samples @ modes  ->  apply similarity.
    # Vectorized: stack samples as (n_samples, K).
    # modes: (K, V, 3)  ->  samples @ modes.reshape(K, V*3): (n_samples, V*3)
    # then reshape to (n_samples, V, 3).
    deformations = samples @ modes.reshape(len(modes), -1)
    deformations = deformations.reshape(n_samples, V, 3)
    surfaces_local = mean[None, :, :] + deformations
    # Apply similarity: world = s * (R @ surface_local.T).T + t per sample
    # surfaces_world: (n_samples, V, 3)
    surfaces_world = s * surfaces_local @ R.T + t
    mean_world = surfaces_world.mean(0)
    per_vert_std_vec = surfaces_world.std(axis=0)              # (V, 3)
    per_vert_std_mag = np.linalg.norm(per_vert_std_vec, axis=1)  # (V,)
    return per_vert_std_mag.astype(np.float32), mean_world


def ssm_fit(obs: np.ndarray, mean: np.ndarray, modes: np.ndarray,
            eigenvalues: np.ndarray, max_iter: int = 20,
            lambda_alpha: float = 1.0, outlier_pct: float = 90.0,
            tol: float = 1e-4, verbose: bool = True,
            n_posterior_samples: int = 100):
    """Fit similarity + PCA coefficients to noisy observations.

    Returns dict with keys: alpha, R, t, s, fitted_world, residuals, history.
    """
    V = len(mean)
    K = len(modes)
    alpha = np.zeros(K)

    # Initial similarity: Umeyama from mean-shape centroid to obs centroid +
    # bounding box scale match.
    mean_centroid = mean.mean(0)
    obs_centroid = obs.mean(0)
    # Per-axis std as a scale proxy
    s = float(obs.std(0).mean() / max(mean.std(0).mean(), 1e-9))
    R = np.eye(3)
    t = obs_centroid - s * (R @ mean_centroid)
    if verbose:
        print(f"[ssm_fit] init s={s:.4f}  |obs|={len(obs)}  V={V}  K={K}")

    history = []
    for it in range(max_iter):
        # 1. Current SSM surface in world frame.
        surface = mean + np.tensordot(alpha, modes, axes=1)  # (V, 3)
        surface_world = (s * (R @ surface.T)).T + t

        # 2. Correspondences: nearest vertex per observation, with
        #    percentile-based outlier rejection.
        tree = cKDTree(surface_world)
        dists, idx = tree.query(obs)
        if outlier_pct < 100:
            keep = dists < np.percentile(dists, outlier_pct)
        else:
            keep = np.ones(len(obs), dtype=bool)
        obs_k = obs[keep]
        target_k = surface_world[idx[keep]]
        rmse = float(np.sqrt((dists[keep] ** 2).mean()))

        # 3. Similarity update (Umeyama on corresponded pairs).
        s_new, R_new, t_new = umeyama(surface[idx[keep]], obs_k)
        # Anchor the similarity to a multiplicative update — combines
        # with current alpha non-linearly. For simplicity, replace.
        s, R, t = s_new, R_new, t_new

        # 4. Alpha update: transform observations into SSM frame, fit alpha.
        q_obs = ((obs_k - t) @ R) / max(s, 1e-9)
        alpha = fit_alpha(q_obs, mean, modes, eigenvalues,
                          idx[keep], lambda_alpha=lambda_alpha)
        if verbose:
            print(f"[ssm_fit] iter {it:>2}  rmse={rmse:.4f}  "
                  f"s={s:.4f}  ||alpha||={float(np.linalg.norm(alpha)):.3f}  "
                  f"n_kept={keep.sum()}/{len(obs)}")
        history.append({"iter": it, "rmse": rmse, "s": s,
                        "alpha_norm": float(np.linalg.norm(alpha)),
                        "n_kept": int(keep.sum())})
        if it > 0 and abs(history[-2]["rmse"] - rmse) < tol:
            if verbose:
                print(f"[ssm_fit] converged.")
            break

    # Final fitted surface (MAP).
    surface = mean + np.tensordot(alpha, modes, axes=1)
    fitted_world = (s * (R @ surface.T)).T + t

    # Heuristic proxy: distance to nearest observation per vertex.
    obs_tree = cKDTree(obs)
    per_vert_dist, _ = obs_tree.query(fitted_world)

    # Proper posterior sampling. Build the design matrix at the final
    # correspondences and sample n_posterior_samples alphas from
    # N(alpha_MAP, sigma^2 * (A^T A + lambda * I)^-1).
    if verbose:
        print(f"[ssm_fit] sampling N={n_posterior_samples} alphas from posterior...")
    # Final correspondences using the converged surface.
    tree = cKDTree(fitted_world)
    dists, idx_final = tree.query(obs)
    keep = (dists < np.percentile(dists, outlier_pct))
    obs_k = obs[keep]
    idx_k = idx_final[keep]
    q_obs = ((obs_k - t) @ R) / max(s, 1e-9)  # transform obs into SSM frame
    A_final = modes[:, idx_k, :].transpose(1, 2, 0).reshape(len(obs_k) * 3,
                                                            len(modes))
    b_final = (q_obs - mean[idx_k]).reshape(-1)
    samples, sigma2_hat, Sigma_alpha = posterior_sample_alpha(
        A_final, b_final, alpha, lambda_alpha,
        n_samples=n_posterior_samples,
    )
    per_vert_std, per_vert_mean = per_vertex_world_uncertainty(
        samples, mean, modes, R, t, s)
    if verbose:
        print(f"[ssm_fit] posterior: sigma^2_hat={sigma2_hat:.6f}  "
              f"(rmse={np.sqrt(sigma2_hat):.4f})  "
              f"per-vert std median={float(np.median(per_vert_std)):.4f}  "
              f"p95={float(np.percentile(per_vert_std, 95)):.4f}")

    return dict(
        alpha=alpha, R=R, t=t, s=float(s),
        fitted_world=fitted_world,
        per_vert_dist=per_vert_dist.astype(np.float32),
        per_vert_std=per_vert_std,
        per_vert_posterior_mean=per_vert_mean.astype(np.float32),
        posterior_samples=samples.astype(np.float32),
        Sigma_alpha=Sigma_alpha.astype(np.float32),
        sigma2_hat=float(sigma2_hat),
        history=history,
    )


def write_mesh_ply(path: Path, verts: np.ndarray, faces: np.ndarray,
                   vertex_colors_u8: np.ndarray = None) -> None:
    """Write a triangle mesh (with optional per-vertex colors)."""
    if vertex_colors_u8 is None:
        vertex_data = np.empty(len(verts),
                               dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertex_data["x"], vertex_data["y"], vertex_data["z"] = \
            verts[:, 0], verts[:, 1], verts[:, 2]
    else:
        vertex_data = np.empty(len(verts),
                               dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                      ("red", "u1"), ("green", "u1"),
                                      ("blue", "u1")])
        vertex_data["x"], vertex_data["y"], vertex_data["z"] = \
            verts[:, 0], verts[:, 1], verts[:, 2]
        vertex_data["red"] = vertex_colors_u8[:, 0]
        vertex_data["green"] = vertex_colors_u8[:, 1]
        vertex_data["blue"] = vertex_colors_u8[:, 2]
    face_data = np.empty(len(faces),
                         dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    PlyData(
        [PlyElement.describe(vertex_data, "vertex"),
         PlyElement.describe(face_data, "face")],
        text=False,
    ).write(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--recon_dir", required=True, type=Path,
                    help="Directory containing points.ply + poses.npz")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--lambda_alpha", type=float, default=1.0)
    ap.add_argument("--outlier_pct", type=float, default=90.0)
    ap.add_argument("--max_points", type=int, default=80000,
                    help="Subsample observations for speed.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load SSM.
    with h5py.File(args.corpus, "r") as f:
        mean = f["metadata/ssm_surface/mean_shape"][:].astype(np.float64)
        modes = f["metadata/ssm_surface/modes"][:].astype(np.float64)
        eigenvalues = f["metadata/ssm_surface/eigenvalues"][:].astype(np.float64)
        faces = f["metadata/surface_template_faces"][:].astype(np.int32)
    print(f"[ssm_fit] SSM: V={len(mean)}  K={len(modes)}  faces={len(faces)}")
    print(f"[ssm_fit] eigenvalues range: "
          f"{eigenvalues.min():.1e} - {eigenvalues.max():.1e}")
    # Pre-scale modes so that alpha is in z-score units (alpha=1 -> 1 sigma
    # deformation). After this, lambda_alpha=1 means "1-unit penalty per
    # 1-sigma deviation per mode" -- comparable to data residuals if those
    # are also order-unity.
    std_per_mode = np.sqrt(np.maximum(eigenvalues, 1e-9))
    modes = modes * std_per_mode[:, None, None]   # absorb sigma into modes
    eigenvalues = np.ones_like(eigenvalues)        # now unit-variance prior
    print(f"[ssm_fit] modes normalized: alpha is now in z-score units")

    # Load observations: dust3r-foe points, drop dead-pose contributions.
    ply = PlyData.read(str(args.recon_dir / "points.ply"))
    obs = np.stack([ply["vertex"]["x"], ply["vertex"]["y"],
                    ply["vertex"]["z"]], 1).astype(np.float64)
    poses = np.load(args.recon_dir / "poses.npz")["c2w"]
    nz = ~np.all(poses[:, :3, 3] == 0, axis=1)
    N = len(poses)
    if (~nz).any():
        # Assume points are concatenated frame-by-frame in equal counts.
        n_per_frame = max(1, len(obs) // N)
        per_pt_frame = np.minimum(np.arange(len(obs)) // n_per_frame, N - 1)
        keep_mask = nz[per_pt_frame]
        obs = obs[keep_mask]
        print(f"[ssm_fit] dropped {(~keep_mask).sum()} dead-pose points; "
              f"{len(obs)} remaining")

    if len(obs) > args.max_points:
        np.random.seed(0)
        sub = np.random.choice(len(obs), args.max_points, replace=False)
        obs = obs[sub]
        print(f"[ssm_fit] subsampled to {len(obs)} observations")

    t0 = time.time()
    result = ssm_fit(obs, mean, modes, eigenvalues,
                     max_iter=args.max_iter,
                     lambda_alpha=args.lambda_alpha,
                     outlier_pct=args.outlier_pct)
    print(f"[ssm_fit] done in {time.time() - t0:.1f}s")

    # Save MAP mesh + posterior diagnostics.
    fitted_world = result["fitted_world"]
    per_vert_std = result["per_vert_std"]
    per_vert_dist = result["per_vert_dist"]
    # Color by posterior std (proper uncertainty), not by residual proxy.
    # Green = tight CI; red = wide CI.
    vmax_std = float(np.percentile(per_vert_std, 95))
    norm_std = np.clip(per_vert_std / max(vmax_std, 1e-9), 0, 1)
    colors_std = np.stack([
        (255 * norm_std).astype(np.uint8),
        (255 * (1 - norm_std)).astype(np.uint8),
        np.full(len(per_vert_std), 64, np.uint8),
    ], 1)
    write_mesh_ply(args.out_dir / "mesh_ssm.ply",
                   fitted_world.astype(np.float32), faces, colors_std)
    # Also write a residual-colored version for comparison with the earlier
    # heuristic uncertainty proxy.
    vmax_res = float(np.percentile(per_vert_dist, 95))
    norm_res = np.clip(per_vert_dist / max(vmax_res, 1e-9), 0, 1)
    colors_res = np.stack([
        (255 * norm_res).astype(np.uint8),
        (255 * (1 - norm_res)).astype(np.uint8),
        np.full(len(per_vert_dist), 64, np.uint8),
    ], 1)
    write_mesh_ply(args.out_dir / "mesh_ssm_residual_color.ply",
                   fitted_world.astype(np.float32), faces, colors_res)

    np.savez(args.out_dir / "ssm_fit.npz",
             alpha=result["alpha"], R=result["R"], t=result["t"],
             s=result["s"],
             per_vert_std=result["per_vert_std"],
             per_vert_dist=result["per_vert_dist"],
             posterior_samples=result["posterior_samples"],
             Sigma_alpha=result["Sigma_alpha"],
             sigma2_hat=result["sigma2_hat"])
    (args.out_dir / "ssm_fit.json").write_text(
        json.dumps({"alpha": result["alpha"].tolist(),
                    "s": result["s"],
                    "history": result["history"],
                    "sigma2_hat": result["sigma2_hat"],
                    "per_vert_std_median": float(np.median(per_vert_std)),
                    "per_vert_std_p95": vmax_std,
                    "per_vert_dist_median": float(np.median(per_vert_dist)),
                    "per_vert_dist_p95": vmax_res}, indent=2))
    print(f"[ssm_fit] wrote {args.out_dir / 'mesh_ssm.ply'} "
          f"({len(fitted_world)} verts, {len(faces)} tris)")
    print(f"[ssm_fit] per-vert posterior STD (units): "
          f"median={np.median(per_vert_std):.4f}  p95={vmax_std:.4f}  "
          f"max={per_vert_std.max():.4f}")
    print(f"[ssm_fit] per-vert nearest-obs distance: "
          f"median={np.median(per_vert_dist):.4f}  p95={vmax_res:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
