"""STAGE 4: cross-section profile + scale-invariant %narrowing on recon_airway.

Pipeline:
  1. Centerline = smooth parametric cubic spline through registered camera
     centers, ordered by frame index. Re-parametrize by arc length.
  2. Resample finely (step ~1 scene unit) along the centerline. Tangent at
     each sample is the spline derivative, unit-normalized.
  3. At each sample p with tangent t:
       - gather DENSE MVS points in a thin slab (|along-tangent| <= SLAB_HALF)
       - project into the plane perpendicular to t (basis e1, e2)
       - angular coverage on 36 polar bins from p; keep slice if coverage>=60%
       - alpha-shape area on the in-plane points (primary)
       - convex-hull area + polar-median-radius area (sanity)
       - Deq = 2*sqrt(A_alpha/pi)
  4. Identify A_min and A_ref (widest well-covered slice above the narrowing
     and globally), and report %obstruction = (1 - A_min/A_ref)*100.
  5. Save: profile plot + 3D render with the kept rings.

All quantities in SCENE UNITS — this stage is scale-free.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull, Delaunay, cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
RECON = PROJECT / "runs/gated2/2_v2/recon_airway"
OUT = RECON / "stage4"
OUT.mkdir(exist_ok=True)

SLAB_HALF = 0.3      # scene units (recon is very compact)
N_BINS = 36          # 10 deg
MIN_COVERAGE = 0.60
SPLINE_SMOOTH = 0.5  # splprep s parameter
SAMPLE_STEP = 0.15   # scene units between centerline samples
OUTLIER_R_FROM_CAMS = 8.0  # keep dense points within this radial dist of any cam


def load_cameras():
    sub = r"""
import pycolmap, json, sys, re, numpy as np
rec = pycolmap.Reconstruction(sys.argv[1])
out = []
for img in rec.images.values():
    m = re.search(r'f(\d+)\.png', img.name)
    fi = int(m.group(1)) if m else -1
    cfw = img.cam_from_world()
    M = np.array(cfw.matrix())
    R = M[:3, :3]; t = M[:3, 3]
    C = (-R.T @ t).tolist()
    a = (R.T @ np.array([0,0,1])).tolist()
    out.append({"name": img.name, "frame_idx": fi, "C": C, "axis": a,
                 "image_id": int(img.image_id)})
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", sub, str(RECON / "sfm")],
                          capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return sorted(json.loads(res.stdout.strip().splitlines()[-1]),
                   key=lambda r: r["frame_idx"])


def alpha_shape_area(pts2d, alpha):
    if len(pts2d) < 4:
        return 0.0
    try:
        tri = Delaunay(pts2d)
    except Exception:
        return 0.0
    sim = tri.simplices
    P = pts2d[sim]
    a = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    b = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
    c = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
    s = (a + b + c) / 2
    tri_area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    R = (a * b * c) / (4 * tri_area + 1e-12)
    keep = (R < 1.0 / alpha) & (tri_area > 1e-9)
    return float(tri_area[keep].sum())


def polar_polygon_area(pts2d, n_bins=36):
    if len(pts2d) < 6:
        return 0.0
    theta = np.arctan2(pts2d[:, 1], pts2d[:, 0])
    r = np.linalg.norm(pts2d, axis=1)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(theta, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    medians = np.zeros(n_bins)
    for i in range(n_bins):
        m = (bin_idx == i)
        if m.any():
            medians[i] = np.median(r[m])
    # Fill empty bins by adjacent-bin linear interpolation around the circle
    empty = (medians == 0)
    if empty.all():
        return 0.0
    if empty.any():
        ix = np.arange(n_bins)
        ix_known = ix[~empty]
        for i in np.where(empty)[0]:
            d_circ = np.minimum(
                np.abs(i - ix_known),
                n_bins - np.abs(i - ix_known))
            nearest = ix_known[d_circ.argmin()]
            medians[i] = medians[nearest]
    xs = medians * np.cos(bin_centers)
    ys = medians * np.sin(bin_centers)
    a = 0.0
    for i in range(n_bins):
        j = (i + 1) % n_bins
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return float(abs(a) / 2.0)


def main():
    print("=== STAGE 4: profile + %obstruction (scene units) ===")
    cams = load_cameras()
    centers = np.array([c["C"] for c in cams], dtype=float)
    fis = [c["frame_idx"] for c in cams]
    print(f"  loaded {len(cams)} cameras (fi {fis[0]}..{fis[-1]})")
    print(f"  camera bbox: dx={float(centers[:,0].max()-centers[:,0].min()):.3f} "
          f"dy={float(centers[:,1].max()-centers[:,1].min()):.3f} "
          f"dz={float(centers[:,2].max()-centers[:,2].min()):.3f}")

    pcd = o3d.io.read_point_cloud(str(RECON / "dense" / "fused.ply"))
    pts_all = np.asarray(pcd.points)
    # Outlier filter: keep dense points within OUTLIER_R_FROM_CAMS of any camera
    cam_tree = cKDTree(centers)
    d_min, _ = cam_tree.query(pts_all, k=1)
    pts = pts_all[d_min <= OUTLIER_R_FROM_CAMS]
    print(f"  dense MVS: {len(pts_all)} pts; after outlier filter "
          f"(r<={OUTLIER_R_FROM_CAMS} from any cam): {len(pts)} pts")

    # 1) Smooth parametric spline; re-parameterize by arc length
    tck, u = splprep(centers.T, s=SPLINE_SMOOTH, k=3)
    u_fine = np.linspace(0, 1, 5000)
    p_fine = np.array(splev(u_fine, tck)).T
    diffs = np.diff(p_fine, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_s = float(s_cum[-1])
    print(f"  centerline arc length: {total_s:.2f} scene units")

    # Map camera positions to arc length (projection)
    cam_s = []
    tree = cKDTree(p_fine)
    for C in centers:
        _, ix = tree.query(C, k=1)
        cam_s.append(float(s_cum[int(ix)]))
    cam_s = np.array(cam_s)

    # 2) Uniform-arc-length resampling
    s_target = np.arange(0, total_s + 1e-6, SAMPLE_STEP)
    u_resamp = np.interp(s_target, s_cum, u_fine)
    p_resamp = np.array(splev(u_resamp, tck)).T
    tg_resamp = np.array(splev(u_resamp, tck, der=1)).T
    tg_resamp = tg_resamp / (np.linalg.norm(tg_resamp, axis=1, keepdims=True) + 1e-9)
    print(f"  centerline samples: {len(s_target)} (step {SAMPLE_STEP} scene units)")

    # 3) Per-slice extraction
    slices = []
    pts_tree = cKDTree(pts)
    for k, (P, T, s_val) in enumerate(zip(p_resamp, tg_resamp, s_target)):
        rel = pts - P
        d_along = rel @ T
        mask = np.abs(d_along) <= SLAB_HALF
        if mask.sum() < 10:
            continue
        in_slab = rel[mask]
        # Perpendicular basis
        up = np.array([0., 0., 1.])
        if abs(T @ up) > 0.95:
            up = np.array([0., 1., 0.])
        e1 = up - (up @ T) * T
        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
        e2 = np.cross(T, e1)
        in_plane = np.stack([in_slab @ e1, in_slab @ e2], axis=1)
        theta = np.arctan2(in_plane[:, 1], in_plane[:, 0])
        bins = np.linspace(-np.pi, np.pi, N_BINS + 1)
        counts = np.histogram(theta, bins=bins)[0]
        cov = float((counts > 0).mean())
        n_in = int(len(in_plane))
        # Adaptive alpha = 1/(3*median_nn_dist)
        if n_in >= 4:
            t2 = cKDTree(in_plane)
            d, _ = t2.query(in_plane, k=2)
            median_nn = float(np.median(d[:, 1])) if d.shape[1] > 1 else 0.0
            alpha = 1.0 / (3.0 * median_nn + 1e-9)
        else:
            alpha = 0.1
        A_alpha = alpha_shape_area(in_plane, alpha)
        try:
            A_conv = float(ConvexHull(in_plane).volume) if n_in >= 3 else 0.0
        except Exception:
            A_conv = 0.0
        A_polar = polar_polygon_area(in_plane, N_BINS)
        Deq = 2.0 * np.sqrt(A_alpha / np.pi) if A_alpha > 0 else 0.0
        slices.append({
            "k": k, "s": float(s_val),
            "P": P.tolist(), "T": T.tolist(),
            "e1": e1.tolist(), "e2": e2.tolist(),
            "n_pts": n_in,
            "coverage": cov,
            "alpha_used": float(alpha),
            "area_alpha": float(A_alpha),
            "area_convex": float(A_conv),
            "area_polar_median_r": float(A_polar),
            "Deq_alpha": float(Deq),
            "in_plane": in_plane.tolist(),
        })

    kept = [s for s in slices if s["coverage"] >= MIN_COVERAGE]
    dropped = len(slices) - len(kept)
    print(f"  slices with >=10 pts: {len(slices)}")
    print(f"  measurable slices (coverage >= {int(MIN_COVERAGE*100)}%): {len(kept)} "
          f"(dropped {dropped})")

    if not kept:
        print("  no measurable slices; aborting")
        return

    ss = np.array([s["s"] for s in kept])
    A_alpha = np.array([s["area_alpha"] for s in kept])
    A_conv = np.array([s["area_convex"] for s in kept])
    A_polar = np.array([s["area_polar_median_r"] for s in kept])
    Deq = np.array([s["Deq_alpha"] for s in kept])
    covs = np.array([s["coverage"] for s in kept])

    # Minimum (narrowing) by alpha-shape area
    i_min = int(A_alpha.argmin())
    A_min = float(A_alpha[i_min]); s_min = float(ss[i_min])
    print(f"\n  A_min = {A_min:.3f} scene-units^2 at s={s_min:.2f} "
          f"(coverage={covs[i_min]*100:.0f}%)")

    # Reference = widest well-covered ABOVE the narrowing (lower s);
    # also report widest BELOW (higher s) and global widest.
    before = A_alpha[:i_min]
    after = A_alpha[i_min + 1:]
    A_ref_above = float(before.max()) if before.size else None
    s_ref_above = float(ss[:i_min][int(before.argmax())]) if before.size else None
    A_ref_below = float(after.max()) if after.size else None
    s_ref_below = (float(ss[i_min + 1:][int(after.argmax())])
                    if after.size else None)
    A_ref_global = float(A_alpha.max())
    s_ref_global = float(ss[int(A_alpha.argmax())])
    # Pick A_ref as widest above the narrowing (Myer-Cotton convention)
    A_ref = A_ref_above if (A_ref_above is not None and A_ref_above > 0) \
            else A_ref_global
    s_ref = s_ref_above if A_ref_above and A_ref_above > 0 else s_ref_global
    pct_obstr = (1 - A_min / A_ref) * 100 if A_ref > 0 else None
    print(f"  A_ref (widest above narrowing) = {A_ref:.3f} at s={s_ref:.2f}")
    print(f"  A_ref (widest below narrowing) = {A_ref_below}")
    print(f"  A_ref (global widest)          = {A_ref_global:.3f} at s={s_ref_global:.2f}")
    print(f"\n  %obstruction (Myer-Cotton convention, A_ref above) = "
          f"{pct_obstr:.2f} %")

    # ============== Save profile plot ==============
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    sc1 = axes[0].scatter(ss, A_alpha, c=covs * 100, cmap="viridis",
                          s=30, edgecolors="black", linewidths=0.4,
                          vmin=60, vmax=100, zorder=3)
    axes[0].plot(ss, A_alpha, "-", c="gray", alpha=0.5, lw=0.8)
    axes[0].plot(ss, A_conv, "-.", c="orange", alpha=0.5, lw=0.7,
                 label="convex hull (sanity)")
    axes[0].plot(ss, A_polar, ":", c="purple", alpha=0.5, lw=0.7,
                 label="polar median-r polygon (sanity)")
    axes[0].axvline(s_min, c="red", lw=1.2, ls="--", alpha=0.7)
    axes[0].axvline(s_ref, c="green", lw=1.2, ls="--", alpha=0.7)
    axes[0].annotate(f"A_min = {A_min:.2f}\ns={s_min:.1f}",
                     xy=(s_min, A_min), xytext=(s_min + 5, A_min),
                     fontsize=10, color="red")
    axes[0].annotate(f"A_ref = {A_ref:.2f}\ns={s_ref:.1f}",
                     xy=(s_ref, A_ref), xytext=(s_ref + 5, A_ref),
                     fontsize=10, color="green")
    # Camera-position rugs
    for cs in cam_s:
        axes[0].axvline(cs, ymin=0.97, ymax=1.0, c="black", lw=0.6, alpha=0.5)
    axes[0].set_ylabel("Cross-section area (scene-units²)\nalpha-shape, primary")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label="coverage %", pad=0.01)
    axes[0].set_title(
        f"recon_airway profile (scene units, no scale anchor applied)\n"
        f"measurable slices: {len(kept)} (>={int(MIN_COVERAGE*100)}% coverage); "
        f"%obstruction (A_min vs widest-above) = {pct_obstr:.2f} %"
    )

    sc2 = axes[1].scatter(ss, Deq, c=covs * 100, cmap="viridis",
                          s=30, edgecolors="black", linewidths=0.4,
                          vmin=60, vmax=100, zorder=3)
    axes[1].plot(ss, Deq, "-", c="gray", alpha=0.5, lw=0.8)
    axes[1].axvline(s_min, c="red", lw=1.2, ls="--", alpha=0.7)
    axes[1].axvline(s_ref, c="green", lw=1.2, ls="--", alpha=0.7)
    for cs in cam_s:
        axes[1].axvline(cs, ymin=0.97, ymax=1.0, c="black", lw=0.6, alpha=0.5)
    axes[1].set_xlabel("Arc length s (scene units)  |  black rugs = camera positions")
    axes[1].set_ylabel("Deq = 2·sqrt(A_alpha/π)  (scene units)")
    plt.colorbar(sc2, ax=axes[1], label="coverage %", pad=0.01)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT / "profile.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'profile.png'}")

    # ============== 3D render with kept rings ==============
    fig = plt.figure(figsize=(18, 6))
    titles = ["side (XZ)", "top (XY)", "front (YZ)"]
    proj = [("X", "Z", 0, 2), ("X", "Y", 0, 1), ("Y", "Z", 1, 2)]
    pcd_pts = pts
    if len(pcd_pts) > 50000:
        ix = np.random.default_rng(0).choice(len(pcd_pts), 50000, replace=False)
        pcd_pts = pcd_pts[ix]
    # Color of each kept slice's ring by coverage
    cov_norm = (covs - MIN_COVERAGE) / max(1.0 - MIN_COVERAGE, 1e-9)
    cmap = plt.get_cmap("viridis")
    for k, (lx, ly, ix, iy) in enumerate(proj):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(pcd_pts[:, ix], pcd_pts[:, iy],
                   c="lightgray", s=0.3, alpha=0.4, marker='.',
                   linewidths=0)
        # Centerline polyline
        ax.plot(p_resamp[:, ix], p_resamp[:, iy], "-",
                c="white", lw=2.2)
        ax.plot(p_resamp[:, ix], p_resamp[:, iy], "-",
                c="black", lw=1.0)
        # Each kept slice ring: project the in_plane 2D points back to 3D
        # ring = P + (e1*x + e2*y) for x,y on the kept-bin polygon
        for j, s_dat in enumerate(kept):
            P = np.array(s_dat["P"]); e1 = np.array(s_dat["e1"]); e2 = np.array(s_dat["e2"])
            ip = np.array(s_dat["in_plane"])
            if len(ip) == 0: continue
            # Convex hull for visualization (more intuitive than alpha)
            try:
                hull = ConvexHull(ip)
                ring2d = ip[hull.vertices]
                ring2d = np.vstack([ring2d, ring2d[:1]])
                ring3d = P + ring2d[:, :1] * e1 + ring2d[:, 1:] * e2
                ax.plot(ring3d[:, ix], ring3d[:, iy], "-",
                         c=cmap(cov_norm[j]), lw=0.8, alpha=0.7)
            except Exception:
                continue
        # Cameras
        ax.scatter(centers[:, ix], centers[:, iy], c="red", s=20,
                   marker="o", edgecolors="black", linewidths=0.4,
                   label="cameras", zorder=4)
        ax.scatter([p_resamp[i_min, ix]], [p_resamp[i_min, iy]],
                   c="red", marker="X", s=120, edgecolors="black",
                   linewidths=0.6, label="A_min", zorder=5)
        ax.scatter([p_resamp[int(np.where(s_target == s_target[np.searchsorted(s_target, s_ref)])[0][0]), ix]
                     if s_ref in s_target else 0],
                   [p_resamp[int(np.where(s_target == s_target[np.searchsorted(s_target, s_ref)])[0][0]), iy]
                     if s_ref in s_target else 0],
                   c="green", marker="X", s=120, edgecolors="black",
                   linewidths=0.6, label="A_ref", zorder=5)
        ax.set_title(f"{titles[k]}  ({lx},{ly})")
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"recon_airway rings  |  centerline arc length = {total_s:.1f}  |  "
        f"kept slices = {len(kept)} (>={int(MIN_COVERAGE*100)}% coverage)\n"
        f"rings colored by coverage %",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(str(OUT / "rings_3view.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'rings_3view.png'}")

    # ============== Save JSON ==============
    # Identify slices that were 'too sparse to trust' (n_pts < some threshold)
    sparse_threshold = 25
    sparse_slices = [
        {"s": s["s"], "n_pts": s["n_pts"], "coverage": s["coverage"]}
        for s in kept if s["n_pts"] < sparse_threshold
    ]
    diag = {
        "units": "scene units (no scale anchor applied)",
        "slab_half_scene_units": SLAB_HALF,
        "n_angular_bins": N_BINS,
        "min_coverage": MIN_COVERAGE,
        "sample_step_scene_units": SAMPLE_STEP,
        "centerline_total_arc_length": total_s,
        "n_cameras": len(cams),
        "camera_arc_lengths": cam_s.tolist(),
        "camera_frame_indices": fis,
        "n_centerline_samples": int(len(s_target)),
        "n_slices_with_points": len(slices),
        "n_measurable_slices": len(kept),
        "n_slices_dropped_low_coverage": dropped,
        "A_min": A_min, "s_min": s_min,
        "A_ref_above_narrowing": A_ref_above, "s_ref_above": s_ref_above,
        "A_ref_below_narrowing": A_ref_below, "s_ref_below": s_ref_below,
        "A_ref_global": A_ref_global, "s_ref_global": s_ref_global,
        "A_ref_chosen": A_ref, "s_ref_chosen": s_ref,
        "pct_obstruction_alpha_AmIn_vs_above": pct_obstr,
        "pct_obstruction_alpha_AmIn_vs_global": (
            (1 - A_min / A_ref_global) * 100 if A_ref_global > 0 else None
        ),
        "Deq_min": float(Deq.min()),
        "Deq_max": float(Deq.max()),
        "profile_records": [
            {"s": float(s["s"]), "n_pts": int(s["n_pts"]),
             "coverage": float(s["coverage"]),
             "area_alpha": float(s["area_alpha"]),
             "area_convex": float(s["area_convex"]),
             "area_polar_median_r": float(s["area_polar_median_r"]),
             "Deq_alpha": float(s["Deq_alpha"])}
            for s in kept
        ],
        "sparse_slices_in_kept_set": sparse_slices,
        "profile_png": str(OUT / "profile.png"),
        "rings_3view_png": str(OUT / "rings_3view.png"),
    }
    (OUT / "stage4_results.json").write_text(json.dumps(diag, indent=2))
    print(f"saved {OUT/'stage4_results.json'}")


if __name__ == "__main__":
    main()
