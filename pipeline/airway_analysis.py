"""Persistent airway-analysis toolkit (survives /tmp wipes — reads clouds from /home).

Cloud-medial-axis centerline + per-slice caliber (D_CE) and cross-sectional area (CSA) with a
coverage gate and 3 area estimators (circle / convex-hull / polar-polygon) whose agreement is the
validity check. Isotropic Sim3 registration (scale+flip) for CT-paired cases. Matplotlib 3D-scatter
renderer (no o3d offscreen, so it can't be broken by EGL/headless issues).

Used by ct3_report.py (CT accuracy) and csa_report.py (scale-free CSA / %obstruction)."""
import numpy as np, open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from scipy.ndimage import median_filter
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def clean(fp, nb=20, std=2.0):
    pcd = fp if isinstance(fp, o3d.geometry.PointCloud) else o3d.io.read_point_cloud(str(fp))
    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb, std)
    return pcd


def dce_of(gt):
    """Circle-equivalent diameter from a GroundTruth CSA profile (mm)."""
    return median_filter(2 * np.sqrt(np.clip(gt.csa_mm2, 0, None) / np.pi), 7)


def axis_centerline(P, N=45, smooth=2.0):
    """Medial-axis centerline: order the cloud along its PCA principal axis, take the per-slab
    centroid, fit a smoothing spline, resample to N points. Returns C (pts), T (unit tangents), arc."""
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    u = Vt[0]
    s = c @ u
    edges = np.linspace(s.min(), s.max(), min(N, 40) + 1)
    cen = []
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() >= 8:
            cen.append(P[m].mean(0))
    cen = np.array(cen)
    if len(cen) < 5:
        return None, None, None
    tck, _ = splprep(cen.T, u=np.linspace(0, 1, len(cen)), s=len(cen) * smooth)
    us = np.linspace(0, 1, N)
    C = np.array(splev(us, tck)).T
    T = np.array(splev(us, tck, der=1)).T
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    return C, T, arc


def slice_metrics(P, C, T, arc, nbins=36):
    """Per-centerline-slice: median radius, angular coverage, and 3 area estimators.
    Returns dict of arrays (len N): r_med, cov, a_circle, a_hull, a_polar (scene units)."""
    N = len(C)
    slab = (arc[-1] / (N - 1)) * 0.8
    out = {k: np.full(N, np.nan) for k in ("r_med", "a_circle", "a_hull", "a_polar")}
    out["cov"] = np.zeros(N)
    for i in range(N):
        p, t = C[i], T[i]
        d = P - p
        al = d @ t
        sel = np.abs(al) < slab / 2
        q = d[sel] - np.outer(al[sel], t)
        r = np.linalg.norm(q, axis=1)
        if len(r) < 15:
            continue
        keep = r < np.percentile(r, 90)
        q, r = q[keep], r[keep]
        if len(r) < 12:
            continue
        e1 = np.array([t[1], -t[0], 0.0])
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 = np.cross(t, e1)
        x, y = q @ e1, q @ e2
        ang = np.arctan2(y, x)
        hist, _ = np.histogram(ang, nbins, (-np.pi, np.pi))
        out["cov"][i] = (hist > 0).sum() / nbins
        rmed = np.median(r)
        out["r_med"][i] = rmed
        out["a_circle"][i] = np.pi * rmed ** 2
        try:
            out["a_hull"][i] = ConvexHull(np.c_[x, y]).volume  # 2D hull area
        except Exception:
            pass
        # polar polygon: per-bin median radius -> area of the star polygon
        rb = []
        for b in range(nbins):
            lo, hi = -np.pi + b * 2 * np.pi / nbins, -np.pi + (b + 1) * 2 * np.pi / nbins
            mb = (ang >= lo) & (ang < hi)
            rb.append(np.median(r[mb]) if mb.sum() else np.nan)
        rb = np.array(rb)
        if np.isfinite(rb).sum() >= nbins * 0.6:
            rb = np.where(np.isfinite(rb), rb, np.nanmedian(rb))
            out["a_polar"][i] = 0.5 * np.sum(rb * np.roll(rb, -1)) * np.sin(2 * np.pi / nbins)
    return out


def valid_mask(m, cov_thr=0.6, agree=2.5):
    """Slices with coverage>=thr AND the 3 area estimators agreeing within `agree`x."""
    A = np.vstack([m["a_circle"], m["a_hull"], m["a_polar"]])
    with np.errstate(invalid="ignore"):
        ratio = np.nanmax(A, 0) / np.nanmin(A, 0)
    ok = (m["cov"] >= cov_thr) & np.isfinite(m["r_med"]) & np.isfinite(ratio) & (ratio <= agree)
    if ok.sum() >= 6:  # robust radius outlier trim
        med = np.median(m["r_med"][ok]); mad = np.median(np.abs(m["r_med"][ok] - med)) + 1e-9
        ok &= np.abs(m["r_med"] - med) < 3.5 * mad
    return ok


def robust_register(a_r, r_m, ct_arc, ct_dce):
    """Isotropic scale + flip search: fit recon radius profile to CT D_CE. Returns dict."""
    a_r = a_r - a_r[0]
    o = np.argsort(a_r); a_r, r_m = a_r[o], r_m[o]
    ss = np.linspace(0.2, 20, 991); best = None
    for flip in (False, True):
        C = ct_dce[::-1] if flip else ct_dce
        A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
        for s in ss:
            e = 2 * s * r_m - np.interp(s * a_r, A, C, left=np.nan, right=np.nan)
            e = e[np.isfinite(e)]
            if len(e) > 5:
                rm = np.sqrt(np.mean(e ** 2))
                if best is None or rm < best[0]:
                    best = (rm, s, flip)
    _, s, flip = best
    C = ct_dce[::-1] if flip else ct_dce
    A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
    dce = 2 * s * r_m; am = s * a_r
    ct = np.interp(am, A, C, left=np.nan, right=np.nan)
    m = np.isfinite(ct)
    err = dce[m] - ct[m]
    return dict(scale=float(s), flip=bool(flip), am=am, dce=dce, ct=ct, m=m, A=A, C=C,
                rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)), n=int(m.sum()))


def color_axis(P):
    """viridis value in [0,1] along the PCA principal axis (for consistent depth colouring)."""
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    t = c @ Vt[0]
    return (t - t.min()) / (np.ptp(t) + 1e-9)


def scatter3d(ax, P, sub=45000, s=1.6, elev=6, azim=-60, rad_pct=95, box=None, zoom=1.0):
    """Robust matplotlib 3D scatter of an airway tube. Aligns the PCA principal axis to the
    vertical, trims the radial outlier spray, and views from the side so the lumen reads as a
    vertical column. viridis along the axis. No o3d offscreen needed (can't break on headless)."""
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    Q = c @ Vt.T                                   # col0 = along-axis, col1/2 = perpendicular
    rad = np.hypot(Q[:, 1], Q[:, 2])
    Q = Q[rad < np.percentile(rad, rad_pct)]       # drop radial spray
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Q), min(sub, len(Q)), replace=False)
    Qs = Q[idx]
    t = (Qs[:, 0] - Q[:, 0].min()) / (np.ptp(Q[:, 0]) + 1e-9)
    # axis -> vertical (z); perpendicular -> x,y
    ax.scatter(Qs[:, 1], Qs[:, 2], Qs[:, 0], c=t, cmap="viridis", s=s, marker=".", linewidths=0, alpha=0.85)
    ax.set_box_aspect(box or (np.ptp(Q[:, 1]), np.ptp(Q[:, 2]), np.ptp(Q[:, 0])), zoom=zoom)   # box=None: true proportions; e.g. (1,1,0.5) for a down-axis view
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
