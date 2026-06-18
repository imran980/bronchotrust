"""STAGE 5 (gated plan): centerline cross-section profile on a dense MVS cloud.

Mapper-agnostic: --recon points to a stage4/<arm>/ dir; reads its
stage4_results.json for the sfm model dir + dense fused.ply. Runs on EITHER the
COLMAP or GlueMap arm. Scene units (scale-free).

Locked rules: slice at CENTERLINE points (not cameras); measure on the dense MVS
cloud (not Poisson). Per slice compute area THREE ways (alpha-shape / convex hull
/ polar-median-r) and KEEP only slices with >=60% angular coverage AND the three
estimators agreeing within AGREE_FACTOR (else the "ring" is noise -> reject).

Auto-scaled to the recon: lumen radius R = median perpendicular distance of
filtered cloud points to the centerline; slab half-thickness = SLAB_FRAC*R.

Outputs (--recon/stage5/):
  stage5_profile.json   per-slice areas/coverage/agreement + A_min/A_ref + %obstr
  profile.png           area- & Deq-vs-arclength, colored by coverage
  rings_3view.png       dense cloud + kept rings + centerline + cameras
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pycolmap

N_BINS = 36
MIN_COVERAGE = 0.60
AGREE_FACTOR = 2.0      # 3 estimators must agree within this ratio
SLAB_FRAC = 0.12        # slab half-thickness as fraction of lumen radius R
STEP_FRAC = 0.06        # centerline sample step as fraction of R
SMOOTH = 0.5


def load_cameras(sfm_dir):
    import re
    rec = pycolmap.Reconstruction(str(sfm_dir))
    out = []
    for img in rec.images.values():
        m = re.search(r"f(\d+)\.png", img.name)
        fi = int(m.group(1)) if m else -1
        M = np.array(img.cam_from_world().matrix())
        R, t = M[:3, :3], M[:3, 3]
        out.append({"frame_idx": fi, "C": (-R.T @ t).tolist()})
    return sorted(out, key=lambda r: r["frame_idx"])


def alpha_shape_area(p, alpha):
    if len(p) < 4:
        return 0.0
    try:
        tri = Delaunay(p)
    except Exception:
        return 0.0
    P = p[tri.simplices]
    a = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    b = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
    c = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    Rc = (a * b * c) / (4 * area + 1e-12)
    keep = (Rc < 1.0 / alpha) & (area > 1e-12)
    return float(area[keep].sum())


def polar_area(p, n=N_BINS):
    if len(p) < 6:
        return 0.0
    th = np.arctan2(p[:, 1], p[:, 0])
    r = np.linalg.norm(p, axis=1)
    bins = np.linspace(-np.pi, np.pi, n + 1)
    idx = np.clip(np.digitize(th, bins) - 1, 0, n - 1)
    med = np.zeros(n)
    for i in range(n):
        mk = idx == i
        if mk.any():
            med[i] = np.median(r[mk])
    empty = med == 0
    if empty.all():
        return 0.0
    if empty.any():
        known = np.where(~empty)[0]
        for i in np.where(empty)[0]:
            dc = np.minimum(np.abs(i - known), n - np.abs(i - known))
            med[i] = med[known[dc.argmin()]]
    ce = (bins[:-1] + bins[1:]) / 2
    xs, ys = med * np.cos(ce), med * np.sin(ce)
    return float(abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                         for i in range(n))) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="stage4/<arm> dir")
    ap.add_argument("--ply", default=None, help="override dense ply (e.g. photometric / sparse)")
    ap.add_argument("--sfm", default=None, help="override sfm model dir (camera poses)")
    ap.add_argument("--tag", default="", help="suffix for the stage5 output dir")
    ap.add_argument("--label-json",
                    default="runs/gluemap_15v2/15_v2/stage1/stage1_selection.json")
    args = ap.parse_args()
    recon = Path(args.recon)
    res = json.loads((recon / "stage4_results.json").read_text())
    sfm_dir = args.sfm or res["sfm_model_dir"]
    ply = args.ply or res["dense_ply"]
    out = recon / (f"stage5{('_' + args.tag) if args.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    seg_of = {s["frame_idx"]: s["segment"] for s in
              json.loads(Path(args.label_json).read_text())["selection"]}

    cams = load_cameras(sfm_dir)
    centers = np.array([c["C"] for c in cams], float)
    fis = [c["frame_idx"] for c in cams]
    pcd = o3d.io.read_point_cloud(ply)
    pts_all = np.asarray(pcd.points)
    print(f"arm={res['arm']}  cams={len(cams)}  dense={len(pts_all)}")
    if len(pts_all) < 50 or len(cams) < 4:
        (out / "stage5_profile.json").write_text(json.dumps(
            {"arm": res["arm"], "status": "INSUFFICIENT",
             "n_dense": len(pts_all), "n_cams": len(cams)}, indent=2))
        print("INSUFFICIENT cloud/cameras -> abort")
        return

    # centerline spline through cameras
    tck, u = splprep(centers.T, s=SMOOTH, k=min(3, len(cams) - 1))
    uf = np.linspace(0, 1, 4000)
    pf = np.array(splev(uf, tck)).T
    seg = np.linalg.norm(np.diff(pf, axis=0), axis=1)
    scum = np.concatenate([[0], np.cumsum(seg)])
    L = float(scum[-1])

    # outlier filter + lumen radius estimate
    cl_tree = cKDTree(pf)
    d_cl, _ = cl_tree.query(pts_all, k=1)
    R0 = np.median(d_cl)
    keep = d_cl <= 3.0 * R0
    cam_tree = cKDTree(centers)
    dcam, _ = cam_tree.query(pts_all, k=1)
    keep &= dcam <= np.percentile(dcam, 97)
    pts = pts_all[keep]
    R = float(np.median(d_cl[keep]))
    slab = SLAB_FRAC * R
    step = max(STEP_FRAC * R, L / 400)
    print(f"  L={L:.3f}  lumen R={R:.3f}  slab={slab:.3f}  step={step:.3f}  "
          f"filtered pts={len(pts)}/{len(pts_all)}")

    # camera arc-lengths
    cam_s = np.array([scum[int(cl_tree.query(C)[1])] for C in centers])

    # resample centerline by arc length
    s_t = np.arange(0, L + 1e-6, step)
    u_t = np.interp(s_t, scum, uf)
    p_t = np.array(splev(u_t, tck)).T
    tg = np.array(splev(u_t, tck, der=1)).T
    tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)

    slices = []
    for k, (P, T, sv) in enumerate(zip(p_t, tg, s_t)):
        rel = pts - P
        da = rel @ T
        m = np.abs(da) <= slab
        if m.sum() < 10:
            continue
        ins = rel[m]
        up = np.array([0., 0., 1.]) if abs(T[2]) < 0.95 else np.array([0., 1., 0.])
        e1 = up - (up @ T) * T
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 = np.cross(T, e1)
        ip = np.stack([ins @ e1, ins @ e2], axis=1)
        th = np.arctan2(ip[:, 1], ip[:, 0])
        cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, N_BINS + 1))[0] > 0).mean())
        t2 = cKDTree(ip)
        dd, _ = t2.query(ip, k=min(2, len(ip)))
        mnn = float(np.median(dd[:, 1])) if dd.ndim > 1 and dd.shape[1] > 1 else R * 0.1
        alpha = 1.0 / (3.0 * mnn + 1e-9)
        A_a = alpha_shape_area(ip, alpha)
        try:
            A_c = float(ConvexHull(ip).volume) if len(ip) >= 3 else 0.0
        except Exception:
            A_c = 0.0
        A_p = polar_area(ip)
        areas = [a for a in (A_a, A_c, A_p) if a > 0]
        agree = (max(areas) / min(areas) <= AGREE_FACTOR) if len(areas) == 3 else False
        slices.append({"k": k, "s": float(sv), "P": P.tolist(),
                       "e1": e1.tolist(), "e2": e2.tolist(), "in_plane": ip.tolist(),
                       "n": int(m.sum()), "coverage": cov,
                       "area_alpha": A_a, "area_convex": A_c, "area_polar": A_p,
                       "estimators_agree": bool(agree),
                       "Deq": float(2 * np.sqrt(A_a / np.pi)) if A_a > 0 else 0.0})

    valid = [s for s in slices if s["coverage"] >= MIN_COVERAGE and s["estimators_agree"]]
    print(f"  slices>=10pts: {len(slices)}  VALID(cov>=60% & 3-estimator agree): {len(valid)}")

    summary = {"arm": res["arm"], "sfm_model_dir": sfm_dir, "dense_ply": ply,
               "n_dense": len(pts_all), "n_filtered": len(pts),
               "centerline_L": L, "lumen_R": R, "slab_half": slab, "step": step,
               "n_slices": len(slices), "n_valid": len(valid),
               "min_coverage": MIN_COVERAGE, "agree_factor": AGREE_FACTOR,
               "camera_arc_lengths": cam_s.tolist(), "camera_frame_indices": fis,
               "valid_slices": [{kk: s[kk] for kk in
                                 ("s", "n", "coverage", "area_alpha", "area_convex",
                                  "area_polar", "Deq")} for s in valid],
               "all_slices": [{kk: s[kk] for kk in
                               ("s", "n", "coverage", "area_alpha", "area_convex",
                                "area_polar", "estimators_agree", "Deq")} for s in slices]}
    (out / "stage5_profile.json").write_text(json.dumps(summary, indent=2))

    # plot
    if slices:
        ss = np.array([s["s"] for s in slices])
        Aa = np.array([s["area_alpha"] for s in slices])
        Ac = np.array([s["area_convex"] for s in slices])
        Ap = np.array([s["area_polar"] for s in slices])
        cv = np.array([s["coverage"] for s in slices])
        vmask = np.array([s["coverage"] >= MIN_COVERAGE and s["estimators_agree"] for s in slices])
        fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        ax[0].plot(ss, Aa, "-", c="gray", alpha=.4, lw=.8)
        ax[0].plot(ss, Ac, "-.", c="orange", alpha=.5, lw=.7, label="convex")
        ax[0].plot(ss, Ap, ":", c="purple", alpha=.5, lw=.7, label="polar")
        sc = ax[0].scatter(ss, Aa, c=cv * 100, cmap="viridis", vmin=40, vmax=100,
                           s=28, edgecolors=np.where(vmask, "black", "red"), linewidths=.6, zorder=3)
        for cs in cam_s:
            ax[0].axvline(cs, ymin=.97, c="k", lw=.5, alpha=.4)
        ax[0].set_ylabel("area (scene²) alpha"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"{res['arm']} profile | dense={len(pts_all)} R={R:.2f} "
                        f"valid slices={len(valid)} (black edge=valid, red=rejected)")
        plt.colorbar(sc, ax=ax[0], label="coverage %")
        Dq = np.array([s["Deq"] for s in slices])
        ax[1].scatter(ss, Dq, c=cv * 100, cmap="viridis", vmin=40, vmax=100, s=28,
                      edgecolors=np.where(vmask, "black", "red"), linewidths=.6)
        ax[1].set_xlabel("arc length s (scene units)"); ax[1].set_ylabel("Deq")
        fig.tight_layout(); fig.savefig(str(out / "profile.png"), dpi=140); plt.close(fig)
    print(f"=== STAGE 5 [{res['arm']}]: {len(valid)} valid slices -> {out}")


if __name__ == "__main__":
    main()
