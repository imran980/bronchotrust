"""Slice the 7-V1 model-0 dense cloud perpendicular to the LUMEN axis (the airway
the scope looks down), NOT the camera path. Centerline = track of slab-centroids
along the cloud's principal axis (medial-axis approximation). Show in-plane
cloud scatter at 10 nodes; the scope/subglottis end is nearest the cameras.

Visual only. Run in depth-eval env.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OD = Path("/home/mi3dr/projects/bronchotrust/runs/barbour_dense/7-V1")
DIAG = OD / "diag"


def main():
    global OD, DIAG
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--recon", default=str(OD)); a = ap.parse_args()
    OD = Path(a.recon); DIAG = OD / "diag"; DIAG.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(OD / "sparse/0"))
    cams = []
    for im in rec.images.values():
        fi = int(re.search(r"f(\d+)", im.name).group(1))
        M = np.array(im.cam_from_world().matrix()); cams.append((fi, -M[:3, :3].T @ M[:3, 3]))
    cams.sort(); fis = np.array([c[0] for c in cams]); C = np.array([c[1] for c in cams])
    cam_centroid = C.mean(0)
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    # remove gross outliers (robust): keep within 4*MAD of centroid on each axis
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    keep = np.all(np.abs(P - med) < 8 * mad, axis=1)
    P = P[keep]
    print(f"cloud {keep.sum()} (removed {(~keep).sum()} outliers)")

    # lumen axis = cloud PC1
    Pc = P - P.mean(0)
    _, sv, Vt = np.linalg.svd(Pc[np.random.default_rng(0).choice(len(Pc), min(len(Pc), 60000), replace=False)], full_matrices=False)
    axis = Vt[0]
    # orient axis so + points AWAY from cameras (into the airway)
    if (P.mean(0) - cam_centroid) @ axis < 0:
        axis = -axis
    s = (P - P.mean(0)) @ axis
    # centerline: per-bin centroid along axis
    nb = 40
    edges = np.linspace(np.percentile(s, 1), np.percentile(s, 99), nb + 1)
    cen = []
    for i in range(nb):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() > 20:
            cen.append(P[m].mean(0))
    cen = np.array(cen)
    tck, _ = splprep(cen.T, s=0.3 * len(cen), k=3)
    uf = np.linspace(0, 1, 2000); pf = np.array(splev(uf, tck)).T
    scum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    L = scum[-1]
    # scope end = centerline endpoint nearest the cameras
    if np.linalg.norm(pf[0] - cam_centroid) > np.linalg.norm(pf[-1] - cam_centroid):
        uf = uf[::-1]; pf = pf[::-1]; scum = scum[-1] - scum[::-1]
    print(f"lumen centerline L={L:.2f}  (node 0 = scope/subglottis end)")

    s_nodes = np.linspace(0.03 * L, 0.97 * L, 10)
    u_n = np.interp(s_nodes, scum, uf)
    Pn = np.array(splev(u_n, tck)).T
    Tn = np.array(splev(u_n, tck, der=1)).T; Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
    tree = cKDTree(P); Rmed = np.median(cKDTree(pf).query(P)[0])
    slab = 0.4

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    rep = []
    for k, (p0, tg, sv2) in enumerate(zip(Pn, Tn, s_nodes)):
        ax = axes.flat[k]
        rel = P[tree.query_ball_point(p0, 6 * Rmed)] - p0
        da = rel @ tg; ins = rel[np.abs(da) <= slab]
        up = np.array([0, 0, 1.]) if abs(tg[2]) < .95 else np.array([0, 1., 0])
        e1 = up - (up @ tg) * tg; e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(tg, e1)
        ip = np.stack([ins @ e1, ins @ e2], 1) if len(ins) else np.zeros((0, 2))
        if len(ip):
            ax.scatter(ip[:, 0], ip[:, 1], s=1.5, c="0.4", alpha=.5)
            th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
            cov = (np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean()
            rmed = np.median(r); rstd = np.std(r)
        else:
            cov = 0; rmed = rstd = 0
        ax.plot(0, 0, "r+", ms=12, mew=2)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        end = "SCOPE/subglottis" if k <= 2 else ("deep/trachea" if k >= 7 else "mid")
        ax.set_title(f"node {k} [{end}]  n={len(ip)} cov={cov*100:.0f}%\nr_med={rmed:.2f} r_std={rstd:.2f}", fontsize=8)
        rep.append({"node": k, "s": float(sv2), "n": int(len(ip)), "coverage": float(cov),
                    "r_median": float(rmed), "r_std": float(rstd)})
    fig.suptitle("7-V1 model 0: cross-sections PERPENDICULAR TO LUMEN AXIS (gray=wall points, +=lumen center)\n"
                 "node 0 = subglottis end (near scope); a clean ring = low r_std/r_med", fontsize=11)
    fig.tight_layout(); fig.savefig(DIAG / "lumen_slices.png", dpi=130); plt.close(fig)

    # render cloud + lumen centerline
    Pp = P[np.random.default_rng(0).choice(len(P), min(len(P), 90000), replace=False)]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (ix, iy, lx, ly) in zip(ax, [(0, 2, "X", "Z"), (0, 1, "X", "Y"), (1, 2, "Y", "Z")]):
        a.scatter(Pp[:, ix], Pp[:, iy], s=.4, c="0.6", alpha=.5)
        a.plot(pf[:, ix], pf[:, iy], "-", c="blue", lw=2, label="lumen centerline")
        a.scatter(C[:, ix], C[:, iy], s=6, c="red", label="cameras")
        a.set_title(f"{lx}{ly}"); a.set_aspect("equal", "datalim"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("7-V1 model 0: dense cloud + LUMEN centerline (blue) vs cameras (red)")
    fig.tight_layout(); fig.savefig(DIAG / "lumen_centerline.png", dpi=130); plt.close(fig)

    (DIAG / "lumen_report.json").write_text(json.dumps({"L": float(L), "Rmed": float(Rmed), "nodes": rep}, indent=2))
    print("\nnode  end              n_pts  cov%   r_med  r_std  r_std/r_med")
    for r in rep:
        ratio = r["r_std"] / max(r["r_median"], 1e-6)
        print(f"  {r['node']:<4} s={r['s']:5.2f}  n={r['n']:<5} cov={r['coverage']*100:3.0f}%  "
              f"r_med={r['r_median']:.2f} r_std={r['r_std']:.2f}  ratio={ratio:.2f}")
    print(f"saved lumen_slices.png + lumen_centerline.png -> {DIAG}")


if __name__ == "__main__":
    main()
