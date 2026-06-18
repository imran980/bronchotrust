"""Barbour 3-landmark cross-sections on the COLMAP 32-V2 dense cloud:
glottis (cord plane ~f297), proximal subglottis (~5mm below), distal subglottis
(~10mm below). Glottis located from the camera of frame f297 projected onto the
medial-axis centerline. Per landmark: dense-cloud slice (3D) + medial-axis ring
(2D) + r_std/r_med + closed?. NO stenosis%/CSA/DCE. SCALE IS PROVISIONAL (blade-
gated): the 5/10mm placement uses a stated provisional k mm/scene-unit; the ring
metrics are dimensionless and DO NOT depend on k. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, re
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geometry_debug_32v2 as G

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
GLOTTIS_FRAME = 297
R_PHYS_MM = 2.0           # PROVISIONAL subglottic radius assumption -> sets scale k (FLAGGED)
COV_MIN, RATIO_MAX = 0.60, 0.35


def frame_center(rec, frame):
    for im in rec.images.values():
        if int(re.search(r"f(\d+)", im.name).group(1)) == frame:
            M = np.array(im.cam_from_world().matrix()); return -M[:3, :3].T @ M[:3, 3]
    return None


def section(P, tree, p0, t, Rmed, slab=0.4, ball=6):
    rel = P[tree.query_ball_point(p0, ball * Rmed)] - p0
    ins3 = rel[np.abs(rel @ t) <= slab] + p0
    e1, e2 = G.frame(t)
    rel_in = ins3 - p0
    ip = np.stack([rel_in @ e1, rel_in @ e2], 1) if len(ins3) else np.zeros((0, 2))
    if len(ip):
        th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
        cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
        return ins3, ip, cov, float(np.median(r)), float(np.std(r)), e1, e2
    return ins3, ip, 0.0, 0.0, 0.0, e1, e2


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); C = G.cam_centers(rec); cam_centroid = C.mean(0)
    tree = cKDTree(P)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck_m, uf_m, pf_m = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    pf = np.array(splev(uf_m, tck_m)).T
    scum = G.arclen(pf); L = scum[-1]

    cg = frame_center(rec, GLOTTIS_FRAME)
    ig = int(np.argmin(np.linalg.norm(pf - cg, axis=1))); s_glottis = scum[ig]
    k = R_PHYS_MM / Rmed                       # PROVISIONAL mm per scene unit
    off5, off10 = 5.0 / k, 10.0 / k            # scene units for 5mm / 10mm
    print(f"glottis frame f{GLOTTIS_FRAME}: s_glottis={s_glottis:.2f} (of L={L:.2f} scene units)")
    print(f"PROVISIONAL scale k={k:.3f} mm/unit (from R_PHYS={R_PHYS_MM}mm / Rmed={Rmed:.3f}u) -> 5mm={off5:.2f}u 10mm={off10:.2f}u")

    locs = [("glottis", s_glottis, 0.0), ("proximal_subglottis(+5mm)", s_glottis + off5, 5.0),
            ("distal_subglottis(+10mm)", s_glottis + off10, 10.0)]
    rows = []; cache = []
    for name, s, mm in locs:
        s = min(max(s, scum[1]), scum[-2])
        u = np.interp(s, scum, uf_m); p0 = np.array(splev(u, tck_m)).T
        t = np.array(splev(u, tck_m, der=1)).T; t /= np.linalg.norm(t) + 1e-9
        if (p0 - cam_centroid) @ t < 0: t = -t
        ins3, ip, cov, rm, rs, e1, e2 = section(P, tree, p0, t, Rmed)
        ratio = rs / max(rm, 1e-6); closed = bool(cov >= COV_MIN and ratio <= RATIO_MAX)
        rows.append({"landmark": name, "mm_below_glottis_PROVISIONAL": mm, "s_scene": round(float(s), 2),
                     "n_slab_pts": int(len(ip)), "coverage": round(cov, 3), "r_med_scene": round(rm, 3),
                     "r_std_scene": round(rs, 3), "r_std/r_med": round(ratio, 3), "closed_contour": closed})
        cache.append((name, mm, p0, t, e1, e2, ins3, ip, cov, rm, rs, ratio, closed))
        print(f"  {name:<26} s={s:6.2f}  n={len(ip):<5} cov={cov*100:3.0f}%  r_med={rm:.2f} r_std={rs:.2f} "
              f"r_std/r_med={ratio:.2f}  closed={'YES' if closed else 'no'}")

    # ---- overview 3D: cloud + centerline + 3 planes + glottis camera ----
    Psub = P[np.random.default_rng(0).choice(len(P), min(len(P), 45000), replace=False)]
    fig = plt.figure(figsize=(8, 9)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Psub[:, 0], Psub[:, 1], Psub[:, 2], s=.4, c="0.8", alpha=.25)
    ax.plot(pf[:, 0], pf[:, 1], pf[:, 2], c="blue", lw=2, label="medial centerline")
    ax.scatter(*cg, c="red", s=80, marker="^", label=f"glottis cam f{GLOTTIS_FRAME}")
    cols = {"glottis": "purple", "proximal_subglottis(+5mm)": "green", "distal_subglottis(+10mm)": "orange"}
    for name, mm, p0, t, e1, e2, ins3, ip, cov, rm, rs, ratio, closed in cache:
        half = 1.6 * max(rm, .4)
        q = np.array([p0 + a*half*e1 + b*half*e2 for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        ax.add_collection3d(Poly3DCollection([q], alpha=.25, facecolor=cols[name], edgecolor=cols[name]))
        ax.scatter(ins3[:, 0], ins3[:, 1], ins3[:, 2], s=3, c=cols[name], alpha=.5)
    ax.legend(fontsize=8); ax.set_title("32-V2 Barbour landmarks on dense cloud + medial centerline\n"
                                        "(glottis=purple, +5mm=green, +10mm=orange; 5/10mm PROVISIONAL scale)", fontsize=9)
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass
    fig.tight_layout(); fig.savefig(DIAG / "barbour3_overview3d.png", dpi=120); plt.close(fig)

    # ---- per-landmark: dense-cloud slab (3D) + medial ring (2D) ----
    fig = plt.figure(figsize=(16, 9))
    for j, (name, mm, p0, t, e1, e2, ins3, ip, cov, rm, rs, ratio, closed) in enumerate(cache):
        ax3 = fig.add_subplot(2, 3, j + 1, projection="3d")
        loc = Psub[np.linalg.norm(Psub - p0, axis=1) < 4 * Rmed]
        ax3.scatter(loc[:, 0], loc[:, 1], loc[:, 2], s=1, c="0.7", alpha=.2)
        ax3.scatter(ins3[:, 0], ins3[:, 1], ins3[:, 2], s=4, c=cols[name], alpha=.6)
        half = 1.6 * max(rm, .4)
        q = np.array([p0 + a*half*e1 + b*half*e2 for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        ax3.add_collection3d(Poly3DCollection([q], alpha=.18, facecolor=cols[name], edgecolor=cols[name]))
        r = 4 * Rmed
        ax3.set_xlim(p0[0]-r, p0[0]+r); ax3.set_ylim(p0[1]-r, p0[1]+r); ax3.set_zlim(p0[2]-r, p0[2]+r)
        ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
        ax3.set_title(f"DENSE CLOUD slice — {name}", fontsize=9, color=cols[name])
        ax2 = fig.add_subplot(2, 3, j + 4)
        if len(ip): ax2.scatter(ip[:, 0], ip[:, 1], s=3, c=cols[name], alpha=.55)
        ax2.plot(0, 0, "k+", ms=12, mew=2); ax2.set_aspect("equal"); ax2.grid(alpha=.3)
        ax2.set_title(f"MEDIAL-AXIS slice — {name}\ncov={cov*100:.0f}%  r_std/r_med={ratio:.2f}  "
                      f"{'CLOSED' if closed else 'open/noisy'}", fontsize=9,
                      color=("green" if closed else "crimson"))
    fig.suptitle(f"32-V2 Barbour 3 landmarks (glottis f{GLOTTIS_FRAME}, +5mm, +10mm below). "
                 f"5/10mm PROVISIONAL scale k={k:.2f}mm/u (blade-gated). Ring metrics are scale-FREE.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(DIAG / "barbour3_landmarks.png", dpi=120); plt.close(fig)

    out = {"glottis_frame": GLOTTIS_FRAME, "Rmed_scene": round(Rmed, 3), "L_scene": round(float(L), 2),
           "provisional_scale_mm_per_unit": round(k, 3), "provisional_note": "5/10mm placement only; metrics scale-free; final scale blade-gated",
           "landmarks": rows}
    (DIAG / "barbour3_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved barbour3_overview3d.png + barbour3_landmarks.png + barbour3_report.json -> {DIAG}")


if __name__ == "__main__":
    main()
