"""32-V2 Barbour landmark SCALE-ROBUSTNESS sweep. Glottis fixed at f297 / s=0.
Because the true blade scale is unknown, sweep the assumed scene->mm scale k over
+-50% around the current k0=1.694 mm/scene-unit. At each k, place +5mm and +10mm
below the glottis (arclength offset = mm / k) and re-slice on the medial-axis
centerline. Report per placement: arclength, coverage, r_std/r_med, closed?, r_med
(scene units), + slice image grid. NO CSA / DCE / Myer-Cotton% / final-mm values.
depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, re
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
GLOTTIS_FRAME = 297
K0 = 1.694                                  # current scale (mm per scene unit)
MULTS = [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]   # -40% .. +50% around K0
COV_MIN, RATIO_MAX = 0.60, 0.35


def frame_center(rec, frame):
    for im in rec.images.values():
        if int(re.search(r"f(\d+)", im.name).group(1)) == frame:
            M = np.array(im.cam_from_world().matrix()); return -M[:3, :3].T @ M[:3, 3]
    return None


def section(P, tree, p0, t, Rmed, slab=0.4, ball=6):
    rel = P[tree.query_ball_point(p0, ball * Rmed)] - p0
    ins = rel[np.abs(rel @ t) <= slab]
    e1, e2 = G.frame(t)
    ip = np.stack([ins @ e1, ins @ e2], 1) if len(ins) else np.zeros((0, 2))
    if len(ip):
        th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
        cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
        return ip, cov, float(np.median(r)), float(np.std(r))
    return ip, 0.0, 0.0, 0.0


def place(tck, scum, uf, s, cam_centroid):
    s = min(max(s, scum[1]), scum[-2])
    u = np.interp(s, scum, uf); p0 = np.array(splev(u, tck)).T
    t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
    if (p0 - cam_centroid) @ t < 0: t = -t
    return s, p0, t


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); C = G.cam_centers(rec); cam_centroid = C.mean(0)
    tree = cKDTree(P)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    scum = G.arclen(pf); L = scum[-1]
    cg = frame_center(rec, GLOTTIS_FRAME)
    s_glottis = scum[int(np.argmin(np.linalg.norm(pf - cg, axis=1)))]
    print(f"glottis f{GLOTTIS_FRAME} fixed at s={s_glottis:.2f}; L={L:.2f}; k0={K0} mm/unit; Rmed={Rmed:.3f}u\n")

    rows = []; cells = {"+5mm": [], "+10mm": []}
    for m in MULTS:
        k = m * K0
        for mm, key in [(5.0, "+5mm"), (10.0, "+10mm")]:
            s_target = s_glottis + mm / k
            beyond = s_target > L
            s, p0, t = place(tck, scum, uf, s_target, cam_centroid)
            ip, cov, rmd, rsd = section(P, tree, p0, t, Rmed)
            ratio = rsd / max(rmd, 1e-6); closed = bool(cov >= COV_MIN and ratio <= RATIO_MAX)
            rows.append({"landmark": key, "scale_mult": m, "k_mm_per_unit": round(k, 3),
                         "s_scene": round(float(s), 2), "beyond_centerline_end": bool(beyond),
                         "coverage": round(cov, 3), "r_med_scene": round(rmd, 3),
                         "r_std/r_med": round(ratio, 3), "closed": closed})
            cells[key].append((m, k, s, cov, rmd, ratio, closed, ip, beyond))

    # ---- slice grid: rows = scale mult, cols = [+5mm, +10mm] ----
    nR = len(MULTS)
    fig, axes = plt.subplots(nR, 2, figsize=(8, 3.1 * nR))
    for ci, key in enumerate(["+5mm", "+10mm"]):
        for ri, (m, k, s, cov, rmd, ratio, closed, ip, beyond) in enumerate(cells[key]):
            ax = axes[ri, ci]
            col = "green" if key == "+5mm" else "orange"
            if len(ip): ax.scatter(ip[:, 0], ip[:, 1], s=2, c=col, alpha=.5)
            ax.plot(0, 0, "k+", ms=10, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            flag = "  [BEYOND END]" if beyond else ""
            ax.set_title(f"{key}  scale x{m:.2f} (k={k:.2f})  s={s:.2f}{flag}\n"
                         f"cov={cov*100:.0f}%  r_std/r_med={ratio:.2f}  r_med={rmd:.2f}  "
                         f"{'CLOSED' if closed else 'open'}", fontsize=8,
                         color=("green" if closed else "crimson"))
    fig.suptitle(f"32-V2 Barbour landmarks vs ASSUMED SCALE (glottis fixed f{GLOTTIS_FRAME}/s={s_glottis:.1f}). "
                 f"k0={K0}mm/u; sweep -40%..+50%.\nLEFT=+5mm proximal subglottis, RIGHT=+10mm distal. "
                 f"Ring metrics are scale-FREE; only PLACEMENT moves with k.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(DIAG / "scalesweep_slices.png", dpi=115); plt.close(fig)

    # ---- summary: ratio & coverage vs scale ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for key, col in [("+5mm", "green"), ("+10mm", "orange")]:
        ms = [c[0] for c in cells[key]]
        ax[0].plot(ms, [c[5] for c in cells[key]], "o-", c=col, label=f"{key} proximal" if key == "+5mm" else f"{key} distal")
        ax[1].plot(ms, [c[3]*100 for c in cells[key]], "o-", c=col, label=key)
    ax[0].axhline(RATIO_MAX, color="0.5", ls="--", label=f"closed threshold {RATIO_MAX}")
    ax[0].axvline(1.0, color="0.7", ls=":"); ax[0].set_xlabel("scale multiplier (k / k0)")
    ax[0].set_ylabel("r_std/r_med"); ax[0].set_title("Ring noise vs assumed scale"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axhline(COV_MIN*100, color="0.5", ls="--"); ax[1].axvline(1.0, color="0.7", ls=":")
    ax[1].set_xlabel("scale multiplier (k / k0)"); ax[1].set_ylabel("coverage %"); ax[1].set_ylim(0, 105)
    ax[1].set_title("Coverage vs assumed scale"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("32-V2 subglottic landmark robustness to scale uncertainty (glottis fixed)", fontsize=11)
    fig.tight_layout(); fig.savefig(DIAG / "scalesweep_summary.png", dpi=130); plt.close(fig)

    (DIAG / "scalesweep_report.json").write_text(json.dumps(
        {"glottis_frame": GLOTTIS_FRAME, "s_glottis": round(float(s_glottis), 2), "L_scene": round(float(L), 2),
         "k0_mm_per_unit": K0, "Rmed_scene": round(Rmed, 3), "rows": rows}, indent=2))

    print(f"{'landmark':<8}{'xScale':>7}{'k':>7}{'s':>7}{'cov%':>7}{'r_med':>7}{'ratio':>7}{'closed':>8}{'note':>14}")
    for r in rows:
        note = "BEYOND END" if r["beyond_centerline_end"] else ""
        print(f"{r['landmark']:<8}{r['scale_mult']:>7}{r['k_mm_per_unit']:>7}{r['s_scene']:>7}"
              f"{r['coverage']*100:>6.0f}{r['r_med_scene']:>7}{r['r_std/r_med']:>7}"
              f"{('Y' if r['closed'] else 'n'):>8}{note:>14}")
    print("\nsaved scalesweep_slices.png + scalesweep_summary.png + scalesweep_report.json")


if __name__ == "__main__":
    main()
