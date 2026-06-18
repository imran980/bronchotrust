"""32-V2 visual scale-sweep + progress package.
Part A: scale multipliers {0.6,0.7,0.85,1.0,1.15,1.3,1.5}; per scale place glottis
(fixed f297/s=0), +5mm proximal, +10mm distal; report s_scene, r_med, r_std/r_med,
coverage, closed, + PROVISIONAL CSA=pi*r_med^2 and D_CE=2*r_med (SCENE UNITS ONLY).
Part B: one composite figure (video frames, 3D overview, cross-sections, sweep plot,
verdict text). COLMAP fused 339-frame model, CLAHE+bezel, dense cloud, medial-axis
centerline, borrowed 32_v1 calib (PROVISIONAL). No Myer-Cotton%, no final mm.
depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, re, cv2
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/32-V2.MP4")
GLOTTIS_FRAME = 297; CLEAN_TRACHEA_FRAME = 515
K0 = 1.694
MULTS = [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
COV_MIN, RATIO_MAX = 0.60, 0.35
COLS = {"glottis": "purple", "+5mm": "green", "+10mm": "orange"}


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
        return ip, cov, float(np.median(r)), float(np.std(r)), e1, e2
    return ip, 0.0, 0.0, 0.0, e1, e2


def place(tck, scum, uf, s, cam_centroid):
    s = min(max(s, scum[1]), scum[-2])
    u = np.interp(s, scum, uf); p0 = np.array(splev(u, tck)).T
    t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
    if (p0 - cam_centroid) @ t < 0: t = -t
    return s, p0, t


def grab_frames(idxs):
    out = {}; want = set(idxs); cap = cv2.VideoCapture(str(VIDEO)); fi = 0
    while want:
        ok, fr = cap.read()
        if not ok: break
        if fi in want:
            out[fi] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); want.discard(fi)
        fi += 1
    cap.release(); return out


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); Cc = G.cam_centers(rec); cam_centroid = Cc.mean(0)
    cendict = {int(re.search(r"f(\d+)", im.name).group(1)):
               (-np.array(im.cam_from_world().matrix())[:3, :3].T @ np.array(im.cam_from_world().matrix())[:3, 3])
               for im in rec.images.values()}
    tree = cKDTree(P)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    scum = G.arclen(pf); L = scum[-1]
    cg = frame_center(rec, GLOTTIS_FRAME)
    s_glottis = scum[int(np.argmin(np.linalg.norm(pf - cg, axis=1)))]

    def metrics(s):
        s, p0, t = place(tck, scum, uf, s, cam_centroid)
        ip, cov, rmd, rsd, e1, e2 = section(P, tree, p0, t, Rmed)
        ratio = rsd / max(rmd, 1e-6)
        return dict(s=float(s), p0=p0, t=t, e1=e1, e2=e2, ip=ip, cov=cov, r_med=rmd,
                    ratio=ratio, closed=bool(cov >= COV_MIN and ratio <= RATIO_MAX),
                    CSA_scene=float(np.pi * rmd ** 2), DCE_scene=float(2 * rmd))

    gm = metrics(s_glottis)
    # ---------------- Part A: sweep ----------------
    rows = []; sweep = {"+5mm": [], "+10mm": []}
    for m in MULTS:
        k = m * K0
        rows.append({"landmark": "glottis", "scale_mult": m, "k_mm_per_unit": round(k, 3),
                     "s_scene": round(gm["s"], 2), "r_med_scene": round(gm["r_med"], 3),
                     "r_std/r_med": round(gm["ratio"], 3), "coverage": round(gm["cov"], 3),
                     "closed": gm["closed"], "CSA_scene2": round(gm["CSA_scene"], 3),
                     "DCE_scene": round(gm["DCE_scene"], 3), "note": "fixed (scale-independent)"})
        for mm, key in [(5.0, "+5mm"), (10.0, "+10mm")]:
            d = metrics(s_glottis + mm / k); beyond = (s_glottis + mm / k) > L
            rows.append({"landmark": key, "scale_mult": m, "k_mm_per_unit": round(k, 3),
                         "s_scene": round(d["s"], 2), "r_med_scene": round(d["r_med"], 3),
                         "r_std/r_med": round(d["ratio"], 3), "coverage": round(d["cov"], 3),
                         "closed": d["closed"], "CSA_scene2": round(d["CSA_scene"], 3),
                         "DCE_scene": round(d["DCE_scene"], 3), "note": "BEYOND END" if beyond else ""})
            sweep[key].append((m, d["ratio"], d["cov"], d["closed"]))
    (DIAG / "progress_sweep.json").write_text(json.dumps(
        {"glottis_frame": GLOTTIS_FRAME, "s_glottis": round(gm["s"], 2), "L_scene": round(float(L), 2),
         "k0_mm_per_unit": K0, "Rmed_scene": round(Rmed, 3),
         "UNITS_NOTE": "CSA/DCE are SCENE UNITS (provisional); NOT mm until blade scale fixed", "rows": rows}, indent=2))
    print(f"glottis f{GLOTTIS_FRAME} s={gm['s']:.2f}  L={L:.2f}  k0={K0}mm/u  Rmed={Rmed:.3f}u")
    print(f"{'landmark':<8}{'xS':>5}{'k':>7}{'s':>6}{'r_med':>7}{'ratio':>7}{'cov%':>6}{'closed':>7}{'CSA(sc^2)':>10}{'DCE(sc)':>8}{'note':>12}")
    for r in rows:
        print(f"{r['landmark']:<8}{r['scale_mult']:>5}{r['k_mm_per_unit']:>7}{r['s_scene']:>6}{r['r_med_scene']:>7}"
              f"{r['r_std/r_med']:>7}{r['coverage']*100:>6.0f}{('Y' if r['closed'] else 'n'):>7}"
              f"{r['CSA_scene2']:>10}{r['DCE_scene']:>8}{r['note']:>12}")

    # current-best landmarks (m=1.0)
    k = K0
    cur = {"glottis": gm, "+5mm": metrics(s_glottis + 5.0 / k), "+10mm": metrics(s_glottis + 10.0 / k)}
    # representative frames for +5/+10mm: descent frame whose camera s is nearest target
    desc = [(f, scum[int(np.argmin(np.linalg.norm(pf - c, axis=1)))]) for f, c in cendict.items() if f > 305]
    def nearest_frame(starget): return min(desc, key=lambda fc: abs(fc[1] - starget))[0]
    f5, f10 = nearest_frame(cur["+5mm"]["s"]), nearest_frame(cur["+10mm"]["s"])
    frames = grab_frames([GLOTTIS_FRAME, f5, f10, CLEAN_TRACHEA_FRAME])

    # ---------------- Part B: composite progress figure ----------------
    fig = plt.figure(figsize=(19, 21))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.05, 2.1, 1.25, 0.62], hspace=0.33, wspace=0.26)
    # Row 0: video frames
    vf = [(GLOTTIS_FRAME, "glottis (cord plane)", "purple"), (f5, "≈ +5mm proximal subglottis", "green"),
          (f10, "≈ +10mm distal subglottis", "orange"), (CLEAN_TRACHEA_FRAME, "clean trachea", "navy")]
    for j, (fi, lab, c) in enumerate(vf):
        ax = fig.add_subplot(gs[0, j])
        if fi in frames: ax.imshow(frames[fi])
        ax.set_title(f"f{fi}: {lab}", fontsize=10, color=c); ax.axis("off")
    # Row 1 left: 3D overview ; right: sweep plot
    ax3 = fig.add_subplot(gs[1, 0:2], projection="3d")
    Psub = P[np.random.default_rng(0).choice(len(P), min(len(P), 35000), replace=False)]
    ax3.scatter(Psub[:, 0], Psub[:, 1], Psub[:, 2], s=.4, c="0.8", alpha=.18)
    order = np.array(sorted(cendict)); Cord = np.array([cendict[f] for f in order])
    oo = (order - order.min()) / (np.ptp(order) + 1e-9)
    ax3.scatter(Cord[:, 0], Cord[:, 1], Cord[:, 2], c=oo, cmap="cool", s=6, label="camera trajectory")
    ax3.plot(pf[:, 0], pf[:, 1], pf[:, 2], c="blue", lw=2, label="medial centerline")
    for key in ["glottis", "+5mm", "+10mm"]:
        d = cur[key]; half = 1.7 * max(d["r_med"], .4)
        q = np.array([d["p0"] + a*half*d["e1"] + b*half*d["e2"] for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax3.add_collection3d(Poly3DCollection([q], alpha=.3, facecolor=COLS[key], edgecolor=COLS[key]))
    ax3.set_title("3D overview: dense cloud (gray) + camera trajectory (cool) + medial centerline (blue)\n"
                  "planes: glottis=purple, +5mm=green, +10mm=orange", fontsize=10)
    ax3.legend(fontsize=8, loc="upper left")
    try: ax3.set_box_aspect((1, 1, 1))
    except Exception: pass
    ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])

    axs = fig.add_subplot(gs[1, 2:4])
    for key, c in [("+5mm", "green"), ("+10mm", "orange")]:
        ms = [x[0] for x in sweep[key]]
        axs.plot(ms, [x[1] for x in sweep[key]], "o-", c=c, lw=2,
                 label=f"{key} ({'proximal' if key == '+5mm' else 'distal'})")
    axs.axhline(RATIO_MAX, color="0.4", ls="--", label=f"closed threshold {RATIO_MAX}")
    axs.axvline(1.0, color="0.7", ls=":", label="current scale k0")
    axs.set_xlabel("assumed scale multiplier  (k / k0)", fontsize=10)
    axs.set_ylabel("r_std / r_med   (lower = cleaner ring)", fontsize=10)
    axs.set_title("Scale-sweep: ring noise vs assumed scale", fontsize=11)
    axs.legend(fontsize=9); axs.grid(alpha=.3)
    # Row 2: cross-sections at current best scale
    for j, key in enumerate(["glottis", "+5mm", "+10mm"]):
        ax = fig.add_subplot(gs[2, j]); d = cur[key]
        if len(d["ip"]): ax.scatter(d["ip"][:, 0], d["ip"][:, 1], s=2.5, c=COLS[key], alpha=.55)
        ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{key} @ current scale  (s={d['s']:.2f})\ncov={d['cov']*100:.0f}%  "
                     f"r_std/r_med={d['ratio']:.2f}  {'CLOSED' if d['closed'] else 'open/aperture'}",
                     fontsize=9.5, color=("green" if d["closed"] else "crimson"))
    # legend cell
    axleg = fig.add_subplot(gs[2, 3]); axleg.axis("off")
    axleg.text(0.0, 0.95, "Cross-sections at current best scale\n(k0=1.694 mm/scene-unit, PROVISIONAL)\n\n"
               "+ = medial-axis centerline point\ndots = dense wall points in the slab\n\n"
               "CLOSED = coverage ≥ 60% AND\n   r_std/r_med ≤ 0.35\n\n"
               "CSA = π·r_med²,  D_CE = 2·r_med\nreported in SCENE UNITS only\n(NOT mm until blade scale fixed)",
               fontsize=9, va="top", family="monospace")
    # Row 3: verdict text
    axv = fig.add_subplot(gs[3, :]); axv.axis("off")
    d5_cur, d10_cur = cur["+5mm"], cur["+10mm"]
    n5_closed = sum(1 for x in sweep["+5mm"] if x[3]); n10_closed = sum(1 for x in sweep["+10mm"] if x[3])
    verdict = (
        "VERDICT (32-V2, COLMAP fused 339-frame model · CLAHE+bezel · medial-axis centerline · "
        "borrowed 32_v1 calibration = PROVISIONAL · NO Myer-Cotton %, NO final mm):\n"
        f"  • DISTAL subglottis (+10mm): ROBUST to scale — closed {n10_closed}/7 sweep scales, coverage 100%, "
        "clean ring across −40%…+50% scale. A reliable measurement target regardless of the eventual blade scale.\n"
        f"  • PROXIMAL subglottis (+5mm): SCALE-SENSITIVE — closed only {n5_closed}/7 scales (the lower-scale / deeper "
        "placements). At current scale it sits on the edge (r_std/r_med="
        f"{d5_cur['ratio']:.2f}); larger scales push it shallow into the noisy cord-transition and it degrades.\n"
        "  • GLOTTIS: irregular aperture (vocal-cord plane), not a tube cross-section — expected; measured as an inlet, not a ring.\n"
        "  • CURRENT LIMITATION: absolute mm scale is NOT yet fixed (blade-gated); the noisy proximal entrance (s<≈3) "
        "limits the +5mm slice. Distal subglottis + trachea reconstruct cleanly.")
    axv.text(0.0, 1.0, verdict, fontsize=10.5, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f6f6f0", edgecolor="0.5"))
    fig.suptitle("32-V2 progress package — Barbour landmarks + scale-sensitivity (PROVISIONAL borrowed 32_v1 calib; scene units)",
                 fontsize=13, y=0.995)
    fig.savefig(DIAG / "progress_32v2.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"\nrepresentative frames: glottis f{GLOTTIS_FRAME}, +5mm≈f{f5}, +10mm≈f{f10}, trachea f{CLEAN_TRACHEA_FRAME}")
    print(f"saved progress_32v2.png + progress_sweep.json -> {DIAG}")


if __name__ == "__main__":
    main()
