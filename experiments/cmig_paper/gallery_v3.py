"""Feasibility gallery, re-rendered in the manuscript's common style (render-only; tiers/numbers come from yield_table.json,
which gallery_v2.py computed). Each tile = one reconstruction: dense point cloud from the side (PCA axis vertical, true
proportions) and along the lumen axis (all stations superimposed: closed ring = fully imaged wall, arc = one-sided), radially
trimmed at the 95th percentile, coloured along the axis. Tile frame colour = measurability tier. Points come from the
capture_cache/*_pts.npy cleaned+subsampled cache (same cleaner as the yield table); falls back to the catalogue .ply.
Writes fig_gallery_v2.png (the file the manuscript references)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Patch
from figstyle import apply_style, OKABE
from airway_analysis import scatter3d
apply_style()

R = f"{ROOT}/runs/own_data"; REP = f"{R}/reports"; CACHE = f"{REP}/capture_cache"; PLY = f"{R}/all_reconstructions_ply"; OUT = f"{R}/renders/new"
COL = {"CT-validated": OKABE["green"], "measurable tube": OKABE["blue"], "partially measurable": OKABE["orange"], "not measurable": "0.5"}
ORDER = ["CT-validated", "measurable tube", "partially measurable", "not measurable"]
rows = sorted(json.load(open(f"{REP}/yield_table.json")), key=lambda r: (ORDER.index(r["tier"]), -r["n_gated"]))


def pts(c, cap=150000):
    fp = f"{CACHE}/{c}_pts.npy"
    if os.path.exists(fp): return np.load(fp)
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(f"{PLY}/{c}.ply").points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1); P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8); P = np.asarray(pc.points)
    if len(P) > cap: P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]
    np.save(fp, P.astype(np.float32)); return P


def views(P, rad_pct=95, sub=35000, seed=0):
    """PCA-aligned, radially trimmed, subsampled points: returns (along-axis coord u, perpendicular v, w) and colour t in [0,1].
    Same alignment/trim as airway_analysis.scatter3d; drawn as orthographic 2D projections (side = (v,u), axial = (v,w))."""
    c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); Q = c @ Vt.T
    rad = np.hypot(Q[:, 1], Q[:, 2]); Q = Q[rad < np.percentile(rad, rad_pct)]
    Q = Q[np.random.default_rng(seed).choice(len(Q), min(sub, len(Q)), replace=False)]
    t = (Q[:, 0] - Q[:, 0].min()) / (np.ptp(Q[:, 0]) + 1e-9); return Q[:, 0], Q[:, 1], Q[:, 2], t


def draw2d(ax, x, y, t, s=1.1, margin=0.06):
    ax.scatter(x, y, c=t, cmap="viridis", s=s, marker=".", linewidths=0, alpha=0.85)
    ax.set_aspect("equal"); ax.set_axis_off(); mx, my = np.ptp(x) * margin, np.ptp(y) * margin
    ax.set_xlim(x.min() - mx, x.max() + mx); ax.set_ylim(y.min() - my, y.max() + my)


ncol = 6; nrow = (len(rows) + ncol - 1) // ncol
fig = plt.figure(figsize=(3.3 * ncol, 4.2 * nrow + 1.0))
outer = gridspec.GridSpec(nrow, ncol, left=0.012, right=0.988, top=0.95, bottom=0.062, hspace=0.26, wspace=0.07)
for k, r in enumerate(rows):
    c = r["case"]; col = COL[r["tier"]]; P = pts(c)
    cell = outer[k // ncol, k % ncol]; sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[0.45, 1.0], wspace=0.04)
    u, v, w, t = views(P); o = np.argsort(-w)                      # far side first so the near wall draws on top
    ax = fig.add_subplot(sub[0, 0]); draw2d(ax, v[o], u[o], t[o])   # side view: axis vertical, true proportions
    ax = fig.add_subplot(sub[0, 1]); draw2d(ax, v, w, t)            # along the lumen axis: all stations superimposed
    bb = cell.get_position(fig); pad = 0.004
    fig.add_artist(Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad, transform=fig.transFigure, fill=False, ec=col, lw=2.2, zorder=10))
    fig.text(bb.x0 + bb.width / 2, bb.y1 + 0.006, f"{c}  ·  {r['n_gated']}/{r['n_stations']} gated  ·  cov {r['median_cov']:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=col)
    print(f"{c}: {len(P):,} cached pts, {r['tier']}", flush=True)
fig.legend(handles=[Patch(color=COL[t], label=t) for t in ORDER], loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.004))
fig.text(0.5, 0.034, "each tile: orthographic views of the dense point cloud from the side (axis vertical, true proportions) and along the lumen axis (all stations superimposed; closed ring = fully imaged wall); "
         "radial 95th-percentile trim; colour = position along the axis", ha="center", fontsize=9, color="0.4")
fig.suptitle("Reconstructed pediatric airways — dense point clouds tiered by measurability (gated cross-sections out of 48 interior stations)", fontsize=13.5, fontweight="bold", y=0.985)
fig.savefig(f"{OUT}/fig_gallery_v2.png", dpi=130); print("saved fig_gallery_v2.png")
