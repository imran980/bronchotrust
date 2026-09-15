"""'Take-home story' pipeline figure (replaces the box-and-arrow schematic as fig_pipeline.png).
Top band: the five stages shown on REAL data from one examination (2-V2): video frame -> SfM (camera path + sparse points,
pinned intrinsics) -> dense MVS cloud (side + along-axis) -> gated cross-sections -> gated CSA profile with %obstruction;
under each stage the design choices that make the measurement defensible.
Bottom band: what the layer delivers — (i) calibre validated against CT (2-V2 profile + pooled stats over 3 patients),
(ii) scale-free %obstruction across the measurable cohort with the empirical noise floor, (iii) cohort yield and the failure
mechanisms (capture behaviour, not optics). Every number is read from the same JSONs as the tables."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json, re
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, Patch
from scipy.ndimage import median_filter
import pycolmap
from figstyle import apply_style, OKABE, PATIENT, ROLE
from csa_partialarc import profile
import ct_rings as CR
apply_style()

R = f"{ROOT}/runs/own_data"; REP = f"{R}/reports"; OUT = f"{R}/renders/new"; CAT = f"{R}/all_reconstructions_ply"
VIDEO = "/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4"; FRAME = 3120; WS = f"{R}/retry/rr_2-V2"
COV, RESID, AGREE, END = 0.75, 0.15, 0.30, 0.10
BLUE, GREEN, ORANGE, VERM, GREY = OKABE["blue"], OKABE["green"], OKABE["orange"], OKABE["vermillion"], "0.45"
TIERCOL = {"CT-validated": GREEN, "measurable tube": BLUE, "partially measurable": ORANGE, "not measurable": "0.6"}
YLD = json.load(open(f"{REP}/yield_table.json")); ROB = json.load(open(f"{REP}/robust_csa_table.json"))
POOL = json.load(open(f"{REP}/ct3_pooled_stats_v2.json")); M1 = json.load(open(f"{REP}/rerun_eval_2-V2.json"))["methods"]["M1 cam-centerline + cov>=0.75"]
EVAL2 = json.load(open(f"{REP}/rerun_eval_2-V2_f2870-3370.json"))["methods"]["M1 cam-centerline + cov>=0.75"]


# ---------- data for the top band (2-V2, single-pass withdrawal cloud = catalogue) ----------
def frame_sequential(video, idx):
    cap = cv2.VideoCapture(video); i = 0
    while True:
        ok, f = cap.read()
        if not ok: return None
        if i == idx: cap.release(); return f
        i += 1


def crop_field(im, thr=10, pad=8):
    g = im.max(axis=2); rows = np.where(g.mean(1) > thr)[0]; cols = np.where(g.mean(0) > thr)[0]
    return im[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]


fr = crop_field(cv2.cvtColor(frame_sequential(VIDEO, FRAME), cv2.COLOR_BGR2RGB))
P = CR.load_clean(f"{CAT}/2-V2.ply")                                  # yield-table cleaner
mu = P.mean(0); _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)   # one PCA frame for every 3-D panel of this case
def proj(X): Q = (X - mu) @ Vt.T; return Q[:, 0], Q[:, 1], Q[:, 2]     # u along axis, v/w perpendicular
u, v, w = proj(P); rad = np.hypot(v, w); keep = rad < np.percentile(rad, 95); u, v, w = u[keep], v[keep], w[keep]
sub = np.random.default_rng(0).choice(len(u), min(40000, len(u)), replace=False); t = (u - u.min()) / np.ptp(u)
rec = pycolmap.Reconstruction(f"{WS}/sparse/0")
X = np.array([p.xyz for p in rec.points3D.values()]); cams = sorted(((int(re.search(r"f(\d+)", im.name).group(1)), np.asarray(im.projection_center())) for im in rec.images.values()), key=lambda x: x[0])
Cc = np.array([c[1] for c in cams]); su, sv, sw = proj(X); srad = np.hypot(sv, sw); sk = srad < np.percentile(srad, 95); cu, cv_, cw = proj(Cc)
print(f"2-V2: frame {FRAME}, dense {len(P):,} pts, sparse {len(X):,} pts, {len(Cc)} cameras, reproj {rec.compute_mean_reprojection_error():.2f} px")
# rings + profile (same gates as robust_csa)
rows, cl = profile(P, nb=60); t0 = float(cl[:, 0].min())
tt = np.array([r["t"] for r in rows]); cov = np.array([r["cov"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows]); ce = np.array([r["csa_ell"] for r in rows]); res = np.array([r["resid"] for r in rows])
with np.errstate(invalid="ignore", divide="ignore"): agree = np.abs(ce - cc) / cc
ok = (cov >= COV) & (res < RESID) & np.isfinite(agree) & (agree < AGREE) & np.isfinite(cc) & (cc > 0)
n = len(tt); lo, hi = int(np.floor(END * n)), int(np.ceil((1 - END) * n)); inter = np.zeros(n, bool); inter[lo:hi] = True; ok &= inter
sm = median_filter(cc[ok], size=3, mode="nearest"); a_min = float(sm.min()); a_ref = float(np.percentile(sm, 90)); tmin = tt[ok][int(np.argmin(sm))]; pct = (1 - a_min / a_ref) * 100
st, _ = CR.station_data(P); tacc = tt[ok]; tol = 1e-3 * np.ptp(tt)
ok_st = sorted([s_ for s_ in st if s_ and np.min(np.abs(tacc - (s_["t"] + t0))) < tol], key=lambda s_: s_["t"]); ts = np.array([s_["t"] for s_ in ok_st])
picks = [ok_st[int(np.argmin(np.abs(ts - (ts.min() + q * (ts.max() - ts.min())))))] for q in (0.25, 0.5, 0.75)]
print(f"  gated {int(ok.sum())}/{n} stations, A_min {a_min:.2f}, A_ref {a_ref:.2f}, %obs {pct:.0f}% (table {ROB['2-V2']['pct_obstruction']:.0f}%)")

# ---------- figure: one row, five panels, one label each, three numbers ----------
fig = plt.figure(figsize=(20, 5.6))
top = gridspec.GridSpec(1, 5, left=0.01, right=0.99, top=0.86, bottom=0.06, wspace=0.28, width_ratios=[1, 1.05, 1.25, 1.1, 1.3])
STAGE = ["video", "camera path", "dense lumen cloud", "gated cross-sections", "calibre profile"]
th = np.linspace(0, 2 * np.pi, 200)
# 1 frame
ax = fig.add_subplot(top[0, 0]); ax.imshow(fr); ax.set_axis_off()
# 2 sfm: sparse points + camera path (colour = time) — number 1
ax = fig.add_subplot(top[0, 1])
ax.scatter(sv[sk], su[sk], s=4, color="0.75", linewidths=0, alpha=0.8)
ax.scatter(cv_, cu, c=np.linspace(0, 1, len(cu)), cmap="plasma", s=26, linewidths=0, zorder=3)
ax.set_aspect("equal"); ax.set_axis_off()
ax.text(0.5, -0.02, f"{len(Cc)}/{len(Cc)} frames", transform=ax.transAxes, fontsize=13, fontweight="bold", color="0.2", ha="center", va="top")
# 3 dense: side + along-axis
g3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top[0, 2], width_ratios=[0.5, 1], wspace=0.04)
ax = fig.add_subplot(g3[0, 0]); o = np.argsort(-w[sub]); ax.scatter(v[sub][o], u[sub][o], c=t[sub][o], cmap="viridis", s=1.6, marker=".", linewidths=0, alpha=0.85); ax.set_aspect("equal"); ax.set_axis_off()
ax2 = fig.add_subplot(g3[0, 1]); ax2.scatter(v[sub], w[sub], c=t[sub], cmap="viridis", s=1.6, marker=".", linewidths=0, alpha=0.85); ax2.set_aspect("equal"); ax2.set_axis_off()
# 4 rings — number 2
g4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=top[0, 3], wspace=0.08, hspace=0.08)
for j, stn in enumerate(picks):
    ax = fig.add_subplot(g4[j // 2, j % 2]); xy = stn["xy"]; Rr = stn["R"]
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color=BLUE, alpha=0.55, linewidths=0, zorder=2); ax.plot(Rr * np.cos(th), Rr * np.sin(th), color="0.15", lw=1.3, ls="--", zorder=1)
    lim = Rr * 1.45; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(GREEN); sp.set_linewidth(1.6)
ax = fig.add_subplot(g4[1, 1]); ax.set_axis_off()
ax.text(0.5, 0.5, f"{int(ok.sum())} stations\naccepted", ha="center", va="center", fontsize=13, fontweight="bold", color=GREEN, transform=ax.transAxes)
# 5 profile — number 3
ax = fig.add_subplot(top[0, 4])
ax.plot(tt, cc, "-", color="0.85", lw=1.0, zorder=0); ax.scatter(tt[~ok], cc[~ok], s=16, color="0.72")
ax.scatter(tt[ok], cc[ok], s=30, c=cov[ok], cmap="viridis", vmin=0.6, vmax=1.0)
ax.axhline(a_ref, color=BLUE, ls="--", lw=1.3); ax.axhline(a_min, color=VERM, ls="--", lw=1.3); ax.plot([tmin], [a_min], "v", color=VERM, ms=10)
ax.text(tt.max(), a_ref, "A$_{ref}$ ", color=BLUE, fontsize=10, va="bottom", ha="right"); ax.text(tt.max(), a_min, "A$_{min}$ ", color=VERM, fontsize=10, va="top", ha="right")
ax.set_ylim(0, np.nanpercentile(cc[ok], 99) * 1.45); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("along the airway", fontsize=10, color="0.35"); ax.set_ylabel("cross-sectional area", fontsize=10, color="0.35")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.text(0.03, 0.96, f"%obstruction {pct:.0f}%", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", color="0.2")
# stage labels + arrows
tops = [top[0, i].get_position(fig) for i in range(5)]
for i, bb in enumerate(tops):
    fig.text((bb.x0 + bb.x1) / 2, 0.90, STAGE[i], ha="center", va="bottom", fontsize=15, fontweight="bold", color="0.15")
    if i < 4:
        nb = tops[i + 1]; fig.add_artist(FancyArrowPatch((bb.x1 + 0.003, 0.47), (nb.x0 - 0.003, 0.47), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color="0.4", lw=1.8))
fig.savefig(f"{OUT}/fig_pipeline.png", dpi=160); print("saved fig_pipeline.png")
