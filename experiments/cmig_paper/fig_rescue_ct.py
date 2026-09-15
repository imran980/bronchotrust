"""Figure for the overnight CT-cohort rescue: (a) 19-V1, a patient that previously produced no reconstruction at all,
now a measurable tube validated against CT; (b) its caliber profile against the CT trachea; (c) the span guard --
30-V2's near-occlusive stenosis yields a fit with an apparently excellent RMSE that maps only a few millimetres of a
51 mm CT segment, which is why a mapped-span requirement is part of the validity gate.
Reads ct_gt/*.npz and reports/ctscore_rr_*.json. Writes renders/new/fig_rescue_ct.png."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json, subprocess
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from figstyle import apply_style, OKABE, ROLE
import ct_rings as CR
apply_style()

R = f"{ROOT}/runs/own_data"; REP = f"{R}/reports"; OUT = f"{R}/renders/new"; GT = f"{R}/ct_gt"
VID19 = "/home/mi3dr/dataset/New broncho/Videos with Matched CT Scans/19-V1 - 6-10-2025.MP4"
FRAMES = [1120, 1170, 1220]
BLUE, GREEN, VERM, ORANGE = OKABE["blue"], OKABE["green"], OKABE["vermillion"], OKABE["orange"]


def frames_sequential(video, idxs):
    want, got, cap, i = set(idxs), {}, cv2.VideoCapture(video), 0
    while want - set(got):
        ok, f = cap.read()
        if not ok: break
        if i in want: got[i] = f
        i += 1
    cap.release(); return got


def crop(im, thr=10, pad=6):
    g = im.max(axis=2); ys = np.where(g.mean(1) > thr)[0]; xs = np.where(g.mean(0) > thr)[0]
    return im[max(ys[0]-pad,0):ys[-1]+pad, max(xs[0]-pad,0):xs[-1]+pad]


def views(P, rad_pct=95, sub=30000):
    c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); Q = c @ Vt.T
    rad = np.hypot(Q[:, 1], Q[:, 2]); Q = Q[rad < np.percentile(rad, rad_pct)]
    Q = Q[np.random.default_rng(0).choice(len(Q), min(sub, len(Q)), replace=False)]
    return Q[:, 0], Q[:, 1], Q[:, 2], (Q[:, 0] - Q[:, 0].min()) / (np.ptp(Q[:, 0]) + 1e-9)


def draw2d(ax, x, y, t, s=1.4):
    ax.scatter(x, y, c=t, cmap="viridis", s=s, marker=".", linewidths=0, alpha=0.85)
    ax.set_aspect("equal"); ax.set_axis_off()


S19 = json.load(open(f"{REP}/ctscore_rr_19-V1_tr1.json"))
M19 = S19["methods"]["M1 cam-centerline + cov>=0.75"]
S30 = json.load(open(f"{REP}/ctscore_rr_30-V2_v3mid.json"))
M30 = S30["methods"]["M2 partial-arc gated"]
g19 = np.load(f"{GT}/gt_19-V1.npz"); g30 = np.load(f"{GT}/gt_30-V2.npz")

fig = plt.figure(figsize=(16, 8.6))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.15], hspace=0.34, wspace=0.28,
                       left=0.05, right=0.985, top=0.85, bottom=0.09, width_ratios=[1, 1, 1, 1.5])
# (a) frames + cloud
fr = frames_sequential(VID19, FRAMES)
for j, k in enumerate(FRAMES):
    ax = fig.add_subplot(gs[0, j]); ax.set_axis_off()
    if k in fr: ax.imshow(crop(cv2.cvtColor(fr[k], cv2.COLOR_BGR2RGB))); ax.set_title(f"frame {k}", fontsize=9, color="0.4", pad=2)
P = CR.load_clean(f"{R}/retry/rr_19-V1_tr1/dense0/fused.ply")
gsub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 3], width_ratios=[0.55, 1], wspace=0.04)
u, v, w, t = views(P); o = np.argsort(-w)
ax = fig.add_subplot(gsub[0]); draw2d(ax, v[o], u[o], t[o]); ax.set_title("side", fontsize=9, color="0.4", pad=2)
ax = fig.add_subplot(gsub[1]); draw2d(ax, v, w, t); ax.set_title("along the lumen axis", fontsize=9, color="0.4", pad=2)
# (b) 19-V1 profile vs CT
ax = fig.add_subplot(gs[1, :2])
ax.plot(g19["arclength_mm"], g19["dce_mm"], color=ROLE["ct"], lw=2.2, label="CT (TotalSegmentator trachea)")
am, dce = np.array(M19["am"]), np.array(M19["dce"]); o = np.argsort(am)
ax.plot(am[o], dce[o], "-o", color=BLUE, ms=5, lw=1.8, label=f"reconstruction (gated camera-centerline)")
ax.set_xlabel("arclength from the subglottis (mm)"); ax.set_ylabel("D$_{CE}$ (mm)"); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="lower right")
ax.set_title(f"19-V1 — RMSE {M19['rmse']:.2f} mm, bias {M19['bias']:+.2f} mm, n = {M19['n']}, over {M19['length_mm']:.0f} of {S19['ct_arclen_mm']:.0f} mm",
             fontsize=11, color=BLUE, fontweight="bold")
# (c) 30-V2 span-guard demonstration
ax = fig.add_subplot(gs[1, 2:])
ax.plot(g30["arclength_mm"], g30["dce_mm"], color=ROLE["ct"], lw=2.2, label="CT: near-occlusive stenosis")
am, dce = np.array(M30["am"]), np.array(M30["dce"]); o = np.argsort(am)
ax.plot(am[o], dce[o], "-o", color=VERM, ms=5, lw=1.8, label=f"fit: RMSE {M30['rmse']:.2f} mm — but see span")
ax.axvspan(am.min(), am.max(), color=VERM, alpha=0.12, lw=0)
ax.annotate(f"the fit maps only {M30['length_mm']:.1f} mm\nof the {S30['ct_arclen_mm']:.0f} mm CT segment ({100*M30['span_frac_of_CT']:.0f}%)\n→ REJECTED by the span guard",
            xy=(am.mean(), dce.mean()), xytext=(0.42, 0.62), textcoords="axes fraction", color=VERM, fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=1.4))
ax.set_xlabel("arclength from the subglottis (mm)"); ax.set_ylabel("D$_{CE}$ (mm)"); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper right")
ax.set_title("30-V2 — why a low RMSE is not enough", fontsize=11, color=VERM, fontweight="bold")
bb_top = gs[0, 0].get_position(fig); bb_bot = gs[1, 0].get_position(fig); bb_r = gs[1, 2].get_position(fig)
fig.text(0.05, bb_top.y1 + 0.012, "(a) 19-V1: a patient that previously produced no reconstruction in any window", fontsize=12, fontweight="bold", va="bottom")
fig.text(0.05, bb_bot.y1 + 0.055, "(b) validated against CT", fontsize=12, fontweight="bold", va="bottom", color=BLUE)
fig.text(bb_r.x0, bb_bot.y1 + 0.055, "(c) the span guard", fontsize=12, fontweight="bold", va="bottom", color=VERM)
fig.suptitle("Recovering a CT-paired patient, and the safeguard that keeps a good-looking fit honest", fontsize=14, fontweight="bold", y=0.965)
fig.savefig(f"{OUT}/fig_rescue_ct.png", dpi=150); print("saved fig_rescue_ct.png")
