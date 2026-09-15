"""26-V2 wall-collapse figure (replaces the raw 8-frame strip fig_wallcollapse_26-V2.png). Three panels:
 (a) frames across the event f1880-1950 (decoded sequentially), the collapse f1912-1918 highlighted;
 (b) per-frame dark-lumen area (largest dark blob inside the endoscope field, as a fraction of the field) over f1840-1990 —
     the quantitative signature of the transient posterior-wall collapse — with the SfM model membership of each frame;
 (c) the two reconstructions either side of the event (segment A f1620-1901, segment B f1924-2065): side and along-axis views.
Writes renders/new/fig_wallcollapse_26-V2.png."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from figstyle import apply_style, OKABE
apply_style()

R = f"{ROOT}/runs/own_data"; REP = f"{R}/reports"; CACHE = f"{REP}/capture_cache"; OUT = f"{R}/renders/new"
VIDEO = "/home/mi3dr/dataset/New broncho/more videos with calib/26-V2.MP4"
STRIP = [1880, 1895, 1905, 1912, 1918, 1925, 1935, 1950]; EVENT = (1912, 1918); TRACE = (1840, 1990)
SEG_A = (1620, 1901); SEG_B = (1924, 2065)
EVA = json.load(open(f"{R}/retry/rr_26-V2_pull/eval.json")); EVB = json.load(open(f"{R}/retry/rr_26-V2_pullB/eval.json"))
PLY_B = next(p for p in (f"{R}/all_reconstructions_ply/baseline/26-V2_segmentB.ply", f"{R}/retry/rr_26-V2_pullB/dense0/fused.ply") if os.path.exists(p))


def field_and_lumen(f, dark=0.45):
    """endoscope field = bright circle in the black 16:9 frame; lumen = largest connected dark blob inside it. Returns (area fraction, crop)."""
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); field = (f.max(axis=2) > 10).astype(np.uint8)
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    med = np.median(g[field > 0]); lum = ((g < dark * med) & (field > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lum)
    area = stats[1:, cv2.CC_STAT_AREA].max() / max(field.sum(), 1) if n > 1 else 0.0
    ys, xs = np.where(field > 0); crop = f[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return float(area), crop


frames, trace = {}, {}; cap = cv2.VideoCapture(VIDEO); i = 0
while i <= max(TRACE[1], max(STRIP)):
    ok, f = cap.read()
    if not ok: break
    if TRACE[0] <= i <= TRACE[1] or i in STRIP:
        a, crop = field_and_lumen(f); trace[i] = a
        if i in STRIP: frames[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    i += 1
cap.release(); print(f"decoded to frame {i-1}; strip frames {sorted(frames)}; trace {len(trace)} frames")


def load_pts(path, cap_n=150000):
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(path).points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1); P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8); P = np.asarray(pc.points)
    return P[np.random.default_rng(0).choice(len(P), cap_n, replace=False)] if len(P) > cap_n else P


def views(P, rad_pct=95, sub=35000):
    c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); Q = c @ Vt.T
    rad = np.hypot(Q[:, 1], Q[:, 2]); Q = Q[rad < np.percentile(rad, rad_pct)]
    Q = Q[np.random.default_rng(0).choice(len(Q), min(sub, len(Q)), replace=False)]
    t = (Q[:, 0] - Q[:, 0].min()) / (np.ptp(Q[:, 0]) + 1e-9); return Q[:, 0], Q[:, 1], Q[:, 2], t


def draw2d(ax, x, y, t, s=1.3, margin=0.06):
    ax.scatter(x, y, c=t, cmap="viridis", s=s, marker=".", linewidths=0, alpha=0.85); ax.set_aspect("equal"); ax.set_axis_off()
    mx, my = np.ptp(x) * margin, np.ptp(y) * margin; ax.set_xlim(x.min() - mx, x.max() + mx); ax.set_ylim(y.min() - my, y.max() + my)


PA = np.load(f"{CACHE}/26-V2_pts.npy"); PB = load_pts(PLY_B); print(f"segment A {len(PA):,} pts · segment B {len(PB):,} pts ({PLY_B.split('/')[-1]})")
cA, cB, cE = OKABE["blue"], OKABE["orange"], OKABE["vermillion"]

# data-driven event interval: longest contiguous run of frames with dark-lumen area < 60% of the pre-event baseline (median over f1840-1894)
ks = np.array(sorted(trace)); va = np.array([trace[k] for k in ks]) * 100; base = np.median(va[ks < 1895]); low = va < 0.6 * base
runs, start = [], None
for k, l in zip(ks, low):
    if l and start is None: start = k
    if not l and start is not None: runs.append((start, k - 1)); start = None
if start is not None: runs.append((start, ks[-1]))
EV = max(runs, key=lambda r: r[1] - r[0]); print(f"baseline {base:.1f}% of field; event run (area < 60% of baseline): f{EV[0]}-{EV[1]}; runs {runs}")

fig = plt.figure(figsize=(15, 11.0))
gs = gridspec.GridSpec(3, 8, height_ratios=[1.0, 1.2, 1.3], hspace=0.5, wspace=0.12, left=0.05, right=0.98, top=0.9, bottom=0.05)
gsb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], height_ratios=[1, 0.16], hspace=0.06)
# (a) strip
for j, k in enumerate(STRIP):
    ax = fig.add_subplot(gs[0, j]); ax.set_axis_off()
    if k in frames: ax.imshow(frames[k])
    inev = EV[0] <= k <= EV[1]
    ax.set_title(f"f{k}", fontsize=10, fontweight="bold" if inev else "normal", color=cE if inev else "0.3", pad=3)
    if inev: ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=cE, lw=3))
# (b) trace
axT = fig.add_subplot(gsb[0]); axT.axvspan(EV[0], EV[1], color=cE, alpha=0.15, lw=0)
axT.axhline(base, color="0.6", ls="--", lw=1); axT.text(1862, base + 0.25, "pre-event baseline", fontsize=8, color="0.45", va="bottom")
axT.plot(ks, va, "-", color="0.25", lw=1.6); axT.scatter(ks, va, s=12, color="0.25")
for k in STRIP: axT.axvline(k, color="0.75", lw=0.7, ls=":")
axT.set_xlim(TRACE); axT.set_ylabel("dark-lumen area (% of field)"); axT.grid(alpha=0.25); axT.tick_params(labelbottom=False)
axT.annotate(f"posterior wall collapses\n(dark-lumen area < 60% of baseline, f{EV[0]}–{EV[1]})", xy=(EV[1] + 0.5, 5.0), xytext=(1958, 5.5), color=cE, fontsize=9.5, fontweight="bold",
             ha="center", va="center", arrowprops=dict(arrowstyle="->", color=cE, lw=1.2))
# model-membership bar in its own row
axB = fig.add_subplot(gsb[1], sharex=axT); axB.set_ylim(0, 1); axB.set_yticks([]); axB.set_xlabel("frame index", loc="right", fontsize=9, color="0.35")
for sp in axB.spines.values(): sp.set_visible(False)
axB.add_patch(Rectangle((TRACE[0], 0.15), SEG_A[1] - TRACE[0], 0.7, color=cA)); axB.add_patch(Rectangle((SEG_B[0], 0.15), TRACE[1] - SEG_B[0], 0.7, color=cB))
axB.add_patch(Rectangle((SEG_A[1], 0.15), SEG_B[0] - SEG_A[1], 0.7, color="0.82"))
axB.text(TRACE[0] + 2, 0.5, f"SfM model A  (f{SEG_A[0]}–{SEG_A[1]}, {EVA['registered']} frames)", va="center", fontsize=8.5, color="white", fontweight="bold")
axB.text(TRACE[1] - 2, 0.5, f"SfM model B  (f{SEG_B[0]}–{SEG_B[1]}, {EVB['registered']} frames)", va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
axB.text((SEG_A[1] + SEG_B[0]) / 2, 0.5, "no frame registers", ha="center", va="center", fontsize=7.5, color="0.3")
# (c) the two reconstructions
for col0, (P, lab, ev, cc) in zip((0, 4), ((PA, "segment A — slow centred pullback before the event", EVA, cA), (PB, "segment B — after the wall reopens", EVB, cB))):
    u, v, w, t = views(P); o = np.argsort(-w)
    ax1 = fig.add_subplot(gs[2, col0:col0 + 1]); draw2d(ax1, v[o], u[o], t[o]); ax1.set_title("side", fontsize=9, color="0.4", pad=2)
    ax2 = fig.add_subplot(gs[2, col0 + 1:col0 + 4]); draw2d(ax2, v, w, t); ax2.set_title("along the lumen axis", fontsize=9, color="0.4", pad=2)
    bb1, bb2 = ax1.get_position(), ax2.get_position()
    fig.text((bb1.x0 + bb2.x1) / 2, bb2.y1 + 0.042, lab, ha="center", fontsize=10.5, fontweight="bold", color=cc)
    fig.text((bb1.x0 + bb2.x1) / 2, bb2.y1 + 0.028, f"f{ev['f_lo']}–{ev['f_hi']} · {ev['registered']} frames in one model · reproj {ev['reproj']:.2f} px · {ev['n_gated']}/{ev['n_stations']} stations accepted · median coverage {ev['median_cov']:.2f}",
             ha="center", fontsize=8.5, color="0.35")
for row, lab, dy in ((0, "(a) frames across the event — the posterior membranous wall bulges in, nearly closes the lumen, then reopens", 0.04),
                     (1, "(b) dark-lumen area per frame, and the SfM model each frame registered to", 0.02),
                     (2, "(c) the two rigid reconstructions either side of the event (this is the rescued 26-V2)", 0.075)):
    bb = gs[row, 0].get_position(fig); fig.text(0.05, bb.y1 + dy, lab, fontsize=11, fontweight="bold", va="bottom")
fig.suptitle("26-V2 — a transient, non-rigid wall collapse breaks the reconstruction into two models", fontsize=13, fontweight="bold", y=0.985)
fig.text(0.5, 0.008, "Frames decoded sequentially. Dark-lumen area = largest connected region darker than 0.45× the field median, inside the endoscope field (the f1901 spike is a glare flash). "
         "Clouds: orthographic views, radial 95th-percentile trim, colour = axial position.", ha="center", fontsize=8, color="0.4")
fig.savefig(f"{OUT}/fig_wallcollapse_26-V2.png", dpi=140); print("saved fig_wallcollapse_26-V2.png")
