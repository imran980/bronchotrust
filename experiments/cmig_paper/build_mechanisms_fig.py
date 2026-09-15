"""Capture-mechanism figure (replaces the pre-rescue fig_capture_mechanisms.png): one SUCCESS row for contrast, then one row per
confirmed failure mechanism, each with three video frames (decoded sequentially, never seeked), the dense cloud from the side and
along the lumen axis (same orthographic style as the gallery), tier and gated count. Rows and frame indices are the reviewed ones
behind fig_failure_frames.png / capture_notes.tex; 26-V2 is shown as the rescued case (transient collapse breaks the chain, both
sides reconstruct). Clouds come from capture_cache/*_pts.npy (yield-table cleaner). Writes renders/new/fig_capture_mechanisms.png."""
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
from figstyle import apply_style, OKABE
apply_style()

R = f"{ROOT}/runs/own_data"; REP = f"{R}/reports"; CACHE = f"{REP}/capture_cache"; OUT = f"{R}/renders/new"
YIELD = {r["case"]: r for r in json.load(open(f"{REP}/yield_table.json"))}
COL = {"CT-validated": OKABE["green"], "measurable tube": OKABE["blue"], "partially measurable": OKABE["orange"], "not measurable": "0.5"}
ROWS = [  # (case, frames, mechanism title, one-line evidence)
    ("2-V2", (3060, 3180, 3294), "Success — sustained, centred pass over ring-textured mucosa", "withdrawal pass; lumen centred in 98% of frames, no glare; every accepted station is a closed ring"),
    ("20-V2", (908, 1136, 2499), "Dynamic wall motion — lumen collapses round → slit → puckered star", "non-rigid wall on pale, ring-poor mucosa; SfM has no rigid scene to register; 0/48 stations"),
    ("26-V2", (1895, 1912, 1925), "Dynamic wall motion, transient — posterior wall collapses at f1912–1918, then reopens (RESCUED)", "the event breaks the feature chain into two models; the slow centred pullback before it reconstructs to 48/48"),
    ("32-V2", (608, 1064, 1520), "Mucosal abnormality — inflamed, irregular wall with an intruding fold; glare from wall contact", "texture is irregular rather than ringed and the wall moves; 2/48 stations"),
    ("24-V1", (608, 2253, 2619), "Lesion / no lumen — rounded mass, wet tissue against the lens, froth", "no lumen is imaged at all; 0/38 stations"),
    ("10-V1", (533, 853, 1386), "Instrument in the field — suction catheter occupies the lumen for most of the pass", "a moving object in front of an occluded wall; 8/48 stations"),
    ("23-V2", (630, 1892, 2522), "Not an airway traversal — inside the endotracheal tube, drapes, operating room", "printed tube markings and the room are reconstructed instead of the airway; 4/48 stations"),
]


def find_video(c):
    for p in [f"{c}.mp4", f"{c} - *.mp4", f"{c}_*.mp4", f"*{c}*.mp4"]:
        r = [x for x in subprocess.run(["find", "/home/mi3dr/dataset", "-type", "f", "-iname", p, "!", "-iname", "*alib*"], capture_output=True, text=True).stdout.split("\n") if x]
        if r: return sorted(r)[0]
    raise FileNotFoundError(c)


def frames_sequential(video, idxs):
    want, got, cap, i = set(idxs), {}, cv2.VideoCapture(video), 0
    while want - set(got):
        ok, f = cap.read()
        if not ok: break
        if i in want: got[i] = f
        i += 1
    cap.release(); return got


def crop_field(im, thr=10, pad=8):
    g = im.max(axis=2); rows = np.where(g.mean(1) > thr)[0]; cols = np.where(g.mean(0) > thr)[0]
    if len(rows) < 10 or len(cols) < 10: return im
    return im[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]


def views(P, rad_pct=95, sub=35000):
    c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); Q = c @ Vt.T
    rad = np.hypot(Q[:, 1], Q[:, 2]); Q = Q[rad < np.percentile(rad, rad_pct)]
    Q = Q[np.random.default_rng(0).choice(len(Q), min(sub, len(Q)), replace=False)]
    t = (Q[:, 0] - Q[:, 0].min()) / (np.ptp(Q[:, 0]) + 1e-9); return Q[:, 0], Q[:, 1], Q[:, 2], t


def draw2d(ax, x, y, t, s=1.2, margin=0.06):
    ax.scatter(x, y, c=t, cmap="viridis", s=s, marker=".", linewidths=0, alpha=0.85); ax.set_aspect("equal"); ax.set_axis_off()
    mx, my = np.ptp(x) * margin, np.ptp(y) * margin; ax.set_xlim(x.min() - mx, x.max() + mx); ax.set_ylim(y.min() - my, y.max() + my)


fig = plt.figure(figsize=(16, 2.3 * len(ROWS) + 1.0))
outer = gridspec.GridSpec(len(ROWS), 1, left=0.01, right=0.99, top=0.915, bottom=0.05, hspace=0.6)
for i, (c, idxs, title, note) in enumerate(ROWS):
    y = YIELD[c]; col = OKABE["green"] if c == "26-V2" and i == 2 else COL[y["tier"]]
    g = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[i], wspace=0.08, width_ratios=[1, 1, 1, 0.55, 1.05])
    fr = frames_sequential(find_video(c), idxs)
    for j, k in enumerate(idxs):
        ax = fig.add_subplot(g[0, j]); ax.set_axis_off()
        if k in fr: ax.imshow(crop_field(cv2.cvtColor(fr[k], cv2.COLOR_BGR2RGB))); ax.set_title(f"f{k}", fontsize=8, color="0.4", pad=1)
    P = np.load(f"{CACHE}/{c}_pts.npy"); u, v, w, t = views(P); o = np.argsort(-w)
    ax = fig.add_subplot(g[0, 3]); draw2d(ax, v[o], u[o], t[o]); ax.set_title("cloud · side", fontsize=8, color="0.4", pad=1)
    ax = fig.add_subplot(g[0, 4]); draw2d(ax, v, w, t); ax.set_title("cloud · along axis", fontsize=8, color="0.4", pad=1)
    bb = outer[i].get_position(fig)
    fig.text(0.01, bb.y1 + 0.035, f"{c} · {y['tier']} · {y['n_gated']}/{y['n_stations']} gated — {title}", fontsize=10.5, fontweight="bold", color=col, va="bottom")
    fig.text(0.01, bb.y1 + 0.021, note, fontsize=8.5, color="0.35", va="bottom")
    print(f"{c}: frames {sorted(fr)} · {len(P):,} pts · {y['tier']} {y['n_gated']}/{y['n_stations']}", flush=True)
fig.text(0.5, 0.008, "Frames decoded sequentially from the analysed window. Clouds: orthographic views of the dense point cloud from the side (axis vertical) and along the lumen axis\n"
         "(all stations superimposed; closed ring = fully imaged wall), radial 95th-percentile trim, colour = axial position. Motion blur alone is not a mechanism: 33-V1 (80% blurred frames) is a measurable tube.",
         ha="center", va="bottom", fontsize=8, color="0.4", linespacing=1.4)
fig.suptitle("Capture mechanisms behind success and failure — one exemplar per confirmed class", fontsize=12.5, fontweight="bold", y=0.985)
fig.savefig(f"{OUT}/fig_capture_mechanisms.png", dpi=130); print("saved fig_capture_mechanisms.png")
