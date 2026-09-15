"""Failure taxonomy figure (replaces the pre-rescue fig_failure_taxonomy.png): EVERY examination that is not a fully measurable tube
(5 partially measurable + 6 not measurable) plus the rescued 26-V2 as the contrast case, grouped by reviewed mechanism class
(capture_notes.tex: W dynamic wall motion / collapse; A mucosal abnormality or lesion; I instrument; H/L haze, laryngeal-only or
non-airway footage). Per row: three reviewed frames (decoded sequentially, never seeked; indices as in fig_failure_frames.png), the
dense cloud from the side and along the lumen axis (orthographic, same style as the gallery), tier and gated count.
Clouds from capture_cache/*_pts.npy (yield-table cleaner). Writes renders/new/fig_failure_taxonomy.png."""
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
GROUPS = [
    ("W — dynamic wall motion / collapse (non-rigid scene: the wall itself moves)", [
        ("20-V2", (908, 1136, 2499), "lumen collapses round → slit → puckered star on pale, ring-poor mucosa"),
        ("14-V1", (714, 816, 1224), "posterior wall bulges into a crescent lumen along the whole ringed segment"),
        ("31-V1", (541, 974, 1299), "clean subglottic rings, then a smooth wall bulge dominates the field; one-sided cup"),
        ("16-V2", (708, 816, 876), "scope nearly static at the carina while the lumen changes shape → six interleaved rigid models"),
        ("13-V1", (780, 1156, 1340), "vibrating posterior wall → eight interleaved models; best fragment is a carina cup with a thin tracheal stub"),
        ("26-V2", (1895, 1912, 1925), "transient posterior-wall collapse at f1912–1918 breaks the chain; the slow centred pullback before it reconstructs to 48/48 (RESCUED)"),
    ]),
    ("A — mucosal abnormality or lesion (irregular, non-ringed, often wet tissue)", [
        ("32-V2", (608, 1064, 1520), "inflamed, irregular mucosa with an intruding wall fold; glare from wall contact"),
        ("24-V2", (5818, 6492, 8851), "compressed crescent lumen, froth and secretions, oedematous cobblestoned arytenoids; prolonged laryngeal manipulation"),
        ("24-V1", (608, 2253, 2619), "no lumen imaged: wet tissue against the lens with glare and froth; rounded mass and irregular tissue"),
    ]),
    ("I — instrument in the field (moving object in front of an occluded wall)", [
        ("10-V1", (533, 853, 1386), "suction catheter in the lumen for most of the pass; secretions; compliant bulging wall"),
    ]),
    ("H / L — lens haze, laryngeal-only or non-airway footage", [
        ("1-V3", (133, 466, 733), "lens haze (low contrast) with an instrument at the cords; short traversal"),
        ("23-V2", (630, 1892, 2522), "not an airway traversal: inside the endotracheal tube (printed markings), drapes, operating room"),
    ]),
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


PAGES = [("p1", [0]), ("p2", [1, 2, 3])]        # page 1: dynamic wall motion (6 rows); page 2: abnormality, instrument, haze/non-airway (6 rows)
for tag, gidx in PAGES:
    cells = []
    for gi in gidx:
        cells.append(("H", gi)); cells += [("R", gi, c, idxs, note) for c, idxs, note in GROUPS[gi][1]]
    hr = [0.26 if k[0] == "H" else 1.0 for k in cells]; n_r = sum(1 for k in cells if k[0] == "R")
    fig = plt.figure(figsize=(16, 2.6 * n_r + 0.9 * len(gidx) + 1.0))
    outer = gridspec.GridSpec(len(cells), 1, height_ratios=hr, left=0.01, right=0.99, top=0.955, bottom=0.035, hspace=0.62)
    for i, cell in enumerate(cells):
        bb = outer[i].get_position(fig)
        if cell[0] == "H":
            fig.text(0.01, bb.y0 + bb.height * 0.15, GROUPS[cell[1]][0], fontsize=13, fontweight="bold", color="0.15", va="bottom")
            fig.add_artist(plt.Line2D([0.01, 0.99], [bb.y0 + bb.height * 0.08, bb.y0 + bb.height * 0.08], transform=fig.transFigure, color="0.6", lw=0.8)); continue
        _, gi, c, idxs, note = cell; y = YIELD[c]; col = OKABE["green"] if c == "26-V2" else COL[y["tier"]]
        g = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[i], wspace=0.08, width_ratios=[1, 1, 1, 0.55, 1.05])
        fr = frames_sequential(find_video(c), idxs)
        for j, k in enumerate(idxs):
            ax = fig.add_subplot(g[0, j]); ax.set_axis_off()
            if k in fr: ax.imshow(crop_field(cv2.cvtColor(fr[k], cv2.COLOR_BGR2RGB))); ax.set_title(f"f{k}", fontsize=9.5, color="0.4", pad=1)
        P = np.load(f"{CACHE}/{c}_pts.npy"); u, v, w, t = views(P); o = np.argsort(-w)
        ax = fig.add_subplot(g[0, 3]); draw2d(ax, v[o], u[o], t[o]); ax.set_title("cloud · side", fontsize=9.5, color="0.4", pad=1)
        ax = fig.add_subplot(g[0, 4]); draw2d(ax, v, w, t); ax.set_title("cloud · along axis", fontsize=9.5, color="0.4", pad=1)
        fig.text(0.01, bb.y1 + 0.016, f"{c} · {y['tier']} · {y['n_gated']}/{y['n_stations']} gated — {note}", fontsize=11, fontweight="bold", color=col, va="bottom")
        print(f"{tag} {c}: frames {sorted(fr)} · {y['tier']} {y['n_gated']}/{y['n_stations']}", flush=True)
    fig.text(0.5, 0.006, "Frames decoded sequentially; clouds: orthographic views of the dense point cloud from the side (axis vertical) and along the lumen axis (all stations superimposed; closed ring = fully imaged wall), radial 95th-percentile trim, colour = axial position.",
             ha="center", va="bottom", fontsize=9, color="0.4")
    fig.suptitle("Failure taxonomy (1/2) — dynamic wall motion; 26-V2 shown as the rescued contrast case" if tag == "p1" else "Failure taxonomy (2/2) — mucosal abnormality, instrument, haze / non-airway footage", fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(f"{OUT}/fig_failure_taxonomy_{tag}.png", dpi=125); plt.close(fig); print(f"saved fig_failure_taxonomy_{tag}.png")
