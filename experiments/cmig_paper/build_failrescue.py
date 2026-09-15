"""Two figures (matplotlib clouds, GPU-safe while the loop runs):
  fig_failure_taxonomy.png  — per mechanism: two frames + the resulting broken cloud
  fig_rescue.png            — 17-V1 & 26-V2: a frame + failed-window cloud vs rescued-window cloud"""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, subprocess, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import open3d as o3d
from airway_analysis import scatter3d, clean; from figstyle import apply_style, OKABE; apply_style()
R = f"{ROOT}/runs/own_data"; OUT = f"{R}/renders/new"; PLY = f"{R}/all_reconstructions_ply"


def video(c):
    u = c.replace("-", "_")
    for p in [f"{c}.mp4", f"{c} - *.mp4", f"{u}.mp4", f"*{c}*.mp4"]:
        r = [x for x in subprocess.run(["find", "/home/mi3dr/dataset", "-type", "f", "-iname", p, "!", "-iname", "*alib*"], capture_output=True, text=True).stdout.split("\n") if x]
        if r: return r[0]


def frames(c, idx):
    cap = cv2.VideoCapture(video(c)); want = set(idx); got = {}; fi = 0
    while True:
        ok, f = cap.read()
        if not ok or fi > max(idx): break
        if fi in want: got[fi] = cv2.cvtColor(cv2.resize(f, (300, 169)), cv2.COLOR_BGR2RGB)
        fi += 1
    cap.release(); return [got[i] for i in idx]


def cloudpts(fp):
    return np.asarray(clean(fp).points)


# ---------------- failure taxonomy ----------------
FAIL = [("20-V2", [908, 1136], "Dynamic wall collapse: round → slit → star (pale, ring-poor mucosa)"),
        ("14-V1", [714, 1224], "Dynamic wall collapse: posterior wall bulges into a crescent lumen"),
        ("32-V2", [608, 1064], "Mucosal abnormality: inflamed irregular wall + intruding fold + glare"),
        ("24-V1", [608, 2253], "Lesion / no lumen: rounded mass, wet tissue against the lens, froth"),
        ("10-V1", [533, 1386], "Instrument: suction catheter occupies the lumen"),
        ("23-V2", [630, 2522], "Not an airway traversal: inside the ETT / operating room")]
fig = plt.figure(figsize=(10.5, 2.15 * len(FAIL))); gs = fig.add_gridspec(len(FAIL), 3, width_ratios=[1, 1, 1.15], wspace=0.04, hspace=0.42)
for i, (c, idx, lab) in enumerate(FAIL):
    frs = frames(c, idx)
    for j in range(2):
        ax = fig.add_subplot(gs[i, j]); ax.imshow(frs[j]); ax.set_axis_off(); ax.set_title(f"f{idx[j]}", fontsize=8, color="0.4", pad=1)
    ax = fig.add_subplot(gs[i, 2], projection="3d"); scatter3d(ax, cloudpts(f"{PLY}/{c}.ply"), sub=40000, s=1.6, azim=-60)
    fig.text(0.5, 1 - (i + 0.02) / len(FAIL), f"{c} — {lab}", ha="center", va="top", fontsize=9.3, fontweight="bold", color="0.2")
fig.subplots_adjust(left=0.01, right=0.99, top=0.975, bottom=0.01)
fig.savefig(f"{OUT}/fig_failure_taxonomy.png", dpi=125); plt.close(fig); print("saved fig_failure_taxonomy.png")

# ---------------- rescue before/after ----------------
RES = [("17-V1", 900, f"{PLY}/baseline/17-V1.ply", "whole-traversal window\n(fragment, 9/48 gated)", f"{PLY}/17-V1.ply", "descent + carina + pullback loop\n(one model, 25/48; carina + bronchi)"),
       ("26-V2", 1750, f"{PLY}/baseline/26-V2.ply", "whole traversal\n(not measurable, 0/48)", f"{PLY}/26-V2.ply", "centred pullback window\n(measurable tube, 48/48)")]
fig = plt.figure(figsize=(10.5, 4.6 * len(RES))); gs = fig.add_gridspec(len(RES), 3, width_ratios=[0.8, 1, 1], wspace=0.05, hspace=0.2)
for i, (c, fi, bad, badlab, good, goodlab) in enumerate(RES):
    ax = fig.add_subplot(gs[i, 0]); ax.imshow(frames(c, [fi])[0]); ax.set_axis_off(); ax.set_title(f"{c}  (f{fi})", fontsize=10, fontweight="bold", pad=2)
    ax = fig.add_subplot(gs[i, 1], projection="3d"); scatter3d(ax, cloudpts(bad), sub=40000, s=1.6, azim=-60); ax.set_title(badlab, fontsize=9.5, color="0.45")
    ax = fig.add_subplot(gs[i, 2], projection="3d"); scatter3d(ax, cloudpts(good), sub=40000, s=1.6, azim=-60); ax.set_title(goodlab, fontsize=9.5, color=OKABE["green"])
fig.suptitle("Rescue by re-windowing onto the withdrawal pass", fontsize=13, fontweight="bold", y=0.995)
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
fig.savefig(f"{OUT}/fig_rescue.png", dpi=130); print("saved fig_rescue.png")
