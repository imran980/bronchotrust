"""Methods figures for the manuscript (figstyle): fig_pipeline (schematic), fig_crosssection (one
gated partial-arc measurement on a real slab), fig_capture (centred vs off-centre capture -> tube vs
degraded cloud, 50-V2), plus the calibration table (markdown + LaTeX) from the pinned-intrinsics files.
Outputs go to runs/own_data/renders/new/ (the paper's \graphicspath)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, glob, os, re
import numpy as np, cv2, open3d as o3d, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge, Circle
from figstyle import apply_style, OKABE
from csa_partialarc import centerline, fit_circle, fit_ellipse_area
from airway_analysis import scatter3d
apply_style()
R = f"{ROOT}/runs"; OUT = f"{R}/own_data/renders/new"; REP = f"{R}/own_data/reports"
BLUE, ORANGE, GREEN, GREY, VERM = OKABE["blue"], OKABE["orange"], OKABE["green"], "0.45", OKABE["vermillion"]

# ---------------- (a) pipeline schematic ----------------
fig, ax = plt.subplots(figsize=(12.5, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fc="#f1f4f8", ec="0.35", fs=9.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="0.15", wrap=True)
def arrow(x1, y1, x2, y2, c="0.35"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=c, shrinkA=2, shrinkB=2))
main = [("Bronchoscopy\nvideo", 1), ("Sequential decode\nCLAHE · quality gate\n→ frame window", 13), ("COLMAP SfM\npinned intrinsics\nexhaustive matching", 29),
        ("Multi-view stereo\ndense lumen cloud", 45), ("Centreline\n(camera path /\nmedial axis)", 59), ("Perpendicular\ncross-sections\npartial-arc fit + gates", 73)]
for t, x in main:
    box(x, 22, 11, 12, t)
for i in range(len(main) - 1):
    arrow(main[i][1] + 11, 28, main[i + 1][1], 28)
box(87, 22, 12, 12, "$D_{CE}$ / CSA\nprofile", fc="#e8f1fb", ec=BLUE, bold=True); arrow(84, 28, 87, 28)
# calibration input
box(29, 3, 11, 8, "Per-session\ncheckerboard\ncalibration", fc="#fff6e5", ec=ORANGE, fs=8.6); arrow(34.5, 11, 34.5, 22, c=ORANGE)
# two output branches
box(56, 2, 22, 12, "With matched CT:\nsegmentation → CT $D_{CE}$ profile\nisotropic Sim(3) (scale + flip, constrained)\n→ absolute mm · self-consistency check", fc="#e9f6f1", ec=GREEN, fs=7.9)
box(80, 2, 19, 12, "Without CT:\nscale-free %obstruction\n= 1 − A$_{min}$/A$_{ref}$\n(strict gates; ends & spikes excluded)", fc="#f3f3f3", ec=GREY, fs=7.9)
arrow(90, 22, 67, 14, c=GREEN); arrow(94, 22, 89.5, 14, c=GREY)
ax.text(1, 37.5, "Monocular airway calibre pipeline — one unchanged pipeline for all cohorts", fontsize=11.5, fontweight="bold", color="0.15")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline_boxes.png", dpi=170); plt.close(fig); print("fig_pipeline_boxes.png (schematic; the manuscript figure fig_pipeline.png is built by build_pipeline_story.py)")

# ---------------- (b) one gated partial-arc cross-section (2-V2 re-run, full rings) ----------------
fp = f"{R}/own_data/retry/rr_2-V2/dense0/fused.ply"
P = np.asarray(o3d.io.read_point_cloud(fp).points); m0 = np.median(P, 0); dd = np.linalg.norm(P - m0, axis=1)
P = P[dd < np.median(dd) + 4 * np.median(np.abs(dd - np.median(dd)))]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pcd, _ = pcd.remove_statistical_outlier(24, 1.8); pcd, _ = pcd.remove_radius_outlier(16, np.median(dd) * 0.03)
P = np.asarray(pcd.points); cl, axv, Vt = centerline(P, 60); e1, e2 = Vt[1], Vt[2]; Pc = P - P.mean(0); tall = Pc @ axv
slab = 0.6 * np.median(np.diff(cl[:, 0]))
st = cl[len(cl) // 2]; sel = np.abs(tall - st[0]) < slab; Q = P[sel] - st[1:]; xy = np.c_[Q @ e1, Q @ e2]
cx, cy, Rc, rr = fit_circle(xy); a_ell = fit_ellipse_area(xy); az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
hist = np.histogram(az, 36, (-180, 180))[0]; cov = (hist > 0).mean()
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
# left: side view of cloud (axis vertical) with the slab highlighted
Qs = Pc @ Vt.T; idx = np.random.default_rng(0).choice(len(Qs), min(40000, len(Qs)), replace=False)
axs[0].scatter(Qs[idx, 1], Qs[idx, 0], s=1.2, c="0.75", linewidths=0)
band = np.abs(tall - st[0]) < slab; axs[0].scatter(Qs[band, 1], Qs[band, 0], s=2.2, c=BLUE, linewidths=0)
axs[0].set_aspect("equal"); axs[0].set_axis_off()
axs[0].set_title("Dense lumen cloud (2-V2, side view) — one perpendicular slab", fontsize=10.5)
# right: the cross-section
axs[1].scatter(xy[:, 0], xy[:, 1], s=5, c=BLUE, alpha=0.6, linewidths=0, label=f"slab points (n={len(xy)})")
axs[1].add_patch(Circle((cx, cy), Rc, fill=False, color=VERM, lw=2.0, label=f"circle fit  R={Rc:.2f}  resid={rr:.3f}"))
axs[1].plot([cx], [cy], "+", color=VERM, ms=10, mew=2)
ro = Rc * 1.25
for b in range(36):
    axs[1].add_patch(Wedge((cx, cy), ro * 1.06, -180 + b * 10, -170 + b * 10, width=ro * 0.06, color=GREEN if hist[b] > 0 else "0.85", lw=0))
axs[1].set_aspect("equal"); axs[1].set_xlabel("in-plane x (scene units)"); axs[1].set_ylabel("in-plane y (scene units)")
axs[1].set_title(f"Partial-arc fit: coverage {cov:.2f} · circle–ellipse CSA difference {abs(a_ell - np.pi*Rc**2)/(np.pi*Rc**2)*100:.1f}%", fontsize=10.5)
axs[1].legend(fontsize=8, loc="lower right"); axs[1].grid(alpha=0.2)
axs[1].text(0.02, 0.98, "gates: coverage ≥ 0.75 · residual < 0.15\ncircle/ellipse agree ≤ 30% · radius-MAD trim", transform=axs[1].transAxes, va="top", fontsize=8.5, color="0.3",
            bbox=dict(fc="white", ec="0.8", boxstyle="round,pad=0.3"))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_crosssection.png", dpi=170); plt.close(fig); print("fig_crosssection.png")

# ---------------- (c) capture behaviour: centred vs off-centre (50-V2) ----------------
vid = "/home/mi3dr/dataset/New broncho/Videos with Matched CT Scans/50-V2 - 6-1-2026.MP4"
want = {300: None, 640: None}; cap = cv2.VideoCapture(vid); fi = 0
while True:
    ok, f = cap.read()
    if not ok or fi > 640: break
    if fi in want: want[fi] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    fi += 1
cap.release()
def onesided(P):
    Pc = P - P.mean(0); _, _, Vt = np.linalg.svd(Pc[::10], full_matrices=False); perp = Pc - np.outer(Pc @ Vt[0], Vt[0])
    azm = np.degrees(np.arctan2(perp @ Vt[2], perp @ Vt[1])); b = np.histogram(azm, 36, (-180, 180))[0]; return float(b[np.argsort(b)[::-1][:18]].sum() / b.sum())
clouds = {"centred window f230–520\n(re-run)": f"{R}/own_data/retry/rr_50-V2/dense0/fused.ply",
          "off-centre / carina window\n(baseline)": f"{R}/own_data/ct_validated_3/50-V2_reconstruction.ply"}
fig = plt.figure(figsize=(10.5, 9.8)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.7], hspace=0.10, wspace=0.05)
for j, (fr, title) in enumerate([(300, "centred lumen (f300)"), (640, "off-centre, at the carina (f640)")]):
    a = fig.add_subplot(gs[0, j]); a.imshow(want[fr]); a.set_axis_off(); a.set_title(title, fontsize=10.5, color=GREEN if j == 0 else ORANGE)
# bottom row: recon-vs-CT calibre profiles for the two windows (the measurable consequence of capture)
J = json.load(open(f"{R}/own_data/ct_validated_3/CT_validation_metrics.json"))["50-V2"]
M2 = json.load(open(f"{REP}/rerun_eval_50-V2.json"))["methods"]["M2 partial-arc gated"]
panels = [("centred window f230–520 (re-run)", np.array(M2["ct_arc"]), np.array(M2["ct_dce"]), np.array(M2["am"]), np.array(M2["dce"]), f"RMSE {M2['rmse']:.2f} mm · n={M2['n']} gated sections", GREEN),
          ("off-centre / carina window (baseline)", np.array(J["arc"])[:len(J["ct_dce"])], np.array(J["ct_dce"]), np.array(J["arc"])[:len(J["recon_dce"])], np.array(J["recon_dce"]), f"RMSE {J['rmse']:.2f} mm · n={J['n']} sections (all slices)", ORANGE)]
for j, (lab, ca, cd, am, dce, sub, col) in enumerate(panels):
    a = fig.add_subplot(gs[1, j]); a.plot(ca, cd, color="0.15", lw=2.2, label="CT (ground truth)")
    a.plot(am, dce, "-o", color=col, ms=4.2, lw=1.5, label="reconstruction"); a.set_xlabel("arclength from subglottis (mm)")
    if j == 0: a.set_ylabel("D$_{CE}$ (mm)")
    a.set_ylim(3, 17); a.grid(alpha=0.25); a.legend(fontsize=8, loc="upper left"); a.set_title(f"{lab}\n{sub}", fontsize=9.8, color=col)
fig.suptitle("Capture behaviour decides reconstruction quality (50-V2, same video, same pipeline)", fontsize=12, fontweight="bold", y=0.98)
fig.savefig(f"{OUT}/fig_capture.png", dpi=160, bbox_inches="tight"); plt.close(fig); print("fig_capture.png")

# ---------------- (d) calibration table (single generator: calib_table.py) ----------------
import subprocess
subprocess.run([sys.executable, f"{REP}/calib_table.py"], check=True)
