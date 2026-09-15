"""Headline CT figure (Angle 1) with the ADOPTED gated results: (a) trimmed renders of the three clouds that were
measured, (b) gated D_CE profile vs CT with the all-slice baseline in grey, (c) one reconstructed ring vs the CT lumen
per case, (d) pooled Bland-Altman, paired agreement and the before->after table. Reads rerun_eval_*.json /
ct3_regate.json (same selection logic as ct3_report_v2.py) and CT_validation_metrics.json for the baseline.
Writes ct3_validation_figure.png here and in renders/new (the all-slice figure is kept as *_baseline.png)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, os, shutil
sys.path.insert(0, ROOT); import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from airway_analysis import clean, scatter3d
from figstyle import apply_style, PATIENT, ROLE, OKABE
import ct_rings as CR
import cam_rings as CRg
apply_style()

D = f"{ROOT}/runs/own_data/ct_validated_3"; OUT = f"{ROOT}/runs/own_data/reports"; REN = f"{ROOT}/runs/own_data/renders/new"
CAT = f"{ROOT}/runs/own_data/all_reconstructions_ply"
J = json.load(open(f"{D}/CT_validation_metrics.json")); CASES = ["2-V2", "20-V1", "50-V2"]
PLY = {"2-V2": f"{CAT}/baseline/2-V2_loop.ply", "20-V1": f"{CAT}/20-V1.ply", "50-V2": f"{CAT}/50-V2.ply"}
RLAB = {"2-V2": "dual-pass loop model, f880–3370\n(measured along the withdrawal pass)", "20-V1": "f2500–3000", "50-V2": "f230–520 (centred trachea, pre-carina)"}
RING_PLY = {p: f"{CAT}/{p}.ply" for p in CASES}   # cloud-axis ring fit for 20-V1 / 50-V2 (their adopted partial-arc method)
CTTYPE = {"2-V2": "insp 1 mm", "20-V1": "insp/exp 4D", "50-V2": "insp 1.5 mm"}


def improved(pid):
    """adopted gated result: M2 partial-arc, or M1 camera-centerline+gate when M2 is scale-degenerate (self-consistency rule)."""
    f = f"{OUT}/rerun_eval_{pid}.json"
    if os.path.exists(f):
        meth = json.load(open(f))["methods"]; m = meth.get("M2 partial-arc gated"); src = "partial-arc"
        if meth.get("M2_scale_degenerate") and meth.get("M1 cam-centerline + cov>=0.75", {}).get("am"):
            m = meth["M1 cam-centerline + cov>=0.75"]; src = "camera-centerline + gate"
        return dict(am=np.array(m["am"]), dce=np.array(m["dce"]), ct=np.array(m["ct"]), rmse=m["rmse"], bias=m["bias"], n=m["n"], src=src,
                    ct_arc=np.array(m["ct_arc"]), ct_dce=np.array(m["ct_dce"]))
    g = json.load(open(f"{OUT}/ct3_regate.json"))[pid]["circle-only"]; k = np.isfinite(np.array(g["ct"]))
    v = J[pid]
    return dict(am=np.array(g["am"])[k], dce=np.array(g["dce"])[k], ct=np.array(g["ct"])[k], rmse=g["rmse"], bias=g["bias"], n=g["n"], src="partial-arc",
                ct_arc=np.array(v["arc"])[:len(v["ct_dce"])], ct_dce=np.array(v["ct_dce"]))


imp = {p: improved(p) for p in CASES}
E = np.concatenate([imp[p]["dce"] - imp[p]["ct"] for p in CASES]); Rr = np.concatenate([imp[p]["dce"] for p in CASES]); C = np.concatenate([imp[p]["ct"] for p in CASES])
p_rmse = float(np.sqrt(np.mean(E ** 2))); p_bias = float(np.mean(E)); p_sd = float(np.std(E, ddof=1)); loa = (p_bias - 1.96 * p_sd, p_bias + 1.96 * p_sd)
rr = float(np.corrcoef(Rr, C)[0, 1]); N = len(E); mean_cases = float(np.mean([imp[p]["rmse"] for p in CASES]))
b_rmse = float(np.sqrt(np.mean(np.concatenate([np.array(J[p]["recon_dce"]) - np.array(J[p]["ct_dce"]) for p in CASES]) ** 2))); b_mean = float(np.mean([J[p]["rmse"] for p in CASES]))
print(f"gated: mean {mean_cases:.2f} pooled {p_rmse:.2f} bias {p_bias:+.2f} LoA [{loa[0]:+.2f},{loa[1]:+.2f}] r {rr:.2f} n {N}   (baseline mean {b_mean:.2f} pooled {b_rmse:.2f})")
for p in CASES: print(f"  {p}: {J[p]['rmse']:.2f} -> {imp[p]['rmse']:.2f} (bias {imp[p]['bias']:+.2f}, n {imp[p]['n']}, {imp[p]['src']})")

fig = plt.figure(figsize=(13.5, 17.2))
gs = gridspec.GridSpec(4, 3, height_ratios=[1.15, 1.0, 1.05, 1.0], hspace=0.58, wspace=0.3, left=0.06, right=0.98, top=0.925, bottom=0.065)


def gapbreak(a, d, k=2.5):
    """insert NaNs where consecutive accepted stations are more than k x the median spacing apart (no line across gaps)"""
    a, d = np.asarray(a, float), np.asarray(d, float); o = np.argsort(a); a, d = a[o], d[o]
    if len(a) < 3: return a, d
    g = np.where(np.diff(a) > k * np.median(np.diff(a)))[0]
    return np.insert(a, g + 1, np.nan), np.insert(d, g + 1, np.nan)
# (a) renders of the measured clouds, radially trimmed
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[0, j], projection="3d"); P = np.asarray(clean(PLY[pid]).points)
    scatter3d(ax, P, sub=40000, s=1.3, rad_pct=95)
    ax.set_title(f"{pid}", color=PATIENT.get(pid, "0.2"), fontweight="bold", fontsize=13, pad=-4)
    ax.text2D(0.5, -0.02, RLAB[pid], transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.35")
    print(f"  render {pid}: {len(P)} pts from {PLY[pid].split('/')[-1]}")
# (b) gated profile vs CT, baseline in grey
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[1, j]); r = imp[pid]; v = J[pid]
    ax.plot(r["ct_arc"], r["ct_dce"], color=ROLE["ct"], lw=2.2, label="CT (ground truth)")
    ax.plot(np.array(v["arc"])[:len(v["recon_dce"])], v["recon_dce"], "-", color="0.72", lw=1.2, label=f"all-slice baseline ({v['rmse']:.2f} mm)")
    ga, gd = gapbreak(r["am"], r["dce"])
    ax.plot(ga, gd, "-o", color=PATIENT.get(pid, "#1f77b4"), ms=4.2, lw=1.6, label=f"gated ({r['rmse']:.2f} mm, n={r['n']})")
    ax.set_title(f"{pid}: RMSE {r['rmse']:.2f} mm, bias {r['bias']:+.2f} mm", fontsize=11); ax.set_xlabel("arclength (mm)")
    if j == 0: ax.set_ylabel("D$_{CE}$ (mm)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="best")
# (c) one reconstructed ring vs CT lumen per case. 2-V2: a station of the ADOPTED camera-centerline registration (dual-pass model,
#     withdrawal pass; asserted to reproduce rerun_eval_2-V2.json). 20-V1 / 50-V2: cloud-axis station of the partial-arc registration (their adopted method).
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[2, j]); th = np.linspace(0, 2 * np.pi, 200); col = PATIENT.get(pid, OKABE["blue"])
    if pid == "2-V2":
        m1 = json.load(open(f"{OUT}/rerun_eval_2-V2.json"))["methods"]["M1 cam-centerline + cov>=0.75"]
        rings, chk = CRg.m1_rings(CRg.light_clean(f"{ROOT}/runs/own_data/retry/rr_2-V2_loop/dense0/fused.ply"), f"{ROOT}/runs/own_data/retry/rr_2-V2_loop/sparse/0", m1, fr=(2840, 3370), quantiles=(0.5,))
        r = rings[0]; xy, Rm, ctd, am, cov = r["xy"], r["R"], r["ct_d"], r["am"], r["cov"]; print(f"  ring 2-V2 (camera-centerline, loop/withdrawal): {chk}; station {am:.0f} mm, CT {ctd:.1f} vs recon {2*Rm:.1f} mm, cov {cov:.2f}")
    else:
        ca, csa = CR.ct_profile(pid); cd = 2 * np.sqrt(np.clip(csa, 0, None) / np.pi)
        st, _ = CR.station_data(CR.load_clean(RING_PLY[pid])); ok, a_r, s, flip, A, Cc = CR.register(st, ca, cd)
        stn = ok[len(ok) // 2]; xy = stn["xy"] * s; Rm = stn["R"] * s; am = (stn["t"] - a_r.min()) * s; ctd = float(np.interp(am, A, Cc)); cov = stn["cov"]
        print(f"  ring {pid}: station {am:.0f} mm, CT {ctd:.1f} vs recon {2*Rm:.1f} mm, cov {cov:.2f}, {len(ok)} accepted stations")
    ax.plot(ctd / 2 * np.cos(th), ctd / 2 * np.sin(th), color="0.15", lw=2.2, label=f"CT lumen  D={ctd:.1f} mm", zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color=col, alpha=0.5, linewidths=0, label=f"reconstructed wall  D={2*Rm:.1f} mm", zorder=2)
    lim = max(ctd / 2, Rm) * 1.65; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{pid} · arclength {am:.0f} mm · coverage {cov:.2f}", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower center", handletextpad=0.3, framealpha=0.9)
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.6)
# (d) pooled agreement + table
axb = fig.add_subplot(gs[3, 0])
for pid in CASES:
    r = imp[pid]; axb.scatter((r["dce"] + r["ct"]) / 2, r["dce"] - r["ct"], s=16, color=PATIENT.get(pid, "0.4"), label=pid, alpha=0.85)
axb.axhline(p_bias, color="0.2", lw=1.4); [axb.axhline(l, color="0.5", ls="--", lw=1) for l in loa]; axb.axhline(0, color="0.85", lw=0.8, zorder=0)
axb.set_xlabel("mean of CT & reconstruction D$_{CE}$ (mm)"); axb.set_ylabel("reconstruction − CT (mm)")
axb.set_title(f"Bland–Altman: bias {p_bias:+.2f}, LoA [{loa[0]:+.2f}, {loa[1]:+.2f}] mm", fontsize=10.5); axb.legend(fontsize=8); axb.grid(alpha=0.25)
axs = fig.add_subplot(gs[3, 1]); lim = [min(Rr.min(), C.min()) - 0.5, max(Rr.max(), C.max()) + 0.5]; axs.plot(lim, lim, "0.6", ls="--", lw=1)
for pid in CASES:
    r = imp[pid]; axs.scatter(r["ct"], r["dce"], s=16, color=PATIENT.get(pid, "0.4"), alpha=0.85, label=pid)
axs.set_xlabel("CT D$_{CE}$ (mm)"); axs.set_ylabel("reconstruction D$_{CE}$ (mm)"); axs.set_title(f"Paired agreement: r = {rr:.2f}, n = {N}", fontsize=10.5)
axs.set_xlim(lim); axs.set_ylim(lim); axs.set_aspect("equal"); axs.grid(alpha=0.25)
axt = fig.add_subplot(gs[3, 2]); axt.axis("off")
tb_rows = [["patient", "CT", "all-slice", "gated", "bias", "n"]] + [[p, CTTYPE[p], f"{J[p]['rmse']:.2f}", f"{imp[p]['rmse']:.2f}", f"{imp[p]['bias']:+.2f}", str(imp[p]["n"])] for p in CASES]
tb_rows += [["mean", "", f"{b_mean:.2f}", f"{mean_cases:.2f}", "", ""], ["pooled", "", f"{b_rmse:.2f}", f"{p_rmse:.2f}", f"{p_bias:+.2f}", str(N)]]
tb = axt.table(cellText=tb_rows, loc="center", cellLoc="center", colWidths=[0.15, 0.27, 0.16, 0.14, 0.14, 0.09]); tb.auto_set_font_size(False); tb.set_fontsize(8.8); tb.scale(1.12, 1.6)
for c in range(6): tb[(0, c)].set_facecolor("#e8e8e8"); tb[(0, c)].set_text_props(fontweight="bold")
for rI in (len(tb_rows) - 2, len(tb_rows) - 1):
    for c in range(6): tb[(rI, c)].set_text_props(fontweight="bold")
axt.set_title("RMSE (mm): all-slice baseline → gated estimator", fontsize=10.5, pad=12)
LABELS = ["(a) reconstructions measured (radially trimmed dense clouds, colour = axial position)",
          "(b) circle-equivalent diameter along the airway: CT vs reconstruction",
          "(c) reconstructed cross-section vs CT lumen at one accepted station per patient (mm)",
          "(d) pooled agreement over accepted cross-sections"]
for i, lab in enumerate(LABELS):   # anchor each row label just above that row's axes (above the axes titles)
    bb = gs[i, 0].get_position(fig); fig.text(0.06, bb.y1 + (0.012 if i == 0 else 0.03), lab, fontsize=11, fontweight="bold", ha="left", va="bottom")
fig.suptitle("Calibre accuracy against CT — three CT-paired patients (gated estimator · constrained isotropic Sim(3))", fontsize=13.5, fontweight="bold", y=0.988)
fig.text(0.5, 0.006, "Grey = all-slice median-radius baseline on the original windows. Gated = coverage ≥ 0.75, circle residual < 0.15, partial-arc circle fit;\n"
         "where the partial-arc scale is degenerate (2-V2, self-consistency ratio 0.57) the camera-centerline profile of the same gated cloud is reported.\n"
         "(c): wall points scaled to mm by each case's reported registration — 2-V2 at a camera-centerline station of the dual-pass model's withdrawal pass, 20-V1 and 50-V2 at cloud-axis stations of the partial-arc fit.",
         ha="center", va="bottom", fontsize=8.2, color="0.4", linespacing=1.4)
fig.savefig(f"{OUT}/ct3_validation_figure.png", dpi=160); shutil.copy(f"{OUT}/ct3_validation_figure.png", f"{REN}/ct3_validation_figure.png")
print("saved ct3_validation_figure.png (reports + renders/new)")
