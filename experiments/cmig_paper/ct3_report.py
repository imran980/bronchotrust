"""CT-validation report (Angle 1): the 3 CT-paired cases (2-V2, 20-V1, 50-V2).
Uses the VALIDATED camera-trajectory-centerline results in CT_validation_metrics.json (the
cloud-centerline is a rougher fallback). 20-V1 render uses the refined cloud (cleaner window,
RMSE 1.09 ~ unchanged). Outputs ct3_validation_figure.png + pooled stats + a metrics table."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from airway_analysis import clean, scatter3d
from figstyle import apply_style, PATIENT, ROLE
apply_style()

D = f"{ROOT}/runs/own_data/ct_validated_3"
OUT = f"{ROOT}/runs/own_data/reports"
J = json.load(open(f"{D}/CT_validation_metrics.json"))
CASES = ["2-V2", "20-V1", "50-V2"]
CTTYPE = {"2-V2": "insp thin-slice", "20-V1": "insp/exp 4D", "50-V2": "insp thin-slice"}
NOTE20 = "refined window (RMSE 1.09, ~unchanged)"

# pooled errors
allE, allR, allC = [], [], []
for pid in CASES:
    r = np.array(J[pid]["recon_dce"]); c = np.array(J[pid]["ct_dce"])
    allE.append(r - c); allR.append(r); allC.append(c)
E = np.concatenate(allE); R = np.concatenate(allR); C = np.concatenate(allC)
p_rmse = float(np.sqrt(np.mean(E ** 2))); p_bias = float(np.mean(E)); p_sd = float(np.std(E, ddof=1))
loa = (p_bias - 1.96 * p_sd, p_bias + 1.96 * p_sd)
rr = float(np.corrcoef(R, C)[0, 1]); Ntot = len(E)

print("=== POOLED CT VALIDATION (n=%d cross-sections, %d patients) ===" % (Ntot, len(CASES)))
for pid in CASES:
    v = J[pid]
    print(f"  {pid:6s} RMSE {v['rmse']:.2f} mm  bias {v['bias']:+.2f}  n {v['n']:2d}  scale {v['scale']:.2f}  ({CTTYPE[pid]})")
print(f"  POOLED RMSE {p_rmse:.2f} mm | bias {p_bias:+.2f} | LoA [{loa[0]:+.2f}, {loa[1]:+.2f}] | r {rr:.2f}")

# ---- figure ----
fig = plt.figure(figsize=(13.5, 12.5))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.15, 1.0, 1.05], hspace=0.36, wspace=0.28,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)
# row 0: renders
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[0, j], projection="3d")
    P = np.asarray(clean(f"{D}/{pid}_reconstruction.ply").points)
    scatter3d(ax, P, sub=35000, s=1.4)
    ttl = f"{pid}" + ("  ✓" if True else "")
    ax.set_title(ttl, color=PATIENT.get(pid, "0.2"), fontweight="bold", fontsize=13, pad=-2)
# row 1: profiles
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[1, j]); v = J[pid]
    arc = np.array(v["arc"])[:len(v["recon_dce"])]
    ax.plot(arc, v["ct_dce"], color=ROLE["ct"], lw=2.3, label="CT (ground truth)")
    ax.plot(arc, v["recon_dce"], "-o", color=PATIENT.get(pid, "#1f77b4"), ms=4, lw=1.6, label="reconstruction")
    ax.set_xlabel("arclength from subglottis (mm)");
    if j == 0: ax.set_ylabel("D$_{CE}$ (mm)")
    ax.set_title(f"{pid}: RMSE {v['rmse']:.2f} mm (n={v['n']})", fontsize=11)
    ax.grid(alpha=0.25)
    if j == 0: ax.legend(fontsize=8, loc="lower center")
# row 2: pooled BA + scatter + table
axb = fig.add_subplot(gs[2, 0])
off = 0
for pid in CASES:
    n = J[pid]["n"]; sl = slice(off, off + n); off += n
    axb.scatter((R[sl] + C[sl]) / 2, R[sl] - C[sl], s=14, color=PATIENT.get(pid, "0.4"), label=pid, alpha=0.8)
axb.axhline(p_bias, color="0.2", lw=1.4); axb.axhline(loa[0], color="0.5", ls="--", lw=1)
axb.axhline(loa[1], color="0.5", ls="--", lw=1); axb.axhline(0, color="0.8", lw=0.8, zorder=0)
axb.set_xlabel("mean of CT & recon D$_{CE}$ (mm)"); axb.set_ylabel("recon − CT (mm)")
axb.set_title(f"Bland–Altman (bias {p_bias:+.2f}, LoA ±{1.96*p_sd:.1f} mm)", fontsize=11)
axb.legend(fontsize=8); axb.grid(alpha=0.25)

axs = fig.add_subplot(gs[2, 1])
lim = [min(R.min(), C.min()) - 0.5, max(R.max(), C.max()) + 0.5]
axs.plot(lim, lim, "0.6", ls="--", lw=1)
off = 0
for pid in CASES:
    n = J[pid]["n"]; sl = slice(off, off + n); off += n
    axs.scatter(C[sl], R[sl], s=14, color=PATIENT.get(pid, "0.4"), alpha=0.8, label=pid)
axs.set_xlabel("CT D$_{CE}$ (mm)"); axs.set_ylabel("recon D$_{CE}$ (mm)")
axs.set_title(f"Paired agreement (r = {rr:.2f})", fontsize=11); axs.set_xlim(lim); axs.set_ylim(lim)
axs.set_aspect("equal"); axs.grid(alpha=0.25)

axt = fig.add_subplot(gs[2, 2]); axt.axis("off")
rows = [["case", "RMSE", "bias", "n", "CT"]]
for pid in CASES:
    v = J[pid]; rows.append([pid, f"{v['rmse']:.2f}", f"{v['bias']:+.2f}", str(v["n"]), CTTYPE[pid]])
rows.append(["POOLED", f"{p_rmse:.2f}", f"{p_bias:+.2f}", str(Ntot), f"r={rr:.2f}"])
tb = axt.table(cellText=rows, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.7)
for c in range(5): tb[(0, c)].set_facecolor("#e8e8e8"); tb[(0, c)].set_text_props(fontweight="bold")
for c in range(5): tb[(len(rows) - 1, c)].set_text_props(fontweight="bold")
axt.set_title("Per-patient caliber accuracy", fontsize=11, pad=14)

fig.suptitle("CT validation of monocular airway reconstruction — 3 CT-paired paediatric cases",
             fontsize=15, fontweight="bold", y=0.975)
fig.text(0.5, 0.005, f"D$_{{CE}}$ = circle-equivalent diameter; scale resolved by isotropic Sim(3). "
         f"20-V1 render = {NOTE20}.", ha="center", fontsize=8.5, color="0.4")
fig.savefig(f"{OUT}/ct3_validation_figure.png", dpi=160)
print(f"\nsaved {OUT}/ct3_validation_figure.png")

json.dump(dict(pooled=dict(rmse=p_rmse, bias=p_bias, sd=p_sd, loa=list(loa), r=rr, n=Ntot),
               per_case={p: {k: J[p][k] for k in ("rmse", "bias", "n", "scale")} for p in CASES}),
          open(f"{OUT}/ct3_pooled_stats.json", "w"), indent=2)
print("saved ct3_pooled_stats.json")
