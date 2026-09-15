"""CT validation v2 — BEFORE (baseline camera-centerline, no gate) vs AFTER (pre-declared uniform
method M2: partial-arc circle fit, cov>=0.75, resid<0.15, radius-MAD trim, constrained isotropic Sim3;
50-V2 and 2-V2 additionally re-windowed to centred passes). Builds ct3_validation_figure_v2.png and
ct3_pooled_stats_v2.json. Falls back to the baseline for any case whose re-run score is missing."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from airway_analysis import dce_of
from evaluation.ground_truth.ground_truth import GroundTruth
from figstyle import apply_style, PATIENT, ROLE
apply_style()

D = f"{ROOT}/runs/own_data/ct_validated_3"; OUT = f"{ROOT}/runs/own_data/reports"
J = json.load(open(f"{D}/CT_validation_metrics.json")); CASES = ["2-V2", "20-V1", "50-V2"]
WINDOW = {"2-V2": "dual-pass loop f880-3370, measured along the withdrawal pass", "20-V1": "existing refined cloud (f2500-3000)", "50-V2": "re-windowed f230-520 (centred trachea, pre-carina)"}


def ct_curve(pid):
    try:
        g = GroundTruth.load(f"{D}/ts_{pid}/gt_{pid}.json"); return np.asarray(g.arclength_mm), dce_of(g)
    except Exception:
        v = J[pid]; a = np.array(v["arc"])[:len(v["ct_dce"])]; return a, np.array(v["ct_dce"])


def improved(pid):
    """returns dict(am, dce, ct, rmse, bias, n, source) for the pre-declared M2 method, or None."""
    f = f"{OUT}/rerun_eval_{pid}.json"
    if os.path.exists(f):
        meth = json.load(open(f))["methods"]
        m = meth.get("M2 partial-arc gated"); src = "re-run + M2"
        # UNIFORM self-consistency rule: M2 and the camera-centerline registration of the SAME cloud must agree on
        # median calibre within 30%; otherwise the isotropic fit is scale-degenerate (flat profile) and M2 is rejected
        # in favour of the camera-centerline + coverage-gate result (M1).
        if meth.get("M2_scale_degenerate") and meth.get("M1 cam-centerline + cov>=0.75", {}).get("am"):
            m = meth["M1 cam-centerline + cov>=0.75"]; src = f"re-run + M1 (M2 rejected: scale-degenerate, calibre ratio {meth['M2_vs_M1_calibre_ratio']:.2f})"
        if m and "am" in m:
            am, dce, ct = map(np.array, (m["am"], m["dce"], m["ct"]))
            return dict(am=am, dce=dce, ct=ct, rmse=m["rmse"], bias=m["bias"], n=m["n"], source=src, ct_arc=np.array(m["ct_arc"]), ct_dce=np.array(m["ct_dce"]))
    g = json.load(open(f"{OUT}/ct3_regate.json")).get(pid, {}).get("circle-only")
    if g:
        am, dce, ct = map(np.array, (g["am"], g["dce"], g["ct"])); k = np.isfinite(ct)
        return dict(am=am[k], dce=dce[k], ct=ct[k], rmse=g["rmse"], bias=g["bias"], n=g["n"], source="existing cloud + M2")
    return None


rows, allE, allR, allC = [], [], [], []
imp = {}
for pid in CASES:
    r = improved(pid); imp[pid] = r
    if r is None:   # fall back to baseline
        v = J[pid]; r = dict(am=np.array(v["arc"])[:len(v["recon_dce"])], dce=np.array(v["recon_dce"]), ct=np.array(v["ct_dce"]), rmse=v["rmse"], bias=v["bias"], n=v["n"], source="baseline (no re-run score yet)"); imp[pid] = r
    allE.append(r["dce"] - r["ct"]); allR.append(r["dce"]); allC.append(r["ct"])
    rows.append((pid, J[pid]["rmse"], J[pid]["bias"], J[pid]["n"], r["rmse"], r["bias"], r["n"], r["source"]))
E, Rr, C = map(np.concatenate, (allE, allR, allC))
p_rmse = float(np.sqrt(np.mean(E ** 2))); p_bias = float(np.mean(E)); p_sd = float(np.std(E, ddof=1)); loa = (p_bias - 1.96 * p_sd, p_bias + 1.96 * p_sd)
rr = float(np.corrcoef(Rr, C)[0, 1]); N = len(E)
b_rmse = float(np.sqrt(np.mean(np.concatenate([np.array(J[p]["recon_dce"]) - np.array(J[p]["ct_dce"]) for p in CASES]) ** 2)))
mean_cases = float(np.mean([r[4] for r in rows])); b_mean = float(np.mean([J[p]["rmse"] for p in CASES]))

print("=== CT VALIDATION: before -> after ===")
for pid, b_r, b_b, b_n, a_r, a_b, a_n, src in rows:
    print(f"  {pid:6s} baseline {b_r:.2f} mm (n={b_n:2d})  ->  {a_r:.2f} mm (bias {a_b:+.2f}, n={a_n:2d})   [{src}]")
print(f"  mean of cases {b_mean:.2f} -> {mean_cases:.2f} | pooled RMSE {b_rmse:.2f} -> {p_rmse:.2f} | bias {p_bias:+.2f} | LoA [{loa[0]:+.2f},{loa[1]:+.2f}] | r {rr:.2f} | n {N}")

fig = plt.figure(figsize=(13.5, 8.6))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.3, left=0.06, right=0.98, top=0.90, bottom=0.08)
for j, pid in enumerate(CASES):
    ax = fig.add_subplot(gs[0, j]); r = imp[pid]; v = J[pid]
    ca, cd = (r["ct_arc"], r["ct_dce"]) if "ct_arc" in r else ct_curve(pid)
    ax.plot(ca, cd, color=ROLE["ct"], lw=2.2, label="CT (ground truth)")
    ba = np.array(v["arc"])[:len(v["recon_dce"])]
    ax.plot(ba, v["recon_dce"], "-", color="0.72", lw=1.2, label=f"baseline ({v['rmse']:.2f} mm)")
    ax.plot(r["am"], r["dce"], "-o", color=PATIENT.get(pid, "#1f77b4"), ms=4.2, lw=1.6, label=f"improved ({r['rmse']:.2f} mm)")
    ax.set_title(f"{pid}: {v['rmse']:.2f} → {r['rmse']:.2f} mm  (n={r['n']})", fontsize=11); ax.set_xlabel("arclength (mm)")
    if j == 0: ax.set_ylabel("D$_{CE}$ (mm)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="best")
axb = fig.add_subplot(gs[1, 0]); off = 0
for pid in CASES:
    r = imp[pid]; n = len(r["dce"]); axb.scatter((r["dce"] + r["ct"]) / 2, r["dce"] - r["ct"], s=16, color=PATIENT.get(pid, "0.4"), label=pid, alpha=0.85)
axb.axhline(p_bias, color="0.2", lw=1.4); [axb.axhline(l, color="0.5", ls="--", lw=1) for l in loa]; axb.axhline(0, color="0.85", lw=0.8, zorder=0)
axb.set_xlabel("mean of CT & recon D$_{CE}$ (mm)"); axb.set_ylabel("recon − CT (mm)"); axb.set_title(f"Bland–Altman (improved): bias {p_bias:+.2f}, LoA ±{1.96*p_sd:.1f} mm", fontsize=10.5); axb.legend(fontsize=8); axb.grid(alpha=0.25)
axs = fig.add_subplot(gs[1, 1]); lim = [min(Rr.min(), C.min()) - 0.5, max(Rr.max(), C.max()) + 0.5]; axs.plot(lim, lim, "0.6", ls="--", lw=1)
for pid in CASES:
    r = imp[pid]; axs.scatter(r["ct"], r["dce"], s=16, color=PATIENT.get(pid, "0.4"), alpha=0.85, label=pid)
axs.set_xlabel("CT D$_{CE}$ (mm)"); axs.set_ylabel("recon D$_{CE}$ (mm)"); axs.set_title(f"Paired agreement (improved): r = {rr:.2f}", fontsize=10.5); axs.set_xlim(lim); axs.set_ylim(lim); axs.set_aspect("equal"); axs.grid(alpha=0.25)
axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
tb_rows = [["case", "before", "after", "n", "bias"]] + [[p, f"{b:.2f}", f"{a:.2f}", str(n), f"{ab:+.2f}"] for p, b, _, _, a, ab, n, _ in rows]
tb_rows.append(["mean", f"{b_mean:.2f}", f"{mean_cases:.2f}", "", ""]); tb_rows.append(["pooled", f"{b_rmse:.2f}", f"{p_rmse:.2f}", str(N), f"{p_bias:+.2f}"])
tb = axt.table(cellText=tb_rows, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.6)
for c in range(5): tb[(0, c)].set_facecolor("#e8e8e8"); tb[(0, c)].set_text_props(fontweight="bold")
for rI in (len(tb_rows) - 2, len(tb_rows) - 1):
    for c in range(5): tb[(rI, c)].set_text_props(fontweight="bold")
axt.set_title("RMSE (mm): baseline → improved", fontsize=10.5, pad=12)
fig.suptitle("CT validation — baseline vs improved (centred windows · coverage gate · partial-arc fit · constrained isotropic Sim(3))", fontsize=13, fontweight="bold", y=0.975)
fig.text(0.5, 0.005, "Grey = baseline reconstruction profile; coloured = improved. Improved segments are shorter where the window was cut before the carina (50-V2).", ha="center", fontsize=8.5, color="0.4")
fig.savefig(f"{OUT}/ct3_validation_figure_v2.png", dpi=160); print("saved ct3_validation_figure_v2.png")
json.dump(dict(pooled=dict(rmse=p_rmse, bias=p_bias, sd=p_sd, loa=list(loa), r=rr, n=N, baseline_pooled_rmse=b_rmse, mean_cases=mean_cases, baseline_mean_cases=b_mean),
               per_case={p: dict(before=b, after=a, bias=ab, n=n, source=s, window=WINDOW[p]) for p, b, _, _, a, ab, n, s in rows}),
          open(f"{OUT}/ct3_pooled_stats_v2.json", "w"), indent=2); print("saved ct3_pooled_stats_v2.json")
