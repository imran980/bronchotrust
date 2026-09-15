"""Robust scale-free CSA + %obstruction on the complete-tube reconstructions.
Validated partial-arc circle+ellipse fits (csa_partialarc.profile), then STRICT gates:
coverage>=0.75, circle residual<0.15, circle-vs-ellipse agreement<30%; end 10% of stations
excluded (funnel/carina ends); 3-station median filter kills single-station spikes.
A_min = narrowest INTERIOR valid slice; A_ref = 90th-percentile valid CSA (robust normal calibre).
%obstruction = (1 - A_min/A_ref)*100 ; D_CE ratio = sqrt(A_min/A_ref). Scene units (scale-free).
Outputs robust_csa_table.json/.md + robust_csa_figure.png."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json
import numpy as np, open3d as o3d
from scipy.ndimage import median_filter
from csa_partialarc import profile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply_style, OKABE
apply_style()

PLY = f"{ROOT}/runs/own_data/all_reconstructions_ply"
OUT = f"{ROOT}/runs/own_data/reports"
CASES = ["26-V2", "12-V1", "33-V1", "27-V1", "30-V2", "16-V1", "2-V3", "15-V2", "21-V1", "22-V2", "25-V1", "9-V2", "10-V2", "17-V1", "7-V1",
         "24-V2", "13-V1", "14-V1", "31-V1", "1-V3", "32-V2", "10-V1", "2-V2", "20-V1", "50-V2"]
CT = {"2-V2", "20-V1", "50-V2"}
COV, RESID, AGREE, END, MINVALID = 0.75, 0.15, 0.30, 0.10, 10


def load_clean(fp):
    P = np.asarray(o3d.io.read_point_cloud(fp).points)
    m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=np.median(d) * 0.03)
    return np.asarray(pcd.points)


results, panels = {}, []
for pid in CASES:
    P = load_clean(f"{PLY}/{pid}.ply")
    rows, _ = profile(P, nb=60)
    if not rows:
        results[pid] = dict(measurable=False, reason="no stations"); continue
    t = np.array([r["t"] for r in rows]); cov = np.array([r["cov"] for r in rows])
    cc = np.array([r["csa_circ"] for r in rows]); ce = np.array([r["csa_ell"] for r in rows])
    res = np.array([r["resid"] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = np.abs(ce - cc) / cc
    ok = (cov >= COV) & (res < RESID) & np.isfinite(agree) & (agree < AGREE) & np.isfinite(cc) & (cc > 0)
    # drop end stations (funnel / carina)
    n = len(t); lo, hi = int(np.floor(END * n)), int(np.ceil((1 - END) * n))
    interior = np.zeros(n, bool); interior[lo:hi] = True
    ok &= interior
    nv = int(ok.sum())
    if nv < MINVALID:
        results[pid] = dict(measurable=False, reason=f"only {nv} strict-valid interior stations", n_valid=nv)
        panels.append((pid, t, cc, cov, ok, None)); continue
    tv, cv = t[ok], cc[ok]
    sm = median_filter(cv, size=3, mode="nearest")            # kill single-station spikes
    imin = int(np.argmin(sm)); a_min = float(sm[imin])
    a_ref = float(np.percentile(sm, 90))                       # robust "normal" calibre
    pct = (1 - a_min / a_ref) * 100; dce_ratio = float(np.sqrt(a_min / a_ref))
    # is the minimum a genuine interior dip (not at the edge of the valid run)?
    edge = imin <= 1 or imin >= len(sm) - 2
    results[pid] = dict(measurable=True, n_valid=nv, mean_cov=float(cov[ok].mean()),
                        agree_med=float(np.median(agree[ok])), A_min=a_min, A_ref=a_ref,
                        pct_obstruction=float(pct), dce_ratio=dce_ratio, cv=float(np.std(sm) / np.mean(sm)),
                        min_at_edge=bool(edge), ct_case=pid in CT)
    panels.append((pid, t, cc, cov, ok, (tv[imin], a_min, a_ref)))

# ---------- table ----------
md = ["| case | n valid | cov | circ/ell agree | A_min | A_ref | **%obstr** | D_CE ratio | CSA CV | min interior? |",
      "|---|---|---|---|---|---|---|---|---|---|"]
print(f"{'case':7s} {'nval':>4s} {'cov':>4s} {'agr':>5s} {'A_min':>7s} {'A_ref':>7s} {'%obs':>6s} {'DCEr':>5s} {'CV':>5s} interior")
for pid in CASES:
    r = results[pid]
    if not r["measurable"]:
        print(f"{pid:7s} -- NOT measurable: {r['reason']}"); md.append(f"| {pid} | {r.get('n_valid','-')} | | | | | *not measurable* | | | {r['reason']} |"); continue
    flag = "edge" if r["min_at_edge"] else "yes"
    print(f"{pid:7s} {r['n_valid']:4d} {r['mean_cov']:4.2f} {r['agree_med']:5.1%} {r['A_min']:7.2f} {r['A_ref']:7.2f} {r['pct_obstruction']:5.0f}% {r['dce_ratio']:5.2f} {r['cv']:5.2f} {flag}")
    md.append(f"| {pid}{' (CT)' if r['ct_case'] else ''} | {r['n_valid']} | {r['mean_cov']:.2f} | {r['agree_med']:.0%} | {r['A_min']:.2f} | {r['A_ref']:.2f} | **{r['pct_obstruction']:.0f}%** | {r['dce_ratio']:.2f} | {r['cv']:.2f} | {flag} |")
json.dump(results, open(f"{OUT}/robust_csa_table.json", "w"), indent=2)
open(f"{OUT}/robust_csa_table.md", "w").write("\n".join(md) + "\n")

# ---------- figure: gated profiles with A_min / A_ref ----------
meas = [p for p in panels if p[5] is not None]
cols = 4; rws = (len(meas) + cols - 1) // cols
fig, axs = plt.subplots(rws, cols, figsize=(5.2 * cols, 3.1 * rws)); axs = np.array(axs).ravel()
for k, (pid, t, cc, cov, ok, (tmin, a_min, a_ref)) in enumerate(meas):
    ax = axs[k]; r = results[pid]
    ax.plot(t, cc, "-", color="0.82", lw=0.9, zorder=0)
    ax.scatter(t[~ok], cc[~ok], s=12, color="0.75", label="rejected")
    sc = ax.scatter(t[ok], cc[ok], s=22, c=cov[ok], cmap="viridis", vmin=0.6, vmax=1.0, label="valid")
    ax.axhline(a_ref, color=OKABE["blue"], ls="--", lw=1.2); ax.axhline(a_min, color=OKABE["vermillion"], ls="--", lw=1.2)
    ax.plot([tmin], [a_min], "v", color=OKABE["vermillion"], ms=9)
    ax.set_title(f"{pid}{'  (CT case)' if r['ct_case'] else ''}: %obstr = {r['pct_obstruction']:.0f}%  (n={r['n_valid']}, cov {r['mean_cov']:.2f})", fontsize=11)
    ax.set_xlabel("along-axis station (scene units)"); ax.set_ylabel("CSA (scene u²)"); ax.grid(alpha=0.25)
    ymax = np.nanpercentile(cc[ok], 99) * 1.35; ax.set_ylim(0, ymax)
    if k == 0: ax.legend(fontsize=7.5, loc="upper left")
for k in range(len(meas), len(axs)): axs[k].axis("off")
fig.suptitle("Robust scale-free CSA + %obstruction — strict gates (cov≥0.75, resid<0.15, circ/ell agree), ends & spikes excluded\n"
             "blue dashed = A_ref (robust normal calibre) · red dashed/▼ = A_min (narrowest interior valid slice)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/robust_csa_figure.png", dpi=130)
print(f"\nsaved robust_csa_figure.png + robust_csa_table.md ({len(meas)} measurable)")
