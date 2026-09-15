"""Exemplar-cases figure: three successful reconstructions walked through the whole measurement layer —
video frames from the measured window -> dense cloud -> gated cross-section rings -> gated CSA profile with %obstruction.
Cases: 2-V2 (CT-validated, clinically normal), 17-V1 (full descent-carina-pullback loop, carina + bronchi), 26-V2
(rescued by re-windowing onto the slow centred pullback; 48/48 gated). Frames are decoded SEQUENTIALLY (never seeked).
Rings and CSA use the same cloud-only estimators as the cohort tables (ct_rings.station_data / csa_partialarc.profile with
robust_csa gates), so every number in the figure is the one in the tables. Writes fig_exemplars.png to renders/new."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, os, subprocess
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import median_filter
from airway_analysis import clean, scatter3d
from figstyle import apply_style, PATIENT, OKABE
from csa_partialarc import profile
import ct_rings as CR
import cam_rings as CRg
apply_style()

R = f"{ROOT}/runs/own_data"; CAT = f"{R}/all_reconstructions_ply"; OUT = f"{R}/renders/new"; REP = f"{R}/reports"
COV, RESID, AGREE, END = 0.75, 0.15, 0.30, 0.10          # identical to robust_csa.py
CASES = [
    dict(pid="2-V2", video="/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4", lo=2870, hi=3370, ws=f"{R}/retry/rr_2-V2",
         role="CT-validated, clinically normal airway", window="withdrawal pass f2870–3370", ct=True,
         m1_json=f"{R}/reports/rerun_eval_2-V2_f2870-3370.json", fused=f"{R}/retry/rr_2-V2/dense0/fused.ply",
         extra="CT: RMSE 1.28 mm (bias +0.60, n 34) on this single-pass cloud; 1.07 mm (n 40) on the dual-pass model reported in the accuracy table"),
    dict(pid="17-V1", video="/home/mi3dr/dataset/New broncho/more videos with calib/17-V1.MP4", lo=500, hi=1330, ws=f"{R}/retry/rr_17-V1_full",
         role="complete descent–carina–pullback loop", window="full loop f500–1330", ct=False,
         extra="trachea, Y-shaped carina and both mainstem bronchi in one model"),
    dict(pid="26-V2", video="/home/mi3dr/dataset/New broncho/more videos with calib/26-V2.MP4", lo=1620, hi=1901, ws=f"{R}/retry/rr_26-V2_pull",
         role="rescued by re-windowing onto the slow, centred pullback", window="pullback f1620–1901", ct=False,
         extra="whole-traversal window: 0/48 gated (instrument, glottic dwell, wall collapse at f1912–1918)"),
]
ROB = json.load(open(f"{REP}/robust_csa_table.json")); YLD = {d["case"]: d for d in json.load(open(f"{REP}/yield_table.json"))}
COLMAP = _os.environ.get("BRONCHO_COLMAP", "colmap")


def registered(ws):
    """registered images + mean reprojection of the workspace's sparse model (eval.json if present, else colmap model_analyzer)."""
    ej = f"{ws}/eval.json"
    if os.path.exists(ej):
        e = json.load(open(ej)); return e.get("registered"), e.get("reproj")
    best = (0, None)
    for m in sorted(os.listdir(f"{ws}/sparse")):
        try:
            txt = subprocess.run([COLMAP, "model_analyzer", "--path", f"{ws}/sparse/{m}"], capture_output=True, text=True, timeout=300).stdout + \
                  subprocess.run([COLMAP, "model_analyzer", "--path", f"{ws}/sparse/{m}"], capture_output=True, text=True, timeout=300).stderr
        except Exception: continue
        import re
        mreg = re.search(r"Registered images:\s*(\d+)", txt); mrp = re.search(r"Mean reprojection error:\s*([\d.]+)px", txt)   # glog-prefixed lines
        n = int(mreg.group(1)) if mreg else 0; rp = float(mrp.group(1)) if mrp else None
        if n > best[0]: best = (n, rp)
    return best


def frames_sequential(video, idxs):
    """decode sequentially with cap.read() up to max(idxs); return {idx: BGR frame}. Never uses CAP_PROP_POS_FRAMES."""
    want = set(idxs); got = {}; cap = cv2.VideoCapture(video); i = 0
    while want - set(got):
        ok, f = cap.read()
        if not ok: break
        if i in want: got[i] = f
        i += 1
    cap.release(); return got


def gated_profile(P):
    rows, cl = profile(P, nb=60); t0 = float(cl[:, 0].min())      # t0: ct_rings.station_data zero-bases its stations; profile() does not
    t = np.array([r["t"] for r in rows]); cov = np.array([r["cov"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows])
    ce = np.array([r["csa_ell"] for r in rows]); res = np.array([r["resid"] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"): agree = np.abs(ce - cc) / cc
    ok = (cov >= COV) & (res < RESID) & np.isfinite(agree) & (agree < AGREE) & np.isfinite(cc) & (cc > 0)
    n = len(t); lo, hi = int(np.floor(END * n)), int(np.ceil((1 - END) * n)); inter = np.zeros(n, bool); inter[lo:hi] = True; ok &= inter
    if ok.sum() == 0: raise SystemExit(f"no accepted stations (cov>={COV}, resid<{RESID}) — cloud/cleaner mismatch with robust_csa?")
    sm = median_filter(cc[ok], size=3, mode="nearest"); imin = int(np.argmin(sm))
    return t, cc, cov, ok, (t[ok][imin], float(sm[imin]), float(np.percentile(sm, 90))), t0


fig = plt.figure(figsize=(17.5, 12.6))
outer = gridspec.GridSpec(3, 1, hspace=0.62, left=0.015, right=0.99, top=0.895, bottom=0.085)
ROWCOL = {"2-V2": PATIENT.get("2-V2", OKABE["blue"]), "17-V1": OKABE["green"], "26-V2": OKABE["vermillion"]}


def crop_field(im, thr=10, pad=8):
    """crop the endoscope image circle out of the black 16:9 frame (bbox of pixels brighter than thr)"""
    g = im.max(axis=2); rows = np.where(g.mean(1) > thr)[0]; cols = np.where(g.mean(0) > thr)[0]
    if len(rows) < 10 or len(cols) < 10: return im
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, im.shape[0]); c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, im.shape[1])
    return im[r0:r1, c0:c1]
for i, C in enumerate(CASES):
    pid = C["pid"]; col = ROWCOL[pid]; rob = ROB[pid]; y = YLD[pid]
    g = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[i], wspace=0.25, width_ratios=[1, 1, 1, 1.45, 0.95, 0.95, 0.95, 2.1])
    # --- frames from the measured window (sequential decode) ---
    idxs = [int(C["lo"] + q * (C["hi"] - C["lo"])) for q in (0.15, 0.5, 0.85)]; fr = frames_sequential(C["video"], idxs)
    for j, k in enumerate(idxs):
        ax = fig.add_subplot(g[0, j]); ax.axis("off")
        if k in fr:
            im = crop_field(cv2.cvtColor(fr[k], cv2.COLOR_BGR2RGB))
            ax.imshow(im); ax.set_title(f"frame {k}", fontsize=8.5, color="0.35", pad=2)
        else: ax.text(0.5, 0.5, f"frame {k}\nnot decodable", ha="center", va="center", transform=ax.transAxes)
    # --- dense cloud ---
    ax = fig.add_subplot(g[0, 3], projection="3d"); P = CR.load_clean(f"{CAT}/{pid}.ply"); scatter3d(ax, P, sub=40000, s=1.2, rad_pct=95)   # same cleaner as robust_csa / yield table
    nreg, rp = registered(C["ws"]); nwin = C["hi"] - C["lo"] + 1
    ax.set_title(f"{nreg}/{nwin} frames registered · {len(P)/1e6:.2f} M points" + (f" · reproj {rp:.2f} px" if rp else ""), fontsize=8.5, color="0.35", pad=-2)
    print(f"{pid}: registered {nreg}/{nwin}, reproj {rp}, clean pts {len(P)}")
    # --- gated CSA profile (computed first; same gates as the cohort table) ---
    t, cc, cov, ok, (tmin, a_min, a_ref), t0 = gated_profile(P); ring_ts = []
    # --- rings at three accepted stations ---
    th = np.linspace(0, 2 * np.pi, 200)
    def style(ax, lim):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.4)
    if C["ct"]:   # the stations of the REPORTED camera-centerline registration of this cloud (M1 in the rerun_eval json), not a cloud-only re-registration
        m1 = json.load(open(C["m1_json"]))["methods"]["M1 cam-centerline + cov>=0.75"]
        rings, chk = CRg.m1_rings(CRg.light_clean(C["fused"]), f"{C['ws']}/sparse/0", m1, fr=(C["lo"], C["hi"])); print(f"  rings: {chk}; scale {m1['scale']:.2f}, RMSE {m1['rmse']:.2f} n {m1['n']}")
        for j, r in enumerate(rings):
            ax = fig.add_subplot(g[0, 4 + j]); ax.scatter(r["xy"][:, 0], r["xy"][:, 1], s=3, color=col, alpha=0.55, linewidths=0, zorder=2)
            ax.plot(r["ct_d"] / 2 * np.cos(th), r["ct_d"] / 2 * np.sin(th), color="0.15", lw=2.0, zorder=1)
            ax.set_title(f"{r['am']:.0f} mm · cov {r['cov']:.2f}\nCT {r['ct_d']:.1f} / recon {2*r['R']:.1f} mm", fontsize=8, color="0.3", pad=2); style(ax, max(r["R"], r["ct_d"] / 2) * 1.5)
    else:         # cloud-only stations passing the same gates as the CSA profile (interior, coverage, residual, enough points)
        st, _ = CR.station_data(P); tacc = t[ok]; tol = 1e-3 * (t.max() - t.min())
        ok_st = sorted([s_ for s_ in st if s_ and np.min(np.abs(tacc - (s_["t"] + t0))) < tol], key=lambda s_: s_["t"])   # exactly the CSA-accepted stations
        ts = np.array([s_["t"] for s_ in ok_st]); picks = [ok_st[int(np.argmin(np.abs(ts - (ts.min() + q * (ts.max() - ts.min())))))] for q in (0.25, 0.5, 0.75)]
        for j, stn in enumerate(picks):
            ax = fig.add_subplot(g[0, 4 + j]); xy = stn["xy"]; Rr = stn["R"]; ring_ts.append(stn["t"] + t0)
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color=col, alpha=0.55, linewidths=0, zorder=2); ax.plot(Rr * np.cos(th), Rr * np.sin(th), color="0.15", lw=1.3, ls="--", zorder=1)
            ax.set_title(f"station {stn['t'] + t0:.1f} · cov {stn['cov']:.2f}\nresid {stn['resid']:.2f}", fontsize=8, color="0.3", pad=2); style(ax, Rr * 1.5)
    # --- gated CSA profile + %obstruction (same gates as the cohort table) ---
    ax = fig.add_subplot(g[0, 7])
    for rt in ring_ts: ax.axvline(rt, color=col, lw=0.9, ls=":", alpha=0.8, zorder=0)      # the stations shown as rings
    ax.plot(t, cc, "-", color="0.85", lw=0.9, zorder=0); ax.scatter(t[~ok], cc[~ok], s=11, color="0.72", label="rejected station")
    ax.scatter(t[ok], cc[ok], s=20, c=cov[ok], cmap="viridis", vmin=0.6, vmax=1.0, label="accepted (colour = coverage)")
    ax.axhline(a_ref, color=OKABE["blue"], ls="--", lw=1.1); ax.axhline(a_min, color=OKABE["vermillion"], ls="--", lw=1.1); ax.plot([tmin], [a_min], "v", color=OKABE["vermillion"], ms=8)
    ax.set_ylim(0, np.nanpercentile(cc[ok], 99) * 1.4); ax.grid(alpha=0.25); ax.set_xlabel("station along the lumen axis (scene units)", fontsize=8.5); ax.set_ylabel("CSA (scene u$^2$)", fontsize=8.5)
    ax.set_title(f"{rob['n_valid']}/{y['n_stations']} accepted · %obstruction {rob['pct_obstruction']:.0f}% · CV {rob['cv']:.2f}", fontsize=9)
    if i == 0: ax.legend(fontsize=7, loc="upper left")
    print(f"  gated: {int(ok.sum())} accepted, A_min {a_min:.2f} A_ref {a_ref:.2f} -> {(1-a_min/a_ref)*100:.0f}%  (table: {rob['pct_obstruction']:.0f}%)")
    # --- row header ---
    bb = outer[i].get_position(fig)
    fig.text(0.015, bb.y1 + 0.036, f"({'abc'[i]}) {pid} — {C['role']}", fontsize=12, fontweight="bold", color=col, va="bottom")
    fig.text(0.015, bb.y1 + 0.02, f"{C['window']} · {C['extra']}", fontsize=9, color="0.35", va="bottom")
fig.text(0.19, 0.948, "video frames (measured window)", fontsize=10, ha="center", color="0.3", fontweight="bold")
fig.text(0.465, 0.948, "dense cloud", fontsize=10, ha="center", color="0.3", fontweight="bold")
fig.text(0.66, 0.948, "gated cross-sections", fontsize=10, ha="center", color="0.3", fontweight="bold")
fig.text(0.88, 0.948, "gated CSA profile → %obstruction", fontsize=10, ha="center", color="0.3", fontweight="bold")
fig.suptitle("Three exemplar cases through the measurement layer: frames → cloud → accepted rings → scale-free calibre", fontsize=13.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.012, "Rings: wall points at the station with the fitted circle (dashed); for 2-V2 the stations are those of the reported camera-centerline registration and the solid circle is the CT lumen of equal area there. Accepted station = coverage ≥ 0.75, "
         "circle residual < 0.15, circle/ellipse agreement < 30%, end 10% excluded.\n%obstruction = 1 − A$_{min}$/A$_{ref}$ with A$_{ref}$ the 90th-percentile accepted CSA; "
         "values ≤ 30% lie within the empirical noise floor (21-V1, clinically normal, reads 29%). Frames decoded sequentially from the measured window.",
         ha="center", va="bottom", fontsize=8, color="0.4", linespacing=1.4)
fig.savefig(f"{OUT}/fig_exemplars.png", dpi=150); print("saved fig_exemplars.png")
