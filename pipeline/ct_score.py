"""Score ANY reconstruction against a CT ground-truth profile (--gt path/to/gt_CASE.npz) with THREE methods (a clean per-lever ablation):
  M0  camera-trajectory centerline, median radius, NO gate       (= baseline method)
  M1  camera-trajectory centerline, median radius, coverage>=0.75 (lever a)
  M2  cloud partial-arc circle fit, cov>=0.75 + resid<0.15 + radius-MAD trim (levers a+c)
All registered with the SAME constrained isotropic Sim3, then two validity rules: SELF-CONSISTENCY (M2 and M1 must
agree on median calibre within 30%, else the partial-arc scale is degenerate and M1 is reported) and the SPAN GUARD
(the mapped physical length must be >=10 mm and >=25% of the CT segment, else the fit has collapsed onto a sliver and
is not a measurement). Reported method priority: M2, then M1, then M0.
Usage: python pipeline/ct_score.py <case> <workspace> --gt ct_gt/gt_<case>.npz [--lo L --hi H] [--out-dir DIR]
  --lo/--hi restrict the camera path to one monotonic pass (a loop's path folds back on itself)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json, re
import numpy as np, open3d as o3d, pycolmap
from scipy.interpolate import splprep, splev
from csa_partialarc import profile as pa_profile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figstyle import apply_style, PATIENT, ROLE; apply_style()

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("case"); _ap.add_argument("workspace"); _ap.add_argument("--gt", required=True)
_ap.add_argument("--lo", type=int); _ap.add_argument("--hi", type=int); _ap.add_argument("--out-dir", default=None)
_a = _ap.parse_args(); CASE, WS, GTNPZ = _a.case, _a.workspace.rstrip("/"), _a.gt
FR = (_a.lo, _a.hi) if _a.lo is not None and _a.hi is not None else None
TAG = f"_f{FR[0]}-{FR[1]}" if FR else ""
OUT = _a.out_dir or WS
BASE = float("nan")


def gt_profile(pid):
    z = np.load(GTNPZ); return np.asarray(z["arclength_mm"]), np.asarray(z["dce_mm"]), os.path.basename(GTNPZ)


def robust_register(a_r, r_m, ct_arc, ct_dce, min_frac=0.6):
    a_r = a_r - a_r[0]; o = np.argsort(a_r); a_r, r_m = a_r[o], r_m[o]
    rmed = float(np.median(r_m)); s_lo, s_hi = 0.5 * ct_dce.min() / (2 * rmed), 2.0 * ct_dce.max() / (2 * rmed)
    best = None
    for flip in (False, True):
        C = ct_dce[::-1] if flip else ct_dce; A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
        for s in np.linspace(s_lo, s_hi, 1500):
            e = 2 * s * r_m - np.interp(s * a_r, A, C, left=np.nan, right=np.nan); m = np.isfinite(e)
            if m.sum() < max(6, min_frac * len(a_r)): continue
            rm = np.sqrt(np.mean(e[m] ** 2))
            if best is None or rm < best[0]: best = (rm, s, flip)
    if best is None: return None
    _, s, flip = best; C = ct_dce[::-1] if flip else ct_dce; A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
    dce = 2 * s * r_m; am = s * a_r; ct = np.interp(am, A, C, left=np.nan, right=np.nan); m = np.isfinite(ct); err = dce[m] - ct[m]
    return dict(scale=float(s), flip=bool(flip), am=am, dce=dce, ct=ct, m=m, A=A, C=C,
                rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)), n=int(m.sum()))


def cam_profile(P, sparse, NST=45):
    """Baseline method: spline through registered camera centres (ordered by frame), slabs at
    centerline points, median radius + 36-bin angular coverage."""
    rec = pycolmap.Reconstruction(sparse)
    cams = sorted(((int(re.search(r"f(\d+)", im.name).group(1)), np.asarray(im.projection_center())) for im in rec.images.values()), key=lambda x: x[0])
    if FR: cams = [c for c in cams if FR[0] <= c[0] <= FR[1]]   # monotonic pass only (a loop's camera path folds back)
    Cc = np.array([c[1] for c in cams]); tck, _ = splprep(Cc.T, u=np.linspace(0, 1, len(Cc)), s=len(Cc) * 2.0); us = np.linspace(0, 1, NST)
    C = np.array(splev(us, tck)).T; T = np.array(splev(us, tck, der=1)).T; T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]; slab = (arc[-1] / (NST - 1)) * 0.8
    rmed = np.full(NST, np.nan); cov = np.zeros(NST)
    for i, (p, t) in enumerate(zip(C, T)):
        d = P - p; al = d @ t; sel = np.abs(al) < slab / 2; q = d[sel] - np.outer(al[sel], t); r = np.linalg.norm(q, axis=1)
        if len(r) < 15: continue
        k = r < np.percentile(r, 90); q, r = q[k], r[k]
        if len(r) < 12: continue
        e1 = np.array([t[1], -t[0], 0.]); e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(t, e1)
        cov[i] = (np.histogram(np.arctan2(q @ e2, q @ e1), 36, (-np.pi, np.pi))[0] > 0).sum() / 36; rmed[i] = np.median(r)
    return arc, rmed, cov


def trim(a, r):
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() >= 6:
        med = np.median(r[ok]); mad = np.median(np.abs(r[ok] - med)) + 1e-9; ok &= np.abs(r - med) < 3.5 * mad
    return ok


fused = f"{WS}/dense0/fused.ply"; sparse = f"{WS}/sparse/0"
pc = o3d.io.read_point_cloud(fused); pc, _ = pc.remove_statistical_outlier(20, 2.0); P_light = np.asarray(pc.points)
ca, cd, src = gt_profile(CASE)
print(f"### {CASE} | {len(P_light)} pts | GT: {src} | arclen {ca.max():.0f} mm, CT D_CE {np.median(cd):.1f} mm ###")
res = {}
HAVE_POSES = os.path.isdir(sparse)
if not HAVE_POSES:
    print("  (no sparse model in this workspace -> cloud-only M2 )", flush=True)
    arc = rmed = cov = np.zeros(0)
else:
    arc, rmed, cov = cam_profile(P_light, sparse)
for name, gate in ([] if not HAVE_POSES else (("M0 cam-centerline, no gate", np.ones_like(cov, bool)), ("M1 cam-centerline + cov>=0.75", cov >= 0.75))):
    ok = gate & trim(arc, rmed)
    R = robust_register(arc[ok], rmed[ok], ca, cd) if ok.sum() >= 6 else None
    res[name] = None if R is None else {k: R[k] for k in ("rmse", "bias", "n", "scale")}
    if R is not None:   # registered median calibre + implied physical length + arrays (for the self-consistency rule / fallback)
        res[name].update(med_dce=float(np.median(R["dce"][R["m"]])), length_mm=float(R["scale"] * (arc[ok].max() - arc[ok].min())),
                         am=R["am"][R["m"]].tolist(), dce=R["dce"][R["m"]].tolist(), ct=R["ct"][R["m"]].tolist(), ct_arc=R["A"].tolist(), ct_dce=R["C"].tolist())
    print(f"  {name:32s} stations {int(ok.sum()):3d} -> " + (f"RMSE {R['rmse']:.2f}  bias {R['bias']:+.2f}  n {R['n']}  scale {R['scale']:.2f}  medDCE {np.median(R['dce'][R['m']]):.1f}mm  len {R['scale']*(arc[ok].max()-arc[ok].min()):.0f}mm" if R else "too few"))
# M2 partial-arc (csa_run cleaning)
P = np.asarray(o3d.io.read_point_cloud(fused).points); m0 = np.median(P, 0); dd = np.linalg.norm(P - m0, axis=1)
P = P[dd < np.median(dd) + 4 * np.median(np.abs(dd - np.median(dd)))]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pcd, _ = pcd.remove_statistical_outlier(24, 1.8); pcd, _ = pcd.remove_radius_outlier(16, np.median(dd) * 0.03)
rows, _ = pa_profile(np.asarray(pcd.points), nb=60)
t = np.array([r["t"] for r in rows]); cv = np.array([r["cov"] for r in rows]); Rr = np.array([r["R"] for r in rows]); rs = np.array([r["resid"] for r in rows])
n = len(t); lo, hi = int(np.floor(0.1 * n)), int(np.ceil(0.9 * n)); inter = np.zeros(n, bool); inter[lo:hi] = True
ok = (cv >= 0.75) & (rs < 0.15) & np.isfinite(Rr) & (Rr > 0) & inter
if ok.sum() >= 4:
    rm = np.median(Rr[ok]); mad = np.median(np.abs(Rr[ok] - rm)) + 1e-9; ok &= (np.abs(Rr - rm) < 4 * mad) & (Rr > 0.4 * rm) & (Rr < 2.5 * rm)
R2 = robust_register(t[ok], Rr[ok], ca, cd) if ok.sum() >= 6 else None
res["M2 partial-arc gated"] = None if R2 is None else {k: R2[k] for k in ("rmse", "bias", "n", "scale")}
if R2 is not None:   # save the fitted arrays so the pooled Bland-Altman / final figure can be rebuilt
    res["M2 partial-arc gated"].update(am=R2["am"][R2["m"]].tolist(), dce=R2["dce"][R2["m"]].tolist(), ct=R2["ct"][R2["m"]].tolist(),
                                       ct_arc=R2["A"].tolist(), ct_dce=R2["C"].tolist(),
                                       med_dce=float(np.median(R2["dce"][R2["m"]])), length_mm=float(R2["scale"] * (t[ok].max() - t[ok].min())))
    m1 = res.get("M1 cam-centerline + cov>=0.75")
    if m1 and m1.get("med_dce"):
        ratio = res["M2 partial-arc gated"]["med_dce"] / m1["med_dce"]
        res["M2_vs_M1_calibre_ratio"] = float(ratio); res["M2_scale_degenerate"] = bool(abs(ratio - 1) > 0.30)
        print(f"  self-consistency: M2 median calibre / M1 median calibre = {ratio:.2f}  -> " + ("M2 REJECTED (scale-degenerate, >30% apart)" if abs(ratio - 1) > 0.30 else "consistent"))
print(f"  {'M2 partial-arc gated (a+c)':32s} stations {int(ok.sum()):3d} -> " + (f"RMSE {R2['rmse']:.2f}  bias {R2['bias']:+.2f}  n {R2['n']}  scale {R2['scale']:.2f}" if R2 else "too few"))
# PRE-DECLARED SPAN GUARD: an isotropic fit can collapse the reconstruction onto a short sub-segment of the CT
# and report a flatteringly small RMSE (the documented 30-V2 false-positive). Require the mapped physical length
# to be at least 10 mm and at least 25% of the CT extent. This is a validity flag, not a tuning knob.
ct_len = float(ca.max() - ca.min())
for k, v in list(res.items()):
    if isinstance(v, dict) and v.get("length_mm") is not None:
        v["span_frac_of_CT"] = float(v["length_mm"] / max(ct_len, 1e-6))
        v["collapsed"] = bool(v["length_mm"] < max(10.0, 0.25 * ct_len))
        if v["collapsed"]:
            print(f"  !! {k}: mapped length {v['length_mm']:.1f} mm = {100*v['span_frac_of_CT']:.0f}% of the {ct_len:.0f} mm CT segment -> COLLAPSED FIT, not a valid measurement")
json.dump(dict(case=CASE, workspace=os.path.basename(WS.rstrip("/")), baseline=BASE, gt_source=src, ct_arclen_mm=ct_len, methods=res), open(f"{OUT}/ctscore_{os.path.basename(WS.rstrip(chr(47)))}{TAG}.json", "w"), indent=1)

# figure: best method profile vs CT
# plot the REPORTED method: lowest RMSE among fits that pass BOTH the self-consistency rule and the span guard
def _valid(k, v):
    if not isinstance(v, dict) or "rmse" not in v: return False
    if v.get("collapsed"): return False
    if k.startswith("M2") and res.get("M2_scale_degenerate"): return False
    return True
# method priority follows the manuscript: gated partial-arc, then gated camera-centerline, then the ungated baseline
PRIORITY = ["M2 partial-arc gated", "M1 cam-centerline + cov>=0.75", "M0 cam-centerline, no gate"]
cands = [(k, res[k]) for k in PRIORITY if _valid(k, res.get(k))]
if not cands: print("  NO VALID FIT (all collapsed or scale-degenerate) -> no figure")
if cands:
    bestk = cands[0][0]
    print(f"  REPORTED: {bestk} RMSE {res[bestk]['rmse']:.2f} mm, bias {res[bestk]['bias']:+.2f}, n {res[bestk]['n']}, spans {res[bestk].get('length_mm',float('nan')):.0f}/{ct_len:.0f} mm")
    Rb = R2 if (bestk.startswith("M2") or not HAVE_POSES) else robust_register(*( (arc[(cov >= 0.75) & trim(arc, rmed)], rmed[(cov >= 0.75) & trim(arc, rmed)]) if bestk.startswith("M1") else (arc[trim(arc, rmed)], rmed[trim(arc, rmed)]) ), ca, cd)
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.plot(Rb["A"], Rb["C"], color=ROLE["ct"], lw=2.2, label="CT (ground truth)")
    ax.plot(Rb["am"][Rb["m"]], Rb["dce"][Rb["m"]], "-o", color=PATIENT.get(CASE, "#1f77b4"), ms=4.5, lw=1.6, label=f"reconstruction ({bestk.split()[0]})")
    ax.set_xlabel("arclength (mm)"); ax.set_ylabel("D$_{CE}$ (mm)"); ax.grid(alpha=0.25); ax.legend()
    ax.set_title(f"{CASE}: {bestk.split()[0]} RMSE {Rb['rmse']:.2f} mm, n={Rb['n']}, spans {res[bestk].get('length_mm',0):.0f} of {ct_len:.0f} mm CT", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/ctscore_{os.path.basename(WS.rstrip(chr(47)))}{TAG}.png", dpi=150); print(f"  saved ctscore_{os.path.basename(WS.rstrip(chr(47)))}{TAG}.png")
