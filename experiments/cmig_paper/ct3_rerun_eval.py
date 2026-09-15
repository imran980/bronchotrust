"""Score a re-reconstructed CT case against CT with THREE methods (a clean per-lever ablation):
  M0  camera-trajectory centerline, median radius, NO gate       (= baseline method)
  M1  camera-trajectory centerline, median radius, coverage>=0.75 (lever a)
  M2  cloud partial-arc circle fit, cov>=0.75 + resid<0.15 + radius-MAD trim (levers a+c)
All registered with the SAME constrained isotropic Sim3. Usage: ct3_rerun_eval.py <case> <workspace>"""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, re
import numpy as np, open3d as o3d, pycolmap
from scipy.interpolate import splprep, splev
from csa_partialarc import profile as pa_profile
from airway_analysis import dce_of
from evaluation.ground_truth.ground_truth import GroundTruth
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figstyle import apply_style, PATIENT, ROLE; apply_style()

CASE, WS = sys.argv[1], sys.argv[2]
FR = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) >= 5 else None   # optional: restrict camera path to a monotonic pass
TAG = f"_f{FR[0]}-{FR[1]}" if FR else ""
D = f"{ROOT}/runs/own_data/ct_validated_3"; OUT = f"{ROOT}/runs/own_data/reports"
J = json.load(open(f"{D}/CT_validation_metrics.json")); BASE = J[CASE]["rmse"]


def gt_profile(pid):
    try:
        g = GroundTruth.load(f"{D}/ts_{pid}/gt_{pid}.json"); return np.asarray(g.arclength_mm), dce_of(g), "gt.json"
    except Exception:
        v = J[pid]; a = np.array(v["arc"])[:len(v["ct_dce"])]; return a, np.array(v["ct_dce"]), "CT samples in JSON"


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
print(f"### {CASE} re-run eval | {len(P_light)} pts | GT: {src} | baseline RMSE {BASE:.2f} ###")
res = {}
arc, rmed, cov = cam_profile(P_light, sparse)
for name, gate in (("M0 cam-centerline, no gate", np.ones_like(cov, bool)), ("M1 cam-centerline + cov>=0.75", cov >= 0.75)):
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
json.dump(dict(case=CASE, baseline=BASE, gt_source=src, methods=res), open(f"{OUT}/rerun_eval_{CASE}{TAG}.json", "w"), indent=1)

# figure: best method profile vs CT
cands = [(k, v) for k, v in res.items() if isinstance(v, dict)]
if cands:
    bestk = min(cands, key=lambda kv: kv[1]["rmse"])[0]
    if res.get("M2_scale_degenerate") and bestk.startswith("M2"): bestk = "M1 cam-centerline + cov>=0.75"   # plot the REPORTED method, not a rejected fit
    Rb = R2 if bestk.startswith("M2") else robust_register(*( (arc[(cov >= 0.75) & trim(arc, rmed)], rmed[(cov >= 0.75) & trim(arc, rmed)]) if bestk.startswith("M1") else (arc[trim(arc, rmed)], rmed[trim(arc, rmed)]) ), ca, cd)
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.plot(Rb["A"], Rb["C"], color=ROLE["ct"], lw=2.2, label="CT (ground truth)")
    ax.plot(Rb["am"][Rb["m"]], Rb["dce"][Rb["m"]], "-o", color=PATIENT.get(CASE, "#1f77b4"), ms=4.5, lw=1.6, label=f"re-run ({bestk.split()[0]})")
    ax.set_xlabel("arclength (mm)"); ax.set_ylabel("D$_{CE}$ (mm)"); ax.grid(alpha=0.25); ax.legend()
    ax.set_title(f"{CASE} re-run: RMSE {Rb['rmse']:.2f} mm (n={Rb['n']})  vs baseline {BASE:.2f} mm", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/rerun_eval_{CASE}{TAG}.png", dpi=150); print(f"  saved rerun_eval_{CASE}.png")
