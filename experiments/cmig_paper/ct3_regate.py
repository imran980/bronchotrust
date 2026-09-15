"""Levers (a)+(c) on the EXISTING CT-case clouds: validated partial-arc circle fit per slice
(csa_partialarc.profile) + coverage/residual gate, then isotropic Sim3 registration to the CT D_CE
profile. Compares to the baseline camera-centerline RMSE (CT_validation_metrics.json).
Two gate variants: STRICT (cov>=0.75, resid<0.15, circle/ellipse agree<30%) and CIRCLE-ONLY
(cov>=0.75, resid<0.15; no ellipse agreement — 2-V2's ellipse fit is unstable)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json
import numpy as np, open3d as o3d
from csa_partialarc import profile
from airway_analysis import dce_of
from evaluation.ground_truth.ground_truth import GroundTruth


def robust_register(a_r, r_m, ct_arc, ct_dce, min_span=0.0, min_frac=0.6):
    """Isotropic scale + flip, CONSTRAINED so it cannot collapse: (i) the recon's median calibre must
    map into the CT calibre range (0.5*min .. 2*max) — this alone excludes the degenerate tiny-scale
    solution; (ii) >=60% of recon stations must land on the CT. No span requirement: legitimately
    short reconstructions (e.g. 20-V1 covers ~9 mm of a 65 mm CT profile) must not be penalised."""
    a_r = a_r - a_r[0]; o = np.argsort(a_r); a_r, r_m = a_r[o], r_m[o]
    rmed = float(np.median(r_m)); L = float(ct_arc[-1] - ct_arc[0])
    s_lo, s_hi = 0.5 * ct_dce.min() / (2 * rmed), 2.0 * ct_dce.max() / (2 * rmed)
    ss = np.linspace(s_lo, s_hi, 1500); best = None
    for flip in (False, True):
        C = ct_dce[::-1] if flip else ct_dce; A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
        for s in ss:
            if s * a_r[-1] < min_span * L: continue
            e = 2 * s * r_m - np.interp(s * a_r, A, C, left=np.nan, right=np.nan); m = np.isfinite(e)
            if m.sum() < max(6, min_frac * len(a_r)): continue
            rm = np.sqrt(np.mean(e[m] ** 2))
            if best is None or rm < best[0]: best = (rm, s, flip)
    if best is None: return None
    _, s, flip = best; C = ct_dce[::-1] if flip else ct_dce; A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
    dce = 2 * s * r_m; am = s * a_r; ct = np.interp(am, A, C, left=np.nan, right=np.nan); m = np.isfinite(ct); err = dce[m] - ct[m]
    return dict(scale=float(s), flip=bool(flip), am=am, dce=dce, ct=ct, m=m, rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)), n=int(m.sum()))

D = f"{ROOT}/runs/own_data/ct_validated_3"
J = json.load(open(f"{D}/CT_validation_metrics.json"))
END = 0.10


def load_clean(fp):
    P = np.asarray(o3d.io.read_point_cloud(fp).points)
    m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=np.median(d) * 0.03)
    return np.asarray(pcd.points)


def gt_profile(pid):
    """Full CT D_CE profile. 2-V2/20-V1 from gt json; 50-V2 from the registered CT samples in the JSON."""
    try:
        g = GroundTruth.load(f"{D}/ts_{pid}/gt_{pid}.json"); return np.asarray(g.arclength_mm), dce_of(g), "gt.json"
    except Exception:
        v = J[pid]; a = np.array(v["arc"])[:len(v["ct_dce"])]; return a, np.array(v["ct_dce"]), "CT samples in JSON"


out = {}
print(f"{'case':6s} {'gate':11s} {'nst':>4s} {'RMSE':>6s} {'bias':>6s} {'n':>3s} {'scale':>6s}  baseline")
for pid in ["2-V2", "20-V1", "50-V2"]:
    P = load_clean(f"{D}/{pid}_reconstruction.ply")
    rows, _ = profile(P, nb=60)
    t = np.array([r["t"] for r in rows]); cov = np.array([r["cov"] for r in rows]); Rr = np.array([r["R"] for r in rows])
    res = np.array([r["resid"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows]); ce = np.array([r["csa_ell"] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"): agree = np.abs(ce - cc) / cc
    n = len(t); lo, hi = int(np.floor(END * n)), int(np.ceil((1 - END) * n)); interior = np.zeros(n, bool); interior[lo:hi] = True
    base = (cov >= 0.75) & (res < 0.15) & np.isfinite(Rr) & (Rr > 0) & interior
    # radius-outlier trim: a near-straight partial arc fits a huge circle with a tiny RELATIVE residual,
    # so the resid gate alone lets it through — reject radii far from the robust median (uniform rule)
    if base.sum() >= 4:
        rm = np.median(Rr[base]); mad = np.median(np.abs(Rr[base] - rm)) + 1e-9
        base &= (np.abs(Rr - rm) < 4.0 * mad) & (Rr > 0.4 * rm) & (Rr < 2.5 * rm)
    gates = {"circle-only": base, "strict": base & np.isfinite(agree) & (agree < 0.30)}
    ca, cd, src = gt_profile(pid); out[pid] = {"gt_source": src, "baseline_rmse": J[pid]["rmse"]}
    for gname, ok in gates.items():
        if ok.sum() < 6:
            print(f"{pid:6s} {gname:11s} {int(ok.sum()):4d}   -- too few stations --"); out[pid][gname] = None; continue
        Rg = robust_register(t[ok], Rr[ok], ca, cd)
        out[pid][gname] = {k: Rg[k] for k in ("rmse", "bias", "n", "scale", "flip")}
        out[pid][gname].update(n_stations=int(ok.sum()), am=Rg["am"].tolist(), dce=Rg["dce"].tolist(), ct=np.where(Rg["m"], Rg["ct"], np.nan).tolist())
        print(f"{pid:6s} {gname:11s} {int(ok.sum()):4d} {Rg['rmse']:6.2f} {Rg['bias']:+6.2f} {Rg['n']:3d} {Rg['scale']:6.2f}  {J[pid]['rmse']:.2f}  (GT: {src})")
json.dump(out, open(f"{ROOT}/runs/own_data/reports/ct3_regate.json", "w"), indent=1)
print("saved ct3_regate.json")
