"""Scale-free calibre measurement on one dense cloud: gated cross-sectional-area profile and %obstruction.

Same estimator and gates as the paper's cohort table. Stations are slices perpendicular to the lumen axis; each is
fitted with a partial-arc circle AND an ellipse. A station is accepted only if angular coverage >= 0.75, circle
residual < 0.15 and the two fits agree within 30%; the end 10% of stations (funnel / carina) are excluded and a
3-station median filter removes single-station spikes. %obstruction = 1 - A_min / A_ref with A_ref the 90th
percentile of accepted CSA. Values <= 30% lie within the empirical noise floor (a clinically normal airway read 29%).
CV of the accepted profile > 0.15 flags unreliable geometry. Output is in scene units: it is a RATIO, no scale needed.

Usage:
  python pipeline/measure_csa.py <fused.ply> [--out result.json] [--plot profile.png]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, open3d as o3d
from scipy.ndimage import median_filter
from csa_partialarc import profile

COV, RESID, AGREE, END, MINVALID = 0.75, 0.15, 0.30, 0.10, 10


def load_clean(fp):
    P = np.asarray(o3d.io.read_point_cloud(fp).points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc, _ = pc.remove_statistical_outlier(24, 1.8); pc, _ = pc.remove_radius_outlier(16, np.median(d) * 0.03)
    return np.asarray(pc.points)


def measure(P):
    rows, _ = profile(P, nb=60)
    t = np.array([r["t"] for r in rows]); cov = np.array([r["cov"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows])
    ce = np.array([r["csa_ell"] for r in rows]); res = np.array([r["resid"] for r in rows])
    with np.errstate(invalid="ignore", divide="ignore"): agree = np.abs(ce - cc) / cc
    ok = (cov >= COV) & (res < RESID) & np.isfinite(agree) & (agree < AGREE) & np.isfinite(cc) & (cc > 0)
    n = len(t); lo, hi = int(np.floor(END * n)), int(np.ceil((1 - END) * n)); inter = np.zeros(n, bool); inter[lo:hi] = True; ok &= inter
    out = dict(n_stations=int(n), n_accepted=int(ok.sum()), median_coverage=float(np.median(cov[lo:hi])) if n else 0.0,
               tier="measurable tube" if ok.sum() >= 20 else ("partially measurable" if ok.sum() >= 10 else "not measurable"))
    if ok.sum() < MINVALID:
        out.update(measurable=False, reason=f"only {int(ok.sum())} strict-valid interior stations"); return out, (t, cc, cov, ok, None)
    sm = median_filter(cc[ok], size=3, mode="nearest"); imin = int(np.argmin(sm)); a_min = float(sm[imin]); a_ref = float(np.percentile(sm, 90))
    out.update(measurable=True, A_min=a_min, A_ref=a_ref, pct_obstruction=float((1 - a_min / a_ref) * 100), dce_ratio=float(np.sqrt(a_min / a_ref)),
               cv=float(np.std(sm) / np.mean(sm)), min_at_edge=bool(imin <= 1 or imin >= len(sm) - 2),
               note=("within the ~30% empirical noise floor" if (1 - a_min / a_ref) * 100 <= 30 else "above the noise floor") + ("; CV>0.15 -> unreliable geometry" if np.std(sm) / np.mean(sm) > 0.15 else ""))
    return out, (t, cc, cov, ok, (t[ok][imin], a_min, a_ref))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply"); ap.add_argument("--out"); ap.add_argument("--plot")
    a = ap.parse_args()
    P = load_clean(a.ply); out, (t, cc, cov, ok, mark) = measure(P); out["n_clean_points"] = int(len(P)); out["cloud"] = a.ply
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))
    if a.out: json.dump(out, open(a.out, "w"), indent=2)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(t, cc, "-", color="0.85", lw=0.9); ax.scatter(t[~ok], cc[~ok], s=12, color="0.7", label="rejected")
        ax.scatter(t[ok], cc[ok], s=22, c=cov[ok], cmap="viridis", vmin=0.6, vmax=1, label="accepted (colour = coverage)")
        if mark: ax.axhline(mark[2], color="#0072B2", ls="--"); ax.axhline(mark[1], color="#D55E00", ls="--"); ax.plot([mark[0]], [mark[1]], "v", color="#D55E00", ms=9)
        ax.set_xlabel("station along the lumen axis (scene units)"); ax.set_ylabel("CSA (scene units$^2$)"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax.set_title(f"{os.path.basename(a.ply)}: {out['n_accepted']}/{out['n_stations']} accepted" + (f" · %obstruction {out['pct_obstruction']:.0f}% · CV {out['cv']:.2f}" if out.get("measurable") else " · not measurable"))
        fig.tight_layout(); fig.savefig(a.plot, dpi=130); print("saved", a.plot)


if __name__ == "__main__":
    main()
