"""Measure the re-reconstructed 25_V1 distal ring (both geometric + photometric fusion). Refined
cloud medial-axis centerline, PERPENDICULAR slicing, CSA 3 ways (polar/ellipse/hull). Emphasis on
THIN-slab coverage (no thick-slab faking). Accept iff thin-slab coverage >=0.75 AND polar~ellipse
AND 3-way spread <=1.3 AND r_std/r_med tight AND stable. Same (new) connected model only; scene
units. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, open3d as o3d, cv2
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import centerline_csa as cc, ring_completeness as rc

ROOT = Path("/home/mi3dr/projects/bronchotrust")
WORK = ROOT / "runs/barbour30/airwayfit/recon_25v1_distal"
MODEL = str((WORK / "sparse/0"))
RNG = (390, 470)
THIN = 0.30
NBIN = 36


def _poly(xy): x, y = xy[:, 0], xy[:, 1]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def measure_declutter(seg, p, t, thick=THIN):
    """THIN-slab ring with radial-MAD trim + 2D statistical-outlier removal (lever 4: drop the
    scattered stray MVS points that inflate the convex hull). Then CSA 3 ways."""
    t, e1, e2 = cc.basis(t); d = (seg - p) @ t
    r0 = np.median(np.linalg.norm((seg - p) - np.outer(d, t), axis=1))
    band = seg[np.abs(d) <= thick * r0]
    if len(band) < 60: return None
    x, y = (band - p) @ e1, (band - p) @ e2; r = np.hypot(x, y)
    rm = np.median(r); mad = np.median(np.abs(r - rm)) + 1e-9
    m = (np.abs(r - rm) <= 3.5 * mad) & (r > 0.4 * rm) & (r < 1.8 * rm); x, y = x[m], y[m]
    if len(x) < 40: return None
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.c_[x, y, np.zeros(len(x))])
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    q = np.asarray(pc.points); x, y = q[:, 0], q[:, 1]
    if len(x) < 40: return None
    r = np.hypot(x, y); th = np.arctan2(y, x); r_med = float(np.median(r)); ratio = float(np.std(r) / max(r_med, 1e-9))
    bins = np.linspace(-np.pi, np.pi, NBIN + 1); idx = np.digitize(th, bins) - 1
    rb = np.array([np.median(r[idx == b]) if (idx == b).any() else np.nan for b in range(NBIN)])
    ang = (bins[:-1] + bins[1:]) / 2; good = np.isfinite(rb); cov = float(good.mean())
    if good.sum() < 6: return None
    rbf = np.interp(ang, ang[good], rb[good], period=2 * np.pi)
    csa_polar = _poly(np.c_[rbf * np.cos(ang), rbf * np.sin(ang)])
    try: (_, _), (MA, ma), _ = cv2.fitEllipse(np.c_[x, y].astype(np.float32)); csa_ell = np.pi * (MA / 2) * (ma / 2)
    except Exception: csa_ell = np.nan
    try: csa_hull = ConvexHull(np.c_[x, y]).volume
    except Exception: csa_hull = np.nan
    vals = np.array([v for v in (csa_polar, csa_ell, csa_hull) if np.isfinite(v) and v > 0])
    spread = float(vals.max() / vals.min()) if len(vals) >= 2 else np.inf
    csa = float(np.median(vals))
    return dict(cov=cov, r_med=r_med, ratio=ratio, csa_polar=float(csa_polar), csa_ell=float(csa_ell),
                csa_hull=float(csa_hull), spread=spread, csa=csa, dce=float(2 * np.sqrt(csa / np.pi)),
                xy=np.c_[x, y], rb=rbf, ang=ang, pe=float(max(csa_polar, csa_ell) / min(csa_polar, csa_ell)))


def measure_cloud(cloud_path, tag):
    P, n0, n1 = rc.load_clean(str(cloud_path))
    cams, axz = cc.region_cams(MODEL, RNG)
    axis0 = axz.mean(0); axis0 /= np.linalg.norm(axis0)
    Cl, Tg, seg = rc.refined_centerline(P, cams, axis0)
    if Cl is None: return dict(fusion=tag, status="FAIL", reason="centerline"), None
    slices = []
    for i in range(10, 190, 2):
        m = measure_declutter(seg, Cl[i], Tg[i], THIN)   # THIN slab + declutter
        if m: m["i"] = i; slices.append(m)
    if not slices: return dict(fusion=tag, status="FAIL", reason="no slices"), None
    acc = [m for m in slices if m["cov"] >= 0.75 and m["spread"] <= 1.30 and m["ratio"] <= 0.35]
    best = max(acc, key=lambda m: (round(m["cov"], 2), -m["spread"])) if acc else None   # prefer near-full coverage
    # best high-coverage attempt regardless (to characterize)
    hi = [m for m in slices if m["cov"] >= 0.75]
    bcov = max(slices, key=lambda m: m["cov"])
    stab = None
    if best:
        d = [m["dce"] for m in slices if abs(m["i"] - best["i"]) <= 12 and m["cov"] >= 0.70 and m["spread"] <= 1.35]
        if len(d) >= 2: stab = round(100 * (max(d) - min(d)) / max(np.median(d), 1e-9), 1)
    r = dict(fusion=tag, status="OK", cloud_pts=n1, n_slices=len(slices), n_accepted=len(acc),
             thin_slab_max_coverage=round(bcov["cov"], 2),
             median_pe_at_hi_cov=(round(float(np.median([m["pe"] for m in hi])), 3) if hi else None))
    if best and stab is not None and stab <= 25:
        r.update(accepted=True, coverage=round(best["cov"], 2), r_std_over_r_med=round(best["ratio"], 3),
                 CSA=round(best["csa"], 4), DCE=round(best["dce"], 4), estimator_spread=round(best["spread"], 3),
                 csa_polar=round(best["csa_polar"], 4), csa_ellipse=round(best["csa_ell"], 4), csa_hull=round(best["csa_hull"], 4),
                 stability_pct=stab, reason=f"THIN-slab coverage {best['cov']:.0%}, estimators {best['spread']:.2f}<=1.3, ratio {best['ratio']:.2f}, stable ±{stab}%")
    else:
        r.update(accepted=False, best_attempt=(None if not bcov else dict(cov=round(bcov["cov"], 2), ratio=round(bcov["ratio"], 3),
                 spread=round(bcov["spread"], 3), dce=round(bcov["dce"], 3), pe=round(bcov["pe"], 3))),
                 reason=f"thin-slab acceptance not met (max thin cov {bcov['cov']:.0%}, best-cov spread {bcov['spread']:.2f}, pe {bcov['pe']:.2f})")
    return r, dict(Cl=Cl, seg=seg, best=(best or bcov), slices=slices)


def main():
    results, extras = [], []
    for typ in ("geometric", "photometric"):
        cp = WORK / "dense" / f"fused_{typ}.ply"
        if not cp.exists(): print(f"[{typ}] missing {cp}"); continue
        r, e = measure_cloud(cp, typ); results.append(r); extras.append(e)
        if r["status"] != "OK": print(f"[{typ}] FAIL {r['reason']}"); continue
        print(f"[{typ}] thin-cov max={r['thin_slab_max_coverage']} accepted={r['accepted']} "
              f"DCE={r.get('DCE','-')} spread={r.get('estimator_spread','-')} stab={r.get('stability_pct','-')} :: {r['reason']}", flush=True)
    # figure
    ok = [(r, e) for r, e in zip(results, extras) if r["status"] == "OK" and e]
    if ok:
        fig = plt.figure(figsize=(6 * len(ok), 5))
        for j, (r, e) in enumerate(ok):
            b = e["best"]
            ax = fig.add_subplot(1, len(ok), j + 1)
            ax.scatter(b["xy"][:, 0], b["xy"][:, 1], s=4, c="tab:blue", alpha=.5)
            ax.plot(b["rb"] * np.cos(b["ang"]), b["rb"] * np.sin(b["ang"]), "r-", lw=1.6); ax.plot(0, 0, "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_title(f"25_V1 distal RE-RECON [{r['fusion']}]\n{'ACCEPTED' if r.get('accepted') else 'best attempt'} "
                         f"thin-cov={b['cov']:.0%} ratio={b['ratio']:.2f}\nDCE={b['dce']:.2f} spread={b['spread']:.2f} (thin slab)", fontsize=9)
        fig.suptitle("25_V1 distal re-reconstruction — thin-slab ring (denser frames + exhaustive + fresh MVS)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(WORK / "measured_ring.png", dpi=120); plt.close(fig)
    (WORK / "measure_report.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nfigure -> {WORK}/measured_ring.png")


if __name__ == "__main__":
    main()
