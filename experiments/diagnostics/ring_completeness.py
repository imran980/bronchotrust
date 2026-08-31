"""Improve distal-ring COVERAGE on 25_V1 / 32_V2 using ONLY the existing global dense cloud
(MVS depth maps were freed, so no cheap re-fusion; no re-COLMAP here). Legitimate levers only:
(4) remove glare/outlier points (color + statistical + radial-MAD), (5) REFINED cloud medial-axis
centerline (iterate: slice perp to local tangent -> recompute medial centroid -> re-spline),
(6) re-slice perpendicular to that centerline, and a slab-THICKNESS sweep (real points from nearby
axial offsets of the SAME straight-ish ring -- NOT hallucinated fill; the r_std/r_med + estimator
gates catch any taper/union artifact). Report polar / ellipse / hull separately.

Accept a ring only if coverage >=0.75 AND estimator spread <=1.3x AND r_std/r_med tight AND stable
across nearby slices. Same global model only; scene units; no mm; no Myer-Cotton grade. If coverage
stays <75%, REJECT honestly (genuine circumferential wall gap -> needs re-capture/re-recon).
depth-eval env. -> runs/barbour30/airwayfit/ring_completeness/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, open3d as o3d, cv2
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import centerline_csa as cc

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit/ring_completeness"
REGIONS = [
    dict(tag="25_V1 dist-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", rng=(405, 452)),
    dict(tag="32_V2 dist-subglottis", model="runs/fuse_32v2/sparse/0",    cloud="runs/fuse_32v2/dense0/fused.ply",    rng=(330, 358)),
]
NBIN = 36


def load_clean(path):
    pc = o3d.io.read_point_cloud(str(ROOT / path)); P = np.asarray(pc.points); C = np.asarray(pc.colors)
    fin = np.isfinite(P).all(1); P, C = P[fin], (C[fin] if len(C) == len(fin) else np.zeros((len(P), 3)))
    n0 = len(P)
    # (4) glare / dark-noise removal by luminance
    if len(C) == len(P):
        L = C @ [0.299, 0.587, 0.114]
        keep = (L < 0.92) & (L > 0.03); P, C = P[keep], C[keep]
    # statistical outlier removal
    pc2 = o3d.geometry.PointCloud(); pc2.points = o3d.utility.Vector3dVector(P)
    pc2, _ = pc2.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    P = np.asarray(pc2.points)
    return P, n0, len(P)


def refined_centerline(P, cams, axis0, iters=2):
    Cl, Tg, seg = cc.medial_centerline(P, cams, axis0)
    if Cl is None: return None, None, seg
    for _ in range(iters):                                   # refine: recompute medial centroids on perp slices
        pts = []
        for i in range(0, len(Cl), 4):
            t, e1, e2 = cc.basis(Tg[i]); d = (seg - Cl[i]) @ t
            r0 = np.median(np.linalg.norm((seg - Cl[i]) - np.outer(d, t), axis=1))
            band = seg[np.abs(d) <= 0.35 * r0]
            if len(band) < 30: continue
            rad = np.linalg.norm((band - Cl[i]) - np.outer((band - Cl[i]) @ t, t), axis=1)
            ring = band[rad < 1.6 * np.median(rad)]
            pts.append(np.median(ring, 0))                   # medial centroid in-plane
        pts = np.array(pts)
        if len(pts) < 6: break
        k = min(3, len(pts) - 1); tck, _ = splprep(pts.T, s=len(pts) * 0.04, k=k)
        ss = np.linspace(0, 1, 200); Cl = np.array(splev(ss, tck)).T
        Tg = np.array(splev(ss, tck, der=1)).T; Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
    return Cl, Tg, seg


def poly_area(xy):
    x, y = xy[:, 0], xy[:, 1]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def measure(seg, p, t, thick):
    t, e1, e2 = cc.basis(t); d = (seg - p) @ t
    r0 = np.median(np.linalg.norm((seg - p) - np.outer(d, t), axis=1))
    band = seg[np.abs(d) <= thick * r0]
    if len(band) < 40: return None
    x, y = (band - p) @ e1, (band - p) @ e2; r = np.hypot(x, y); th = np.arctan2(y, x)
    rm0 = np.median(r); mad = np.median(np.abs(r - rm0)) + 1e-9
    ring = (np.abs(r - rm0) <= 3.5 * mad) & (r > 0.4 * rm0) & (r < 1.8 * rm0)
    x, y, r, th = x[ring], y[ring], r[ring], th[ring]
    if len(r) < 30: return None
    r_med = float(np.median(r)); ratio = float(np.std(r) / max(r_med, 1e-9))
    bins = np.linspace(-np.pi, np.pi, NBIN + 1); idx = np.digitize(th, bins) - 1
    rb = np.full(NBIN, np.nan)
    for b in range(NBIN):
        if (idx == b).any(): rb[b] = np.median(r[idx == b])
    cov = float(np.isfinite(rb).mean()); ang = (bins[:-1] + bins[1:]) / 2; good = np.isfinite(rb)
    if good.sum() < 6: return None
    rb_f = np.interp(ang, ang[good], rb[good], period=2 * np.pi)
    csa_polar = poly_area(np.c_[rb_f * np.cos(ang), rb_f * np.sin(ang)])
    try:
        (_, _), (MA, ma), _ = cv2.fitEllipse(np.c_[x, y].astype(np.float32)); csa_ell = np.pi * (MA / 2) * (ma / 2)
    except Exception: csa_ell = np.nan
    try: csa_hull = ConvexHull(np.c_[x, y]).volume
    except Exception: csa_hull = np.nan
    vals = np.array([v for v in (csa_polar, csa_ell, csa_hull) if np.isfinite(v) and v > 0])
    spread = float(vals.max() / vals.min()) if len(vals) >= 2 else np.inf
    csa = float(np.median(vals))
    return dict(cov=cov, r_med=r_med, ratio=ratio, csa_polar=float(csa_polar), csa_ell=float(csa_ell),
                csa_hull=float(csa_hull), spread=spread, csa=csa, dce=float(2 * np.sqrt(csa / np.pi)),
                xy=np.c_[x, y], rb=rb_f, ang=ang, empty_bins=int((~good).sum()))


def run(cfg):
    P, n0, n1 = load_clean(cfg["cloud"]); cams, axz = cc.region_cams(cfg["model"], cfg["rng"])
    axis0 = axz.mean(0); axis0 /= np.linalg.norm(axis0)
    Cl, Tg, seg = refined_centerline(P, cams, axis0)
    if Cl is None: return dict(tag=cfg["tag"], status="FAIL", reason="centerline"), None
    THK = [0.30, 0.40, 0.50, 0.60, 0.75]
    # BEFORE = thin 0.30 slab best coverage; AFTER = best config meeting acceptance
    best_before, best_after = None, None
    per_thick = {}
    for thk in THK:
        cand = []
        for i in range(10, 190, 2):
            m = measure(seg, Cl[i], Tg[i], thk)
            if m: m["i"] = i; cand.append(m)
        # best-coverage slice with agreeing estimators at this thickness
        ok = [m for m in cand if m["spread"] <= 1.30 and m["ratio"] <= 0.40]
        pick = max(ok, key=lambda m: m["cov"]) if ok else (max(cand, key=lambda m: m["cov"]) if cand else None)
        per_thick[thk] = (None if not pick else dict(thick=thk, cov=round(pick["cov"], 2), ratio=round(pick["ratio"], 3),
                          spread=round(pick["spread"], 3), dce=round(pick["dce"], 3), i=pick["i"]))
        if thk == 0.30 and pick: best_before = pick
        acc = [m for m in cand if m["cov"] >= 0.75 and m["spread"] <= 1.30 and m["ratio"] <= 0.35]
        if acc:
            c = min(acc, key=lambda m: m["spread"])
            if best_after is None or (c["cov"], -c["spread"]) > (best_after["cov"], -best_after["spread"]):
                best_after = c; best_after["thick"] = thk
    # characterize the best HIGH-COVERAGE ring (>=0.75): is it a strict pass, a hull-blocked
    # near-miss (polar~ellipse agree), or a genuine failure (polar vs ellipse disagree)?
    hi = []
    for thk in THK:
        for i in range(10, 190, 2):
            m = measure(seg, Cl[i], Tg[i], thk)
            if m and m["cov"] >= 0.75 and m["ratio"] <= 0.45:
                m["thick"] = thk; m["pe"] = max(m["csa_polar"], m["csa_ell"]) / min(m["csa_polar"], m["csa_ell"])
                m["dce_pe"] = 2 * np.sqrt(np.median([m["csa_polar"], m["csa_ell"]]) / np.pi); hi.append(m)
    # strict acceptance
    strict = [m for m in hi if m["spread"] <= 1.30 and m["ratio"] <= 0.35]
    best = min(strict, key=lambda m: m["spread"]) if strict else None
    stab = None
    if best:
        near = [measure(seg, Cl[i], Tg[i], best["thick"]) for i in range(max(10, best["i"] - 12), min(190, best["i"] + 13), 2)]
        d = [m["dce"] for m in near if m and m["cov"] >= 0.70 and m["spread"] <= 1.35]
        if len(d) >= 2: stab = round(100 * (max(d) - min(d)) / max(np.median(d), 1e-9), 1)
    # robust (polar+ellipse) picture at high coverage. Classify on the TYPICAL (median) polar-vs-
    # ellipse agreement across high-cov slices, not the best one -- a lone agreeing slice amid a
    # streak-scatter is coincidental, not a real ring.
    robust = min(hi, key=lambda m: m["pe"]) if hi else None
    med_pe = float(np.median([m["pe"] for m in hi])) if hi else np.inf
    maxcov = max((v["cov"] for v in per_thick.values() if v), default=0.0)
    r = dict(tag=cfg["tag"], status="OK", cloud_pts_before=n0, cloud_pts_after_clean=n1,
             coverage_before=(round(best_before["cov"], 2) if best_before else None),
             coverage_after=round(maxcov, 2), thickness_sweep=per_thick)
    if best and stab is not None and stab <= 25:
        r.update(accepted=True, thickness=best["thick"], r_std_over_r_med=round(best["ratio"], 3),
                 CSA=round(best["csa"], 4), DCE=round(best["dce"], 4), estimator_spread=round(best["spread"], 3),
                 csa_polar=round(best["csa_polar"], 4), csa_ellipse=round(best["csa_ell"], 4), csa_hull=round(best["csa_hull"], 4),
                 stability_pct=stab, reason=f"coverage {best_before['cov']:.0%}->{best['cov']:.0%}; estimators {best['spread']:.2f}<=1.3, ratio {best['ratio']:.2f}, stable")
    elif robust and med_pe <= 1.20:
        r.update(accepted=False, failure_mode="NEAR-MISS (hull/ratio-blocked; real ring w/ arc gap)",
                 median_polar_ellipse_ratio=round(med_pe, 3), robust_DCE_polar_ellipse=round(robust["dce_pe"], 3),
                 polar_ellipse_ratio=round(robust["pe"], 3), ratio_at_full_cov=round(robust["ratio"], 3),
                 three_way_spread=round(robust["spread"], 3), csa_polar=round(robust["csa_polar"], 4),
                 csa_ellipse=round(robust["csa_ell"], 4), csa_hull=round(robust["csa_hull"], 4),
                 reason=(f"coverage {best_before['cov']:.0%}->{robust['cov']:.0%} ACHIEVED; polar~ellipse AGREE "
                         f"typically (median pe {med_pe:.2f}, robust DCE~{robust['dce_pe']:.2f}) = a REAL ring, but "
                         f"full coverage needs a thick slab (no thin clean full ring) -> ratio {robust['ratio']:.2f} + "
                         f"hull-inflated 3-way spread {robust['spread']:.2f}>1.3. Strict gate NARROWLY missed; "
                         f"a denser recon could close the residual arc gap."))
    else:
        r.update(accepted=False, failure_mode="GENUINE (radial-streak scatter; robust estimators disagree)",
                 median_polar_ellipse_ratio=(round(med_pe, 3) if np.isfinite(med_pe) else None),
                 reason=(f"coverage nominally reaches {maxcov:.0%} but the distal cloud is a radial-streak MVS scatter, "
                         f"NOT a coherent wall: polar vs ellipse TYPICALLY disagree (median pe {med_pe:.2f}). "
                         f"Genuine low-parallax reconstruction-quality gap (borrowed calib); needs re-capture/re-recon, not slicing."))
    r["_viz"] = dict(Cl=Cl, Tg=Tg, seg=seg, best=(best or robust), before=best_before)
    return r


def figure(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r["status"] == "OK"]
    fig = plt.figure(figsize=(6 * len(ok), 8))
    for j, r in enumerate(ok):
        v = r["_viz"]; Cl, seg = v["Cl"], v["seg"]
        ax = fig.add_subplot(2, len(ok), j + 1, projection="3d")
        sub = seg[np.random.default_rng(0).choice(len(seg), min(4000, len(seg)), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="0.6", alpha=.3)
        ax.plot(Cl[:, 0], Cl[:, 1], Cl[:, 2], "b-", lw=2)
        ax.set_title(f"{r['tag']}\nrefined medial-axis", fontsize=9); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax = fig.add_subplot(2, len(ok), len(ok) + j + 1)
        b = v["best"] or v["before"]
        if b is not None:
            ax.scatter(b["xy"][:, 0], b["xy"][:, 1], s=4, c="tab:blue", alpha=.5)
            ax.plot(b["rb"] * np.cos(b["ang"]), b["rb"] * np.sin(b["ang"]), "r-", lw=1.5); ax.plot(0, 0, "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_title(f"{'ACCEPTED' if r.get('accepted') else 'best attempt'} cov={b['cov']:.0%} ratio={b['ratio']:.2f}\nDCE={b['dce']:.2f} spread={b['spread']:.2f} thick={b.get('thick','?')}", fontsize=8)
        cov_line = f"cov {r.get('coverage_before')}->{r.get('coverage_after')}"
        ax.set_xlabel(cov_line)
    fig.suptitle("Distal ring completeness (refined medial-axis + slab-thickness + declutter; same global cloud)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "ring_completeness.png", dpi=115); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run(c) for c in REGIONS]
    figure(rows)
    clean = [{k: v for k, v in r.items() if k != "_viz"} for r in rows]
    (OUT / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    print("=== RING COMPLETENESS (distal; same global cloud, legitimate levers only) ===")
    print(f"{'video':22}{'cov_before':>11}{'cov_after':>10}{'CSA':>8}{'DCE':>7}{'spread':>8}{'stab%':>7}  acc  reason")
    for r in rows:
        if r["status"] != "OK": print(f"{r['tag']:22}  {r['reason']}"); continue
        print(f"{r['tag']:22}{str(r.get('coverage_before')):>11}{str(r.get('coverage_after')):>10}"
              f"{str(r.get('CSA','-')):>8}{str(r.get('DCE','-')):>7}{str(r.get('estimator_spread','-')):>8}{str(r.get('stability_pct','-')):>7}"
              f"  {'Y' if r.get('accepted') else 'N'}  {r['reason']}")
    print("\nthickness sweep (coverage per slab thickness):")
    for r in rows:
        if r["status"] == "OK":
            sw = " ".join(f"{t}:{(v['cov'] if v else '-')}" for t, v in r["thickness_sweep"].items())
            print(f"  {r['tag']:22} {sw}")
    print(f"\nfigure -> {OUT}/ring_completeness.png")


if __name__ == "__main__":
    main()
