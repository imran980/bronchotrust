"""Centerline-slice CSA on the DISTAL dense rings (where the geometry-gap audit showed usable
triangulation 3.7-4.3deg + 100%-coverage wall rings). Uses the airway MEDIAL-AXIS centerline from
the point cloud (slab-centroid path + smoothing spline), NOT the camera-trajectory PCA; slices
PERPENDICULAR to the local centerline tangent; measures CSA three independent ways (polar-median
polygon, ellipse fit, convex-hull free-contour). Accepts a slice only if coverage is high AND the
three estimators agree within <=1.3x. depth-eval env. -> runs/barbour30/airwayfit/centerline_csa/.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d, cv2
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit/centerline_csa"
REGIONS = [
    dict(tag="2_V2 dist-subglottis",  model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  rng=(985, 1045)),
    dict(tag="32_V2 dist-subglottis", model="runs/fuse_32v2/sparse/0",    cloud="runs/fuse_32v2/dense0/fused.ply",    rng=(330, 358)),
    dict(tag="25_V1 dist-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", rng=(405, 452)),
]
COV_HIGH, SPREAD_MAX = 0.75, 1.30       # acceptance: high coverage AND estimators agree <=1.3x
NBIN = 36


def clean_cloud(path):
    P = np.asarray(o3d.io.read_point_cloud(str(ROOT / path)).points); P = P[np.isfinite(P).all(1)]
    if len(P) < 50: return P
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
    return np.asarray(pc.points)


def region_cams(model, rng):
    rec = pycolmap.Reconstruction(str(ROOT / model)); C, ax = [], []
    for im in rec.images.values():
        f = int(re.search(r"f(\d+)", im.name).group(1))
        if rng[0] <= f <= rng[1]:
            C.append(np.array(im.projection_center())); ax.append(np.array(im.cam_from_world().matrix())[2, :3])
    return np.array(C), np.array(ax)


def basis(t):
    t = t / (np.linalg.norm(t) + 1e-12); a = np.array([0, 0, 1.]) if abs(t[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1) + 1e-12; return t, e1, np.cross(t, e1)


def medial_centerline(P, cams, axis0):
    """slab-centroid medial axis in the distal window, then a smoothing spline."""
    Cc = cams.mean(0); u_cam = (cams - Cc) @ axis0
    u = (P - Cc) @ axis0
    lo, hi = np.percentile(u_cam, 2), np.percentile(u_cam, 98)
    seg = P[(u >= lo - 0.1 * (hi - lo)) & (u <= hi + 0.8 * (hi - lo))]        # distal window + forward
    if len(seg) < 200: return None, None, seg
    us = (seg - Cc) @ axis0
    edges = np.linspace(us.min(), us.max(), 22); cents, uu = [], []
    for i in range(len(edges) - 1):
        m = (us >= edges[i]) & (us < edges[i + 1]); sl = seg[m]
        if len(sl) < 30: continue
        c = np.median(sl, 0)                                                 # slab centroid = medial-axis point
        # trim slab to the ring (radial) then re-centroid for robustness
        rad = np.linalg.norm((sl - c) - ((sl - c) @ axis0)[:, None] * axis0, axis=1)
        keep = sl[rad < 2.0 * np.median(rad)]
        cents.append(np.median(keep, 0)); uu.append(0.5 * (edges[i] + edges[i + 1]))
    cents = np.array(cents)
    if len(cents) < 6: return None, None, seg
    k = min(3, len(cents) - 1); tck, _ = splprep(cents.T, s=len(cents) * 0.05, k=k)
    ss = np.linspace(0, 1, 200); Cl = np.array(splev(ss, tck)).T
    Tg = np.array(splev(ss, tck, der=1)).T; Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
    return Cl, Tg, seg


def poly_area(xy):
    x, y = xy[:, 0], xy[:, 1]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def slice_csa(P, p, t):
    t, e1, e2 = basis(t)
    d = (P - p) @ t
    r0 = np.median(np.linalg.norm((P - p) - np.outer((P - p) @ t, t), axis=1))
    band = P[np.abs(d) <= 0.30 * r0]
    if len(band) < 40: return None
    x, y = (band - p) @ e1, (band - p) @ e2; r = np.hypot(x, y); th = np.arctan2(y, x)
    rm0 = np.median(r); ring = (r > 0.45 * rm0) & (r < 1.7 * rm0)             # isolate the wall annulus
    x, y, r, th = x[ring], y[ring], r[ring], th[ring]
    if len(r) < 30: return None
    r_med = float(np.median(r)); ratio = float(np.std(r) / max(r_med, 1e-9))
    bins = np.linspace(-np.pi, np.pi, NBIN + 1); idx = np.digitize(th, bins) - 1
    rb = np.full(NBIN, np.nan)
    for b in range(NBIN):
        if (idx == b).any(): rb[b] = np.median(r[idx == b])
    cov = float(np.isfinite(rb).mean())
    # fill empty sectors by circular interpolation (for the polar polygon)
    ang = (bins[:-1] + bins[1:]) / 2; good = np.isfinite(rb)
    if good.sum() < 6: return None
    rb_f = np.interp(ang, ang[good], rb[good], period=2 * np.pi)
    csa_polar = poly_area(np.c_[rb_f * np.cos(ang), rb_f * np.sin(ang)])
    # ellipse
    try:
        (cx, cy), (MA, ma), _ = cv2.fitEllipse(np.c_[x, y].astype(np.float32)); csa_ell = np.pi * (MA / 2) * (ma / 2)
    except Exception:
        csa_ell = np.nan
    # convex-hull free-contour
    try:
        csa_hull = ConvexHull(np.c_[x, y]).volume
    except Exception:
        csa_hull = np.nan
    vals = np.array([v for v in (csa_polar, csa_ell, csa_hull) if np.isfinite(v) and v > 0])
    spread = float(vals.max() / vals.min()) if len(vals) >= 2 else np.inf
    csa = float(np.median(vals))
    return dict(cov=cov, r_med=r_med, ratio=ratio, csa_polar=float(csa_polar), csa_ell=float(csa_ell),
                csa_hull=float(csa_hull), spread=spread, csa=csa, dce=float(2 * np.sqrt(csa / np.pi)),
                n=int(len(r)), xy=np.c_[x, y], rb=rb_f, ang=ang)


def run(cfg):
    P = clean_cloud(cfg["cloud"]); cams, axz = region_cams(cfg["model"], cfg["rng"])
    if len(P) < 200 or len(cams) < 4: return dict(tag=cfg["tag"], status="FAIL", reason="cloud/cams too small"), None
    axis0 = axz.mean(0); axis0 /= np.linalg.norm(axis0)                       # airway direction = mean optical axis
    Cl, Tg, seg = medial_centerline(P, cams, axis0)
    if Cl is None: return dict(tag=cfg["tag"], status="FAIL", reason="centerline failed"), None
    slices = []
    for i in range(10, 190, 3):                                              # skip spline ends
        s = slice_csa(seg, Cl[i], Tg[i])
        if s: s["i"] = i; s["p"] = Cl[i]; slices.append(s)
    acc = [s for s in slices if s["cov"] >= COV_HIGH and s["spread"] <= SPREAD_MAX and s["ratio"] <= 0.35]
    best = min(acc, key=lambda s: s["spread"]) if acc else None
    # best ATTEMPT (min-spread slice regardless of acceptance) -> shows HOW rejects failed
    battempt = min(slices, key=lambda s: s["spread"]) if slices else None
    # stability: DCE across accepted slices within +-6 spline steps of best
    band = None; stab_pct = None
    if best:
        near = [s for s in acc if abs(s["i"] - best["i"]) <= 12]
        dces = [s["dce"] for s in near]
        band = [round(min(dces), 3), round(max(dces), 3)]
        stab_pct = round(100 * (max(dces) - min(dces)) / max(np.median(dces), 1e-9), 1)
    r = dict(tag=cfg["tag"], status="OK", n_slices=len(slices), n_accepted=len(acc),
             centerline="cloud medial-axis (slab-centroid + smoothing spline)",
             best=(None if not best else dict(spline_i=best["i"], coverage=round(best["cov"], 2),
                   r_std_over_r_med=round(best["ratio"], 3), CSA=round(best["csa"], 4), DCE=round(best["dce"], 4),
                   csa_polar=round(best["csa_polar"], 4), csa_ellipse=round(best["csa_ell"], 4),
                   csa_hull=round(best["csa_hull"], 4), estimator_spread=round(best["spread"], 3))),
             stability_band_DCE=band, stability_pct=stab_pct,
             best_attempt=(None if not battempt else dict(coverage=round(battempt["cov"], 2),
                   r_std_over_r_med=round(battempt["ratio"], 3), DCE=round(battempt["dce"], 3),
                   estimator_spread=round(battempt["spread"], 3),
                   csa_polar=round(battempt["csa_polar"], 3), csa_ellipse=round(battempt["csa_ell"], 3),
                   csa_hull=round(battempt["csa_hull"], 3))))
    if best and stab_pct is not None and stab_pct <= 25:
        r["verdict"] = "ACCEPT: clean centerline-slice CSA (cov high, estimators <=1.3x, stable)"
    elif best:
        r["verdict"] = "WEAK: accepted slices but unstable/borderline"
    elif battempt and battempt["spread"] <= SPREAD_MAX:
        r["verdict"] = f"REJECT (coverage-limited): best ring cov {battempt['cov']:.0%}, estimators AGREE ({battempt['spread']:.2f}) -> partial wall = recon-completeness/recipe gap"
    else:
        r["verdict"] = "REJECT (noisy ring): estimators disagree >1.3x at every slice"
    return r, dict(P=P, seg=seg, Cl=Cl, Tg=Tg, cams=cams, slices=slices, acc=acc, best=best, axis0=axis0)


def figure(rows, extras):
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [(r, e) for r, e in zip(rows, extras) if r["status"] == "OK" and e]
    if not ok: return
    fig = plt.figure(figsize=(5.2 * len(ok), 9))
    for j, (r, e) in enumerate(ok):
        Cl, seg, best = e["Cl"], e["seg"], e["best"]
        ax = fig.add_subplot(3, len(ok), j + 1, projection="3d")
        sub = seg[np.random.default_rng(0).choice(len(seg), min(4000, len(seg)), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="0.6", alpha=.3)
        ax.plot(Cl[:, 0], Cl[:, 1], Cl[:, 2], "b-", lw=2, label="medial centerline")
        for s in e["acc"]:
            t, e1, e2 = basis(e["Tg"][s["i"]]); th = np.linspace(0, 2 * np.pi, 60)
            ring = s["p"] + np.outer(np.cos(th), e1 * 0) + 0  # placeholder
        if best:
            t, e1, e2 = basis(e["Tg"][best["i"]]); th = np.linspace(0, 2 * np.pi, 80)
            ring = best["p"][None] + best["r_med"] * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], "r-", lw=2, label="best slice")
        ax.set_title(f"{r['tag']}\ncloud medial-axis", fontsize=8); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        # best slice ring 2D
        ax = fig.add_subplot(3, len(ok), len(ok) + j + 1)
        if best:
            ax.scatter(best["xy"][:, 0], best["xy"][:, 1], s=3, c="tab:blue", alpha=.5)
            ax.plot(best["rb"] * np.cos(best["ang"]), best["rb"] * np.sin(best["ang"]), "r-", lw=1.5)
            ax.plot(0, 0, "k+", ms=10); ax.set_aspect("equal")
            ax.set_title(f"best slice cov={best['cov']:.0%} ratio={best['ratio']:.2f}\nDCE={best['dce']:.2f} spread={best['spread']:.2f}", fontsize=8)
        else:
            ax.text(.5, .5, "no accepted slice", ha="center"); ax.axis("off")
        # CSA vs arclength (three estimators)
        ax = fig.add_subplot(3, len(ok), 2 * len(ok) + j + 1)
        ii = [s["i"] for s in e["slices"]]
        ax.plot(ii, [s["csa_polar"] for s in e["slices"]], ".-", ms=3, label="polar")
        ax.plot(ii, [s["csa_ell"] for s in e["slices"]], ".-", ms=3, label="ellipse")
        ax.plot(ii, [s["csa_hull"] for s in e["slices"]], ".-", ms=3, label="hull")
        for s in e["acc"]: ax.axvline(s["i"], color="g", alpha=.15)
        ax.set_title("CSA vs centerline position (green=accepted)", fontsize=8); ax.set_xlabel("spline idx"); ax.legend(fontsize=6)
    fig.suptitle("Centerline-slice CSA on DISTAL dense rings (cloud medial-axis; accept cov high & estimators <=1.3x)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "centerline_csa.png", dpi=115); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, extras = [], []
    for c in REGIONS:
        r, e = run(c); rows.append(r); extras.append(e)
        b = r.get("best")
        print(f"[{r['tag']}] " + (r.get("reason", "") if r["status"] != "OK" else
              (f"REJECT (0/{r['n_slices']} accepted)" if not b else
               f"DCE={b['DCE']} CSA={b['CSA']} cov={b['coverage']} ratio={b['r_std_over_r_med']} "
               f"spread={b['estimator_spread']} band={r['stability_band_DCE']} stab±{r['stability_pct']}%")), flush=True)
    figure(rows, extras)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n=== CENTERLINE-SLICE CSA (distal dense rings) ===")
    print(f"{'video/region':24}{'centerline':14}{'cov':>6}{'r_std/rmed':>11}{'CSA':>9}{'DCE':>8}{'spread':>8}{'stab%':>7}  verdict")
    for r in rows:
        if r["status"] != "OK" or not r.get("best"):
            print(f"{r['tag']:24}{'medial-axis':14}{'--':>6}{'--':>11}{'--':>9}{'--':>8}{'--':>8}{'--':>7}  {r.get('verdict', r.get('reason'))}")
            continue
        b = r["best"]
        print(f"{r['tag']:24}{'medial-axis':14}{b['coverage']:>6}{b['r_std_over_r_med']:>11}{b['CSA']:>9}{b['DCE']:>8}"
              f"{b['estimator_spread']:>8}{str(r['stability_pct']):>7}  {r['verdict']}")
    print(f"\nfigure -> {OUT}/centerline_csa.png")


if __name__ == "__main__":
    main()
