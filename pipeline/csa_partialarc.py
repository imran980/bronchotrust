"""CSA from a ONE-SIDED (partial-arc) airway cloud, validated then applied to 7-V1 smooth-run.

PART A (validation): synthetic stenosis tube r(z)=RN-(RN-RT)exp(-.5((z-zs)/sig)^2), RN=5 RT=3
(true CSA ratio 0.36). Sample surface points over an ARC of `span` degrees (mimic 7-V1's missing
wall), add noise. Run the SAME slice->fit pipeline. Check recovered CSA profile + stenosis ratio vs
GT as a function of arc coverage. Tells us how much arc we need and the missing-sector uncertainty.

PART B: apply identical pipeline to the 7-V1 smooth-run cleaned cloud. Report CSA-vs-arclength
(scene units), per-station arc coverage, circle-vs-ellipse agreement, and whether there is a real
local narrowing (stenosis) or just the reconstruction funnel. NO absolute scale -> scale-free.
"""
import sys, numpy as np, open3d as o3d
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def fit_circle(xy):
    """Geometric circle fit (robust, arc-safe). Returns (cx,cy,R,rms_resid_frac)."""
    x, y = xy[:, 0], xy[:, 1]
    c0 = xy.mean(0); r0 = np.median(np.hypot(x - c0[0], y - c0[1]))
    def res(p): return np.hypot(x - p[0], y - p[1]) - p[2]
    s = least_squares(res, [c0[0], c0[1], r0], loss="soft_l1", f_scale=r0 * 0.1)
    cx, cy, R = s.x; rr = np.abs(res(s.x))
    return cx, cy, abs(R), float(np.sqrt(np.mean(rr ** 2)) / (abs(R) + 1e-9))


def fit_ellipse_area(xy):
    """Direct ellipse fit (Fitzgibbon); return CSA=pi*a*b or nan."""
    x, y = xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()
    D = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
    S = D.T @ D
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    try:
        ev, evec = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return np.nan
    a = evec[:, np.argmax(np.real(ev))].real
    b, c, d, f, g, aa = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = 2 * (aa * f * f + c * d * d + g * b * b - 2 * b * d * f - aa * c * g)
    den1 = (b * b - aa * c) * ((c - aa) * np.sqrt(1 + 4 * b * b / ((aa - c) ** 2 + 1e-12)) - (c + aa))
    den2 = (b * b - aa * c) * ((aa - c) * np.sqrt(1 + 4 * b * b / ((aa - c) ** 2 + 1e-12)) - (c + aa))
    if den1 == 0 or den2 == 0: return np.nan
    ax1 = np.sqrt(abs(num / den1)); ax2 = np.sqrt(abs(num / den2))
    return float(np.pi * ax1 * ax2)


def centerline(P, nb=60):
    c0 = P.mean(0); Pc = P - c0; _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    ax = Vt[0]; t = Pc @ ax; lo, hi = np.percentile(t, [2, 98])
    edges = np.linspace(lo, hi, nb + 1); cl = []
    for i in range(nb):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() > 30: cl.append([0.5 * (edges[i] + edges[i + 1]), *P[m].mean(0)])
    cl = np.array(cl)
    for j in (1, 2, 3):                                    # smooth
        cl[:, j] = np.convolve(cl[:, j], np.ones(5) / 5, mode="same")
    return cl, ax, Vt


def profile(P, nb=60, slab=None):
    """Slice at centerline stations perpendicular to local axis; fit circle+ellipse to each arc."""
    cl, ax, Vt = centerline(P, nb); e1, e2 = Vt[1], Vt[2]
    Pc = P - P.mean(0); tall = Pc @ ax
    if slab is None: slab = 0.6 * np.median(np.diff(cl[:, 0]))
    rows = []
    for st in cl:
        tstar = st[0]; sel = np.abs(tall - tstar) < slab
        if sel.sum() < 40: continue
        Q = P[sel] - st[1:]                                # relative to centerline point
        xy = np.c_[Q @ e1, Q @ e2]
        az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
        cov = (np.histogram(az, bins=36, range=(-180, 180))[0] > 0).mean()
        cx, cy, R, rr = fit_circle(xy); csa_c = np.pi * R * R
        csa_e = fit_ellipse_area(xy)
        rows.append(dict(t=tstar, cov=cov, R=R, csa_circ=csa_c, csa_ell=csa_e, resid=rr, n=int(sel.sum())))
    return rows, cl


# ---------------- PART A: synthetic validation ----------------
def synth_tube(span_deg, RN=5.0, RT=3.0, zs=25.0, sig=4.0, L=50.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed); pts = []
    th0 = rng.uniform(0, 360)
    for z in np.linspace(2, L - 2, 400):
        r = RN - (RN - RT) * np.exp(-0.5 * ((z - zs) / sig) ** 2)
        th = np.radians(th0) + np.radians(np.linspace(0, span_deg, max(8, int(span_deg / 3))))
        x = r * np.cos(th) + rng.normal(0, noise, len(th)); y = r * np.sin(th) + rng.normal(0, noise, len(th))
        for xi, yi in zip(x, y): pts.append([xi, yi, z + rng.normal(0, noise)])
    return np.array(pts)


def validate():
    print("PART A — partial-arc CSA validation on synthetic stenosis tube (RN=5 RT=3, true ratio 0.36)")
    for span in (360, 300, 240, 180, 120):
        P = synth_tube(span)
        rows, _ = profile(P, nb=40)
        rows = [r for r in rows if r["cov"] > 0.0]
        if not rows: print(f"  span {span:3d}deg: no slices"); continue
        z = np.array([r["t"] for r in rows]); csa = np.array([r["csa_circ"] for r in rows])
        # map station-t back to z is monotonic; throat = min CSA, ref = max
        ratio = csa.min() / csa.max()
        # GT at the throat/ref radii
        print(f"  span {span:3d}deg: stations={len(rows)} | mean arc-cov={np.mean([r['cov'] for r in rows]):.2f} "
              f"| recovered ratio={ratio:.3f} (true 0.36) | R_throat={np.sqrt(csa.min()/np.pi):.2f} R_ref={np.sqrt(csa.max()/np.pi):.2f}")


# ---------------- PART B: 7-V1 smooth-run ----------------
def load_7v1():
    pcd = o3d.io.read_point_cloud("runs/own_data/recon_7v1_smooth/dense0/fused.ply")
    P = np.asarray(pcd.points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=np.median(d) * 0.03)
    return np.asarray(pcd.points)


def apply_7v1():
    print("\nPART B — 7-V1 smooth-run cleaned cloud (scene units, NO absolute scale)")
    P = load_7v1(); rows, cl = profile(P, nb=60)
    arc = np.arange(len(rows))
    good = [r for r in rows if r["cov"] >= 0.55 and r["resid"] < 0.15]
    print(f"  {len(rows)} stations, {len(good)} valid (arc-cov>=0.55 & circle resid<0.15)")
    if good:
        csa = np.array([r["csa_circ"] for r in good]); ce = np.array([r["csa_ell"] for r in good])
        agree = np.nanmedian(np.abs(ce - csa) / csa)
        print(f"  valid arc-cov range {min(r['cov'] for r in good):.2f}-{max(r['cov'] for r in good):.2f} | "
              f"circle-vs-ellipse median disagreement {agree:.0%}")
        print(f"  CSA(scene^2): min={csa.min():.3f} max={csa.max():.3f} scale-free ratio min/max={csa.min()/csa.max():.3f}")
        # is the profile monotonic (funnel) or does it have a real local minimum (stenosis)?
        z = np.array([r["t"] for r in good]); order = np.argsort(z); cs = csa[order]
        imin = cs.argmin()
        interior_min = 0 < imin < len(cs) - 1
        print(f"  profile: {'LOCAL minimum in interior (candidate narrowing)' if interior_min else 'monotonic -> looks like reconstruction FUNNEL, not a stenosis'}")
    # figure
    fig, ax = plt.subplots(figsize=(9, 4.5))
    zt = np.array([r["t"] for r in rows]); cc = np.array([r["csa_circ"] for r in rows])
    cov = np.array([r["cov"] for r in rows])
    sc = ax.scatter(zt, cc, c=cov, cmap="viridis", vmin=0.3, vmax=1.0, s=28)
    ax.plot(zt, cc, "-", color="0.6", lw=0.8, zorder=0)
    fig.colorbar(sc, label="arc coverage (frac of 360)"); ax.set_xlabel("along-axis station (scene units)")
    ax.set_ylabel("CSA (scene units^2)"); ax.set_title("7-V1 smooth-run: partial-arc CSA profile (scene units)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("runs/own_data/renders/7v1_csa_profile.png", dpi=120)
    print("  saved runs/own_data/renders/7v1_csa_profile.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"): validate()
    if which in ("b", "all"): apply_7v1()
