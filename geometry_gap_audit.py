"""GEOMETRY-GAP AUDIT: is each real subglottic region truly parallax-limited (CAPTURE gap), or
does usable triangulation geometry exist and we're missing the reconstruction/measurement recipe
(RECIPE / SEGMENTATION gap)? The earlier 'near-zero parallax' call used only the optical-axis CONE
angle -- but parallel-axis lateral translation gives a ~0 cone yet real triangulation angle at a
near target. So here we compute the TRUE target-aware triangulation angle, track support, and ring
quality, and only call 'capture-limited' if triangulation AND track support are actually below
usable thresholds. depth-eval env. -> runs/barbour30/airwayfit/geomgap/.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit/geomgap"
REGIONS = [
    dict(tag="2_V2 prox-subglottis",  model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  rng=(875, 945)),
    dict(tag="2_V2 dist-subglottis",  model="runs/batch4/2-V2/sparse/0",  cloud="runs/batch4/2-V2/dense0/fused.ply",  rng=(985, 1045)),
    dict(tag="25_V1 prox-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", rng=(335, 388)),
    dict(tag="25_V1 dist-subglottis", model="runs/batch4/25-V1/sparse/0", cloud="runs/batch4/25-V1/dense0/fused.ply", rng=(405, 452)),
    dict(tag="32_V2 prox-subglottis", model="runs/fuse_32v2/sparse/0",    cloud="runs/fuse_32v2/dense0/fused.ply",    rng=(300, 328)),
    dict(tag="32_V2 dist-subglottis", model="runs/fuse_32v2/sparse/0",    cloud="runs/fuse_32v2/dense0/fused.ply",    rng=(330, 358)),
]
# usable thresholds (defensible; do not call impossible above these)
TRI_POOR, TRI_USABLE = 1.5, 3.0          # deg, median target triangulation angle
PTS_MIN, VIEWS_MIN = 150, 3.0            # track support
RING_TIGHT, COV_OK = 0.35, 0.60         # ring quality


def load(model):
    rec = pycolmap.Reconstruction(str(ROOT / model))
    fid = {int(re.search(r"f(\d+)", im.name).group(1)): im.image_id for im in rec.images.values()}
    centers = {im.image_id: np.array(im.projection_center()) for im in rec.images.values()}
    axes = {im.image_id: np.array(im.cam_from_world().matrix())[2, :3] for im in rec.images.values()}
    return rec, fid, centers, axes


def tri_angles(rec, region_ids, centers):
    """max pairwise triangulation angle (deg) at each 3D point over the REGION cameras that see it,
    plus track length (all views) and region-view count. Returns arrays + the point xyz."""
    ang, tlen, rviews, XYZ = [], [], [], []
    rid = set(region_ids)
    for p in rec.points3D.values():
        obs = [e.image_id for e in p.track.elements]
        rc = [centers[i] for i in obs if i in rid]
        if len(rc) < 2: continue
        X = np.array(p.xyz); V = np.array(rc) - X
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        D = np.clip(V @ V.T, -1, 1); a = np.degrees(np.arccos(D))
        ang.append(a.max()); tlen.append(len(obs)); rviews.append(len(rc)); XYZ.append(X)
    return np.array(ang), np.array(tlen), np.array(rviews), np.array(XYZ) if XYZ else np.zeros((0, 3))


def clean_cloud(path):
    p = ROOT / path
    if not p.exists(): return np.zeros((0, 3))
    P = np.asarray(o3d.io.read_point_cloud(str(p)).points); P = P[np.isfinite(P).all(1)]
    if len(P) < 50: return P
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    return P[np.all(np.abs(P - med) < 8 * mad, axis=1)]


def ring_quality(P, ctr, axis, r_guess):
    """slice dense cloud at target plane perp to axis; ring tightness, coverage, shell thickness."""
    if len(P) < 50: return None
    d = (P - ctr) @ axis; slab = P[np.abs(d) <= 0.5 * r_guess]
    if len(slab) < 30: return dict(n=len(slab), r_med=None, tightness=None, coverage=0.0, shell=None)
    e1 = np.cross(axis, [0, 0, 1.]);
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(axis, [1., 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    x, y = (slab - ctr) @ e1, (slab - ctr) @ e2; r = np.hypot(x, y); th = np.arctan2(y, x)
    r_med = float(np.median(r)); cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
    return dict(n=int(len(slab)), r_med=r_med, tightness=float(np.std(r) / max(r_med, 1e-9)),
                coverage=cov, shell=float(np.std(r)))


def audit(cfg):
    rec, fid, centers, axes = load(cfg["model"])
    rframes = sorted(f for f in range(cfg["rng"][0], cfg["rng"][1] + 1) if f in fid)
    if len(rframes) < 4: return dict(tag=cfg["tag"], status="FAIL", reason=f"only {len(rframes)} frames registered")
    rids = [fid[f] for f in rframes]; C = np.array([centers[i] for i in rids]); A = np.array([axes[i] for i in rids])
    # --- trajectory ---
    ctr0 = C.mean(0); _, _, Vt = np.linalg.svd(C - ctr0, full_matrices=False); axis = Vt[0]
    if (C[-1] - C[0]) @ axis < 0: axis = -axis
    axial = (C - ctr0) @ axis; perp = (C - ctr0) - np.outer(axial, axis)
    axial_span = float(np.ptp(axial)); lateral_span = float(np.ptp(perp @ Vt[1]))
    baseline = float(np.max(np.linalg.norm(C[:, None] - C[None], axis=2)))
    path = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))); net = float(np.linalg.norm(C[-1] - C[0]))
    tortuosity = float(path / (net + 1e-9))
    cone = float(np.max([np.degrees(np.arccos(np.clip(A[i] @ A[j], -1, 1))) for i in range(len(A)) for j in range(i + 1, len(A))]))
    # --- target-aware triangulation ---
    ang, tlen, rviews, XYZ = tri_angles(rec, rids, centers)
    if len(ang) < 10: return dict(tag=cfg["tag"], status="FAIL", reason=f"only {len(ang)} co-observed target points")
    tgt_ctr = np.median(XYZ, 0); depth = float(np.median(np.linalg.norm(XYZ - ctr0, axis=1)))
    # restrict to points AT the target ring wall (thin axial slab + on the wall annulus)
    rel = XYZ - tgt_ctr; ax_c = rel @ axis; rad_v = rel - np.outer(ax_c, axis); rad = np.linalg.norm(rad_v, axis=1)
    r_guess = float(np.median(rad))
    near = (np.abs(ax_c) <= 0.6 * r_guess) & (rad >= 0.5 * r_guess) & (rad <= 1.6 * r_guess)
    ang_ring = ang[near] if near.sum() >= 10 else ang
    tri = dict(median=float(np.median(ang_ring)), p10=float(np.percentile(ang_ring, 10)),
               p90=float(np.percentile(ang_ring, 90)), n_ring_pts=int(near.sum()),
               median_allco=float(np.median(ang)))
    # per-point local baseline/depth is captured by the tri angle; this traj-level ratio is an UPPER bound
    bd_ratio = float(baseline / (depth + 1e-9))
    track = dict(n_pts=int(len(ang)), med_track_len=float(np.median(tlen)), med_region_views=float(np.median(rviews)))
    # --- ring quality (dense) ---
    P = clean_cloud(cfg["cloud"]); r_guess = float(np.median(np.linalg.norm((XYZ - tgt_ctr) - ((XYZ - tgt_ctr) @ axis)[:, None] * axis, axis=1)))
    ring = ring_quality(P, tgt_ctr, axis, max(r_guess, 1e-3)) if len(P) else None
    # --- verdict ---
    usable_tri = tri["median"] >= TRI_USABLE; marginal_tri = TRI_POOR <= tri["median"] < TRI_USABLE
    usable_track = track["n_pts"] >= PTS_MIN and track["med_region_views"] >= 2.0 and track["med_track_len"] >= VIEWS_MIN
    ring_ok = ring and ring["tightness"] is not None and ring["tightness"] <= RING_TIGHT and ring["coverage"] >= COV_OK
    if tri["median"] < TRI_POOR or track["n_pts"] < PTS_MIN or track["med_region_views"] < 2.0:
        verdict = "CAPTURE gap (insufficient triangulation/track support)"
    elif not usable_tri:
        verdict = "MARGINAL capture (triangulation borderline 1.5-3 deg)"
    elif ring_ok:
        verdict = "SEGMENTATION/MEASUREMENT gap (geometry + surface exist; slicing/measure is the limit)"
    else:
        verdict = "RECIPE gap (usable triangulation+tracks, but dense ring surface is poor)"
    return dict(tag=cfg["tag"], status="OK", n_frames=len(rframes),
                trajectory=dict(axial_span=round(axial_span, 3), lateral_span=round(lateral_span, 3),
                                baseline=round(baseline, 3), path_len=round(path, 3), net_disp=round(net, 3),
                                tortuosity_dwell=round(tortuosity, 2)),
                optical_axis_cone_deg=round(cone, 2),
                triangulation_deg={k: round(v, 2) for k, v in tri.items()},
                baseline_to_depth=round(bd_ratio, 4), target_depth=round(depth, 3),
                track_support={k: round(v, 1) if isinstance(v, float) else v for k, v in track.items()},
                ring_quality=(None if ring is None else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ring.items()}),
                verdict=verdict,
                _viz=dict(C=C, axis=axis, tgt_ctr=tgt_ctr, XYZ=XYZ, ang=ang, r_guess=r_guess, e_perp=Vt[1]))


def figure(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r["status"] == "OK"]
    n = len(ok); fig = plt.figure(figsize=(4.6 * n, 9))
    for j, r in enumerate(ok):
        v = r["_viz"]; C, axis, tgt, XYZ, ang = v["C"], v["axis"], v["tgt_ctr"], v["XYZ"], v["ang"]
        ax = fig.add_subplot(2, n, j + 1, projection="3d")
        ax.plot(C[:, 0], C[:, 1], C[:, 2], "c-o", ms=2, lw=1, label="cameras")
        sub = XYZ[np.random.default_rng(0).choice(len(XYZ), min(600, len(XYZ)), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=2, c="0.6", alpha=.4)
        # target ring
        e1 = np.cross(axis, [0, 0, 1.]); e1 = e1 / (np.linalg.norm(e1) + 1e-9); e2 = np.cross(axis, e1)
        th = np.linspace(0, 2 * np.pi, 80); ring = tgt + v["r_guess"] * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], "r-", lw=2, label="target ring")
        # sample triangulation rays from extreme cameras to target center
        for k in (0, len(C) // 2, len(C) - 1):
            ax.plot([C[k, 0], tgt[0]], [C[k, 1], tgt[1]], [C[k, 2], tgt[2]], "orange", lw=0.7, alpha=.7)
        ax.set_title(f"{r['tag']}\ncone {r['optical_axis_cone_deg']}° | tri med {r['triangulation_deg']['median']}°", fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax = fig.add_subplot(2, n, n + j + 1)
        ax.hist(ang, bins=40, color="steelblue"); ax.axvline(TRI_USABLE, c="g", ls="--", label=f"usable {TRI_USABLE}°")
        ax.axvline(TRI_POOR, c="r", ls="--", label=f"poor {TRI_POOR}°"); ax.axvline(np.median(ang), c="k", label=f"median {np.median(ang):.1f}°")
        ax.set_title(f"target triangulation angle\np10/50/90={r['triangulation_deg']['p10']}/{r['triangulation_deg']['median']}/{r['triangulation_deg']['p90']}°  b/d={r['baseline_to_depth']}", fontsize=7.5)
        ax.set_xlabel("deg"); ax.legend(fontsize=6)
    fig.suptitle("Geometry-gap audit — optical-axis CONE vs TRUE target triangulation angle", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "geomgap_summary.png", dpi=115); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit(c) for c in REGIONS]
    figure(rows)
    clean = [{k: val for k, val in r.items() if k != "_viz"} for r in rows]
    (OUT / "geomgap_report.json").write_text(json.dumps(clean, indent=2, default=str))
    print("=== GEOMETRY-GAP AUDIT ===")
    print(f"{'region':24}{'cone°':>7}{'tri_med°':>9}{'tri_p90°':>9}{'b/d':>7}{'n_pts':>7}{'views':>7}{'ring_tt':>8}{'cov':>6}  verdict")
    for r in rows:
        if r["status"] != "OK": print(f"{r['tag']:24}  {r['status']}: {r['reason']}"); continue
        t = r["triangulation_deg"]; k = r["track_support"]; rq = r["ring_quality"] or {}
        print(f"{r['tag']:24}{r['optical_axis_cone_deg']:>7}{t['median']:>9}{t['p90']:>9}{r['baseline_to_depth']:>7}"
              f"{k['n_pts']:>7}{k['med_region_views']:>7}{str(rq.get('tightness')):>8}{str(rq.get('coverage')):>6}  {r['verdict']}")
    print(f"\nfigure -> {OUT}/geomgap_summary.png ; report -> {OUT}/geomgap_report.json")


if __name__ == "__main__":
    main()
