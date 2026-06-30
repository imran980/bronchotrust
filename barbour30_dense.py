"""Barbour 30-frame audit — DENSE + landmark CSA/DCE measurement (stable COLMAP).
For each viable batch: MVS (same flags as baseline) -> measure the landmark cross-section
at the centerline point nearest the landmark CENTER frame, THREE ways:
  CSA_polar   : angular-median-radius polygon (robust to gaps/outliers)
  CSA_ellipse : direct ellipse fit (handles eccentric lumen)
  CSA_mesh    : double-Poisson surface, vertices in the slab -> polygon
DCE = 2*sqrt(CSA/pi). Reports ring coverage, r_std/r_med, estimator spread.
Figures per batch: A_organ_cloud, B_centerline(+slice plane), slice(3-estimator),
reproject-vs-real. Cleans stereo after each. depth-eval env (colmap-cuda binary)."""
from __future__ import annotations
import argparse, json, shutil, subprocess, re
from pathlib import Path
import numpy as np, cv2, pycolmap, open3d as o3d
if not hasattr(np, "in1d"): np.in1d = np.isin
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G
from batch_render import double_poisson, section as br_section

ROOT = Path("/home/mi3dr/projects/bronchotrust"); OUT = ROOT / "runs/barbour30"
DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
VIDFILE = {"2-V2": "2-V2.MP4", "25-V1": "25-V1.MP4", "32-V2": "32-V2.MP4"}


def run(cmd):
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0: print("  ERR", cmd[0], r.stderr[-200:], flush=True)
    return r


def ply_n(p):
    with open(p, "rb") as f:
        for _ in range(40):
            ln = f.readline().decode("latin1", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
    return 0


def run_mvs(bdir):
    fused = bdir / "dense0/fused.ply"
    if fused.exists() and ply_n(fused) > 0: return fused
    dense = bdir / "dense0"; shutil.rmtree(dense, ignore_errors=True)
    run(["image_undistorter", "--image_path", str(bdir / "images"), "--input_path", str(bdir / "sparse/0"),
         "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run(["patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"])
    run(["stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
    return fused


def cam_by_frame(rec):
    out = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix())
        out[int(re.search(r"f(\d+)", im.name).group(1))] = (-M[:3, :3].T @ M[:3, 3], im)
    return out


def csa_polar(ip, nb=36):
    th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
    edges = np.linspace(-np.pi, np.pi, nb + 1); idx = np.digitize(th, edges) - 1
    ang, rr = [], []
    for b in range(nb):
        rb = r[idx == b]
        if len(rb): ang.append((edges[b] + edges[b + 1]) / 2); rr.append(np.median(rb))
    cov = len(rr) / nb
    if len(rr) < 8: return 0.0, cov
    ang = np.array(ang); rr = np.array(rr); o = np.argsort(ang); ang, rr = ang[o], rr[o]
    a = 0.5 * abs(np.sum(rr * np.roll(rr, -1) * np.sin(np.roll(ang, -1) - ang)))
    return float(a), cov


def csa_ellipse(ip, r_med):
    keep = (np.linalg.norm(ip, axis=1) > 0.3 * r_med) & (np.linalg.norm(ip, axis=1) < 2.0 * r_med)
    pts = ip[keep].astype(np.float32)
    if len(pts) < 8: return 0.0
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts)
    return float(np.pi * (MA / 2) * (ma / 2))


def csa_hull(ip):
    if len(ip) < 5: return 0.0
    try: return float(ConvexHull(ip).volume)
    except Exception: return 0.0


def section_at(P, tree, p0, t, Rmed, slab=0.5, ball=6):
    e1, e2 = G.frame(t)
    rel = P[tree.query_ball_point(p0, ball * Rmed)] - p0
    ins = rel[np.abs(rel @ t) <= slab * Rmed]
    if len(ins) < 10: return None
    ip = np.stack([ins @ e1, ins @ e2], 1)
    return ip


def measure(name, center, do_mesh=True, do_reproj=True):
    bdir = OUT / "batches" / name
    rec = pycolmap.Reconstruction(str(bdir / "sparse/0"))
    # dense cloud
    pc = o3d.io.read_point_cloud(str(bdir / "dense0/fused.ply"))
    P = np.asarray(pc.points); cols = np.asarray(pc.colors); fin = np.isfinite(P).all(1)
    P, cols = P[fin], (cols[fin] if len(cols) else np.zeros((fin.sum(), 3)))
    keep = np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4; P, cols = P[keep], cols[keep]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    inl = np.all(np.abs(P - med) < 8 * mad, axis=1); P, cols = P[inl], cols[inl]
    # centerline from the CLOUD axis (forward-scope: cloud is the tube AHEAD of the cameras), medial-refined
    Cc = G.cam_centers(rec); cam_centroid = Cc.mean(0)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    scum = G.arclen(pf); L = scum[-1]; tree = cKDTree(P)
    # scan mid candidates, slice perpendicular, pick the cleanest (avoid ends/bends)
    best = None
    for frac in (0.30, 0.40, 0.50, 0.60, 0.70):
        u = np.interp(frac * L, scum, uf); p0 = np.array(splev(u, tck)).T
        t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
        if (p0 - cam_centroid) @ t < 0: t = -t
        ipc, cov, rm, rs, e1, e2 = br_section(P, tree, p0, t, Rmed)
        if len(ipc) < 12: continue
        score = cov - rs / max(rm, 1e-6)
        if best is None or score > best["score"]:
            best = {"frac": frac, "p0": p0, "t": t, "ip": ipc, "cov": cov, "rm": rm, "rs": rs, "score": score}
    res = {"name": name, "center_frame": center, "n_dense": int(len(P)), "L_scene": round(float(L), 3),
           "Rmed_scene": round(Rmed, 4)}
    if best is None:
        res.update({"measurable": False}); (bdir / "measure.json").write_text(json.dumps(res, indent=2)); return res
    p0, t, ip = best["p0"], best["t"], best["ip"]
    res["slice_frac"] = best["frac"]
    r = np.linalg.norm(ip, axis=1); r_med = float(best["rm"]); r_std = float(best["rs"])
    A_polar, cov = csa_polar(ip); A_ell = csa_ellipse(ip, r_med); A_hull = csa_hull(ip)
    dce = lambda A: round(2 * np.sqrt(A / np.pi), 4) if A > 0 else None
    # mesh slab CSA
    A_mesh = 0.0; ip_mesh = None
    if do_mesh:
        try:
            Vm = double_poisson(P, bdir)
            e1, e2 = G.frame(t); rel = Vm - p0; ins = rel[np.abs(rel @ t) <= 0.5 * Rmed]
            if len(ins) > 12:
                ip_mesh = np.stack([ins @ e1, ins @ e2], 1); A_mesh, _ = csa_polar(ip_mesh)
        except Exception as e:
            print("  mesh fail", name, str(e)[:80])
    csas = {"polar": round(A_polar, 4), "ellipse": round(A_ell, 4), "hull": round(A_hull, 4), "mesh": round(A_mesh, 4)}
    dces = {kk: dce(vv) for kk, vv in csas.items()}
    good = [v for v in (dces["polar"], dces["ellipse"], dces["mesh"]) if v]
    spread = round(max(good) / min(good), 2) if len(good) >= 2 and min(good) > 0 else None
    res.update({"measurable": True, "coverage": round(cov, 3), "r_med_scene": round(r_med, 4),
                "r_std/r_med": round(r_std / max(r_med, 1e-6), 3),
                "closed": bool(cov >= 0.60 and r_std / max(r_med, 1e-6) <= 0.35),
                "CSA_scene": csas, "DCE_scene": dces, "DCE_estimator_spread": spread})
    (bdir / "measure.json").write_text(json.dumps(res, indent=2))
    # ---- figures ----
    make_figs(bdir, name, P, cols, Cc, pf, p0, t, ip, ip_mesh, res, rec, center)
    return res


def make_figs(bdir, name, P, cols, Cs, curve, p0, t, ip, ip_mesh, res, rec, center):
    e1, e2 = G.frame(t)
    # A: organ cloud (3 views) colored by axial position
    s_axis = (P - p0) @ t
    fig = plt.figure(figsize=(13, 5), facecolor="black")
    for i, (el, az) in enumerate([(12, -60), (88, -90), (4, 0)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        idx = np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)
        ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], c=s_axis[idx], cmap="turbo", s=1.0, alpha=.5, linewidths=0)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c="cyan", lw=1.5)
        ax.set_axis_off(); ax.set_facecolor("black")
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.view_init(el, az)
    fig.suptitle(f"{name}  A: local cloud + centerline  (n={res['n_dense']})", color="white", fontsize=11)
    fig.savefig(bdir / "A_organ_cloud.png", dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    # B: centerline + slice plane
    fig = plt.figure(figsize=(7, 6), facecolor="black"); ax = fig.add_subplot(111, projection="3d")
    idx = np.random.default_rng(0).choice(len(P), min(len(P), 50000), replace=False)
    ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], c="0.5", s=0.8, alpha=.15, linewidths=0)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c="cyan", lw=2)
    hh = 1.6 * res["r_med_scene"]
    quad = np.array([p0 + a * hh * e1 + b * hh * e2 for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([quad], alpha=.5, facecolor="#39ff14", edgecolor="#39ff14"))
    ax.set_axis_off(); ax.set_facecolor("black")
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass
    ax.view_init(14, -60)
    ax.set_title(f"{name}  B: centerline + landmark slice plane", color="white", fontsize=10)
    fig.savefig(bdir / "B_centerline.png", dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    # slice / landmark with 3 estimators
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ip[:, 0], ip[:, 1], s=4, c="tab:blue", alpha=.5, label="dense cloud")
    if ip_mesh is not None and len(ip_mesh): ax.scatter(ip_mesh[:, 0], ip_mesh[:, 1], s=3, c="tab:orange", alpha=.3, label="mesh")
    r_med = res["r_med_scene"]
    try:
        pts = ip[(np.linalg.norm(ip, axis=1) > .3 * r_med) & (np.linalg.norm(ip, axis=1) < 2 * r_med)].astype(np.float32)
        if len(pts) >= 8:
            el = cv2.fitEllipse(pts); th = np.linspace(0, 2 * np.pi, 80)
            (cx, cy), (MA, ma), a0 = el; a0 = np.deg2rad(a0)
            ex = cx + (MA / 2) * np.cos(th) * np.cos(a0) - (ma / 2) * np.sin(th) * np.sin(a0)
            ey = cy + (MA / 2) * np.cos(th) * np.sin(a0) + (ma / 2) * np.sin(th) * np.cos(a0)
            ax.plot(ex, ey, "r-", lw=2, label="ellipse fit")
    except Exception: pass
    ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    d = res["DCE_scene"]
    ax.set_title(f"{name}\ncov={res['coverage']*100:.0f}% r_std/r_med={res['r_std/r_med']:.2f} "
                 f"{'CLOSED' if res['closed'] else 'open'}\nDCE_scene polar={d['polar']} ell={d['ellipse']} "
                 f"mesh={d['mesh']}  spread={res['DCE_estimator_spread']}", fontsize=9,
                 color=("green" if res["closed"] else "crimson"))
    fig.tight_layout(); fig.savefig(bdir / "landmarks.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="comma-separated batch names, or 'viable'")
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args()
    screen = {r["name"]: r for r in json.loads((OUT / "sparse_screen.json").read_text())}
    if a.names == "viable":
        names = [n for n, r in screen.items() if r.get("viable")]
    else:
        names = a.names.split(",")
    print(f"densifying+measuring {len(names)} batches", flush=True)
    rows = []
    for i, nm in enumerate(names):
        bdir = OUT / "batches" / nm
        if not (bdir / "sparse/0").exists(): print(f"[{i+1}] {nm} no sparse, skip"); continue
        center = screen[nm]["center"]
        run_mvs(bdir)
        try:
            r = measure(nm, center, do_mesh=not a.no_mesh)
        except Exception as e:
            print(f"[{i+1}] {nm} MEASURE FAIL {str(e)[:120]}"); r = {"name": nm, "measurable": False, "error": str(e)[:200]}
        rows.append(r)
        m = "measurable" if r.get("measurable") else "NOT measurable"
        d = r.get("DCE_scene", {})
        print(f"[{i+1}/{len(names)}] {nm:40} {m} cov={r.get('coverage','-')} ratio={r.get('r_std/r_med','-')} "
              f"DCE(polar/ell/mesh)={d.get('polar','-')}/{d.get('ellipse','-')}/{d.get('mesh','-')} spread={r.get('DCE_estimator_spread','-')}", flush=True)
        shutil.rmtree(bdir / "dense0/stereo", ignore_errors=True)
        shutil.rmtree(bdir / "dense0/images", ignore_errors=True)
    (OUT / "dense_measure.json").write_text(json.dumps(rows, indent=2))
    print(f"\nDENSE+MEASURE done -> {OUT}/dense_measure.json")


if __name__ == "__main__":
    main()
