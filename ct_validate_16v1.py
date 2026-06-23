"""16_v1 vs 4D-CT lower-airway validation — final analysis + report.
Step 1 (anatomy) PASSED visually: 16_v1 reaches distal trachea -> carina (f1056-1160)
-> one mainstem (f1170-1340). But reconstruction is the wall: default strict pipeline
FRAGMENTS, lenient COLLAPSES (301/437 coincident cams). Strict mapper on the dense-
feature DB -> 8 disconnected fragments, NONE spanning trachea+mainstem. So there is no
connected model with a common scale, and with no blade there is no recoverable metric
scale -> a non-circular absolute DCE vs CT-envelope comparison is NOT achievable.

This script: (1) builds the fragmentation map; (2) measures the locally-reconstructable
distal-trachea ring (best fragment m0) in SCENE UNITS to show the lumen IS captured (limit
is scale/connectivity, not local capture); (3) writes report.json + the comparison table
(marked N/A with reasons) + an evidence figure. NO fabricated mm DCE. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/mi3dr/projects/bronchotrust"); OUT = ROOT / "runs/ct_validation_16v1"
SP = OUT / "recon_connect/sparse_strict"
CT = json.loads((OUT / "ct_reference.json").read_text())
ZONES = [("distal_trachea(<1056)", lambda f: f < 1056), ("carina(1056-1160)", lambda f: 1056 <= f <= 1160),
         ("mainstem(>1160)", lambda f: f > 1160)]


def model_info(d):
    rec = pycolmap.Reconstruction(str(d))
    rows = []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); C = -M[:3, :3].T @ M[:3, 3]
        ax = M[:3, :3].T @ np.array([0, 0, 1.0])
        rows.append((int(re.search(r'f(\d+)', im.name).group(1)), C, ax / (np.linalg.norm(ax) + 1e-9)))
    rows.sort(); F = np.array([r[0] for r in rows]); Cc = np.array([r[1] for r in rows]); A = np.array([r[2] for r in rows])
    med = np.median(Cc, 0); coincident = int((np.linalg.norm(Cc - med, axis=1) < 0.01).sum())
    cone = float(np.degrees(np.arccos(np.clip(A @ A.T, -1, 1))).max()) if len(A) > 1 else 0.0
    zc = {nm: int(sum(1 for f in F if fn(f))) for nm, fn in ZONES}
    return dict(name=d.name, reg=rec.num_reg_images(), sparse=rec.num_points3D(),
                fmin=int(F[0]), fmax=int(F[-1]), coincident=coincident, n=len(F), cone=round(cone, 1),
                zones=zc, F=F, C=Cc, rec=rec)


def sparse_xyz(rec):
    return np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))


def ring_at(rec, frame, halfwin=18):
    """Scene-unit cross-section near `frame`: centerline = spline through this model's
    cameras (ordered by frame); slab perpendicular to local tangent; polar coverage,
    r_med/r_std, CSA via angular-median polygon, DCE=2*sqrt(CSA/pi)."""
    rows = []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); rows.append((int(re.search(r'f(\d+)', im.name).group(1)), -M[:3, :3].T @ M[:3, 3]))
    rows.sort(); F = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    keep = np.abs(F - frame) <= halfwin
    if keep.sum() < 6: return None
    Cw = C[keep]
    # robust: drop flung cams
    med = np.median(Cw, 0); mad = np.median(np.abs(Cw - med), 0) + 1e-9
    Cw = Cw[np.all(np.abs(Cw - med) < 6 * mad * 1.4826, axis=1)]
    if len(Cw) < 6: return None
    # tangent = principal axis of the local camera segment
    u, s, vt = np.linalg.svd(Cw - Cw.mean(0), full_matrices=False); t = vt[0]; t /= np.linalg.norm(t)
    p0 = Cw.mean(0)
    P = sparse_xyz(rec)
    if len(P) < 30: return None
    medP = np.median(P, 0); madP = np.median(np.abs(P - medP), 0) + 1e-9
    P = P[np.all(np.abs(P - medP) < 8 * madP, axis=1)]
    Rmed0 = np.median(cKDTree(P).query(p0)[0])
    rel = P - p0
    inslab = P[np.abs(rel @ t) <= 0.6 * max(Rmed0, 1e-3)]
    if len(inslab) < 12: return None
    # local recenter (medial) once
    e1 = np.cross(t, [0, 0, 1.0]); e1 = e1 / (np.linalg.norm(e1) + 1e-9); e2 = np.cross(t, e1)
    c2 = np.stack([(inslab - p0) @ e1, (inslab - p0) @ e2], 1)
    p0 = p0 + c2.mean(0)[0] * e1 + c2.mean(0)[1] * e2
    ip = np.stack([(inslab - p0) @ e1, (inslab - p0) @ e2], 1)
    th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
    bins = np.linspace(-np.pi, np.pi, 37); cov = float((np.histogram(th, bins=bins)[0] > 0).mean())
    # angular-median polygon CSA
    idx = np.digitize(th, bins) - 1; rmed_bin = []
    ang = []
    for b in range(36):
        rb = r[idx == b]
        if len(rb): rmed_bin.append(np.median(rb)); ang.append((bins[b] + bins[b + 1]) / 2)
    csa = 0.0
    if len(rmed_bin) >= 8:
        rr = np.array(rmed_bin); aa = np.array(ang)
        o = np.argsort(aa); rr = rr[o]; aa = aa[o]
        csa = 0.5 * abs(np.sum(rr * np.roll(rr, -1) * np.sin(np.roll(aa, -1) - aa)))
    rmed = float(np.median(r)); rstd = float(np.std(r))
    dce = 2 * np.sqrt(csa / np.pi) if csa > 0 else 2 * rmed
    return dict(frame=int(frame), n_pts=int(len(inslab)), coverage=round(cov, 3),
                r_med_scene=round(rmed, 4), ratio=round(rstd / max(rmed, 1e-6), 3),
                CSA_scene=round(float(csa), 4), DCE_scene=round(float(dce), 4),
                closed=bool(cov >= 0.60 and rstd / max(rmed, 1e-6) <= 0.35), ip=ip)


def ring_dense(ply_path, rec, frame, halfwin=14):
    """Same cross-section as ring_at but on a DENSE MVS cloud (for the best fragment m0)."""
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points); P = P[np.isfinite(P).all(1)]
    if len(P) < 50: return None
    medP = np.median(P, 0); madP = np.median(np.abs(P - medP), 0) + 1e-9
    P = P[np.all(np.abs(P - medP) < 8 * madP, axis=1)]
    rows = []
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); rows.append((int(re.search(r'f(\d+)', im.name).group(1)), -M[:3, :3].T @ M[:3, 3]))
    rows.sort(); F = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    Cw = C[np.abs(F - frame) <= halfwin]
    if len(Cw) < 6: return None
    u, s, vt = np.linalg.svd(Cw - Cw.mean(0), full_matrices=False); t = vt[0] / np.linalg.norm(vt[0]); p0 = Cw.mean(0)
    Rmed0 = np.median(cKDTree(P).query(p0)[0]); rel = P - p0; sl = P[np.abs(rel @ t) <= 0.6 * max(Rmed0, 1e-3)]
    if len(sl) < 15: return None
    e1 = np.cross(t, [0, 0, 1.0]); e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(t, e1)
    ip = np.stack([(sl - p0) @ e1, (sl - p0) @ e2], 1); ip = ip - ip.mean(0)
    th = np.arctan2(ip[:, 1], ip[:, 0]); r = np.linalg.norm(ip, axis=1)
    cov = float((np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean())
    rmed = float(np.median(r)); rstd = float(np.std(r))
    return dict(frame=int(frame), n_pts=int(len(sl)), coverage=round(cov, 3), source="dense MVS (m0)",
                r_med_scene=round(rmed, 4), ratio=round(rstd / max(rmed, 1e-6), 3),
                DCE_scene=round(2 * rmed, 4), closed=bool(cov >= 0.60 and rstd / max(rmed, 1e-6) <= 0.35), ip=ip)


def main():
    models = sorted([d for d in SP.iterdir() if d.is_dir()],
                    key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images(), reverse=True)
    infos = [model_info(d) for d in models]
    through = [m for m in infos if m["zones"]["distal_trachea(<1056)"] >= 5 and m["zones"]["mainstem(>1160)"] >= 5]

    # distal-trachea ring from the densified fragment m0 = sparse_strict/0 (998-1088, cone 11.1deg), slice f1040
    trach = next(m for m in infos if m["name"] == "0")
    dense_m0 = OUT / "recon_connect/dense_m0/fused.ply"
    tr_ring = ring_dense(dense_m0, trach["rec"], frame=1040) if dense_m0.exists() else ring_at(trach["rec"], frame=1030)
    # best-parallax mainstem fragment
    ms_frags = [m for m in infos if m["zones"]["mainstem(>1160)"] >= 10]
    ms = max(ms_frags, key=lambda m: m["cone"]) if ms_frags else None
    ms_ring = ring_at(ms["rec"], frame=int((ms["fmin"] + ms["fmax"]) / 2)) if ms else None

    def seg_row(label, ct_key, ring, model):
        d = CT[ct_key]
        return {"segment": label, "recon_DCE_mm": "NOT RECOVERABLE (no metric scale)",
                "recon_DCE_scene_units": (ring["DCE_scene"] if ring else None),
                "ring_coverage": (ring["coverage"] if ring else None),
                "ring_r_std/r_med": (ring["ratio"] if ring else None),
                "ring_closed": (ring["closed"] if ring else None),
                "from_model": (model["name"] if model else None),
                "model_frames": (f"{model['fmin']}-{model['fmax']}" if model else None),
                "model_cone_deg": (model["cone"] if model else None),
                "CT_PEEP5": d["PEEP5"], "CT_PEEP7": d["PEEP7"], "CT_PEEP8": d["PEEP8"],
                "pass_fail": "N/A — cannot scale", }

    report = {
        "video": "16_v1", "ct_type": "4D dynamic chest/airway CT (lower airway)",
        "step1_reaches_lower_airway": True,
        "step1_evidence": "Carina (saddle spur + two mainstem orifices) clearly visible f1056-1160; "
                          "scope enters one mainstem f1170-1340. See zoom_carina_1000_1340.png.",
        "reconstruction_outcome": {
            "default_strict_batch_dense": "fragmented: best model only 66 frames f1014-1079 (carina approach)",
            "lenient_mapper": "COLLAPSED: 301/437 cameras coincident (degenerate, not real geometry)",
            "strict_mapper_on_dense_features": f"{len(infos)} disconnected fragments; NONE span trachea+mainstem",
            "through_path_connected_model": bool(through),
            "fragments": [{k: m[k] for k in ("name", "reg", "sparse", "fmin", "fmax", "cone", "coincident", "zones")} for m in infos],
        },
        "root_cause": "Slow dwelling scope -> near-zero frame-to-frame parallax (locked-rule-6) on the "
                      "lowest-texture cohort video. Strict gating fragments; lenient gating collapses. "
                      "Carina/deep-mainstem fragments have cones 3-5deg (insufficient parallax).",
        "metric_scale": "UNRECOVERABLE: no blade in view; no 16_v1 calibration (borrowed 2_v2 optics, "
                        "PROVISIONAL); trachea and mainstem are in separate models with unrelatable scales, "
                        "so even a scale-free caliber ratio is unavailable; per-segment CT-anchoring would be "
                        "circular/tautological. CT 3D model unavailable (only scalar DCE), so similarity "
                        "registration to CT geometry is impossible.",
        "verdict": "CT comparison CANNOT BE COMPLETED for 16_v1 — NOT because the CT region is unimaged "
                   "(it is imaged), but because the lower airway is not metrically reconstructable "
                   "(low-parallax fragmentation/collapse) and no metric scale is recoverable. "
                   "No valid reconstructed DCE in mm can be produced; reporting one would be the "
                   "'plausible number with no validity check' trap.",
        "what_is_locally_reconstructable_scene_units_only": {
            "distal_trachea": tr_ring and {k: tr_ring[k] for k in tr_ring if k != "ip"},
            "mainstem": ms_ring and {k: ms_ring[k] for k in ms_ring if k != "ip"},
        },
        "comparison_table": [
            seg_row("distal trachea", "trachea", tr_ring, trach),
            seg_row("right_or_left mainstem (unresolved)", "right_mainstem", ms_ring, ms),
        ],
        "ct_reference_used": {k: {kk: CT[k][kk] for kk in ("PEEP5", "PEEP7", "PEEP8", "overall_range")} for k in CT},
        "note_left_vs_right": "Mainstem L/R could not be resolved: caliber-based ID requires a metric DCE, "
                              "which is unrecoverable; the mainstem fragment is too small/low-parallax for a "
                              "reliable eccentricity read.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- evidence figure: fragmentation map + cones + the trachea ring ----
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    for i, m in enumerate(infos):
        col = "tab:red" if m["coincident"] > 0.3 * m["n"] else "tab:blue"
        ax1.barh(i, m["fmax"] - m["fmin"], left=m["fmin"], color=col, alpha=.8)
        ax1.text(m["fmin"], i, f" m{m['name']} reg{m['reg']} cone{m['cone']}°", va="center", fontsize=8)
    for fr, lab in [(1056, "carina start"), (1160, "mainstem")]:
        ax1.axvline(fr, color="k", ls="--", lw=1); ax1.text(fr, len(infos), lab, fontsize=8, rotation=90, va="top")
    ax1.set_xlabel("frame index"); ax1.set_yticks([]); ax1.set_title("16_v1 lower-airway: DISCONNECTED fragments (none span trachea→mainstem)")
    ax1.set_xlim(890, 1350)

    ax2 = fig.add_subplot(2, 2, 2)
    names = [f"m{m['name']}\n{m['fmin']}-{m['fmax']}" for m in infos]; cones = [m["cone"] for m in infos]
    ax2.bar(range(len(infos)), cones, color=["tab:green" if c >= 10 else "tab:orange" if c >= 5 else "tab:red" for c in cones])
    ax2.axhline(5, color="k", ls="--", lw=1); ax2.text(0, 5.2, "parallax floor ~5°", fontsize=8)
    ax2.set_xticks(range(len(infos))); ax2.set_xticklabels(names, fontsize=7); ax2.set_ylabel("viewing-cone (deg)")
    ax2.set_title("Per-fragment parallax (cone); carina/deep-mainstem are starved")

    ax3 = fig.add_subplot(2, 2, 3)
    if tr_ring:
        ax3.scatter(tr_ring["ip"][:, 0], tr_ring["ip"][:, 1], s=4, c="tab:blue", alpha=.6)
        ax3.plot(0, 0, "k+", ms=12, mew=2); ax3.set_aspect("equal"); ax3.grid(alpha=.3)
        ax3.set_title(f"distal-trachea ring (SCENE UNITS, m{trach['name']} f{tr_ring['frame']})\n"
                      f"cov={tr_ring['coverage']*100:.0f}% r_std/r_med={tr_ring['ratio']:.2f} "
                      f"DCE_scene={tr_ring['DCE_scene']:.3f}  {'closed' if tr_ring['closed'] else 'open'}", fontsize=9)
    else:
        ax3.text(.5, .5, "distal-trachea ring not measurable", ha="center"); ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4); ax4.axis("off")
    ax4.text(0.0, 1.0, "VERDICT", fontsize=13, fontweight="bold", va="top")
    ax4.text(0.0, 0.90,
             "• 16_v1 DOES reach distal trachea / carina / one mainstem (Step 1 PASS).\n"
             "• But the lower airway is NOT metrically reconstructable:\n"
             "   – default strict → fragments (best 66 frames)\n"
             "   – lenient → COLLAPSE (301/437 coincident cameras)\n"
             "   – strict on dense features → 8 disconnected fragments\n"
             "• Root cause: slow dwelling scope → near-zero parallax\n"
             "   (carina/deep-mainstem cones 3–5°), lowest-texture clip.\n"
             "• No connected trachea↔mainstem model → NO common scale.\n"
             "• No blade, no 16_v1 calib → NO recoverable metric scale.\n"
             "• ⇒ Absolute DCE-vs-CT envelope CANNOT be computed validly.\n"
             "   Reporting mm here would be a number with no validity check.\n"
             "• CT is NOT the limiter (region is imaged); reconstruction is.",
             fontsize=9.5, va="top", family="monospace")
    fig.suptitle("16_v1 vs 4D-CT lower airway — reconstruction not metrically viable (evidence)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "ct_validation_evidence.png", dpi=130); plt.close(fig)

    print("through-path connected model:", bool(through))
    print("distal-trachea ring (scene units):", {k: tr_ring[k] for k in ("coverage", "ratio", "DCE_scene", "closed")} if tr_ring else None)
    print("mainstem ring (scene units):", {k: ms_ring[k] for k in ("coverage", "ratio", "DCE_scene", "closed")} if ms_ring else None)
    print("VERDICT: CT comparison cannot be completed — lower airway not metrically reconstructable (no recoverable scale).")
    print("wrote report.json + ct_validation_evidence.png")


if __name__ == "__main__":
    main()
