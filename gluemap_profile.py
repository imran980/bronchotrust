"""A/B lumen profile: COLMAP vs GlueMap dense clouds, BOTH sliced with the SAME
medial-axis centerline method. Side-by-side cross-section grid + profile plot +
per-node table (focus nodes 0-2 subglottis). NO scale. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
COV_MIN, RATIO_MAX = 0.60, 0.35
ARMS = {"COLMAP": (OD / "dense0/fused.ply", OD / "sparse/0"),
        "GlueMap": (OD / "gluemap/dense0/fused.ply", OD / "gluemap/gluemap_aba")}


def medial_profile(fused, model):
    P = np.asarray(o3d.io.read_point_cloud(str(fused)).points)
    P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P, axis=1) > 1e-4]          # drop failed-fusion origin points (MVS artifact)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(model)); C = G.cam_centers(rec); cam_centroid = C.mean(0)
    tree = cKDTree(P)
    _, _, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck_m, uf_m, pf_m = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    fracs = np.linspace(0.03, 0.97, 10)
    Pn, Tn, pfm = G.nodes_from(tck_m, uf_m, P, cam_centroid, fracs)
    scum = G.arclen(pfm); L = scum[-1]; rows = []
    for k, (p0, t) in enumerate(zip(Pn, Tn)):
        _, ip, cov, rm, rs = G.section(P, tree, p0, t, Rmed)
        ratio = rs / max(rm, 1e-6)
        rows.append({"node": k, "s_arc": round(float(fracs[k] * L), 2), "n": int(len(ip)),
                     "coverage": round(cov, 3), "r_med": round(rm, 3), "ratio": round(ratio, 3),
                     "closed": bool(cov >= COV_MIN and ratio <= RATIO_MAX), "_ip": ip})
    return rows, int(P.shape[0])


def grid(prof, path):
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    for ai, (arm, color, base) in enumerate([("COLMAP", "tab:green", 0), ("GlueMap", "tab:purple", 2)]):
        for k, r in enumerate(prof[arm]):
            ax = axes[base + k // 5, k % 5]; ip = r["_ip"]
            if len(ip): ax.scatter(ip[:, 0], ip[:, 1], s=1.5, c=color, alpha=.5)
            ax.plot(0, 0, "k+", ms=11, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_title(f"{arm} n{r['node']} (s={r['s_arc']})\ncov={r['coverage']*100:.0f}% "
                         f"r_std/r_med={r['ratio']:.2f}\n{'CLOSED' if r['closed'] else 'open/noisy'}",
                         fontsize=8.5, color=("green" if r["closed"] else "crimson"))
    fig.suptitle("32-V2 medial-axis cross-sections — COLMAP (rows 1-2) vs GlueMap (rows 3-4). "
                 "node0=subglottis entrance -> node9=deep trachea. CLOSED = cov>=60% & r_std/r_med<=0.35\n"
                 "(GlueMap intrinsics NOT pinned: focal 602->748.8)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=110); plt.close(fig)


def profileplot(prof, path):
    x = list(range(10))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].axhline(RATIO_MAX, color="0.5", ls="--", label=f"closed threshold {RATIO_MAX}")
    for arm, c in [("COLMAP", "tab:green"), ("GlueMap", "tab:purple")]:
        ax[0].plot(x, [r["ratio"] for r in prof[arm]], "o-", c=c, label=arm)
        ax[1].plot(x, [r["coverage"] * 100 for r in prof[arm]], "o-", c=c, label=arm)
    ax[0].axvspan(-0.5, 2.5, color="orange", alpha=.08); ax[0].text(1, ax[0].get_ylim()[1]*.95, "subglottis", ha="center", fontsize=8)
    ax[0].set_xlabel("node (0=subglottis -> 9=deep trachea)"); ax[0].set_ylabel("r_std/r_med (lower=cleaner)")
    ax[0].set_title("Ring noise per node (medial-axis)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axhline(COV_MIN*100, color="0.5", ls="--"); ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("node"); ax[1].set_ylabel("coverage %"); ax[1].set_title("Angular coverage"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("32-V2 A/B medial-axis profile: COLMAP vs GlueMap (no scale)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    prof = {}; dens = {}
    for arm, (fused, model) in ARMS.items():
        rows, npts = medial_profile(fused, model); prof[arm] = rows; dens[arm] = npts
        print(f"{arm}: dense(filtered)={npts}")
    grid(prof, DIAG / "ab_gluemap_grid.png")
    profileplot(prof, DIAG / "ab_gluemap_plot.png")
    out = {}
    for arm in ARMS:
        cl = sum(r["closed"] for r in prof[arm]); medr = float(np.median([r["ratio"] for r in prof[arm]]))
        sub = [prof[arm][i]["ratio"] for i in (0, 1, 2)]
        out[arm] = {"dense_filtered": dens[arm], "closed_rings": cl, "median_ratio": round(medr, 3),
                    "subglottis_0_2_ratio": sub,
                    "nodes": [{k: v for k, v in r.items() if k != "_ip"} for r in prof[arm]]}
    (DIAG / "ab_gluemap_compare.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'node':<5}{'region':<11}{'COLMAP cov/ratio/closed':>26}{'GlueMap cov/ratio/closed':>27}")
    for a, b in zip(prof["COLMAP"], prof["GlueMap"]):
        reg = "subglottis" if a["node"] <= 2 else ("trachea" if a["node"] >= 7 else "mid")
        pa = f"{a['coverage']*100:.0f}% / {a['ratio']:.2f} / {'Y' if a['closed'] else 'n'}"
        pb = f"{b['coverage']*100:.0f}% / {b['ratio']:.2f} / {'Y' if b['closed'] else 'n'}"
        print(f"{a['node']:<5}{reg:<11}{pa:>26}{pb:>27}")
    print("-" * 70)
    for arm in ARMS:
        print(f"{arm:<8} closed={out[arm]['closed_rings']}/10  median_ratio={out[arm]['median_ratio']}  "
              f"subglottis(0-2) ratios={[round(x,2) for x in out[arm]['subglottis_0_2_ratio']]}")
    print(f"\nfigures: {DIAG}/ab_gluemap_grid.png + ab_gluemap_plot.png")


if __name__ == "__main__":
    main()
