"""Recompute the FULL 32-V2 lumen profile (all 10 nodes) with the LOCAL
MEDIAL-AXIS centerline, vs the current GLOBAL-PCA centerline. Before/after:
closed-ring count, median r_std/r_med, per-node r_std/r_med, node locations.
Side-by-side cross-section grid + profile plot. NO SfM/MVS. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G          # reuse pca/medial centerline + section

OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2"); DIAG = OD / "diag"
COV_MIN, RATIO_MAX = 0.60, 0.35


def profile(Pn, Tn, P, tree, Rmed, scum, fracs):
    L = scum[-1]; rows = []
    for k, (p0, t) in enumerate(zip(Pn, Tn)):
        ins3, ip, cov, rm, rs = G.section(P, tree, p0, t, Rmed)
        ratio = rs / max(rm, 1e-6)
        rows.append({"node": k, "s_arc": round(float(fracs[k] * L), 2),
                     "pos": [round(float(x), 2) for x in p0], "n": int(len(ip)),
                     "coverage": round(cov, 3), "r_med": round(rm, 3), "r_std": round(rs, 3),
                     "ratio": round(ratio, 3), "closed": bool(cov >= COV_MIN and ratio <= RATIO_MAX),
                     "_ip": ip})
    return rows


def grid(rows_p, rows_m, path):
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    for src, base in [(rows_p, 0), (rows_m, 2)]:
        for k, r in enumerate(src):
            ax = axes[base + k // 5, k % 5]; ip = r["_ip"]
            c = "tab:blue" if base == 0 else "tab:orange"
            if len(ip): ax.scatter(ip[:, 0], ip[:, 1], s=1.5, c=c, alpha=.5)
            ax.plot(0, 0, "k+", ms=11, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
            tag = "PCA" if base == 0 else "MEDIAL"
            ax.set_title(f"{tag} n{r['node']} (s={r['s_arc']})\ncov={r['coverage']*100:.0f}% "
                         f"r_std/r_med={r['ratio']:.2f}\n{'CLOSED' if r['closed'] else 'open/noisy'}",
                         fontsize=8.5, color=("green" if r["closed"] else "crimson"))
    fig.suptitle("32-V2 full profile: GLOBAL-PCA centerline (rows 1-2) vs LOCAL MEDIAL-AXIS centerline (rows 3-4)\n"
                 "node0=subglottis entrance -> node9=deep trachea.  CLOSED = cov>=60% & r_std/r_med<=0.35", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=110); plt.close(fig)


def profileplot(rows_p, rows_m, path):
    x = [r["node"] for r in rows_p]
    rp = [r["ratio"] for r in rows_p]; rm = [r["ratio"] for r in rows_m]
    cp = [r["coverage"] * 100 for r in rows_p]; cm = [r["coverage"] * 100 for r in rows_m]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].axhline(RATIO_MAX, color="0.5", ls="--", label=f"closed threshold {RATIO_MAX}")
    ax[0].plot(x, rp, "o-", c="tab:blue", label="PCA")
    ax[0].plot(x, rm, "s-", c="tab:orange", label="medial-axis")
    for xi, a, b in zip(x, rp, rm):
        ax[0].annotate("", xy=(xi, b), xytext=(xi, a),
                       arrowprops=dict(arrowstyle="->", color=("green" if b < a else "red"), alpha=.5))
    ax[0].axvspan(-0.5, 2.5, color="orange", alpha=.08); ax[0].text(1, ax[0].get_ylim()[1]*.95, "subglottis", ha="center", fontsize=8)
    ax[0].set_xlabel("node (0=subglottis entrance -> 9=deep trachea)"); ax[0].set_ylabel("r_std / r_med  (lower=cleaner ring)")
    ax[0].set_title("Ring noise per node: PCA vs medial-axis"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, cp, "o-", c="tab:blue", label="PCA"); ax[1].plot(x, cm, "s-", c="tab:orange", label="medial-axis")
    ax[1].axhline(COV_MIN * 100, color="0.5", ls="--", label=f"min coverage {COV_MIN*100:.0f}%")
    ax[1].set_xlabel("node"); ax[1].set_ylabel("angular coverage %"); ax[1].set_title("Angular coverage per node")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 105)
    fig.suptitle("32-V2 centerline swap — full-profile before/after (calib 32_v1 BORROWED; no scale)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    P = np.asarray(o3d.io.read_point_cloud(str(OD / "dense0/fused.ply")).points)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); C = G.cam_centers(rec); cam_centroid = C.mean(0)
    tree = cKDTree(P)
    tck_p, uf_p, pf_p = G.pca_centerline(P, cam_centroid)
    Rmed = float(np.median(cKDTree(pf_p).query(P)[0]))
    tck_m, uf_m, pf_m = G.medial_centerline(P, pf_p, Rmed, cam_centroid)
    fracs = np.linspace(0.03, 0.97, 10)
    Pn_p, Tn_p, pfp = G.nodes_from(tck_p, uf_p, P, cam_centroid, fracs)
    Pn_m, Tn_m, pfm = G.nodes_from(tck_m, uf_m, P, cam_centroid, fracs)
    scum_p = G.arclen(pfp); scum_m = G.arclen(pfm)
    rows_p = profile(Pn_p, Tn_p, P, tree, Rmed, scum_p, fracs)
    rows_m = profile(Pn_m, Tn_m, P, tree, Rmed, scum_m, fracs)

    grid(rows_p, rows_m, DIAG / "profile_pca_vs_medial_grid.png")
    profileplot(rows_p, rows_m, DIAG / "profile_pca_vs_medial_plot.png")

    cl_p = sum(r["closed"] for r in rows_p); cl_m = sum(r["closed"] for r in rows_m)
    med_p = float(np.median([r["ratio"] for r in rows_p])); med_m = float(np.median([r["ratio"] for r in rows_m]))
    # threshold sensitivity
    sens = {f"<= {th}": [sum(r["ratio"] <= th and r["coverage"] >= COV_MIN for r in rows_p),
                         sum(r["ratio"] <= th and r["coverage"] >= COV_MIN for r in rows_m)] for th in (0.35, 0.5)}
    for r in rows_p + rows_m: r.pop("_ip", None)
    out = {"closed_pca": cl_p, "closed_medial": cl_m, "median_ratio_pca": round(med_p, 3),
           "median_ratio_medial": round(med_m, 3), "sensitivity_closed_pca_vs_medial": sens,
           "pca": rows_p, "medial": rows_m}
    (DIAG / "profile_medial_compare.json").write_text(json.dumps(out, indent=2))

    print("============ 32-V2 FULL PROFILE: PCA vs MEDIAL-AXIS centerline ============")
    print(f"{'node':<5}{'s_arc':>7}{'region':<12}{'PCA cov/ratio/closed':>26}{'MEDIAL cov/ratio/closed':>27}")
    for a, b in zip(rows_p, rows_m):
        reg = "subglottis" if a["node"] <= 2 else ("trachea" if a["node"] >= 7 else "mid")
        pc = f"{a['coverage']*100:.0f}% / {a['ratio']:.2f} / {'Y' if a['closed'] else 'n'}"
        mc = f"{b['coverage']*100:.0f}% / {b['ratio']:.2f} / {'Y' if b['closed'] else 'n'}"
        print(f"{a['node']:<5}{a['s_arc']:>7}{reg:<12}{pc:>26}{mc:>27}")
    print("-" * 75)
    print(f"CLOSED RINGS:    PCA = {cl_p}/10     MEDIAL = {cl_m}/10")
    print(f"MEDIAN r_std/r_med: PCA = {med_p:.3f}   MEDIAL = {med_m:.3f}")
    print(f"sensitivity (closed if cov>=60% & ratio<=th):")
    for k, (vp, vm) in sens.items(): print(f"   ratio {k}:  PCA={vp}/10   MEDIAL={vm}/10")
    print(f"\nfigures: {DIAG}/profile_pca_vs_medial_grid.png + profile_pca_vs_medial_plot.png")


if __name__ == "__main__":
    main()
