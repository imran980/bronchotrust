"""Stability-band % obstruction (2-V2, 25-V1) from the global model. Scan the centerline
finely, measure every ring (coverage, r_std/r_med, DCE, CSA), keep only VALID slices
(cov>=0.90 & ratio<=0.35), and near the narrowest point pick the best-covered stenosis
slice. Report a BAND (min-median-max % over nearby valid stenosis x valid trachea slices)
instead of one number. Eyeball figure per video: DCE/CSA profile + best stenosis ring +
reference ring. Scene units only, NO mm, within-video only. depth-eval env."""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/mi3dr/projects/bronchotrust")
import json
import numpy as np, pycolmap, open3d as o3d, re
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geometry_debug_32v2 as G
from batch_render import section as br_section

OUT = "/home/mi3dr/projects/bronchotrust/runs/barbour30"
CFG = {"2-V2":  dict(Rdir="runs/batch4/2-V2",  sten_zone=(1003, 1036), ref_zone=(1055, 1140), label="2-V2"),
       "25-V1": dict(Rdir="runs/batch4/25-V1", sten_zone=(348, 462),   ref_zone=(476, 596),  label="25-V1")}
COVOK, RATIOOK = 0.90, 0.35


def build(Rdir):
    P = np.asarray(o3d.io.read_point_cloud(f"{Rdir}/dense0/fused.ply").points); P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(f"{Rdir}/sparse/0"); cbf = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); cbf[int(re.search(r"f(\d+)", im.name).group(1))] = -M[:3, :3].T @ M[:3, 3]
    cc = G.cam_centers(rec).mean(0)
    _, _, pfp = G.pca_centerline(P, cc); Rmed = float(np.median(cKDTree(pfp).query(P)[0]))
    tck, uf, pf = G.medial_centerline(P, pfp, Rmed, cc); scum = G.arclen(pf)
    return P, cKDTree(P), cbf, cc, tck, uf, pf, scum, Rmed


def profile(video, cfg):
    P, tree, cbf, cc, tck, uf, pf, scum, Rmed = build(cfg["Rdir"])
    L = scum[-1]; camfr = sorted(cbf); camC = np.array([cbf[f] for f in camfr])
    rows = []
    for frac in np.linspace(0.05, 0.96, 60):
        u = np.interp(frac * L, scum, uf); p0 = np.array(splev(u, tck)).T
        t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
        if (p0 - cc) @ t < 0: t = -t
        ip, cov, rm, rs, e1, e2 = br_section(P, tree, p0, t, Rmed)
        fr = camfr[int(np.argmin(np.linalg.norm(camC - p0, axis=1)))]
        rows.append(dict(frac=float(frac), frame=int(fr), cov=float(cov), ratio=float(rs / max(rm, 1e-6)),
                         DCE=float(2 * rm), CSA=float(np.pi * rm * rm), ip=ip))
    valid = [r for r in rows if r["cov"] >= COVOK and r["ratio"] <= RATIOOK]
    sten_v = [r for r in valid if cfg["sten_zone"][0] <= r["frame"] <= cfg["sten_zone"][1]]
    ref_v = [r for r in valid if cfg["ref_zone"][0] <= r["frame"] <= cfg["ref_zone"][1]]
    if not sten_v or not ref_v:
        return dict(rows=rows, ok=False, sten_v=sten_v, ref_v=ref_v)
    # narrowest valid stenosis slices (within 12% of the min CSA) x widest valid trachea (within 12% of max)
    minC = min(x["CSA"] for x in sten_v); maxC = max(x["CSA"] for x in ref_v)
    sten_band = [x for x in sten_v if x["CSA"] <= minC * 1.12]
    ref_band = [x for x in ref_v if x["CSA"] >= maxC * 0.88]
    pcts = [100 * (1 - s["CSA"] / r["CSA"]) for s in sten_band for r in ref_band]
    dpcts = [100 * (1 - s["DCE"] / r["DCE"]) for s in sten_band for r in ref_band]
    best_sten = max(sten_band, key=lambda x: x["cov"])   # best-covered near narrowest
    best_ref = max(ref_v, key=lambda x: x["CSA"])
    return dict(rows=rows, ok=True, sten_v=sten_v, ref_v=ref_v, sten_band=sten_band, ref_band=ref_band,
                best_sten=best_sten, best_ref=best_ref,
                area_band=(round(min(pcts), 1), round(float(np.median(pcts)), 1), round(max(pcts), 1)),
                diam_band=(round(min(dpcts), 1), round(float(np.median(dpcts)), 1), round(max(dpcts), 1)))


def fig(video, cfg, R):
    rows = R["rows"]
    fig = plt.figure(figsize=(16, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.24)
    axp = fig.add_subplot(gs[0, :])
    fr = [r["frame"] for r in rows]; dce = [r["DCE"] for r in rows]
    cols = ["#2ca02c" if (r["cov"] >= COVOK and r["ratio"] <= RATIOOK) else ("#ff9b1a" if r["cov"] >= 0.6 else "#d62728") for r in rows]
    axp.axvspan(*cfg["sten_zone"], color="#ff4d4d", alpha=.08); axp.axvspan(*cfg["ref_zone"], color="#39ff14", alpha=.08)
    axp.scatter(fr, dce, c=cols, s=45, zorder=3, edgecolor="k", linewidth=0.3)
    axp.plot(fr, dce, c="0.6", lw=0.8, zorder=1)
    if R["ok"]:
        bs, br = R["best_sten"], R["best_ref"]
        axp.scatter([bs["frame"]], [bs["DCE"]], s=200, facecolors="none", edgecolors="red", lw=2.2, zorder=4, label="stenosis (best cov)")
        axp.scatter([br["frame"]], [br["DCE"]], s=200, facecolors="none", edgecolors="green", lw=2.2, zorder=4, label="reference")
        axp.legend(fontsize=9, loc="upper left")
        a = R["area_band"]; d = R["diam_band"]
        axp.set_title(f"{cfg['label']} — DCE profile (scene units); green=valid ring (cov>=90% & r_std/r_med<=0.35)\n"
                      f"AREA obstruction band = {a[0]:.0f}–{a[2]:.0f}% (median {a[1]:.0f}%)   ·   "
                      f"DIAMETER band = {d[0]:.0f}–{d[2]:.0f}% (median {d[1]:.0f}%)", fontsize=12)
    axp.set_xlabel("frame (proximal → distal)"); axp.set_ylabel("DCE (scene units)"); axp.grid(alpha=.3)
    # rings
    def ring(ax, d, color, title):
        if d is None: ax.axis("off"); return
        ip = d["ip"]; r = d["DCE"] / 2; th = np.linspace(0, 2 * np.pi, 100)
        ax.scatter(ip[:, 0], ip[:, 1], s=5, c=color, alpha=.5); ax.plot(r*np.cos(th), r*np.sin(th), "k--", lw=1.3)
        ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{title}\ncov={d['cov']*100:.0f}%  r_std/r_med={d['ratio']:.2f}  DCE={d['DCE']:.2f}  CSA={d['CSA']:.2f}",
                     fontsize=9.5, color=("green" if d["cov"] >= COVOK and d["ratio"] <= RATIOOK else "crimson"))
    if R["ok"]:
        ring(fig.add_subplot(gs[1, 0]), R["best_sten"], "#ff4d4d", f"STENOSIS ring (best cov) — f{R['best_sten']['frame']}")
        ring(fig.add_subplot(gs[1, 1]), R["best_ref"], "#2ca02c", f"REFERENCE ring — f{R['best_ref']['frame']}")
        axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
        a = R["area_band"]; d = R["diam_band"]
        axt.text(0, 1, f"{cfg['label']}  STABILITY BAND", fontsize=13, fontweight="bold", va="top")
        axt.text(0, 0.80, f"valid stenosis slices : {len(R['sten_band'])}\n"
                          f"valid trachea slices  : {len(R['ref_band'])}\n\n"
                          f"AREA obstruction : {a[0]:.0f}–{a[2]:.0f}%  (med {a[1]:.0f}%)\n"
                          f"DIAM narrowing   : {d[0]:.0f}–{d[2]:.0f}%  (med {d[1]:.0f}%)",
                 fontsize=12, va="top", family="monospace")
        axt.text(0, 0.34, "• same global-model scale\n• NO mm (scene-unit area ratio)\n"
                          "• within-video only, no cross-patient\n• NOT a final Myer–Cotton grade",
                 fontsize=9.5, va="top", color="#8a1f1f", bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d98a8a"))
    fig.suptitle(f"{cfg['label']} — stenosis stability band (best-coverage slices)", fontsize=14, fontweight="bold")
    path = f"{OUT}/band_{cfg['label']}.png"; fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    summary = {}
    for v, cfg in CFG.items():
        R = profile(v, cfg); p = fig(v, cfg, R)
        if R["ok"]:
            bs = R["best_sten"]
            summary[v] = {"best_stenosis_frame": bs["frame"], "best_stenosis_cov": round(bs["cov"], 2),
                          "best_stenosis_DCE": round(bs["DCE"], 3),
                          "area_band_pct": R["area_band"], "diam_band_pct": R["diam_band"],
                          "n_valid_sten": len(R["sten_band"]), "n_valid_ref": len(R["ref_band"])}
            a = R["area_band"]; d = R["diam_band"]
            print(f"{v}: best-cov stenosis f{bs['frame']} cov={bs['cov']*100:.0f}% DCE={bs['DCE']:.2f} | "
                  f"AREA {a[0]:.0f}-{a[2]:.0f}% (med {a[1]:.0f}%) DIAM {d[0]:.0f}-{d[2]:.0f}% (med {d[1]:.0f}%) -> {p}", flush=True)
        else:
            summary[v] = {"ok": False}; print(f"{v}: no valid stenosis/reference slices", flush=True)
    json.dump(summary, open(f"{OUT}/obstruction_band.json", "w"), indent=2)
    print("saved obstruction_band.json")


if __name__ == "__main__":
    main()
