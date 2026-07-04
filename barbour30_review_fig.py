"""Review figures for the scale-free % obstruction (2-V2, 25-V1). BOTH rings measured in
the ONE global long-window model (same native scene-unit scale). Per video: 3D global
airway + centerline + stenosis/reference slice planes; the two 2D rings; CSA ratio, DCE
ratio, area-obstruction %, diameter-narrowing %. Clearly labelled: same global scale,
NO mm, NO cross-patient comparison, NOT a final Myer-Cotton grade. depth-eval env."""
from __future__ import annotations
import sys; sys.path.insert(0, "/home/mi3dr/projects/bronchotrust")
import numpy as np, pycolmap, open3d as o3d, re
from scipy.spatial import cKDTree
from scipy.interpolate import splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geometry_debug_32v2 as G
from batch_render import section as br_section

OUT = "/home/mi3dr/projects/bronchotrust/runs/barbour30"
STEN = {"#ff4d4d"}  # colors
CFG = {"2-V2":  dict(Rdir="runs/batch4/2-V2",  sten=918, ref=1120, label="2-V2"),
       "25-V1": dict(Rdir="runs/batch4/25-V1", sten=390, ref=520,  label="25-V1")}
rng = np.random.default_rng(0)


def load(Rdir):
    P = np.asarray(o3d.io.read_point_cloud(f"{Rdir}/dense0/fused.ply").points); P = P[np.isfinite(P).all(1)]
    P = P[np.linalg.norm(P - np.median(P, 0), axis=1) > 1e-4]
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0) + 1e-6
    P = P[np.all(np.abs(P - med) < 8 * mad, axis=1)]
    rec = pycolmap.Reconstruction(f"{Rdir}/sparse/0"); cbf = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix()); cbf[int(re.search(r"f(\d+)", im.name).group(1))] = -M[:3, :3].T @ M[:3, 3]
    return P, rec, cbf


def ring(P, tree, pf, scum, uf, tck, cc, Rmed, fr, cbf):
    c = cbf[min(cbf, key=lambda x: abs(x - fr))]; i = int(np.argmin(np.linalg.norm(pf - c, axis=1)))
    u = np.interp(scum[i], scum, uf); p0 = np.array(splev(u, tck)).T
    t = np.array(splev(u, tck, der=1)).T; t /= np.linalg.norm(t) + 1e-9
    if (p0 - cc) @ t < 0: t = -t
    ip, cov, rm, rs, e1, e2 = br_section(P, tree, p0, t, Rmed)
    return dict(p0=p0, t=t, e1=e1, e2=e2, ip=ip, cov=cov, r_med=rm, ratio=rs / max(rm, 1e-6),
                DCE=2 * rm, CSA=float(np.pi * rm * rm))


def draw_ring(ax, d, color, title):
    ip = d["ip"]
    ax.scatter(ip[:, 0], ip[:, 1], s=4, c=color, alpha=.5)
    th = np.linspace(0, 2 * np.pi, 100); r = d["r_med"]
    ax.plot(r * np.cos(th), r * np.sin(th), "k--", lw=1.4, label=f"equiv Ø (DCE={d['DCE']:.2f})")
    ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{title}\ncov={d['cov']*100:.0f}%  r_std/r_med={d['ratio']:.2f}  "
                 f"CSA={d['CSA']:.2f}  DCE={d['DCE']:.2f}  (scene units)", fontsize=9.5,
                 color=("green" if d["ratio"] <= 0.35 else "crimson"))


def main():
    for key, cfg in CFG.items():
        P, rec, cbf = load(cfg["Rdir"])
        Cc = G.cam_centers(rec); cc = Cc.mean(0)
        _, _, pfp = G.pca_centerline(P, cc); Rmed = float(np.median(cKDTree(pfp).query(P)[0]))
        tck, uf, pf = G.medial_centerline(P, pfp, Rmed, cc); scum = G.arclen(pf); tree = cKDTree(P)
        S = ring(P, tree, pf, scum, uf, tck, cc, Rmed, cfg["sten"], cbf)
        Rf = ring(P, tree, pf, scum, uf, tck, cc, Rmed, cfg["ref"], cbf)
        csa_ratio = S["CSA"] / Rf["CSA"]; dce_ratio = S["DCE"] / Rf["DCE"]
        area_obs = 100 * (1 - csa_ratio); diam_narrow = 100 * (1 - dce_ratio)

        # canonical PCA frame for the 3D render
        center = P.mean(0); sub = P[rng.choice(len(P), min(len(P), 60000), replace=False)] - center
        _, _, Vt = np.linalg.svd(sub, full_matrices=False); V = Vt.T
        if (pf[0] - center) @ V[:, 0] > (pf[-1] - center) @ V[:, 0]: V[:, 0] = -V[:, 0]
        if np.linalg.det(V) < 0: V[:, 2] = -V[:, 2]
        rot = lambda X: (X - center) @ V; rotd = lambda D: D @ V
        Pc = rot(P); pfc = rot(pf); s_pt = scum[cKDTree(pf).query(P)[1]]
        lo = np.percentile(Pc, 1, 0); hi = np.percentile(Pc, 99, 0); cen = (lo + hi) / 2; half = 0.6 * max(hi - lo)
        lims = [(cen[i] - half, cen[i] + half) for i in range(3)]

        fig = plt.figure(figsize=(16, 9.5), facecolor="white")
        gs = GridSpec(2, 3, height_ratios=[1.15, 1], hspace=0.28, wspace=0.24)
        ax3d = fig.add_subplot(gs[0, :2], projection="3d", facecolor="black")
        idx = rng.choice(len(Pc), min(len(Pc), 140000), replace=False)
        ax3d.scatter(Pc[idx, 0], Pc[idx, 1], Pc[idx, 2], c=s_pt[idx] / scum[-1], cmap="turbo", s=1.0, alpha=.5, linewidths=0)
        ax3d.plot(pfc[:, 0], pfc[:, 1], pfc[:, 2], c="cyan", lw=2.2)
        for d, col, nm in [(S, "#ff4d4d", "STENOSIS"), (Rf, "#39ff14", "REFERENCE (trachea)")]:
            p0c = rot(d["p0"]); e1c, e2c = rotd(d["e1"]), rotd(d["e2"]); hh = 1.7 * d["r_med"]
            q = np.array([p0c + a*hh*e1c + b*hh*e2c for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
            ax3d.add_collection3d(Poly3DCollection([q], alpha=.55, facecolor=col, edgecolor=col))
            ax3d.text(p0c[0], p0c[1], p0c[2] + hh * 1.6, nm, color=col, fontsize=10, fontweight="bold", ha="center")
        for i, sl in enumerate((ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim)): sl(*lims[i])
        ax3d.set_axis_off()
        try: ax3d.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax3d.view_init(16, -62)
        ax3d.set_title(f"{cfg['label']} — global airway model (ONE connected model, same native scale)\n"
                       "colour = proximal→distal;  slice planes at stenosis (red) & reference (green)",
                       color="black", fontsize=11)

        axtxt = fig.add_subplot(gs[0, 2]); axtxt.axis("off")
        axtxt.text(0.0, 1.0, f"{cfg['label']}  —  SCALE-FREE OBSTRUCTION", fontsize=13, fontweight="bold", va="top")
        axtxt.text(0.0, 0.86,
                   f"stenosis  : {cfg['label']} subglottis (f{cfg['sten']})\n"
                   f"reference : trachea (f{cfg['ref']})\n\n"
                   f"CSA ratio (sten/ref)   = {csa_ratio:.3f}\n"
                   f"DCE ratio (sten/ref)   = {dce_ratio:.3f}\n\n"
                   f"AREA obstruction  = 1 - CSA_s/CSA_r = {area_obs:.1f}%\n"
                   f"DIAMETER narrowing = 1 - DCE_s/DCE_r = {diam_narrow:.1f}%",
                   fontsize=12, va="top", family="monospace")
        axtxt.text(0.0, 0.30,
                   "• SAME global-model scale (both rings, one model)\n"
                   "• NO mm — scene-unit area ratio only\n"
                   "• NO cross-patient comparison (per-video only)\n"
                   "• NOT a final Myer–Cotton grade\n"
                   "  (reference = this patient's trachea caliber,\n"
                   "   not a normative airway size)",
                   fontsize=10, va="top", color="#8a1f1f",
                   bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d98a8a"))

        draw_ring(fig.add_subplot(gs[1, 0]), S, "#ff4d4d", f"STENOSIS ring — subglottis (f{cfg['sten']})")
        draw_ring(fig.add_subplot(gs[1, 1]), Rf, "#2ca02c", f"REFERENCE ring — trachea (f{cfg['ref']})")
        axb = fig.add_subplot(gs[1, 2])
        x = np.arange(2)
        axb.bar(x - 0.18, [S["CSA"], Rf["CSA"]], 0.36, label="CSA", color=["#ff4d4d", "#2ca02c"])
        axb2 = axb.twinx()
        axb2.plot(x, [S["DCE"], Rf["DCE"]], "ko--", label="DCE")
        axb.set_xticks(x); axb.set_xticklabels(["stenosis", "reference"]); axb.set_ylabel("CSA (scene²)")
        axb2.set_ylabel("DCE (scene)"); axb.set_title(f"CSA & DCE (scene units)\narea obstruction ≈ {area_obs:.0f}%", fontsize=10)
        axb.grid(alpha=.3, axis="y")

        fig.suptitle(f"{cfg['label']} — scale-free % obstruction (REVIEW)   "
                     f"area {area_obs:.0f}% · diameter {diam_narrow:.0f}%", fontsize=14, fontweight="bold")
        path = f"{OUT}/review_{cfg['label']}.png"
        fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight"); plt.close(fig)
        print(f"{cfg['label']}: CSAratio={csa_ratio:.3f} DCEratio={dce_ratio:.3f} area={area_obs:.1f}% "
              f"diam={diam_narrow:.1f}% | sten ring cov{S['cov']*100:.0f}%/{S['ratio']:.2f} "
              f"ref ring cov{Rf['cov']*100:.0f}%/{Rf['ratio']:.2f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
