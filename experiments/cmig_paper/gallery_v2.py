"""Feasibility gallery + yield table, CONSISTENT and defensible: tiers from ring coverage measured by the
validated partial-arc cross-section method (not an ad-hoc PCA one-sidedness). Per cloud: csa_run cleaning,
profile(nb=60), interior stations (10-90%), n_gated = cov>=0.75 & resid<0.15, median coverage.
Tier: CT-validated | complete tube (n_gated>=20) | partial rings (10-19) | one-sided (<10).
Renders each cloud (Open3D offscreen if available, else matplotlib) with viridis along the axis."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, glob, json, numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.cm as cm
from csa_partialarc import profile
from figstyle import apply_style, OKABE; apply_style()
PLY = "runs/own_data/all_reconstructions_ply"; OUT = "runs/own_data/renders/new"; REP = "runs/own_data/reports"
OLD = {"10-V2", "15-V2", "16-V1", "25-V1", "31-V1", "32-V2", "33-V1", "7-V1"}; CT = {"2-V2", "20-V1", "50-V2"}
COL = {"CT-validated": OKABE["green"], "measurable tube": OKABE["blue"], "partially measurable": OKABE["orange"], "not measurable": "0.5"}

def load_clean(fp):
    P = np.asarray(o3d.io.read_point_cloud(fp).points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8)
    pc, _ = pc.remove_radius_outlier(16, np.median(d) * 0.03); return np.asarray(pc.points), len(d)

def axis_colors(P):
    c = P - P.mean(0); _, _, Vt = np.linalg.svd(c[::max(1, len(c)//50000)], full_matrices=False); t = c @ Vt[0]
    t = (t - t.min()) / (np.ptp(t) + 1e-9); return cm.viridis(t)[:, :3], Vt

def render_o3d(P, cols, Vt, w=520, h=700):
    import open3d.visualization.rendering as rd
    r = rd.OffscreenRenderer(w, h); r.scene.set_background([0.93, 0.93, 0.93, 1.0])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.colors = o3d.utility.Vector3dVector(cols)
    mat = rd.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 2.6; r.scene.add_geometry("pc", pc, mat)
    bb = pc.get_axis_aligned_bounding_box(); c = np.asarray(bb.get_center(), dtype=np.float32); ext = float(np.linalg.norm(bb.get_extent()))
    eye = (c + Vt[1] * ext * 1.35).astype(np.float32); up = np.asarray(Vt[0], dtype=np.float32)
    r.setup_camera(40.0, c, eye, up)            # (fov, center, eye, up) overload — float32 vectors required
    return np.asarray(r.render_to_image())

def render_mpl(P, w=520, h=700):
    from airway_analysis import scatter3d
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100); ax = fig.add_subplot(111, projection="3d"); scatter3d(ax, P, sub=60000, s=2.0)
    fig.subplots_adjust(0, 0, 1, 1); fig.canvas.draw(); im = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig); return im

rows, tiles, mode = [], {}, None
for f in sorted(glob.glob(f"{PLY}/*.ply"), key=lambda s: (int(os.path.basename(s).split("-")[0]), s)):
    pid = os.path.basename(f)[:-4]
    try:
        P, n_raw = load_clean(f); rs, _ = profile(P, nb=60); n = len(rs); lo, hi = int(0.1 * n), int(0.9 * n); rs = rs[lo:hi]
        cov = np.array([r["cov"] for r in rs]); res = np.array([r["resid"] for r in rs]); ng = int(((cov >= 0.75) & (res < 0.15)).sum()); mc = float(np.median(cov)) if len(cov) else 0.0
        tier = "CT-validated" if pid in CT else ("measurable tube" if ng >= 20 else ("partially measurable" if ng >= 10 else "not measurable"))
        reason = "" if ng >= 20 else ("one-sided (low coverage)" if mc < 0.75 else "noisy wall (fit residual)")
        rows.append(dict(case=pid, batch="first" if pid in OLD else "second", n_raw=n_raw, n_clean=len(P), n_stations=len(rs), n_gated=ng, median_cov=round(mc, 2), tier=tier, reason=reason, ct=pid in CT))
        cols, Vt = axis_colors(P)
        if mode != "mpl":
            try: tiles[pid] = render_o3d(P, cols, Vt); mode = mode or "o3d"
            except Exception as e: mode = "mpl"; print("o3d offscreen unavailable ->", str(e)[:80])
        if pid not in tiles: tiles[pid] = render_mpl(P)
        print(f"{pid}: clean {len(P):,}  gated {ng}/{len(rs)}  medcov {mc:.2f}  -> {tier}", flush=True)
    except Exception as e:
        print(f"{pid}: ERR {e}", flush=True)
json.dump(rows, open(f"{REP}/yield_table.json", "w"), indent=1)
from collections import Counter; print("TIERS:", dict(Counter(r["tier"] for r in rows)), "| renderer:", mode)
# tables
md = ["| Case | Batch | Points (clean) | Gated sections | Median coverage | Tier | Limiting factor | CT |", "|---|---|---|---|---|---|---|---|"]
tex = ["\\begin{tabular}{llrrclll}", "\\toprule", "Case & Batch & Points & Gated sections & Median cov. & Tier & Limiting factor & CT \\\\", "\\midrule"]
order = {"CT-validated": 0, "measurable tube": 1, "partially measurable": 2, "not measurable": 3}
for r in sorted(rows, key=lambda r: (order[r["tier"]], -r["n_gated"])):
    md.append(f"| {r['case']} | {r['batch']} | {r['n_clean']:,} | {r['n_gated']}/{r['n_stations']} | {r['median_cov']:.2f} | {r['tier']} | {r['reason']} | {'yes' if r['ct'] else ''} |")
    tex.append(f"{r['case']} & {r['batch']} & {r['n_clean']:,} & {r['n_gated']}/{r['n_stations']} & {r['median_cov']:.2f} & {r['tier']} & {r['reason']} & {'yes' if r['ct'] else ''} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(f"{REP}/yield_table.md", "w").write("\n".join(md) + "\n"); open("paper/yield_table.tex", "w").write("\n".join(tex) + "\n")
# gallery
srt = sorted(rows, key=lambda r: (order[r["tier"]], -r["n_gated"])); cols_n = 6; rows_n = (len(srt) + cols_n - 1) // cols_n
fig, axs = plt.subplots(rows_n, cols_n, figsize=(2.9 * cols_n, 3.9 * rows_n)); axs = np.array(axs).ravel()
for k, r in enumerate(srt):
    ax = axs[k]; ax.imshow(tiles[r["case"]]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(COL[r["tier"]]); s.set_linewidth(2.4)
    ax.set_title(f"{r['case']}  ·  {r['n_gated']} gated · cov {r['median_cov']:.2f}", fontsize=9.5, fontweight="bold", color=COL[r["tier"]], pad=2)
for k in range(len(srt), len(axs)): axs[k].axis("off")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL[t], label=t) for t in order], loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Reconstructed pediatric airways — raw dense point clouds, tiered by measurability (gated cross-sections out of 48 interior stations)", fontsize=13, fontweight="bold", y=0.995)
fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.04, wspace=0.04, hspace=0.16)
fig.savefig(f"{OUT}/fig_gallery_v2.png", dpi=130); print("saved fig_gallery_v2.png; tables -> yield_table.md / paper/yield_table.tex")
