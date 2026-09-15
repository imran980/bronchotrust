"""Per-case CAPTURE-BEHAVIOUR analysis: why each reconstruction succeeded or failed, with evidence from the
video. Two phases so it can run in parallel workers:
  phase 1  python capture_analysis.py CASE [CASE ...]   -> per-case cache in reports/capture_cache/CASE.npz
  phase 2  python capture_analysis.py assemble          -> tables + figures from the cache
Per frame (every STRIDE-th frame, analysis width 480 px, inside the endoscope field):
  offset     dark-lumen blob centroid distance from field centre, in field radii (0 = centred)
  sharpness  variance of Laplacian of L*          motion  mean |L*_t - L*_{t-1}|
  specular   fraction of pixels with L* > 245     Lmean   mean L* (dark-frame gate at 12)
Segment analysed = known reconstruction window (re-runs) else the longest lumen-visible traversal."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, json, subprocess, numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.cm as cm
from figstyle import apply_style, OKABE; apply_style()
from airway_analysis import scatter3d
R = "runs/own_data"; PLY = f"{R}/all_reconstructions_ply"; REP = f"{R}/reports"; OUT = f"{R}/renders/new"; CACHE = f"{REP}/capture_cache"; os.makedirs(CACHE, exist_ok=True)
YIELD = {r["case"]: r for r in json.load(open(f"{REP}/yield_table.json"))}
WINDOWS = {"2-V2": (2870, 3370), "50-V2": (230, 520), "20-V1": (2500, 3000), "17-V1": (500, 1330), "26-V2": (1620, 1901), "13-V1": (660, 1460)}
COL = {"CT-validated": OKABE["green"], "measurable tube": OKABE["blue"], "partially measurable": OKABE["orange"], "not measurable": "0.5"}
TIER_ORDER = ["CT-validated", "measurable tube", "partially measurable", "not measurable"]
W, STRIDE = 480, 2


def find_video(c):
    u = c.replace("-", "_")
    for p in [f"{c}.mp4", f"{c} - *.mp4", f"{u}.mp4", f"{c}_*.mp4", f"{u}_*.mp4", f"*{c}*.mp4", f"*{u}*.mp4"]:
        r = [x for x in subprocess.run(["find", "/home/mi3dr/dataset", "-type", "f", "-iname", p, "!", "-iname", "*alib*"], capture_output=True, text=True).stdout.split("\n") if x]
        if r: return r[0]
    return None


def field_mask(video):
    cap = cv2.VideoCapture(video); N = int(cap.get(7)); acc = None
    for fi in np.linspace(0, max(N - 1, 0), 60).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if not ok: continue
        g = cv2.cvtColor(cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1]))), cv2.COLOR_BGR2GRAY)
        acc = (g > 8).astype(np.int32) if acc is None else acc + (g > 8)
    cap.release(); m = (acc >= 12).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1: m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = cv2.erode(m, np.ones((7, 7), np.uint8), iterations=2); ys, xs = np.nonzero(m)
    return m.astype(bool), (xs.mean(), ys.mean()), np.sqrt(m.sum() / np.pi), N


def analyse_case(c):
    v = find_video(c)
    if not v: print(f"{c}: VIDEO MISSING", flush=True); return
    mask, centre, Rf, N = field_mask(v); cap = cv2.VideoCapture(v); rows = []; prev = None; fi = 0; cache = {}
    while True:
        ok, f = cap.read()
        if not ok: break
        if fi % STRIDE:
            fi += 1; continue
        s = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1]))); L = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)[:, :, 0]; Lm = float(L[mask].mean())
        if Lm < 12: rows.append((fi, np.nan, np.nan, np.nan, np.nan, Lm, 0)); prev = None; fi += 1; continue
        thr = np.percentile(L[mask], 8); dark = ((L < thr) & mask).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(dark); vis = 0; off = np.nan
        if n > 1:
            k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); af = st[k, cv2.CC_STAT_AREA] / mask.sum()
            if 0.004 < af < 0.35: vis = 1; off = float(np.hypot(cen[k][0] - centre[0], cen[k][1] - centre[1]) / Rf)
        sharp = float(cv2.Laplacian(L, cv2.CV_64F)[mask].var()); spec = float((L[mask] > 245).mean())
        mot = float(np.abs(L.astype(np.int16) - prev.astype(np.int16))[mask].mean()) if prev is not None else np.nan
        rows.append((fi, off, sharp, mot, spec, Lm, vis)); prev = L
        if fi % 40 == 0: cache[fi] = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        fi += 1
    cap.release(); F = np.array(rows, dtype=float)
    idx = F[:, 0].astype(int); vis = F[:, 6] > 0; win = WINDOWS.get(c)
    if win: seg = (idx >= win[0]) & (idx <= win[1])
    else:   # longest lumen-visible run allowing gaps <= 15 frames
        best, cs, last = (0, 0, 0), None, None
        for i in np.nonzero(vis)[0]:
            if cs is None or idx[i] - idx[last] > 15:
                if cs is not None and idx[last] - idx[cs] > best[0]: best = (idx[last] - idx[cs], idx[cs], idx[last])
                cs = i
            last = i
        if cs is not None and idx[last] - idx[cs] > best[0]: best = (idx[last] - idx[cs], idx[cs], idx[last])
        seg = (idx >= best[1]) & (idx <= best[2])
    S = F[seg]; sidx = S[:, 0].astype(int); picks = [sidx[int(q * (len(sidx) - 1))] for q in (0.15, 0.38, 0.62, 0.85)]
    ck = np.array(sorted(cache)); frames = np.stack([cache[ck[np.argmin(np.abs(ck - p))]] for p in picks])
    np.savez_compressed(f"{CACHE}/{c}.npz", F=F, seg=seg, frames=frames, picks=np.array(picks), video=os.path.basename(v), N=N, window_known=c in WINDOWS)
    print(f"{c}: {os.path.basename(v)} N={N} seg f{sidx[0]}-{sidx[-1]} ({seg.sum()} sampled frames) -> cached", flush=True)


def load_pts(c, cap=150000):
    """cleaned point array for the render column (median trim + statistical outlier, as the yield table); cached per case."""
    fp = f"{CACHE}/{c}_pts.npy"
    if os.path.exists(fp): return np.load(fp)
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(f"{PLY}/{c}.ply").points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1); P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8); P = np.asarray(pc.points)
    if len(P) > cap: P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]
    np.save(fp, P.astype(np.float32)); return P


def note(s):
    p = ["steady, centred lumen for most of the pass" if s["pct_centred"] >= 70 else ("lumen centred only intermittently" if s["pct_centred"] >= 45 else "lumen off-centre / wall-facing for most of the pass")]
    if s["pct_blur"] > 35: p.append("frequent defocus or motion blur")
    if s["motion_rel"] > 1.5: p.append("fast scope motion")
    if s["pct_specular"] > 25: p.append("frequent glare / secretions")
    if s["n_seg_frames"] < 200: p.append("short traversal")
    return "; ".join(p)


def assemble():
    results, strips = [], {}
    cases = sorted(YIELD, key=lambda c: (TIER_ORDER.index(YIELD[c]["tier"]), -YIELD[c]["n_gated"]))
    for c in cases:
        fp = f"{CACHE}/{c}.npz"
        if not os.path.exists(fp): print(f"{c}: no cache"); continue
        z = np.load(fp, allow_pickle=True); F, seg = z["F"], z["seg"]; S = F[seg]
        off, sh, mo, sp, vis = S[:, 1], S[:, 2], S[:, 3], S[:, 4], S[:, 6] > 0
        offv, shv, mov, spv = off[np.isfinite(off)], sh[np.isfinite(sh)], mo[np.isfinite(mo)], sp[np.isfinite(sp)]
        s = dict(case=c, tier=YIELD[c]["tier"], n_gated=YIELD[c]["n_gated"], median_cov=YIELD[c]["median_cov"], video=str(z["video"]), n_video=int(z["N"]),
                 seg_lo=int(S[0, 0]), seg_hi=int(S[-1, 0]), n_seg_frames=int(S[-1, 0] - S[0, 0] + 1), window_known=bool(z["window_known"]),
                 pct_lumen_visible=float(100 * vis.mean()), pct_centred=float(100 * (offv < 0.35).mean()) if len(offv) else 0.0,
                 median_offset=float(np.median(offv)) if len(offv) else np.nan, pct_edge=float(100 * (offv > 0.6).mean()) if len(offv) else 0.0,
                 median_sharp=float(np.median(shv)) if len(shv) else np.nan, pct_blur=float(100 * (shv < 0.3 * np.percentile(shv, 90)).mean()) if len(shv) else np.nan,
                 median_motion=float(np.median(mov)) if len(mov) else np.nan, pct_specular=float(100 * (spv > 0.015).mean()) if len(spv) else np.nan)
        results.append(s); strips[c] = dict(frames=z["frames"], picks=z["picks"], trace=(S[:, 0].astype(int), off), P=load_pts(c))
    med_mo = np.nanmedian([r["median_motion"] for r in results])
    for r in results: r["motion_rel"] = float(r["median_motion"] / med_mo) if np.isfinite(r["median_motion"]) else np.nan; r["note"] = note(r)
    json.dump(results, open(f"{REP}/capture_table.json", "w"), indent=1)
    md = ["| Case | Tier | Gated | Segment (frames) | Lumen centred (%) | Median offset (R) | Blur (%) | Motion (rel.) | Glare (%) | Capture note |", "|---|---|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{llrrrrrrl}", "\\toprule", "Case & Tier & Gated & Centred (\\%) & Offset (R) & Blur (\\%) & Motion (rel.) & Glare (\\%) & Capture note \\\\", "\\midrule"]
    for r in results:
        md.append(f"| {r['case']} | {r['tier']} | {r['n_gated']} | {r['seg_lo']}–{r['seg_hi']}{'' if r['window_known'] else '*'} | {r['pct_centred']:.0f} | {r['median_offset']:.2f} | {r['pct_blur']:.0f} | {r['motion_rel']:.2f} | {r['pct_specular']:.0f} | {r['note']} |")
        tex.append(f"{r['case']} & {r['tier']} & {r['n_gated']} & {r['pct_centred']:.0f} & {r['median_offset']:.2f} & {r['pct_blur']:.0f} & {r['motion_rel']:.2f} & {r['pct_specular']:.0f} & {r['note']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(f"{REP}/capture_table.md", "w").write("\n".join(md) + "\n"); open("paper/capture_table.tex", "w").write("\n".join(tex) + "\n")
    per_page = 6
    for pg in range(0, len(results), per_page):
        chunk = results[pg:pg + per_page]; fig = plt.figure(figsize=(16, 2.7 * len(chunk)))
        gs = fig.add_gridspec(len(chunk), 7, width_ratios=[1, 1, 1, 1, 0.75, 1.0, 1.55], hspace=0.38, wspace=0.12)
        for i, r in enumerate(chunk):
            Sd = strips[r["case"]]; col = COL[r["tier"]]
            for j in range(4):
                ax = fig.add_subplot(gs[i, j]); ax.imshow(Sd["frames"][j]); ax.set_axis_off(); ax.set_title(f"f{int(Sd['picks'][j])}", fontsize=9.5, color="0.35", pad=1)
            ax = fig.add_subplot(gs[i, 4], projection="3d"); scatter3d(ax, Sd["P"], sub=30000, s=1.0, rad_pct=95, zoom=1.45)             # side view, PCA axis vertical, true proportions
            if i == 0: ax.set_title("cloud · side", fontsize=9.5, color="0.35", pad=1)
            ax = fig.add_subplot(gs[i, 5], projection="3d"); scatter3d(ax, Sd["P"], sub=30000, s=1.0, elev=90, azim=-90, rad_pct=95, box=(1, 1, 0.4), zoom=1.12)  # down the lumen axis (ring closure)
            if i == 0: ax.set_title("cloud · along axis", fontsize=9.5, color="0.35", pad=1)
            ax = fig.add_subplot(gs[i, 6]); idx, off = Sd["trace"]; ax.plot(idx, off, ".", ms=2, color=col, alpha=0.6); ax.axhline(0.35, color="0.6", ls="--", lw=0.8)
            ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=8.5); ax.set_ylabel("lumen offset (R)", fontsize=9); ax.grid(alpha=0.2)
            ax.set_title(f"centred {r['pct_centred']:.0f}% · blur {r['pct_blur']:.0f}% · glare {r['pct_specular']:.0f}%", fontsize=9, color="0.35", pad=2)
            tier_lab = r["tier"].replace("partially measurable", "partially\nmeasurable")
            fig.text(0.005, 1 - (i + 0.5) / len(chunk), f"{r['case']}\n{tier_lab}\n{r['n_gated']} gated", va="center", fontsize=11, fontweight="bold", color=col)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.02); fig.savefig(f"{OUT}/fig_capture_cases_p{pg // per_page + 1}.png", dpi=130); plt.close(fig)
    from scipy.stats import spearmanr; from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, key, lab in zip(axs, ["pct_centred", "pct_blur", "motion_rel"], ["lumen centred (% of pass)", "blurred frames (%)", "scope motion (relative to cohort median)"]):
        x = np.array([r[key] for r in results], float); y = np.array([r["n_gated"] for r in results], float); k = np.isfinite(x)
        for r in results: ax.scatter(r[key], r["n_gated"], s=46, color=COL[r["tier"]], edgecolor="white", linewidth=0.8, zorder=3); ax.annotate(r["case"], (r[key], r["n_gated"]), textcoords="offset points", xytext=(4, 3), fontsize=8.5, color="0.4")
        rho, p = spearmanr(x[k], y[k]); ax.set_title(f"Spearman ρ = {rho:+.2f} (p = {p:.2f})", fontsize=12); ax.axhline(20, color="0.75", ls="--", lw=1); ax.axhline(10, color="0.85", ls=":", lw=1); ax.set_xlabel(lab); ax.grid(alpha=0.25)
    axs[0].set_ylabel("gated cross-sections (of 48)")
    fig.legend(handles=[Patch(color=COL[t], label=t) for t in COL], loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Capture behaviour vs measurability", fontsize=12.5, fontweight="bold"); fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(f"{OUT}/fig_capture_summary.png", dpi=150)
    print(f"ASSEMBLED {len(results)} cases -> capture_table.*, {(len(results) + per_page - 1) // per_page} strip pages, fig_capture_summary.png"); print("\n".join(md))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble": assemble()
    else:
        for c in sys.argv[1:]:
            try: analyse_case(c)
            except Exception as e: print(f"{c}: ERR {e}", flush=True)
