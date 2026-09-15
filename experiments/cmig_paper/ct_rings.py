"""CT cross-section ring overlay: at three stations per CT-paired case, plot the reconstructed wall
points (mm, scaled by the case's isotropic factor) against the CT lumen circle of the matched
arclength. Shows the actual measured ring vs CT — the visual behind the D_CE numbers. CPU/matplotlib."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, numpy as np
import open3d as o3d, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from csa_partialarc import centerline, fit_circle
from figstyle import apply_style, OKABE, PATIENT; apply_style()
D = f"{ROOT}/runs/own_data/ct_validated_3"; OUT = f"{ROOT}/runs/own_data/renders/new"
J = json.load(open(f"{D}/CT_validation_metrics.json"))


def ct_profile(pid):
    try:
        z = np.load(f"{D}/ts_{pid}/gt_{pid}.json/ground_truth.npz"); return np.asarray(z["arclength_mm"]), np.asarray(z["csa_mm2"])
    except Exception:
        v = J[pid]; a = np.array(v["arc"])[:len(v["ct_dce"])]; return a, np.pi * (np.array(v["ct_dce"]) / 2) ** 2


def load_clean(fp):
    P = np.asarray(o3d.io.read_point_cloud(fp).points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1)
    P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8); pc, _ = pc.remove_radius_outlier(16, np.median(d) * 0.03)
    return np.asarray(pc.points)


def station_data(P, nb=60):
    cl, ax, Vt = centerline(P, nb); e1, e2 = Vt[1], Vt[2]; Pc = P - P.mean(0); tall = Pc @ ax
    t0 = cl[:, 0].min(); cl[:, 0] -= t0; tall = tall - t0        # zero-base the axis coordinate
    slab = 0.6 * np.median(np.diff(cl[:, 0])); out = []
    for st in cl:
        sel = np.abs(tall - st[0]) < slab
        if sel.sum() < 40: out.append(None); continue
        Q = P[sel] - st[1:]; xy = np.c_[Q @ e1, Q @ e2]; cx, cy, Rr, rr = fit_circle(xy)
        az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))  # coverage about the centerline point (validated convention)
        cov = (np.histogram(az, 36, (-180, 180))[0] > 0).mean()
        out.append(dict(t=st[0], xy=xy - [cx, cy], R=Rr, resid=rr, cov=cov))
    return out, cl


def register(stations, ca, cd_dce):
    ok = [s for s in stations if s and s["cov"] >= 0.75 and s["resid"] < 0.15]
    a_r = np.array([s["t"] for s in ok]); r_m = np.array([s["R"] for s in ok]); o = np.argsort(a_r); a_r, r_m = a_r[o] - a_r.min(), r_m[o]
    ok = [ok[i] for i in o]; rmed = np.median(r_m); ss = np.linspace(0.5 * cd_dce.min() / (2 * rmed), 2 * cd_dce.max() / (2 * rmed), 1500); best = None
    for flip in (False, True):
        C = cd_dce[::-1] if flip else cd_dce; A = (ca[-1] - ca[::-1]) if flip else ca
        for s in ss:
            e = 2 * s * r_m - np.interp(s * a_r, A, C, left=np.nan, right=np.nan); m = np.isfinite(e)
            if m.sum() > 5:
                rm = np.sqrt(np.mean(e[m] ** 2))
                if best is None or rm < best[0]: best = (rm, s, flip)
    if best is None:
        print(f"  register: len(ok)={len(ok)} rmed={rmed:.3f} ss=[{ss.min():.2f},{ss.max():.2f}] a_r=[{a_r.min():.2f},{a_r.max():.2f}] ct=[{ca.min():.0f},{ca.max():.0f}] dce=[{cd_dce.min():.1f},{cd_dce.max():.1f}]", flush=True)
        raise SystemExit
    _, s, flip = best; C = cd_dce[::-1] if flip else cd_dce; A = (ca[-1] - ca[::-1]) if flip else ca
    return ok, a_r, s, flip, A, C


def main():
    CASES = ["2-V2", "20-V1", "50-V2"]
    fig, axs = plt.subplots(len(CASES), 3, figsize=(9.5, 9.8))
    for i, pid in enumerate(CASES):
        ca, csa = ct_profile(pid); cd_dce = 2 * np.sqrt(np.clip(csa, 0, None) / np.pi)
        P = load_clean(f"{ROOT}/runs/own_data/all_reconstructions_ply/{pid}.ply"); st, cl = station_data(P)
        ok, a_r, s, flip, A, C = register(st, ca, cd_dce)
        picks = [ok[int(q * (len(ok) - 1))] for q in (0.2, 0.5, 0.8)]; col = PATIENT.get(pid, OKABE["blue"])
        for j, stn in enumerate(picks):
            ax = axs[i, j]; xy = stn["xy"] * s; am = (stn["t"] - a_r.min()) * s
            ctdce = float(np.interp(am, A, C)); ctR = ctdce / 2
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(ctR * np.cos(th), ctR * np.sin(th), color="0.15", lw=2.2, label=f"CT circle  D={ctdce:.1f} mm", zorder=1)
            ax.scatter(xy[:, 0], xy[:, 1], s=5, color=col, alpha=0.5, linewidths=0, label=f"recon wall  D={2*stn['R']*s:.1f} mm", zorder=2)
            lim = max(ctR, stn["R"] * s) * 1.5; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{pid} · arclength {am:.0f} mm · cov {stn['cov']:.2f}", fontsize=9)
            ax.legend(fontsize=6.6, loc="lower center", handletextpad=0.3, framealpha=0.85)
            for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.6)
        print(f"{pid}: s={s:.2f} flip={flip}, stations at arclen {[f'{(p[chr(116)]-a_r.min())*s:.0f}' for p in picks]} mm", flush=True)
    fig.suptitle("Reconstructed cross-sections vs CT lumen (mm) — the rings behind the $D_{CE}$ numbers", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Recon wall points scaled to mm by the case's isotropic Sim(3) factor; CT circle = lumen of equal area at the matched arclength.", ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98)); fig.savefig(f"{OUT}/fig_ct_rings.png", dpi=150); print("saved fig_ct_rings.png")


if __name__ == "__main__":
    main()
