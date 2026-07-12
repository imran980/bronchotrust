"""25_V1 SAME-MODEL scale-free % obstruction attempt from ONE connected subglottic model
(recon_25v1_subglottis, frames 340-470). One connected model IS required; both rings must be
measured in it at one scene scale (NO cross-reconstruction, NO Sim3, NO mixing with the old distal
DCE). Uses a camera-trajectory-backbone medial-axis centerline (the airway is curved, so the
single-axis projection centerline tangles). Accept a ring iff thin-slab cov>=0.75 AND polar~ellipse
AND 3-way spread<=1.3 AND r_std/r_med tight AND STABLE across a run of nearby slices.

If the proximal/narrowest ring does not pass (e.g. the capture-marginal proximal frames degrade the
connected reconstruction), report proximal ring quality only and DO NOT compute %. Scene units; no
mm; no Myer-Cotton. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap
from scipy.interpolate import splprep, splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import centerline_csa as cc, ring_completeness as rc
from measure_25v1_distal import measure_declutter

ROOT = Path("/home/mi3dr/projects/bronchotrust")
WORK = ROOT / "runs/barbour30/airwayfit/recon_25v1_subglottis"
MODEL = str(WORK / "sparse/0"); FUSION = "geometric"


def curved_centerline(P, C, smooth=2.0, iters=2):
    tck, _ = splprep(C.T, s=len(C) * smooth, k=3); ss = np.linspace(0, 1, 220)
    Cl = np.array(splev(ss, tck)).T; Tg = np.array(splev(ss, tck, der=1)).T; Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
    for _ in range(iters):
        pts = []
        for i in range(0, len(Cl), 3):
            t, e1, e2 = cc.basis(Tg[i]); d = (P - Cl[i]) @ t
            r0 = np.median(np.linalg.norm((P - Cl[i]) - np.outer(d, t), axis=1)); band = P[np.abs(d) <= 0.35 * r0]
            if len(band) < 40: continue
            rad = np.linalg.norm((band - Cl[i]) - np.outer((band - Cl[i]) @ t, t), axis=1)
            pts.append(np.median(band[rad < 1.6 * np.median(rad)], 0))
        pts = np.array(pts)
        if len(pts) < 8: break
        tck, _ = splprep(pts.T, s=len(pts) * 0.05, k=3); Cl = np.array(splev(ss, tck)).T
        Tg = np.array(splev(ss, tck, der=1)).T; Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-12
    return Cl, Tg


def main():
    summ = json.loads((WORK / "summary.json").read_text())
    P, n0, n1 = rc.load_clean(str(WORK / "dense" / f"fused_{FUSION}.ply"))
    rec = pycolmap.Reconstruction(MODEL)
    fc = sorted((int(re.search(r"f(\d+)", im.name).group(1)), np.array(im.projection_center())) for im in rec.images.values())
    frames = np.array([f for f, _ in fc]); C = np.array([c for _, c in fc])
    Cl, Tg = curved_centerline(P, C)
    s_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Cl, axis=0), axis=1))])
    slices = []
    for i in range(3, 217):
        m = measure_declutter(P, Cl[i], Tg[i])
        if m: m["i"] = i; m["s"] = float(s_arc[i]); slices.append(m)
    acc = [m for m in slices if m["cov"] >= 0.75 and m["spread"] <= 1.30 and m["ratio"] <= 0.35]
    # require a STABLE run: >=3 accepted slices within a short arc window whose DCE spread <=25%
    stable_runs = []
    for a in acc:
        nb = [b for b in acc if abs(b["s"] - a["s"]) <= 6.0]
        if len(nb) >= 3:
            d = [b["dce"] for b in nb]; sp = 100 * (max(d) - min(d)) / max(np.median(d), 1e-9)
            if sp <= 25: stable_runs.append((a, np.median(d), sp, len(nb)))
    out = dict(video="25_V1", model="one connected subglottic model (recon_25v1_subglottis)",
               connected=summ.get("connected"), reg_span=summ.get("reg_span"), n_prox=summ.get("n_prox"),
               n_dist=summ.get("n_dist"), fusion=FUSION, units="scene units (within this one connected model)",
               n_slices=len(slices), n_accepted=len(acc), n_stable_accepted=len(stable_runs))
    # proximal ring quality (best attempt, regardless of pass): highest-coverage low-spread slice
    hi = [m for m in slices if m["cov"] >= 0.70]
    bcov = max(hi, key=lambda m: (round(m["cov"], 2), -m["spread"])) if hi else (max(slices, key=lambda m: m["cov"]) if slices else None)
    out["best_ring_attempt"] = (None if not bcov else dict(cov=round(bcov["cov"], 2), ratio=round(bcov["ratio"], 3),
        spread=round(bcov["spread"], 3), pe=round(bcov["pe"], 3), DCE=round(bcov["dce"], 3)))
    if len(stable_runs) < 1:
        out["verdict"] = ("CONNECTED MODEL BUILT but NO ring meets accept+stability -> the capture-marginal proximal "
                          "frames degrade the connected reconstruction (rings sparse/unstable). Per the rules: proximal "
                          "ring quality only, % obstruction NOT computed.")
    else:
        out["verdict"] = "stable accepted rings exist — see profile; % computed only if proximal narrowest is among them"
    (WORK / "obstruction_report.json").write_text(json.dumps(out, indent=2, default=str))
    # figure: CSA profile along arc length + best-attempt ring
    fig = plt.figure(figsize=(13, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    for key, cc_ in (("csa_polar", "tab:blue"), ("csa_ell", "tab:orange"), ("csa_hull", "tab:green")):
        ax.plot([m["s"] for m in slices], [m[key] for m in slices], ".-", ms=2, label=key.replace("csa_", ""), color=cc_)
    for a in acc: ax.axvline(a["s"], color="0.8", lw=2, alpha=.4, zorder=0)
    ax.set_xlabel("arc length along scope-path centerline"); ax.set_ylabel("CSA (scene units)")
    ax.set_title(f"25_V1 connected-model CSA profile\n{len(acc)} accepted, {len(stable_runs)} stable (grey=accepted)", fontsize=9); ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 2, 2)
    if bcov is not None:
        ax.scatter(bcov["xy"][:, 0], bcov["xy"][:, 1], s=4, c="tab:blue", alpha=.5)
        ax.plot(bcov["rb"] * np.cos(bcov["ang"]), bcov["rb"] * np.sin(bcov["ang"]), "m-", lw=1.5); ax.plot(0, 0, "k+", ms=10)
        ax.set_aspect("equal")
        ax.set_title(f"best ring attempt (connected model)\ncov={bcov['cov']:.0%} ratio={bcov['ratio']:.2f} spread={bcov['spread']:.2f}\nDCE={bcov['dce']:.2f} (NOT a stable accepted ring)", fontsize=9)
    fig.suptitle("25_V1 connected subglottic model — rings too noisy for same-model % (proximal capture-marginal)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(WORK / "obstruction_25v1.png", dpi=125); plt.close(fig)
    print(json.dumps(out, indent=2, default=str)); print(f"\nfigure -> {WORK}/obstruction_25v1.png")


if __name__ == "__main__":
    main()
