"""2_V2 scale-free % obstruction, SAME global dense model only.

One continuous airway medial-axis centerline through the 2_V2 subglottis (global dense cloud
runs/batch4/2-V2/dense0/fused.ply, frames 875-1045), sliced perpendicular to the local tangent,
CSA measured 3 ways (polar polygon / ellipse / convex-hull). Both the DISTAL reference ring and the
PROXIMAL narrowest ring are measured on this ONE centerline -> one scene-unit scale (NOT local-batch
DCE, NOT mixed scales). Accept a slice only if coverage>=0.75 AND estimators agree <=1.3x AND
r_std/r_med<=0.35.

  CSA obstruction     = 1 - CSA_narrow / CSA_ref
  diameter narrowing  = 1 - DCE_narrow / DCE_ref

Scene units only; scale-free within-video; NOT absolute mm; NOT a final Myer-Cotton grade.
Rejects the proximal measurement if coverage/estimator agreement fails. depth-eval env.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import centerline_csa as cc

ROOT = Path("/home/mi3dr/projects/bronchotrust")
OUT = ROOT / "runs/barbour30/airwayfit/obstruction_2v2"
MODEL = "runs/batch4/2-V2/sparse/0"; CLOUD = "runs/batch4/2-V2/dense0/fused.ply"
RNG = (875, 1045)                       # proximal subglottis (875-945) .. distal subglottis (985-1045)
DISTAL_FRAMES = (985, 1045); PROX_FRAMES = (875, 945)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = cc.clean_cloud(CLOUD); cams, axz = cc.region_cams(MODEL, RNG)
    axis0 = axz.mean(0); axis0 /= np.linalg.norm(axis0)
    Cl, Tg, seg = cc.medial_centerline(P, cams, axis0)
    assert Cl is not None, "centerline failed"
    # per-centerline-sample axial coord (distal = larger, since axis0 = mean optical axis points distally)
    cam_c = cams.mean(0); u_cl = (Cl - cam_c) @ axis0
    # camera axial coords for prox / distal windows -> map centerline position to anatomy
    def cam_axial(rng):
        c, _ = cc.region_cams(MODEL, rng); return ((c - cam_c) @ axis0)
    u_prox = cam_axial(PROX_FRAMES); u_dist = cam_axial(DISTAL_FRAMES)
    slices = []
    for i in range(8, 192, 2):
        s = cc.slice_csa(seg, Cl[i], Tg[i])
        if s: s["i"] = i; s["p"] = Cl[i]; s["u"] = float(u_cl[i]); slices.append(s)
    acc = [s for s in slices if s["cov"] >= cc.COV_HIGH and s["spread"] <= cc.SPREAD_MAX and s["ratio"] <= 0.35]
    acc.sort(key=lambda s: s["u"])
    # NARROWEST = min-CSA accepted slice (proximal-ward: smaller u). REFERENCE = distal plateau
    # (accepted slices in the distal 40% of the accepted u-range). Both in the SAME global model.
    au = np.array([s["u"] for s in acc]) if acc else np.array([])
    narrow = min(acc, key=lambda s: s["csa"]) if acc else None
    plateau = [s for s in acc if s["u"] >= np.percentile(au, 60)] if acc else []
    ref = min(plateau, key=lambda s: abs(s["dce"] - np.median([q["dce"] for q in plateau]))) if plateau else None
    # is the narrowest a genuine interior local-min, or the proximal EDGE of clean reconstruction?
    edge_limited = bool(narrow and au.size and (narrow["u"] - au.min()) <= 0.10 * (au.max() - au.min()))

    def stab(center, wu=0.6):
        near = [s for s in acc if abs(s["u"] - center["u"]) <= wu]
        d = [s["dce"] for s in near]; return ([round(min(d), 3), round(max(d), 3)],
               round(100 * (max(d) - min(d)) / max(np.median(d), 1e-9), 1)) if len(d) >= 2 else (None, None)

    out = dict(video="2_V2", model="same global dense model (runs/batch4/2-V2/dense0/fused.ply)",
               centerline="one continuous cloud medial-axis (frames 875-1045)", units="scene units (scale-free, within-video)",
               n_slices=len(slices), n_accepted=len(acc))
    if ref:
        b, p = stab(ref)
        out["distal_reference"] = dict(u=round(ref["u"], 2), coverage=round(ref["cov"], 2), r_std_over_r_med=round(ref["ratio"], 3),
            CSA=round(ref["csa"], 4), DCE=round(ref["dce"], 4), estimator_spread=round(ref["spread"], 3),
            csa_polar=round(ref["csa_polar"], 4), csa_ellipse=round(ref["csa_ell"], 4), csa_hull=round(ref["csa_hull"], 4),
            stability_band_DCE=b, stability_pct=p)
    if narrow:
        b, p = stab(narrow)
        out["proximal_narrowest"] = dict(u=round(narrow["u"], 2), coverage=round(narrow["cov"], 2), r_std_over_r_med=round(narrow["ratio"], 3),
            CSA=round(narrow["csa"], 4), DCE=round(narrow["dce"], 4), estimator_spread=round(narrow["spread"], 3),
            csa_polar=round(narrow["csa_polar"], 4), csa_ellipse=round(narrow["csa_ell"], 4), csa_hull=round(narrow["csa_hull"], 4),
            stability_band_DCE=b, stability_pct=p, edge_limited=edge_limited)
    if ref and narrow:
        out["CSA_obstruction_pct"] = round(100 * (1 - narrow["csa"] / ref["csa"]), 1)
        out["diameter_narrowing_pct"] = round(100 * (1 - narrow["dce"] / ref["dce"]), 1)
        cn = [s["csa"] for s in acc if abs(s["u"] - narrow["u"]) <= 0.6]
        cr = [s["csa"] for s in plateau]
        lo = 100 * (1 - max(cn) / min(cr)); hi = 100 * (1 - min(cn) / max(cr))
        out["CSA_obstruction_band_pct"] = [round(min(lo, hi), 1), round(max(lo, hi), 1)]
        out["is_lower_bound"] = edge_limited
        out["verdict"] = ("ACCEPT (LOWER BOUND): scale-free % obstruction from same-model accepted rings; "
                          "narrowest is at the PROXIMAL EDGE of clean reconstruction (monotonic profile) so the true "
                          "subglottic throat is more proximal/unreconstructed -> real obstruction >= this."
                          if edge_limited else
                          "ACCEPT: scale-free % obstruction from same-model accepted rings (interior narrowest)")
    else:
        out["verdict"] = "REJECT: reference and/or narrowest not accepted"

    # figure
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sub = seg[np.random.default_rng(0).choice(len(seg), min(5000, len(seg)), replace=False)]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="0.6", alpha=.25)
    ax.plot(Cl[:, 0], Cl[:, 1], Cl[:, 2], "b-", lw=2, label="medial centerline")
    for s, col, lab in ((ref, "r", "distal ref"), (narrow, "m", "prox narrowest")):
        if s:
            t, e1, e2 = cc.basis(Tg[s["i"]]); th = np.linspace(0, 2 * np.pi, 80)
            ring = s["p"][None] + s["r_med"] * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], col + "-", lw=2, label=lab)
    ax.legend(fontsize=7); ax.set_title("2_V2 subglottis — one global centerline", fontsize=9)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for k, (s, col, ttl) in enumerate([(narrow, "m", "proximal narrowest"), (ref, "r", "distal reference")]):
        ax = fig.add_subplot(1, 3, 2 + k)
        if s:
            ax.scatter(s["xy"][:, 0], s["xy"][:, 1], s=3, c="tab:blue", alpha=.5)
            ax.plot(s["rb"] * np.cos(s["ang"]), s["rb"] * np.sin(s["ang"]), col + "-", lw=1.5)
            ax.plot(0, 0, "k+", ms=10); ax.set_aspect("equal")
            ax.set_title(f"{ttl}\ncov={s['cov']:.0%} ratio={s['ratio']:.2f} DCE={s['dce']:.2f}\nspread={s['spread']:.2f}", fontsize=8)
        else:
            ax.text(.5, .5, f"{ttl}\nREJECTED", ha="center", va="center"); ax.axis("off")
    fig.suptitle(f"2_V2 scale-free % obstruction (scene units; same global model) — {out.get('CSA_obstruction_pct','n/a')}% CSA", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "obstruction_rings.png", dpi=125); plt.close(fig)
    # CSA profile figure
    fig, ax = plt.subplots(figsize=(9, 4.2))
    us = [s["u"] for s in slices]
    ax.plot(us, [s["csa_polar"] for s in slices], ".-", ms=3, label="polar", color="tab:blue")
    ax.plot(us, [s["csa_ell"] for s in slices], ".-", ms=3, label="ellipse", color="tab:orange")
    ax.plot(us, [s["csa_hull"] for s in slices], ".-", ms=3, label="hull", color="tab:green")
    for s in acc: ax.axvline(s["u"], color="0.85", lw=3, alpha=.4, zorder=0)
    if narrow: ax.axvline(narrow["u"], color="m", lw=2, label=f"narrowest DCE={narrow['dce']:.2f} (proximal edge)")
    if ref: ax.axvline(ref["u"], color="r", lw=2, label=f"distal ref DCE={ref['dce']:.2f}")
    ax.set_xlabel("axial position along centerline (scene units; distal ->)"); ax.set_ylabel("CSA (scene units)")
    ax.set_title("2_V2 CSA profile along one global centerline (grey = accepted slices)"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "obstruction_profile.png", dpi=125); plt.close(fig)

    (OUT / "report.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k in ("distal_reference", "proximal_narrowest")}, indent=2, default=str))
    print(f"\nfigures -> {OUT}/obstruction_rings.png , obstruction_profile.png")


if __name__ == "__main__":
    main()
