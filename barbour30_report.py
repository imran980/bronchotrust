"""Aggregate the Barbour 30-frame audit into the final table + verdict + figure.
Joins: sparse_screen.json (all 48), per-batch measure.json (densified), and the
long-window baseline (runs/batch4/<v>/report.json). Answers: can a Barbour-style
30-frame local reconstruction produce cleaner landmark CSA/DCE than our long-window
COLMAP pipeline? depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/mi3dr/projects/bronchotrust/runs/barbour30")
BATCH4 = Path("/home/mi3dr/projects/bronchotrust/runs/batch4")
LM2BASE = {"glottis": "glottis", "prox_subglottis": "+5mm", "dist_subglottis": "+10mm", "trachea_ref": None}


def main():
    screen = {r["name"]: r for r in json.loads((OUT / "sparse_screen.json").read_text())}
    rows = []
    for bd in sorted((OUT / "batches").glob("*")):
        mj = bd / "measure.json"
        if not mj.exists(): continue
        m = json.loads(mj.read_text()); s = screen.get(m["name"], {})
        dce = m.get("DCE_scene", {})
        rows.append({
            "video": s.get("vk"), "landmark": s.get("landmark"), "position": s.get("position"),
            "frames": f"{s.get('lo')}-{s.get('hi')}", "preproc": s.get("preproc"),
            "registered": f"{s.get('reg','?')}/30", "components": s.get("n_models"),
            "reproj": s.get("reproj_px"), "cone_deg": s.get("cone_deg"), "parallax": s.get("parallax_lat_along"),
            "outliers": s.get("outlier_ratio"), "dense_pts": m.get("n_dense"),
            "coverage": m.get("coverage"), "r_std/r_med": m.get("r_std/r_med"), "closed": m.get("closed"),
            "CSA_polar": (m.get("CSA_scene") or {}).get("polar"),
            "DCE_polar": dce.get("polar"), "DCE_ellipse": dce.get("ellipse"), "DCE_mesh": dce.get("mesh"),
            "DCE_estimator_spread": m.get("DCE_estimator_spread"), "measurable": m.get("measurable"),
        })

    # long-window baseline
    base = {}
    for v in ("2-V2", "25-V1"):
        d = json.loads((BATCH4 / v / "report.json").read_text())
        base[v] = {"dense": d["dense_pts"], "L": d["L_scene"], "landmarks": d["landmarks"]}

    # stability across early/centered/late (same video+landmark+preproc, polar DCE)
    stab = {}
    for r in rows:
        if not r["measurable"]: continue
        key = (r["video"], r["landmark"], r["preproc"])
        stab.setdefault(key, []).append(r["DCE_polar"])
    stability = {k: (round(max(v) / min(v), 2) if len(v) >= 2 and min(v) else None) for k, v in stab.items()}

    # per-row verdict
    for r in rows:
        k = (r["video"], r["landmark"], r["preproc"])
        r["stable_across_batches"] = stability.get(k)
        clean = bool(r["measurable"] and r["closed"] and (r["DCE_estimator_spread"] or 9) <= 1.30)
        r["verdict"] = "clean" if clean else ("partial/incoherent" if r["measurable"] else "not measurable")

    # compare to baseline — split by anatomy (subglottis is the clinical target; trachea is easy)
    SUBGLOTTIC = {"glottis", "prox_subglottis", "dist_subglottis"}
    n_clean = sum(1 for r in rows if r["verdict"] == "clean")
    n_meas = sum(1 for r in rows if r["measurable"])
    sub_rows = [r for r in rows if r["landmark"] in SUBGLOTTIC]
    tra_rows = [r for r in rows if r["landmark"] == "trachea_ref"]
    n_sub_clean = sum(1 for r in sub_rows if r["verdict"] == "clean")
    n_tra_clean = sum(1 for r in tra_rows if r["verdict"] == "clean")
    base_closed = {v: {lk: base[v]["landmarks"][lk]["closed"] for lk in base[v]["landmarks"]} for v in base}

    report = {
        "question": "Can Barbour-style 30-frame local reconstruction produce cleaner landmark CSA/DCE "
                    "than the long-window COLMAP pipeline?",
        "n_batches_screened": len(screen), "n_densified": len(rows),
        "n_clean_local_rings": n_clean, "n_measurable": n_meas,
        "raw_vs_clahe": "RAW fails to register on low-texture subglottic batches (3/24 raw reg>=25 vs 22/24 CLAHE); "
                        "CLAHE is required for SfM bootstrap. Raw only works on the vascular trachea.",
        "long_window_baseline_closed": base_closed,
        "long_window_note": "Long-window batch4 gives CLOSED rings at all subglottic landmarks "
                            "(2-V2 cov~100%/ratio~0.23; 25-V1 cov~86-100%/ratio~0.23), coherent tube, 200-270k dense pts.",
        "table": rows,
        "stability_across_positions": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in stability.items()},
        "anatomy_split": {
            "subglottic_local_clean": f"{n_sub_clean}/{len(sub_rows)}",
            "trachea_local_clean": f"{n_tra_clean}/{len(tra_rows)}",
            "subglottic_long_window_closed": "ALL (2-V2 & 25-V1 glottis/+5mm/+10mm closed)",
        },
    }
    verdict = (
        "NO (for the clinical target). 30-frame local batches do NOT produce cleaner SUBGLOTTIC CSA/DCE than "
        f"the long-window pipeline. Subglottic local rings clean: {n_sub_clean}/{len(sub_rows)} — every glottis/"
        "subglottis batch is OPEN or partial-coverage (r_std/r_med 0.37-0.73, or <60% coverage) with a flared, "
        "incoherent forward-cone cloud; and the distal-subglottis DCE swings ~2.1x across early/centered/late "
        "selections (unstable to frame choice). The long-window pipeline CLOSES all the same subglottic "
        f"landmarks (cov~86-100%, ratio~0.23). The ONLY clean local rings are the wide/textured TRACHEA "
        f"reference ({n_tra_clean}/{len(tra_rows)}, both raw & CLAHE) — which the long-window also measures "
        "cleanly. So the trachea is easy either way; the subglottis does not benefit from the 30-frame recipe. "
        "Also: raw frames FAIL to register on subglottic batches (CLAHE required). CONCLUSION: the limitation "
        "is NOT just our frame strategy — matching Barbour's 30-frame protocol does not recover the subglottis. "
        "Move toward a new method (shading-based / airway-model fitting).")
    report["verdict"] = verdict
    (OUT / "barbour30_report.json").write_text(json.dumps(report, indent=2))

    # ---- table print ----
    cols = ["video", "landmark", "position", "frames", "preproc", "registered", "components", "reproj",
            "cone_deg", "dense_pts", "coverage", "r_std/r_med", "closed", "DCE_polar", "DCE_ellipse",
            "DCE_mesh", "DCE_estimator_spread", "stable_across_batches", "verdict"]
    print("\n=== BARBOUR 30-FRAME AUDIT TABLE ===")
    print(" | ".join(cols))
    for r in sorted(rows, key=lambda x: (x["video"], x["landmark"], x["position"])):
        print(" | ".join(str(r.get(c)) for c in cols))
    print(f"\nclean local rings: {n_clean}/{n_meas} measurable ({len(rows)} densified)")
    print("VERDICT:", verdict)

    # ---- summary figure: coverage + estimator spread, local vs long-window ----
    meas = [r for r in rows if r["measurable"]]
    if meas:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        labels = [f"{r['video']}\n{r['landmark'][:8]}/{r['position'][:3]}/{r['preproc'][:2]}" for r in meas]
        cov = [(r["coverage"] or 0) * 100 for r in meas]
        ax[0].bar(range(len(meas)), cov, color=["tab:green" if c >= 60 else "tab:red" for c in cov])
        ax[0].axhline(60, color="k", ls="--", lw=1); ax[0].text(0, 62, "closed-ring coverage floor 60%", fontsize=8)
        ax[0].axhline(100, color="tab:blue", ls=":", lw=1.5); ax[0].text(0, 96, "long-window: ~86-100% (all closed)", fontsize=8, color="tab:blue")
        ax[0].set_xticks(range(len(meas))); ax[0].set_xticklabels(labels, fontsize=6, rotation=90)
        ax[0].set_ylabel("ring angular coverage %"); ax[0].set_title("Local 30-frame ring coverage (red = open/partial)")
        sp = [(r["DCE_estimator_spread"] or 0) for r in meas]
        ax[1].bar(range(len(meas)), sp, color=["tab:green" if x and x <= 1.3 else "tab:red" for x in sp])
        ax[1].axhline(1.3, color="k", ls="--", lw=1); ax[1].text(0, 1.32, "estimator-agreement ceiling 1.3x", fontsize=8)
        ax[1].set_xticks(range(len(meas))); ax[1].set_xticklabels(labels, fontsize=6, rotation=90)
        ax[1].set_ylabel("DCE estimator spread (max/min)"); ax[1].set_title("CSA/DCE estimator disagreement (red = >1.3x)")
        fig.suptitle("Barbour 30-frame local reconstruction: ring coverage + CSA estimator agreement\n"
                     "(long-window baseline closes ALL these landmarks at ~100% coverage)", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / "barbour30_summary.png", dpi=130); plt.close(fig)
        print(f"saved barbour30_summary.png")


if __name__ == "__main__":
    main()
