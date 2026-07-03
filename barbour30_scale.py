"""Metric-scale determination for the accepted smart local-batch rings.
Critical rule: each local 30-frame SfM batch has its OWN arbitrary (gauge-free) scale.
mm is valid ONLY if a known-size object is in the SAME model, or scale is transferred
from a global model (containing the object) via Sim(3). Evidence gathered:
  - Parsons laryngoscope aperture (13x15 mm) NOT visible in 2-V2 / 25-V1 / 32-V2:
    entry frames show only laryngeal ANATOMY (epiglottis, vocal cords) — see
    runs/barbour30/scale/approach_*.png, zoom_*.png; quantitative metallic scan finds no
    persistent large low-saturation (metal) blob (2-V2 max blob 4.4%, 25-V1 6.6%; 32-V2's
    bright frames are deep specular glare, not an aperture — check_32v2_deep.png).
  - Hopkins telescope images FORWARD -> its own 4 mm shaft is never in frame; OD/length
    cannot be an in-image scale object (rule 6).
  - '4 cm' OD is a typo -> 4 mm (a 40 mm scope exceeds the trachea; CT gave 2-6 mm DCE).
=> No connected scale bridge exists -> absolute mm NOT valid; DCE stays in scene units.
Emits the required per-ring table + scale_report.json. depth-eval env."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("/home/mi3dr/projects/bronchotrust/runs/barbour30")

PHYS = {"parsons_size2_aperture_vertical_mm": 13.0, "parsons_size2_aperture_horizontal_mm": 15.0,
        "parsons_size2_length_mm": 90.0, "hopkins_OD_mm": 4.0, "hopkins_working_length_mm": 300.0,
        "hopkins_OD_4cm_is_typo": True}

# CORRECTED (user caught the earlier error): metal IS present in the cohort. Ruler chosen by
# user = the visible Hopkins telescope SHAFT outer diameter (4 mm); do NOT use the tube inner
# bore unless measured; do NOT use Parsons 13x15 unless the slot is visible+reconstructed in
# the same model. Metal presence (improved grey/low-sat detector + visual):
METAL_PRESENT = {
    "2-V2": "4mm rod/shaft at f8-45 (thin, specular, MOVING)",
    "5_v1_2": "large metal TUBE bore (scope inside it)", "10_v2": "metal TUBE bore",
    "13_v2": "metal TUBE (white-out)", "15_v2": "metal TUBE/aperture rim",
    "5_v1_1": "thin rod", "7-V1": "thin bright rim",
    "25-V1": "none usable", "32-V2": "none usable"}


def main():
    rows = json.loads((OUT / "pipeline_report.json").read_text())
    accepted = [r for r in rows if r.get("verdict") == "ACCEPT"]
    out_rows = []
    for r in accepted:
        v, lm = r["video"], r["landmark"]
        name = f"{v}_{lm}_pipe_p{r['chosen_pool']}"
        mj = OUT / "batches" / name / "measure.json"
        csa_scene = None
        if mj.exists():
            m = json.loads(mj.read_text()); csa_scene = (m.get("CSA_scene") or {}).get("polar")
        if lm == "glottis":
            note = "glottis is an aperture (cord plane), not a tube ring"
        elif v == "2-V2":
            note = "4mm rod IS visible but MOVING+specular -> not reconstructable as a cylinder; no anchor"
        else:
            note = "no rod/blade anchor in the airway-connected model -> scene units"
        out_rows.append({
            "video": v, "landmark": lm, "local_DCE_scene": r.get("DCE_polar"),
            "local_CSA_scene": csa_scene, "scale_source": "unavailable",
            "scale_mm_per_unit": None, "DCE_mm": None, "CSA_mm2": None,
            "scale_validity": "NOT VALID (no static known-size object in a connected stable model)",
            "notes": note})
    scale_source_breakdown = {
        "direct_blade_in_same_model": 0, "global_to_local_Sim3_transfer": 0,
        "unavailable": len(out_rows)}
    report = {
        "physical_dimensions_mm": PHYS,
        "CORRECTION": "Earlier claim 'no metal visible' was WRONG (user caught it). Metal IS present across the "
                      "cohort — see METAL_PRESENT. My first metallic scan thresholded bright-white specular "
                      "(V>230) and missed dull-grey metal; corrected detector (low-sat grey S<55 & V>135) finds "
                      "large metal TUBES in 5_v1_2/10_v2/13_v2/15_v2 and a thin 4mm rod in 2-V2.",
        "chosen_ruler": "Visible Hopkins telescope SHAFT outer diameter = 4 mm (user). Do NOT use tube inner "
                        "bore unless measured; do NOT use Parsons 13x15 unless the slot is visible+reconstructed "
                        "in the same model. '4 cm' OD confirmed a TYPO -> 4 mm.",
        "metal_present": METAL_PRESENT,
        "scale_attempts": {
            "2-V2_rod_reconstructability": "FAILS. The rod-visible segment f0-60 reconstructs (61/61, 1 model) "
                "but the 4mm rod does NOT form a fittable cylinder: metallic (grey) points are scattered through "
                "the tissue (radial IQR/median 1.52; a rod shell would be ~0.1-0.3). ROOT CAUSE: the rod is a "
                "MOVING instrument (metal centroid sweeps cy 0.44->0.75, area 1%->7.5% over f5-45) -> SfM cannot "
                "reconstruct a non-static object, and it is specular. => the 4mm OD cannot be measured.",
            "2-V2_global_bridge": "MOOT. Even if a connected rod->subglottis model existed, the moving/specular "
                "rod is not measurable, so no 4mm anchor can be fit. (Bridge would also have to span ~880 frames "
                "through the low-parallax cords, which fragments — see prior audits.) -> scale UNAVAILABLE.",
            "tube_videos (5_v1_2/10_v2/13_v2/15_v2)": "the visible metal there is the tube INNER BORE, not the "
                "4mm Hopkins shaft; user said do NOT use the bore unless its diameter is measured. A 10_v2 "
                "tube->airway test fragmented (40/91, 3 models) and the points do not form a clean cylinder. "
                "Not usable as the 4mm ruler.",
            "25-V1 / 32-V2": "no scale anchor (rod/blade) appears in the airway-connected model -> scene units.",
            "hopkins_working_length": "not used — the scope shaft is a moving instrument, not a static "
                "reconstructed object (rule 6).",
        },
        "scale_source_breakdown": scale_source_breakdown,
        "verdict": "ABSOLUTE mm STILL NOT VALID for any accepted ring — but now for the CORRECT reason: the 4mm "
                   "Hopkins shaft IS visible (2-V2) yet is a MOVING, specular instrument that SfM cannot "
                   "reconstruct into a fittable cylinder, so no 4mm anchor can be measured in a connected model; "
                   "25-V1/32-V2 have no anchor at all. Per the rule (no cross-batch/cross-video transfer; anchor "
                   "must be in ONE connected stable model), scale = unavailable -> all DCE/CSA stay SCENE UNITS. "
                   "A valid mm anchor would need a STATIC known-size object (a held-still shaft, or the Parsons "
                   "slot) imaged in the SAME connected pass as the airway ring.",
        "table": out_rows,
    }
    (OUT / "scale_report.json").write_text(json.dumps(report, indent=2))

    cols = ["video", "landmark", "local_DCE_scene", "scale_source", "scale_mm_per_unit", "DCE_mm",
            "CSA_mm2", "scale_validity", "notes"]
    print("\n=== METRIC SCALE — ACCEPTED RINGS ===")
    print(" | ".join(cols))
    for r in out_rows:
        print(" | ".join(str(r.get(c)) for c in cols))
    print("\nscale source breakdown:", json.dumps(scale_source_breakdown))
    print("Hopkins OD '4cm' -> TYPO, correct = 4 mm. Ruler = visible 4mm Hopkins SHAFT (user).")
    print("2-V2: 4mm rod IS visible but is a MOVING, specular instrument -> not reconstructable as a cylinder")
    print("      (metallic pts scattered, radial IQR/median 1.52; centroid sweeps) -> 4mm not measurable.")
    print("25-V1/32-V2: no rod/blade anchor in a connected model.")
    print("VERDICT: absolute mm NOT valid (no STATIC known-size object in a connected model) -> scene units.")
    print(f"report -> {OUT}/scale_report.json")


if __name__ == "__main__":
    main()
