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

BLADE_VISIBLE = {"2-V2": False, "25-V1": False, "32-V2": False}   # from visual + metallic scan


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
        note = ("glottis is an aperture (cord plane), not a tube ring" if lm == "glottis"
                else "monocular gauge-free; this local batch has its own arbitrary scale")
        out_rows.append({
            "video": v, "landmark": lm, "local_DCE_scene": r.get("DCE_polar"),
            "local_CSA_scene": csa_scene, "scale_source": "unavailable",
            "scale_mm_per_unit": None, "DCE_mm": None, "CSA_mm2": None,
            "scale_validity": "NOT VALID (no known-size object imaged; no Sim(3) bridge)",
            "notes": note})
    scale_source_breakdown = {
        "direct_blade_in_same_model": 0, "global_to_local_Sim3_transfer": 0,
        "unavailable": len(out_rows)}
    report = {
        "physical_dimensions_mm": PHYS,
        "hopkins_OD_confirmation": "'4 cm' is a TYPO -> 4 mm (40 mm exceeds the trachea; CT DCE was 2-6 mm). "
                                   "Telescope images forward, so its own shaft is never in frame -> OD/length "
                                   "cannot be used as an in-image scale object.",
        "blade_search": {
            "task1_frames_with_parsons_aperture": "NONE in 2-V2 / 25-V1 / 32-V2. Entry frames (video start->"
                "glottis) show laryngeal ANATOMY only (epiglottis = smooth pale curved mucosa; vocal cords = "
                "dark vertical slit); no rigid metal 13x15mm aperture rim. Quantitative metallic scan (bright "
                "V>230 & low-sat S<45): no persistent large blob (2-V2 max 4.4%, 25-V1 6.6%; 32-V2 bright "
                "frames are DEEP specular glare, not the entry aperture).",
            "task2_blade_airway_same_model": "N/A — no blade to reconstruct.",
            "task3_13_15mm_scale": "cannot apply — aperture not imaged.",
            "task4_global_Sim3_transfer": "no blade in any potential global model to anchor absolute mm; and a "
                "connected model spanning entry->deep subglottis fragments/collapses in these low-parallax "
                "videos (see 16v1-ct-validation, long-window audits) — so no absolute-mm bridge is available.",
            "task6_hopkins_shaft": "not visible/reconstructed (forward-imaging) -> working length NOT used.",
        },
        "scale_source_breakdown": scale_source_breakdown,
        "verdict": "ABSOLUTE mm NOT VALID for any accepted ring. The only candidate scale object (Parsons "
                   "aperture) is not imaged; the Hopkins telescope cannot self-image. All accepted DCE/CSA "
                   "remain in SCENE UNITS. NOTE: because each accepted ring is a SEPARATE local 30-frame batch "
                   "with its own gauge, the rings are not even mutually comparable in scale -> a Sim(3) bridge "
                   "to a common (long-window) model would be needed for scale-free %-obstruction (A_min/A_ref), "
                   "though that yields COMMON-RELATIVE scale, still not mm without a visible blade.",
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
    print("Hopkins OD '4cm' -> TYPO, correct = 4 mm. Telescope cannot self-image (rule 6 upheld).")
    print("VERDICT: absolute mm NOT valid (no blade imaged) -> DCE stays scene-units.")
    print(f"report -> {OUT}/scale_report.json")


if __name__ == "__main__":
    main()
