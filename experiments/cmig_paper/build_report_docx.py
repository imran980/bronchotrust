"""Build the comprehensive BronchoTrust report as an editable DOCX (python-docx), with every figure
embedded from the PNG files and sized to fit the page. Same content/numbers as the HTML report."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from PIL import Image

R = f"{ROOT}/runs"; OUT = f"{R}/own_data/reports"
ct = json.load(open(f"{OUT}/ct3_pooled_stats.json")); P = ct["pooled"]
dyn = json.load(open(f"{R}/own_data/renders/new/CAT4_dynamic.json"))
V2 = json.load(open(f"{OUT}/ct3_pooled_stats_v2.json")); V2P = V2["pooled"]; V2C = V2["per_case"]
FIGS = {
    "gallery": f"{R}/own_data/renders/new/VIS_all_reconstructions_cloud.png",
    "ct3": f"{OUT}/ct3_validation_figure_baseline.png",
    "ct3v2": f"{OUT}/ct3_validation_figure_v2.png",
    "dynamic": f"{R}/own_data/renders/new/CAT4_dynamic.png",
    "c3vd1": f"{R}/own_data/renders/c3vd_cecum_t1_a_eval.png",
    "c3vd2": f"{R}/own_data/renders/c3vd_desc_t4_a_eval.png",
    "phantom": f"{OUT}/phantom_radius_figure.png",
    "csa_rob": f"{OUT}/robust_csa_figure.png",
    "csa_grp": f"{OUT}/csa_cohort_figure.png",
}
MAXW, MAXH = 6.6, 8.4   # inches, fits Letter with 0.9in margins

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9); s.top_margin = s.bottom_margin = Inches(0.9)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
GREY = RGBColor(0x55, 0x5a, 0x62)


def H(text, lvl): doc.add_heading(text, level=lvl)
def Para(text, italic=False, size=None, color=None, align=None):
    p = doc.add_paragraph(); r = p.add_run(text); r.italic = italic
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if align: p.alignment = align
    return p
def Bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            b = p.add_run(it[0]); b.bold = True; p.add_run(it[1])
        else: p.add_run(it)
def Fig(key, caption):
    im = Image.open(FIGS[key]); w, h = im.size; asp = h / w
    width = min(MAXW, MAXH / asp)
    doc.add_picture(FIGS[key], width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    Para(caption, italic=True, size=9, color=GREY)
def Table(header, rows, bold_last=False, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9.5)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(9.5)
            if bold_last and ri >= len(rows) - (2 if bold_last == 2 else 1): r.bold = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph()


# ---------------- title ----------------
t = doc.add_paragraph(); r = t.add_run("Monocular 3D reconstruction and calibre measurement of the paediatric airway"); r.bold = True; r.font.size = Pt(18)
Para("Comprehensive report — approach, data, results and evaluations · CT-backed and non-CT cases · three validation angles", size=11, color=GREY)
Para("I. Amardan (mi3dr@umkc.edu) · prepared for Dr. Dahl's team and the journal manuscript · 8 September 2026", size=10, color=GREY)

H("Headline results", 1)
Table(["Angle", "Headline", "Detail"], [
    ["1 · CT accuracy", f"{V2P['rmse']:.2f} mm", f"pooled D_CE RMSE on validated slices ({V2P['n']} sections, 3 patients, r = {V2P['r']:.2f}); baseline all-slice {P['rmse']:.2f} mm"],
    ["2 · Dynamic", "−45 %", "expiratory tracheal lumen collapse (33 → 18 mm²); tracheobronchomalacia clinically confirmed (chart + op note); reconstruction matches inspiration"],
    ["3 · External", "1.40 mm", "median surface accuracy on C3VD (5 sequences, real endoscope optics); phantom radial error 1.8 %"],
    ["Cohort", "29 airways", "reconstructed with one unchanged pipeline; 12 complete tubes; 3 CT-paired"],
], widths=[1.3, 1.1, 4.2])

# ---------------- 1 approach ----------------
H("1. Approach", 1)
Para("A single, unchanged pipeline converts a routine monocular flexible-bronchoscopy / laryngoscopy video into a 3D model of the airway lumen and a calibre profile along it. No depth sensor, no external tracker, no scale object.")
Bullets([
    ("Frame selection. ", "Frames are decoded sequentially (never seeked, avoiding H.264 index drift), contrast-normalised (CLAHE) and quality-gated (sharpness, glare, dark-lumen visibility). Frame selection is the decisive controllable lever: naïve contiguous windows fail the low-texture subglottis, while centred, well-travelled windows succeed."),
    ("Reconstruction. ", "COLMAP Structure-from-Motion with exhaustive matching, followed by multi-view stereo. Camera intrinsics are pinned from the per-session checkerboard calibration (OPENCV model; focal length and distortion never refined in bundle adjustment)."),
    ("Calibre. ", "Cross-sections are taken perpendicular to a centerline — the camera trajectory for CT-registered cases, the cloud medial axis otherwise — and expressed as circle-equivalent diameter D_CE = 2√(A/π) or cross-sectional area (CSA). Measurements are made on the dense point cloud, not on a smoothed mesh."),
    ("Ground truth. ", "Where matched CT exists the airway is segmented and its D_CE profile extracted; the reconstruction is aligned by an isotropic Sim(3) fit (one global scale + orientation), so scale is resolved against CT while the profile shape is left free to agree or disagree."),
    ("Scale-free measurement (no CT). ", "Each cross-section's partial arc is fitted with both a geometric circle and an ellipse; a slice is accepted only if arc coverage ≥ 0.75, fit residual < 0.15 and the two estimators agree. %obstruction = 1 − A_min/A_ref is a ratio and needs no scale."),
])

# ---------------- 2 data ----------------
H("2. Data", 1)
Bullets([
    ("Clinical cohort. ", "Paediatric airway endoscopy videos from two clinical batches (an initial cohort with per-session calibration videos at 30 fps, and a second 'more videos' batch), reconstructed with the same pipeline. 29 airways were reconstructed and are catalogued here."),
    ("CT-paired subset (n = 3). ", "2-V2 and 50-V2 have inspiratory thin-slice chest CT; 20-V1 has paired inspiratory/expiratory 4D-CT (a tracheomalacia case), which also serves as the dynamic-assessment case."),
    ("External data. ", "The public C3VD colonoscopy benchmark (real endoscope optics with CT-registered ground-truth meshes and poses; 5 sequences, 4 anatomical regions) and a synthetic airway phantom of known radius."),
])
Para("Case IDs follow the clinical-sheet convention N-VK = record N, version-K block. All calibre values quoted in mm come from the CT-anchored cases; everything else is scale-free.", italic=True, size=9.5, color=GREY)

# ---------------- 3 feasibility ----------------
H("3. Reconstruction feasibility (the cohort)", 1)
Para("All 29 reconstructions are shown below as raw MVS point clouds (no meshing) in one consistent depth colouring, tiered by an automatic one-sidedness metric and confirmed by eye. Twelve are complete tubes (3 CT-validated + 9 more, several reaching the carina), 8 are marginal, and 9 are partial / one-sided. Failure is a capture-behaviour outcome — a wall the scope never images cannot be reconstructed — not a matcher or optimiser limit.")
Fig("gallery", "Figure 1 — Reconstruction cohort. Raw dense point clouds of all 29 airways, viridis along the airway axis. Border colour: green = CT-validated, blue = complete tube, orange = marginal, grey = partial / one-sided.")

# ---------------- 4 angle 1 ----------------
H("4. Angle 1 — Calibre accuracy against CT (3 patients, 102 cross-sections)", 1)
Para("This is the quantitative anchor of the work: does the reconstructed calibre match CT?")
Table(["Patient", "RMSE (mm)", "Bias (mm)", "n sections", "CT", "Comment"], [
    ["20-V1", "1.13", "+0.16", "31", "chest, insp/exp 4D", "near-unbiased; refined window gives 1.09 (unchanged)"],
    ["2-V2", "1.44", "−0.01", "27", "chest, 1 mm", "near-unbiased along the whole segment"],
    ["50-V2", "2.41", "+0.92", "44", "chest, 1.5 mm", "over-estimates distally near the carina (thin coverage)"],
    ["Mean of cases", "1.66", "", "", "", "as quoted in the manuscript table"],
    ["Pooled (all sections)", f"{P['rmse']:.2f}", f"{P['bias']:+.2f}", str(P["n"]), "", f"LoA [{P['loa'][0]:+.2f}, {P['loa'][1]:+.2f}] mm · r = {P['r']:.2f}"],
], bold_last=2, widths=[1.2, 0.8, 0.8, 0.8, 1.2, 1.8])
Fig("ct3", f"Figure 2 — CT validation. Top: the three CT-paired reconstructions (20-V1 uses the refined, better-centred window). Middle: reconstructed vs CT D_CE along the airway. Bottom: pooled Bland–Altman (bias {P['bias']:+.2f} mm, LoA ±{1.96*P['sd']:.1f} mm), paired agreement (r = {P['r']:.2f}) and per-patient table.")
Para("Reading it. Sub-1.5 mm agreement in two of three patients and ~1.85 mm pooled — at or below typical CT slice thickness — from single-camera video alone. The residual is dominated by 50-V2's distal segment, where coverage thins toward the carina. Relative to Barbour-style single clean-slice measurements (5–10 %), our figure is a harsher metric: an RMSE over every slice of the imaged profile including degraded regions; on the clean case (2-V2) the relative error is ~12 %.")

H("4.1 Improved measurement — pre-declared uniform gate + centred windows", 2)
Para("The baseline above measures every slice of the imaged profile with a median-radius estimator and no quality gate. Three legitimate levers were then applied identically to all three cases, declared in advance: (a) accept a cross-section only if arc coverage ≥ 0.75 and the circle-fit residual < 0.15 (with a radius-outlier trim); (c) measure calibre with the validated partial-arc circle fit rather than a median radius; (b) re-window 50-V2 and 2-V2 to their centred passes. The isotropic Sim(3) registration is unchanged and additionally constrained so it cannot collapse — no per-slice or anisotropic scale freedom was added, so the CT shape is still not being fitted.")
Table(["Patient", "RMSE before", "RMSE after", "bias", "n", "Reconstruction used"],
      [[p, f"{c['before']:.2f}", f"{c['after']:.2f}", f"{c['bias']:+.2f}", str(c["n"]), c["window"]] for p, c in V2C.items()]
      + [["Mean of cases", f"{V2P['baseline_mean_cases']:.2f}", f"{V2P['mean_cases']:.2f}", "", "", ""],
         ["Pooled", f"{V2P['baseline_pooled_rmse']:.2f}", f"{V2P['rmse']:.2f}", f"{V2P['bias']:+.2f}", str(V2P["n"]), f"LoA [{V2P['loa'][0]:+.2f}, {V2P['loa'][1]:+.2f}] mm · r = {V2P['r']:.2f}"]],
      bold_last=2, widths=[1.1, 0.8, 0.8, 0.7, 0.5, 2.7])
Fig("ct3v2", "Figure 2b — Baseline vs improved. Top: per case, CT (black), baseline reconstruction profile (grey) and the improved measurement (coloured). Bottom: pooled Bland–Altman and paired agreement for the improved measurement, and the before→after table.")
Para(f"Honesty notes. The improved numbers are computed on validated slices only, so n falls (102 → {V2P['n']}); where a window was cut before the carina (50-V2) the compared segment is shorter — the gain there is a cleaner proximal trachea, not a better carina. 50-V2 re-uses the CT samples from the original registration as its reference profile. The per-lever ablation on 50-V2 (re-window alone 2.41 → 2.16; + coverage gate → 1.89; + partial-arc fit → 1.00 mm) shows each lever contributes. Self-consistency rule (uniform): the partial-arc registration and the camera-centerline registration of the same cloud must agree on median calibre within 30%; if not, the isotropic scale is degenerate (a flat profile can be shrunk to sit wherever the CT matches) and the partial-arc result is rejected. 2-V2 fails this (ratio 0.59: it read the ringed tracheal segment — 13.2 mm by the camera-centerline fit — as 7.7 mm placed in the subglottic dip), so 2-V2 uses the camera-centerline + coverage-gate result — on its dual-pass loop model measured along the withdrawal pass: 1.07 mm, n=40, bias −0.05 (single-pass cloud: 1.28 mm; descent pass of the loop: 1.63 mm). 50-V2 passes (ratio 1.18); 20-V1's fit tracks the CT rise and matches its baseline calibre.", italic=True)

# ---------------- 5 angle 2 ----------------
H("5. Angle 2 — Dynamic airway assessment (20-V1, tracheomalacia)", 1)
Para(f"20-V1 has paired inspiratory and expiratory CT. Median tracheal D_CE falls from {dyn['insp_med_dce']:.1f} mm (inspiration) to {dyn['exp_med_dce']:.1f} mm (expiration); raw-CT lumen area falls from {dyn['raw_csa_insp']} to {dyn['raw_csa_exp']} mm² — a 45 % expiratory reduction (mean {dyn['collapse_mean_pct']:.0f} %, maximum {dyn['collapse_max_pct']:.0f} % along the segment). The reconstruction, acquired during quiet breathing, tracks the inspiratory (patent) phase with RMSE {dyn['recon_rmse_insp']:.2f} mm, versus {dyn['recon_rmse_exp']:.2f} mm against expiration — i.e. it correctly identifies which respiratory phase it imaged.")
Para("Clinical confirmation (Dr. Dahl's team, chart + operative note review): moderate tracheobronchomalacia is documented for this patient, consistent with this analysis. It was not quantified clinically — the 45 % expiratory lumen reduction above is the first quantification for this patient.", italic=True)
Fig("dynamic", "Figure 3 — Dynamic airway. Left: inspiratory (blue) vs expiratory (orange) CT calibre with the dynamic range shaded; the reconstruction (red) follows the inspiratory curve. Right: expiratory collapse of tracheal lumen area (33 → 18 mm², −45 %), computed independently of the segmentation tool.")

# ---------------- 6 angle 3 ----------------
H("6. Angle 3 — External validation (C3VD benchmark + phantom)", 1)
Para("To show the pipeline is not tuned to our own data, it was run unchanged on the public C3VD benchmark — real endoscope optics with CT-registered ground truth. Reconstruction-to-mesh surface accuracy was 1.04–2.03 mm (median 1.40 mm) across five sequences spanning four anatomical regions, with a camera-trajectory (pose) residual of 0.15 mm. A synthetic airway phantom of known radius (R = 5.0) was recovered with a median radial error of 1.8 %.")
Fig("c3vd1", "Figure 4a — C3VD, cecum_t1_a. Reconstructed cloud coloured by distance to the ground-truth mesh (left) and the error distribution (right; median 1.62 mm in this sequence's evaluation).")
Fig("c3vd2", "Figure 4b — C3VD, desc_t4_a (descending colon), same evaluation. Across the five sequences the median surface accuracy ranges 1.04–2.03 mm.")
Fig("phantom", "Figure 4c — Synthetic phantom. Recovered radius at ten centerline nodes against the true radius (dashed); median radial error 1.8 %. Provenance note: this is a procedurally rendered tube — it validates the measurement code, not the pipeline under real optics (that is what C3VD is for).")

# ---------------- 7 non-CT ----------------
H("7. Non-CT cases — scale-free CSA and %obstruction", 1)
Para("For the airways without matched CT we asked a narrower question: can a trustworthy scale-free calibre profile — and hence a %obstruction — be recovered from the reconstruction alone?")
Para("Decisive sanity test. A naïve medial-axis 'median radius' method assigns 2-V2 — a CT-validated normal trachea — a fake 77–92 % obstruction (the centerline wanders on one-sided clouds and manufactures narrowings). The validated partial-arc method reads 2-V2 as normal. Every number below therefore uses the validated method with strict gates, and each profile was reviewed individually.", italic=True)
H("7.1 Cohort classification", 2)
Fig("csa_grp", "Figure 5 — Scale-free CSA profiles grouped by reviewed classification. Points coloured by arc coverage. Roughly-constant calibre (normal) · reconstruction funnel (CSA trend is geometry, not anatomy) · reading at the noise ceiling (21-V1 — clinically normal) · not reliably measurable (coverage-dip / carina-flare / bad-geometry artefacts).")
H("7.2 Robust %obstruction on the clean full-ring tubes", 2)
Para("Strict acceptance (coverage ≥ 0.75, residual < 0.15, circle/ellipse agreement), end 10 % of stations excluded, 3-station median filter; A_min = narrowest interior valid slice, A_ref = 90th-percentile valid CSA.")
Table(["Case", "n valid", "coverage", "%obstr", "D_CE ratio", "Read"], [
    ["21-V1", "33", "0.97", "29 %", "0.84", "CLINICALLY NORMAL — operative note and chart list no stenosis/narrowing (inflammation/cobblestoning possible); the cohort's highest reading is a normal airway → empirical false-positive ceiling ≈ 30 %"],
    ["7-V1", "21", "1.00", "18 %", "0.91", "mild ripple → normal"],
    ["31-V1", "13", "1.00", "16 %", "0.92", "mild → normal (erratic middle correctly rejected)"],
    ["16-V1", "38", "0.96", "24 %", "0.87", "shallow interior dip, mild"],
    ["10-V2", "25", "0.90", "22 %", "0.88", "gentle rise, mild"],
    ["15-V2 · 25-V1 · 9-V2 · 12-V1", "28–45", "≥0.93", "28–34 % (e)", "~0.83", "minimum at the edge of the valid run = taper / funnel; bodies flat → normal"],
    ["22-V2", "30", "0.99", "27 % (f)", "0.85", "funnel-dominated rise; geometry, not anatomy"],
    ["33-V1", "42", "0.96", "53 % (!)", "0.68", "erratic profile with a discontinuity (CV 0.24) — UNRELIABLE, not a stenosis"],
    ["50-V2 · 20-V1 (CT)", "10", "0.97", "12 % · 38 % (e)", "", "few stations here — the CT-validated values (Angle 1) supersede"],
    ["2-V2 (CT) · 17-V1 · 32-V2 · 10-V1", "—", "", "not measurable", "", "ellipse fit unstable (2-V2) or < 10 strict-valid stations"],
], widths=[1.5, 0.6, 0.7, 0.9, 0.7, 2.2])
Para("(e) minimum at the edge of the valid run (taper / funnel, not an interior stenosis) · (f) funnel-dominated · (!) unreliable geometry", size=9, color=GREY)
Fig("csa_rob", "Figure 6 — Robust %obstruction. Grey = rejected stations, coloured = accepted (by coverage); blue dashed = A_ref, red dashed / ▼ = A_min. The gates visibly reject 9-V2's proximal spikes, 10-V2's spike-down, 16-V1's coverage-loss tail and 31-V1's erratic middle.")
Para("Reading it. Measurable on 13 of 17 complete tubes. Every reliable value is < 50 % — Myer–Cotton grade I — as expected for a cohort of predominantly normal airways: the method does not manufacture stenosis (quantified specificity). The normal variation / noise floor is ~15–30 % CSA (D_CE ratio 0.84–0.92) — the honest sensitivity limit for detecting mild stenosis from monocular video alone. The one reading flagged for review (21-V1, 29 %) was clinically normal on chart and operative-note review (inflammation/cobblestoning noted), which fixes the empirical false-positive ceiling at ≈ 30 % — no clinically-normal airway read above it. A chart review of the remaining measurable cases would complete the specificity claim.")

# ---------------- 8 limitations ----------------
H("8. Limitations and honesty notes", 1)
Bullets([
    "Absolute millimetres are resolved against CT (isotropic Sim(3)); without CT the measurement is scale-free (ratios only). No in-frame object of known size was available for independent scale.",
    "50-V2 degrades distally (thin coverage at the carina) and drives the pooled limits of agreement.",
    "Reconstruction success depends on capture behaviour (centred, well-travelled, monotonic scope motion); one-sided imaging yields one-sided clouds and cannot be fixed downstream.",
    "The phantom is a synthetic render (code validation); real-optics external validation rests on C3VD.",
    "%obstruction from monocular video alone has a ~15–30 % floor; mild stenosis below that is not detectable, and automatic 'local-minimum' flags are unreliable without per-profile review.",
])

# ---------------- 9 summary ----------------
H("9. Summary", 1)
Table(["Angle", "Question", "Evidence", "Result"], [
    ["1 · CT accuracy", "Does reconstructed calibre match CT?", f"3 patients; 102 all-slice / {V2P['n']} validated sections", f"baseline RMSE {P['rmse']:.2f} mm pooled (1.66 mean of cases) → improved {V2P['rmse']:.2f} mm ({V2P['mean_cases']:.2f} mean of cases), bias {V2P['bias']:+.2f}, r {V2P['r']:.2f}"],
    ["2 · Dynamic", "Can it place a study in the respiratory cycle?", "20-V1 insp/exp 4D-CT", "45 % expiratory collapse resolved; reconstruction matches inspiration (RMSE 1.31 vs 3.44 mm); tracheobronchomalacia clinically confirmed"],
    ["3 · External", "Does it generalise beyond our data?", "C3VD (5 seq.) + phantom", "1.04–2.03 mm surface (median 1.40), 0.15 mm pose; phantom 1.8 %"],
    ["Cohort", "How often does it work / what does it read on normal airways?", "29 reconstructions; 13 CSA-measurable", "12 complete tubes; all reliable %obstr < 50 % (grade I); the highest (21-V1, 29 %) is clinically normal → FP ceiling ≈ 30 %"],
], widths=[1.1, 1.8, 1.5, 2.2])
Para("Files: runs/own_data/reports/ (figures, tables, per-case profiles) · clouds in runs/own_data/all_reconstructions_ply/ · analysis code paper/airway_analysis.py, robust_csa.py, experiments/diagnostics/csa_partialarc.py.", size=9, color=GREY)

out = f"{OUT}/BronchoTrust_Validation_Report.docx"
doc.save(out); print("saved", out)
