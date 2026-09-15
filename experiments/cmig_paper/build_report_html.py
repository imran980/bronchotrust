"""Assemble the comprehensive BronchoTrust report (approach, data, results, CT-backed, non-CT,
three validation angles) as a single self-contained HTML page with every figure embedded.
Generates the missing phantom chart, downsizes/JPEG-encodes all figures, writes bronchotrust_report.html."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json, base64, io
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from figstyle import apply_style, OKABE
apply_style()

R = f"{ROOT}/runs"
OUT = f"{R}/own_data/reports"

# ---------------- phantom chart (no figure existed) ----------------
ph = json.load(open(f"{R}/phantom/phantom_gt_report.json"))
rec = np.array(ph["lumen_axis"]["recovered_R_GTunits"]); gt = ph["R_GT"]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(1, len(rec) + 1)
ax.axhline(gt, color="0.45", ls="--", lw=1.4, zorder=1)
ax.text(len(rec) + 0.25, gt + 0.015, f"ground truth R = {gt:.1f}", va="bottom", fontsize=9, color="0.35")
ax.plot(x, rec, "-", color=OKABE["blue"], lw=1.6, zorder=2)
ax.scatter(x, rec, s=42, color=OKABE["blue"], zorder=3, edgecolor="white", linewidth=1.2)
for i in (int(np.argmin(rec)), int(np.argmax(rec))):
    ax.annotate(f"{rec[i]:.2f}", (x[i], rec[i]), textcoords="offset points", xytext=(0, 9 if rec[i] > gt else -14),
                ha="center", fontsize=8.5, color="0.25")
ax.set_ylim(gt * 0.9, gt * 1.1); ax.set_xlim(0.5, len(rec) + 2.4); ax.set_xticks(x)
ax.set_xlabel("centerline node along the tube"); ax.set_ylabel("recovered radius (GT units)")
ax.set_title(f"Synthetic phantom: recovered vs true radius — median radial error {ph['lumen_axis']['median_radial_error_pct']:.1f}%  (n={ph['n_common_cams']} cameras)", fontsize=10.5)
ax.grid(alpha=0.25, axis="y")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/phantom_radius_figure.png", dpi=150); plt.close(fig)
print("phantom chart saved")


# ---------------- figure embedding ----------------
def embed(path, max_w=1700, q=82):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, optimize=True)
    b = base64.b64encode(buf.getvalue()).decode()
    print(f"  {path.split('/')[-1]:45s} {len(buf.getvalue())//1024:5d} KB  {im.width}x{im.height}")
    return f"data:image/jpeg;base64,{b}"

FIG = {
    "gallery":  embed(f"{R}/own_data/renders/new/VIS_all_reconstructions_cloud.png", 1500, 80),
    "ct3":      embed(f"{OUT}/ct3_validation_figure_baseline.png"),
    "dynamic":  embed(f"{R}/own_data/renders/new/CAT4_dynamic.png"),
    "c3vd1":    embed(f"{R}/own_data/renders/c3vd_cecum_t1_a_eval.png", 1500),
    "c3vd2":    embed(f"{R}/own_data/renders/c3vd_desc_t4_a_eval.png", 1500),
    "phantom":  embed(f"{OUT}/phantom_radius_figure.png"),
    "csa_rob":  embed(f"{OUT}/robust_csa_figure.png"),
    "csa_grp":  embed(f"{OUT}/csa_cohort_figure.png", 1300, 80),
}
ct = json.load(open(f"{OUT}/ct3_pooled_stats.json"))
dyn = json.load(open(f"{R}/own_data/renders/new/CAT4_dynamic.json"))
P = ct["pooled"]
V2 = json.load(open(f"{OUT}/ct3_pooled_stats_v2.json")); V2P = V2["pooled"]; V2C = V2["per_case"]
FIG["ct3v2"] = embed(f"{OUT}/ct3_validation_figure_v2.png")
v2rows = "".join(f"<tr><td>{p}</td><td class=\"num\">{c['before']:.2f}</td><td class=\"num\">{c['after']:.2f}</td><td class=\"num\">{c['bias']:+.2f}</td><td class=\"num\">{c['n']}</td><td>{c['window']}</td></tr>" for p, c in V2C.items())
sec41 = f"""
<h3>4.1 Improved measurement — pre-declared uniform gate + centred windows</h3>
<p>The baseline above measures <i>every</i> slice of the imaged profile with a median-radius estimator and no quality gate. Three legitimate levers were then applied <b>identically to all three cases</b>, declared in advance: (a) accept a cross-section only if arc coverage ≥ 0.75 and the circle-fit residual &lt; 0.15 (with a radius-outlier trim), (c) measure calibre with the validated <b>partial-arc circle fit</b> rather than a median radius, and (b) re-window 50-V2 and 2-V2 to their <b>centred</b> passes. The isotropic Sim(3) registration is unchanged and additionally constrained so it cannot collapse — no per-slice or anisotropic scale freedom was added, so the CT shape is still not being fitted.</p>
<div class="tbl"><table>
<tr><th>Patient</th><th class="num">RMSE before</th><th class="num">RMSE after</th><th class="num">bias</th><th class="num">n</th><th>Reconstruction used</th></tr>
{v2rows}
<tr><td><b>Mean of cases</b></td><td class="num"><b>{V2P['baseline_mean_cases']:.2f}</b></td><td class="num"><b>{V2P['mean_cases']:.2f}</b></td><td></td><td></td><td></td></tr>
<tr><td><b>Pooled</b></td><td class="num"><b>{V2P['baseline_pooled_rmse']:.2f}</b></td><td class="num"><b>{V2P['rmse']:.2f}</b></td><td class="num"><b>{V2P['bias']:+.2f}</b></td><td class="num"><b>{V2P['n']}</b></td><td>LoA [{V2P['loa'][0]:+.2f}, {V2P['loa'][1]:+.2f}] mm · r = {V2P['r']:.2f}</td></tr>
</table></div>
<figure><img src="{FIG['ct3v2']}" alt="CT validation before vs after">
<figcaption><b>Figure 2b — Baseline vs improved.</b> Top: per case, CT (black), baseline reconstruction profile (grey) and the improved measurement (coloured). Bottom: pooled Bland–Altman and paired agreement for the improved measurement, and the before→after table.</figcaption></figure>
<div class="note"><b>Honesty notes.</b> The improved numbers are computed on <i>validated</i> slices only, so <b>n falls</b> (102 → {V2P['n']}); and where a window was cut before the carina (50-V2) the compared segment is shorter — the gain there is a cleaner proximal trachea, not a better carina. 50-V2 also re-uses the CT samples from the original registration as its reference profile. The per-lever ablation on 50-V2 (re-window alone 2.41 → 2.16; + coverage gate → 1.89; + partial-arc fit → 1.00 mm) shows each lever contributes. <b>Self-consistency rule (uniform):</b> the partial-arc registration and the camera-centerline registration of the <i>same</i> cloud must agree on median calibre within 30%; if not, the isotropic scale is degenerate (a flat profile can be shrunk to sit wherever the CT matches) and the partial-arc result is rejected. <b>2-V2 fails this</b> (ratio 0.59: it read the ringed tracheal segment — 13.2 mm by the camera-centerline fit — as 7.7 mm placed in the subglottic dip), so 2-V2 uses the camera-centerline + coverage-gate result — on its dual-pass loop model measured along the withdrawal pass: 1.07 mm, n=40, bias −0.05 (single-pass cloud: 1.28 mm; descent pass of the loop: 1.63 mm). 50-V2 passes (ratio 1.18); 20-V1's fit tracks the CT rise and matches its baseline calibre.</div>
"""

CSS = """
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1c1e21;--ink2:#4a4f57;--muted:#7a808a;--line:#e3e6ea;
--acc:#0072B2;--acc2:#009E73;--warn:#E69F00;--bad:#D55E00;--figbg:#ffffff;--chip:#eef2f6;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131518;--card:#1c1f24;--ink:#e8eaed;--ink2:#b9bec6;
--muted:#8a9099;--line:#2c3038;--acc:#56B4E9;--acc2:#3fcf9e;--warn:#F0C24B;--bad:#F08A4B;--figbg:#f4f5f7;--chip:#252a31;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#131518;--card:#1c1f24;--ink:#e8eaed;--ink2:#b9bec6;--muted:#8a9099;--line:#2c3038;
--acc:#56B4E9;--acc2:#3fcf9e;--warn:#F0C24B;--bad:#F08A4B;--figbg:#f4f5f7;--chip:#252a31;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:22px;margin:44px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:26px 0 8px;color:var(--ink)}p{margin:8px 0;color:var(--ink2)}
.sub{color:var(--muted);font-size:14px;margin-bottom:18px}
.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:18px 0 6px}
.toc a{color:var(--acc);text-decoration:none;margin-right:14px;white-space:nowrap}.toc a:hover{text-decoration:underline}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:26px;font-weight:700;margin:2px 0;color:var(--ink)}.tile .d{font-size:13px;color:var(--ink2)}
figure{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:18px 0}
figure img{display:block;width:100%;height:auto;background:var(--figbg);border-radius:6px}
figcaption{font-size:13.5px;color:var(--ink2);margin-top:10px;line-height:1.5}
figcaption b{color:var(--ink)}
.tbl{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:14px;min-width:560px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.note{border-left:3px solid var(--warn);background:var(--chip);padding:10px 14px;border-radius:6px;margin:14px 0;color:var(--ink2);font-size:14px}
.good{border-left-color:var(--acc2)}.bad{border-left-color:var(--bad)}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12.5px;color:var(--ink2);margin:2px 4px 2px 0}
.pill{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
ul{color:var(--ink2);margin:6px 0 6px 18px;padding:0}li{margin:4px 0}
.small{font-size:13px;color:var(--muted)}
@media (max-width:640px){h1{font-size:24px}.wrap{padding:18px 14px}}
"""

html = f"""<title>BronchoTrust Validation Report</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Monocular 3D reconstruction and calibre measurement of the paediatric airway</h1>
<div class="sub">Comprehensive report — approach, data, results and evaluations · CT-backed and non-CT cases · three validation angles<br>
I. Amardan (mi3dr@umkc.edu) · prepared for Dr. Dahl's team and the journal manuscript · 8 September 2026</div>

<div class="toc"><b>Contents</b><br>
<a href="#s1">1 Approach</a><a href="#s2">2 Data</a><a href="#s3">3 Reconstruction feasibility</a>
<a href="#s4">4 Angle 1 — CT accuracy</a><a href="#s5">5 Angle 2 — Dynamic</a><a href="#s6">6 Angle 3 — External</a>
<a href="#s7">7 Non-CT cases: CSA / %obstruction</a><a href="#s8">8 Limitations</a><a href="#s9">9 Summary</a></div>

<div class="tiles">
<div class="tile"><div class="k">Angle 1 · CT accuracy</div><div class="v">{V2P['rmse']:.2f} mm</div><div class="d">pooled D<sub>CE</sub> RMSE on validated slices ({V2P['n']} sections, 3 patients, r = {V2P['r']:.2f}); baseline all-slice {P['rmse']:.2f} mm</div></div>
<div class="tile"><div class="k">Angle 2 · Dynamic</div><div class="v">−45 %</div><div class="d">expiratory tracheal lumen collapse (33 → 18 mm²) — tracheobronchomalacia <b>clinically confirmed</b> (chart + op note)</div></div>
<div class="tile"><div class="k">Angle 3 · External</div><div class="v">1.40 mm</div><div class="d">median surface accuracy, C3VD (5 sequences) · phantom radial error 1.8 %</div></div>
<div class="tile"><div class="k">Cohort</div><div class="v">29</div><div class="d">reconstructed airways · 12 complete tubes · 3 CT-paired</div></div>
</div>

<h2 id="s1">1 · Approach</h2>
<p>A single, unchanged pipeline converts a routine <b>monocular</b> flexible-bronchoscopy / laryngoscopy video into a 3D model of the airway lumen and a calibre profile along it. No depth sensor, no external tracker, no scale object.</p>
<ul>
<li><b>Frame selection.</b> Frames are decoded sequentially (never seeked, to avoid H.264 index drift), contrast-normalised (CLAHE) and quality-gated (sharpness, glare, dark-lumen visibility). Frame <i>selection</i> is the decisive controllable lever: naïve contiguous windows fail the low-texture subglottis, while centred, well-travelled windows succeed.</li>
<li><b>Reconstruction.</b> COLMAP Structure-from-Motion with exhaustive matching, followed by multi-view stereo. Camera intrinsics are <b>pinned</b> from the per-session checkerboard calibration (OPENCV model; focal length and distortion never refined in bundle adjustment).</li>
<li><b>Calibre.</b> Cross-sections are taken perpendicular to a centerline — the camera trajectory for CT-registered cases, the cloud medial axis otherwise — and expressed as circle-equivalent diameter D<sub>CE</sub> = 2√(A/π) or cross-sectional area (CSA). Measurements are made on the dense point cloud, not on a smoothed mesh.</li>
<li><b>Ground truth.</b> Where matched CT exists the airway is segmented and its D<sub>CE</sub> profile extracted; the reconstruction is aligned by an <b>isotropic Sim(3)</b> fit (one global scale + orientation), so scale is resolved against CT while the profile <i>shape</i> is left free to agree or disagree.</li>
<li><b>Scale-free measurement</b> (no CT). Each cross-section's partial arc is fitted with both a geometric circle and an ellipse; a slice is accepted only if arc coverage ≥ 0.75, fit residual &lt; 0.15 and the two estimators agree. %obstruction = 1 − A<sub>min</sub>/A<sub>ref</sub> is a ratio and therefore needs no scale.</li>
</ul>

<h2 id="s2">2 · Data</h2>
<ul>
<li><b>Clinical cohort.</b> Paediatric airway endoscopy videos from two clinical batches (an initial cohort with per-session calibration videos at 30 fps, and a second "more videos" batch), reconstructed with the same pipeline. <b>29 airways</b> were reconstructed and are catalogued in this report.</li>
<li><b>CT-paired subset (n = 3).</b> 2-V2 and 50-V2 have inspiratory thin-slice chest CT; <b>20-V1</b> has paired inspiratory / expiratory 4D-CT (a tracheomalacia case), which also serves as the dynamic-assessment case.</li>
<li><b>External data.</b> The public <b>C3VD</b> colonoscopy benchmark (real endoscope optics with CT-registered ground-truth meshes and poses; 5 sequences, 4 anatomical regions) and a synthetic airway phantom of known radius.</li>
</ul>
<div class="note">Case IDs follow the clinical sheet convention <span class="chip">N-VK</span> = record N, version-K block. All calibre values quoted in mm come from the CT-anchored cases; everything else is scale-free.</div>

<h2 id="s3">3 · Reconstruction feasibility (the cohort)</h2>
<p>All 29 reconstructions are shown below as <b>raw MVS point clouds</b> (no meshing) in one consistent depth colouring, tiered by an automatic one-sidedness metric and confirmed by eye. Twelve are complete tubes (3 CT-validated + 9 more, several reaching the carina), 8 are marginal, and 9 are partial / one-sided. Failure is a <i>capture-behaviour</i> outcome — a wall the scope never images cannot be reconstructed — not a matcher or optimiser limit.</p>
<figure><img src="{FIG['gallery']}" alt="All 29 reconstructed airways as point clouds">
<figcaption><b>Figure 1 — Reconstruction cohort.</b> Raw dense point clouds of all 29 airways, viridis along the airway axis. Border: <span class="pill" style="background:#009E73"></span>CT-validated · <span class="pill" style="background:#0072B2"></span>complete tube · <span class="pill" style="background:#E69F00"></span>marginal · <span class="pill" style="background:#777"></span>partial / one-sided.</figcaption></figure>

<h2 id="s4">4 · Angle 1 — Calibre accuracy against CT (n = 3 patients, 102 cross-sections)</h2>
<p>This is the quantitative anchor of the work: does the reconstructed calibre match CT?</p>
<div class="tbl"><table>
<tr><th>Patient</th><th class="num">RMSE (mm)</th><th class="num">Bias (mm)</th><th class="num">n sections</th><th>CT</th><th>Comment</th></tr>
<tr><td>20-V1</td><td class="num">1.13</td><td class="num">+0.16</td><td class="num">31</td><td>chest, insp/exp 4D</td><td>near-unbiased; refined window gives 1.09 (unchanged)</td></tr>
<tr><td>2-V2</td><td class="num">1.44</td><td class="num">−0.01</td><td class="num">27</td><td>chest, 1 mm</td><td>near-unbiased along the whole segment</td></tr>
<tr><td>50-V2</td><td class="num">2.41</td><td class="num">+0.92</td><td class="num">44</td><td>chest, 1.5 mm</td><td>over-estimates distally near the carina (thin coverage)</td></tr>
<tr><td><b>Mean of cases</b></td><td class="num"><b>1.66</b></td><td></td><td></td><td></td><td>as quoted in the manuscript table</td></tr>
<tr><td><b>Pooled (all sections)</b></td><td class="num"><b>{P['rmse']:.2f}</b></td><td class="num"><b>{P['bias']:+.2f}</b></td><td class="num"><b>{P['n']}</b></td><td></td><td>LoA [{P['loa'][0]:+.2f}, {P['loa'][1]:+.2f}] mm · r = {P['r']:.2f}</td></tr>
</table></div>
<figure><img src="{FIG['ct3']}" alt="CT validation figure">
<figcaption><b>Figure 2 — CT validation.</b> Top: the three CT-paired reconstructions (20-V1 uses the refined, better-centred window). Middle: reconstructed vs CT D<sub>CE</sub> along the airway. Bottom: pooled Bland–Altman (bias {P['bias']:+.2f} mm, LoA ±{1.96*P['sd']:.1f} mm), paired agreement (r = {P['r']:.2f}) and per-patient table.</figcaption></figure>
<div class="note good"><b>Reading it.</b> Sub-1.5 mm agreement in two of three patients and ~1.85 mm pooled — at or below typical CT slice thickness — from single-camera video alone. The residual is dominated by 50-V2's distal segment, where coverage thins toward the carina. Relative to Barbour-style single clean-slice measurements (5–10 %), our figure is a harsher metric: an RMSE over <i>every</i> slice of the imaged profile including degraded regions; on the clean case (2-V2) the relative error is ~12 %.</div>

{sec41}
<h2 id="s5">5 · Angle 2 — Dynamic airway assessment (20-V1, tracheomalacia)</h2>
<p>20-V1 has paired inspiratory and expiratory CT. Median tracheal D<sub>CE</sub> falls from <b>{dyn['insp_med_dce']:.1f} mm</b> (inspiration) to <b>{dyn['exp_med_dce']:.1f} mm</b> (expiration); raw-CT lumen area falls from {dyn['raw_csa_insp']} to {dyn['raw_csa_exp']} mm² — a <b>45 % expiratory reduction</b> (mean {dyn['collapse_mean_pct']:.0f} %, maximum {dyn['collapse_max_pct']:.0f} % along the segment). The reconstruction, acquired during quiet breathing, tracks the <i>inspiratory</i> (patent) phase with RMSE {dyn['recon_rmse_insp']:.2f} mm, versus {dyn['recon_rmse_exp']:.2f} mm against expiration — i.e. it correctly identifies which respiratory phase it imaged.</p>
<div class="note good"><b>Clinical confirmation (Dr. Dahl's team, chart + operative note review):</b> moderate tracheobronchomalacia is documented for this patient, consistent with this analysis. It was <i>not</i> quantified clinically — the 45 % expiratory lumen reduction above is the first quantification for this patient.</div>
<figure><img src="{FIG['dynamic']}" alt="Dynamic airway figure">
<figcaption><b>Figure 3 — Dynamic airway.</b> Left: inspiratory (blue) vs expiratory (orange) CT calibre with the dynamic range shaded; the reconstruction (red) follows the inspiratory curve. Right: expiratory collapse of tracheal lumen area (33 → 18 mm², −45 %), computed independently of the segmentation tool.</figcaption></figure>

<h2 id="s6">6 · Angle 3 — External validation (C3VD benchmark + phantom)</h2>
<p>To show the pipeline is not tuned to our own data, it was run unchanged on the public <b>C3VD</b> benchmark — real endoscope optics with CT-registered ground truth. Reconstruction-to-mesh surface accuracy was <b>1.04–2.03 mm (median 1.40 mm)</b> across five sequences spanning four anatomical regions, with a camera-trajectory (pose) residual of <b>0.15 mm</b>. A synthetic airway phantom of known radius (R = 5.0) was recovered with a <b>median radial error of 1.8 %</b>.</p>
<figure><img src="{FIG['c3vd1']}" alt="C3VD cecum_t1_a evaluation">
<figcaption><b>Figure 4a — C3VD, cecum_t1_a.</b> Reconstructed cloud coloured by distance to the ground-truth mesh (left) and the error distribution (right; median 1.62 mm in this sequence's evaluation).</figcaption></figure>
<figure><img src="{FIG['c3vd2']}" alt="C3VD desc_t4_a evaluation">
<figcaption><b>Figure 4b — C3VD, desc_t4_a</b> (descending colon), same evaluation. Across the five sequences the median surface accuracy ranges 1.04–2.03 mm.</figcaption></figure>
<figure><img src="{FIG['phantom']}" alt="Phantom radius recovery">
<figcaption><b>Figure 4c — Synthetic phantom.</b> Recovered radius at ten centerline nodes against the true radius (dashed); median radial error 1.8 %. <b>Provenance note:</b> this is a procedurally rendered tube — it validates the measurement code, not the pipeline under real optics (that is what C3VD is for).</figcaption></figure>

<h2 id="s7">7 · Non-CT cases — scale-free CSA and %obstruction</h2>
<p>For the airways without matched CT we asked a narrower question: can a trustworthy <b>scale-free</b> calibre profile — and hence a %obstruction — be recovered from the reconstruction alone?</p>
<div class="note bad"><b>Decisive sanity test.</b> A naïve medial-axis "median radius" method assigns 2-V2 — a CT-validated <i>normal</i> trachea — a fake 77–92 % obstruction (the centerline wanders on one-sided clouds and manufactures narrowings). The validated partial-arc method reads 2-V2 as normal. Every number below therefore uses the validated method with strict gates, and each profile was reviewed individually.</div>
<h3>7.1 Cohort classification</h3>
<figure><img src="{FIG['csa_grp']}" alt="CSA cohort classification">
<figcaption><b>Figure 5 — Scale-free CSA profiles grouped by reviewed classification.</b> Points coloured by arc coverage. Roughly-constant calibre (normal) · reconstruction funnel (CSA trend is geometry, not anatomy) · reading at the noise ceiling (21-V1 — clinically normal) · not reliably measurable (coverage-dip / carina-flare / bad-geometry artefacts).</figcaption></figure>
<h3>7.2 Robust %obstruction on the clean full-ring tubes</h3>
<p>Strict acceptance (coverage ≥ 0.75, residual &lt; 0.15, circle/ellipse agreement), end 10 % of stations excluded, 3-station median filter; A<sub>min</sub> = narrowest interior valid slice, A<sub>ref</sub> = 90th-percentile valid CSA.</p>
<div class="tbl"><table>
<tr><th>Case</th><th class="num">n valid</th><th class="num">coverage</th><th class="num">%obstr</th><th class="num">D<sub>CE</sub> ratio</th><th>Read</th></tr>
<tr><td><b>21-V1</b></td><td class="num">33</td><td class="num">0.97</td><td class="num"><b>29 %</b></td><td class="num">0.84</td><td><b>Clinically normal</b> — operative note and chart list no stenosis or narrowing (inflammation / cobblestoning possible). The highest reading in the cohort is a normal airway → empirical false-positive ceiling ≈ 30 %</td></tr>
<tr><td>7-V1</td><td class="num">21</td><td class="num">1.00</td><td class="num">18 %</td><td class="num">0.91</td><td>mild ripple → normal</td></tr>
<tr><td>31-V1</td><td class="num">13</td><td class="num">1.00</td><td class="num">16 %</td><td class="num">0.92</td><td>mild → normal (erratic middle correctly rejected)</td></tr>
<tr><td>16-V1</td><td class="num">38</td><td class="num">0.96</td><td class="num">24 %</td><td class="num">0.87</td><td>shallow interior dip, mild</td></tr>
<tr><td>10-V2</td><td class="num">25</td><td class="num">0.90</td><td class="num">22 %</td><td class="num">0.88</td><td>gentle rise, mild</td></tr>
<tr><td>15-V2 · 25-V1 · 9-V2 · 12-V1</td><td class="num">28–45</td><td class="num">≥0.93</td><td class="num">28–34 %ᵉ</td><td class="num">~0.83</td><td>minimum at the edge of the valid run = taper / funnel; bodies flat → <b>normal</b></td></tr>
<tr><td>22-V2</td><td class="num">30</td><td class="num">0.99</td><td class="num">27 %ᶠ</td><td class="num">0.85</td><td>funnel-dominated rise; geometry, not anatomy</td></tr>
<tr><td>33-V1</td><td class="num">42</td><td class="num">0.96</td><td class="num">53 % ⚠</td><td class="num">0.68</td><td>erratic profile with a discontinuity (CV 0.24) — <b>unreliable, not a stenosis</b></td></tr>
<tr><td>50-V2 · 20-V1 (CT)</td><td class="num">10</td><td class="num">0.97</td><td class="num">12 % · 38 %ᵉ</td><td></td><td>few stations here — the CT-validated values (Angle 1) supersede</td></tr>
<tr><td>2-V2 (CT) · 17-V1 · 32-V2 · 10-V1</td><td colspan="4"><i>not measurable</i></td><td>ellipse fit unstable (2-V2) or &lt; 10 strict-valid stations</td></tr>
</table></div>
<p class="small">ᵉ minimum at the edge of the valid run (taper / funnel, not an interior stenosis) · ᶠ funnel-dominated · ⚠ unreliable geometry</p>
<figure><img src="{FIG['csa_rob']}" alt="Robust CSA with A_min and A_ref">
<figcaption><b>Figure 6 — Robust %obstruction.</b> Grey = rejected stations, coloured = accepted (by coverage); blue dashed = A<sub>ref</sub>, red dashed / ▼ = A<sub>min</sub>. The gates visibly reject 9-V2's proximal spikes, 10-V2's spike-down, 16-V1's coverage-loss tail and 31-V1's erratic middle.</figcaption></figure>
<div class="note good"><b>Reading it.</b> Measurable on 13 of 17 complete tubes. <b>Every reliable value is &lt; 50 % — Myer–Cotton grade I</b>, as expected for a cohort of predominantly normal airways: the method does not manufacture stenosis (quantified specificity). The normal variation / noise floor is <b>~15–30 % CSA</b> (D<sub>CE</sub> ratio 0.84–0.92) — the honest sensitivity limit for detecting <i>mild</i> stenosis from monocular video alone. The one reading flagged for review (21-V1, 29 %) was <b>clinically normal</b> on chart and operative-note review (inflammation / cobblestoning noted), which fixes the empirical false-positive ceiling at ≈ 30 % — no clinically-normal airway read above it. A chart review of the remaining measurable cases would complete the specificity claim.</div>

<h2 id="s8">8 · Limitations and honesty notes</h2>
<ul>
<li>Absolute millimetres are resolved <i>against CT</i> (isotropic Sim(3)); without CT the measurement is scale-free (ratios only). No in-frame object of known size was available for independent scale.</li>
<li>50-V2 degrades distally (thin coverage at the carina) and drives the pooled limits of agreement.</li>
<li>Reconstruction success depends on capture behaviour (centred, well-travelled, monotonic scope motion); one-sided imaging yields one-sided clouds and cannot be fixed downstream.</li>
<li>The phantom is a synthetic render (code validation); real-optics external validation rests on C3VD.</li>
<li>%obstruction from monocular video alone has a ~15–30 % floor; mild stenosis below that is not detectable, and automatic "local-minimum" flags are unreliable without per-profile review.</li>
</ul>

<h2 id="s9">9 · Summary</h2>
<div class="tbl"><table>
<tr><th>Angle</th><th>Question</th><th>Evidence</th><th>Result</th></tr>
<tr><td><b>1 · CT accuracy</b></td><td>Does reconstructed calibre match CT?</td><td>3 patients; 102 all-slice / {V2P['n']} validated sections</td><td>baseline RMSE {P['rmse']:.2f} mm pooled (1.66 mean of cases) → <b>improved {V2P['rmse']:.2f} mm</b> ({V2P['mean_cases']:.2f} mean of cases), bias {V2P['bias']:+.2f}, r {V2P['r']:.2f}</td></tr>
<tr><td><b>2 · Dynamic</b></td><td>Can it place a study in the respiratory cycle?</td><td>20-V1 insp/exp 4D-CT</td><td>45 % expiratory collapse resolved; reconstruction matches inspiration (RMSE 1.31 vs 3.44 mm); <b>tracheobronchomalacia clinically confirmed</b></td></tr>
<tr><td><b>3 · External</b></td><td>Does it generalise beyond our data?</td><td>C3VD (5 seq.) + phantom</td><td>1.04–2.03 mm surface (median 1.40), 0.15 mm pose; phantom 1.8 %</td></tr>
<tr><td><b>Cohort</b></td><td>How often does it work / what does it read on normal airways?</td><td>29 reconstructions; 13 CSA-measurable</td><td>12 complete tubes; all reliable %obstr &lt; 50 % (grade I); the highest (21-V1, 29 %) is clinically normal → FP ceiling ≈ 30 %</td></tr>
</table></div>
<p class="small">Files: <code>runs/own_data/reports/</code> (figures, tables, per-case profiles) · clouds in <code>runs/own_data/all_reconstructions_ply/</code> · analysis code <code>paper/airway_analysis.py</code>, <code>robust_csa.py</code>, <code>experiments/diagnostics/csa_partialarc.py</code>.</p>
</div>
"""
open(f"{OUT}/bronchotrust_report.html", "w").write(html)
print(f"\nwrote bronchotrust_report.html  ({len(html)//1024} KB)")
