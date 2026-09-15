"""CSA / %-obstruction feasibility report (non-CT successes). Groups the validated partial-arc CSA
profiles by a MANUALLY-REVIEWED classification (each profile eyeballed for coverage, estimator
agreement, and bend/flare artifacts). Outputs csa_cohort_figure.png + csa_summary.json."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from figstyle import apply_style, OKABE
apply_style()

D = f"{ROOT}/runs/own_data/reports/csa"
# classification from careful per-profile review (coverage steadiness, circle-vs-ellipse agreement,
# and rejection of coverage-dip / carina-flare / ellipse-blowup artifacts)
GROUPS = [
    ("Roughly-constant caliber — normal (specificity: no false stenosis)", OKABE["green"],
     ["15-V2", "17-V1", "31-V1", "25-V1", "7-V1"]),
    ("Reconstruction funnel — CSA trend is geometry, not anatomy", OKABE["orange"],
     ["22-V2", "12-V1", "10-V2", "16-V1"]),
    ("Reading at the noise ceiling — clinically NORMAL (op note/chart: no stenosis; inflammation/cobblestoning)", OKABE["blue"],
     ["21-V1"]),
    ("Not reliably measurable — artifact / bad geometry", "0.5",
     ["33-V1", "9-V2", "2-V3"]),
]
NOTES = {
    "15-V2": "flat body ~3.5-4; end spikes only", "17-V1": "wavy 6-9.7, no single stenosis",
    "31-V1": "flat ~18-20 + single-station spikes", "25-V1": "taper, noisy proximal end",
    "7-V1": "roughly constant 4.2-5.3 (20% ripple)",
    "22-V2": "monotonic rise = funnel opening", "12-V1": "monotonic rise = funnel",
    "10-V2": "rise + proximal spike-down artifact", "16-V1": "distal drop tracks coverage loss",
    "21-V1": "29% dip, full coverage — clinically normal → FP ceiling ≈30%",
    "33-V1": "min = coverage dip; max = carina flare (fake 83%)",
    "9-V2": "proximal spikes 1.5->5.5->1.5 (noise)", "2-V3": "curled-sheet recon (bad geometry)",
}

rows = sum(len(g[2]) for g in GROUPS)
fig = plt.figure(figsize=(15, 2.4 * rows + 1.2))
gs = fig.add_gridspec(rows, 1, hspace=0.05)
r = 0
summary = {}
for title, col, pids in GROUPS:
    for i, pid in enumerate(pids):
        ax = fig.add_subplot(gs[r, 0]); r += 1
        try:
            ax.imshow(mpimg.imread(f"{D}/csa_{pid}.png"))
        except Exception:
            ax.text(.5, .5, pid, ha="center")
        ax.axis("off")
        band = title if i == 0 else ""
        ax.text(-0.01, 0.5, pid, transform=ax.transAxes, ha="right", va="center",
                fontsize=13, fontweight="bold", color=col)
        ax.text(1.005, 0.5, NOTES.get(pid, ""), transform=ax.transAxes, ha="left", va="center",
                fontsize=8.5, color="0.35", wrap=True)
        if band:
            ax.text(0.5, 1.02, band, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold", color=col)
        summary[pid] = {"group": title.split(" —")[0], "note": NOTES.get(pid, "")}
fig.suptitle("Scale-free CSA profiles across the non-CT reconstruction cohort (validated partial-arc method)",
             fontsize=15, fontweight="bold", y=0.997)
fig.subplots_adjust(left=0.10, right=0.80, top=0.965, bottom=0.01)
fig.savefig(f"{ROOT}/runs/own_data/reports/csa_cohort_figure.png", dpi=115)
json.dump(summary, open(f"{ROOT}/runs/own_data/reports/csa_summary.json", "w"), indent=2)
print("saved csa_cohort_figure.png +", len(summary), "cases classified")
