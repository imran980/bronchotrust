"""One algorithm-agnostic comparison report: CT  ->  Method A / Method B / Method C, identical
metrics. Nothing in the figures/tables depends on which reconstruction produced the numbers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .metrics.metrics import EvaluationResult


def make_report(gt, results, out_dir, title=None):
    """results: list[EvaluationResult] (one per method). Writes report.json + report.png + a CSV."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.summary_row() for r in results]
    report = dict(ct_id=gt.ct_id, mode=gt.mode, ground_truth_checksum=gt._checksum,
                  metrics=["CSA_err_pct_p50", "DCE_err_pct_p50", "centerline_mm", "stenosis_loc_mm",
                           "obstruction_err_pct", "calib_factor", "within_band", "runtime_s"],
                  methods={r.method: r.to_dict() for r in results}, summary=rows)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: None if o != o else o))
    # CSV
    keys = list(rows[0].keys())
    with open(out_dir / "report.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows: f.write(",".join("" if r[k] is None else f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    # figure: profile overlay (if profile mode) + a metrics bar grid
    fig = plt.figure(figsize=(15, 8))
    if gt.mode == "profile":
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(gt.arclength_mm, gt.dce_mm, "k-", lw=2.5, label="CT (ground truth)")
        for r in results:
            d = r.detail.get("per_slice")
            if d: ax.plot(d["ct_arclength_mm"], np.array(d["gt_dce"]) + np.array(d["dce_err_mm"]), ".-", ms=3, label=r.method, alpha=.8)
        ax.set_xlabel("CT arclength (mm)"); ax.set_ylabel("DCE (mm)"); ax.set_title("DCE profile: CT vs methods"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    panels = [("CSA_err_pct_p50", "CSA error % (p50)"), ("DCE_err_pct_p50", "DCE error % (p50)"),
              ("centerline_mm", "centerline err (mm)"), ("stenosis_loc_mm", "stenosis loc err (mm)"),
              ("obstruction_err_pct", "obstruction err %")]
    names = [r["method"] for r in rows]
    for k, (key, lab) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 2)
        vals = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.bar(range(len(names)), vals, color="tab:blue")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, fontsize=8); ax.set_ylabel(lab); ax.set_title(lab, fontsize=9)
    fig.suptitle(title or f"Evaluation vs CT ({gt.ct_id}) — identical metrics, algorithm-agnostic", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_dir / "report.png", dpi=120); plt.close(fig)
    return report


def print_table(results):
    rows = [r.summary_row() for r in results]; keys = list(rows[0].keys())
    w = {k: max(len(k), *(len(("" if r[k] is None else f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]))) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print(" | ".join(("" if r[k] is None else f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k])).ljust(w[k]) for k in keys))
