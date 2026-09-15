"""Calibration table for the manuscript, from every pinned-intrinsics file (retriage_first15/_calib/*/intrinsics_pinned.json
and own_data/retry/*_intrinsics.json; pigs excluded). One row per session, sorted by case number, with a Note column that
flags rows a reviewer must know about: assumed intrinsics (no calibration video), a calibration shared from another
session, and marginal board RMS (>0.5 px). Writes reports/calib_table.{md,tex} and copies the .tex into paper/."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import glob, json, os, re, shutil
import numpy as np

R = f"{ROOT}/runs"; REP = f"{R}/own_data/reports"; PAPER = f"{ROOT}/experiments/cmig_paper"
RMS_STRICT, RMS_MARG = 0.5, 0.7                      # bands: strict < 0.5 px; marginal 0.5-0.7 px; poor > 0.7 px
# second-batch sessions whose intrinsics were re-derived on 2026-09-12 from the same board videos (original workspaces lost)
REDERIVED = {"20-V1", "30-V2", "9-V2", "27-V1", "2-V3", "12-V1", "21-V1", "22-V2", "14-V1", "1-V3", "10-V1", "24-V2", "23-V2", "20-V2", "24-V1"}

files = sorted(glob.glob(f"{R}/retriage_first15/_calib/*/intrinsics_pinned.json")) + sorted(glob.glob(f"{R}/own_data/retry/*_intrinsics.json"))
rows, seen = [], set()
for f in files:
    if "pig" in f: continue
    sess = os.path.basename(os.path.dirname(f)) if "retriage" in f else os.path.basename(f).replace("_intrinsics.json", "")
    key = sess.replace("_", "-").upper()
    if key in seen: continue                       # prefer the first-cohort (retriage) file when both exist
    d = json.load(open(f)); p = d.get("params_colmap") or d.get("params") or []
    fx = d.get("fx", p[0] if len(p) > 3 else np.nan); fy = d.get("fy", p[1] if len(p) > 3 else np.nan)
    w = d.get("width", 1920); h = d.get("height", 1080)
    hf = d.get("hfov_deg", 2 * np.degrees(np.arctan(w / (2 * fx)))); rms = d.get("rms_reproj_px", d.get("rms", np.nan))
    nfr = d.get("n_frames")
    src = os.path.basename(d.get("source_calib_video") or "")
    m = re.match(r"(\d+[-_][vV]\d)", src); src_key = m.group(1).replace("_", "-").upper() if m else None
    if d.get("approx"): prov = f"assumed ({d.get('fov_assumed_deg', 97)}$^\\circ$ FOV, no distortion)"; rms = np.nan
    elif src_key and src_key != key: prov = f"borrowed from {src_key}"
    elif key in REDERIVED: prov = "re-derived"
    else: prov = "original"
    band = "--" if not np.isfinite(rms) else ("strict" if rms < RMS_STRICT else "marginal")
    seen.add(key); rows.append(dict(key=key, fx=fx, fy=fy, r=fy / fx, hf=hf, rms=rms, n=nfr, band=band, prov=prov, note=prov))

# sessions whose reconstruction was run on another session's board (same patient, no own calibration video)
BORROW = {"32-V2": "32-V1"}
byk = {r["key"]: r for r in rows}
for k, srck in BORROW.items():
    if k not in byk and srck in byk:
        r = dict(byk[srck]); r["key"] = k; r["prov"] = f"borrowed from {srck} (same patient)"; r["note"] = r["prov"]; rows.append(r)

# restrict to the reconstructed cohort (yield table) so the two tables list the same cases; list the rest in a note
recon = [d["case"] for d in json.load(open(f"{REP}/yield_table.json"))]
others = sorted(r["key"] for r in rows if r["key"] not in recon)
rows = [r for r in rows if r["key"] in recon]
missing = sorted(set(recon) - {r["key"] for r in rows})
rows.sort(key=lambda r: (int(re.match(r"(\d+)", r["key"]).group(1)), r["key"]))
fmt_rms = lambda v: "--" if not np.isfinite(v) else f"{v:.3f}"
fmt_n = lambda v: "--" if v in (None, "") else str(v)
md = ["| Session | f_x (px) | f_y (px) | f_y/f_x | HFOV (°) | Board frames | RMS (px) | RMS band | Provenance |", "|---|---|---|---|---|---|---|---|---|"]
tex = ["\\begin{tabular}{lrrrrrrll}", "\\toprule", "Session & $f_x$ (px) & $f_y$ (px) & $f_y/f_x$ & HFOV ($^\\circ$) & Board frames & RMS (px) & RMS band & Provenance \\\\", "\\midrule"]
for r in rows:
    md.append(f"| {r['key']} | {r['fx']:.1f} | {r['fy']:.1f} | {r['r']:.4f} | {r['hf']:.1f} | {fmt_n(r['n'])} | {fmt_rms(r['rms'])} | {r['band']} | {r['prov']} |")
    tex.append(f"{r['key']} & {r['fx']:.1f} & {r['fy']:.1f} & {r['r']:.4f} & {r['hf']:.1f} & {fmt_n(r['n'])} & {fmt_rms(r['rms'])} & {r['band']} & {r['prov']} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(f"{REP}/calib_table.md", "w").write("\n".join(md) + "\n"); open(f"{REP}/calib_table.tex", "w").write("\n".join(tex) + "\n")
shutil.copy(f"{REP}/calib_table.tex", f"{PAPER}/calib_table.tex")
ok = [r for r in rows if np.isfinite(r["rms"])]
n_strict = sum(1 for r in rows if r["band"] == "strict"); n_marg = sum(1 for r in rows if r["band"] == "marginal"); n_poor = sum(1 for r in rows if r["band"] == "marginal" and r["rms"] > RMS_MARG)
n_bor = sum(1 for r in rows if r["prov"].startswith("borrowed")); n_red = sum(1 for r in rows if r["prov"] == "re-derived"); n_orig = sum(1 for r in rows if r["prov"] == "original")
marg = ", ".join(f"{r['key']} {r['rms']:.2f}" for r in sorted(rows, key=lambda r: -r['rms'] if np.isfinite(r['rms']) else 0) if r["band"] == "marginal")
note = (f"Calibration of the {len(rows)} reconstructed sessions (OPENCV model, $k_3=0$, 14$\\times$13 inner-corner checkerboard; intrinsics pinned during bundle adjustment). "
        f"Board RMS: {n_strict} sessions meet the strict threshold $<{RMS_STRICT}$ px; {n_marg} are marginal ($\\geq{RMS_STRICT}$ px; {marg}), {n_poor} of them above {RMS_MARG} px. "
        f"Provenance: {n_orig} original (the file used at reconstruction time), {n_red} re-derived (second-batch sessions recalibrated from the same board videos after the original workspaces were lost; "
        f"same recipe, so values match the originals up to frame subsampling but were not read back from the SfM models), {n_bor} borrowed (no own board video; 16-V1 uses 2-V2, 32-V2 uses 32-V1 of the same patient). "
        f"No reconstructed session uses assumed intrinsics. Three optics are present (HFOV $\\approx$97--99$^\\circ$; $\\approx$105--107$^\\circ$ for 16-V2, 20-V1, 23-V2; 78$^\\circ$ for 24-V2). "
        + (f"Calibrated sessions without a reconstruction in this study: {', '.join(others)}." if others else ""))
open(f"{REP}/calib_table_note.tex", "w").write(note + "\n"); shutil.copy(f"{REP}/calib_table_note.tex", f"{PAPER}/calib_table_note.tex")
print(f"calib table: {len(rows)} sessions | strict {n_strict} / marginal {n_marg} (of which >{RMS_MARG} px: {n_poor}) | provenance original {n_orig} / re-derived {n_red} / borrowed {n_bor} -> paper/calib_table.tex + calib_table_note.tex")
print(f"reconstructed cases WITHOUT a row: {missing or 'none'} | calibrated-not-reconstructed (in note): {others}")
print("\n".join(md))
