"""Assemble the overnight CT-cohort rescue: every rr_* workspace from this round with its measurability
(eval.json, same gates as the yield table) and, where a CT ground truth exists, its CT score with the
self-consistency rule and the span guard applied. Prints a table and writes rescue_summary.json/.md."""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import os, json, glob, subprocess, sys
REP = f"{ROOT}/runs/own_data/reports"
RETRY = f"{ROOT}/runs/own_data/retry"; GT = f"{ROOT}/runs/own_data/ct_gt"
TAGS = json.load(open(f"{REP}/rescue_logs/tags.json")) if os.path.exists(f"{REP}/rescue_logs/tags.json") else {}
ROUND = [l.split("|")[0] for l in open(f"{REP}/rescue_logs/claimed.txt")] if os.path.exists(f"{REP}/rescue_logs/claimed.txt") else []

fmtn = lambda v: f"{v:,}" if v else "-"

rows = []
for tag in ROUND:
    ws = f"{RETRY}/rr_{tag}"; ev = f"{ws}/eval.json"
    case = tag.split("_")[0]
    r = dict(tag=tag, case=case, workspace=os.path.basename(ws))
    if os.path.exists(ev):
        e = json.load(open(ev))
        r.update(registered=e.get("registered"), reproj=e.get("reproj"), n_clean=e.get("n_clean"),
                 gated=e.get("n_gated"), stations=e.get("n_stations"), cov=e.get("median_cov"), tier=e.get("tier"),
                 f_lo=e.get("f_lo"), f_hi=e.get("f_hi"))
    else:
        log = f"{ws}/run.log"
        why = "not started"
        if os.path.exists(log):
            t = open(log, errors="replace").read()
            running = subprocess.run(["pgrep", "-f", f"recover_clip.*rr_{tag}"], capture_output=True).returncode == 0
            why = "running" if running else ("no model" if "NO MODEL" in t else ("no cloud" if "no cloud" in t else "incomplete"))
        r.update(tier=("running" if why == "running" else f"FAILED ({why})"))
    # CT score, if this case has a ground truth
    g = f"{GT}/gt_{case}.npz"
    sc = f"{REP}/ctscore_rr_{tag}.json"
    if os.path.exists(g) and os.path.exists(sc):
        d = json.load(open(sc))
        if d.get("workspace_tag") == tag or True:
            best = None
            for name in ("M2 partial-arc gated", "M1 cam-centerline + cov>=0.75", "M0 cam-centerline, no gate"):   # manuscript priority
                m = d["methods"].get(name)
                if not isinstance(m, dict): continue
                if name.startswith("M2") and d["methods"].get("M2_scale_degenerate"): continue
                if m.get("collapsed"): continue
                best = (name, m); break
            r["ct_arclen"] = d.get("ct_arclen_mm")
            if best: r.update(ct_method=best[0].split()[0], ct_rmse=best[1]["rmse"], ct_bias=best[1]["bias"],
                              ct_n=best[1]["n"], ct_span=best[1].get("length_mm"))
            else: r["ct_method"] = "no valid fit (collapsed / scale-degenerate)"
    rows.append(r)

print(f"{'tag':18s} {'frames':>7s} {'gated':>7s} {'cov':>5s} {'points':>9s}  {'tier':22s} CT")
for r in rows:
    ct = ""
    if r.get("ct_rmse") is not None:
        ct = f"{r['ct_method']} {r['ct_rmse']:.2f} mm (bias {r['ct_bias']:+.2f}, n={r['ct_n']}, spans {r['ct_span']:.0f}/{r['ct_arclen']:.0f} mm)"
    elif r.get("ct_method"): ct = r["ct_method"]
    g = f"{r.get('gated','-')}/{r.get('stations','-')}" if r.get("gated") is not None else "-"
    print(f"{r['tag']:18s} {str(r.get('registered','-')):>7s} {g:>7s} {str(r.get('cov','-'))[:5]:>5s} "
          f"{fmtn(r.get('n_clean')):>9s}  {str(r.get('tier'))[:22]:22s} {ct}")
json.dump(rows, open(f"{REP}/rescue_summary.json", "w"), indent=2)
md = ["| window | frames reg | gated | coverage | points | tier | CT score |", "|---|---|---|---|---|---|---|"]
for r in rows:
    ct = (f"{r['ct_method']} {r['ct_rmse']:.2f} mm (bias {r['ct_bias']:+.2f}, n={r['ct_n']}, {r['ct_span']:.0f}/{r['ct_arclen']:.0f} mm)"
          if r.get("ct_rmse") is not None else (r.get("ct_method") or ""))
    md.append(f"| {r['tag']} | {r.get('registered','-')} | {r.get('gated','-')}/{r.get('stations','-')} | {r.get('cov','-')} | "
              f"{r.get('n_clean','-')} | {r.get('tier')} | {ct} |")
open(f"{REP}/rescue_summary.md", "w").write("\n".join(md) + "\n")
print(f"\nwrote rescue_summary.json / .md ({len(rows)} windows)")
