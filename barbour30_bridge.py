"""Scale-free % obstruction via Sim(3) bridge of accepted local smart rings.
Each accepted local 30-frame batch has its OWN arbitrary gauge. To compute a within-video
% obstruction (CSA ratio) we must put the stenosis and reference rings in ONE common scale.
Bridge = align each local batch to the video's long-window GLOBAL model (batch4 / fuse — one
connected model, one scale) using SHARED camera frames (Umeyama Sim(3)). Stability is judged
by (a) the spread of pairwise camera-distance ratios global/local (CoV) and (b) the Umeyama
residual, both relative to the airway. Only if the stenosis AND reference bridges are stable
do we compute % obstruction. NO mm. NO cross-video comparison. depth-eval env."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap
from itertools import combinations

ROOT = Path("/home/mi3dr/projects/bronchotrust"); OUT = ROOT / "runs/barbour30"
GLOBALS = {"2-V2": "runs/batch4/2-V2/sparse/0", "25-V1": "runs/batch4/25-V1/sparse/0",
           "32-V2": "runs/fuse_32v2/sparse/0"}
SUBGLOTTIC = {"prox_subglottis", "dist_subglottis"}
COV_MAX, RESREL_MAX, NMIN = 0.15, 0.12, 5   # stability thresholds


def cam_centers(model):
    rec = pycolmap.Reconstruction(str(model)); out = {}
    for im in rec.images.values():
        M = np.array(im.cam_from_world().matrix())
        out[int(re.search(r"f(\d+)", im.name).group(1))] = -M[:3, :3].T @ M[:3, 3]
    return out


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X); U, D, Vt = np.linalg.svd(S)
    Dm = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ Dm @ Vt
    var = (Xc ** 2).sum() / len(X); s = float(np.trace(np.diag(D) @ Dm) / var)
    t = my - s * R @ mx; Yp = (s * (R @ X.T).T + t)
    res = float(np.sqrt(((Y - Yp) ** 2).sum(1).mean()))
    return s, res


def bridge(video, name):
    gC = cam_centers(GLOBALS[video]); lC = cam_centers(OUT / "batches" / name / "sparse/0")
    shared = sorted(set(gC) & set(lC))
    if len(shared) < NMIN:
        return {"n_shared": len(shared), "stable": False, "why": "too few shared frames"}
    X = np.array([lC[f] for f in shared]); Y = np.array([gC[f] for f in shared])
    s, res = umeyama(X, Y)
    gspread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    resrel = res / max(gspread, 1e-9)
    # pairwise distance ratios global/local (scale consistency)
    idx = list(combinations(range(len(shared)), 2))
    dl = np.array([np.linalg.norm(X[i] - X[j]) for i, j in idx])
    dg = np.array([np.linalg.norm(Y[i] - Y[j]) for i, j in idx])
    ok = dl > 1e-6; ratios = dg[ok] / dl[ok]
    cov = float(np.std(ratios) / max(np.median(ratios), 1e-9))
    stable = bool(len(shared) >= NMIN and cov <= COV_MAX and resrel <= RESREL_MAX)
    return {"n_shared": len(shared), "s_umeyama": round(s, 4), "s_pairwise_med": round(float(np.median(ratios)), 4),
            "scale_CoV": round(cov, 3), "resid_rel": round(resrel, 3), "stable": stable,
            "why": "ok" if stable else f"CoV={cov:.2f}(>{COV_MAX}) or resid_rel={resrel:.2f}(>{RESREL_MAX})"}


def main():
    rows = json.loads((OUT / "pipeline_report.json").read_text())
    acc = [r for r in rows if r["verdict"] == "ACCEPT"]
    per_video = {}
    for r in acc:
        v = r["video"]; lm = r["landmark"]; name = f"{v}_{lm}_pipe_p{r['chosen_pool']}"
        m = json.loads((OUT / "batches" / name / "measure.json").read_text())
        br = bridge(v, name)
        s = br.get("s_pairwise_med")   # local->global scale
        rec = {"landmark": lm, "name": name,
               "CSA_local": (m.get("CSA_scene") or {}).get("polar"), "DCE_local": (m.get("DCE_scene") or {}).get("polar"),
               "bridge": br}
        if br["stable"] and s:
            rec["CSA_global"] = round(rec["CSA_local"] * s * s, 4)
            rec["DCE_global"] = round(rec["DCE_local"] * s, 4)
        per_video.setdefault(v, []).append(rec)

    table = []
    for v, recs in per_video.items():
        bridged = [x for x in recs if x["bridge"]["stable"] and "CSA_global" in x]
        sub = [x for x in bridged if x["landmark"] in SUBGLOTTIC]
        if not sub or len(bridged) < 2:
            table.append({"video": v, "stenosis": None, "reference": None, "bridge_valid": False,
                          "CSA_ratio": None, "DCE_ratio": None, "pct_obstruction": None,
                          "verdict": "Scale-free % obstruction NOT valid from separate local gauges "
                                     f"({'no stable subglottic ring' if not sub else 'need >=2 stably-bridged rings'})",
                          "_recs": recs})
            continue
        sten = min(sub, key=lambda x: x["CSA_global"])          # narrowest subglottic
        ref = max(bridged, key=lambda x: x["CSA_global"])       # widest bridged ring
        if ref["name"] == sten["name"]:
            cand = [x for x in bridged if x["name"] != sten["name"]]
            ref = max(cand, key=lambda x: x["CSA_global"]) if cand else ref
        csa_ratio = sten["CSA_global"] / ref["CSA_global"]; dce_ratio = sten["DCE_global"] / ref["DCE_global"]
        pct = 100 * (1 - csa_ratio)
        table.append({"video": v, "stenosis": sten["landmark"], "reference": ref["landmark"],
                      "bridge_valid": True, "CSA_ratio": round(csa_ratio, 3), "DCE_ratio": round(dce_ratio, 3),
                      "pct_obstruction": round(pct, 1),
                      "verdict": f"% obstruction (CSA) = {pct:.1f}% ; diameter-equiv = {100*(1-dce_ratio):.1f}% "
                                 "(scene-units common scale via Sim(3); NO mm; within-video only)",
                      "_recs": recs})
    (OUT / "obstruction_report.json").write_text(json.dumps(
        {"stability_thresholds": {"scale_CoV_max": COV_MAX, "resid_rel_max": RESREL_MAX, "n_shared_min": NMIN},
         "table": table}, indent=2, default=str))

    print("\n=== Sim(3) BRIDGE STABILITY (each accepted ring -> global model) ===")
    for v, recs in per_video.items():
        for x in recs:
            b = x["bridge"]
            print(f"  {v:6} {x['landmark']:16} shared={b.get('n_shared')} s={b.get('s_pairwise_med')} "
                  f"CoV={b.get('scale_CoV')} resid_rel={b.get('resid_rel')} -> {'STABLE' if b['stable'] else 'unstable: '+b['why']}")
    print("\n=== % OBSTRUCTION TABLE ===")
    cols = ["video", "stenosis", "reference", "bridge_valid", "CSA_ratio", "DCE_ratio", "pct_obstruction", "verdict"]
    print(" | ".join(cols))
    for t in table:
        print(" | ".join(str(t.get(c)) for c in cols))
    print(f"\nreport -> {OUT}/obstruction_report.json")


if __name__ == "__main__":
    main()
