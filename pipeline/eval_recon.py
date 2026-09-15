"""Quick assessment of a re-run reconstruction: measurability (gated sections of 48 interior stations,
median arc coverage — same gates as the yield table), registered frames / reprojection from the sparse
model, and a two-view render. The tiers are the paper's: >=20 gated interior stations = measurable tube,
10-19 = partially measurable, <10 = not measurable.
Usage: python pipeline/eval_recon.py <workspace_dir> [--label NAME] [--out-dir DIR]   (default: writes into the workspace)"""
import os as _os, sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]          # repository root (was a hard-coded absolute path)
for _p in (str(ROOT), str(ROOT / "pipeline"), str(ROOT / "experiments/cmig_paper")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import sys, os, re, json, numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.cm as cm
from csa_partialarc import profile
try:
    from figstyle import apply_style; apply_style()
except Exception: pass
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("workspace"); _ap.add_argument("--label", default=None); _ap.add_argument("--out-dir", default=None)
_a = _ap.parse_args(); WS = _a.workspace.rstrip("/"); LABEL = _a.label or os.path.basename(WS); OUT = _a.out_dir or WS
fused = f"{WS}/dense0/fused.ply"
if not os.path.exists(fused): print(f"{LABEL}: NO CLOUD"); sys.exit()
# registration stats from the run log (pycolmap + Open3D-CUDA in one process segfaults)
log = open(f"{WS}/run.log").read() if os.path.exists(f"{WS}/run.log") else ""; mm = re.search(r"\[model\] (\d+) reg, span f(\d+)-f?(\d+), reproj ([\d.]+)px", log)
n_reg, fr, mre = (int(mm.group(1)), (int(mm.group(2)), int(mm.group(3))), float(mm.group(4))) if mm else (0, (0, 0), float("nan"))
class _R: pass
rec = _R(); rec.num_reg_images = lambda: n_reg
P = np.asarray(o3d.io.read_point_cloud(fused).points); m = np.median(P, 0); d = np.linalg.norm(P - m, axis=1); P = P[d < np.median(d) + 4 * np.median(np.abs(d - np.median(d)))]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc, _ = pc.remove_statistical_outlier(24, 1.8); pc, _ = pc.remove_radius_outlier(16, np.median(d) * 0.03); P = np.asarray(pc.points)
rows, _ = profile(P, nb=60); n = len(rows); rs = rows[int(0.1 * n):int(0.9 * n)]
cov = np.array([r["cov"] for r in rs]); res = np.array([r["resid"] for r in rs]); ng = int(((cov >= 0.75) & (res < 0.15)).sum()); mc = float(np.median(cov)) if len(cov) else 0
tier = "measurable tube" if ng >= 20 else ("partially measurable" if ng >= 10 else "not measurable")
print(f"{LABEL}: registered {rec.num_reg_images()} frames f{fr[0]}-f{fr[-1]} | reproj {mre:.2f} px | clean pts {len(P):,} | gated {ng}/{len(rs)} | median cov {mc:.2f} -> {tier}")
# render two views with matplotlib (Open3D's EGL renderer segfaults while COLMAP saturates the GPUs)
from airway_analysis import scatter3d
fig = plt.figure(figsize=(8.5, 5.6))
for k, az in enumerate((-60, 30)):
    a = fig.add_subplot(1, 2, k + 1, projection="3d"); scatter3d(a, P, sub=60000, s=2.0, azim=az)
fig.suptitle(f"{LABEL}: {rec.num_reg_images()} frames f{fr[0]}-{fr[-1]} · reproj {mre:.2f} px · gated {ng}/{len(rs)} · cov {mc:.2f} · {tier}", fontsize=10.5)
fig.tight_layout(); out = f"{OUT}/eval_{LABEL.replace(' ', '_')}.png"; fig.savefig(out, dpi=130); print("saved", out)
json.dump(dict(label=LABEL, registered=rec.num_reg_images(), f_lo=fr[0], f_hi=fr[-1], reproj=mre, n_clean=len(P), n_gated=ng, n_stations=len(rs), median_cov=mc, tier=tier), open(f"{WS}/eval.json", "w"), indent=1)
