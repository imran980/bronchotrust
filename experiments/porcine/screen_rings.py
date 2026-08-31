"""Serial sparse-only ring/funnel screen over a list of chunks. For each chunk: run the pipeline SfM
(feature-capped, fast), then render sparse cross-sections perpendicular to the camera-trajectory axis.
Stacks one row per chunk into a tall montage. NO MVS (fast, no thrash). Judges ring vs funnel by eye.
Reads chunks from argv as  wsname:video:lo:hi  tokens (stride 2, cap 4000 baked in)."""
import subprocess, os, re, glob, sys
import numpy as np, pycolmap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

SC = "/tmp/claude-100461304/-home-mi3dr-projects-bronchotrust/32888f63-b0c2-4d41-9ad1-91a5dfc651e6/scratchpad"
D = "/home/mi3dr/dataset/Porcine Airway Endoscopies"
CAL = "runs/own_data/porcine/pig1/intrinsics_pinned.json"
PY = "/home/mi3dr/.conda/envs/depth-eval/bin/python"
chunks = [t.split(":") for t in sys.argv[1:]]                       # wsname:videofile:lo:hi

rows = []
for tag, vfile, lo, hi in chunks:
    r = subprocess.run([PY, "-u", str(Path(__file__).resolve().parent / "recon_pig_sfm.py"), f"{D}/{vfile}", CAL, lo, hi, "2", tag, "4000"],
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith(tag + ":")]
    summary = line[-1] if line else f"{tag}: NO OUTPUT"
    print(summary, flush=True)
    mods = glob.glob(f"{SC}/{tag}/sparse/*")
    mods = [m for m in mods if os.path.exists(f"{m}/images.bin")]
    if not mods: rows.append((tag, summary, None)); continue
    best = max(mods, key=lambda d: pycolmap.Reconstruction(d).num_reg_images())
    rec = pycolmap.Reconstruction(best)
    C = np.array([np.asarray(im.projection_center()) for im in rec.images.values()])
    P = np.array([np.asarray(p.xyz) for p in rec.points3D.values()])
    if len(P) < 50 or len(C) < 8: rows.append((tag, summary, None)); continue
    # trim gross outliers
    m = np.median(P, 0); dd = np.linalg.norm(P - m, axis=1); P = P[dd < np.median(dd) + 4 * np.median(np.abs(dd - np.median(dd)))]
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False); ax = Vt[0]
    ref = np.eye(3)[np.argmin(np.abs(ax))]; u = np.cross(ax, ref); u /= np.linalg.norm(u); v = np.cross(ax, u)
    Pc = P - C.mean(0); pa = Pc @ ax; pu = Pc @ u; pv = Pc @ v; ca = Cc @ ax; cu = Cc @ u; cv = Cc @ v
    travel = ca.max() - ca.min(); rad = np.median(np.linalg.norm(Pc - np.outer(pa, ax), axis=1))
    rows.append((tag, summary + f"  travel/radius {travel/max(rad,1e-6):.2f}", (pa, pu, pv, ca, cu, cv)))

nR = len(rows); fig = plt.figure(figsize=(20, 3.1 * nR))
for i, (tag, summary, geom) in enumerate(rows):
    if geom is None:
        a = fig.add_subplot(nR, 5, i * 5 + 1); a.text(0.02, 0.5, summary, fontsize=8, family="monospace"); a.axis("off"); continue
    pa, pu, pv, ca, cu, cv = geom
    lo_, hi_ = np.percentile(pa, 5), np.percentile(pa, 95); cuts = np.linspace(lo_, hi_, 6)[1:-1]; sw = (hi_ - lo_) / 10
    for k, cut in enumerate(cuts):
        sel = np.abs(pa - cut) < sw; a = fig.add_subplot(nR, 5, i * 5 + k + 1)
        a.scatter(pu[sel], pv[sel], s=2, c='steelblue')
        cs = np.abs(ca - cut) < sw * 2
        if cs.any(): a.scatter(cu[cs], cv[cs], c='red', s=25, marker='+')
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if k == 0: a.set_ylabel(tag, fontsize=8)
        a.set_title(f"a={cut:.1f}", fontsize=7)
    a = fig.add_subplot(nR, 5, i * 5 + 4); a.scatter(pa[::2], pu[::2], s=1, c='steelblue'); a.plot(ca, cu, 'r.-', ms=1, lw=.4)
    a.set_title("side", fontsize=7); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(nR, 5, i * 5 + 5); a.text(0.0, 0.5, summary, fontsize=7, family="monospace", wrap=True); a.axis("off")
fig.suptitle("Ring/funnel screen — sparse cross-sections perpendicular to camera axis (red +=cams). "
             "RING=closed loop w/ cam inside; FUNNEL=open arc, cam at edge", fontsize=11)
fig.tight_layout(); out = "runs/own_data/porcine/ring_screen.png"; fig.savefig(out, dpi=80); print(f"saved {out}", flush=True)
print("SCREEN_DONE", flush=True)
