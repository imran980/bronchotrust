"""Stepwise visual diagnostic: WHERE does 15_v2 break? (matcher -> mapper -> measure)

Produces (runs/gluemap_15v2/15_v2/diag/):
  1_match_matrix.png    36x36 LightGlue match-count heatmap, frames grouped by
                        segment -> shows which pairs match (and which DON'T,
                        esp. insertion<->withdrawal sub-cord = the parallax pairs).
  2_match_examples.png  example pair overlays: a strong within-segment pair vs the
                        cross-pass sub-cord pair we NEED (few matches).
  3_mapper.png          COLMAP trajectory (sane) vs GlueMap (outlier cams) +
                        per-segment registration; sparse points.
  4_why_no_measure.png  triangulation-angle histogram (the parallax) + a sub-cord
                        slice scatter that fails to close into a ring.
  diag_tables.json      per-frame registration + match stats.
Run in depth-eval env.
"""
from __future__ import annotations
import json, re, sqlite3
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pycolmap
from hloc.utils.io import get_keypoints, get_matches

RUN = Path("/home/mi3dr/projects/bronchotrust/runs/gluemap_15v2/15_v2")
ST1, ST4 = RUN / "stage1", RUN / "stage4"
CARM, GARM = ST4 / "colmap", ST4 / "gluemap"
IMG = ST4 / "undist/images"
OUT = RUN / "diag"; OUT.mkdir(parents=True, exist_ok=True)

SEG_COLOR = {"glottis_in": "#28c8f0", "subglot_in": "#5adc5a", "trachea_in": "#dca032",
             "trachea_out": "#c8783c", "subglot_out": "#3cb43c", "glottis_out": "#1ea0c8"}

sel = json.loads((ST1 / "stage1_selection.json").read_text())["selection"]
seg_of = {s["frame_idx"]: s["segment"] for s in sel}
frames = sorted(seg_of)                       # 36 frame indices, temporal order
names = [f"f{fi:05d}.png" for fi in frames]
idx_of_name = {n: i for i, n in enumerate(names)}


def reg_frames(model_dir):
    try:
        rec = pycolmap.Reconstruction(str(model_dir))
    except Exception:
        return {}, None
    out = {}
    for img in rec.images.values():
        m = re.search(r"f(\d+)\.png", img.name)
        if not m:
            continue
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        out[int(m.group(1))] = {"C": (-R.T @ t),
                                "nobs": len([p for p in img.points2D if p.has_point3D()])
                                if hasattr(img, "points2D") else 0}
    return out, rec


# ---------- match-count matrix (raw LightGlue) ----------
feat = next(p for p in CARM.glob("feats*.h5") if "matches" not in p.name)
match = next(p for p in CARM.glob("*matches*.h5"))
N = len(names)
Mraw = np.zeros((N, N), int)
for i in range(N):
    for j in range(i + 1, N):
        try:
            mm, _ = get_matches(match, names[i], names[j])
            Mraw[i, j] = Mraw[j, i] = len(mm)
        except Exception:
            pass

# ---------- registration both arms ----------
creg, crec = reg_frames(CARM / "sfm")
greg, grec = reg_frames(GARM / "gm/gluemap_aba")
gctr = np.median(np.array([v["C"] for v in greg.values()]), 0) if greg else np.zeros(3)
gout = {fi for fi, v in greg.items()
        if np.linalg.norm(v["C"] - gctr) > 5 * np.median(
            [np.linalg.norm(vv["C"] - gctr) for vv in greg.values()])} if greg else set()

# ================= FIG 1: match matrix =================
fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(np.log10(Mraw + 1), cmap="magma", origin="upper")
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels([str(f) for f in frames], rotation=90, fontsize=6)
ax.set_yticklabels([str(f) for f in frames], fontsize=6)
# color the tick labels by segment + draw segment block boundaries
segs = [seg_of[f] for f in frames]
bnds = [0] + [k for k in range(1, N) if segs[k] != segs[k - 1]] + [N]
for a, b in zip(bnds[:-1], bnds[1:]):
    ax.add_patch(Rectangle((a - .5, a - .5), b - a, b - a, fill=False,
                           edgecolor=SEG_COLOR[segs[a]], lw=2.5))
    ax.text(N + 0.5, (a + b) / 2 - .5, segs[a], color=SEG_COLOR[segs[a]],
            fontsize=8, va="center")
plt.colorbar(im, ax=ax, label="log10(LightGlue matches + 1)", shrink=.8)
ax.set_title("STEP 1 — MATCHER: LightGlue match count between every frame pair\n"
             "diagonal blocks = within-segment (OK). The CROSS-PASS sub-cord block\n"
             "(subglot_in rows x subglot_out cols) is what gives the subglottic ring\n"
             "its parallax — watch whether it is bright or dark.")
fig.tight_layout(); fig.savefig(OUT / "1_match_matrix.png", dpi=140); plt.close(fig)

# cross-pass sub-cord block stats
si = [i for i, f in enumerate(frames) if seg_of[f] == "subglot_in"]
so = [i for i, f in enumerate(frames) if seg_of[f] == "subglot_out"]
block = Mraw[np.ix_(si, so)]
within_in = Mraw[np.ix_(si, si)][np.triu_indices(len(si), 1)]
print(f"sub-cord WITHIN-insertion median matches: {np.median(within_in):.0f}")
print(f"sub-cord CROSS-PASS (in x out) median matches: {np.median(block):.0f}  max:{block.max()}")

# ================= FIG 2: example overlays =================
def overlay(n0, n1, axp, title):
    i0 = cv2.imread(str(IMG / n0)); i1 = cv2.imread(str(IMG / n1))
    k0 = get_keypoints(feat, n0); k1 = get_keypoints(feat, n1)
    mm, _ = get_matches(match, n0, n1)
    H = max(i0.shape[0], i1.shape[0]); W = i0.shape[1] + i1.shape[1]
    canv = np.zeros((H, W, 3), np.uint8); canv[:i0.shape[0], :i0.shape[1]] = i0
    canv[:i1.shape[0], i0.shape[1]:] = i1
    off = i0.shape[1]
    rng = np.random.default_rng(0)
    for a, b in (mm[rng.choice(len(mm), min(len(mm), 60), replace=False)] if len(mm) else []):
        p0 = tuple(np.int32(k0[a])); p1 = (int(k1[b][0] + off), int(k1[b][1]))
        c = tuple(int(x) for x in rng.integers(60, 255, 3))
        cv2.line(canv, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canv, p0, 3, c, -1); cv2.circle(canv, p1, 3, c, -1)
    axp.imshow(cv2.cvtColor(canv, cv2.COLOR_BGR2RGB))
    axp.set_title(f"{title}\n{n0} <-> {n1}: {len(mm)} matches", fontsize=10)
    axp.axis("off")

# strong within subglot_in; weak cross-pass subglot_in<->subglot_out
def closest(seg, target):
    cand = [f for f in frames if seg_of[f] == seg]
    return f"f{min(cand, key=lambda f: abs(f - target)):05d}.png"
fig, ax = plt.subplots(2, 1, figsize=(13, 9))
overlay(closest("subglot_in", 80), closest("subglot_in", 95), ax[0],
        "STEP 1b — STRONG pair (within insertion sub-cord, adjacent views)")
overlay(closest("subglot_in", 84), closest("subglot_out", 1600), ax[1],
        "STEP 1b — THE PARALLAX PAIR WE NEED (insertion sub-cord <-> withdrawal sub-cord)")
fig.tight_layout(); fig.savefig(OUT / "2_match_examples.png", dpi=140); plt.close(fig)

# ================= FIG 3: mapper trajectories =================
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for axp, reg, rec, name, outl in [(ax[0], creg, crec, "COLMAP arm (sane geometry)", set()),
                                  (ax[1], greg, grec, "GlueMap arm (3 outlier cams)", gout)]:
    if not reg:
        axp.set_title(name + " — empty"); continue
    C = np.array([reg[f]["C"] for f in sorted(reg)]); ff = sorted(reg)
    pts = np.array([p.xyz for p in rec.points3D.values()]) if rec else np.empty((0, 3))
    # clip view to inlier cameras for GlueMap so the plot is readable
    if outl:
        inl = np.array([np.linalg.norm(reg[f]["C"] - gctr) <= 5 * np.median(
            [np.linalg.norm(reg[g]["C"] - gctr) for g in reg]) for f in ff])
        lim = C[inl]
    else:
        lim = C
    if len(pts):
        pc = pts - np.median(pts, 0)
        pts = pts[np.linalg.norm(pc, axis=1) <= 5 * np.median(np.linalg.norm(pc, axis=1))]
        axp.scatter(pts[:, 0], pts[:, 2], s=1, c="lightgray", alpha=.4)
    for f, c in zip(ff, C):
        col = SEG_COLOR[seg_of[f]]
        mk = "X" if f in outl else "o"
        axp.scatter(c[0], c[2], c=col, s=80 if f in outl else 40, marker=mk,
                    edgecolors="red" if f in outl else "black", linewidths=1.2 if f in outl else .4, zorder=3)
    axp.plot(C[:, 0], C[:, 2], "-", c="gray", lw=.6, alpha=.5)
    if len(lim):
        axp.set_xlim(lim[:, 0].min() - 5, lim[:, 0].max() + 5)
        axp.set_ylim(lim[:, 2].min() - 5, lim[:, 2].max() + 5)
    axp.set_title(f"STEP 2 — MAPPER: {name}\nreg={len(reg)}/36  "
                  f"{'outliers(X)='+str(sorted(outl)) if outl else 'reproj 1.4px'}", fontsize=10)
    axp.set_xlabel("X"); axp.set_ylabel("Z"); axp.set_aspect("equal", "datalim"); axp.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "3_mapper.png", dpi=140); plt.close(fig)

# ================= FIG 4: why no measurement =================
# tri-angles (COLMAP)
cc = {img.image_id: (lambda M: -M[:3, :3].T @ M[:3, 3])(np.array(img.cam_from_world().matrix()))
      for img in crec.images.values()}
tri = []
for p in crec.points3D.values():
    obs = [el.image_id for el in p.track.elements if el.image_id in cc]
    if len(obs) < 2: continue
    X = np.array(p.xyz); V = np.array([cc[i] - X for i in obs]); V /= np.linalg.norm(V, axis=1, keepdims=True)
    tri.append(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max())
tri = np.array(tri)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].hist(tri, bins=30, color="#c0392b", alpha=.8)
ax[0].axvline(np.median(tri), c="k", ls="--", label=f"median {np.median(tri):.1f}°")
ax[0].axvline(15, c="green", ls=":", label="comfortable ≥15°")
ax[0].set_title(f"STEP 3 — PARALLAX: COLMAP per-point triangulation angle\n"
                f"{len(tri)} points, median {np.median(tri):.1f}° (need ≥~10-15° for dense MVS)")
ax[0].set_xlabel("triangulation angle (deg)"); ax[0].legend()
# sub-cord slice on photometric cloud (the densest COLMAP cloud)
import open3d as o3d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
ph = CARM / "dense/fused_photometric.ply"
if ph.exists():
    P = np.asarray(o3d.io.read_point_cloud(str(ph)).points)
    C = np.array([creg[f]["C"] for f in sorted(creg)])
    sub = np.array([creg[f]["C"] for f in sorted(creg) if seg_of[f].startswith("subglot")])
    p0 = sub.mean(0) if len(sub) else C.mean(0)
    tck, _ = splprep(C.T, s=0.5, k=3); tg = np.array(splev(np.linspace(0, 1, 200), tck, der=1)).T
    T = tg[np.argmin(np.linalg.norm(np.array(splev(np.linspace(0, 1, 200), tck)).T - p0, axis=1))]
    T /= np.linalg.norm(T)
    rel = P - p0; da = rel @ T
    R = np.median(np.linalg.norm(rel - np.outer(da, T), axis=1))
    m = np.abs(da) <= 0.12 * R
    up = np.array([0, 0, 1.]) if abs(T[2]) < .95 else np.array([0, 1., 0])
    e1 = up - (up @ T) * T; e1 /= np.linalg.norm(e1); e2 = np.cross(T, e1)
    ip = np.stack([rel[m] @ e1, rel[m] @ e2], 1)
    ax[1].scatter(ip[:, 0], ip[:, 1], s=4, c="#34495e", alpha=.6)
    ax[1].scatter(0, 0, c="red", marker="+", s=200)
    th = np.arctan2(ip[:, 1], ip[:, 0]) if len(ip) else np.array([])
    cov = (np.histogram(th, bins=np.linspace(-np.pi, np.pi, 37))[0] > 0).mean() if len(ip) else 0
    ax[1].set_title(f"STEP 4 — MEASURE: a sub-cord cross-section slab\n"
                    f"{len(ip)} pts, angular coverage {cov*100:.0f}% (need ≥60%) — "
                    f"{'NO closed ring' if cov<0.6 else 'ring'}")
    ax[1].set_aspect("equal"); ax[1].set_xlabel("e1"); ax[1].set_ylabel("e2")
fig.tight_layout(); fig.savefig(OUT / "4_why_no_measure.png", dpi=140); plt.close(fig)

# ================= tables =================
rows = []
for f in frames:
    rows.append({"frame": f, "segment": seg_of[f],
                 "colmap_registered": f in creg,
                 "colmap_nobs": creg.get(f, {}).get("nobs", 0),
                 "gluemap_registered": f in greg,
                 "gluemap_outlier": f in gout,
                 "median_matches_to_others": int(np.median(Mraw[idx_of_name[f"f{f:05d}.png"]]))})
tables = {"per_frame": rows,
          "colmap_failed_frames": [f for f in frames if f not in creg],
          "gluemap_outlier_frames": sorted(gout),
          "subcord_within_insertion_median_matches": float(np.median(within_in)),
          "subcord_crosspass_median_matches": float(np.median(block)),
          "subcord_crosspass_max_matches": int(block.max()),
          "colmap_tri_angle_median": float(np.median(tri))}
(OUT / "diag_tables.json").write_text(json.dumps(tables, indent=2))
print("\n=== PER-FRAME (frame seg | COLMAP reg,nobs | GlueMap reg,outlier | medMatches) ===")
for r in rows:
    print(f"  f{r['frame']:<5d} {r['segment']:<11s} | C:{'Y' if r['colmap_registered'] else 'n'} {r['colmap_nobs']:>3d}obs "
          f"| G:{'Y' if r['gluemap_registered'] else 'n'}{'!OUT' if r['gluemap_outlier'] else '   '} "
          f"| med_matches={r['median_matches_to_others']}")
print(f"\nCOLMAP failed to register: {tables['colmap_failed_frames']}")
print(f"GlueMap outlier (mis-placed) cams: {tables['gluemap_outlier_frames']}")
print(f"saved figures + diag_tables.json -> {OUT}")
