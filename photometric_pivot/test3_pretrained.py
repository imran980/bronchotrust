"""TEST 3 -- pretrained monocular depth quick probe on 2_V2 (timeboxed).
Endoscopy-specific models (PPSNet/LightDepth) need repo clone + weights; within the timebox the
cleanly-runnable option is Depth-Anything-V2-Small (a GENERAL, NON-endoscopic monocular depth
model). It is used ONLY as a quick learned-depth probe -- NOT ground truth, NOT endoscopy-trained,
relative (scale/shift-ambiguous) depth only. Back-project the lumen boundary with the model depth,
fit a relative ring, and check the proximal-vs-distal ordering / ratio vs the COLMAP sanity result.
NO mm, NO clinical claim."""
from __future__ import annotations
import json
import numpy as np, cv2, torch
from PIL import Image
from transformers import pipeline
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C

K, DIST = C.intrinsics(); KINV = np.linalg.inv(K)
PIPE = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if torch.cuda.is_available() else -1)


def depth_map(bgr):
    out = PIPE(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    d = out["predicted_depth"]
    d = d.squeeze().float().cpu().numpy() if hasattr(d, "cpu") else np.array(out["depth"], float)
    return cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def measure(fr):
    from test2_inverse_square import lumen_boundary
    bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g)
    bnd = lumen_boundary(g, fov)
    if bnd is None: return None
    dm = depth_map(bgr)
    # orient so the lumen CENTER is FAR (large depth). Compare center vs wall raw values.
    ctr = bnd.mean(0); H, W = g.shape
    cy, cx = int(np.clip(ctr[1], 0, H - 1)), int(np.clip(ctr[0], 0, W - 1))
    center_val = float(np.median(dm[max(0, cy - 20):cy + 20, max(0, cx - 20):cx + 20]))
    wall_val = float(np.median(dm[fov]))
    depth = dm if center_val > wall_val else (dm.max() - dm)   # ensure center(far) = large
    ii = np.clip(bnd[:, 1].astype(int), 0, H - 1); jj = np.clip(bnd[:, 0].astype(int), 0, W - 1)
    db = depth[ii, jj]
    und = cv2.undistortPoints(bnd.reshape(-1, 1, 2).astype(np.float32), K, DIST, P=K).reshape(-1, 2)
    rays = (KINV @ np.c_[und, np.ones(len(und))].T).T; rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    P = rays * db[:, None]; Pc = P - P.mean(0); _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    x, y = Pc @ Vt[0], Pc @ Vt[1]; rr = np.hypot(x, y); rm = np.median(rr); keep = (rr > 0.4 * rm) & (rr < 2.0 * rm)
    if keep.sum() < 10: return None
    try:
        (_, _), (MA, ma), _ = cv2.fitEllipse(np.c_[x[keep], y[keep]].astype(np.float32))
        csa = float(np.pi * (MA / 2) * (ma / 2)); dce = float(2 * np.sqrt((MA / 2) * (ma / 2)))
    except Exception:
        return None
    return dict(frame=fr, CSA_rel=csa, DCE_rel=dce, oriented=bool(center_val <= wall_val), depth=dm, bnd=bnd, ctr=ctr.tolist())


def main():
    P = [m for m in (measure(f) for f in C.PROX) if m]
    D = [m for m in (measure(f) for f in C.DIST) if m]
    def stat(rows, k): v = np.array([r[k] for r in rows]); return dict(median=float(np.median(v)), cov_pct=round(float(100 * np.std(v) / max(np.median(v), 1e-9)), 1))
    csa_ratio = stat(P, "CSA_rel")["median"] / stat(D, "CSA_rel")["median"]
    dce_ratio = stat(P, "DCE_rel")["median"] / stat(D, "DCE_rel")["median"]
    out = dict(test="pretrained depth (Depth-Anything-V2-Small, GENERAL/non-endoscopic; quick probe, NOT GT)",
               model="depth-anything/Depth-Anything-V2-Small-hf", endoscopy_specific=False,
               proximal={"n": len(P), "CSA_rel": stat(P, "CSA_rel"), "DCE_rel": stat(P, "DCE_rel")},
               distal={"n": len(D), "CSA_rel": stat(D, "CSA_rel"), "DCE_rel": stat(D, "DCE_rel")},
               CSA_ratio_prox_over_dist=round(csa_ratio, 3), DCE_ratio_prox_over_dist=round(dce_ratio, 3),
               implied_CSA_obstruction_pct=round(100 * (1 - csa_ratio), 1),
               COLMAP_sanity={"CSA_ratio": 0.68, "DCE_ratio": 0.82, "obstruction_CSA_pct": 32.0},
               ordering_prox_lt_dist=bool(csa_ratio < 1.0), ratio_near_colmap=bool(0.5 <= csa_ratio <= 0.85),
               stable=bool(stat(P, "CSA_rel")["cov_pct"] <= 25 and stat(D, "CSA_rel")["cov_pct"] <= 25))
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.bar([0, 1], [stat(P, "CSA_rel")["median"], stat(D, "CSA_rel")["median"]], color=["tab:red", "tab:blue"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["proximal", "distal"]); ax.set_ylabel("relative CSA (learned depth)")
    ax.set_title(f"pretrained-depth rel CSA (prox/dist={csa_ratio:.2f})\nCOLMAP sanity≈0.68", fontsize=10)
    for k, (rows, ttl) in enumerate([(P, "proximal"), (D, "distal")]):
        r = rows[len(rows) // 2]; ax = fig.add_subplot(1, 3, 2 + k)
        dm = r["depth"]; ax.imshow(dm, cmap="turbo"); ax.plot(r["bnd"][:, 0], r["bnd"][:, 1], ".", ms=2, c="w")
        ax.set_title(f"{ttl} f{r['frame']} depth (turbo)\nrel CSA={r['CSA_rel']:.1f}", fontsize=9); ax.axis("off")
    fig.suptitle("TEST 3 — pretrained (general) depth probe: relative CSA (NOT endoscopy-trained, NOT GT)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(C.OUT / "test3_pretrained.png", dpi=120); plt.close(fig)
    clean = {k: v for k, v in out.items() if k not in ("depth",)}
    (C.OUT / "test3_pretrained.json").write_text(json.dumps(clean, indent=2))
    print("=== TEST 3 ==="); print(json.dumps({k: v for k, v in clean.items() if not isinstance(v, dict) or k in ("proximal", "distal", "COLMAP_sanity")}, indent=2))
    print(f"figure -> {C.OUT}/test3_pretrained.png")


if __name__ == "__main__":
    main()
