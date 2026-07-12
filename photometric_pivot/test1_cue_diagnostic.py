"""TEST 1 -- photometric cue diagnostic on 2_V2. For proximal/narrowest vs distal reference frames:
mask specular + black bezel, find lumen center, inspect the radial brightness profile from the lumen
center outward, and judge whether brightness/falloff is consistent enough to imply depth ordering.
Output: intensity-profile plots, specular/bezel mask overlays, a short verdict. NO mm, NO GT."""
from __future__ import annotations
import json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C


def radial_profile(b, center, fov, nbin=40):
    H, W = b.shape; ys, xs = np.mgrid[0:H, 0:W]
    r = np.hypot(xs - center[0], ys - center[1])
    rmax = np.percentile(r[fov], 88)                     # ignore the far bezel edge
    edges = np.linspace(0, rmax, nbin + 1); prof, rr = [], []
    for i in range(nbin):
        m = fov & (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(b)
        if m.sum() > 30: prof.append(float(np.nanmedian(b[m]))); rr.append(0.5 * (edges[i] + edges[i + 1]))
    return np.array(rr), np.array(prof), rmax


def analyze(frames, tag):
    rows = []
    for fr in frames:
        bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g); spec = C.specular_mask(g, fov)
        b = C.clean_brightness(bgr, g, fov, spec); ctr = C.lumen_center(g, fov)
        rr, prof, rmax = radial_profile(b, ctr, fov)
        # monotonicity: fraction of outward steps that increase (dark center -> bright wall)
        mono = float((np.diff(prof) > 0).mean()) if len(prof) > 3 else 0.0
        # depth-order proxy: brightness rises then plateaus/peaks; the radius where brightness
        # reaches 60% of its span ~ where the near wall dominates (relates to lumen aperture size)
        span = prof.max() - prof.min() + 1e-9; thr = prof.min() + 0.6 * span
        r60 = rr[np.argmax(prof >= thr)] if (prof >= thr).any() else np.nan
        rows.append(dict(frame=fr, center=[float(ctr[0]), float(ctr[1])], mono_frac=round(mono, 3),
                         r60_px=round(float(r60), 1), spec_px=int(spec.sum()), rr=rr.tolist(), prof=prof.tolist()))
    return rows


def main():
    C.OUT.mkdir(exist_ok=True)
    res = {"prox": analyze(C.PROX, "prox"), "dist": analyze(C.DIST, "dist")}
    # figure: profiles + a masked overlay per group
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(2, 2, 1)
    for r in res["prox"]: ax.plot(r["rr"], r["prof"], "-", color="tab:red", alpha=.6)
    for r in res["dist"]: ax.plot(r["rr"], r["prof"], "-", color="tab:blue", alpha=.6)
    ax.plot([], [], "tab:red", label="proximal/narrowest"); ax.plot([], [], "tab:blue", label="distal reference")
    ax.set_xlabel("radius from lumen center (px)"); ax.set_ylabel("median corrected brightness")
    ax.set_title("radial brightness profiles (dark center -> bright wall = near-light depth cue)"); ax.legend(); ax.grid(alpha=.3)
    ax = fig.add_subplot(2, 2, 2)
    mp = [r["mono_frac"] for r in res["prox"]]; md = [r["mono_frac"] for r in res["dist"]]
    ax.boxplot([mp, md], labels=["prox", "dist"]); ax.axhline(0.75, c="g", ls="--", label="monotone 0.75")
    ax.set_ylabel("outward-increasing fraction (monotonicity)"); ax.set_title("brightness monotonicity per group"); ax.legend(fontsize=8)
    # mask overlays (one prox, one dist)
    for k, fr in enumerate([C.PROX[len(C.PROX) // 2], C.DIST[len(C.DIST) // 2]]):
        bgr, g = C.load_gray(fr); fov, _ = C.fov_mask(g); spec = C.specular_mask(g, fov); ctr = C.lumen_center(g, fov)
        ov = bgr.copy(); ov[~fov] = (0, 60, 0); ov[spec] = (0, 255, 255)
        cv2.circle(ov, (int(ctr[0]), int(ctr[1])), 10, (255, 0, 255), -1)
        ax = fig.add_subplot(2, 2, 3 + k); ax.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        ax.set_title(f"f{fr} masks: green=bezel, yellow=specular, magenta=lumen center", fontsize=9); ax.axis("off")
    fig.suptitle("TEST 1 — 2_V2 photometric cue diagnostic (scene units; not a clinical result)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(C.OUT / "test1_cue.png", dpi=120); plt.close(fig)

    mono_all = np.array([r["mono_frac"] for r in res["prox"] + res["dist"]])
    r60_prox = np.median([r["r60_px"] for r in res["prox"]]); r60_dist = np.median([r["r60_px"] for r in res["dist"]])
    verdict = dict(median_monotonicity=round(float(np.median(mono_all)), 3),
                   all_frames_monotone_ge_0p7=bool((mono_all >= 0.7).all()),
                   r60_prox_px=round(float(r60_prox), 1), r60_dist_px=round(float(r60_dist), 1),
                   note="r60 = radius where brightness reaches 60% of its span; a near-light cue exists if profiles rise monotonically outward and are consistent within a group")
    verdict["cue_present"] = bool(verdict["median_monotonicity"] >= 0.7)
    res["verdict"] = verdict
    (C.OUT / "test1_cue.json").write_text(json.dumps(res, indent=2))
    print("=== TEST 1 verdict ==="); print(json.dumps(verdict, indent=2))
    print(f"figure -> {C.OUT}/test1_cue.png")


if __name__ == "__main__":
    main()
