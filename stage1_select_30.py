"""STAGE 1 (gated plan): curate ~36 frames for 15_v2, DUAL-PASS sub-cord.

Clinical insight (user, confirmed on contact sheet): 15_v2 images the
subglottis TWICE -- insertion (f50-105) and withdrawal (f1500-1740). The two
opposing/maneuvering viewpoints of the same subglottic ring are the parallax
source (triage cone 17.5 deg, lateral/along 7.15). So we sample the sub-cord
region DENSELY from BOTH passes, with a trachea backbone (descending f105-400
and ascending f1300-1500, which image overlapping trachea wall) so the two
passes fuse into ONE connected reconstruction.

Improvements over the old stage1 + stage1b two-step:
  - SINGLE sequential extraction (no cap.set(POS_FRAMES) -> no H.264 drift).
  - Within each window, pick k frames at equal cumulative-motion intervals
    (optical-flow proxy), then SNAP each pick to the locally sharpest frame
    (Laplacian variance over the ROI) to avoid motion-blur / mucus frames.
  - HASH-VERIFY every saved frame's 32x18 SHA1 against the Stage-0 canonical
    frame_stats.json hash at that index (report matches/total; must be N/N).

Outputs (--out, default runs/gluemap_15v2/15_v2/stage1/):
  images/f<NNNNN>.png      curated frames (named by sequential index)
  contact_curated.png      labeled contact sheet (segment + index)
  stage1_selection.json    selection + per-window picks + hash-verify result
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DEF_VIDEO = "/home/mi3dr/dataset/validation-videos/First 13 Videos Trimmed/15_v2.mp4"
DEF_STAGE0 = "/home/mi3dr/projects/bronchotrust/runs/gluemap_15v2/15_v2/stage0"
DEF_OUT = "/home/mi3dr/projects/bronchotrust/runs/gluemap_15v2/15_v2/stage1"

# (name, lo_inclusive, hi_exclusive, count) -- dual-pass, disjoint windows
DEFAULT_WINDOWS = [
    ("glottis_in",  0,    50,   3),
    ("subglot_in",  50,   105,  9),   # ROI dense (insertion)
    ("trachea_in",  105,  400,  6),
    ("trachea_out", 1300, 1500, 6),
    ("subglot_out", 1500, 1740, 9),   # ROI dense (withdrawal)
    ("glottis_out", 1740, 1822, 3),
]

SEG_COLOR = {
    "glottis_in":  (40, 200, 240), "subglot_in":  (90, 220, 90),
    "trachea_in":  (220, 160, 50), "trachea_out": (200, 120, 40),
    "subglot_out": (60, 180, 60),  "glottis_out": (30, 160, 200),
}
DIFF_DS = 6
SHARP_SNAP = 3  # +/- candidate frames to search for the sharpest


def frame_sha1(fr):
    return hashlib.sha1(
        cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA).tobytes()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEF_VIDEO)
    ap.add_argument("--stage0", default=DEF_STAGE0)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    stage0 = Path(args.stage0)
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)

    stats = np.load(stage0 / "frame_stats.npz")
    valid = stats["valid"].astype(bool)
    N = len(valid)
    canon = json.loads((stage0 / "frame_stats.json").read_text())
    canon_sha = {r["idx"]: r["sha1"] for r in canon}
    mask = cv2.imread(str(stage0 / "content_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    print(f"frames N={N} valid={int(valid.sum())}; ROI px={int(mask.sum())}")

    # ---- Pass 1: sequential motion (diff) + sharpness (Laplacian var) ----
    diff = np.zeros(N, np.float32)
    sharp = np.zeros(N, np.float32)
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cap = cv2.VideoCapture(str(video))
    last_g = None
    for fi in range(N):
        ok, fr = cap.read()
        if not ok:
            break
        if not valid[fi]:
            last_g = None
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        roi = g[y0:y1, x0:x1]
        sharp[fi] = cv2.Laplacian(roi, cv2.CV_64F).var()
        gd = cv2.resize(g, (g.shape[1] // DIFF_DS, g.shape[0] // DIFF_DS),
                        interpolation=cv2.INTER_AREA)
        if last_g is not None:
            diff[fi] = float(np.abs(gd.astype(np.int16)
                                    - last_g.astype(np.int16)).mean())
        last_g = gd
        if fi % 400 == 0:
            print(f"  scan {fi}/{N}", flush=True)
    cap.release()

    # ---- select per window ----
    selection = []
    for name, lo, hi, k in DEFAULT_WINDOWS:
        cand = np.array([fi for fi in range(lo, min(hi, N)) if valid[fi]], int)
        if len(cand) == 0:
            print(f"WARN {name}: no valid frames in [{lo},{hi})")
            continue
        if len(cand) <= k:
            picked = cand.tolist()
        else:
            d = diff[cand].copy()
            d[0] = 0.0
            cum = np.cumsum(d)
            tot = float(cum[-1])
            targets = (np.linspace(0, tot, k) if tot > 1e-6
                       else np.linspace(0, len(cand) - 1, k))
            base = ([int(np.clip(np.searchsorted(cum, t), 0, len(cand) - 1))
                     for t in targets] if tot > 1e-6
                    else [int(round(t)) for t in targets])
            # snap each base pick to the locally sharpest candidate
            snapped, seen = [], set()
            for j in base:
                lo_j, hi_j = max(0, j - SHARP_SNAP), min(len(cand), j + SHARP_SNAP + 1)
                local = cand[lo_j:hi_j]
                best = int(local[int(np.argmax(sharp[local]))])
                jj = int(np.where(cand == best)[0][0])
                if jj not in seen:
                    seen.add(jj)
                    snapped.append(jj)
            # fill if dedup shrank: largest-gap midpoints
            while len(snapped) < k:
                s = sorted(snapped)
                gaps = [(b - a, (a + b) // 2) for a, b in zip(s[:-1], s[1:])]
                if not gaps:
                    break
                _, mid = max(gaps)
                if mid in seen:
                    break
                seen.add(mid)
                snapped.append(mid)
            picked = sorted(int(cand[j]) for j in snapped[:k])
        for fi in picked:
            selection.append({"segment": name, "frame_idx": int(fi),
                              "sharp": round(float(sharp[fi]), 1)})
        print(f"{name:12s} [{lo:4d},{hi:4d}) valid={len(cand):3d} -> {len(picked)}: {picked}")

    selection.sort(key=lambda s: s["frame_idx"])
    targets = [s["frame_idx"] for s in selection]
    seg_of = {s["frame_idx"]: s["segment"] for s in selection}
    print(f"\ntotal curated: {len(selection)}")

    # ---- Pass 2: sequential extract + hash-verify against Stage-0 canon ----
    for p in img_dir.glob("*.png"):
        p.unlink()
    cap = cv2.VideoCapture(str(video))
    want = sorted(targets)
    ni = 0
    fi = 0
    imgs = {}
    matches = 0
    hash_report = []
    while ni < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if fi == want[ni]:
            h = frame_sha1(fr)
            ok_h = (canon_sha.get(fi) == h)
            matches += int(ok_h)
            hash_report.append({"idx": fi, "match": ok_h})
            cv2.imwrite(str(img_dir / f"f{fi:05d}.png"), fr,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            imgs[fi] = fr
            ni += 1
        fi += 1
    cap.release()
    print(f"hash-verify vs Stage-0 canonical: {matches}/{len(want)} match")

    # ---- contact sheet ----
    cols = 6
    rows = (len(selection) + cols - 1) // cols
    tw, th, lh = 480, 270, 40
    canvas = np.full((rows * (th + lh), cols * tw, 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, s in enumerate(selection):
        fi = s["frame_idx"]
        r, c = divmod(idx, cols)
        yy, xx = r * (th + lh), c * tw
        im = imgs.get(fi)
        if im is None:
            continue
        canvas[yy:yy + th, xx:xx + tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        canvas[yy + th:yy + th + lh, xx:xx + tw] = SEG_COLOR.get(s["segment"], (120, 120, 120))
        cv2.putText(canvas, f"{s['segment']} f{fi:05d}", (xx + 8, yy + th + 28),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (xx, yy), (xx + tw - 1, yy + th + lh - 1), (50, 50, 50), 1)
    cv2.imwrite(str(out / "contact_curated.png"), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    n_sub = sum(1 for s in selection if s["segment"].startswith("subglot"))
    summary = {
        "video": str(video), "n_curated": len(selection),
        "windows": [{"name": n, "lo": lo, "hi": hi, "count": k}
                    for (n, lo, hi, k) in DEFAULT_WINDOWS],
        "selection": selection,
        "subcord_frames": [s["frame_idx"] for s in selection
                           if s["segment"].startswith("subglot")],
        "n_subcord": n_sub,
        "hash_verify": {"matches": matches, "total": len(want),
                        "all_match": matches == len(want),
                        "detail": hash_report},
        "images_dir": str(img_dir), "contact": str(out / "contact_curated.png"),
        "extraction": "SEQUENTIAL cv2.read() (no POS_FRAMES)",
        "selection_method": "equal-cumulative-motion + Laplacian-sharpness snap",
    }
    (out / "stage1_selection.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== STAGE 1: {len(selection)} frames, {n_sub} sub-cord, "
          f"hash {matches}/{len(want)} -> {out}")


if __name__ == "__main__":
    main()
