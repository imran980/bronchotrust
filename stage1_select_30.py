"""STAGE 1: select 30 frames spanning blade->glottis->subglottis->trachea->tube.

Segment ranges (assumed defaults; user can swap individual frames at the gate):
  - blade            : f0   .. f15    (laryngoscope blade, pre-glottis)
  - glottis          : f15  .. f80    (vocal cords; L1=f20)
  - subglottic       : f80  .. f700   (descent into upper trachea; L2=f140)
  - trachea (mid+distal): f700 .. f3300 (long stretch; L3=f2250)
  - transition+tube  : f3300 .. f3484  (trachea -> 4-mm cylinder, packed)

Counts:
  blade=2, glottis=4, subglottic=6, trachea=6, transition+tube=12 (incl. ~3 inside-tube)
  total = 30

Spacing: within each segment, pick frames at equal cumulative-grayscale-diff
intervals between consecutive VALID frames (proxy for optical flow). This avoids
clustering on stretches where the camera is momentarily still.

Outputs (runs/gated2/2_v2/):
  stage1_frames/seg_<NAME>_f<NNNNN>.png   : 30 full-res PNGs
  contact_30.png                          : 5x6 labeled contact sheet
  stage1_selection.json                   : per-frame metadata
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/2_v2")
FRAMES = OUT / "stage1_frames"
FRAMES.mkdir(parents=True, exist_ok=True)

SEGMENTS = [
    # (name, lo_inclusive, hi_exclusive, count)
    ("blade",       0,    15,   2),
    ("glottis",     15,   80,   4),
    ("subglottic",  80,   700,  6),
    ("trachea",     700,  3300, 6),
    ("transition_tube", 3300, 3485, 12),  # includes ~3 inside-tube tail
]

# Diff downsample factor for the cumulative-motion proxy
DIFF_DS = 6  # 1920/6 = 320


def main():
    stats = np.load(OUT / "frame_stats.npz")
    valid = stats["valid"].astype(bool)
    L_mean = stats["L_mean"]
    N = len(valid)
    print(f"loaded {N} frames, valid={int(valid.sum())}")

    # Pass A: compute consecutive grayscale-absdiff for every valid frame pair
    # (we need this for spacing inside any segment that has many valid frames).
    cap = cv2.VideoCapture(str(VIDEO))
    diff_per_frame = np.zeros(N, dtype=np.float32)  # diff between frame i-1 and i (valid only)
    last_g = None
    last_fi = -1
    for fi in range(N):
        ok, fr = cap.read()
        if not ok:
            break
        if not valid[fi]:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // DIFF_DS, g.shape[0] // DIFF_DS),
                       interpolation=cv2.INTER_AREA)
        if last_g is not None:
            d = float(np.abs(g.astype(np.int16) - last_g.astype(np.int16)).mean())
            diff_per_frame[fi] = d
        last_g = g
        last_fi = fi
        if fi % 500 == 0:
            print(f"  diff scan {fi}/{N}  d={diff_per_frame[fi]:.2f}")
    cap.release()

    # Pass B: select per segment by equal-cumulative-diff partitions.
    selection = []  # list of dicts
    for name, lo, hi, k in SEGMENTS:
        # candidate valid indices in [lo, hi)
        cand = np.array([fi for fi in range(lo, min(hi, N)) if valid[fi]],
                        dtype=int)
        if len(cand) < k:
            print(f"WARN: segment {name} only has {len(cand)} valid; "
                  f"using all + may shortfall ({k} requested)")
            picked = cand.tolist()
        elif k == 1:
            picked = [int(cand[len(cand) // 2])]
        else:
            # cumulative diff along cand (diff_per_frame is keyed by absolute fi)
            d = diff_per_frame[cand]
            d[0] = 0.0
            cum = np.cumsum(d)
            total = float(cum[-1])
            if total < 1e-6:
                picked = np.linspace(cand[0], cand[-1], k).round().astype(int).tolist()
            else:
                targets = np.linspace(0, total, k)
                pick_ix = []
                for t in targets:
                    j = int(np.searchsorted(cum, t))
                    j = min(max(j, 0), len(cand) - 1)
                    pick_ix.append(j)
                # dedup while preserving order
                seen = set(); uniq = []
                for j in pick_ix:
                    if j not in seen:
                        seen.add(j); uniq.append(j)
                # if dedup shrank, fill greedily with max-gap insertions
                while len(uniq) < k:
                    # find largest gap between consecutive picked positions
                    uniq_sorted = sorted(uniq)
                    best_gap, best_j = -1, None
                    for a, b in zip(uniq_sorted[:-1], uniq_sorted[1:]):
                        gap = b - a
                        if gap > best_gap:
                            best_gap, best_j = gap, (a + b) // 2
                    if best_j is None or best_j in seen:
                        break
                    seen.add(best_j); uniq.append(best_j)
                picked = [int(cand[j]) for j in sorted(uniq)]
        for fi in picked[:k]:
            selection.append({"segment": name, "frame_idx": int(fi)})
        print(f"{name:18s} {lo:5d}..{hi:5d}  picked {len(picked[:k])}: "
              f"{[s for s in picked[:k]]}")

    # If short, top up; if over, trim (shouldn't happen).
    if len(selection) != 30:
        print(f"WARN: total selected = {len(selection)} (expected 30)")

    # Pass C: extract full-res frames and build contact sheet
    want = sorted({s["frame_idx"] for s in selection})
    cap = cv2.VideoCapture(str(VIDEO))
    seg_of = {s["frame_idx"]: s["segment"] for s in selection}
    frame_imgs = {}
    for fi in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            print(f"WARN failed to read {fi}")
            continue
        seg = seg_of[fi]
        p = FRAMES / f"seg_{seg}_f{fi:05d}.png"
        cv2.imwrite(str(p), fr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        frame_imgs[fi] = fr
        print(f"  saved {p.name}")
    cap.release()

    # Build 5x6 grid (5 rows, 6 cols). Order: by segment, then by frame_idx.
    seg_order = [s[0] for s in SEGMENTS]
    sel_sorted = sorted(selection,
                        key=lambda s: (seg_order.index(s["segment"]),
                                       s["frame_idx"]))
    cols, rows = 6, 5
    THUMB_W = 480
    THUMB_H = int(round(THUMB_W * 1080 / 1920))  # 270
    LABEL_H = 40
    cell_w = THUMB_W
    cell_h = THUMB_H + LABEL_H
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    seg_color = {
        "blade":            (60, 60, 240),
        "glottis":          (40, 200, 240),
        "subglottic":       (90, 220, 90),
        "trachea":          (220, 160, 50),
        "transition_tube":  (240, 90, 200),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, s in enumerate(sel_sorted[:rows * cols]):
        fi = s["frame_idx"]
        seg = s["segment"]
        r = idx // cols; c = idx % cols
        y0 = r * cell_h; x0 = c * cell_w
        img = frame_imgs.get(fi)
        if img is None:
            continue
        t = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = t
        ly0 = y0 + THUMB_H
        canvas[ly0:ly0 + LABEL_H, x0:x0 + cell_w] = seg_color[seg]
        txt = f"{seg}  f{fi:05d}"
        cv2.putText(canvas, txt, (x0 + 10, ly0 + 28),
                    font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (50, 50, 50), 1)
    out_png = OUT / "contact_30.png"
    cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"contact sheet: {canvas.shape} -> {out_png}")

    summary = {
        "video": str(VIDEO),
        "segments": [
            {"name": n, "lo": lo, "hi": hi, "count": k}
            for (n, lo, hi, k) in SEGMENTS
        ],
        "selection": sel_sorted,
        "frame_dir": str(FRAMES),
        "contact_path": str(out_png),
        "n_selected": len(selection),
        "diff_proxy": "consecutive-valid-frame absdiff at 1/{} grayscale".format(DIFF_DS),
        "cylinder_inner_diameter_mm": 4.0,
        "transition_range": [3300, 3469],
        "user_note": "blade<->glottis disjoint; subglottic is f80..700; trachea f700..3300; transition+tube f3300..3484 (12 frames, ~3 inside tube)",
    }
    (OUT / "stage1_selection.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT/'stage1_selection.json'}")


if __name__ == "__main__":
    main()
