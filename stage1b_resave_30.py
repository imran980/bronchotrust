"""STAGE 1b: re-extract the 30 frames using SEQUENTIAL decoding (no seek).

Why: stage2b_index_check.json shows 21/30 of the Stage-1-saved frames are
mismatched against the sequential-read frame at the same index. Cause: OpenCV's
cap.set(CAP_PROP_POS_FRAMES, fi) does not land on the requested H.264 frame.

Changes:
  - Drop f56 (glottis 4->3, still bracketed by f35 and f76 around L1=f20)
  - Drop f3403 (transition_tube cluster near f3396) and f3475 (adjacent to f3474)
  - Add 3 sub-cord frames: f100, f125, f150 (between f83 and the existing f164,
    bracketing L2 prox subglottis = f140)

New counts: blade 2, glottis 3, subglottic 9, trachea 6, transition_tube 10 = 30

Linkage density f3300..3469 unchanged (still 8 frames in this zone after drops).

Also writes:
  intrinsics_pinned.json  (NEW OPENCV calibration, RMS 0.444 px)
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/2_v2")
FRAMES = OUT / "stage1_frames"

# (drop, add) sets applied to the prior selection
DROP = {56, 3403, 3475}
ADD = [
    ("subglottic", 100),
    ("subglottic", 125),
    ("subglottic", 150),
]

# NEW OPENCV calibration (pinned)
PINNED = {
    "model": "OPENCV (k1,k2,p1=0,p2=0,k3=0)",
    "image_size": [1920, 1080],
    "fx": 835.9017208676929,
    "fy": 836.5695555793831,
    "cx": 984.2944316622135,
    "cy": 458.73851870454973,
    "k1": -0.10604076425522294,
    "k2": -0.023907873153004158,
    "p1": 0.0, "p2": 0.0,
    "rms_reproj_px": 0.44369500806171164,
    "hfov_deg": 97.90580332076522,
    "vfov_deg": 65.68393922498068,
    "n_frames_used": 79,
    "source": "stage2_calib_compare on 2_v2 Calibration Video, "
              "findChessboardCornersSB + cornerSubPix, CALIB_FIX_K3 | "
              "CALIB_ZERO_TANGENT_DIST | USE_INTRINSIC_GUESS",
}

SEG_COLOR = {
    "blade":            (60, 60, 240),
    "glottis":          (40, 200, 240),
    "subglottic":       (90, 220, 90),
    "trachea":          (220, 160, 50),
    "transition_tube":  (240, 90, 200),
}


def img_hash(img):
    return hashlib.sha1(img.tobytes()).hexdigest()


def main():
    sel0 = json.loads((OUT / "stage1_selection.json").read_text())["selection"]
    keep = [s for s in sel0 if s["frame_idx"] not in DROP]
    print(f"prior: {len(sel0)};  after drop: {len(keep)};  + add {len(ADD)} -> "
          f"{len(keep) + len(ADD)}")
    new_sel = keep + [{"segment": seg, "frame_idx": fi} for seg, fi in ADD]
    seg_order = ["blade", "glottis", "subglottic", "trachea", "transition_tube"]
    new_sel.sort(key=lambda s: (seg_order.index(s["segment"]), s["frame_idx"]))
    assert len(new_sel) == 30, f"expected 30 got {len(new_sel)}"

    want = sorted({s["frame_idx"]: s["segment"] for s in new_sel}.items(),
                  key=lambda kv: kv[0])
    target_indices = [fi for fi, _ in want]
    seg_of = {fi: seg for fi, seg in want}
    print("targets:", target_indices)

    # Wipe old frames
    if FRAMES.exists():
        for p in FRAMES.iterdir():
            p.unlink()
    FRAMES.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    remaining = sorted(target_indices)
    next_i = 0
    fi = 0
    saved = {}  # fi -> path
    sample_hashes = {}
    frame_imgs = {}
    while next_i < len(remaining):
        ok, fr = cap.read()
        if not ok or fi >= N:
            break
        if fi == remaining[next_i]:
            seg = seg_of[fi]
            p = FRAMES / f"seg_{seg}_f{fi:05d}.png"
            cv2.imwrite(str(p), fr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            saved[fi] = str(p)
            sample_hashes[fi] = img_hash(fr)
            frame_imgs[fi] = fr
            print(f"  saved {p.name}  sha1[:10]={sample_hashes[fi][:10]}")
            next_i += 1
        fi += 1
    cap.release()
    print(f"saved {len(saved)} frames (sequential extraction)")

    # Build 5x6 contact sheet
    sel_sorted = new_sel  # already segment-grouped, frame-sorted
    cols, rows = 6, 5
    THUMB_W = 480
    THUMB_H = int(round(THUMB_W * 1080 / 1920))  # 270
    LABEL_H = 40
    cell_w = THUMB_W
    cell_h = THUMB_H + LABEL_H
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, s in enumerate(sel_sorted[:rows * cols]):
        fi = s["frame_idx"]; seg = s["segment"]
        r = idx // cols; c = idx % cols
        y0 = r * cell_h; x0 = c * cell_w
        img = frame_imgs.get(fi)
        if img is None:
            continue
        t = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = t
        ly0 = y0 + THUMB_H
        canvas[ly0:ly0 + LABEL_H, x0:x0 + cell_w] = SEG_COLOR[seg]
        txt = f"{seg}  f{fi:05d}"
        cv2.putText(canvas, txt, (x0 + 10, ly0 + 28),
                    font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (50, 50, 50), 1)
    out_png = OUT / "contact_30.png"
    cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"contact sheet: {canvas.shape} -> {out_png}")

    # Save updated selection + index-check note + pinned intrinsics
    new_summary = {
        "video": str(VIDEO),
        "extraction_method": "SEQUENTIAL cap.read() (no seek)",
        "drop": sorted(DROP),
        "add": [{"segment": s, "frame_idx": fi} for s, fi in ADD],
        "selection": sel_sorted,
        "n_selected": len(sel_sorted),
        "frame_dir": str(FRAMES),
        "contact_path": str(out_png),
        "linkage_zone_indices_f3300_3469":
            [s["frame_idx"] for s in sel_sorted
             if 3299 <= s["frame_idx"] <= 3469],
        "inside_tube_indices":
            [s["frame_idx"] for s in sel_sorted
             if s["frame_idx"] >= 3470],
        "subglottic_frames":
            [s["frame_idx"] for s in sel_sorted if s["segment"] == "subglottic"],
        "user_added_subcord": [fi for _, fi in ADD],
    }
    (OUT / "stage1_selection.json").write_text(json.dumps(new_summary, indent=2))
    print(f"saved {OUT/'stage1_selection.json'}")

    (OUT / "intrinsics_pinned.json").write_text(json.dumps(PINNED, indent=2))
    print(f"saved {OUT/'intrinsics_pinned.json'}")


if __name__ == "__main__":
    main()
