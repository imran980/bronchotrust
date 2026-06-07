"""Verify that the frame-index space is consistent across stages.

Stage 0 indexed frames by SEQUENTIAL cap.read() order (0..N-1).
Stage 1's diff scan also used SEQUENTIAL cap.read().
Stage 1's extraction used cap.set(CAP_PROP_POS_FRAMES, fi) + cap.read().

H.264 seek can land on a different frame than the sequential read at the same
index. If that happened, the labels `f00000.png ... f03479.png` saved in
stage1_frames/ do NOT correspond to the same frames that frame_stats.npz
tagged 'valid'.

Tests:
  1. For each of the 30 selected indices, sequentially read the video up to
     that index and compare its SHA1 pixel hash to the hash of the saved
     stage1_frames/seg_*_f<fi>.png. Report all mismatches.
  2. Sanity-check the dark-drop boundary: a 'valid' frame should not be dark;
     report (frame_idx, L_mean) for sample of saved-extracted frames.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/2_v2")
FRAMES = OUT / "stage1_frames"


def img_hash(img):
    return hashlib.sha1(img.tobytes()).hexdigest()


def main():
    sel = json.loads((OUT / "stage1_selection.json").read_text())["selection"]
    want = sorted({s["frame_idx"]: s["segment"] for s in sel}.items(),
                  key=lambda kv: kv[0])
    targets = [fi for fi, _ in want]
    seg_of = {fi: seg for fi, seg in want}
    print(f"checking {len(targets)} frames")

    stats = np.load(OUT / "frame_stats.npz")
    L_mean = stats["L_mean"]
    valid = stats["valid"].astype(bool)

    cap = cv2.VideoCapture(str(VIDEO))
    target_set = set(targets)
    seq_hash = {}
    seq_mean = {}
    fi = 0
    next_target = min(target_set) if target_set else None
    remaining = set(target_set)
    while remaining:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in remaining:
            seq_hash[fi] = img_hash(fr)
            seq_mean[fi] = float(fr.mean())
            remaining.discard(fi)
        fi += 1
    cap.release()

    rows = []
    n_mismatch = 0
    for fi in targets:
        seg = seg_of[fi]
        path = FRAMES / f"seg_{seg}_f{fi:05d}.png"
        img = cv2.imread(str(path))
        if img is None:
            rows.append((fi, seg, "MISSING", None, None, None, None, False))
            n_mismatch += 1
            continue
        saved_h = img_hash(img)
        seq_h = seq_hash.get(fi)
        match = saved_h == seq_h
        if not match:
            n_mismatch += 1
        rows.append((
            fi, seg,
            "OK" if match else "MISMATCH",
            saved_h[:10],
            seq_h[:10] if seq_h else "n/a",
            float(L_mean[fi]),
            bool(valid[fi]),
            match,
        ))

    print(f"\n{'fi':>5}  {'segment':<18}  {'status':<10}  "
          f"{'saved':<11}  {'seq':<11}  {'L_mean':>7}  valid")
    for r in rows:
        fi, seg, st, sh, qh, lm, vld, _ = r
        lm_s = f"{lm:7.2f}" if lm is not None else "  n/a "
        v_s = "Y" if vld else "N"
        print(f"{fi:5d}  {seg:<18}  {st:<10}  {sh or '':<11}  {qh or '':<11}  "
              f"{lm_s}  {v_s}")

    print(f"\nTOTAL: {len(rows)} frames, mismatches: {n_mismatch}")

    out = {
        "n_targets": len(targets),
        "n_mismatch": n_mismatch,
        "all_match": n_mismatch == 0,
        "rows": [
            {"frame_idx": fi, "segment": seg, "status": st,
             "saved_sha1_10": sh, "seq_sha1_10": qh,
             "L_mean_from_stats": lm, "valid_from_stats": vld,
             "match": ok}
            for (fi, seg, st, sh, qh, lm, vld, ok) in rows
        ],
    }
    (OUT / "stage2b_index_check.json").write_text(json.dumps(out, indent=2))
    print(f"saved {OUT/'stage2b_index_check.json'}")


if __name__ == "__main__":
    main()
