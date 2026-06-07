"""STAGE 0: extract + dark-drop + contact sheet on untrimmed 2_v2.

Pass A: auto-detect the bronchoscope content ROI (drop the letterbox/bezel)
        by OR-ing 'g>8' masks over a sample of 60 frames evenly spread.
Pass B: read every frame; compute LAB-L mean over the content mask;
        drop frames with mean < DARK_L_THRESH;
        collect every ~15th valid frame as a labeled thumbnail.

Outputs (runs/gated2/2_v2/):
  content_mask.png   : binary mask of the bronchoscope content ROI
  frame_stats.npz    : frame_idx (N,), L_mean (N,), valid (N,)
  contact_sheet.png  : labeled contact sheet of valid frames
  stage0_summary.json: counts + parameters
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/2_v2")
OUT.mkdir(parents=True, exist_ok=True)

DARK_L_THRESH = 30.0
SAMPLE_STRIDE = 15
THUMB_W = 320
LABEL_H = 28
COLS = 12
N_BEZEL_SAMPLES = 60


def detect_content_mask(video: Path):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_ix = np.linspace(0, N - 1, N_BEZEL_SAMPLES).astype(int)
    cum = np.zeros((H, W), dtype=np.int32)
    for fi in sample_ix:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        g = fr.mean(axis=2)
        cum += (g > 8).astype(np.int32)
    cap.release()
    # require >=20% of samples to be lit at a pixel -> content
    mask = (cum >= max(int(0.20 * N_BEZEL_SAMPLES), 5))
    # tighten: morphological open then close to remove specks; take largest CC
    m_u8 = (mask.astype(np.uint8)) * 255
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # largest connected component
    nlab, lab = cv2.connectedComponents(m_u8)
    if nlab > 1:
        sizes = [int((lab == i).sum()) for i in range(1, nlab)]
        keep = int(np.argmax(sizes)) + 1
        m_u8 = ((lab == keep).astype(np.uint8)) * 255
    # erode 4 px to avoid bezel halo
    m_u8 = cv2.erode(m_u8, np.ones((5, 5), np.uint8), iterations=4)
    return m_u8 > 0, (W, H)


def main():
    print("detecting content ROI...")
    mask, (W, H) = detect_content_mask(VIDEO)
    print(f"  W x H = {W} x {H}; content pixels = {int(mask.sum())} "
          f"({100*mask.sum()/(W*H):.1f}%)")
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    print(f"  content bbox (x0,y0,x1,y1) = {bbox}")
    cv2.imwrite(str(OUT / "content_mask.png"), (mask.astype(np.uint8) * 255))

    cap = cv2.VideoCapture(str(VIDEO))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    THUMB_H = int(round(THUMB_W * H / W))
    L_mean = np.zeros(N, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    thumbs: list[tuple[int, np.ndarray]] = []
    valid_count = 0

    mask_f = mask.astype(bool)
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok or fi >= N:
            break
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        m = float(L[mask_f].mean())
        L_mean[fi] = m
        if m >= DARK_L_THRESH:
            valid[fi] = True
            if valid_count % SAMPLE_STRIDE == 0:
                tw = cv2.resize(fr, (THUMB_W, THUMB_H),
                                interpolation=cv2.INTER_AREA)
                thumbs.append((fi, tw))
            valid_count += 1
        if fi % 300 == 0:
            print(f"  {fi}/{N}  L_roi={m:.1f}  valid={valid_count}  "
                  f"thumbs={len(thumbs)}")
        fi += 1
    cap.release()

    n_total = fi
    n_valid = int(valid.sum())
    print(f"\ntotal read: {n_total}  valid (L_roi>={DARK_L_THRESH}): "
          f"{n_valid}  thumbs: {len(thumbs)}")

    np.savez(OUT / "frame_stats.npz",
             frame_idx=np.arange(n_total, dtype=np.int32),
             L_mean=L_mean[:n_total], valid=valid[:n_total])

    # Contact sheet
    n_thumbs = len(thumbs)
    rows = (n_thumbs + COLS - 1) // COLS
    cell_w = THUMB_W
    cell_h = THUMB_H + LABEL_H
    canvas = np.full((rows * cell_h, COLS * cell_w, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (fi, t) in enumerate(thumbs):
        r = idx // COLS
        c = idx % COLS
        y0 = r * cell_h
        x0 = c * cell_w
        canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = t
        ly0 = y0 + THUMB_H
        canvas[ly0:ly0 + LABEL_H, x0:x0 + cell_w] = (255, 255, 255)
        txt = f"f{fi:05d}"
        cv2.putText(canvas, txt, (x0 + 8, ly0 + 20),
                    font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (200, 200, 200), 1)
    out_png = OUT / "contact_sheet.png"
    Hc, Wc = canvas.shape[:2]
    max_h = 14000
    if Hc > max_h:
        scale = max_h / Hc
        canvas_s = cv2.resize(canvas, (int(Wc * scale), int(Hc * scale)),
                              interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_png), canvas_s,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"contact sheet (downscaled): {canvas_s.shape}")
    else:
        cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"contact sheet: {canvas.shape}")

    thumb_indices = [int(fi) for fi, _ in thumbs]
    summary = {
        "video": str(VIDEO),
        "n_frames_total": int(n_total),
        "fps": float(fps),
        "WH": [int(W), int(H)],
        "content_bbox_x0y0x1y1": list(bbox),
        "content_mask_path": str(OUT / "content_mask.png"),
        "dark_L_thresh": DARK_L_THRESH,
        "n_valid": int(n_valid),
        "sample_stride": int(SAMPLE_STRIDE),
        "n_thumbs": int(n_thumbs),
        "cols": int(COLS),
        "rows": int(rows),
        "thumb_size": [int(THUMB_W), int(THUMB_H)],
        "thumb_frame_indices": thumb_indices,
        "contact_sheet_path": str(out_png),
        "frame_stats_path": str(OUT / "frame_stats.npz"),
    }
    (OUT / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT/'stage0_summary.json'}")


if __name__ == "__main__":
    main()
