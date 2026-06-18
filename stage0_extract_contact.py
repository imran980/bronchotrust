"""STAGE 0 (gated plan): integrity + sequential frame extraction + contact sheet.

Parametrized for any video (default: 15_v2 trimmed). Reuses the proven
content-ROI / dark-frame logic; fixes the over-strict "any ffmpeg error =>
CORRUPTED" verdict by CLASSIFYING each ffmpeg error line:

    cosmetic : "non-monotonic ... dts to muxer" (timestamp dedup, harmless)
    real     : corrupt / concealing / error while decoding / invalid data

Hard-rule compliance (CLAUDE.md gated plan, failure-signature list):
  - Frame EXTRACTION uses sequential cv2.read() with a running index, never
    cap.set(CAP_PROP_POS_FRAMES) (which drifts on H.264). POS_FRAMES is used
    ONLY for the cosmetic bezel-mask sampling, where approximate positions
    are fine.
  - Each decoded frame gets a SHA1 (of a 32x18 downsample) stored in
    frame_stats.json so later stages can hash-verify a re-extracted frame
    against the canonical sequential frame at that index (no index drift).

Outputs (--out, default runs/gluemap_15v2/15_v2/stage0/):
  content_mask.png    binary bronchoscope content ROI (bezel dropped)
  frame_stats.json    per-frame: idx, L_mean (over ROI), valid, sha1
  frame_stats.npz     same arrays, compact
  contact_sheet.png   labeled contact sheet of every Nth valid frame
  stage0_summary.json integrity verdict + error classification + params
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

DEF_VIDEO = "/home/mi3dr/dataset/validation-videos/First 13 Videos Trimmed/15_v2.mp4"
DEF_OUT = "/home/mi3dr/projects/bronchotrust/runs/gluemap_15v2/15_v2/stage0"
DEF_FFMPEG = "/home/mi3dr/.conda/envs/depth-eval/bin/ffmpeg"

# Error classification keywords
REAL_KEYS = ("corrupt", "concealing", "error while decoding", "invalid data",
             "no frame", "decode_slice_header error")
COSMETIC_KEYS = ("non monotonic", "non-monotonic", "monoton", "dts to muxer")


def frame_sha1(fr: np.ndarray) -> str:
    small = cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def ffmpeg_classify(video: Path, ffmpeg: str) -> dict:
    """Full decode to null muxer; classify every stderr error line."""
    cmd = [ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return {"timeout": True, "n_err": -1, "n_real": -1, "n_cosmetic": -1,
                "sample_real": [], "sample_cosmetic": []}
    lines = [ln for ln in r.stderr.strip().split("\n") if ln.strip()]
    real, cosmetic, other = [], [], []
    for ln in lines:
        low = ln.lower()
        is_cos = any(k in low for k in COSMETIC_KEYS)
        is_real = any(k in low for k in REAL_KEYS)
        if is_real and not is_cos:
            real.append(ln)
        elif is_cos:
            cosmetic.append(ln)
        else:
            other.append(ln)
    return {
        "timeout": False,
        "n_err": len(lines),
        "n_real": len(real),
        "n_cosmetic": len(cosmetic),
        "n_other": len(other),
        "sample_real": real[:10],
        "sample_cosmetic": cosmetic[:5],
        "sample_other": other[:10],
    }


def detect_content_mask(video: Path, n_samples: int = 60):
    """OR 'lit' masks over n_samples evenly-spread frames -> content ROI.

    POS_FRAMES seek is acceptable here: the mask only needs the lit-region
    geometry, which is stable regardless of exact frame index.
    """
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_ix = np.linspace(0, max(N - 1, 0), n_samples).astype(int)
    cum = np.zeros((H, W), dtype=np.int32)
    for fi in sample_ix:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        cum += (fr.mean(axis=2) > 8).astype(np.int32)
    cap.release()
    mask = cum >= max(int(0.20 * n_samples), 5)
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    nlab, lab = cv2.connectedComponents(m)
    if nlab > 1:
        sizes = [int((lab == i).sum()) for i in range(1, nlab)]
        keep = int(np.argmax(sizes)) + 1
        m = ((lab == keep).astype(np.uint8)) * 255
    m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=4)
    return m > 0, (W, H)


def sequential_scan(video: Path, mask: np.ndarray, dark_thresh: float,
                    sample_stride: int, thumb_w: int):
    """One SEQUENTIAL pass: L_mean over ROI, valid flag, sha1, thumbs."""
    cap = cv2.VideoCapture(str(video))
    N_claim = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    thumb_h = int(round(thumb_w * H / W))
    mask_f = mask.astype(bool)

    L_mean, valid, shas = [], [], []
    thumbs: list[tuple[int, np.ndarray]] = []
    n_black = 0
    last_hash = None
    cur_freeze = longest_freeze = n_freeze_runs = 0
    read_fail_idx = []
    valid_count = 0
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            if N_claim > 0 and fi < N_claim - 1:
                read_fail_idx.append(fi)
            break
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        m_roi = float(L[mask_f].mean())
        if float(L.mean()) < 5.0:
            n_black += 1
        h = frame_sha1(fr)
        if last_hash is not None and h == last_hash:
            cur_freeze += 1
        else:
            if cur_freeze >= 2:
                n_freeze_runs += 1
                longest_freeze = max(longest_freeze, cur_freeze + 1)
            cur_freeze = 0
        last_hash = h

        L_mean.append(m_roi)
        shas.append(h)
        is_valid = m_roi >= dark_thresh
        valid.append(bool(is_valid))
        if is_valid:
            if valid_count % sample_stride == 0:
                tw = cv2.resize(fr, (thumb_w, thumb_h),
                                interpolation=cv2.INTER_AREA)
                thumbs.append((fi, tw))
            valid_count += 1
        if fi % 300 == 0:
            print(f"  {fi}/{N_claim}  L_roi={m_roi:.1f}  valid={valid_count}",
                  flush=True)
        fi += 1
    if cur_freeze >= 2:
        n_freeze_runs += 1
        longest_freeze = max(longest_freeze, cur_freeze + 1)
    cap.release()
    return {
        "claimed_n": N_claim, "decodable_n": fi, "WH": [W, H], "fps": fps,
        "thumb_h": thumb_h, "n_black": n_black, "n_freeze_runs": n_freeze_runs,
        "longest_freeze": longest_freeze, "read_fail_idx": read_fail_idx[:20],
        "L_mean": np.array(L_mean, np.float32), "valid": np.array(valid, bool),
        "sha1": shas, "thumbs": thumbs,
    }


def build_contact_sheet(thumbs, thumb_w, thumb_h, cols, out_png: Path):
    label_h = 28
    n = len(thumbs)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = thumb_w, thumb_h + label_h
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (fi, t) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * cell_w
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = t
        canvas[y0 + thumb_h:y0 + cell_h, x0:x0 + cell_w] = (255, 255, 255)
        cv2.putText(canvas, f"f{fi:05d}", (x0 + 8, y0 + thumb_h + 20),
                    font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (200, 200, 200), 1)
    Hc = canvas.shape[0]
    if Hc > 14000:
        s = 14000 / Hc
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * s), 14000),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return canvas.shape, rows


def verdict(scan, ff) -> tuple[str, str]:
    reasons = []
    if ff.get("n_real", 0) > 0:
        reasons.append(f"{ff['n_real']} REAL ffmpeg errors")
    if ff.get("timeout"):
        reasons.append("ffmpeg decode timeout")
    if scan["claimed_n"] > 0:
        ratio = scan["decodable_n"] / scan["claimed_n"]
        if ratio < 0.95:
            reasons.append(
                f"decodable {scan['decodable_n']}/{scan['claimed_n']} "
                f"({ratio*100:.1f}%)")
    if scan["read_fail_idx"]:
        reasons.append(f"mid-stream read fails {scan['read_fail_idx'][:5]}")
    if reasons:
        return "CORRUPTED", "; ".join(reasons)
    note = ""
    if ff.get("n_cosmetic", 0) > 0:
        note = (f" ({ff['n_cosmetic']} cosmetic DTS warnings ignored)")
    return "INTACT", "all checks ok" + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEF_VIDEO)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--ffmpeg", default=DEF_FFMPEG)
    ap.add_argument("--dark-thresh", type=float, default=30.0)
    ap.add_argument("--sample-stride", type=int, default=15)
    ap.add_argument("--thumb-w", type=int, default=320)
    ap.add_argument("--cols", type=int, default=12)
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"VIDEO: {video}\nOUT:   {out}\n")

    print("[1/4] ffmpeg full-decode integrity classification ...")
    ff = ffmpeg_classify(video, args.ffmpeg)
    print(f"   n_err={ff['n_err']}  real={ff['n_real']}  "
          f"cosmetic={ff['n_cosmetic']}  other={ff.get('n_other')}")
    for ln in ff.get("sample_real", []):
        print(f"   ! REAL: {ln[:120]}")

    print("[2/4] content ROI (bezel) detection ...")
    mask, (W, H) = detect_content_mask(video)
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    cv2.imwrite(str(out / "content_mask.png"), mask.astype(np.uint8) * 255)
    print(f"   WxH={W}x{H}  ROI px={int(mask.sum())} "
          f"({100*mask.sum()/(W*H):.1f}%)  bbox={bbox}")

    print("[3/4] SEQUENTIAL decode pass (L_mean, sha1, thumbs) ...")
    scan = sequential_scan(video, mask, args.dark_thresh,
                           args.sample_stride, args.thumb_w)
    n_valid = int(scan["valid"].sum())
    print(f"   decodable={scan['decodable_n']}/{scan['claimed_n']}  "
          f"valid={n_valid}  black={scan['n_black']}  "
          f"freeze_runs={scan['n_freeze_runs']}  thumbs={len(scan['thumbs'])}")

    print("[4/4] contact sheet + save ...")
    shape, rows = build_contact_sheet(
        scan["thumbs"], args.thumb_w, scan["thumb_h"], args.cols,
        out / "contact_sheet.png")

    idx = np.arange(scan["decodable_n"], dtype=np.int32)
    np.savez(out / "frame_stats.npz", frame_idx=idx,
             L_mean=scan["L_mean"], valid=scan["valid"])
    frame_stats = [
        {"idx": int(i), "L_mean": round(float(scan["L_mean"][i]), 2),
         "valid": bool(scan["valid"][i]), "sha1": scan["sha1"][i]}
        for i in range(scan["decodable_n"])]
    (out / "frame_stats.json").write_text(json.dumps(frame_stats))

    v, why = verdict(scan, ff)
    summary = {
        "video": str(video), "WH": [W, H], "fps": scan["fps"],
        "integrity_verdict": v, "integrity_reason": why,
        "ffmpeg": {k: ff[k] for k in ("n_err", "n_real", "n_cosmetic",
                                      "n_other", "sample_real",
                                      "sample_cosmetic")},
        "claimed_n": scan["claimed_n"], "decodable_n": scan["decodable_n"],
        "n_valid": n_valid, "n_black": scan["n_black"],
        "n_freeze_runs": scan["n_freeze_runs"],
        "longest_freeze": scan["longest_freeze"],
        "content_bbox_x0y0x1y1": bbox,
        "dark_thresh": args.dark_thresh, "sample_stride": args.sample_stride,
        "n_thumbs": len(scan["thumbs"]), "cols": args.cols, "rows": rows,
        "thumb_size": [args.thumb_w, scan["thumb_h"]],
        "thumb_frame_indices": [int(fi) for fi, _ in scan["thumbs"]],
        "contact_sheet_shape": list(shape),
    }
    (out / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== STAGE 0 VERDICT: {v} ({why}) ===")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
