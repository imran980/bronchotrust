"""Video-integrity check across the bronchoscopy cohort folders.

For every .MP4/.mp4 file in:
  First 15 Videos/
  First 13 Videos Trimmed/

report:
  - byte size  (<10 KB -> STUB)
  - codec_name, width x height, container-claimed nb_frames + duration
    (via ffprobe)
  - ffmpeg full-decode error count and any error messages quoted
  - full sequential decode (OpenCV cv2.VideoCapture) producing:
      * actual decodable frame count
      * count of black/near-black frames (LAB-L<5 on whole frame)
      * count of frozen runs (consecutive identical SHA1 hashes) and longest
      * any frame index where cv2.read returned False before EOF

Verdict per file:
  STUB        size < 10 KB
  CORRUPTED   any ffmpeg decode error, OR decodable < 95% of claimed
              (excluding STUB), OR any read-failure mid-stream
  INTACT      otherwise

ALSO call out: 15_v2, 2_v1, 5_v1, 7_v1, 13_v2, 2_v2 explicitly.

NO RECONSTRUCTION. Pure file-integrity diagnostic.
"""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

DATASET = Path("/home/mi3dr/dataset/validation-videos")
FOLDERS = [
    DATASET / "First 15 Videos",
    DATASET / "First 13 Videos Trimmed",
]
FFMPEG = Path("/home/mi3dr/.conda/envs/depth-eval/bin/ffmpeg")
FFPROBE = Path("/home/mi3dr/.conda/envs/depth-eval/bin/ffprobe")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/gated2/video_integrity.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

STUB_BYTES = 10_000
BLACK_L_THRESH = 5.0


def ffprobe_info(p: Path) -> dict:
    cmd = [str(FFPROBE), "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(p)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"_error": "ffprobe timeout"}
    if r.returncode != 0:
        return {"_error": r.stderr.strip()[:500]}
    try:
        d = json.loads(r.stdout)
    except Exception as e:
        return {"_error": f"ffprobe-json: {e}"}
    out: dict = {}
    fmt = d.get("format", {})
    out["duration_sec"] = float(fmt.get("duration", 0.0) or 0.0)
    out["bit_rate"] = int(fmt.get("bit_rate", 0) or 0)
    streams = d.get("streams", [])
    vs = [s for s in streams if s.get("codec_type") == "video"]
    if not vs:
        out["_error"] = "no video stream"
        return out
    s = vs[0]
    out["codec_name"] = s.get("codec_name")
    out["pix_fmt"] = s.get("pix_fmt")
    out["width"] = s.get("width")
    out["height"] = s.get("height")
    out["nb_frames_container"] = (
        int(s["nb_frames"]) if s.get("nb_frames", "0").isdigit() else None
    )
    afr = s.get("avg_frame_rate", "0/0")
    try:
        n, m = afr.split("/")
        out["avg_frame_rate"] = (float(n) / float(m)) if float(m) else 0.0
    except Exception:
        out["avg_frame_rate"] = None
    return out


def ffmpeg_decode_pass(p: Path) -> dict:
    """Full decode to null muxer; count error lines."""
    cmd = [str(FFMPEG), "-v", "error", "-i", str(p), "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"timeout": True, "err_lines": [], "n_err": -1}
    err = r.stderr.strip()
    lines = [ln for ln in err.split("\n") if ln.strip()]
    return {"timeout": False, "err_lines": lines[:30], "n_err": len(lines),
            "returncode": r.returncode}


def cv2_full_scan(p: Path) -> dict:
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {"opened": False}
    N_claim = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_read = 0
    n_black = 0
    last_hash = None
    cur_freeze = 0
    longest_freeze = 0
    freeze_runs = []
    read_fail_indices = []
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            # End reached, or a mid-stream failure
            if fi < N_claim - 1 and N_claim > 0:
                # potential mid-stream read failure; tolerate one trailing fail
                read_fail_indices.append(fi)
            break
        n_read += 1
        # Black detection (LAB L mean on whole frame)
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        if float(L.mean()) < BLACK_L_THRESH:
            n_black += 1
        # Frame hash (downsample for speed)
        small = cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA)
        h = hashlib.sha1(small.tobytes()).hexdigest()
        if last_hash is not None and h == last_hash:
            cur_freeze += 1
        else:
            if cur_freeze >= 2:
                freeze_runs.append({"end_frame": fi - 1,
                                     "length": cur_freeze + 1})
                longest_freeze = max(longest_freeze, cur_freeze + 1)
            cur_freeze = 0
        last_hash = h
        fi += 1
    if cur_freeze >= 2:
        freeze_runs.append({"end_frame": fi - 1,
                             "length": cur_freeze + 1})
        longest_freeze = max(longest_freeze, cur_freeze + 1)
    cap.release()
    return {
        "opened": True,
        "claimed_n": int(N_claim),
        "decodable_n": int(n_read),
        "WxH": [W, H],
        "fps": fps,
        "n_black": int(n_black),
        "n_freeze_runs": int(len(freeze_runs)),
        "longest_freeze_run": int(longest_freeze),
        "read_fail_indices": read_fail_indices[:20],
    }


def verdict(size_b, fp, dec, scan) -> tuple[str, str]:
    if size_b < STUB_BYTES:
        return "STUB", f"size {size_b} B < {STUB_BYTES}"
    reasons = []
    if dec.get("n_err", 0) > 0:
        reasons.append(f"ffmpeg decode errors n={dec['n_err']}")
    if dec.get("timeout"):
        reasons.append("ffmpeg decode timeout")
    if scan.get("opened") is False:
        reasons.append("cv2 cannot open")
    elif scan.get("claimed_n") and scan.get("decodable_n"):
        ratio = scan["decodable_n"] / max(scan["claimed_n"], 1)
        if ratio < 0.95:
            reasons.append(
                f"decodable {scan['decodable_n']}/{scan['claimed_n']} ({ratio*100:.1f}%)")
    if scan.get("read_fail_indices"):
        reasons.append(
            f"mid-stream read fails at {scan['read_fail_indices'][:5]}")
    if reasons:
        return "CORRUPTED", "; ".join(reasons)
    return "INTACT", "all checks ok"


def main():
    files: list[dict] = []
    for folder in FOLDERS:
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in {".mp4", ".mov"}:
                continue
            print(f"\n>>> {folder.name}/{p.name}")
            size_b = p.stat().st_size
            print(f"  size: {size_b} bytes")
            if size_b < STUB_BYTES:
                files.append({
                    "folder": folder.name, "name": p.name,
                    "size_bytes": size_b, "verdict": "STUB",
                    "verdict_reason": f"size {size_b} B < {STUB_BYTES}",
                })
                continue
            fp = ffprobe_info(p)
            print(f"  ffprobe: codec={fp.get('codec_name')}, "
                  f"{fp.get('width')}x{fp.get('height')}, "
                  f"nb_frames={fp.get('nb_frames_container')}, "
                  f"dur={fp.get('duration_sec'):.2f}s, "
                  f"fps={fp.get('avg_frame_rate')}")
            dec = ffmpeg_decode_pass(p)
            print(f"  ffmpeg decode-pass: n_err={dec.get('n_err')}")
            if dec.get("n_err", 0) > 0:
                for ln in dec["err_lines"][:5]:
                    print(f"     ! {ln}")
            scan = cv2_full_scan(p)
            print(f"  cv2 scan: opened={scan.get('opened')}  "
                  f"claimed={scan.get('claimed_n')}  "
                  f"decodable={scan.get('decodable_n')}  "
                  f"black={scan.get('n_black')}  "
                  f"freeze_runs={scan.get('n_freeze_runs')}  "
                  f"longest_freeze={scan.get('longest_freeze_run')}")
            v, why = verdict(size_b, fp, dec, scan)
            print(f"  VERDICT: {v}  ({why})")
            files.append({
                "folder": folder.name, "name": p.name,
                "size_bytes": size_b,
                "ffprobe": fp,
                "ffmpeg_decode_pass": dec,
                "cv2_scan": scan,
                "verdict": v,
                "verdict_reason": why,
            })

    OUT.write_text(json.dumps(files, indent=2))
    print(f"\nsaved {OUT}")

    # Table
    print("\n=== INTEGRITY TABLE ===")
    print(f"{'folder':<32}{'name':<22}{'size_MB':>10}{'codec':>10}"
          f"{'WxH':>14}{'claim':>8}{'decod':>8}{'errs':>7}{'black':>7}"
          f"{'froz':>7}{'verdict':>11}")
    for r in files:
        size_mb = r['size_bytes'] / 1e6
        codec = r.get('ffprobe', {}).get('codec_name', '-')
        wh = r.get('ffprobe', {})
        wh_str = f"{wh.get('width','-')}x{wh.get('height','-')}" \
                  if r.get('ffprobe') else "-"
        clm = r.get('cv2_scan', {}).get('claimed_n', '-')
        dec = r.get('cv2_scan', {}).get('decodable_n', '-')
        errs = r.get('ffmpeg_decode_pass', {}).get('n_err', '-')
        blk = r.get('cv2_scan', {}).get('n_black', '-')
        frz = r.get('cv2_scan', {}).get('longest_freeze_run', '-')
        v = r.get('verdict', '?')
        print(f"{r['folder']:<32}{r['name']:<22}{size_mb:>10.3f}"
              f"{str(codec):>10}{wh_str:>14}"
              f"{str(clm):>8}{str(dec):>8}{str(errs):>7}{str(blk):>7}"
              f"{str(frz):>7}{v:>11}")


if __name__ == "__main__":
    main()
