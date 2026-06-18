"""Phase 1: full-video annotated contact sheets + auto-proposed anatomy ranges.

For each First-15 video: one sequential decode pass computes per-frame lumen
morphology (dark-lumen area fraction, aspect ratio, blob count), proposes
glottis / subglottis / trachea boundaries on the insertion descent, and renders
a FULL-video contact sheet (every Nth valid frame, labeled by sequential index)
with the proposed bands colored. STOP after this for user approval.

Auto-proposal heuristic (transparent; frame fractions only to PROPOSE, never for
final metrics):
  airway_start = first sustained lit run.
  glottis      = airway_start -> round-lumen onset (aspect<1.8 & area rising).
  subglottis   = round-lumen onset -> recovery past the first lumen-area minimum
                 (the narrowing just below the cords).
  trachea      = subglottis end -> carina (first sustained 2-blob frame) or end.
Confidence is flagged when the heuristic falls back to fractions.

NO SfM, NO metrics here. Outputs runs/retriage_first15/anatomy/<video>/.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import cv2, numpy as np

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/anatomy")
VIDEOS = ["2-V2.MP4", "5_v1_1.mp4", "5_v1_2.mp4", "7-V1.MP4", "10_v2.mp4",
          "13_v2.mp4", "15_v2.mp4", "16_v1.mp4", "18-V1.MP4", "25-V1.MP4",
          "31-V1.MP4", "32-V2.MP4", "33-V1.MP4"]
DARK_L = 12.0
REGION_COLOR = {"glottis": (240, 200, 40), "subglottis": (80, 220, 80),
                "trachea": (40, 150, 230), "other": (110, 110, 110)}


def content_mask(video, n=50):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(3)); H = int(cap.get(4))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab = cv2.connectedComponents(m)
    if nlab > 1:
        m = ((lab == 1 + np.argmax([(lab == i).sum() for i in range(1, nlab)])).astype(np.uint8)) * 255
    m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=4)
    return m > 0, (W, H)


def feats(fr, mask, roi_area):
    L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) * 100 / 255
    roivals = L[mask]
    med = float(np.median(roivals)) if len(roivals) else 0.0
    thr = max(8.0, 0.45 * med)
    dark = ((L < thr) & mask).astype(np.uint8)
    nb, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    blobs = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nb)
                    if stats[i, cv2.CC_STAT_AREA] > 0.005 * roi_area], reverse=True)
    if blobs:
        a, i = blobs[0]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        return med, a / roi_area, max(w, h) / max(min(w, h), 1), len(blobs)
    return med, 0.0, 0.0, 0


def smooth(x, k=9):
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def propose(fi, Lroi, area, aspect, nblob):
    """Return dict of proposed ranges (in frame index) + confidence + notes.

    Order matters: find glottis end FIRST (round-lumen onset), then search for
    the carina only AFTER that (early cords show as multiple dark blobs and must
    not be mistaken for a carina)."""
    valid = Lroi >= DARK_L
    vidx = np.where(valid)[0]
    notes = []
    if len(vidx) < 40:
        return {"airway_start": None, "glottis": None, "subglottis": None,
                "trachea": None, "confidence": "none", "notes": ["too few valid frames"]}
    runs = np.split(vidx, np.where(np.diff(vidx) > 3)[0] + 1)
    runs = [r for r in runs if len(r) >= 20]
    start = int(runs[0][0]) if runs else int(vidx[0])
    end_run = int(runs[0][-1]) if runs else int(vidx[-1])
    run = np.arange(start, end_run + 1)
    n = len(run)
    a_s = smooth(area[run]); asp_s = smooth(aspect[run]); nb_s = smooth(nblob[run].astype(float))
    conf = "heuristic"

    # 1) glottis end = first round-lumen onset (aspect<1.8 & area>0.03) sustained 8
    g_pos = None
    for k in range(n - 8):
        if np.all(asp_s[k:k + 8] < 1.8) and np.all(a_s[k:k + 8] > 0.03):
            g_pos = k; break
    if g_pos is None:
        g_pos = int(0.10 * n); conf = "low(frac)"; notes.append("glottis: fraction fallback")
    g_end = int(run[g_pos])

    # 2) carina: first sustained 2-blob AFTER glottis + 15% margin
    c_lo = g_pos + max(8, int(0.15 * n))
    carina = None
    for k in range(c_lo, n - 6):
        if np.all(nb_s[k:k + 6] >= 1.6):
            carina = int(run[k]); break
    if carina is not None:
        ins_end_pos = int(np.searchsorted(run, carina)); notes.append(f"carina(2-lumen) ~f{carina}")
    else:
        ins_end_pos = min(n - 1, g_pos + int(0.55 * n)); notes.append("no carina; trachea capped at fraction (adjust)")
    ins_end = int(run[ins_end_pos])

    # 3) subglottis = narrowing: lumen-area local min after glottis within next 45% of insertion
    win_hi = min(ins_end_pos, g_pos + int(0.45 * n) + 1)
    if win_hi - g_pos >= 5:
        seg = a_s[g_pos:win_hi]
        mpos = g_pos + int(np.argmin(seg)); amin = a_s[mpos]
        rec = win_hi - 1
        for k in range(mpos, win_hi):
            if a_s[k] > 1.4 * max(amin, 1e-3):
                rec = k; break
        s_start, s_end = int(run[g_pos]), int(run[rec])
        notes.append(f"subglottic narrowing min ~f{int(run[mpos])}")
    else:
        s_start = g_end; s_end = int(run[min(g_pos + int(0.12 * n), n - 1)])
        conf = "low(frac)"; notes.append("subglottis: fraction fallback")

    return {"airway_start": start, "glottis": [start, g_end],
            "subglottis": [s_start, s_end], "trachea": [s_end, ins_end],
            "insertion_end": ins_end, "carina": carina,
            "confidence": conf, "notes": notes}


def region_of(fi, P):
    for r in ("glottis", "subglottis", "trachea"):
        rg = P.get(r)
        if rg and rg[0] <= fi <= rg[1]:
            return r
    return "other"


def render_sheet(video, thumbs, P, out_png, WH):
    cols = 12
    cap = 264
    if len(thumbs) > cap:
        step = int(np.ceil(len(thumbs) / cap))
        thumbs = thumbs[::step]
    tw = 220; th = int(round(tw * WH[1] / WH[0])); lh = 26
    rows = (len(thumbs) + cols - 1) // cols
    head = 120
    canvas = np.full((head + rows * (th + lh), cols * tw, 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # header
    cv2.putText(canvas, f"{video.name}  proposed anatomy (ADJUST subglottis as needed)",
                (12, 32), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    txt = (f"glottis {P.get('glottis')}   subglottis {P.get('subglottis')}   "
           f"trachea {P.get('trachea')}   carina {P.get('carina')}   conf={P.get('confidence')}")
    cv2.putText(canvas, txt, (12, 64), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    x = 12
    for r in ("glottis", "subglottis", "trachea", "other"):
        cv2.rectangle(canvas, (x, 80), (x + 24, 100), REGION_COLOR[r][::-1], -1)
        cv2.putText(canvas, r, (x + 30, 96), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        x += 180
    for idx, (fi, t) in enumerate(thumbs):
        rr, cc = divmod(idx, cols)
        y0 = head + rr * (th + lh); x0 = cc * tw
        canvas[y0:y0 + th, x0:x0 + tw] = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        col = REGION_COLOR[region_of(fi, P)][::-1]
        canvas[y0 + th:y0 + th + lh, x0:x0 + tw] = col
        cv2.putText(canvas, f"f{fi:05d}", (x0 + 6, y0 + th + 19), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + tw - 1, y0 + th + lh - 1), (60, 60, 60), 1)
    H = canvas.shape[0]
    if H > 16000:
        s = 16000 / H
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * s), 16000), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def process(fname):
    video = DATA / fname
    od = OUT / video.stem
    od.mkdir(parents=True, exist_ok=True)
    mask, WH = content_mask(video)
    roi_area = float(mask.sum())
    cap = cv2.VideoCapture(str(video))
    Lroi, area, aspect, nblob, shas = [], [], [], [], []
    thumbs = []
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        med, af, asp, nb = feats(fr, mask, roi_area)
        Lroi.append(med); area.append(af); aspect.append(asp); nblob.append(nb)
        shas.append(hashlib.sha1(cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA).tobytes()).hexdigest())
        if med >= DARK_L and (len([t for t in thumbs]) == 0 or fi - thumbs[-1][0] >= 8):
            thumbs.append((fi, cv2.resize(fr, (220, int(220 * WH[1] / WH[0])), interpolation=cv2.INTER_AREA)))
        fi += 1
    cap.release()
    Lroi = np.array(Lroi); area = np.array(area); aspect = np.array(aspect); nblob = np.array(nblob)
    P = propose(np.arange(len(Lroi)), Lroi, area, aspect, nblob)
    render_sheet(video, thumbs, P, od / "contact_anatomy.png", WH)
    rec = {"video": fname, "n_frames": int(len(Lroi)),
           "n_valid": int((Lroi >= DARK_L).sum()), "WH": list(WH),
           "proposed": P, "contact_sheet": str(od / "contact_anatomy.png")}
    (od / "proposed_ranges.json").write_text(json.dumps(rec, indent=2))
    # save features for Phase 2 reuse (frame-accurate, sequential)
    np.savez(od / "features.npz", Lroi=Lroi, area=area, aspect=aspect, nblob=nblob)
    (od / "frame_sha.json").write_text(json.dumps(shas))
    print(f"{fname:<12} valid={rec['n_valid']:>5}/{rec['n_frames']:<5} "
          f"glottis={P.get('glottis')} subglottis={P.get('subglottis')} "
          f"trachea={P.get('trachea')} conf={P.get('confidence')}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", default=",".join(VIDEOS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for v in args.videos.split(","):
        if v not in VIDEOS:
            continue
        print(f"\n#### {v} ####", flush=True)
        try:
            recs.append(process(v))
        except Exception as e:
            print(f"{v} ERROR {type(e).__name__}: {e}", flush=True)
            recs.append({"video": v, "error": str(e)})
    (OUT / "all_proposed.json").write_text(json.dumps(recs, indent=2))
    print(f"\nsaved {OUT/'all_proposed.json'}")


if __name__ == "__main__":
    main()
