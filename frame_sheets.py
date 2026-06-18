"""Per video: ONE dense contact sheet (~110 thumbnails) with frame numbers
clearly overlaid, the proposed anatomy bands colored, and the frames the model
SELECTS for the 3D recon MARKED in-place (magenta border + 'SEL').

Density is biased toward the upper-airway/descent (glottis->trachea) so the
subglottic transition is easy to identify, with sparser coverage of the deep
bronchi / withdrawal.

Outputs runs/retriage_first15/frames/<stem>/contact_dense.png
Run in depth-eval env.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np

DATA = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")
ANAT = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/anatomy")
OUT = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/frames")
VIDEOS = ["2-V2.MP4", "5_v1_1.mp4", "5_v1_2.mp4", "7-V1.MP4", "10_v2.mp4",
          "13_v2.mp4", "15_v2.mp4", "16_v1.mp4", "18-V1.MP4", "25-V1.MP4",
          "31-V1.MP4", "32-V2.MP4", "33-V1.MP4"]
DARK_L = 12.0
EARLY_VALID = 700
NPICK = 36
N_UPPER = 58      # dense thumbs in the upper-airway/descent zone
N_REST = 30       # sparser thumbs over the rest of the video
RC = {"glottis": (40, 200, 240), "subglottis": (80, 220, 80),
      "trachea": (230, 150, 40), "other": (110, 110, 110)}  # BGR
SEL_BORDER = (220, 60, 220)  # magenta


def content_mask(video, n=50):
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    m = cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
                  np.ones((5, 5), np.uint8), iterations=4)
    return m > 0, (W, H)


def region_of(fi, P):
    for r in ("glottis", "subglottis", "trachea"):
        rg = P.get(r)
        if rg and rg[0] <= fi <= rg[1]:
            return r
    return "other"


def select_recon(valid, diff):
    vidx = np.where(valid)[0]
    if len(vidx) == 0:
        return []
    cand = vidx[:EARLY_VALID]
    d = diff[cand].copy(); d[0] = 0
    cd = np.cumsum(d); tot = cd[-1]
    if tot > 1e-6:
        tg = np.linspace(0, tot, NPICK)
        return sorted(set(int(cand[np.clip(np.searchsorted(cd, t), 0, len(cand) - 1)]) for t in tg))
    return sorted(set(int(x) for x in np.linspace(cand[0], cand[-1], NPICK).round()))


def pick_displayed(valid, P):
    """~N_UPPER thumbs in [airway_start, trachea_end], ~N_REST over the rest."""
    vidx = np.where(valid)[0]
    if len(vidx) == 0:
        return []
    a0 = P.get("airway_start") or int(vidx[0])
    t_end = (P.get("trachea") or [a0, a0 + int(0.35 * len(vidx))])[1]
    upper = vidx[(vidx >= a0) & (vidx <= t_end)]
    rest = vidx[(vidx < a0) | (vidx > t_end)]
    def even(arr, k):
        if len(arr) == 0:
            return []
        k = min(k, len(arr))
        return [int(arr[i]) for i in np.linspace(0, len(arr) - 1, k).round().astype(int)]
    return sorted(set(even(upper, N_UPPER) + even(rest, N_REST)))


def render(fname, thumbs_by_fi, displayed, selected, P, tw, th, out_png):
    sel = set(selected)
    cols = 12
    rows = (len(displayed) + cols - 1) // cols
    lh = 30
    head = 60
    canvas = np.full((head + rows * (th + lh), cols * tw, 3), 245, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{fname}  dense contact ({len(displayed)} thumbs)  |  "
                f"magenta border = SELECTED for 3D recon ({len(sel)})",
                (10, 26), f, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    x = 10
    for r in ("glottis", "subglottis", "trachea", "other"):
        cv2.rectangle(canvas, (x, 40), (x + 22, 56), RC[r], -1)
        cv2.putText(canvas, r, (x + 26, 54), f, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x += 140
    cv2.rectangle(canvas, (x, 40), (x + 22, 56), SEL_BORDER, 3)
    cv2.putText(canvas, "selected", (x + 26, 54), f, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for idx, fi in enumerate(displayed):
        rr, cc = divmod(idx, cols)
        y0 = head + rr * (th + lh); x0 = cc * tw
        canvas[y0:y0 + th, x0:x0 + tw] = cv2.resize(thumbs_by_fi[fi], (tw, th), interpolation=cv2.INTER_AREA)
        canvas[y0 + th:y0 + th + lh, x0:x0 + tw] = RC[region_of(fi, P)]
        # clear, outlined frame number
        cv2.putText(canvas, f"{fi}", (x0 + 5, y0 + th + lh - 8), f, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"{fi}", (x0 + 5, y0 + th + lh - 8), f, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if fi in sel:
            cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + tw - 2, y0 + th - 2), SEL_BORDER, 4)
            cv2.putText(canvas, "SEL", (x0 + tw - 46, y0 + 22), f, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, "SEL", (x0 + tw - 46, y0 + 22), f, 0.6, SEL_BORDER, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + tw - 1, y0 + th + lh - 1), (70, 70, 70), 1)
    H = canvas.shape[0]
    if H > 18000:
        s = 18000 / H
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * s), 18000), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def process(fname):
    video = DATA / fname
    od = OUT / video.stem; od.mkdir(parents=True, exist_ok=True)
    pr = ANAT / video.stem / "proposed_ranges.json"
    P = json.loads(pr.read_text())["proposed"] if pr.exists() else {}
    mask, (W, H) = content_mask(video)
    sw = 240; sh = int(round(sw * H / W))
    cap = cv2.VideoCapture(str(video))
    thumbs = {}; L = []; diff = []; last = None; fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        thumbs[fi] = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
        L.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0][mask].mean() * 100 / 255) if mask.sum() else 0.0)
        gd = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        diff.append(0.0 if last is None else float(np.abs(gd.astype(np.int16) - last.astype(np.int16)).mean()))
        last = gd; fi += 1
    cap.release()
    L = np.array(L); diff = np.array(diff); valid = L >= DARK_L
    selected = select_recon(valid, diff)
    displayed = sorted(set(pick_displayed(valid, P)) | set(selected))
    tw = 200; th = int(round(tw * H / W))
    render(fname, thumbs, displayed, selected, P, tw, th, od / "contact_dense.png")
    (od / "sheet_info.json").write_text(json.dumps(
        {"displayed": displayed, "n_displayed": len(displayed),
         "selected": selected, "n_selected": len(selected)}))
    print(f"{fname:<12} displayed={len(displayed)} selected={len(selected)} -> {od/'contact_dense.png'}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", default=",".join(VIDEOS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for v in args.videos.split(","):
        if v not in VIDEOS:
            continue
        print(f"\n#### {v} ####", flush=True)
        try:
            process(v)
        except Exception as e:
            print(f"{v} ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
