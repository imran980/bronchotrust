"""Whole-video quality distribution + best window finder. Samples every Nth frame across the full
video, measures brightness (L*) and sharpness (Laplacian var), reports the distribution and the
single brightest+sharpest ~200-frame (effective, fps-normalized) window. Args: tag:video ..."""
import sys, cv2, numpy as np
N = 15  # sample stride for the scan
for tok in sys.argv[1:]:
    tag, v = tok.split("::")
    cap = cv2.VideoCapture(v); fps = cap.get(5); tot = int(cap.get(7)); fi = 0
    idx, bri, shp = [], [], []
    while True:
        ok, fr = cap.read()
        if not ok: break
        if fi % N == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0]
            idx.append(fi); bri.append(L.mean() * 100 / 255); shp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        fi += 1
    cap.release()
    idx, bri, shp = np.array(idx), np.array(bri), np.array(shp)
    valid = bri >= 12
    # best ~200-effective-frame window: convert 200 effective @30fps to source frames
    win = int(200 * round(fps / 30))
    # slide the window over sampled points, score = fraction-valid * median-sharpness within
    best, bscore, bwin = None, -1, win
    step = max(N, win // 20)
    for start in range(idx.min(), idx.max() - win, step):
        sel = (idx >= start) & (idx < start + win)
        if sel.sum() < 5: continue
        fv = valid[sel].mean(); ms = np.median(shp[sel]) if sel.any() else 0
        score = fv * ms
        if score > bscore: bscore, best = score, (start, start + win, fv, ms, np.median(bri[sel]))
    print(f"=== {tag}  ({tot} frames, {fps:.0f}fps) ===")
    print(f"  WHOLE VIDEO: brightness med {np.median(bri):.1f} (>=12 usable: {100*valid.mean():.0f}% of frames)")
    print(f"               sharpness  med {np.median(shp):.0f}  (p25 {np.percentile(shp,25):.0f} / p75 {np.percentile(shp,75):.0f})")
    if best:
        print(f"  BEST {win}-frame window: f{best[0]}-{best[1]}  valid {100*best[2]:.0f}%  sharp {best[3]:.0f}  bright {best[4]:.1f}")
