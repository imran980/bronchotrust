"""Dense contact sheet of a bronchoscopy video for choosing the reconstruction window by eye.

Frames are decoded SEQUENTIALLY (never seeked, which drifts on H.264) and every thumbnail is labelled with its true
frame index, so the indices you read off the sheet are the ones to pass to recover_clip.py --lo/--hi. Two traces
underneath (field brightness and dark-lumen area) make dark, out-of-body and collapsed stretches obvious.

Manual windowing onto a steady, centred, monotonic pass is the single most effective lever in this pipeline: it
roughly doubled the cohort yield and rescued clips that failed on their whole traversal.

Usage:
  python pipeline/contact_sheet.py --video "<clip.MP4>" --out sheets/<tag>.png [--step 15]
"""
import argparse, os
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=0, help="thumbnail every STEP frames (default: ~120 thumbnails)")
    ap.add_argument("--thumb", type=int, default=160)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video); N = int(cap.get(7)); step = a.step or max(1, N // 120)
    tiles, idxs, stats, i = [], [], [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); field = f.max(axis=2) > 10
            if field.sum() > 1000:
                ys, xs = np.where(field); crop = f[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                med = np.median(g[field]); dark = ((g < 0.45 * med) & field).astype(np.uint8)
                nl, _, st, _ = cv2.connectedComponentsWithStats(dark)
                lum = st[1:, cv2.CC_STAT_AREA].max() / field.sum() if nl > 1 else 0.0
            else:
                crop, med, lum = f, 0, 0
            tiles.append(cv2.cvtColor(cv2.resize(crop, (a.thumb, a.thumb)), cv2.COLOR_BGR2RGB)); idxs.append(i); stats.append((med, lum * 100))
        i += 1
    cap.release(); S = np.array(stats)
    cols = 12; rws = (len(tiles) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 1.5, rws * 1.62 + 2.4))
    gs = fig.add_gridspec(rws + 2, cols, height_ratios=[1] * rws + [0.9, 0.9], hspace=0.35, wspace=0.04)
    for k, (im, ix) in enumerate(zip(tiles, idxs)):
        ax = fig.add_subplot(gs[k // cols, k % cols]); ax.imshow(im); ax.set_axis_off(); ax.set_title(f"{ix}", fontsize=7.5, color="0.25", pad=1)
    ax = fig.add_subplot(gs[rws, :]); ax.plot(idxs, S[:, 0], "-", color="0.3", lw=1.2); ax.set_ylabel("brightness", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[rws + 1, :]); ax.plot(idxs, S[:, 1], "-", color="#0072B2", lw=1.2); ax.set_ylabel("dark-lumen %", fontsize=8); ax.set_xlabel("frame", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    fig.suptitle(f"{os.path.basename(a.video)} — {N} frames, every {step}th", fontsize=13, fontweight="bold", y=0.997)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=88, bbox_inches="tight")
    print(f"{os.path.basename(a.video)}: {N} frames, {len(tiles)} thumbnails every {step} -> {a.out}")


if __name__ == "__main__":
    main()
