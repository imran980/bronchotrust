"""Whole-video numbered contact sheets for pigs 2-5, for manual window selection."""
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/home/mi3dr/dataset/Porcine Airway Endoscopies"
vids = [("pig2a", "Pig Trachea 2 (Video 1).mp4"), ("pig2b", "Pig Trachea 2 (Video 2).mp4"),
        ("pig3", "Pig Trachea 3.mp4"), ("pig4", "Pig Trachea 4.mp4"), ("pig5", "Pig Trachea 5.mp4")]
for tag, fn in vids:
    cap = cv2.VideoCapture(f"{D}/{fn}"); N = int(cap.get(7)); fps = cap.get(5)
    step = 130; idxs = list(range(0, N, step)); cols = 12; rows = int(np.ceil(len(idxs) / cols))
    want = set(idxs); got = {}; fi = 0
    while fi < max(idxs) + 1:
        ok, fr = cap.read()
        if not ok: break
        if fi in want: got[fi] = cv2.convertScaleAbs(fr, alpha=1.7, beta=12)
        fi += 1
    cap.release()
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.25)); ax = np.array(ax).ravel()
    for j, fi in enumerate(idxs):
        if fi in got:
            ax[j].imshow(cv2.cvtColor(got[fi], cv2.COLOR_BGR2RGB))
            ax[j].axhline(got[fi].shape[0] / 2, color='cyan', lw=0.3, alpha=0.5)
            ax[j].axvline(got[fi].shape[1] / 2, color='cyan', lw=0.3, alpha=0.5)
        ax[j].set_title(f"{fi}", fontsize=6, pad=1); ax[j].axis("off")
    for j in range(len(idxs), len(ax)): ax[j].axis("off")
    fig.suptitle(f"{tag} ({fn}) — whole video every {step}f (~{step/fps:.1f}s), {N} frames. cyan=center", fontsize=11)
    fig.tight_layout(); fig.savefig(f"runs/own_data/porcine/thumbs_{tag}.png", dpi=80); plt.close(fig)
    print(f"saved thumbs_{tag}.png ({len(idxs)} thumbs, {N} frames)", flush=True)
print("ALL THUMBS DONE")
