"""Oracle (human-supervised) throat annotation for the oracle-contour test. For each region I
read the gridded montages (runs/.../oracle/montage_<tag>.png), pick the throat/narrowest-lumen
CENTER per frame by eye (SEEDS below), then extract the dark-lumen boundary at that seed (Otsu
in a local window + seed-connected component) and VISUALLY VERIFY every overlay. This removes
detector-selection error: the human chooses the correct anatomical lumen; the boundary is snapped
to it. Writes contours_<tag>.json consumed by oracle_fit.py. depth-eval env."""
from __future__ import annotations
import cv2, numpy as np, json
from pathlib import Path

OUT = Path("/home/mi3dr/projects/bronchotrust/runs/barbour30/airwayfit/oracle"); OUT.mkdir(parents=True, exist_ok=True)
# human-picked throat centers (read off the gridded montages), per region/frame
SEED = {
    "2v2prox": ("runs/batch4/2-V2", {878: (945, 300), 885: (955, 312), 892: (957, 322), 899: (960, 335),
                906: (960, 342), 913: (942, 332), 920: (946, 330), 927: (950, 326), 934: (946, 315), 941: (940, 302)}),
    "25v1prox": ("runs/batch4/25-V1", {337: (865, 255), 342: (880, 250), 347: (895, 250), 352: (900, 262),
                 357: (905, 272), 362: (855, 280), 367: (870, 270), 372: (880, 260), 377: (888, 252), 382: (880, 250)}),
    "25v1dist": ("runs/batch4/25-V1", {407: (720, 215), 412: (760, 203), 417: (762, 200), 422: (790, 200),
                 427: (800, 202), 432: (730, 222), 437: (762, 210), 442: (782, 208), 447: (800, 202), 452: (820, 210)}),
}
ROOT = Path("/home/mi3dr/projects/bronchotrust")


def oracle_contour(gray, seed, win=210):
    cx, cy = seed; H, W = gray.shape
    x0, y0 = max(0, cx - win), max(0, cy - win); x1, y1 = min(W, cx + win), min(H, cy + win)
    sub = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.uint8), (0, 0), 2)
    thr, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = (sub < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    sx, sy = cx - x0, cy - y0; seedlab = lab[min(sy, dark.shape[0] - 1), min(sx, dark.shape[1] - 1)]
    if seedlab == 0:
        best, bd = 0, 1e9
        for i in range(1, n):
            d = np.hypot(cen[i, 0] - sx, cen[i, 1] - sy)
            if st[i, 4] > 400 and d < bd: bd, best = d, i
        seedlab = best
    if seedlab == 0: return None
    cnts, _ = cv2.findContours((lab == seedlab).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    return c[:, 0, :].astype(float) + [x0, y0] if len(c) >= 12 else None


def main():
    for tag, (model, seeds) in SEED.items():
        store, thumbs = {}, []
        for fr, seed in seeds.items():
            img = cv2.imread(str(ROOT / model / "images" / f"f{fr:05d}.png"))
            pts = oracle_contour(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), seed); ov = img.copy()
            if pts is not None:
                store[fr] = pts.tolist(); cv2.polylines(ov, [pts.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.circle(ov, seed, 5, (255, 0, 255), -1)
            th = cv2.resize(ov, (576, 324)); cv2.putText(th, f"f{fr}", (490, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            thumbs.append(th)
        rows = [np.hstack(thumbs[i:i + 5]) for i in range(0, len(thumbs), 5)]
        cv2.imwrite(str(OUT / f"verify_{tag}.png"), np.vstack(rows))
        (OUT / f"contours_{tag}.json").write_text(json.dumps(store))
        print(f"{tag}: {len(store)}/{len(seeds)} contours -> verify_{tag}.png (VERIFY before fitting)")


if __name__ == "__main__":
    main()
