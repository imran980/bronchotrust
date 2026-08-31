"""Recon-free scope-motion detector via optical-flow LOOMING. For each consecutive frame pair, track
features (LK) and fit an isotropic image expansion s (flow ~ s*(p-c)): forward motion in a tube expands
the image (s>0), pullback contracts it (s<0). Integrate s -> a virtual axial-position signal, analogous
to the pose-based along-tube position but needing NO reconstruction.

Validation mode: on 7-V1, correlate the looming axial signal against the pose-based axial (from the
existing recon). High correlation => the recon-free detector is trustworthy on clips with no recon.

Usage: python flow_looming.py validate7v1
       python flow_looming.py signal <video> <lo> <hi>
"""
import sys, re, json, numpy as np, cv2
from pathlib import Path

FIRST15 = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos")


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok: cum += (f.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((7, 7), np.uint8), iterations=6) > 0


def expansion(p, dp):
    """Robust isotropic expansion s: dp ~ s*p + t (shared s, per-axis t). IRLS to reject outliers."""
    px, py = p[:, 0], p[:, 1]; dx, dy = dp[:, 0], dp[:, 1]
    A = np.zeros((2 * len(p), 3)); b = np.zeros(2 * len(p))
    A[0::2, 0] = px; A[0::2, 1] = 1; b[0::2] = dx
    A[1::2, 0] = py; A[1::2, 2] = 1; b[1::2] = dy
    w = np.ones(2 * len(p))
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        r = np.abs(A @ sol - b); sca = np.median(r) + 1e-6; w = 1.0 / (1.0 + (r / (3 * sca)) ** 2)
    return float(sol[0])                                   # expansion rate s


def looming_signal(video, lo, hi, stride=1):
    video = Path(video) if Path(video).is_absolute() else FIRST15 / video
    mask = bezel(video); ys, xs = np.where(mask); ctr = np.array([xs.mean(), ys.mean()])
    feat_mask = (mask.astype(np.uint8)) * 255
    cap = cv2.VideoCapture(str(video)); fi = 0; prev = None; frames = []; s_list = []
    while True:
        ok, fr = cap.read()
        if not ok or fi > hi: break
        if lo <= fi <= hi and (fi - lo) % stride == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                p0 = cv2.goodFeaturesToTrack(prev, 600, 0.01, 7, mask=feat_mask)
                s = 0.0
                if p0 is not None and len(p0) > 20:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, g, p0, None,
                                                           winSize=(21, 21), maxLevel=3)
                    st = st.ravel().astype(bool); a = p0[st].reshape(-1, 2); b = p1[st].reshape(-1, 2)
                    inb = mask[np.clip(b[:, 1].astype(int), 0, mask.shape[0] - 1),
                               np.clip(b[:, 0].astype(int), 0, mask.shape[1] - 1)]
                    a, b = a[inb], b[inb]
                    if len(a) > 20: s = expansion(a - ctr, b - a)
                s_list.append(s); frames.append(fi)
            prev = g
        fi += 1
    cap.release()
    s = np.array(s_list); frames = np.array(frames)
    axial = np.cumsum(s)                                    # virtual along-tube position
    return frames, s, axial


def pose_axial_7v1():
    import pycolmap
    rec = pycolmap.Reconstruction("runs/own_data/recon_7v1_verify/dense/sparse")
    rows = sorted(([int(re.search(r"(\d+)", im.name).group(1)), np.asarray(im.projection_center())]
                   for im in rec.images.values()), key=lambda r: r[0])
    fr = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows])
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    a = Cc @ Vt[0]
    if a[-1] < a[0]: a = -a
    return fr, a - a.min()


def smooth(a, k=9):
    if len(a) < k: return a
    return np.convolve(a, np.ones(k) / k, mode="same")


def monotonicity(axial, k=9):
    a = smooth(axial, k); da = np.diff(a)
    return abs(a[-1] - a[0]) / (np.sum(np.abs(da)) + 1e-9)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate7v1"
    if mode == "validate7v1":
        fr_f, s, axial = looming_signal("7-V1.MP4", 470, 880)
        fr_p, ap = pose_axial_7v1()
        # align by frame index (interp looming-axial onto pose frames)
        ax_on_pose = np.interp(fr_p, fr_f, axial)
        # orient/scale for comparison
        r = np.corrcoef(ax_on_pose, ap)[0, 1]
        print(f"7-V1 looming-vs-pose axial correlation r = {r:.3f}  (n_pose={len(fr_p)}, n_flow={len(fr_f)})")
        print(f"  monotonicity: pose={monotonicity(ap):.2f}  looming(recon-free)={monotonicity(axial):.2f}")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(fr_p, (ap - ap.min()) / (np.ptp(ap) + 1e-9), ".-", ms=3, label="pose axial (from recon)")
        ax.plot(fr_f, (axial - axial.min()) / (np.ptp(axial) + 1e-9), "-", lw=1.2, alpha=0.8, label="looming axial (recon-free)")
        ax.set_title(f"7-V1: recon-free looming vs pose-based axial  (r={r:.3f})"); ax.legend(); ax.grid(alpha=.3)
        ax.set_xlabel("frame"); ax.set_ylabel("normalized along-tube position")
        fig.tight_layout(); fig.savefig("runs/own_data/renders/looming_validate_7v1.png", dpi=120)
        print("  saved runs/own_data/renders/looming_validate_7v1.png")
    elif mode == "signal":
        v, lo, hi = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        fr, s, axial = looming_signal(v, lo, hi)
        print(f"{v} f{lo}-{hi}: {len(fr)} pairs | monotonicity={monotonicity(axial):.2f} | "
              f"net={axial[-1]-axial[0]:.3f} path={np.sum(np.abs(np.diff(axial))):.3f}")
        json.dump(dict(video=v, lo=lo, hi=hi, frames=fr.tolist(), s=s.tolist(), axial=axial.tolist(),
                       monotonicity=float(monotonicity(axial))),
                  open(f"runs/own_data/looming_{Path(v).stem}.json", "w"))
