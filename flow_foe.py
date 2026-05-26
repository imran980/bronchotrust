"""
Optical-flow focus-of-expansion (FoE) sign classifier for consecutive
bronchoscopy frames.

Why this exists: DUSt3R's per-pair AsymmetricCroCo3DStereo is sign-
ambiguous on textureless bronchoscopy frames. It can fit "moved forward
0.3 mm" or "moved backward 0.3 mm" with equally small pointmap-
consistency loss, and the global aligner picks arbitrarily — producing
the 15-27x trajectory zigzag observed in Phase 1.

Optical flow works on smooth mucosa because it needs only local
intensity gradients (RAFT correlation volumes), not the feature
descriptors that fail elsewhere. The focus of expansion / contraction
disambiguates direction:

    Forward motion (camera advances into the scene):
        flow vectors radiate OUTWARD from FoE
        sign(flow . (pixel - FoE)) > 0   for most pixels
    Backward motion (camera retracts):
        flow vectors converge TOWARD FoE
        sign(flow . (pixel - FoE)) < 0

We use this only to determine the SIGN of each DUSt3R pair's relative
translation — DUSt3R still estimates magnitude and rotation, which it
does well per-pair. Confidence below threshold -> "uncertain", no sign
constraint is applied for that pair.

Public API:
    load_raft(device)                                       -> model
    compute_flow(model, img1, img2)                         -> (H, W, 2)
    estimate_foe(flow, min_mag=0.5)                         -> (foe_xy, n_used)
    classify_sign(flow, foe, min_mag=0.5)                   -> (sign, confidence)

Single-shot helper:
    pair_sign(model, img1, img2, ...)                       -> (sign, conf, foe)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------- RAFT ----------

def load_raft(device: str = "cuda") -> torch.nn.Module:
    """Load torchvision's RAFT-Large. Same architecture as the paper."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    return model


def _prep(img_u8: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (1, 3, H, W) float in [-1, 1] (RAFT convention)."""
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=-1)
    t = torch.from_numpy(img_u8).to(device).permute(2, 0, 1).float().unsqueeze(0)
    t = t / 127.5 - 1.0
    # RAFT requires H, W divisible by 8.
    _, _, H, W = t.shape
    Hp = (H // 8) * 8
    Wp = (W // 8) * 8
    if Hp != H or Wp != W:
        t = t[:, :, :Hp, :Wp]
    return t


@torch.no_grad()
def compute_flow(model: torch.nn.Module, img1_u8: np.ndarray,
                 img2_u8: np.ndarray) -> np.ndarray:
    """Returns (H, W, 2) float32 flow in pixels (dx, dy)."""
    device = next(model.parameters()).device
    a = _prep(img1_u8, device)
    b = _prep(img2_u8, device)
    flow_list = model(a, b)
    flow = flow_list[-1][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    return flow


# ---------- FoE ----------

def estimate_foe(flow: np.ndarray, min_mag: float = 0.5
                 ) -> Tuple[np.ndarray, int]:
    """LSQ focus of expansion / contraction.

    Each flow vector defines a line through pixel (x, y) in direction
    (u, v); the FoE is the point closest to all such lines. The
    perpendicular distance from FoE=(fx, fy) to the line through
    (x, y) with direction (u, v) is

        |(fx - x) v  -  (fy - y) u|   (since (v, -u) is the unit normal,
                                       up to scaling by sqrt(u^2 + v^2))

    Minimizing the squared distances is a 2x2 linear system.

    Returns (foe_xy = (fx, fy), n_used). n_used = 0 if not enough pixels.
    """
    H, W, _ = flow.shape
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    mask = mag > min_mag
    if mask.sum() < 50:
        return np.array([W / 2.0, H / 2.0], dtype=np.float32), 0
    ys, xs = np.where(mask)
    u_m = u[mask]
    v_m = v[mask]
    # A * foe = b, where each row is (v_i, -u_i) and b_i = x_i v_i - y_i u_i.
    A = np.stack([v_m, -u_m], axis=1).astype(np.float64)
    b = (xs * v_m - ys * u_m).astype(np.float64)
    # Normal equations
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        foe = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        return np.array([W / 2.0, H / 2.0], dtype=np.float32), 0
    return foe.astype(np.float32), int(mask.sum())


def classify_sign(flow: np.ndarray, foe: np.ndarray, min_mag: float = 0.5,
                  conf_threshold: float = 0.5
                  ) -> Tuple[int, float]:
    """Return (sign, confidence) for the pair's relative motion direction.

    sign = +1   forward  (flow radiates OUTWARD from FoE)
    sign = -1   backward (flow converges TOWARD FoE)
    sign =  0   uncertain (lateral motion, rotation, low flow)

    confidence in [0, 1] = fraction of high-magnitude pixels whose flow
    points consistently away from (or toward) FoE.
    """
    H, W, _ = flow.shape
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    mask = mag > min_mag
    if mask.sum() < 50:
        return 0, 0.0

    ys, xs = np.where(mask)
    dx = xs.astype(np.float32) - foe[0]
    dy = ys.astype(np.float32) - foe[1]
    # Dot product of flow with the FoE->pixel direction.
    dot = u[mask] * dx + v[mask] * dy
    # Sign agreement: +1 if flow points outward, -1 if inward.
    s = np.sign(dot)
    # Confidence: mean of sign-agreements weighted by flow magnitude.
    w = mag[mask]
    weighted_sign = float((s * w).sum() / w.sum())  # in [-1, 1]
    if weighted_sign > conf_threshold:
        return +1, float(weighted_sign)
    if weighted_sign < -conf_threshold:
        return -1, float(-weighted_sign)
    return 0, float(abs(weighted_sign))


def pair_sign(model: torch.nn.Module, img1_u8: np.ndarray,
              img2_u8: np.ndarray, min_mag: float = 0.5,
              conf_threshold: float = 0.5):
    """Convenience: flow + FoE + classification in one call.

    Returns dict with keys sign, confidence, foe, n_pixels, flow_mag_median.
    """
    flow = compute_flow(model, img1_u8, img2_u8)
    foe, n = estimate_foe(flow, min_mag=min_mag)
    sign, conf = classify_sign(flow, foe, min_mag=min_mag,
                               conf_threshold=conf_threshold)
    mag = np.sqrt((flow ** 2).sum(-1))
    return dict(sign=sign, confidence=conf, foe=foe,
                n_pixels=n, flow_mag_median=float(np.median(mag)))


# ---------- CLI smoke test ----------

if __name__ == "__main__":
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True,
                    help="Directory of PNG/JPG frames in numeric order.")
    ap.add_argument("--step", type=int, default=1, help="Pair stride.")
    ap.add_argument("--max_pairs", type=int, default=20)
    args = ap.parse_args()

    from pathlib import Path
    frames = sorted(Path(args.frames_dir).glob("*.png"))
    if not frames:
        raise SystemExit(f"No PNGs in {args.frames_dir}")
    model = load_raft("cuda" if torch.cuda.is_available() else "cpu")
    n = min(args.max_pairs, len(frames) - args.step)
    print(f"{'pair':>6} {'sign':>5} {'conf':>6} {'foe_x':>7} {'foe_y':>7} "
          f"{'n_px':>7} {'mag_med':>8}")
    for k in range(n):
        i = k * args.step
        j = i + args.step
        if j >= len(frames):
            break
        img1 = np.array(Image.open(frames[i]).convert("RGB"))
        img2 = np.array(Image.open(frames[j]).convert("RGB"))
        r = pair_sign(model, img1, img2)
        print(f"{i:>3}->{j:<3} {r['sign']:>5d} {r['confidence']:>6.3f} "
              f"{r['foe'][0]:>7.1f} {r['foe'][1]:>7.1f} "
              f"{r['n_pixels']:>7d} {r['flow_mag_median']:>8.3f}")
