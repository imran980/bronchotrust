"""Real 2_V2 distal test with the CORRECTED invalid-ray handling (no -1e9 sentinel; invalid rays get a
moderate penalty; per-frame/per-ray validity tracked). Manual green contours (23 frames 1000-1044) +
fixed COLMAP poses. K=8 multistart, NO ground-truth init, NO tuning toward DCE 1.5446. Pixel chamfer is
an EVALUATION metric only (not in the objective). Reports tangency-on-valid, valid fractions, valid
frames, DCE spread, residual ranking, and renders every solution across all frames."""
from __future__ import annotations
import re, json
from pathlib import Path
import numpy as np, cv2
import pycolmap
from direct_metrology import segment as SG
from direct_metrology.phantom import Camera

ROOT = Path("/home/mi3dr/projects/bronchotrust")
ANN = ROOT / "runs/batch4/2-V2/images/2v2_distal_annotated"
MODEL = ROOT / "runs/batch4/2-V2/sparse/0"
OUT = Path("/tmp/claude-100461304/-home-mi3dr-projects-bronchotrust/32888f63-b0c2-4d41-9ad1-91a5dfc651e6/scratchpad/real2v2")
ACCEPTED_DCE = 1.5446                    # accepted distal-reference (report only; NEVER used in the fit)
K_STARTS = 8
MIN_VALID_FRAMES = 14                    # ~60% of 23 (coverage requirement)


def green_contour(p):
    im = cv2.imread(str(p)); b, g, r = cv2.split(im.astype(int))
    m = ((g > 120) & (r < 110) & (b < 110)).astype(np.uint8); m = cv2.dilate(m, np.ones((3, 3), np.uint8), 1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0, :].astype(float) if cs else None


def load():
    raw = {int(re.search(r"f(\d+)", p.name).group(1)): green_contour(p) for p in sorted(ANN.glob("f*.png"))}
    rec = pycolmap.Reconstruction(str(MODEL)); cam = list(rec.cameras.values())[0]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]); dist = np.array([k1, k2, p1, p2])
    pos = {}
    for im in rec.images.values():
        mm = re.search(r"f(\d+)", im.name)
        if not mm: continue
        M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]; C = -R.T @ M[:3, 3]
        pos[int(mm.group(1))] = (R, C)
    have = sorted(f for f in raw if f in pos and raw[f] is not None)
    cons_full = {f: raw[f] for f in have}
    cons = []; cams = []
    for f in have:
        und = cv2.undistortPoints(raw[f].reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(-1, 2)
        step = max(1, len(und) // 80); cons.append(und[::step]); R, C = pos[f]; cams.append(Camera(R=R, C=C))
    return have, cons, cons_full, cams, K, dist, pos


def triangulate_axis(cons, cams, K):
    Kinv = np.linalg.inv(K); A = np.zeros((3, 3)); bb = np.zeros(3)
    for c, cm in zip(cons, cams):
        d = Kinv @ np.array([*c.mean(0), 1.]); d /= np.linalg.norm(d); d = cm.R.T @ d
        P = np.eye(3) - np.outer(d, d); A += P; bb += P @ cm.C
    O0 = np.linalg.solve(A, bb)
    Cs = np.array([cm.C for cm in cams]); axis = np.linalg.svd(Cs - Cs.mean(0))[2][0]; axis *= np.sign(axis[2])
    return O0, axis


def undist_to_dist(uv, K, dist):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; k1, k2 = dist[0], dist[1]
    xn = (uv[:, 0] - cx) / fx; yn = (uv[:, 1] - cy) / fy; r2 = xn ** 2 + yn ** 2; f = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[xn * f * fx + cx, yn * f * fy + cy]


def pixel_chamfer(p, cams, cons_full, have, K, dist):
    """EVAL ONLY: median per-frame chamfer (px) between projected silhouette and the manual contour."""
    from scipy.spatial import cKDTree
    per = {}
    for f, cm in zip(have, cams):
        pc = SG.predicted_contour(p, cm, K, ndir=140, rmax_px=700)
        if pc is None or len(pc) < 8:
            per[f] = None; continue
        pcd = undist_to_dist(pc, K, dist)
        per[f] = float(np.median(cKDTree(cons_full[f]).query(pcd)[0]))
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have, cons, cons_full, cams, K, dist, pos = load()
    O0, axis = triangulate_axis(cons, cams, K)
    rpx = np.median([np.median(np.linalg.norm(c - c.mean(0), axis=1)) for c in cons]); fx = K[0, 0]
    r0 = rpx / fx * 0.83                              # naive scene-unit ring radius
    PEN = 3.0 * r0                                    # invalid-ray penalty (moderate; data-driven, not tuned)
    half_len = max(0.5, 6 * r0)
    print("real 2_V2 distal | %d frames | naive r0=%.3f  PEN=%.3f  half_len=%.3f  axis=%s"
          % (len(have), r0, PEN, half_len, np.round(axis, 3)))

    rng = np.random.default_rng(0); sols = []
    for i in range(K_STARTS):
        th = SG.theta_from(O0 + rng.normal(0, 0.15, 3), axis, r_t=max(0.05, r0 * rng.uniform(0.6, 1.6)),
                           s_t=rng.normal(0, 0.2), kappa=[0.1, 0.2, 0.35][i % 3], kappa4=0.0,
                           e=0.0, phi=rng.uniform(0, 6.28), lobe=0.0)
        fs = SG.fit(cons, cams, K, th, half_len=half_len, max_nfev=300, pen=PEN)
        per, nvf, vrms = SG.frame_diagnostics(SG.theta_from(fs.p.O, fs.p.a, fs.p.r_t, fs.p.s_t,
                        np.log(np.expm1(max(fs.p.kappa, 1e-3))), fs.p.kappa4, fs.p.e, fs.p.phi, fs.p.lobe),
                        cons, cams, K, half_len=half_len)
        vf = np.mean([x["valid_frac"] for x in per])
        pch = pixel_chamfer(fs.p, cams, cons_full, have, K, dist)
        pchv = [pch[f] for f in have if pch[f] is not None]
        sols.append(dict(i=i, dce=fs.dce, csa=fs.csa, resid=fs.resid_rms_mm, r_t=fs.p.r_t, e=fs.p.e,
                         lobe=fs.p.lobe, n_valid_frames=nvf, mean_valid_frac=float(vf), valid_tang_rms=vrms,
                         med_pixel_chamfer=float(np.median(pchv)) if pchv else float("nan"),
                         per_frame_valid=[round(x["valid_frac"], 2) for x in per], fs=fs))
    sols.sort(key=lambda s: s["resid"])

    print("\n K=8 multistart (sorted by total residual; invalid rays penalized, not zeroed):")
    print(" rank  DCE    CSA    total_resid  validFrames  meanValidFrac  validTangRMS  medPixChamfer  r_t    e")
    for r, s in enumerate(sols):
        print("  %d   %.3f  %.4f   %8.4f      %2d/%2d        %.2f          %.4f        %6.1f px   %.3f %+.3f" % (
            r, s["dce"], s["csa"], s["resid"], s["n_valid_frames"], len(have), s["mean_valid_frac"],
            s["valid_tang_rms"], s["med_pixel_chamfer"], s["r_t"], s["e"]))
    dces = np.array([s["dce"] for s in sols])
    qualdces = np.array([s["dce"] for s in sols if s["n_valid_frames"] >= MIN_VALID_FRAMES])
    print("\n DCE spread (all 8): min=%.3f max=%.3f ratio=%.2f std/mean=%.2f" % (
        dces.min(), dces.max(), dces.max() / max(dces.min(), 1e-6), dces.std() / dces.mean()))
    print(" solutions meeting coverage (>=%d/%d valid frames): %d/%d" % (
        MIN_VALID_FRAMES, len(have), len(qualdces), len(sols)))
    if len(qualdces):
        print(" DCE spread (coverage-qualified): min=%.3f max=%.3f ratio=%.2f" % (
            qualdces.min(), qualdces.max(), qualdces.max() / max(qualdces.min(), 1e-6)))
    best = sols[0]
    print(" lowest-residual pick: DCE=%.3f  validFrames=%d/%d  medPixChamfer=%.1fpx" % (
        best["dce"], best["n_valid_frames"], len(have), best["med_pixel_chamfer"]))
    print(" residual ranking monotonic with pixel-chamfer? corr(resid,chamfer)=%.2f" % (
        np.corrcoef([s["resid"] for s in sols], [s["med_pixel_chamfer"] for s in sols])[0, 1]))
    print(" accepted distal-reference DCE = %.4f (report only; not used in fit)" % ACCEPTED_DCE)

    # render every solution across a representative set of frames
    show = [f for f in have[::3]][:8]
    tiles_rows = []
    for s in sols:
        row = []
        for f in show:
            cm = Camera(R=pos[f][0], C=pos[f][1]); im = cv2.imread(str(ANN / f"f{f:05d}.png"))
            cv2.polylines(im, [cons_full[f].astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 6)
            pc = SG.predicted_contour(s["fs"].p, cm, K, ndir=140, rmax_px=700)
            valid = pc is not None and len(pc) > 8
            if valid:
                cv2.polylines(im, [undist_to_dist(pc, K, dist).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 5)
            cv2.rectangle(im, (0, 0), (620, 150), (0, 0, 0), -1)
            cv2.putText(im, "f%d %s" % (f, "" if valid else "INVALID"), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 255, 255) if valid else (0, 0, 255), 4)
            cv2.putText(im, "sol%d DCE%.2f" % (s["i"], s["dce"]), (8, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            row.append(cv2.resize(im, (360, 202)))
        tiles_rows.append(np.hstack(row))
    grid = np.vstack(tiles_rows)
    cv2.imwrite(str(OUT / "every_solution_all_frames.png"), grid)
    (OUT / "results.json").write_text(json.dumps(
        dict(pen=PEN, half_len=half_len, min_valid_frames=MIN_VALID_FRAMES, accepted_dce=ACCEPTED_DCE,
             solutions=[{k: v for k, v in s.items() if k != "fs"} for s in sols]), indent=2,
        default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print("\n saved", OUT / "every_solution_all_frames.png", grid.shape)


if __name__ == "__main__":
    main()
