"""7-V1 FULL pipeline (primary candidate). Stages 1-4 here:
  1. SfM: dense sub_in + dense sub_out + bridge, pinned 7_v1 OPENCV, SP+LG, COLMAP.
  2. Verify intrinsics stayed pinned (delta vs pinned).
  3. Report registered/N, reproj, components, outlier ratio, lat/along, median/p95 tri.
  4. GATE: if geometry clean -> run MVS (geometric) on dense cloud (no Poisson).
Writes stage4_results.json for the Stage-5 cross-section step. Honest gate: if
geometry is not clean, STOP before MVS.

Run in depth-eval env. Output runs/phase3_7v1/.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap
import phase2_subcord as P2
import s4_common as C

DATA = P2.DATA
VIDEO = "7-V1.MP4"; SESSION = "7_v1"
WIN = {"sub_in": [130, 220], "sub_out": [1846, 1992]}
OD = Path("/home/mi3dr/projects/bronchotrust/runs/phase3_7v1")
K_IN, K_OUT, K_BRIDGE = 18, 18, 22
OUTLIER_K = 10.0


def main():
    OD.mkdir(parents=True, exist_ok=True)
    intr = P2.calibrate(SESSION)
    assert intr is not None, "no 7_v1 calibration"
    video = DATA / VIDEO
    print(f"7-V1 full: pinned OPENCV fx={intr['fx']:.3f} cx={intr['cx']:.1f} "
          f"k1={intr['k1']:.4f} calib_rms={intr['rms_reproj_px']:.4f}")
    mask, _ = P2.content_mask(video)
    L, diff, shas = P2.scan(video, mask, WIN["sub_out"][1] + 5)
    valid = L >= P2.DARK_L
    picks = sorted(set(
        P2.flow_pick(valid, diff, WIN["sub_in"][0], WIN["sub_in"][1], K_IN) +
        P2.flow_pick(valid, diff, WIN["sub_out"][0], WIN["sub_out"][1], K_OUT) +
        P2.flow_pick(valid, diff, WIN["sub_in"][1] + 1, WIN["sub_out"][0] - 1, K_BRIDGE)))
    img_dir, names, hm = P2.extract(video, picks, mask, shas, OD)
    n_in = sum(1 for p in picks if P2.seg_of(p, WIN) == "sub_in")
    n_out = sum(1 for p in picks if P2.seg_of(p, WIN) == "sub_out")
    n_br = sum(1 for p in picks if P2.seg_of(p, WIN) == "bridge")
    print(f"curated {len(names)} frames (sub_in={n_in}, bridge={n_br}, sub_out={n_out}); hash {hm}/{len(names)}")

    sfm = P2.run_sfm(img_dir, names, intr, OD)

    # ---- analyze ----
    rec = pycolmap.Reconstruction(str(sfm))
    cc, segc = {}, {}
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        cc[img.image_id] = -R.T @ t; segc[img.image_id] = P2.seg_of(fi, WIN)
    Call = np.array(list(cc.values()))
    ctr = np.median(Call, 0); dall = np.linalg.norm(Call - ctr, axis=1)
    outlier_ratio = float(dall.max() / max(np.median(dall), 1e-12))
    ana = P2.analyze(sfm, WIN)
    n_comp = len([d for d in (sfm / "models").iterdir() if d.is_dir()]) if (sfm / "models").exists() else 1
    has_in = any(s == "sub_in" for s in segc.values())
    has_out = any(s == "sub_out" for s in segc.values())

    # pin verification
    cam = next(iter(rec.cameras.values())); cp = list(cam.params)
    pin = [intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["k1"], intr["k2"], 0.0, 0.0]
    pin_delta = float(max(abs(a - b) for a, b in zip(cp, pin)))

    print("\n=== STAGE 3: 7-V1 geometry ===")
    print(f"  registered/N : {ana['n_reg']}/{len(names)}   reproj: {ana['reproj_px']:.2f}px")
    print(f"  components   : {n_comp}   (sub_in present={has_in}, sub_out present={has_out})")
    print(f"  outlier ratio: {outlier_ratio:.2f}  (clean if <= {OUTLIER_K})")
    print(f"  lat/along    : sub-cord {ana['subcord_lat_along']}")
    print(f"  tri (sub-cord): median {ana['subcord_median_tri_deg']}  p95 {ana['subcord_p95_tri_deg']}  (n={ana['subcord_n_points']})")
    print(f"  pin delta    : {pin_delta:.2e}  (pinned OK if ~0)  camera={cam.model.name if hasattr(cam.model,'name') else cam.model}")

    clean = (outlier_ratio <= OUTLIER_K) and (pin_delta < 1e-3) and has_in and has_out and n_comp <= 2
    report = {"video": VIDEO, "n_picked": len(names), "hash_match": f"{hm}/{len(names)}",
              "n_sub_in": n_in, "n_bridge": n_br, "n_sub_out": n_out,
              "n_reg": ana["n_reg"], "reproj_px": ana["reproj_px"], "n_components": n_comp,
              "has_sub_in": has_in, "has_sub_out": has_out, "outlier_ratio": round(outlier_ratio, 3),
              "subcord_lat_along": ana["subcord_lat_along"],
              "subcord_median_tri_deg": ana["subcord_median_tri_deg"],
              "subcord_p95_tri_deg": ana["subcord_p95_tri_deg"],
              "subcord_n_points": ana["subcord_n_points"],
              "pin_delta": pin_delta, "camera_params": cp, "geometry_clean": clean}

    if not clean:
        report["status"] = "STOP_unclean_geometry"
        (OD / "stage4_results.json").write_text(json.dumps(report, indent=2))
        print(f"\n=== GATE FAILED (geometry not clean) -> STOP before MVS ===")
        return

    print("\n=== STAGE 4: MVS (geometric, no Poisson) ===")
    n_dense, ply, logs = C.run_mvs(sfm, img_dir, OD / "dense")
    report.update({"status": "OK", "arm": "colmap_7v1", "sfm_model_dir": str(sfm),
                   "dense_ply": str(ply), "n_dense": int(n_dense), "mvs_logs": logs})
    (OD / "stage4_results.json").write_text(json.dumps(report, indent=2))
    print(f"  MVS dense points: {n_dense}")
    print(f"\nGATE PASSED. stage4_results.json written. Next: Stage 5 cross-section.")


if __name__ == "__main__":
    main()
