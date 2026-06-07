"""STAGE 3c: bridge-attempt reconstruction.

Add 10 mid-trachea bridge frames in f700-f2200 to the existing 30-frame set
(total 40 frames). Re-extract sequentially. Re-run the full pinned Stage-3
pipeline on the 40-frame set. Report whether airway and cylinder now land
in ONE connected reconstruction.

Bridges (segment="trachea"): 850, 950, 1050, 1150, 1350, 1500, 1900, 2100,
2500, 2700. Chosen to break the worst f700->f1212 gap and densify the
mid-trachea descent.

Output: runs/gated2/2_v2/recon_bridged/
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, "/home/mi3dr/projects/bronchotrust")
from stage3_reconstruct import (
    cam_params_str, run_hloc, find_glottic_init, reconstruct,
    run_mvs, run_poisson, analyze, render_3view,
)

PROJECT = Path("/home/mi3dr/projects/bronchotrust")
GATED = PROJECT / "runs/gated2/2_v2"
RECON = GATED / "recon_bridged"
VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
VID = "2_v2"

BRIDGE_FRAMES = [850, 950, 1050, 1150, 1350, 1500, 1900, 2100, 2500, 2700]


def extract_bridges():
    frames_dir = GATED / "stage1_frames"
    cap = cv2.VideoCapture(str(VIDEO))
    targets = set(BRIDGE_FRAMES)
    saved = []
    fi = 0
    while targets:
        ok, fr = cap.read()
        if not ok: break
        if fi in targets:
            p = frames_dir / f"seg_trachea_f{fi:05d}.png"
            cv2.imwrite(str(p), fr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            saved.append(fi)
            targets.discard(fi)
            print(f"  bridge: saved {p.name}")
        fi += 1
    cap.release()
    print(f"saved {len(saved)} bridge frames")
    return saved


def build_selection():
    sel0 = json.loads((GATED / "stage1_selection.json").read_text())["selection"]
    bridges = [{"segment": "trachea", "frame_idx": fi}
                for fi in BRIDGE_FRAMES]
    new = sel0 + bridges
    seg_order = ["blade", "glottis", "subglottic", "trachea", "transition_tube"]
    new.sort(key=lambda s: (seg_order.index(s["segment"]), s["frame_idx"]))
    return new


def preprocess_bridged(out_base: Path, selection, mask):
    img_dir = out_base / "images"
    msk_dir = out_base / "masks"
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(msk_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); msk_dir.mkdir(parents=True)
    bezel = (mask > 0)
    extracted = []
    for s in selection:
        fi = s["frame_idx"]; seg = s["segment"]
        src = GATED / "stage1_frames" / f"seg_{seg}_f{fi:05d}.png"
        if not src.exists():
            print(f"  MISSING source {src.name}"); continue
        img = cv2.imread(str(src))
        img[~bezel] = 0
        name = f"{VID}_f{fi:06d}.png"
        cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(msk_dir / f"{name}.png"),
                    (bezel.astype(np.uint8) * 255),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        extracted.append({"frame_idx": fi, "segment": seg, "filename": name})
    print(f"  preprocessed {len(extracted)} / {len(selection)} frames")
    return extracted


def main():
    print("=== STAGE 3c: bridged-trachea reconstruction ===")
    extract_bridges()
    sel = build_selection()
    print(f"selection: {len(sel)} frames "
          f"(original 30 + {len(BRIDGE_FRAMES)} bridges)")
    intr = json.loads((GATED / "intrinsics_pinned.json").read_text())
    mask = cv2.imread(str(GATED / "content_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert intr["rms_reproj_px"] < 0.5

    RECON.mkdir(parents=True, exist_ok=True)
    for f in RECON.glob("*"):
        if f.is_file(): f.unlink()
        else: shutil.rmtree(f)
    extracted = preprocess_bridged(RECON, sel, mask)
    image_list = sorted([e["filename"] for e in extracted])
    glottic_files = {e["filename"] for e in extracted if e["segment"] == "glottis"}
    print(f"  glottic init pool: {sorted(glottic_files)}")

    feats, matches, sfm_pairs = run_hloc(RECON, image_list)
    n_reg, n_pts = reconstruct(RECON, image_list, intr, feats, matches,
                                  sfm_pairs, glottic_files)
    n_dense = run_mvs(RECON)
    n_mesh_v = run_poisson(RECON)
    print(f"  MVS dense: {n_dense}   Poisson verts (VIEW ONLY): {n_mesh_v}")

    analysis = analyze(RECON, sel, intr)

    text_dir = RECON / "sfm_text"
    text_dir.mkdir(exist_ok=True)
    import subprocess
    COLMAP = Path("/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap")
    subprocess.run([str(COLMAP), "model_converter",
                     "--input_path", str(RECON / "sfm"),
                     "--output_path", str(text_dir),
                     "--output_type", "TXT"], capture_output=True)

    dense_ply = RECON / "dense" / "fused.ply"
    if dense_ply.exists():
        try: render_3view(RECON, analysis, dense_ply)
        except Exception as e: print(f"render err: {e}")

    reg_idx = {r["frame_idx"] for r in analysis["best"]["registered"]}
    all_idx = {s["frame_idx"] for s in sel}
    failed = sorted(all_idx - reg_idx)
    transition_target = sorted(s["frame_idx"] for s in sel
                                 if s["segment"] == "transition_tube")
    transition_reg = sorted(fi for fi in transition_target if fi in reg_idx)
    bridge_reg = sorted(fi for fi in BRIDGE_FRAMES if fi in reg_idx)
    bridge_failed = sorted(fi for fi in BRIDGE_FRAMES if fi not in reg_idx)
    airway_seg = sorted(s["frame_idx"] for s in sel
                          if s["segment"] in {"blade", "glottis", "subglottic"})
    airway_reg = sorted(fi for fi in airway_seg if fi in reg_idx)
    airway_failed = sorted(fi for fi in airway_seg if fi not in reg_idx)
    cyl_seg = sorted(s["frame_idx"] for s in sel
                       if s["segment"] == "transition_tube")
    cyl_reg = sorted(fi for fi in cyl_seg if fi in reg_idx)
    connected = (len(airway_reg) > 0 and len(cyl_reg) > 0)

    n_submodels = len(analysis.get("all_submodels", []))
    cam_params = analysis["best"]["camera_params"]
    pin_check = {
        "delta_fx_px": (cam_params[0] - intr["fx"]) if len(cam_params) >= 1 else None,
        "delta_fy_px": (cam_params[1] - intr["fy"]) if len(cam_params) >= 2 else None,
        "delta_k1": (cam_params[4] - intr["k1"]) if len(cam_params) >= 5 else None,
        "delta_k2": (cam_params[5] - intr["k2"]) if len(cam_params) >= 6 else None,
    }

    diag = {
        "mode": "bridged (40 frames, pinned NEW OPENCV)",
        "n_selected": len(sel),
        "n_registered": int(analysis["best"]["n_reg"]),
        "n_failed_to_register": len(failed),
        "failed_frames": failed,
        "n_sparse_points": int(analysis["best"]["n_sparse"]),
        "n_dense_points": int(n_dense),
        "n_poisson_verts_VIEW_ONLY": int(n_mesh_v),
        "mean_reprojection_error_px":
            analysis["best"]["mean_reprojection_error_px"],
        "bridges": {
            "target": BRIDGE_FRAMES,
            "registered": bridge_reg,
            "failed": bridge_failed,
        },
        "airway_summary": {
            "target": airway_seg,
            "registered": airway_reg,
            "failed": airway_failed,
        },
        "cylinder_summary": {
            "target": cyl_seg,
            "registered": cyl_reg,
        },
        "connected_airway_and_cylinder_in_best": bool(connected),
        "n_submodels": n_submodels,
        "pin_verification": pin_check,
    }
    (RECON / "recon_results.json").write_text(json.dumps(diag, indent=2))
    print(f"saved {RECON/'recon_results.json'}")
    print(f"\nairway_in_best: {len(airway_reg)}/{len(airway_seg)}")
    print(f"cylinder_in_best: {len(cyl_reg)}/{len(cyl_seg)}")
    print(f"CONNECTED (best has both): {connected}")


if __name__ == "__main__":
    main()
