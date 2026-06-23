"""Lightweight CORRECTED sparse-only GlueMap diagnostic (no MVS).
Tests ONE variable vs the rejected run: temporal/sequential pairing
(--is_sequential) instead of retrieval-only top-100. Everything else identical
(same undistorted full-window frames + PINHOLE GT + --use_gt_intrinsics + SHARED
SIMPLE_PINHOLE). Reports the sparse gates the user requires BEFORE any MVS:
  focal drift, camera-outlier ratio, reproj px, registered, trajectory runs.
Reuses the existing undist_images + gt_pinhole from gluemap_audit.py.
depth-eval env for metrics; gluemap env for the front-end subprocess."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import gluemap_audit as A


def run_seq(key, neighbors_seq=30, num_neighbors=None, tag="gm_seq"):
    cfg = A.VIDEOS[key]; aud = A.AUDIT / key
    und = aud / "undist_images"; gt = aud / "gt_pinhole"
    assert und.exists() and gt.exists(), "run gluemap_audit.py first (need undist+GT)"
    write = aud / tag; aba = write / "gluemap_aba"
    if not ((aba / "images.bin").exists() or (aba / "images.txt").exists()):
        shutil.rmtree(write, ignore_errors=True); write.mkdir(parents=True)
        env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
        cmd = [A.GLUEMAP, "--config", str(A.GM_CONFIG), "--images_path", str(und),
               "--write_path", str(write), "--intrinsics_mode", "SHARED",
               "--camera_model", "SIMPLE_PINHOLE",
               "--gt_intrinsics_path", str(gt), "--use_gt_intrinsics",
               "--is_sequential", "--num_neighbors_sequential", str(neighbors_seq)]
        if num_neighbors is not None:
            cmd += ["--num_neighbors", str(num_neighbors)]
        print("CMD:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(A.GM_SRC), capture_output=True, text=True, env=env)
        (write / "run.log").write_text((r.stdout or "") + "\n==STDERR==\n" + (r.stderr or ""))
        if not ((aba / "images.bin").exists() or (aba / "images.txt").exists()):
            raise RuntimeError(f"GlueMap(seq) produced no model; see {write/'run.log'}")
    # gates
    gm = A.sfm_metrics(aba)
    given = A.VIDEOS  # noqa
    j = json.loads((A.CALIB / cfg["session"] / "intrinsics_pinned.json").read_text())
    pinned = j["params_colmap"][0]
    drift = round(gm["focal_fx"] / pinned - 1.0, 4)
    col = A.sfm_metrics(A.BATCH4 / cfg["src"] / "sparse/0")
    gates = {
        "focal_drift": drift, "focal_drift_pass(<1%)": abs(drift) < 0.01,
        "outlier_ratio": gm["outlier_ratio"], "outlier_pass(<2%)": gm["outlier_ratio"] < 0.02,
        "reproj_px": gm["reproj_px"], "colmap_reproj_px": col["reproj_px"],
        "reproj_pass(~colmap)": gm["reproj_px"] <= col["reproj_px"] * 1.3,
        "registered": gm["reg"], "frame_runs": gm["frame_runs"],
        "trajectory_connected(1 run)": len(gm["frame_runs"]) == 1,
    }
    out = {"video": key, "mode": "sequential", "num_neighbors_sequential": neighbors_seq,
           "num_neighbors": num_neighbors, "tag": tag,
           "pinned_focal": round(pinned, 3), "gluemap_focal": gm["focal_fx"], "gates": gates}
    (write / "seq_gates.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    passed = gates["focal_drift_pass(<1%)"] and gates["outlier_pass(<2%)"] and gates["reproj_pass(~colmap)"]
    print(f"\n[{key}] SPARSE GATES {'PASS -> MVS justified' if passed else 'FAIL -> stay rejected (or freeze focal next)'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="2_V2", choices=list(A.VIDEOS))
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--num_neighbors", type=int, default=None)
    ap.add_argument("--tag", default="gm_seq")
    a = ap.parse_args(); run_seq(a.only, a.neighbors, a.num_neighbors, a.tag)


if __name__ == "__main__":
    main()
