"""Dense MVS on the GlueMap model (gluemap_aba), SAME COLMAP MVS settings as the
COLMAP arm. Uses the already-undistorted dense0/images (GlueMap's input) + GlueMap
poses/intrinsics (f=748.8, self-consistent with GlueMap). NO scale. depth-eval env."""
from __future__ import annotations
import subprocess
from pathlib import Path
OD = Path("/home/mi3dr/projects/bronchotrust/runs/fuse_32v2")
COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
IMG = OD / "dense0/images"            # undistorted pinhole frames (GlueMap input)
MODEL = OD / "gluemap/gluemap_aba"    # GlueMap rig-format model
DENSE = OD / "gluemap/dense0"


def run(cmd):
    print(">>", " ".join(cmd[:3]), "...", flush=True)
    r = subprocess.run([COLMAP] + cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("COLMAP ERR:\n", r.stderr[-1500:])
    return r


def ply_n(p):
    with open(p, "rb") as f:
        for _ in range(40):
            ln = f.readline().decode("latin1", "ignore")
            if ln.startswith("element vertex"): return int(ln.split()[-1])
    return 0


fused = DENSE / "fused.ply"
if not fused.exists():
    r = run(["image_undistorter", "--image_path", str(IMG), "--input_path", str(MODEL),
             "--output_path", str(DENSE), "--output_type", "COLMAP", "--max_image_size", "1600"])
    if not (DENSE / "sparse").exists():
        # fallback: convert rig-format model to legacy via pycolmap, then undistort
        import pycolmap
        leg = OD / "gluemap/aba_legacy"; leg.mkdir(parents=True, exist_ok=True)
        rec = pycolmap.Reconstruction(str(MODEL)); rec.write(str(leg))
        run(["image_undistorter", "--image_path", str(IMG), "--input_path", str(leg),
             "--output_path", str(DENSE), "--output_type", "COLMAP", "--max_image_size", "1600"])
    run(["patch_match_stereo", "--workspace_path", str(DENSE), "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.max_image_size", "1200"])
    run(["stereo_fusion", "--workspace_path", str(DENSE), "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", str(fused)])
print(f"GLUEMAP DENSE: {ply_n(fused) if fused.exists() else 'FAILED'} fused points -> {fused}")
