"""Take a user-provided bronchoscopy video and prep it for the pipeline.

What it does:
  1. ffmpeg-extracts frames from the video at --fps into <out_dir>/frames/.
  2. Writes a JSONL manifest with the same shape build_uaal_manifest.py
     produces, so depth_eval_uaal.py / build_endo2dtam_scene.py /
     render_dashboard_v1.py all work without modification. Subset name
     defaults to 'video' (configurable).
  3. Optionally seeds every frame with a stub annotation of
     --deepest_landmark (e.g. "Right main bronchus") — that's how the
     dashboard knows how far down the centerline path to cap the dot.
     For a video that travels nose -> trachea -> carina -> right lobe,
     "Right main bronchus" is the right choice. If your video stops at
     trachea, pass "Trachea". The exact label has to be one the
     dashboard's DEPTH_FRAC table knows about (see render_dashboard_v1.py).
  4. Writes a custom uaal.yaml under endo2dtam_uaal/ with intrinsics
     matched to your video's resolution (assumes 90 deg pinhole FOV —
     overwrite if you have a calibration).
  5. Prints the next-step commands ready to copy/paste.

Usage:
  python process_video_input.py \\
      --video /path/to/my.mp4 \\
      --out_dir runs/user_video \\
      --session my_synthetic \\
      --deepest_landmark "Right main bronchus" \\
      --fps 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

from PIL import Image


VALID_DEEPEST_LANDMARKS = [
    "Vocal cords", "Glottis", "Trachea", "Main carina",
    "Right main bronchus", "Left main bronchus",
    "Intermediate bronchus",
    "Right superior lobar bronchus", "Right middle lobar bronchus",
    "Right inferior lobar bronchus",
    "Left superior lobar bronchus", "Left inferior lobar bronchus",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--session", default="user_video")
    ap.add_argument("--subset", default="video")
    ap.add_argument("--deepest_landmark", default=None,
                    help=f"One of: {', '.join(VALID_DEEPEST_LANDMARKS)}")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Frames per second to extract (use the same fps "
                         "for ffmpeg-stitch at the end).")
    ap.add_argument("--write_yaml", action="store_true",
                    help="Also write a uaal.yaml with intrinsics matched "
                         "to the extracted frames' resolution.")
    args = ap.parse_args()

    if args.deepest_landmark and args.deepest_landmark not in VALID_DEEPEST_LANDMARKS:
        raise SystemExit(
            f"Unknown deepest_landmark '{args.deepest_landmark}'. "
            f"Pick one of: {VALID_DEEPEST_LANDMARKS}")

    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", args.video,
        "-vf", f"fps={args.fps}",
        os.path.join(frames_dir, "%06d.png"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = sorted(f for f in os.listdir(frames_dir)
                    if f.lower().endswith(".png"))
    if not frames:
        raise SystemExit("No frames were extracted.")
    first = Image.open(os.path.join(frames_dir, frames[0]))
    W, H = first.size
    print(f"Extracted {len(frames)} frames @ {W}x{H}")

    stub = []
    if args.deepest_landmark:
        stub = [{
            "label": args.deepest_landmark,
            "bbox": [W // 4, H // 4, W // 2, H // 2],
            "cx": W / 2.0, "cy": H / 2.0,
            "area": (W * H) / 4.0,
            "segmentation": [],
        }]

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for i, name in enumerate(frames):
            row = {
                "subset": args.subset,
                "split": "all",
                "session": args.session,
                "frame": i + 1,
                "path": os.path.abspath(
                    os.path.join(frames_dir, name)),
                "width": W,
                "height": H,
                "annotations": [dict(a) for a in stub],
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote manifest -> {manifest_path}")

    if args.write_yaml:
        yaml_path = os.path.join(args.out_dir, "uaal.yaml")
        fx = fy = W / 2.0
        with open(yaml_path, "w") as f:
            f.write(f"dataset_name: 'c3vd'\n"
                    f"camera_params:\n"
                    f"  image_height: {H}\n"
                    f"  image_width: {W}\n"
                    f"  fx: {fx}\n  fy: {fy}\n"
                    f"  cx: {W/2.0}\n  cy: {H/2.0}\n"
                    f"  png_depth_scale: 2.55\n"
                    f"  crop_edge: 0\n")
        print(f"Wrote yaml -> {yaml_path}")
        print(f"  -> copy to external/Endo-2DTAM/configs/data/uaal.yaml "
              f"to use for SLAM on this video.")

    print()
    print("Next steps:")
    print()
    print("1. Depth estimation (conda activate depth-eval):")
    print(f"   python depth_eval_uaal.py \\")
    print(f"       --manifest {manifest_path} \\")
    print(f"       --subset {args.subset} --session {args.session} \\")
    print(f"       --out_dir {args.out_dir}/depth_out \\")
    print(f"       --model_size small")
    print()
    print("2. Build Endo-2DTAM scene:")
    print(f"   python build_endo2dtam_scene.py \\")
    print(f"       --manifest {manifest_path} \\")
    print(f"       --depth_dir {args.out_dir}/depth_out/depth \\")
    print(f"       --subset {args.subset} --session {args.session} \\")
    print(f"       --out_dir external/Endo-2DTAM/data/UAAL \\")
    print(f"       --scene_name {args.session}")
    print()
    print("3. Update Endo-2DTAM yaml to match this video's resolution:")
    if args.write_yaml:
        print(f"   cp {args.out_dir}/uaal.yaml "
              f"external/Endo-2DTAM/configs/data/uaal.yaml")
    else:
        print(f"   Edit external/Endo-2DTAM/configs/data/uaal.yaml: "
              f"image_height={H}, image_width={W}, "
              f"fx=fy={W/2.0}, cx={W/2.0}, cy={H/2.0}")
    print()
    print("4. Run SLAM:")
    print(f"   conda deactivate && conda activate e2dtam   # SLAM needs the torch-1.13 env")
    print(f"   cd external/Endo-2DTAM")
    print(f"   SCENE_NAME={args.session} OUTPUT_NAME=video_run "
          f"BASE_H={H} BASE_W={W} DOWN_SCALE=1 \\")
    print(f"       python scripts/main.py configs/uaal/uaal_base.py")
    print()
    print("5. Render dashboard:")
    print(f"   conda deactivate && conda activate depth-eval   # back to the torch-2.x env")
    print(f"   cd /home/mi3dr/projects/endomap-clean")
    print(f"   python render_dashboard_v1.py \\")
    print(f"       --corpus atm22_corpus.h5 \\")
    print(f"       --rgb_dir {args.out_dir}/depth_out/rgb \\")
    print(f"       --depth_dir {args.out_dir}/depth_out/depth \\")
    print(f"       --params_npz external/Endo-2DTAM/experiments/"
          f"video_run/{args.session}/params.npz \\")
    print(f"       --uaal_manifest {manifest_path} \\")
    print(f"       --uaal_subset {args.subset} --uaal_session {args.session} \\")
    print(f"       --out_dir {args.out_dir}/dashboard")
    print()
    print(f"6. Stitch:")
    print(f"   ffmpeg -framerate {int(args.fps)} -pattern_type glob \\")
    print(f"       -i '{args.out_dir}/dashboard/frame_*.png' \\")
    print(f"       -c:v libx264 -pix_fmt yuv420p "
          f"{args.out_dir}/dashboard.mp4")


if __name__ == "__main__":
    main()
