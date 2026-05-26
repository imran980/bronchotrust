"""
Reconstruction driver — pluggable backbone for monocular bronchoscopy.

Phase 1 (baseline comparison) and Phase 2 (synthetic eval) both need the
same thing: run a reconstruction method on a directory of frames and emit
a comparable output. This module defines that contract and the three
baselines we benchmark against.

Backbones (Phase 1 set):
    * mast3r-slam : MASt3R-SLAM (Murai et al., 2025) — sequential, learned
                    two-view priors. Primary candidate.
    * dust3r      : DUSt3R / MASt3R pairwise + global alignment.
    * colmap      : Classical SfM + MVS. Sanity-check baseline.

Canonical output (under --out):
    poses.npz          N x 4 x 4 c2w matrices, key 'c2w'.
    mesh.ply           Dense surface mesh (vertices + faces). Required.
    points.ply         Dense point cloud. Optional; used if mesh absent.
    intrinsics.yaml    K matrix used (or estimated). camera-yaml schema.
    confidence.npy     Per-vertex (or per-point) confidence in [0, 1].
                       Optional but expected from learned backbones.
    meta.json          backbone, params, runtime, which outputs present.

Each backbone is a stub right now: check_installed() works, run() raises
NotImplementedError until wired. Use --check-installed to see what's set up.

CLI:
    python reconstruct.py --check-installed
    python reconstruct.py --backbone mast3r-slam \\
        --frames runs/$SESSION/frames --out runs/$SESSION/recon_mast3r
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------- IO contract ----------

@dataclass
class ReconstructionResult:
    """Canonical paths a backbone writes into out_dir.

    Backbones don't have to produce every file — `write_meta` records
    which are present so downstream phases can dispatch (e.g. points-only
    backbones still feed into uncertainty + SSM matching, just via a
    different code path)."""
    out_dir: Path
    backbone: str

    @property
    def poses_npz(self) -> Path:
        return self.out_dir / "poses.npz"

    @property
    def mesh_ply(self) -> Path:
        return self.out_dir / "mesh.ply"

    @property
    def points_ply(self) -> Path:
        return self.out_dir / "points.ply"

    @property
    def intrinsics_yaml(self) -> Path:
        return self.out_dir / "intrinsics.yaml"

    @property
    def confidence_npy(self) -> Path:
        return self.out_dir / "confidence.npy"

    @property
    def meta_json(self) -> Path:
        return self.out_dir / "meta.json"

    def write_meta(self, params: dict, runtime_s: float) -> None:
        meta = {
            "backbone": self.backbone,
            "params": params,
            "runtime_s": float(runtime_s),
            "outputs_present": {
                "poses": self.poses_npz.exists(),
                "mesh": self.mesh_ply.exists(),
                "points": self.points_ply.exists(),
                "intrinsics": self.intrinsics_yaml.exists(),
                "confidence": self.confidence_npy.exists(),
            },
        }
        self.meta_json.write_text(json.dumps(meta, indent=2))


# ---------- Backbone ABC ----------

def external_dir() -> Path:
    return Path(os.environ.get("BRONCHO_EXTERNAL_DIR", "~/external")).expanduser()


class Backbone(ABC):
    """One reconstruction method. Wraps an upstream repo or CLI."""

    name: str = ""
    repo_subdir: str = ""

    @abstractmethod
    def install_instructions(self) -> str:
        ...

    @abstractmethod
    def check_installed(self) -> Tuple[bool, str]:
        """Return (installed, reason_if_not). reason_if_not empty on success."""
        ...

    @abstractmethod
    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        ...

    def ensure_installed(self) -> None:
        ok, why = self.check_installed()
        if not ok:
            raise RuntimeError(
                f"backbone '{self.name}' not installed: {why}\n"
                f"install:\n{self.install_instructions()}"
            )


# ---------- Stubs ----------

class MASt3RSLAMBackbone(Backbone):
    """MASt3R-SLAM (Murai et al., 2025). Sequential SLAM on top of MASt3R
    pairwise priors. Primary candidate for Phase 1.

    Driven by subprocess to upstream main.py — their custom CUDA kernels
    (lietorch) need their own conda env. We don't import mast3r_slam in
    this process. Env python path is configurable via BRONCHO_MAST3R_PYTHON
    (default: /home/mi3dr/.conda/envs/mast3r-slam/bin/python).

    Upstream writes (relative to repo):
        logs/<save_as>/<seq_name>.txt   TUM trajectory: ts x y z qx qy qz qw
        logs/<save_as>/<seq_name>.ply   conf-filtered colored point cloud
        logs/<save_as>/keyframes/<seq_name>/*.png

    We map to our six-file contract by parsing the trajectory into Nx4x4
    c2w (poses.npz), copying the .ply (points.ply), and copying the
    intrinsics yaml if provided.
    """

    name = "mast3r-slam"
    repo_subdir = "MASt3R-SLAM"
    default_python = "/home/mi3dr/.conda/envs/mast3r-slam/bin/python"

    def install_instructions(self) -> str:
        ext = external_dir()
        return (
            f"git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git "
            f"{ext / self.repo_subdir}\n"
            f"cd {ext / self.repo_subdir} && follow upstream README "
            "(torch 2.x + CUDA 12.x; custom kernels)."
        )

    def _python(self) -> str:
        return os.environ.get("BRONCHO_MAST3R_PYTHON", self.default_python)

    def check_installed(self) -> Tuple[bool, str]:
        repo = external_dir() / self.repo_subdir
        if not repo.exists():
            return False, f"no repo at {repo}"
        if not Path(self._python()).exists():
            return False, f"no python at {self._python()} (set BRONCHO_MAST3R_PYTHON)"
        ckpts = repo / "checkpoints"
        needed = [
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
        ]
        missing = [n for n in needed if not (ckpts / n).exists()]
        if missing:
            return False, f"missing checkpoints in {ckpts}: {missing}"
        return True, ""

    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        self.ensure_installed()
        repo = external_dir() / self.repo_subdir
        save_as = out_dir.name
        cmd = [
            self._python(), "main.py",
            "--dataset", str(frames_dir.resolve()),
            "--config", "config/base.yaml",
            "--no-viz",
            "--save-as", save_as,
        ]
        if intrinsics_yaml is not None:
            cmd += ["--calib", str(intrinsics_yaml.resolve())]
        print(f"[mast3r-slam] cwd={repo}")
        print(f"[mast3r-slam] cmd={shlex.join(cmd)}")
        subprocess.run(cmd, cwd=repo, check=True)

        # Map upstream outputs -> our six-file contract.
        seq_name = frames_dir.resolve().stem
        upstream_log = repo / "logs" / save_as
        upstream_traj = upstream_log / f"{seq_name}.txt"
        upstream_ply = upstream_log / f"{seq_name}.ply"
        if not upstream_traj.exists():
            raise FileNotFoundError(
                f"MASt3R-SLAM did not produce trajectory: {upstream_traj}"
            )
        if not upstream_ply.exists():
            raise FileNotFoundError(
                f"MASt3R-SLAM did not produce point cloud: {upstream_ply}"
            )

        result = ReconstructionResult(out_dir=out_dir, backbone=self.name)
        _tum_traj_to_poses_npz(upstream_traj, result.poses_npz)
        shutil.copy(upstream_ply, result.points_ply)
        if intrinsics_yaml is not None:
            shutil.copy(intrinsics_yaml, result.intrinsics_yaml)
        return result


def _tum_traj_to_poses_npz(traj_txt: Path, out_npz: Path) -> None:
    """Convert TUM trajectory (ts tx ty tz qx qy qz qw) to N x 4 x 4 c2w."""
    rows = np.loadtxt(traj_txt)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    timestamps = rows[:, 0]
    t = rows[:, 1:4]
    q = rows[:, 4:8]  # xyzw
    R = _quat_xyzw_to_R(q)
    poses = np.zeros((len(rows), 4, 4), dtype=np.float64)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t
    poses[:, 3, 3] = 1.0
    np.savez(out_npz, c2w=poses, timestamps=timestamps)


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """N x 4 (xyzw) -> N x 3 x 3 rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class DUSt3RBackbone(Backbone):
    """DUSt3R per-pair AsymmetricCroCo3DStereo + global alignment.

    Sidesteps MASt3R-SLAM's retrieval-based tracker failure: DUSt3R works
    on a graph of pairs (sliding window) and never needs a retrieval
    database. We subsample to ~50 evenly-spaced frames because the
    PointCloudOptimizer holds all pair outputs in GPU memory — a complete
    graph on 350+ frames is ~60k pairs and OOMs immediately.

    Lives in the same conda env as MASt3R-SLAM (the dust3r package ships
    inside ~/external/MASt3R-SLAM/thirdparty/mast3r/dust3r). We delegate
    to dust3r_runner.py to keep the heavy deps off the driver process.
    """

    name = "dust3r"
    default_python = "/home/mi3dr/.conda/envs/mast3r-slam/bin/python"
    default_ckpt = ("/home/mi3dr/external/MASt3R-SLAM/checkpoints/"
                    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    dust3r_module = ("/home/mi3dr/external/MASt3R-SLAM/thirdparty/"
                     "mast3r/dust3r")

    def install_instructions(self) -> str:
        return (
            "DUSt3R ships with MASt3R-SLAM at "
            f"{self.dust3r_module}.\n"
            f"Download the DUSt3R checkpoint to {self.default_ckpt}:\n"
            "  wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/"
            "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        )

    def _python(self) -> str:
        return os.environ.get("BRONCHO_MAST3R_PYTHON", self.default_python)

    def _ckpt(self) -> Path:
        return Path(os.environ.get("BRONCHO_DUST3R_CKPT", self.default_ckpt))

    def check_installed(self) -> Tuple[bool, str]:
        if not Path(self.dust3r_module).exists():
            return False, f"no dust3r module at {self.dust3r_module}"
        if not Path(self._python()).exists():
            return False, f"no python at {self._python()}"
        if not self._ckpt().exists():
            return False, f"no DUSt3R checkpoint at {self._ckpt()}"
        return True, ""

    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        self.ensure_installed()
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).parent / "dust3r_runner.py"
        cmd = [
            self._python(), str(runner),
            "--frames_dir", str(frames_dir.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--ckpt", str(self._ckpt()),
        ]
        print(f"[dust3r] cmd={shlex.join(cmd)}")
        subprocess.run(cmd, check=True)
        return ReconstructionResult(out_dir=out_dir, backbone=self.name)


class DUSt3RSmoothBackbone(Backbone):
    """DUSt3R per-pair priors + SE(3) second-difference temporal smoothness.

    Same per-pair AsymmetricCroCo inference as DUSt3R; replaces the global
    aligner with a subclass that adds a pose-acceleration penalty between
    consecutive frames. Phase 1 diagnostic showed DUSt3R's priors work
    but its global aligner produces 15-27x zigzag — this directly
    addresses that defect while keeping the rest of the pipeline
    identical for clean ablation against the baseline.

    See dust3r_smooth_runner.py for the SmoothPointCloudOptimizer
    implementation. Defaults: lam_rot=1.0, lam_trans=1.0 (override via
    BRONCHO_DUST3R_LAM_{ROT,TRANS} env vars).
    """

    name = "dust3r-smooth"
    default_python = "/home/mi3dr/.conda/envs/mast3r-slam/bin/python"
    default_ckpt = ("/home/mi3dr/external/MASt3R-SLAM/checkpoints/"
                    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    dust3r_module = ("/home/mi3dr/external/MASt3R-SLAM/thirdparty/"
                     "mast3r/dust3r")

    def install_instructions(self) -> str:
        return ("Same install as 'dust3r' backbone. This is a different "
                "algorithm reusing the same DUSt3R model + checkpoint.")

    def _python(self) -> str:
        return os.environ.get("BRONCHO_MAST3R_PYTHON", self.default_python)

    def _ckpt(self) -> Path:
        return Path(os.environ.get("BRONCHO_DUST3R_CKPT", self.default_ckpt))

    def check_installed(self) -> Tuple[bool, str]:
        if not Path(self.dust3r_module).exists():
            return False, f"no dust3r module at {self.dust3r_module}"
        if not Path(self._python()).exists():
            return False, f"no python at {self._python()}"
        if not self._ckpt().exists():
            return False, f"no DUSt3R checkpoint at {self._ckpt()}"
        return True, ""

    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        self.ensure_installed()
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).parent / "dust3r_smooth_runner.py"
        lam_rot = os.environ.get("BRONCHO_DUST3R_LAM_ROT", "1.0")
        lam_trans = os.environ.get("BRONCHO_DUST3R_LAM_TRANS", "1.0")
        cmd = [
            self._python(), str(runner),
            "--frames_dir", str(frames_dir.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--ckpt", str(self._ckpt()),
            "--lam_rot", lam_rot,
            "--lam_trans", lam_trans,
        ]
        print(f"[dust3r-smooth] cmd={shlex.join(cmd)}")
        subprocess.run(cmd, check=True)
        return ReconstructionResult(out_dir=out_dir, backbone=self.name)


class DUSt3RFoEBackbone(Backbone):
    """DUSt3R + optical-flow focus-of-expansion sign-fix + SE(3) smoothness.

    Per-pair RAFT flow gives forward/backward sign for each pair via the
    FoE pattern (outward-radial = forward, inward = backward). This fixes
    DUSt3R's per-pair sign ambiguity (the root cause of its zigzag), then
    SE(3) smoothness cleans up residual high-freq wiggle. The full
    "ours" combination.

    Defaults: lam_sign=10, lam_rot=1, lam_trans=1. Override via env vars
    BRONCHO_DUST3R_LAM_{SIGN,ROT,TRANS}.
    """

    name = "dust3r-foe"
    default_python = "/home/mi3dr/.conda/envs/mast3r-slam/bin/python"
    default_ckpt = ("/home/mi3dr/external/MASt3R-SLAM/checkpoints/"
                    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    dust3r_module = ("/home/mi3dr/external/MASt3R-SLAM/thirdparty/"
                     "mast3r/dust3r")

    def install_instructions(self) -> str:
        return ("Same install as 'dust3r' backbone. Additionally needs "
                "torchvision RAFT (bundled with torchvision >= 0.13; no "
                "extra install for the mast3r-slam env).")

    def _python(self) -> str:
        return os.environ.get("BRONCHO_MAST3R_PYTHON", self.default_python)

    def _ckpt(self) -> Path:
        return Path(os.environ.get("BRONCHO_DUST3R_CKPT", self.default_ckpt))

    def check_installed(self) -> Tuple[bool, str]:
        if not Path(self.dust3r_module).exists():
            return False, f"no dust3r module at {self.dust3r_module}"
        if not Path(self._python()).exists():
            return False, f"no python at {self._python()}"
        if not self._ckpt().exists():
            return False, f"no DUSt3R checkpoint at {self._ckpt()}"
        return True, ""

    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        self.ensure_installed()
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).parent / "dust3r_foe_runner.py"
        cmd = [
            self._python(), str(runner),
            "--frames_dir", str(frames_dir.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--ckpt", str(self._ckpt()),
            "--lam_sign", os.environ.get("BRONCHO_DUST3R_LAM_SIGN", "10.0"),
            "--lam_rot",   os.environ.get("BRONCHO_DUST3R_LAM_ROT",   "1.0"),
            "--lam_trans", os.environ.get("BRONCHO_DUST3R_LAM_TRANS", "1.0"),
        ]
        print(f"[dust3r-foe] cmd={shlex.join(cmd)}")
        subprocess.run(cmd, check=True)
        return ReconstructionResult(out_dir=out_dir, backbone=self.name)


class COLMAPBackbone(Backbone):
    """Classical SfM baseline. With bronchoscopy-aware preprocessing:
    CLAHE (LAB-L) on each frame for feature contrast + a circular bezel
    mask so SIFT doesn't waste detectors on the black scope-barrel
    border. Sparse-only for Phase 1 (skip dense MVS).

    COLMAP is installed in the mast3r-slam env (conda-forge); we call its
    binary via absolute path so the driver works from any conda env. The
    actual subprocess wrapper that does CLAHE + mask + the COLMAP stages
    lives in colmap_runner.py (uses cv2 + plyfile from mast3r-slam env)."""

    name = "colmap"
    default_python = "/home/mi3dr/.conda/envs/mast3r-slam/bin/python"
    default_colmap = "/home/mi3dr/.conda/envs/mast3r-slam/bin/colmap"

    def install_instructions(self) -> str:
        return (
            "conda install -n mast3r-slam -c conda-forge colmap\n"
            "or apt: sudo apt install colmap (Ubuntu 22.04+)\n"
            "(or override BRONCHO_COLMAP_BIN env var)"
        )

    def _python(self) -> str:
        return os.environ.get("BRONCHO_MAST3R_PYTHON", self.default_python)

    def _colmap(self) -> str:
        return os.environ.get("BRONCHO_COLMAP_BIN", self.default_colmap)

    def check_installed(self) -> Tuple[bool, str]:
        if not Path(self._colmap()).exists():
            if shutil.which("colmap") is None:
                return False, f"no colmap binary at {self._colmap()} and not on PATH"
        if not Path(self._python()).exists():
            return False, f"no python at {self._python()}"
        return True, ""

    def run(self, frames_dir: Path, out_dir: Path,
            intrinsics_yaml: Optional[Path] = None) -> ReconstructionResult:
        self.ensure_installed()
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).parent / "colmap_runner.py"
        cmd = [
            self._python(), str(runner),
            "--frames_dir", str(frames_dir.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--colmap", self._colmap(),
        ]
        print(f"[colmap] cmd={shlex.join(cmd)}")
        subprocess.run(cmd, check=True)
        return ReconstructionResult(out_dir=out_dir, backbone=self.name)


# ---------- Factory + CLI ----------

BACKBONES = {
    "mast3r-slam": MASt3RSLAMBackbone,
    "dust3r": DUSt3RBackbone,
    "dust3r-smooth": DUSt3RSmoothBackbone,
    "dust3r-foe": DUSt3RFoEBackbone,
    "colmap": COLMAPBackbone,
}


def make_backbone(name: str) -> Backbone:
    if name not in BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}. choose from {sorted(BACKBONES)}."
        )
    return BACKBONES[name]()


def _print_install_status() -> int:
    n_ok = 0
    print(f"BRONCHO_EXTERNAL_DIR = {external_dir()}")
    for name, cls in BACKBONES.items():
        bb = cls()
        ok, why = bb.check_installed()
        if ok:
            n_ok += 1
            print(f"  [OK]      {name}")
        else:
            print(f"  [MISSING] {name}  ({why})")
    print(f"{n_ok}/{len(BACKBONES)} backbones installed.")
    return n_ok


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--check-installed", action="store_true",
                   help="Report which backbones are set up, then exit.")
    p.add_argument("--backbone", choices=sorted(BACKBONES))
    p.add_argument("--frames", type=Path, help="Directory of RGB frames.")
    p.add_argument("--out", type=Path, help="Output directory.")
    p.add_argument("--intrinsics", type=Path, default=None,
                   help="Optional intrinsics yaml (skip if --backbone estimates).")
    args = p.parse_args(argv)

    if args.check_installed:
        _print_install_status()
        return 0

    if not (args.backbone and args.frames and args.out):
        p.error("--backbone, --frames, --out are required unless --check-installed")

    if not args.frames.is_dir():
        p.error(f"--frames not a directory: {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    bb = make_backbone(args.backbone)
    print(f"[reconstruct] backbone={bb.name}  frames={args.frames}  out={args.out}")
    t0 = time.time()
    result = bb.run(args.frames, args.out, args.intrinsics)
    dt = time.time() - t0
    result.write_meta(
        params={"intrinsics_yaml": str(args.intrinsics) if args.intrinsics else None},
        runtime_s=dt,
    )
    print(f"[reconstruct] done in {dt:.1f}s -> {result.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
