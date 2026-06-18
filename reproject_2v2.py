"""2-V2 'real inside' demo.
(1) Color-drape: transfer the dense cloud's real RGB onto the double-Poisson mesh
    vertices -> double_poisson_colored.ply (fly through the inside in a viewer).
(2) Endoluminal reprojection: project the COLORED dense cloud from the actual COLMAP
    camera poses (OPENCV model, z-buffer splat) to synthesize the inside view, next
    to the real video frame (CLAHE+bezel) at the same index -> validation figure.
Visual/validation only; no geometry change. depth-eval env."""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2, open3d as o3d, pycolmap, re
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OD = Path("runs/batch4/2-V2"); DIAG = OD; VIDEO = Path("/home/mi3dr/dataset/validation-videos/First 15 Videos/2-V2.MP4")
FRAMES = [905, 1000, 1095, 1185]


def bezel(video, n=60):
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(7)); H = int(cap.get(4)); W = int(cap.get(3))
    cum = np.zeros((H, W), np.int32)
    for fi in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if ok: cum += (fr.mean(2) > 8).astype(np.int32)
    cap.release()
    m = ((cum >= max(int(0.2 * n), 5)).astype(np.uint8)) * 255
    return cv2.erode(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8), iterations=4) > 0


def real_frame(idx, mask, clahe):
    cap = cv2.VideoCapture(str(VIDEO)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read(); cap.release()
    if not ok: return None
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); im[~mask] = 0
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def synth_view(Pw, cols, im, cam, mask):
    M = np.array(im.cam_from_world().matrix()); R = M[:3, :3]; t = M[:3, 3]
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]); dist = np.array([k1, k2, p1, p2])
    Pc = (R @ Pw.T).T + t; z = Pc[:, 2]; front = z > 1e-3
    Pw2, z2, c2 = Pw[front], z[front], cols[front]
    uv = cv2.projectPoints(Pw2, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
    u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
    H, W = cam.height, cam.width
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z2, c2 = u[inb], v[inb], z2[inb], c2[inb]
    order = np.argsort(-z2)                       # far first, near overwrites
    u, v, c2 = u[order], v[order], (c2[order] * 255).astype(np.uint8)
    img = np.zeros((H, W, 3), np.uint8)
    for du in (-2, -1, 0, 1, 2):
        for dv in (-2, -1, 0, 1, 2):
            uu = np.clip(u + du, 0, W - 1); vv = np.clip(v + dv, 0, H - 1)
            img[vv, uu] = c2
    img[~mask] = 0
    return img


def main():
    pc = o3d.io.read_point_cloud(str(OD / "dense0/fused.ply"))
    Pw = np.asarray(pc.points); cols = np.asarray(pc.colors)
    fin = np.isfinite(Pw).all(1); Pw, cols = Pw[fin], cols[fin]

    # (1) color-drape mesh
    mesh = o3d.io.read_triangle_mesh(str(OD / "double_poisson.ply"))
    Vm = np.asarray(mesh.vertices)
    if len(Vm):
        _, idx = cKDTree(Pw).query(Vm)
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols[idx])
        o3d.io.write_triangle_mesh(str(OD / "double_poisson_colored.ply"), mesh)
        print(f"saved double_poisson_colored.ply ({len(Vm)} verts colored from cloud)")

    # (2) reprojection vs real frames
    rec = pycolmap.Reconstruction(str(OD / "sparse/0")); cam = list(rec.cameras.values())[0]
    by_frame = {int(re.search(r'f(\d+)', im.name).group(1)): im for im in rec.images.values()}
    mask = bezel(VIDEO); clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    rmask = cv2.resize(mask.astype(np.uint8), (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0

    fig, axes = plt.subplots(len(FRAMES), 2, figsize=(11, 5.4 * len(FRAMES)))
    for r, fr in enumerate(FRAMES):
        f = min(by_frame, key=lambda x: abs(x - fr))
        syn = synth_view(Pw, cols, by_frame[f], cam, rmask)
        real = real_frame(f, mask, clahe)
        axes[r, 0].imshow(syn); axes[r, 0].set_title(f"f{f}: SYNTHESIZED from .ply (cloud reprojected)", fontsize=11)
        axes[r, 1].imshow(real if real is not None else np.zeros_like(syn))
        axes[r, 1].set_title(f"f{f}: REAL video frame (CLAHE+bezel)", fontsize=11)
        for c in (0, 1): axes[r, c].axis("off")
    fig.suptitle("2-V2 — inside of the reconstruction reprojected from the real camera pose, vs the real frame\n"
                 "(colors are CLAHE-enhanced; synthesized view has holes where the cloud is sparse)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(OD / "reproject_vs_real.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved reproject_vs_real.png -> {OD}")


if __name__ == "__main__":
    main()
