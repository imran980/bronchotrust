"""Compare phantom recon measurements to ground truth (R_GT).

Aligns recon camera centers to GT centers (Umeyama similarity -> scale s), then
converts the lumen-axis recovered radii (and mesh-slice areas) from recon units
to GT units and reports radial error % vs R_GT.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pycolmap

OD = Path("/home/mi3dr/projects/bronchotrust/runs/phantom")


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, Ds, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (Xc ** 2).sum() / len(X)
    s = (Ds * np.diag(S)).sum() / var
    return s, R, my - s * R @ mx


def main():
    gt = json.loads((OD / "gt_poses.json").read_text())
    R_GT = gt["R_GT"]
    gtC = {p["frame"]: np.array(p["C"]) for p in gt["poses"]}
    rec = pycolmap.Reconstruction(str(OD / "sparse/0"))
    rC = {}
    for im in rec.images.values():
        fi = int(re.search(r"f(\d+)", im.name).group(1))
        M = np.array(im.cam_from_world().matrix()); rC[fi] = -M[:3, :3].T @ M[:3, 3]
    common = sorted(set(gtC) & set(rC))
    X = np.array([rC[f] for f in common]); Y = np.array([gtC[f] for f in common])
    s, R, t = umeyama(X, Y)
    resid = np.linalg.norm((s * (R @ X.T).T + t) - Y, axis=1)
    print(f"alignment: {len(common)} cams  scale s={s:.4f}  median residual={np.median(resid):.3f} (GT units)")

    out = {"R_GT": R_GT, "scale": float(s), "align_residual_med": float(np.median(resid)),
           "n_common_cams": len(common)}

    # lumen-axis recovered radii
    lr = OD / "diag/lumen_report.json"
    if lr.exists():
        d = json.loads(lr.read_text())
        rec_R = np.array([n["r_median"] for n in d["nodes"]]) * s
        err = np.abs(rec_R - R_GT) / R_GT * 100
        out["lumen_axis"] = {"recovered_R_GTunits": [round(float(x), 3) for x in rec_R],
                             "median_recovered_R": round(float(np.median(rec_R)), 3),
                             "radial_error_pct_per_node": [round(float(x), 1) for x in err],
                             "median_radial_error_pct": round(float(np.median(err)), 1)}
        print(f"\nLUMEN-AXIS (cloud median radius): R_GT={R_GT}")
        print(f"  recovered R (GT units): median={np.median(rec_R):.3f}  range={rec_R.min():.2f}-{rec_R.max():.2f}")
        print(f"  radial error %: median={np.median(err):.1f}%  per-node={[round(float(x),0) for x in err]}")

    # mesh-slice areas -> radius
    for tag in ("watertight", "trim1"):
        mr = OD / f"diag/mesh_lumen_report_{tag}.json"
        if mr.exists():
            d = json.loads(mr.read_text())
            areas = [n["area"] for n in d["nodes"] if n.get("area")]
            n_closed = len(areas)
            if areas:
                Aphys = np.array(areas) * s ** 2
                Rmesh = np.sqrt(Aphys / np.pi)
                err = np.abs(Rmesh - R_GT) / R_GT * 100
                out[f"mesh_{tag}"] = {"n_closed": n_closed,
                                     "recovered_R": [round(float(x), 3) for x in Rmesh],
                                     "median_radial_error_pct": round(float(np.median(err)), 1)}
                print(f"\nMESH ({tag}): {n_closed}/10 closed; recovered R (GT) median={np.median(Rmesh):.3f} "
                      f"error median={np.median(err):.1f}%")
            else:
                out[f"mesh_{tag}"] = {"n_closed": 0}
                print(f"\nMESH ({tag}): 0/10 closed")

    (OD / "phantom_gt_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OD/'phantom_gt_report.json'}")


if __name__ == "__main__":
    main()
