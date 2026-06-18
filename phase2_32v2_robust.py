"""32-V2 ROBUST REMAP (poses only, STOP before MVS).

Reuses the 68-frame set (14 sub_in + 40 bridge + 14 sub_out) already extracted
under runs/phase2_subcord/32-V2_bridge40/images. Pinned APPROX intrinsics
(32_v1 calib). Stricter COLMAP mapper, then iterative outlier-camera pruning
(center dist > 10x median from centroid) + intrinsics-FIXED bundle adjustment.

Reports before/after: camera count, components, outlier ratio, lat/along,
median tri, p95 tri; plus separate sub_in / sub_out / fused(sub_in+sub_out)
metrics, and the same-scene-scale check (sub_in vs sub_out centroid distance).

Success = clean fused model (no gross outliers) AND sub_in/sub_out centroids in
the same scene scale. Else: mark 32_V2 UNUSABLE without real calibration.
"""
from __future__ import annotations
import json, re, shutil, subprocess
from pathlib import Path
import numpy as np, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLMAP = "/home/mi3dr/.conda/envs/colmap-cuda/bin/colmap"
SRC = Path("/home/mi3dr/projects/bronchotrust/runs/phase2_subcord/32-V2_bridge40")
OD = Path("/home/mi3dr/projects/bronchotrust/runs/phase2_subcord/32-V2_robust")
CALIB = Path("/home/mi3dr/projects/bronchotrust/runs/retriage_first15/_calib/32_v1/intrinsics_pinned.json")
WIN = {"sub_in": [191, 305], "sub_out": [2229, 2281]}
OUTLIER_K = 10.0


def seg(fi):
    if WIN["sub_in"][0] <= fi <= WIN["sub_in"][1]:
        return "sub_in"
    if WIN["sub_out"][0] <= fi <= WIN["sub_out"][1]:
        return "sub_out"
    return "bridge"


def load(model):
    rec = pycolmap.Reconstruction(str(model))
    cc, segs, names = {}, {}, {}
    for img in rec.images.values():
        fi = int(re.search(r"f(\d+)", img.name).group(1))
        M = np.array(img.cam_from_world().matrix()); R, t = M[:3, :3], M[:3, 3]
        cc[img.image_id] = -R.T @ t; segs[img.image_id] = seg(fi); names[img.image_id] = img.name
    return rec, cc, segs, names


def pca_lat_along(C):
    C = np.asarray(C, float)
    if len(C) < 3:
        return None
    Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    sd = (Cc @ Vt.T).std(0, ddof=1)
    return float(np.sqrt(sd[1] ** 2 + sd[2] ** 2) / sd[0]) if sd[0] > 1e-12 else None


def outlier_ratio(C):
    C = np.asarray(C, float); ctr = np.median(C, 0); d = np.linalg.norm(C - ctr, axis=1)
    return float(d.max() / max(np.median(d), 1e-12)), d


def subset_metrics(cc, segs, which):
    if which == "subcord":
        C = [cc[i] for i in cc if segs[i] in ("sub_in", "sub_out")]
    elif which == "all":
        C = [cc[i] for i in cc]
    else:
        C = [cc[i] for i in cc if segs[i] == which]
    if len(C) < 3:
        return {"n": len(C)}
    r, _ = outlier_ratio(C)
    return {"n": len(C), "lat_along": pca_lat_along(C), "outlier_ratio": round(r, 2)}


def tri_metrics(rec, cc, segs):
    sub_ids = {i for i in cc if segs[i] in ("sub_in", "sub_out")}
    tri = []
    for p in rec.points3D.values():
        obs = [el.image_id for el in p.track.elements if el.image_id in cc]
        if len(obs) < 2 or not (set(obs) & sub_ids):
            continue
        X = np.array(p.xyz); V = np.array([cc[i] - X for i in obs]); V /= np.linalg.norm(V, axis=1, keepdims=True)
        tri.append(float(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max()))
    tri = np.array(tri)
    return (float(np.median(tri)) if len(tri) else None,
            float(np.percentile(tri, 95)) if len(tri) else None, int(len(tri)))


def full_metrics(model):
    rec, cc, segs, names = load(model)
    C_all = [cc[i] for i in cc]
    r_all, _ = outlier_ratio(C_all)
    mt, p95, npts = tri_metrics(rec, cc, segs)
    ci = np.array([cc[i] for i in cc if segs[i] == "sub_in"])
    co = np.array([cc[i] for i in cc if segs[i] == "sub_out"])
    cen_dist = float(np.linalg.norm(ci.mean(0) - co.mean(0))) if len(ci) and len(co) else None
    spread_in = float(max((np.linalg.norm(a - b) for a in ci for b in ci), default=0)) if len(ci) else 0
    return {
        "n_reg": len(cc), "n_sparse": int(rec.num_points3D()),
        "reproj_px": round(float(np.mean([p.error for p in rec.points3D.values()])), 3) if rec.num_points3D() else None,
        "all_outlier_ratio": round(r_all, 2), "all_lat_along": round(pca_lat_along(C_all) or 0, 5),
        "subcord_median_tri": round(mt, 3) if mt else None, "subcord_p95_tri": round(p95, 3) if p95 else None,
        "subcord_n_points": npts,
        "sub_in": subset_metrics(cc, segs, "sub_in"),
        "sub_out": subset_metrics(cc, segs, "sub_out"),
        "fused_subcord": subset_metrics(cc, segs, "subcord"),
        "subin_subout_centroid_dist": round(cen_dist, 3) if cen_dist is not None else None,
        "sub_in_internal_spread": round(spread_in, 3),
        "same_scale": (cen_dist is not None and spread_in > 0 and cen_dist < 5 * spread_in),
        "_cc": {i: cc[i].tolist() for i in cc}, "_segs": {i: segs[i] for i in segs}, "_names": names}


def run(cmd):
    return subprocess.run([COLMAP] + cmd, capture_output=True, text=True)


def prune_and_ba(model_in, model_out, it):
    rec, cc, segs, names = load(model_in)
    C = np.array([cc[i] for i in cc]); ids = list(cc)
    ctr = np.median(C, 0); d = np.linalg.norm(C - ctr, axis=1); med = np.median(d)
    outliers = [names[ids[k]] for k in range(len(ids)) if d[k] > OUTLIER_K * med]
    if not outliers:
        return 0, model_in
    ol_file = OD / f"outliers_{it}.txt"
    ol_file.write_text("\n".join(outliers) + "\n")
    pruned = OD / f"sfm_prune{it}"; pruned.mkdir(exist_ok=True)
    run(["image_deleter", "--input_path", str(model_in), "--output_path", str(pruned),
         "--image_names_path", str(ol_file)])
    run(["point_filtering", "--input_path", str(pruned), "--output_path", str(pruned)])
    model_out.mkdir(exist_ok=True)
    run(["bundle_adjuster", "--input_path", str(pruned), "--output_path", str(model_out),
         "--BundleAdjustment.refine_focal_length", "0",
         "--BundleAdjustment.refine_principal_point", "0",
         "--BundleAdjustment.refine_extra_params", "0",
         "--BundleAdjustment.max_num_iterations", "200"])
    return len(outliers), model_out


def traj_plot(model, title, png):
    rec, cc, segs, names = load(model)
    COL = {"sub_in": "#3cb43c", "sub_out": "#19c819", "bridge": "#3c78e6"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for axp, (ix, iy, lx, ly) in zip(ax, [(0, 2, "X", "Z"), (0, 1, "X", "Y")]):
        for i in cc:
            axp.scatter(cc[i][ix], cc[i][iy], c=COL[segs[i]], s=45, edgecolors="k", linewidths=.4)
        axp.set_xlabel(lx); axp.set_ylabel(ly); axp.set_aspect("equal", "datalim"); axp.grid(alpha=.3); axp.set_title(f"{lx}{ly}")
    import matplotlib.patches as mp
    ax[0].legend(handles=[mp.Patch(color=COL[s], label=s) for s in COL], fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)


def main():
    if OD.exists():
        shutil.rmtree(OD)
    OD.mkdir(parents=True)
    intr = json.loads(CALIB.read_text())
    names = sorted(p.name for p in (SRC / "images").glob("f*.png"))
    print(f"32-V2 robust remap: {len(names)} frames, APPROX intrinsics fx={intr['fx']:.2f}")

    from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction as hr
    img_dir = SRC / "images"
    fconf = extract_features.confs["superpoint_max"]; mconf = match_features.confs["superpoint+lightglue"]
    feats = extract_features.main(fconf, img_dir, OD, image_list=names)
    pairs = OD / "pairs.txt"; pairs_from_exhaustive.main(pairs, image_list=names)
    matches = match_features.main(mconf, pairs, fconf["output"], OD)
    params = ",".join(f"{p:.10g}" for p in [intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["k1"], intr["k2"], 0.0, 0.0])
    sfm = OD / "sfm"; sfm.mkdir()
    print("stricter mapper (init_tri=8, filter_tri=3, filter_reproj=3, abs_err=8) ...", flush=True)
    hr.main(sfm, img_dir, pairs, feats, matches, image_list=names,
            camera_mode=hr.pycolmap.CameraMode.SINGLE,
            image_options={"camera_model": "OPENCV", "camera_params": params},
            verbose=False, mapper_options={
                "min_num_matches": 15, "multiple_models": True, "max_num_models": 50, "min_model_size": 3,
                "ba_refine_focal_length": False, "ba_refine_extra_params": False, "ba_refine_principal_point": False,
                "mapper": {"init_min_tri_angle": 8.0, "init_max_error": 4.0, "init_min_num_inliers": 30,
                           "init_max_forward_motion": 0.95, "abs_pose_max_error": 8.0,
                           "abs_pose_min_num_inliers": 30, "abs_pose_min_inlier_ratio": 0.25,
                           "filter_max_reproj_error": 3.0, "filter_min_tri_angle": 3.0}})
    n_components = len([d for d in (sfm / "models").iterdir() if d.is_dir()]) if (sfm / "models").exists() else 1

    before = full_metrics(sfm)
    traj_plot(sfm, f"32-V2 BEFORE prune  reg={before['n_reg']}  outlier_ratio={before['all_outlier_ratio']}", OD / "traj_before.png")

    # iterative prune + BA
    cur = sfm; total_pruned = 0; iters = []
    for it in range(4):
        npr, nxt = prune_and_ba(cur, OD / f"sfm_ba{it}", it)
        if npr == 0:
            break
        total_pruned += npr; iters.append({"it": it, "pruned": npr}); cur = nxt
        print(f"  iter {it}: pruned {npr} outlier cams", flush=True)

    after = full_metrics(cur)
    after["n_components"] = 1
    traj_plot(cur, f"32-V2 AFTER prune+BA  reg={after['n_reg']}  outlier_ratio={after['all_outlier_ratio']}", OD / "traj_after.png")

    success = (after["all_outlier_ratio"] <= OUTLIER_K) and after["same_scale"]
    verdict = "CLEAN FUSED MODEL" if success else "UNUSABLE without real calibration (still flings cameras)"

    for d in (before, after):
        for k in ("_cc", "_segs", "_names"):
            d.pop(k, None)
    report = {"n_frames": len(names), "approx_intrinsics": True, "n_components_initial": n_components,
              "total_cams_pruned": total_pruned, "prune_iters": iters,
              "before": before, "after": after, "success": success, "verdict": verdict}
    (OD / "robust_report.json").write_text(json.dumps(report, indent=2))

    def line(tag, m):
        return (f"  {tag:<7} reg={m['n_reg']:<3} outlier_ratio={m['all_outlier_ratio']:<12} "
                f"all_lat/along={m['all_lat_along']:<9} medTri={m['subcord_median_tri']} p95={m['subcord_p95_tri']}")
    print("\n=== 32-V2 ROBUST REMAP ===")
    print(f"initial components: {n_components}   cameras pruned: {total_pruned}")
    print(line("BEFORE", before)); print(line("AFTER", after))
    print("\nsubset (AFTER):")
    print(f"  sub_in : {after['sub_in']}")
    print(f"  sub_out: {after['sub_out']}")
    print(f"  fused  : {after['fused_subcord']}")
    print(f"  sub_in<->sub_out centroid dist = {after['subin_subout_centroid_dist']}  "
          f"(sub_in internal spread = {after['sub_in_internal_spread']})  same_scale={after['same_scale']}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
