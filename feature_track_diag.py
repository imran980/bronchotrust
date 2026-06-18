"""Feature-track diagnostic: 7-V1 vs phantom. Keypoints, matches, verified
matches, track lengths, view-count bins, sparse counts. Answers A-E."""
from __future__ import annotations
import sqlite3, sys, json
from pathlib import Path
import numpy as np, pycolmap

MAXID = 2147483647


def db_stats(db):
    c = sqlite3.connect(db); cur = c.cursor()
    kp = dict(cur.execute("SELECT image_id, rows FROM keypoints").fetchall())
    raw = [r for r in cur.execute("SELECT rows FROM matches WHERE rows>0").fetchall()]
    ver = [r[0] for r in cur.execute("SELECT rows FROM two_view_geometries WHERE rows>0").fetchall()]
    n_img = len(cur.execute("SELECT image_id FROM images").fetchall())
    c.close()
    kpv = np.array(list(kp.values()))
    raw = np.array([r[0] for r in raw]) if raw else np.array([0])
    ver = np.array(ver) if ver else np.array([0])
    return {"n_images": n_img,
            "kp_per_image": {"mean": float(kpv.mean()), "median": float(np.median(kpv)),
                             "min": int(kpv.min()), "max": int(kpv.max())},
            "n_raw_matched_pairs": int((raw > 0).sum()),
            "n_verified_pairs": int((ver > 0).sum()),
            "mean_verified_matches_per_pair": float(ver.mean()),
            "median_verified_matches_per_pair": float(np.median(ver)),
            "verified_pair_frac_of_possible": round((ver > 0).sum() / (n_img * (n_img - 1) / 2), 4)}


def track_stats(model):
    rec = pycolmap.Reconstruction(str(model))
    tl = np.array([len(p.track.elements) for p in rec.points3D.values()])
    errs = np.array([p.error for p in rec.points3D.values()])
    bins = {"2": int((tl == 2).sum()), "3-5": int(((tl >= 3) & (tl <= 5)).sum()),
            "6-10": int(((tl >= 6) & (tl <= 10)).sum()), ">10": int((tl > 10).sum())}
    hist, edges = np.histogram(tl, bins=[2, 3, 4, 5, 6, 8, 11, 16, 26, 1000])
    return {"n_reg_images": rec.num_reg_images(), "n_sparse": rec.num_points3D(),
            "track_len": {"mean": float(tl.mean()), "median": float(np.median(tl)),
                          "max": int(tl.max()), "p90": float(np.percentile(tl, 90))},
            "view_bins": bins, "mean_reproj_err_px": float(errs.mean()),
            "hist_counts": hist.tolist(), "hist_edges": [int(e) for e in edges]}


def main():
    cases = {
        "7-V1 (model0 subglottis)": ("runs/barbour_dense/7-V1/database.db", "runs/barbour_dense/7-V1/sparse/0"),
        "PHANTOM": ("runs/phantom/database.db", "runs/phantom/sparse/0"),
    }
    out = {}
    for name, (db, model) in cases.items():
        d = db_stats(db); t = track_stats(model)
        out[name] = {"db": d, "model": t}
        print(f"\n############ {name} ############")
        print(f"  images={d['n_images']}  kp/img: mean={d['kp_per_image']['mean']:.0f} "
              f"median={d['kp_per_image']['median']:.0f} min={d['kp_per_image']['min']} max={d['kp_per_image']['max']}")
        print(f"  raw matched pairs={d['n_raw_matched_pairs']}  verified pairs={d['n_verified_pairs']} "
              f"({d['verified_pair_frac_of_possible']*100:.1f}% of all possible)")
        print(f"  verified matches/pair: mean={d['mean_verified_matches_per_pair']:.1f} median={d['median_verified_matches_per_pair']:.0f}")
        print(f"  reg images={t['n_reg_images']}  SPARSE points={t['n_sparse']}  mean reproj={t['mean_reproj_err_px']:.2f}px")
        print(f"  track len: mean={t['track_len']['mean']:.2f} median={t['track_len']['median']:.0f} "
              f"p90={t['track_len']['p90']:.0f} max={t['track_len']['max']}")
        print(f"  view bins:  2-view={t['view_bins']['2']}  3-5={t['view_bins']['3-5']}  "
              f"6-10={t['view_bins']['6-10']}  >10={t['view_bins']['>10']}")
    # 7-V1 all models sparse
    print("\n### 7-V1 all sparse models ###")
    for m in sorted(Path("runs/barbour_dense/7-V1/sparse").glob("*/")):
        try:
            r = pycolmap.Reconstruction(str(m)); print(f"  {m.name}: reg={r.num_reg_images()} sparse={r.num_points3D()}")
        except Exception:
            pass
    Path("runs/feature_track_diag.json").write_text(json.dumps(out, indent=2))
    print("\nsaved runs/feature_track_diag.json")


if __name__ == "__main__":
    main()
