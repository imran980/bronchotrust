"""List subjects by Procrustes landmark residual.

Run after build_correspondence.py to spot subjects whose alignment to
the template is far worse than average — they're either anatomical
outliers (post-resection, unusual variants) or have a mis-identified
landmark that slipped through step 2A's QA.

Usage:
  python qa_correspondence.py --corpus atm22_corpus.h5 --top 10
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = []
    with h5py.File(args.corpus, "r") as h5:
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        for sid in ids:
            g = h5[f"subjects/{sid}"]
            if "correspondence_residual_mm" not in g:
                continue
            rows.append((sid, float(g["correspondence_residual_mm"][()])))

    if not rows:
        print("No correspondence residuals stored. "
              "Run build_correspondence.py first.")
        return

    rs = np.array([r[1] for r in rows])
    print(f"{len(rows)} subjects with correspondence: "
          f"mean={rs.mean():.2f}  median={np.median(rs):.2f}  "
          f"std={rs.std():.2f}  max={rs.max():.2f}  min={rs.min():.2f}")

    rows.sort(key=lambda r: -r[1])
    print(f"\nTop {args.top} worst (review in a 3D viewer):")
    for sid, r in rows[:args.top]:
        print(f"  {r:7.2f} mm  {sid}")
    print(f"\nBest {args.top}:")
    for sid, r in rows[-args.top:][::-1]:
        print(f"  {r:7.2f} mm  {sid}")


if __name__ == "__main__":
    main()
