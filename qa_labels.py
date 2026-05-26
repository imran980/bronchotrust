"""Per-subject labeling summary; flags subjects that need eyeballing.

Usage:
  python qa_labels.py --corpus atm22_corpus.h5
  python qa_labels.py --corpus atm22_corpus.h5 --csv qa_labels.csv
"""
from __future__ import annotations

import argparse
import csv

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    with h5py.File(args.corpus, "r") as h5:
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        for sid in ids:
            g = h5[f"subjects/{sid}"]
            labels = sorted(set(
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g["label_name"][:])) if "label_name" in g else []
            qa = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in g["qa_flags"][:]] if "qa_flags" in g else []
            rows.append({
                "subject_id": sid,
                "n_labels": len(labels),
                "labels": "|".join(labels),
                "n_bifurcations": int(g.attrs.get("n_bifurcations", 0)),
                "n_qa_flags": len(qa),
                "qa_flags": "|".join(qa),
            })

    flagged = [r for r in rows if r["n_qa_flags"] > 0 or r["n_labels"] < 5]
    clean = len(rows) - len(flagged)
    print(f"{len(rows)} subjects total; {clean} clean, {len(flagged)} flagged.")
    if flagged:
        print("\nFlagged subjects (review with a 3D viewer):")
        for r in sorted(flagged, key=lambda r: (-r["n_qa_flags"],
                                                 r["n_labels"])):
            print(f"  {r['subject_id']:20s}  "
                  f"labels={r['n_labels']}/5  "
                  f"qa={r['qa_flags'] or '-'}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
