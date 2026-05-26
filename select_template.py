"""Pick the canonical template subject for inter-subject correspondence.

Score per subject:
  + 1 per named landmark present (max 5)
  - 2 per QA flag
  tiebreak: smallest absolute deviation from the median bifurcation count

Writes /metadata/template_subject_id back into atm22_corpus.h5.

Usage:
  python select_template.py --corpus atm22_corpus.h5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    with h5py.File(args.corpus, "a") as h5:
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        scores = np.zeros(len(ids), dtype=np.float32)
        n_bifs = np.zeros(len(ids), dtype=np.float32)
        for i, sid in enumerate(ids):
            g = h5[f"subjects/{sid}"]
            if "label_name" not in g:
                continue
            labels = set(s.decode() if isinstance(s, bytes) else str(s)
                         for s in g["label_name"][:])
            qa = list(g["qa_flags"][:])
            scores[i] = len(labels) - 2 * len(qa)
            n_bifs[i] = int(g.attrs.get("n_bifurcations", 0))

        median_nb = float(np.median(n_bifs))
        deviation = np.abs(n_bifs - median_nb)
        order = sorted(range(len(ids)),
                       key=lambda i: (-scores[i], deviation[i]))
        template = ids[order[0]]
        print(f"Template: {template}")
        print(f"  score          = {scores[order[0]]:.0f}")
        print(f"  n_bifurcations = {int(n_bifs[order[0]])} "
              f"(corpus median = {median_nb:.0f})")

        meta = h5["metadata"]
        if "template_subject_id" in meta:
            del meta["template_subject_id"]
        meta.create_dataset(
            "template_subject_id",
            data=np.array(template, dtype=h5py.string_dtype()))


if __name__ == "__main__":
    main()
