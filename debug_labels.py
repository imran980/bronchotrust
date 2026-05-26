"""Debug a single subject's labeling — prints the walker's trace.

Loads one subject from the corpus, identifies the trachea root, and
walks toward the carina, printing each degree-3+ node it visits with
its non-parent side branches' subtree bifurcation counts. Then does
the same one hop further (carina's children). Run this on a flagged
subject to see why labeling fails.

Usage:
  python debug_labels.py --corpus atm22_corpus.h5 --subject ATM_001_0000
"""
from __future__ import annotations

import argparse

import h5py
import networkx as nx
import numpy as np

from label_bifurcations import (
    _build_graph, _find_trachea_root, _subtree_size,
    _walk_to_next_real_bifurcation, MIN_SIDE_RATIO, MIN_SIDE_NODES,
)


def _trace_walk(G, start, parent, nodes, label, max_hops=2000):
    print(f"\n[{label}] start={start} parent={parent}")
    prev, curr = parent, start
    n_shaft = 0
    for hop in range(max_hops):
        deg = G.degree(curr)
        if deg == 1:
            print(f"  hop {hop}: node {curr} deg=1 ENDPOINT — stop "
                  f"(skipped {n_shaft} degree-2 shaft nodes)")
            return curr
        if deg == 2:
            nxts = [n for n in G.neighbors(curr) if n != prev]
            prev, curr = curr, nxts[0]
            n_shaft += 1
            continue
        sides = [(nb, _subtree_size(G, nb, curr))
                 for nb in G.neighbors(curr) if nb != prev]
        if sides:
            max_size = max(s[1] for s in sides)
            threshold = max(MIN_SIDE_NODES, int(max_size * MIN_SIDE_RATIO))
            substantial = [s for s in sides if s[1] >= threshold]
        else:
            substantial = []
            threshold = 0
        pos = nodes[curr]
        print(f"  hop {hop} (after {n_shaft} shaft): node {curr} deg={deg} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
              f"sides={sides} thr={threshold} substantial={len(substantial)}")
        n_shaft = 0
        if len(substantial) >= 2:
            print(f"           -> REAL bifurcation, return")
            return curr
        if not sides:
            return curr
        sides.sort(key=lambda x: -x[1])
        prev, curr = curr, sides[0][0]
        print(f"           -> fake, follow side {prev}->{curr}")
    print(f"  hop budget {max_hops} exhausted at node {curr} deg={G.degree(curr)}")
    return curr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--subject", default=None,
                    help="Subject ID. If omitted, auto-picks the first "
                         "subject flagged carina_underbranched.")
    args = ap.parse_args()

    with h5py.File(args.corpus, "r") as h5:
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        if args.subject is None:
            for sid in ids:
                qa = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in h5[f"subjects/{sid}/qa_flags"][:]]
                if any("carina_underbranched" in f or f.startswith("missing:")
                       for f in qa):
                    args.subject = sid
                    print(f"Auto-selected flagged subject: {sid} qa={qa}")
                    break
            if args.subject is None:
                print("No flagged subjects found.")
                return

        g = h5[f"subjects/{args.subject}"]
        nodes = g["centerline_nodes"][:]
        edges = g["centerline_edges"][:]
        existing_qa = [s.decode() if isinstance(s, bytes) else str(s)
                       for s in g["qa_flags"][:]]
        existing_labels = [s.decode() if isinstance(s, bytes) else str(s)
                           for s in g["label_name"][:]]

    G = _build_graph(len(nodes), edges)
    print(f"\nSubject {args.subject}: {len(nodes)} nodes, {len(edges)} edges")
    print(f"  existing labels = {existing_labels}")
    print(f"  existing qa     = {existing_qa}")
    degs = np.bincount([G.degree(n) for n in G.nodes])
    print(f"degree histogram: " +
          " ".join(f"d{i}={c}" for i, c in enumerate(degs) if c))

    trachea = _find_trachea_root(G, nodes)
    print(f"trachea root: {trachea} z={nodes[trachea, 2]:.1f}\n"
          f"degree histogram: " +
          " ".join(f"d{i}={c}" for i, c in
                   enumerate(np.bincount([G.degree(n) for n in G.nodes]))
                   if c))

    nbrs = list(G.neighbors(trachea))
    carina = _trace_walk(G, nbrs[0], trachea, nodes, "trachea -> carina")
    print(f"\ncarina candidate = {carina} (deg={G.degree(carina)})")
    if G.degree(carina) < 3:
        print("  -> NOT a bifurcation, no_carina")
        return

    for i, nb in enumerate(G.neighbors(carina)):
        # Walk every non-trachea-direction neighbor. The trace itself
        # will reach an endpoint or real bif and tell us which.
        target = _trace_walk(G, nb, carina, nodes,
                             f"carina[child{i}] -> next bif")
        print(f"  child {i} (nb={nb}): target={target} "
              f"deg={G.degree(target)}")


if __name__ == "__main__":
    main()
