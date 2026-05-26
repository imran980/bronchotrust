"""Rule-based anatomical labeling of bifurcations in the ATM22 corpus.

Walks atm22_corpus.h5 and, on each subject's centerline tree, labels
the small set of named landmarks needed to root + orient the tree for
inter-subject correspondence. Labels are written back into the corpus
under /subjects/<sid>/{label_node_idx, label_name, qa_flags}.

Coordinate convention: NIfTI affines are assumed to map voxels to RAS
(nibabel default). +X = patient right, +Y = anterior, +Z = superior.
If a large fraction of subjects come back flagged
``LR_orientation_suspect``, the data is probably LPS-encoded; flip X
upstream and re-run.

Named landmarks (the minimum set to define a rooted, oriented tree
through generation ~3):
  TRACHEA      root endpoint at the top of the trachea
  CARINA       first true bifurcation
  RMB_BIF      bifurcation at the end of the right main bronchus
               (where RUL and bronchus intermedius diverge)
  LMB_BIF      bifurcation at the end of the left main bronchus
               (where LUL and LLL diverge)
  BI_BIF       bifurcation in the bronchus intermedius
               (where RML and RLL diverge)

Beyond these, generation-3+ bifurcations are matched in step 2B by
BFS order within each named subtree.

Usage:
  python label_bifurcations.py --corpus atm22_corpus.h5
"""
from __future__ import annotations

import argparse
import collections

import h5py
import networkx as nx
import numpy as np
from tqdm import tqdm

LABEL_DTYPE = h5py.string_dtype()
EXPECTED_LABELS = ["TRACHEA", "CARINA", "RMB_BIF", "LMB_BIF", "BI_BIF"]
MIN_SIDE_RATIO = 0.1     # smaller side must be >=10% of larger
MIN_SIDE_NODES = 3       # ...and absolutely >=3 nodes
SUBTREE_CAP = 5000
MAX_WALK_HOPS = 2000


def _build_graph(n_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(a), int(b)) for a, b in edges)
    return G


def _side_branch_length(G, start, came_from, nodes):
    """Walk a degree-2 chain from `start` until a non-degree-2 node,
    returning total path length in mm. Used only for tie-breaking
    between two real children of the same parent bifurcation."""
    prev, curr = came_from, start
    length = float(np.linalg.norm(nodes[curr] - nodes[prev]))
    while G.degree(curr) == 2:
        nxts = [n for n in G.neighbors(curr) if n != prev]
        if not nxts:
            return length
        nxt = nxts[0]
        length += float(np.linalg.norm(nodes[nxt] - nodes[curr]))
        prev, curr = curr, nxt
    return length


def _subtree_size(G, start, came_from, cap=SUBTREE_CAP):
    """Count nodes reachable from `start` without crossing back through
    `came_from`. Capped for speed — comparable subtrees in the upper
    airway are thousands of nodes."""
    visited = {came_from, start}
    count = 1
    stack = [start]
    while stack and count < cap:
        u = stack.pop()
        for v in G.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            count += 1
            if count >= cap:
                return count
            stack.append(v)
    return count


def _walk_to_next_real_bifurcation(G, start, parent, nodes):
    """Walk from `start` (entered from `parent`) until reaching either
    (a) an endpoint, or (b) a "real" bifurcation — degree >= 3 where
    at least two non-parent side subtrees are each at least
    MIN_SIDE_RATIO of the largest side's subtree (and at least
    MIN_SIDE_NODES nodes absolutely).

    Returns (target, prev) where `prev` is the neighbor of `target`
    along the path we walked in on. Call sites pass that prev as
    `came_from` when chaining further walks; without it,
    `_children_bifurcations` walks back along the path it just came
    from (because the original `came_from` is many hops away and no
    longer a direct neighbor), reaches the opposite side of the
    parent's parent, and assigns a downstream label to a node that
    already has a different one. That was the source of the spurious
    LMB_BIF -> BI_BIF overwrites in ATM_003 and similar subjects.

    Relative ratio (not absolute count) is the right substantiality
    measure because a real anatomical bifurcation splits the
    downstream airway into comparable halves, while accessory bronchi
    and noise spurs are 1-2 orders of magnitude smaller than the main
    continuation.
    """
    prev, curr = parent, start
    for _ in range(MAX_WALK_HOPS):
        deg = G.degree(curr)
        if deg == 1:
            return curr, prev
        if deg == 2:
            nxts = [n for n in G.neighbors(curr) if n != prev]
            if not nxts:
                return curr, prev
            prev, curr = curr, nxts[0]
            continue
        sides = [(nb, _subtree_size(G, nb, curr))
                 for nb in G.neighbors(curr) if nb != prev]
        if not sides:
            return curr, prev
        max_size = max(s[1] for s in sides)
        threshold = max(MIN_SIDE_NODES, int(max_size * MIN_SIDE_RATIO))
        substantial = [s for s in sides if s[1] >= threshold]
        if len(substantial) >= 2:
            return curr, prev
        sides.sort(key=lambda x: -x[1])
        prev, curr = curr, sides[0][0]
    return curr, prev


def _children_bifurcations(G, parent_bif, came_from, nodes):
    """For each neighbor of `parent_bif` other than `came_from`, walk
    until a real bifurcation and return (target, length_mm, target_prev).
    `target_prev` is the immediate predecessor of `target` on the
    walked path; pass it as `came_from` to chain another call without
    walking back along the path you just came from.

    `came_from` MUST be a direct neighbor of `parent_bif`; otherwise
    the back-direction won't be filtered.
    """
    out = []
    for nb in G.neighbors(parent_bif):
        if nb == came_from:
            continue
        target, target_prev = _walk_to_next_real_bifurcation(
            G, nb, parent_bif, nodes)
        if target is None or G.degree(target) < 3:
            continue
        length = _side_branch_length(G, nb, parent_bif, nodes)
        out.append((target, length, target_prev))
    return out


def _find_trachea_root(G, nodes):
    endpoints = [n for n, d in G.degree() if d == 1]
    if not endpoints:
        return None
    return endpoints[int(np.argmax(nodes[endpoints, 2]))]


def _label_subject(nodes, edges):
    qa = []
    G = _build_graph(len(nodes), edges)

    trachea = _find_trachea_root(G, nodes)
    if trachea is None:
        return {}, ["no_root_endpoint"]

    nbrs = list(G.neighbors(trachea))
    if not nbrs:
        return {int(trachea): "TRACHEA"}, ["isolated_root"]
    carina, carina_prev = _walk_to_next_real_bifurcation(
        G, nbrs[0], trachea, nodes)
    if carina is None or G.degree(carina) < 3:
        return {int(trachea): "TRACHEA"}, ["no_carina"]

    labels = {int(trachea): "TRACHEA", int(carina): "CARINA"}

    children = _children_bifurcations(G, carina, carina_prev, nodes)
    if len(children) < 2:
        qa.append("carina_underbranched")
        return labels, qa
    children.sort(key=lambda x: -x[1])
    c1, c1_prev = children[0][0], children[0][2]
    c2, c2_prev = children[1][0], children[1][2]

    def _horizontality(a, b):
        v = nodes[b] - nodes[a]
        n = float(np.linalg.norm(v))
        return abs(v[0]) / n if n > 1e-6 else 0.0

    # Anatomical convention: RMB is more horizontal than LMB (the
    # heart pushes the left main bronchus downward). Using
    # horizontality as the discriminator is convention-independent;
    # using sign of X requires knowing whether the affine encodes RAS
    # or LPS, and ATM22 NIfTIs were converted from DICOM with
    # inconsistent conventions across subjects.
    h1 = _horizontality(carina, c1)
    h2 = _horizontality(carina, c2)
    if h1 > h2:
        rmb_bif, rmb_bif_prev = c1, c1_prev
        lmb_bif, lmb_bif_prev = c2, c2_prev
    else:
        rmb_bif, rmb_bif_prev = c2, c2_prev
        lmb_bif, lmb_bif_prev = c1, c1_prev
    labels[int(rmb_bif)] = "RMB_BIF"
    labels[int(lmb_bif)] = "LMB_BIF"

    rmb_children = _children_bifurcations(G, rmb_bif, rmb_bif_prev, nodes)
    if len(rmb_children) >= 2:
        rmb_children.sort(key=lambda x: -x[1])
        a, b = rmb_children[0][0], rmb_children[1][0]
        bi_bif = a if nodes[a, 2] < nodes[b, 2] else b
        labels[int(bi_bif)] = "BI_BIF"
    else:
        qa.append("RMB_underbranched")

    missing = [l for l in EXPECTED_LABELS if l not in labels.values()]
    if missing:
        qa.append("missing:" + ",".join(missing))
    return labels, qa


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    flag_summary = collections.Counter()
    with h5py.File(args.corpus, "a") as h5:
        ids = [s.decode() if isinstance(s, bytes) else str(s)
               for s in h5["metadata/subject_ids"][:]]
        for sid in tqdm(ids, desc="labeling"):
            g = h5[f"subjects/{sid}"]
            nodes = g["centerline_nodes"][:]
            edges = g["centerline_edges"][:]
            labels, qa = _label_subject(nodes, edges)

            idxs = np.array(sorted(labels.keys()), dtype=np.int32)
            names = np.array([labels[int(i)] for i in idxs],
                             dtype=LABEL_DTYPE)
            for k in ("label_node_idx", "label_name", "qa_flags"):
                if k in g:
                    del g[k]
            g.create_dataset("label_node_idx", data=idxs)
            g.create_dataset("label_name", data=names)
            g.create_dataset("qa_flags",
                             data=np.array(qa, dtype=LABEL_DTYPE))
            for f in qa:
                flag_summary[f] += 1

    print(f"\nLabeled {len(ids)} subjects.")
    if flag_summary:
        print("QA flag summary:")
        for f, c in flag_summary.most_common():
            print(f"  {f:40s} {c}")
    else:
        print("No QA flags raised.")


if __name__ == "__main__":
    main()
