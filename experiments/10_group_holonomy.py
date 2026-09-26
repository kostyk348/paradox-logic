"""
10 — Group holonomy: the detection ladder + a useful application.

(1) Consistency is polynomial for any fixed finite group (O(E) group ops).
(2) Non-abelian detection is strictly finer than parity: an EVEN corruption
    (3-cycle) leaves the abelianization untouched, so a Z/2 check sees nothing,
    while the S_3 holonomy finds the paradox.
(3) Useful: localising a single corrupted edge in a group-labelled network
    (this is exactly loop-closure error localisation in pose-graph SLAM, with
     rotations in place of group elements).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from groupholonomy import symmetric_group, sign, consistency, find_bad_edge

S3 = symmetric_group(3)
ID = S3.id
A3 = [e for e in S3.els if sign(e) == 0]          # even permutations (order 3, abelian)
THREE_CYCLE = next(e for e in A3 if e != ID)      # an even corruption: parity-invisible

def random_connected_graph(V, p, rng):
    nodes = list(range(V)); edges = []
    for v in range(1, V):                          # random tree
        edges.append((v, int(rng.integers(0, v))))
    for a in range(V):
        for b in range(a + 1, V):
            if rng.random() < p: edges.append((a, b))
    return edges

def make_consistent(V, gedges, rng):
    """x_a = g_ab x_b with labels derived from random potentials -> always consistent."""
    x = {v: S3.els[int(rng.integers(0, len(S3.els)))] for v in range(V)}
    out = []
    for (a, b) in gedges:
        g = S3.mul(x[a], S3.inv(x[b]))             # x_a = g x_b
        out.append((a, b, g))
    return out

V, p, trials = 9, 0.35, 400
rng = np.random.default_rng(0)
z2_det = s3_det = loc_ok = 0
for _ in range(trials):
    E = random_connected_graph(V, p, rng)
    edges = make_consistent(V, E, rng)
    i = int(rng.integers(0, len(edges)))
    a, b, g = edges[i]
    edges[i] = (a, b, S3.mul(THREE_CYCLE, g))      # corrupt by an EVEN element

    # Z/2 (abelianization) detector
    zedges = [(u, w, (sign(gg))) for (u, w, gg) in edges]
    ok_ab, _, _ = consistency(list(range(V)), zedges, __import__("groupholonomy").cyclic_group(2))
    if not ok_ab: z2_det += 1
    # non-abelian detector
    ok_s3, _, _ = consistency(list(range(V)), edges, S3)
    if not ok_s3: s3_det += 1
    # localisation
    if find_bad_edge(list(range(V)), edges, S3) == i: loc_ok += 1

print("Group holonomy on random S_3-labelled graphs (V=9, one EVEN corruption per trial)\n")
print(f"  trials:                        {trials}")
print(f"  Z/2 (abelianization) detects:  {z2_det}/{trials}  ({100*z2_det/trials:.0f}%)")
print(f"  S_3 (non-abelian) detects:     {s3_det}/{trials}  ({100*s3_det/trials:.0f}%)")
print(f"  corrupted edge localised:      {loc_ok}/{trials}  ({100*loc_ok/trials:.0f}%)")
print("\n=> parity is blind to even corruption; non-abelian holonomy detects it AND localises")
print("   the offending edge -- a polynomial-time consistency check (and an SO(3) SLAM tool).")
