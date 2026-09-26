"""
18 — Meaning = gauge invariance.

Constraint  x_a = g_e x_b.  A gauge transformation relabels thoughts: x_v -> c x_v.
Under it the labels transform by CONJUGATION  g_e -> c^{-1} g_e c, and the whole
coherent-thought set maps bijectively onto itself.  Hence:

  * the number of coherent thoughts is gauge-invariant;
  * the holonomy CONJUGACY CLASS is gauge-invariant;
  * any readout that depends only on conjugacy classes is a well-defined meaning,
    while a readout of the raw labels is not.

This is a working definition of meaning as invariance (not reference): the invariant
content of a language is exactly its gauge orbit.
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from groupholonomy import symmetric_group, consistency

S3 = symmetric_group(3)
G = S3.els
ID = S3.id

def prod(seq):
    h = ID
    for g in seq: h = S3.mul(h, g)
    return h

def conj_class(e):
    return frozenset(S3.mul(S3.mul(g, e), S3.inv(g)) for g in G)

def count_solutions(L, labels):
    return sum(1 for x in itertools.product(G, repeat=L)
               if all(S3.mul(labels[i], x[(i + 1) % L]) == x[i] for i in range(L)))

rng = np.random.default_rng(0)
L = 4
n_sol_inv = class_inv = h_eq = 0
trials = 2000
for _ in range(trials):
    labels = [G[int(rng.integers(0, len(G)))] for _ in range(L)]
    c = G[int(rng.integers(0, len(G)))]
    g2 = [S3.mul(S3.mul(S3.inv(c), g), c) for g in labels]      # g -> c^{-1} g c

    if count_solutions(L, labels) == count_solutions(L, g2): n_sol_inv += 1
    if conj_class(prod(labels)) == conj_class(prod(g2)): class_inv += 1
    if prod(g2) == S3.mul(S3.mul(S3.inv(c), prod(labels)), c): h_eq += 1

print("Gauge transformation  x_v -> c x_v  (labels conjugated)\n")
print(f"  trials: {trials}")
print(f"  #coherent-thoughts preserved:     {n_sol_inv}/{trials}")
print(f"  holonomy conjugacy class preserved: {class_inv}/{trials}")
print(f"  holonomy transforms exactly by conjugation: {h_eq}/{trials}")
print("\n=> the invariant content (conjugacy class of holonomy) is gauge-invariant;")
print("   the raw labels are not.  Meaning = the gauge orbit.  Translation = gauge.")
