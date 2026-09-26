"""
05 — Cohomological diagonal lemma.   VERDICT: REFUTED (in its simple form).

Hypothesis: a network whose labels depend on its own holonomy class ("self-gauge")
must be paradoxical.  Test exhaustively: labels n'[i] = c[i] XOR h(n), where h(n) is
the holonomy class of the label vector n. Fixed points of n -> n' are enumerated.
Finding: about half of all base vectors c admit a fixed point with TRIVIAL class.
Self-reference does NOT force paradox.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from holonomy import holonomy_dim, cycle_edges, label_cycle

def selfgauge_fixed_points(L, c):
    base = cycle_edges(L)
    out = []
    for bits in itertools.product((0, 1), repeat=L):
        h = 1 if holonomy_dim(L, label_cycle(base, bits)) > 0 else 0
        if tuple(c[i] ^ h for i in range(L)) == bits:
            out.append((bits, h))
    return out

print("Self-gauge fixed points  (n'[i] = c[i] XOR holonomy(n))\n")
total = with_trivial = 0
for L in range(3, 7):
    triv = 0
    for c in itertools.product((0, 1), repeat=L):
        total += 1
        if any(h == 0 for _, h in selfgauge_fixed_points(L, c)):
            triv += 1; with_trivial += 1
    print(f"  L={L}: base vectors c with a trivial-class fixed point: {triv}/{2**L}")
print(f"\n  overall: {with_trivial}/{total}")
print("  => trivial (consistent) self-referential states exist: the lemma is FALSE in this form.")
