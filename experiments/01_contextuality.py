"""
01 — Paradox is (Mermin/GHZ) contextuality.

Parity constraints on a cycle: a global assignment exists iff #1-labels is even.
When it does not, the network is contextual and max_satisfiable < total.
Shows: the GF(2) holonomy invariant coincides with the contextuality criterion.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holonomy import cycle_edges, label_cycle, consistency, holonomy_dim, max_satisfiable, contextual_fraction

print("Parity constraints on a cycle  (non-contextual <=> satisfiable by a global assignment)\n")
print(f"{'L':>2} {'#NOT':>5} {'max-sat':>9} {'holonomy':>9} {'contextual':>11}")
for L in range(3, 7):
    for k in range(L + 1):
        labels = [1] * k + [0] * (L - k)
        edges = label_cycle(cycle_edges(L), labels)
        sat = max_satisfiable(L, edges)
        print(f"{L:>2} {k:>5} {sat:>4}/{L:<4} {holonomy_dim(L, edges):>9} "
              f"{'YES' if sat < L else 'no':>11}")

print("\nMermin triangle  a^b=1, b^c=1, c^a=1")
edges = label_cycle(cycle_edges(3), [1, 1, 1])
print(f"  consistent={consistency(3, edges)}  holonomy_dim={holonomy_dim(3, edges)}  "
      f"max-sat={max_satisfiable(3, edges)}/3  contextual_fraction={contextual_fraction(3, edges):.3f}")
print("  => this is the canonical parity proof of contextuality; our H^1 invariant matches it.")
