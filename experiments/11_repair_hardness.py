"""
11 — The complexity ladder: consistency is easy, optimal repair is hard.

For a signed graph (Z/2 labels; n=1 <=> antiferromagnetic bond):
  * CONSISTENCY (is there a global assignment at all?):  polynomial (GF(2) rank).
  * MINIMUM REPAIR (delete the fewest labels to make it consistent):
        = the frustration index  =  m - MAX-CUT,
    already NP-hard for Z/2. So "how many paradoxes are there" is easy, but
    "which labels must change" is the hard combinatorial problem.

We brute-force the frustration index on small graphs and show it is a non-trivial
fraction of the edges, while consistency stays instant.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import numpy as np
from holonomy import consistency, gf2_rank_bitrows

def max_satisfied(V, edges):
    best = 0
    for a in itertools.product((0, 1), repeat=V):
        s = sum(1 for (u, v, n) in edges if (a[u] ^ a[v]) == (n & 1))
        best = max(best, s)
    return best

rng = np.random.default_rng(0)
print("Frustration index of random signed graphs (Z/2)\n")
print(f"{'V':>3} {'m':>4} {'consistent':>11} {'max-sat':>8} {'frustration':>12} {'brute(ms)':>10}")
for V in (8, 10, 12, 14):
    edges = []
    for v in range(1, V):
        edges.append((v, int(rng.integers(0, v)), int(rng.integers(0, 2))))
    for a in range(V):
        for b in range(a + 1, V):
            if rng.random() < 0.4:
                edges.append((a, b, int(rng.integers(0, 2))))
    m = len(edges)
    t0 = time.time()
    ok = consistency(V, edges)
    t1 = time.time()
    best = max_satisfied(V, edges)
    t2 = time.time()
    print(f"{V:>3} {m:>4} {str(ok):>11} {best:>8} {m-best:>12} {(t2-t1)*1000:>10.1f}")

print("\nconsistency = one GF(2) rank (microseconds); frustration index = brute force 2^V.")
print("Optimal repair is NP-hard already at the abelian (Z/2) level: it is MAX-CUT.")
print("=> 'count the paradoxes' is polynomial; 'fix them minimally' is not.")
