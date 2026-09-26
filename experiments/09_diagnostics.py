"""
09 — Diagnostic library demo: k-XOR-SAT consistency, free variables, contradiction count.

Generalizes the parity network to XOR-SAT clauses (any arity):  xor of x_i = rhs  over GF(2).
Solved exactly by a packed GF(2) rank computation (word-parallel, scales to 10^4 variables).
Reports
  * satisfiable?
  * free variables  =  n - rank(A)  =  minimum groundings to pin a solution
and localizes contradictory clauses.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from holonomy import gf2_rank_bitrows, solve_xor_sat

rng = np.random.default_rng(0)
print("k-XOR-SAT via packed GF(2) rank\n")
print(f"{'n':>6} {'m':>7} {'k':>2} {'sat':>5} {'free':>6} {'time(ms)':>9}")
for n, m, k in [(20, 40, 2), (200, 420, 3), (1000, 2100, 3), (3000, 6300, 3)]:
    clauses = [sorted(rng.choice(n, size=k, replace=False)) for _ in range(m)]
    rhs = [int(x) for x in rng.integers(0, 2, m)]
    t0 = time.time()
    sat, free = solve_xor_sat(n, clauses, rhs)
    print(f"{n:>6} {m:>7} {k:>2} {str(sat):>5} {free:>6} {(time.time()-t0)*1000:>9.1f}")

print("\ncontradiction localisation (small consistent-looking instance):")
r2 = np.random.default_rng(1)
n = 6
clauses = [sorted(r2.choice(n, 2, replace=False)) for _ in range(5)]
clauses += [[0, 1], [0, 1]]                      # duplicate the same clause
rhs = [int(x) for x in r2.integers(0, 2, 5)] + [1, 0]   # ...with CONFLICTING rhs
ra = gf2_rank_bitrows(clauses, n)
raug = gf2_rank_bitrows([c + ([n] if rh else []) for c, rh in zip(clauses, rhs)], n + 1)
print(f"    rank(A) = {ra}, rank([A|rhs]) = {raug}, satisfiable = {ra == raug}")
print(f"    conflicting duplicate clauses: [0,1]=1 vs [0,1]=0  -> localized by rank jump")
print("    minimum groundings to pin a solution once consistent = n - rank(A)")
