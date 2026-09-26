"""
04 — Novelty = change of the topological class.   CORE: YES.  As criticality signal: NO.

Random VALUE noise cannot change the holonomy class (it is structural), so it produces
zero novelty. Structural edits do. Therefore "novelty != noise" is provable and exact.
But the novelty rate is Poisson in the edit rate -> it is not a criticality signal.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from holonomy import holonomy_dim

def edges_of(n, ref, lab): return [(i, ref[i], lab[i]) for i in range(n)]

rng = np.random.default_rng(7)
n = 16
ref = list(rng.integers(0, n, n)); lab = list(rng.integers(0, 2, n))

# value noise only: flip values many times, class must not move
v = list(rng.integers(0, 2, n)); c0 = holonomy_dim(n, edges_of(n, ref, lab)); changes = 0
for _ in range(20000):
    v = [v[ref[i]] ^ (lab[i] & 1) for i in range(n)]
    v[int(rng.integers(0, n))] ^= 1                          # inject value noise
    changes += holonomy_dim(n, edges_of(n, ref, lab)) != c0
print(f"value noise:   class changes = {changes}  (0 => novelty is orthogonal to randomness)")

# structure noise: flip labels / reroute, count class changes
for mu in (0.002, 0.01, 0.05):
    ref2, lab2 = ref[:], lab[:]; prev = holonomy_dim(n, edges_of(n, ref2, lab2)); ev = 0
    for t in range(4000):
        if rng.random() < mu * n:
            j = int(rng.integers(0, n))
            if rng.random() < 0.5: ref2[j] = int(rng.integers(0, n))
            else: lab2[j] ^= 1
        cur = holonomy_dim(n, edges_of(n, ref2, lab2))
        if cur != prev: ev += 1
        prev = cur
    print(f"structure mu={mu:<5}: class changes = {ev:<5} (novelty > 0, scales ~ linearly)")
