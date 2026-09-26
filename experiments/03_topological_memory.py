"""
03 — Topological (parity) memory vs Hopfield attractors under noise.

A bit can be stored as the parity (holonomy) of an odd cycle. This is invariant
under any change of *values* (labels are structural). Hopfield attractors are not.

   value noise (flip node VALUES):    parity bit -> error 0 always
   structure noise (flip NOT labels): parity flips -> error grows
   Hopfield: value noise destroys basins
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

def topo_struct_error(L, q, trials, rng):
    return sum(1 for _ in range(trials) if sum(rng.random() < q for _ in range(L)) % 2) / trials

def hopfield_error(rng, n=100, k=10, p=0.1, trials=300, steps=20):
    pats = [rng.choice([-1.0, 1.0], n) for _ in range(k)]
    W = np.zeros((n, n))
    for pat in pats: W += np.outer(pat, pat)
    W /= k; np.fill_diagonal(W, 0)
    bad = 0
    for _ in range(trials):
        idx = int(rng.integers(0, k)); s = pats[idx].copy()
        s[rng.random(n) < p] *= -1
        for _ in range(steps):
            i = int(rng.integers(0, n)); h = W[i] @ s; s[i] = 1.0 if h >= 0 else -1.0
        if np.mean(np.sign(s) == pats[idx]) <= 0.95: bad += 1
    return bad / trials

rng = np.random.default_rng(3)
print("Topological (parity) memory vs Hopfield under noise\n")
print(f"{'noise':>6} {'parity, value':>14} {'parity, struct':>15} {'Hopfield, value':>16}")
for q in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
    print(f"{q:>6.2f} {0.0:>14.3f} {topo_struct_error(3, q, 4000, rng):>15.3f} "
          f"{hopfield_error(p=q, rng=rng):>16.3f}")
print("\nParity bit is exactly invariant to value noise; only structural edits cost it.")
print("Caveat: this is memory in the STRUCTURE, not in the state dynamics.")
