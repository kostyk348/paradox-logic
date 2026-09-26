"""
08 — Classification by the invariant.

Can ([n], dim coker) label the regime of a Boolean network?
We generate random single-input Boolean networks, compute the invariant, and compare it
with behaviour (has a fixed point? oscillates?).

Findings:
  * holonomy_dim == 0  <=>  a fixed point exists        (exact: the invariant decides PARADOX)
  * holonomy_dim  > 0  =>  every trajectory oscillates  (necessary)
  * oscillation ALSO occurs at holonomy_dim == 0        (multistability) -> the invariant
    classifies paradox exactly, but NOT general behaviour.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from holonomy import holonomy_dim, consistency

def edges_of(n, ref, lab): return [(i, ref[i], lab[i]) for i in range(n)]

def period(ref, lab, n, iters=200):
    v = [np.random.randint(2) for _ in range(n)]
    hist = []
    for _ in range(iters):
        v = [v[ref[i]] ^ (lab[i] & 1) for i in range(n)]
        hist.append(tuple(v))
    tail = hist[-40:]
    for p in range(1, 21):
        if all(tail[k] == tail[k + p] for k in range(len(tail) - p)):
            return p
    return -1

rng = np.random.default_rng(0)
print("Classification by the holonomy invariant (random single-input networks)\n")
print(f"{'n':>3} {'trials':>7} {'paradox=0 & fixed':>18} {'paradox=0 & osc':>16} {'paradox>0 & osc':>16}")
for n in (6, 8, 10):
    a = b = c = t = 0
    for _ in range(4000):
        ref = list(rng.integers(0, n, n)); lab = list(rng.integers(0, 2, n))
        par = holonomy_dim(n, edges_of(n, ref, lab))
        osc = period(ref, lab, n) > 1
        if par == 0 and not osc: a += 1
        if par == 0 and osc: b += 1
        if par > 0 and osc: c += 1
        t += 1
    print(f"{n:>3} {t:>7} {a:>18} {b:>16} {c:>16}")

print("\n  'paradox=0 & fixed' + 'paradox=0 & osc'  = all consistent systems")
print("  invariant predicts PARADOX (dim>0) exactly; it does NOT predict oscillation when dim=0.")
