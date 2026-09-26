"""
A — Reduction to Hopfield: the exact barrier.  (companion theory)

(1) A Boolean map folds into ONE symmetric Hopfield layer iff every coordinate is a
    linear threshold function (LTF). The LP classifier is validated against the known
    counts of Boolean threshold functions (OEIS A000609: 4, 14, 104 for n = 1, 2, 3).
    All monotone gates and implication are LTFs; XOR / XNOR are not -> the barrier is
    exactly parity.
(2) Goles-Olivos: a SYMMETRIC threshold network (synchronous) has limit cycles of
    length <= 2. Asymmetric coupling escapes it (directed threshold = RNN class).
"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
from scipy.optimize import linprog

def inputs(k): return list(itertools.product((0, 1), repeat=k))

def is_ltf(table, k):
    A_ub, b_ub = [], []
    for idx, v in enumerate(inputs(k)):
        y = 1 if table[idx] else -1
        A_ub.append([y * vi for vi in v] + [-y]); b_ub.append(-1)
    return linprog([0.0] * (k + 1), A_ub=A_ub, b_ub=b_ub,
                   bounds=[(None, None)] * (k + 1), method="highs").status == 0

def gate(k, name):
    t = []
    for v in inputs(k):
        s = sum(v)
        t.append({"AND": int(s == k), "OR": int(s >= 1), "NAND": int(s != k),
                  "MAJ": int(s >= (k + 1) // 2), "XOR": int(s % 2),
                  "XNOR": int(s % 2 == 0)}[name])
    return t

print("(1) LP threshold-function classifier validation (OEIS A000609)")
for k in (1, 2, 3):
    cnt = sum(is_ltf([(b >> i) & 1 for i in range(1 << k)], k) for b in range(1 << (1 << k)))
    print(f"    n={k}: {cnt}   (expected {[4, 14, 104][k-1]})")

print("\n(1) which gates fold into one Hopfield layer?")
for name in ("AND", "OR", "NAND", "MAJ", "XOR", "XNOR"):
    k = 3 if name in ("NAND", "MAJ") else 2
    ok = is_ltf(gate(k, name), k)
    print(f"    {name:5s} (k={k}): {'LTF -> Hopfield' if ok else 'NOT LTF -> escapes (parity barrier)'}")

def max_cycle(n, W, b):
    seen, best = set(), 0
    for x0 in range(1 << n):
        if x0 in seen: continue
        path, x, t = {}, x0, 0
        while x not in path and x not in seen:
            path[x] = t
            v = np.array([(x >> i) & 1 for i in range(n)])
            x = int(np.sum((W @ v + b > 0) * (1 << np.arange(n)))); t += 1
        if x in path: best = max(best, t - path[x])
        seen |= set(path)
    return best

print("\n(2) Goles-Olivos: max limit-cycle length (n=3, 4000 random networks each)")
rng = np.random.default_rng(0)
for sym in (True, False):
    mx = 0
    for _ in range(4000):
        W = rng.integers(-1, 2, (3, 3)).astype(float)
        if sym: W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        b = rng.integers(-1, 2, 3).astype(float)
        mx = max(mx, max_cycle(3, W, b))
    print(f"    {'SYMMETRIC (Hopfield)' if sym else 'ASYMMETRIC (RNN)':22s} max_cycle = {mx}")
print("\n    => period <= 2 is a theorem for symmetric coupling; direction breaks it.")
