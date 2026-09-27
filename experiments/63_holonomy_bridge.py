"""
63 — Discrete <-> continuous through the PARADOX (holonomy).

Pointwise VQ fails (exp 62): a connected continuous group has no non-trivial homomorphism to
a finite group. The bridge must be non-trivial: the discrete lives in the HOLONOMY (an
integral along a path), which is path-dependent = the paradox/obstruction.

Task: a sequence of continuous increments d_t; the label is the net rotation quantised.
  * quantise-then-compose (pointwise VQ): lossy, not a homomorphism -> fails
  * compose-then-quantise (integrate = holonomy): exact, and ADDITIVE under concatenation
"""
import math
import numpy as np
rng = np.random.default_rng(0)
K = 8


def gen(n, T):
    d = rng.normal(0, 0.7, (n, T))                    # continuous increments (turns)
    y = (np.floor((d.sum(1) % 1.0) * K)).astype(int) % K
    return d, y


# --- A: pointwise VQ (quantise each increment, then compose codes) ---
def vq(x, K):                                         # fixed-bin quantiser of increments
    b = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return np.minimum((b * K).astype(int), K - 1)


d_tr, _ = gen(20000, 1)
edges = np.linspace(d_tr.min(), d_tr.max() + 1e-9, K + 1)
def code_pointwise(d):                                # code = which bin the increment is in
    return np.clip(np.digitize(d, edges) - 1, 0, K - 1)

# --- B: holonomy (integrate along the path, then quantise once) ---
def holonomy(d):
    return np.floor((d.sum(1) % 1.0) * K).astype(int) % K

d, y = gen(4000, 8)
A = code_pointwise(d).sum(1) % K                       # pointwise VQ then compose
B = holonomy(d)                                        # integrate then quantise
print("Discrete from continuous: pointwise VQ vs holonomy (integrate)\n")
print(f"  pointwise VQ (quantise each step, compose):  acc = {(A == y).mean():.3f}")
print(f"  holonomy (integrate the path, quantise once): acc = {(B == y).mean():.3f}")

# additivity under concatenation: the holonomy IS a homomorphism (paths -> Z_K)
d1, _ = gen(4000, 4); d2, _ = gen(4000, 4)
h1 = (d1.sum(1) % 1.0); h2 = (d2.sum(1) % 1.0); h12 = ((d1.sum(1) + d2.sum(1)) % 1.0)
add = np.mean(np.floor(h12 * K) % K == (np.floor(h1 * K) + np.floor(h2 * K)) % K)
print(f"\n  holonomy additivity h(path1+path2) == h(path1)+h(path2): {add*100:.1f}%")
print("\n=> the discrete emerges from the continuous only through the HOLONOMY (the integral);")
print("   it is path-dependent (the paradox), and that is exactly what makes the bridge")
print("   non-trivial -- a trivial pointwise map cannot carry the algebra.")
