"""Generate figures/contextuality_memory.png (two panels)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from holonomy import cycle_edges, label_cycle, contextual_fraction, max_satisfiable

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# panel 1: contextual fraction vs cycle length for the "all-NOT" and "all-but-one-NOT" labelings
Ls = list(range(3, 13))
allnot = [contextual_fraction(L, label_cycle(cycle_edges(L), [1] * L)) for L in Ls]
oneless = [contextual_fraction(L, label_cycle(cycle_edges(L), [1] * (L - 1) + [0])) for L in Ls]
ax[0].plot(Ls, allnot, "o-", label="#NOT = L (odd for odd L)")
ax[0].plot(Ls, oneless, "s-", label="#NOT = L-1")
ax[0].set_xlabel("cycle length L")
ax[0].set_ylabel("contextual fraction = 1 - max-sat/L")
ax[0].set_title("Contextuality of parity cycles")
ax[0].grid(alpha=0.3); ax[0].legend()

# panel 2: memory error curves
rng = np.random.default_rng(3)
qs = np.linspace(0, 0.4, 9)
struct = [sum(1 for _ in range(4000) if sum(rng.random() < q for _ in range(3)) % 2) / 4000 for q in qs]
def hop(q):
    n, k, bad = 100, 10, 0
    pats = [rng.choice([-1.0, 1.0], n) for _ in range(k)]
    W = np.zeros((n, n))
    for p in pats: W += np.outer(p, p)
    W /= k; np.fill_diagonal(W, 0)
    for _ in range(300):
        i0 = int(rng.integers(0, k)); s = pats[i0].copy(); s[rng.random(n) < q] *= -1
        for _ in range(20):
            i = int(rng.integers(0, n)); s[i] = 1.0 if W[i] @ s >= 0 else -1.0
        if np.mean(np.sign(s) == pats[i0]) <= 0.95: bad += 1
    return bad / 300
hopf = [hop(q) for q in qs]
ax[1].plot(qs, [0]*len(qs), "o-", label="parity bit, VALUE noise")
ax[1].plot(qs, struct, "s-", label="parity bit, STRUCTURE noise")
ax[1].plot(qs, hopf, "^-", label="Hopfield, VALUE noise")
ax[1].set_xlabel("noise level")
ax[1].set_ylabel("bit error")
ax[1].set_title("Topological memory vs Hopfield")
ax[1].grid(alpha=0.3); ax[1].legend()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "contextuality_memory.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout(); plt.savefig(out, dpi=140)
print("wrote", out)
