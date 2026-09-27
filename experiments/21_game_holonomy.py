"""
21 — Game theory wrap: paradox = no pure Nash equilibrium.

Everything built in this repo is the theory of *equilibrium existence* in games whose
constraints are parity / preference cycles:

    language            -> a game (players = nodes, edges = constraints)
    thought (section)   -> a PURE Nash equilibrium
    paradox             -> NO pure equilibrium
    oscillation         -> best-response dynamics that never settle
    the "act"           -> a MIXED strategy (randomization)
    gauge freedom       -> multiplicity of equilibria

Two canonical verifications:
  (1) matching pennies  = the Z/2 parity game: no pure NE -> best-response cycle.
  (2) Condorcet paradox  = a frustrated 3-cycle of majority preferences = non-trivial holonomy.
"""
import itertools
import numpy as np
from collections import Counter

# ---------------------------------------------------------------- (1) matching pennies
# A wants to MATCH, B wants to MISMATCH -> no pure NE, best-response dynamics cycle.
def bp_cycle(steps=10):
    a, b = 0, 0
    hist = [(a, b)]
    for _ in range(steps):
        b = 1 - a                    # B best-responds to a
        a = b                        # A best-responds to b
        hist.append((a, b))
    return hist

h = bp_cycle(8)
period = next(p for p in range(1, 9) if all(h[i] == h[i + p] for i in range(len(h) - p)))
print("(1) matching pennies (A matches, B mismatches)")
print(f"    trajectory: {h}")
print(f"    no pure Nash equilibrium; best-response period = {period} (oscillation)\n")

# ---------------------------------------------------------------- (2) Condorcet
def condorcet_cycle(m, n_voters, rng):
    """majority tournament on m alternatives; True if it has a directed cycle."""
    # random strict rankings
    prefs = [rng.permutation(m) for _ in range(n_voters)]
    wins = np.zeros((m, m), int)
    for p in prefs:
        pos = {alt: i for i, alt in enumerate(p)}
        for x in range(m):
            for y in range(m):
                if x != y and pos[x] < pos[y]:
                    wins[x, y] += 1
    # majority relation beats[x][y] = True if majority prefers x to y
    beats = wins > (n_voters / 2)
    # detect a directed cycle (DFS)
    adj = [[y for y in range(m) if beats[x, y]] for x in range(m)]
    color = [0] * m
    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1: return True
            if color[v] == 0 and dfs(v): return True
        color[u] = 2
        return False
    return any(color[u] == 0 and dfs(u) for u in range(m))

rng = np.random.default_rng(0)
print("(2) Condorcet paradox: fraction of profiles with NO Condorcet winner")
print(f"{'alternatives':>12} {'voters':>7} {'P(cycle)':>9}")
for m in (3, 4, 5, 6):
    for nv in (3, 5, 11):
        p = np.mean([condorcet_cycle(m, nv, rng) for _ in range(3000)])
        print(f"{m:>12} {nv:>7} {p:>9.3f}")

print("\n=> a Condorcet cycle is a frustrated cycle of preferences = non-trivial holonomy.")
print("   paradox (no global section)  ==  no pure Nash equilibrium.")
print("   the 'act' (ground a node)    ==  a mixed strategy / tie-break: it only picks a")
print("   representative, which is why a parity act and a random act were indistinguishable.")
