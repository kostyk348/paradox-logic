"""
17 — Expressiveness = cycle structure (a quantitative Whorf law).

A "language" is a Γ-labelled graph. The fraction of labelings that admit a global
assignment (coherent languages) is governed by the cycle rank:

        P(consistent)  ~  |Γ|^(-cycle_rank),      cycle_rank = E - V + 1

and the number of coherent thoughts of a consistent language is |Γ|^(V - rank) = |Γ|^c,
c = number of connected components.

So a language constrains thought in proportion to its cycles: a tree binds nothing,
each independent cycle multiplies the constraint by |Γ|.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from groupholonomy import symmetric_group, cyclic_group, consistency

Z2 = cyclic_group(2)
S3 = symmetric_group(3)

def graph(V, p, rng):
    E = [(v, int(rng.integers(0, v))) for v in range(1, V)]
    for a in range(V):
        for b in range(a + 1, V):
            if rng.random() < p:
                E.append((a, b))
    return E

def connected(V, E):
    adj = {v: [] for v in range(V)}
    for a, b in E: adj[a].append(b); adj[b].append(a)
    seen = {0}; st = [0]
    while st:
        u = st.pop()
        for w in adj[u]:
            if w not in seen: seen.add(w); st.append(w)
    return len(seen) == V

rng = np.random.default_rng(0)
print("Fraction of coherent labelings vs cycle rank\n")
print(f"{'Γ':>4} {'V':>3} {'cycle-rank':>11} {'measured':>10} {'|Γ|^-rank':>10}")
for Gx, gname in ((Z2, "Z/2"), (S3, "S_3")):
    for V, p in [(8, 0.15), (8, 0.30), (8, 0.50), (10, 0.40)]:
        E = graph(V, p, rng)
        if not connected(V, E):
            continue
        cr = len(E) - V + 1
        n = 20000
        ok = 0
        for _ in range(n):
            edges = [(a, b, Gx.els[int(rng.integers(0, len(Gx.els)))]) for (a, b) in E]
            if consistency(list(range(V)), edges, Gx)[0]: ok += 1
        print(f"{gname:>4} {V:>3} {cr:>11} {ok/n:>10.4f} {len(Gx.els)**(-cr):>10.4f}")

print("\n=> expressiveness law: each independent cycle multiplies the constraint by |Γ|.")
print("   A tree (cycle rank 0) constrains nothing -> every labeling is a coherent language.")
print("   Thought is bound in proportion to the cycle structure of the grammar.")
