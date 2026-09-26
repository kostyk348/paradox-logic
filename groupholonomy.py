"""
groupholonomy — holonomy of group-labelled graphs.

A labelled graph assigns to each edge (a, b) a group element g meaning

        x_a = g * x_b                (x_v in a group Γ, non-abelian allowed)

A global assignment x exists  <=>  every cycle has trivial holonomy
<=> the labelling defines a TRIVIAL flat Γ-bundle over the graph.
The obstruction is the holonomy representation  rho : pi_1(G) -> Γ.

When Γ = Z/2 this reduces to the parity invariant of holonomy.py.
For non-abelian Γ the invariant is strictly finer: a labelling can have trivial
abelianization and still carry a non-trivial holonomy (e.g. a commutator).

All algorithms are O(E) group multiplications given a multiplication table, so
CONSISTENCY IS POLYNOMIAL for any fixed finite group Γ.
"""
from __future__ import annotations
from itertools import permutations
from collections import deque
from typing import Callable, Hashable, Sequence, Tuple


class Group:
    """A finite group given by its elements and operations."""
    def __init__(self, elements, mul, inv, ident):
        self.els = list(elements)
        self._mul: Callable = mul
        self._inv: Callable = inv
        self.id = ident
        self.index = {e: i for i, e in enumerate(self.els)}

    def mul(self, a, b): return self._mul(a, b)
    def inv(self, a): return self._inv(a)


def symmetric_group(n: int) -> Group:
    els = [tuple(p) for p in permutations(range(n))]
    def mul(a, b): return tuple(a[b[i]] for i in range(n))
    def inv(a):
        r = [0] * n
        for i, v in enumerate(a):
            r[v] = i
        return tuple(r)
    return Group(els, mul, inv, tuple(range(n)))


def sign(a) -> int:
    """Abelianization Z -> Z/2 (even = 0, odd = 1)."""
    invs = sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j])
    return invs % 2


def cyclic_group(n: int) -> Group:
    els = list(range(n))
    return Group(els, lambda a, b: (a + b) % n, lambda a: (-a) % n, 0)


# ---------------------------------------------------------------------------
def consistency(V, edges, G: Group):
    """Is the labelling globally realizable? Returns (ok, defects, potentials).

    edges: sequence of (a, b, g) meaning x_a = g * x_b.
    """
    adj = {v: [] for v in V}
    for (a, b, g) in edges:
        adj[a].append((b, G.inv(g)))
        adj[b].append((a, g))
    root = V[0]
    pot = {root: G.id}
    dq = deque([root])
    defects = set()
    while dq:
        u = dq.popleft()
        for (w, f) in adj[u]:
            val = G.mul(f, pot[u])
            if w not in pot:
                pot[w] = val
                dq.append(w)
            elif val != pot[w]:
                defects.add(frozenset((u, w)))
    return (len(defects) == 0), defects, pot


def find_bad_edge(V, edges, G: Group):
    """If exactly one edge is corrupted, return its index (O(E^2) group ops)."""
    for i in range(len(edges)):
        ok, _, _ = consistency(V, edges[:i] + edges[i + 1:], G)
        if ok:
            return i
    return None
