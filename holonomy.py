"""
holonomy — parity (GF(2)) invariants of Boolean constraint networks.

A Boolean constraint network is a graph whose edges carry a parity label:

    edge (u, v, n)   means   x_u XOR x_v = n      (n in {0,1})

This is exactly the linear layer of "paradox logic": a node of type NOT flips the
value along its reference edge (n=1), a node of type PURE copies it (n=0).

Central fact (the whole repository rests on it):

    A global assignment exists   <=>  n is a coboundary   <=>  every cycle has
                                     an EVEN number of 1-labels.

When no global assignment exists, the system is *paradoxical*; the number of
independent paradoxes is  dim coker(A)  where  A x = n  is the constraint system
over GF(2).  We call this invariant the *holonomy dimension* of the network.

Everything here is exact integer linear algebra over GF(2); no floats.
"""
from __future__ import annotations
import itertools
from typing import Iterable, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int, int]          # (u, v, label in {0,1})


def gf2_rank(M: np.ndarray) -> int:
    """Rank of an integer matrix over GF(2)."""
    M = (np.asarray(M) % 2).astype(int).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def _system(nv: int, edges: Sequence[Edge]) -> Tuple[np.ndarray, np.ndarray]:
    """Constraint matrix A (one row per edge) and rhs n, for A x = n over GF(2)."""
    A = np.zeros((len(edges), nv), dtype=int)
    n = np.zeros(len(edges), dtype=int)
    for k, (u, v, lab) in enumerate(edges):
        A[k, u] ^= 1
        A[k, v] ^= 1
        n[k] = lab & 1
    return A, n


def consistency(nv: int, edges: Sequence[Edge]) -> bool:
    """True iff a global {0,1} assignment satisfies every edge."""
    A, n = _system(nv, edges)
    aug = np.concatenate([A, n[:, None]], axis=1)
    return gf2_rank(A) == gf2_rank(aug)


def holonomy_dim(nv: int, edges: Sequence[Edge]) -> int:
    """Number of independent paradoxes = dim coker(A) = (#contradictions)."""
    A, n = _system(nv, edges)
    aug = np.concatenate([A, n[:, None]], axis=1)
    return gf2_rank(aug) - gf2_rank(A)


def min_groundings(nv: int, edges: Sequence[Edge]) -> int:
    """Acts needed to pin a solution = nv - rank(A) (gauge dimension)."""
    A, _ = _system(nv, edges)
    return nv - gf2_rank(A)


def max_satisfiable(nv: int, edges: Sequence[Edge]) -> int:
    """Max number of edges satisfiable by any *single* global assignment (brute force)."""
    best = 0
    for a in itertools.product((0, 1), repeat=nv):
        s = sum(1 for (u, v, lab) in edges if (a[u] ^ a[v]) == (lab & 1))
        best = max(best, s)
    return best


def contextual_fraction(nv: int, edges: Sequence[Edge]) -> float:
    """1 - (max satisfiable / total). > 0 <=> the network is contextual (Mermin-like)."""
    if not edges:
        return 0.0
    return 1.0 - max_satisfiable(nv, edges) / len(edges)


def cycle_edges(L: int) -> list[Edge]:
    return [(i, (i + 1) % L, 0) for i in range(L)]


def label_cycle(edges: Sequence[Edge], labels: Sequence[int]) -> list[Edge]:
    return [(u, v, labels[i]) for i, (u, v, _) in enumerate(edges)]


def cycle_holonomy(labels: Sequence[int]) -> int:
    """Parity of 1-labels around a cycle (0 = consistent, 1 = paradox)."""
    return sum(l & 1 for l in labels) & 1
