"""
algebraic.core — exact GF(2) consistency of Boolean constraint networks.

A network is a list of edges (u, v, n) meaning x_u XOR x_v = n over GF(2).
Everything here is exact integer linear algebra; no floats.

This is LEVEL 2's primitive: consistency (is the system realisable?), the holonomy
dimension (how many independent contradictions), and the number of groundings needed.
"""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

Edge = Tuple[int, int, int]


def gf2_rank(M: np.ndarray) -> int:
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


def _system(nv: int, edges: Sequence[Edge]):
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
    return gf2_rank(A) == gf2_rank(np.concatenate([A, n[:, None]], axis=1))


def holonomy_dim(nv: int, edges: Sequence[Edge]) -> int:
    """Number of independent contradictions = dim coker(A)."""
    A, n = _system(nv, edges)
    return gf2_rank(np.concatenate([A, n[:, None]], axis=1)) - gf2_rank(A)


def min_groundings(nv: int, edges: Sequence[Edge]) -> int:
    """Groundings (acts) needed to pin a solution = nv - rank(A)."""
    A, _ = _system(nv, edges)
    return nv - gf2_rank(A)


def xor_sat(nvars: int, clauses: Sequence[Sequence[int]], rhs: Sequence[int]):
    """Exact k-XOR-SAT: returns (satisfiable, free_variables)."""
    A = np.zeros((len(clauses), nvars), dtype=int)
    for r, vs in enumerate(clauses):
        for v in vs:
            A[r, v] ^= 1
    b = np.array(rhs, dtype=int)
    return gf2_rank(A) == gf2_rank(np.concatenate([A, b[:, None]], axis=1)), nvars - gf2_rank(A)
