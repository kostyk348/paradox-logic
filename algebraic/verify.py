"""
algebraic.verify — LEVEL 2: programs as monoids.

Composition of processes is a product / wreath of automata; correctness is a cohomology
obstruction (a non-trivial holonomy). `verify` answers: is the process network realisable,
how many independent contradictions, and how many groundings pin it. `min_repair` is the
minimum number of clauses to drop (NP-hard in general; brute force for small systems).
"""
from __future__ import annotations
import itertools
from typing import Callable, Sequence

from .core import consistency, holonomy_dim, min_groundings
from .automaton import DFA


def verify(nv: int, edges: Sequence):
    return {"consistent": consistency(nv, edges),
            "contradictions": holonomy_dim(nv, edges),
            "groundings": min_groundings(nv, edges)}


def min_repair(nv: int, edges: Sequence) -> int:
    """Minimum number of constraints to delete to make the system consistent."""
    m = len(edges)
    for k in range(m + 1):
        for comb in itertools.combinations(range(m), k):
            drop = set(comb)
            if consistency(nv, [e for i, e in enumerate(edges) if i not in drop]):
                return k
    return m


def product(A: DFA, B: DFA) -> DFA:
    """Direct product: states (a, b); recognises the intersection (A AND B)."""
    states = [(a, b) for a in sorted(A.reachable()) for b in sorted(B.reachable())]
    idx = {s: i for i, s in enumerate(states)}
    trans = {}
    for (a, b) in states:
        if a in A.trans and b in B.trans:
            trans[idx[(a, b)]] = {al: idx[(A.trans[a][al], B.trans[b][al])]
                                  for al in A.trans[a] if al in B.trans[b]}
    accept = {idx[(a, b)] for (a, b) in states if a in A.accept and b in B.accept}
    return DFA(len(states), trans, accept, idx[(A.start, B.start)]).minimise()


def cascade(A: DFA, B: DFA, couple: Callable) -> DFA:
    """Wreath/cascade: A drives B. B reads couple(A_state, symbol) instead of the symbol."""
    states = [(a, b) for a in sorted(A.reachable()) for b in sorted(B.reachable())]
    idx = {s: i for i, s in enumerate(states)}
    trans = {}
    for (a, b) in states:
        trans[idx[(a, b)]] = {}
        for al in A.trans[a]:
            na = A.trans[a][al]
            bs = couple(a, al)
            nb = B.trans[b].get(bs, b)
            trans[idx[(a, b)]][al] = idx[(na, nb)]
    accept = {idx[(a, b)] for (a, b) in states if b in B.accept}
    return DFA(len(states), trans, accept, idx[(A.start, B.start)]).minimise()
