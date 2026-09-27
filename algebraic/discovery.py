"""
algebraic.discovery — learn the ALGEBRA from data, no seed (RPNI).

Gradient descent on automata hits a wall (random init fails for Z>=5). Structure can instead
be DISCOVERED symbolically: RPNI (regular positive/negative inference) merges the states of a
prefix tree until the automaton is consistent with the labelled examples. Closes the open
problem "search the algebra" with no prior.
"""
from __future__ import annotations
import copy
from typing import Iterable, Sequence


def _build_pta(P):
    trans, accept = {}, set()
    def new():
        i = len(trans); trans[i] = {}; return i
    root = new()
    for w in P:
        cur = root
        for a in w:
            if a not in trans[cur]:
                trans[cur][a] = new()
            cur = trans[cur][a]
        accept.add(cur)
    return trans, accept, root


def _accepts(trans, accept, root, w):
    s = root
    for a in w:
        if a not in trans.get(s, {}):
            return False
        s = trans[s][a]
    return s in accept


def _consistent(trans, accept, root, P, N):
    return all(_accepts(trans, accept, root, w) for w in P) and \
           not any(_accepts(trans, accept, root, w) for w in N)


def _fold(trans, accept, b, r):
    """merge state b into state r (recursively fold their successors)."""
    if b == r or b not in trans:
        return
    for a, tb in list(trans.get(b, {}).items()):
        if a in trans.get(r, {}):
            _fold(trans, accept, tb, trans[r][a])
        else:
            trans.setdefault(r, {})[a] = tb
    if b in accept:
        accept.add(r)
    trans.pop(b, None)
    for s in list(trans):
        for a in list(trans[s]):
            if trans[s][a] == b:
                trans[s][a] = r


def rpni(P: Iterable[Sequence], N: Iterable[Sequence]):
    P, N = [tuple(w) for w in P], [tuple(w) for w in N]
    trans, accept, root = _build_pta(P)
    red = [root]
    while True:
        reds = set(red)
        blue = sorted({t for s in red for t in trans.get(s, {}).values() if t not in reds})
        if not blue:
            break
        b = blue[0]; merged = False
        for r in list(red):
            if r not in trans:
                continue
            t2, a2 = copy.deepcopy(trans), set(accept)
            _fold(t2, a2, b, r)
            if _consistent(t2, a2, root, P, N):
                trans, accept = t2, a2; merged = True
                break
        if not merged:
            red.append(b)
    states = sorted(trans)
    idx = {s: i for i, s in enumerate(states)}
    tr = {idx[s]: {a: idx[t] for a, t in trans[s].items()} for s in states}
    acc = {idx[s] for s in accept if s in trans}
    return len(states), tr, acc, idx[root]


def classify(nstates, trans):
    if not trans:
        return "empty"
    syms = set().union(*[set(t) for t in trans.values()])
    for a in syms:
        img = {t.get(a) for t in trans.values()}
        if None in img or len(img) != len(trans):
            return "aperiodic"
    return "group"
