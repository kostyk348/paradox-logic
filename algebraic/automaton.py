"""
algebraic.automaton — finite automata, Angluin's L*, Moore minimisation, constrained decode.

LEVEL 1 primitive (learn/find the algebra) and LEVEL 2 primitive (exact state machine).
Pure Python; no torch required.
"""
from __future__ import annotations
import itertools
from typing import Callable, Dict, Sequence, Tuple


class DFA:
    """Deterministic finite automaton: states 0..n-1, start 0, accept set."""
    def __init__(self, n, trans, accept, start=0):
        self.n = n; self.trans = trans; self.accept = set(accept); self.start = start

    def run(self, s):
        r = self.start
        for a in s:
            r = self.trans[r][a]
        return r in self.accept

    def reachable(self):
        seen = {self.start}; stack = [self.start]
        while stack:
            u = stack.pop()
            for a in self.trans[u]:
                v = self.trans[u][a]
                if v not in seen:
                    seen.add(v); stack.append(v)
        return seen

    def minimise(self):
        """Moore partition refinement over the reachable part."""
        reach = sorted(self.reachable())
        part = {s: (1 if s in self.accept else 0) for s in reach}
        changed = True
        while changed:
            changed = False
            sig = {}
            for s in reach:
                key = (1 if s in self.accept else 0, tuple(part[self.trans[s][a]] for a in self.trans[s]))
                sig.setdefault(key, len(sig))
            new = {s: sig[(1 if s in self.accept else 0,
                           tuple(part[self.trans[s][a]] for a in self.trans[s]))] for s in reach}
            if len(set(new.values())) != len(set(part.values())):
                changed = True
            part = new
        classes = sorted(set(part.values()))
        cmap = {c: i for i, c in enumerate(classes)}
        reps = {c: next(s for s in reach if part[s] == c) for c in classes}
        trans = {cmap[part[s]]: {a: cmap[part[self.trans[s][a]]] for a in self.trans[s]} for s in reach}
        accept = {cmap[part[s]] for s in reach if s in self.accept}
        return DFA(len(classes), trans, accept, cmap[part[self.start]])

    def is_group(self):
        """all transitions bijective over reachable states -> group (else aperiodic)."""
        reach = sorted(self.reachable())
        return all(len({self.trans[r][a] for r in reach}) == len(reach) for a in self.trans[reach[0]])


def l_star(alphabet: Sequence, membership: Callable, max_len: int = 8) -> DFA:
    """Angluin's L* with an equivalence teacher (exhaustive counterexample search)."""
    S = [()]; E = [()]
    def row(s): return tuple(membership(s + e) for e in E)

    def hyp():
        reps = {}
        for s in S: reps.setdefault(row(s), s)
        states = list(reps); start = row(())
        trans_r = {r: {a: row(reps[r] + (a,)) for a in alphabet} for r in states}
        acc = {r for r in states if membership(reps[r])}
        return reps, states, trans_r, acc, start

    while True:
        while True:
            reps, states, trans_r, acc, start = hyp()
            known = set(states)
            new = next((s + (a,) for s in S for a in alphabet if row(s + (a,)) not in known), None)
            if new is None: break
            S.append(new)
        inc = None
        for s in S:
            for t in S:
                if row(s) == row(t):
                    for a in alphabet:
                        if row(s + (a,)) != row(t + (a,)):
                            for e in E:
                                if membership(s + (a,) + e) != membership(t + (a,) + e):
                                    inc = (a,) + e
                            if inc: break
                    if inc: break
            if inc: break
        if inc is not None:
            E.append(inc); continue
        reps, states, trans_r, acc, start = hyp()
        idx = {r: i for i, r in enumerate(states)}
        counter = None
        for L in range(max_len + 1):
            for s in itertools.product(alphabet, repeat=L):
                r = start
                for a in s: r = trans_r[r][a]
                if (r in acc) != membership(s):
                    counter = s; break
            if counter: break
        if counter is None:
            trans = {idx[r]: {a: idx[trans_r[r][a]] for a in alphabet} for r in states}
            return DFA(len(states), trans, {idx[r] for r in acc}, idx[start]).minimise()
        for L in range(len(counter) + 1):
            p = counter[:L]
            if p not in S: S.append(p)


def constrained_decode(step_logits: Callable, trans: Dict, accept, start, budget, n_samples=1, rng=None):
    """Generate valid strings: only symbols from which `accept` stays reachable.

    step_logits(prefix) -> dict symbol -> score.  trans[s][a] = next, or None if illegal.
    """
    import random
    rng = rng or random.Random(0)

    def reach(s, rem):                       # accept reachable from s within rem steps
        cur = {s}
        for _ in range(rem):
            nxt = set()
            for u in cur:
                for a in trans[u]:
                    v = trans[u][a]
                    if v is not None: nxt.add(v)
            cur |= nxt
        return accept & cur

    outs = []
    for _ in range(n_samples):
        s = start; seq = []
        for t in range(budget):
            rem = budget - t - 1
            allowed = [a for a in trans[s] if trans[s][a] is not None and reach(trans[s][a], rem)]
            if not allowed: break
            scores = step_logits(seq)
            w = [max(scores.get(a, 0.0), 1e-9) for a in allowed]
            tot = sum(w); r = rng.random() * tot; acc = 0; pick = allowed[0]
            for a, wi in zip(allowed, w):
                acc += wi
                if r <= acc: pick = a; break
            seq.append(pick); s = trans[s][pick]
            if s in accept: break
        outs.append(seq)
    return outs
