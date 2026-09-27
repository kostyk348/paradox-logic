"""
algebraic.runtime — universal wiring for regular tasks:

    quantise(continuous) -> discover the algebra (L*) -> execute exactly -> decode by grammar

The neural net is optional: perception/quantisation (or a learned VQ); the STRUCTURE is
discovered and executed symbolically, so it is exact at any length with no seed.
"""
from __future__ import annotations
from typing import Callable, Iterable, Sequence

from .automaton import l_star, DFA, constrained_decode
from .discovery import rpni, classify


class Runtime:
    def __init__(self, alphabet, membership: Callable = None, dfa: DFA = None):
        self.alphabet = list(alphabet)
        self.dfa = dfa if dfa is not None else l_star(self.alphabet, membership)

    # ---- exact execution ----
    def state_of(self, tokens: Sequence) -> int:
        r = self.dfa.start
        for a in tokens:
            r = self.dfa.trans[r][a]
        return r

    def accepts(self, tokens: Sequence) -> bool:
        return self.dfa.run(tokens)

    def type(self) -> str:
        return "group" if self.dfa.is_group() else "aperiodic"

    # ---- continuous -> symbolic (fixed-bin quantiser; a learned VQ can replace it) ----
    @staticmethod
    def quantize(values: Iterable[float], lo: float, hi: float, n: int):
        out = []
        for v in values:
            i = int((v - lo) / (hi - lo) * n)
            out.append(min(max(i, 0), n - 1))
        return out

    # ---- grammar-constrained generation ----
    def decode(self, step_logits, budget, n_samples=1, accept=None, rng=None):
        from .automaton import constrained_decode
        trans = {s: dict(self.dfa.trans[s]) for s in self.dfa.trans}
        return constrained_decode(step_logits, trans, accept or self.dfa.accept,
                                  self.dfa.start, budget, n_samples, rng)


def discover_from_examples(positives, negatives) -> DFA:
    """RPNI front-end: algebra from labelled examples only (no oracle)."""
    n, trans, accept, start = rpni(positives, negatives)
    return DFA(n, trans, accept, start)
