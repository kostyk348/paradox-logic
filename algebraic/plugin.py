"""
algebraic.plugin — drop the algebra into a real training loop, in one call.

    spec = TaskSpec(alphabet, start, transition, n_states, n_out)   # known
    model = attach(base_seq_model, spec)                            # wrap: base + exact layer
    # or discover the algebra first (L*) and attach it:
    layer = discover_layer(alphabet, membership, n_out)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .layer import AlgebraicLayer
from .automaton import l_star, DFA


@dataclass
class TaskSpec:
    alphabet: Sequence
    start: int
    transition: Callable
    n_states: int
    n_out: int


class AutoAlgebra(nn.Module):
    """base sequence model + exact algebra layer (gated)."""
    def __init__(self, base: nn.Module, spec: TaskSpec, trunk_out: int = 96):
        super().__init__()
        self.base = base
        self.alg = AlgebraicLayer(spec.alphabet, spec.start, spec.transition,
                                  spec.n_states, spec.n_out, trunk_out=trunk_out)

    def forward(self, x):
        feat = self.base(x)[0]
        return self.alg(x, feat)[0]


def attach(base: nn.Module, spec: TaskSpec, trunk_out: int = 96) -> AutoAlgebra:
    return AutoAlgebra(base, spec, trunk_out)


def layer_from_dfa(dfa: DFA, alphabet: Sequence, n_out: int) -> AlgebraicLayer:
    """turn a DISCOVERED DFA into an attachable layer."""
    def transition(s, a):
        return dfa.trans[s].get(a, s)
    return AlgebraicLayer(alphabet, dfa.start, transition, dfa.n, n_out)


def discover_layer(alphabet: Sequence, membership: Callable, n_out: int, max_len: int = 8):
    """L* recovers the algebra, then wrap it as a layer (no architecture prior)."""
    dfa = l_star(list(alphabet), membership, max_len)
    return layer_from_dfa(dfa, alphabet, n_out)
