"""
algebraic.layer — a universal algebraic layer for neural nets.

Given any task family with a finite algebra (group / monoid / finite automaton), the layer
computes the EXACT state per step and offers an exact head; a learned head then decides how
much to trust the algebra vs the network's own features. One API, applied to every family:

    layer = AlgebraicLayer(alphabet, transition, n_states, n_out, trunk_out)
    logits = layer(x_tokens, trunk_features)          # per step

`state_of(tokens)` is exact and length-independent; `hard_head()` reads it back.
"""
from __future__ import annotations
from typing import Callable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlgebraicLayer(nn.Module):
    def __init__(self, alphabet: Sequence, start: int, transition: Callable,
                 n_states: int, n_out: int, trunk_out: int = 0, hid: int = 64):
        super().__init__()
        self.alphabet = list(alphabet)
        self.index = {a: i for i, a in enumerate(self.alphabet)}
        self.start = start
        self.transition = transition
        self.n = n_states
        self.head_alg = nn.Linear(n_states, n_out)
        self.head_net = nn.Linear(trunk_out, n_out) if trunk_out else None
        self.gate = nn.Linear(n_states + trunk_out, 1) if trunk_out else None
        self.hid = hid

    def states_onehot(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T) ids -> (B,T,n_states) EXACT prefix state (no learning)."""
        B, T = x.shape
        out = torch.zeros(B, T, self.n, device=x.device)
        cur = torch.full((B,), self.start, dtype=torch.long)
        for t in range(T):
            cur = torch.tensor([self.transition(int(s), self.alphabet[int(c)])
                                for s, c in zip(cur.tolist(), x[:, t].tolist())])
            out[torch.arange(B), t, cur] = 1.0
        return out

    def forward(self, x: torch.Tensor, trunk_feat: torch.Tensor | None = None):
        st = self.states_onehot(x)                       # (B,T,n)
        alg = self.head_alg(st)
        if self.head_net is None or trunk_feat is None:
            return alg, st
        net = self.head_net(trunk_feat)
        g = torch.sigmoid(self.gate(torch.cat([st, trunk_feat], -1)))
        return g * net + (1 - g) * alg, st

    def hard_head(self):
        """state id -> class, for exact inference."""
        with torch.no_grad():
            return self.head_alg(torch.eye(self.n)).argmax(-1).tolist()
