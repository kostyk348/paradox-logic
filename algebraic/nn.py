"""
algebraic.nn — LEVEL 1: the algebraic core as a drop-in module for neural networks.

Requires torch. Provides
  * SemigroupRNN  — learnable finite automaton (state distribution over a monoid),
  * AlgebraHead   — exact state -> label head (route structure through algebra),
  * Hybrid        — neural trunk + algebra head (perception + exact state),
  * policy_from_dfa — a step-logits function for constrained decoding.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemigroupRNN(nn.Module):
    """state = distribution over n monoid elements; token -> transition table."""
    def __init__(self, n_tokens: int, n_states: int, n_out: int | None = None):
        super().__init__()
        self.n = n_states
        self.T = nn.Parameter(torch.randn(n_tokens, n_states, n_states) * 0.3)
        self.out = nn.Linear(n_states, n_out or n_states)

    def states(self, x):
        B, T = x.shape
        p = torch.zeros(B, self.n, device=x.device); p[:, 0] = 1.0
        q = torch.softmax(self.T, -1); out = []
        for t in range(T):
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p); out.append(p)
        return torch.stack(out, 1)

    def forward(self, x):
        return self.out(self.states(x))


def hard_table(model: SemigroupRNN):
    return torch.softmax(model.T.detach(), -1).argmax(-1).tolist()


class AlgebraHead(nn.Module):
    """exact head: a one-hot state -> class (identity when the label IS the state)."""
    def __init__(self, n_states: int, n_out: int):
        super().__init__(); self.fc = nn.Linear(n_states, n_out)
    def forward(self, state_onehot):
        return self.fc(state_onehot)


class Hybrid(nn.Module):
    """trunk (a net over the sequence) + exact algebra state, combined by a small head.

    The head sees [trunk features, algebra state] and learns the ORCHESTRATION: when the
    answer is a condition on the net's features, when it is the algebra's exact state, and
    any composition of the two.
    """
    def __init__(self, trunk_out: int, n_states: int, n_out: int, hid: int = 64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(trunk_out + n_states, hid), nn.ReLU(),
                                  nn.Linear(hid, n_out))

    def forward(self, trunk_features, state_onehot):
        return self.head(torch.cat([trunk_features, state_onehot], -1)), None


def policy_from_dfa(model: SemigroupRNN, prefix: torch.Tensor) -> Dict[int, float]:
    """step logits for constrained_decode: symbol -> score, from the learned model."""
    with torch.no_grad():
        logits = model(prefix)[0, -1]
        return {i: float(logits[i]) for i in range(logits.shape[0])}
