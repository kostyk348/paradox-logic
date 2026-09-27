"""
grouprnn — a sequence model whose recurrent state is an element of a finite group Γ.

    state_t = state_{t-1} · g_t          (g_t drawn from Γ, learned from the token)

The state is kept as a probability distribution over Γ and composed through the group's
REGULAR REPRESENTATION (the permutation matrix of right multiplication), so the whole
thing is differentiable and exactly implements the holonomy of the token sequence.

This is the "group-valued sequence model": path-dependent memory (non-abelian Γ) that
never saturates, because the state space is the group itself.
"""
from __future__ import annotations
import itertools
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
def build_group(kind: str, k: int | None = None):
    """Return (elements, mul, identity)."""
    if kind == "cyclic":
        n = k
        els = list(range(n))
        return els, lambda a, b: (a + b) % n, 0
    if kind == "symmetric3":
        els = [tuple(p) for p in itertools.permutations(range(3))]
        mul = lambda a, b: tuple(a[b[i]] for i in range(3))     # (a*b)(i)=a(b(i))
        return els, mul, (0, 1, 2)
    raise ValueError(kind)


def regular_rep(els, mul):
    """P[g][h][k] = 1  iff  els[h] = mul(els[k], els[g])   (right multiplication)."""
    idx = {e: i for i, e in enumerate(els)}
    n = len(els)
    P = torch.zeros(n, n, n)
    for gi, g in enumerate(els):
        for ki, kk in enumerate(els):
            P[gi, idx[mul(kk, g)], ki] = 1.0
    return P


class GroupRNN(nn.Module):
    """State = distribution over Γ; update by the regular representation."""
    def __init__(self, kind, n_out, in_dim=None, n_tokens=None, k=None, readout_pool=True):
        super().__init__()
        els, mul, ident = build_group(kind, k)
        self.P = regular_rep(els, mul)                      # (|Γ|,|Γ|,|Γ|)
        order = len(els)
        self.order = order
        self.ident = ident
        self.readout_pool = readout_pool
        if in_dim is None:                                  # discrete tokens
            self.emb = nn.Embedding(n_tokens, order)
        else:                                               # continuous input per step
            self.emb = nn.Linear(in_dim, order)
        self.out = nn.Linear(order * (2 if readout_pool else 1), n_out)

    def forward(self, x):
        P = self.P.to(x.device)
        B, T = x.shape[0], x.shape[1]
        p = torch.zeros(B, self.order, device=x.device)
        p[:, self.ident] = 1.0
        pool = torch.zeros_like(p)
        for t in range(T):
            q = torch.softmax(self.emb(x[:, t]), dim=-1)    # (B,|Γ|) distribution of g_t
            p = torch.einsum("bg,ghk,bk->bh", q, P, p)      # p <- p · g_t
            pool = pool + p
        feat = torch.cat([p, pool / T], dim=-1) if self.readout_pool else p
        return self.out(feat)
