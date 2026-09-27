"""
algebraic.continuous — continuous algebras for perception: steerable + fuzzy.

* SteerableInvariant: polar-FFT magnitude features -> EXACT continuous rotation invariance
  at FFT cost (one pass; no n-x group averaging).
* FuzzyAutomaton: state is an UNNORMALISED membership in [0,1] with max-product composition
  -- a weighted automaton over the semiring ([0,1], max, x); states superpose, graded accept.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SteerableInvariant(nn.Module):
    """rotation-invariant features via |angular FFT| per radius."""
    def __init__(self, R: int = 14, A: int = 64, out: int = 2):
        super().__init__()
        self.R, self.A = R, A
        self.nf = R * (A // 2 + 1)
        self.head = nn.Sequential(nn.Linear(self.nf, 256), nn.ReLU(), nn.Linear(256, out))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        rr = torch.linspace(0.03, 0.97, self.R); th = torch.linspace(0, 2 * math.pi, self.A)
        Rg, Tg = torch.meshgrid(rr, th, indexing="ij")
        grid = torch.stack([Rg * torch.cos(Tg), Rg * torch.sin(Tg)], -1)[None]
        p = F.grid_sample(x, grid.expand(x.size(0), self.R, self.A, 2), align_corners=False)
        return torch.fft.rfft(p, dim=-1).abs().flatten(1)

    def forward(self, x): return self.head(self.features(x))


class FuzzyAutomaton(nn.Module):
    """state in [0,1]^n (sigmoid), max-product composition; graded accept head."""
    def __init__(self, n_tokens: int, n_states: int, n_out: int):
        super().__init__(); self.n = n_states
        self.D = nn.Parameter(torch.randn(n_tokens, n_states, n_states) * 0.3)
        self.q0 = nn.Parameter(torch.zeros(n_states)); self.out = nn.Linear(n_states, n_out)

    def states(self, x):                                   # (B,T) -> (B,T,n)
        B, T = x.shape
        m = torch.sigmoid(self.q0).unsqueeze(0).expand(B, -1).clone()
        outs = []
        for t in range(T):
            D = torch.sigmoid(self.D[x[:, t]])             # (B,n,n)
            m = (D * m.unsqueeze(-1)).max(dim=1).values    # max-product
            outs.append(m)
        return torch.stack(outs, 1)

    def forward(self, x): return self.out(self.states(x)[:, -1])
