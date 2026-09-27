"""
77 — STATISTICS dissected by algebra: sufficient statistics form a MONOID.

The sufficient statistics (count, sum, sums of products) COMBINE associatively:
    S(A ++ B) = S(A) . S(B)      -- a monoid (Chan's parallel algorithm).
So a network can track them EXACTLY at any length. Task: two channels with a fixed
correlation sign; the label is the sign of the covariance (a function of the statistics).

  A: flat net over the sequence (must approximate the statistic)
  B: exact sufficient statistics (monoid) -> sign of covariance   (exact)
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)


def gen(n, T):
    X = np.zeros((n, T, 2), dtype=np.float32); y = np.zeros(n, dtype=int)
    for i in range(n):
        rho = rng.choice([-0.7, 0.7])
        cov = np.array([[1.0, rho], [rho, 1.0]])
        X[i] = rng.multivariate_normal([0, 0], cov, size=T)
        sx = X[i, :, 0].sum(); sy = X[i, :, 1].sum(); sxy = (X[i, :, 0] * X[i, :, 1]).sum()
        covhat = sxy - sx * sy / T
        y[i] = 1 if covhat > 0 else 0
    return torch.tensor(X), torch.tensor(y)


class Flat(nn.Module):                                   # A
    def __init__(self, hid=64):
        super().__init__(); self.rnn = nn.GRU(2, hid, batch_first=True); self.out = nn.Linear(hid, 2)
    def forward(self, x): return self.out(self.rnn(x)[0][:, -1])


def stats_sign(x):                                       # B: exact monoid of sufficient stats
    sx = x[:, :, 0].sum(1); sy = x[:, :, 1].sum(1); sxy = (x[:, :, 0] * x[:, :, 1]).sum(1)
    T = x.shape[1]
    covhat = sxy - sx * sy / T
    return (covhat > 0).long()


def train(m, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(128, 20); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def accA(m, T, n=500):
    x, y = gen(n, T)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def accB(T, n=500):
    x, y = gen(n, T)
    return (stats_sign(x) == y).float().mean().item()


A = Flat(); train(A)
print("Statistics as a monoid: sign of covariance (train T=20)\n")
print(f"{'model':>34}  T=20   T=100  T=500")
print(f"{'A: flat net (approximates the stat)':>34}  " + "  ".join(f"{accA(A,T):.3f}" for T in (20, 100, 500)))
print(f"{'B: exact sufficient stats (monoid)':>34}  " + "  ".join(f"{accB(T):.3f}" for T in (20, 100, 500)))
print("\n=> sufficient statistics combine as a monoid, so they are EXACT at any length; the")
print("   flat net must approximate the statistic and drifts. Statistics IS algebra where")
print("   the statistic is a homomorphism (Darmois-Koopman-Pitman: exponential families).")
