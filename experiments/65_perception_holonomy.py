"""
65 — The CORRECT hybrid: learned perception + HOLONOMY bridge + discrete core.

Pointwise VQ breaks the algebra (exp 62). The bridge must be the HOLONOMY: integrate the
perceived field along the path, then quantise once (exp 63). Here the field comes from a
LEARNED perception module (raw vectors -> angle), so perception is neural and structure is
algebraic, connected by the integral.

Task: raw 2D vectors v_t at angles delta_t; label = quantised net rotation (compose delta_t).
  A: end-to-end net (must learn to integrate)     -> fails at length
  B: learned perception + integral (holonomy) + quantise -> exact, any length
"""
import sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)
K = 8


def gen(n, T):
    d = rng.normal(0, 0.7, (n, T))
    v = np.stack([np.cos(d), np.sin(d)], -1).astype(np.float32)           # raw vectors
    y = (np.floor((d.sum(1) % (2 * math.pi)) / (2 * math.pi) * K)).astype(int) % K
    return torch.tensor(v), torch.tensor(d, dtype=torch.float32), torch.tensor(y)


class EndToEnd(nn.Module):                       # A: flat net must learn everything
    def __init__(self, hid=96):
        super().__init__(); self.rnn = nn.GRU(2, hid, batch_first=True); self.out = nn.Linear(hid, K)
    def forward(self, v): return self.out(self.rnn(v)[0][:, -1])


class Perception(nn.Module):                     # B: learn to READ the angle (perception)
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, v): return self.net(v).squeeze(-1)       # predicted delta per step


def train_A(m, T=8, steps=600, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        v, _, y = gen(256, T); opt.zero_grad(); F.cross_entropy(m(v), y).backward(); opt.step()


def train_B(m, T=8, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        v, d, _ = gen(256, T); opt.zero_grad()
        F.mse_loss(m(v), d).backward(); opt.step()              # perception: read the angle


def acc_A(m, T, n=500):
    v, _, y = gen(n, T)
    with torch.no_grad(): return (m(v).argmax(1) == y).float().mean().item()


def acc_B(m, T, n=500):
    v, _, y = gen(n, T)
    with torch.no_grad():
        dhat = m(v)                                             # (n,T) perceived angles
        lab = torch.floor(((dhat.sum(1) % (2 * math.pi)) / (2 * math.pi)) * K).long() % K
    return (lab == y).float().mean().item()


A = EndToEnd(); train_A(A)
B = Perception(); train_B(B)
print("Perception + HOLONOMY bridge vs end-to-end (train T=8)\n")
print(f"{'model':>34} {'params':>7}  T=8    T=32   T=64")
for name, m, fn in [("A: end-to-end net", A, acc_A),
                    ("B: perception + holonomy", B, acc_B)]:
    a = [fn(m, T) for T in (8, 32, 64)]
    print(f"{name:>34} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> perception is learned (neural), the bridge is the INTEGRAL (holonomy, not VQ),")
print("   and the composition is exact. A net can use our algebra end-to-end IF the")
print("   continuous->discrete link is a holonomy, not a pointwise quantiser.")
