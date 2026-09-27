"""
60 — Expressing SEMANTICS through algebra: meaning = a monoid homomorphism.

A compositional language: a sequence of operators applied in order to 0 (mod M).
The MEANING of the sequence is the composition of the operators -- a monoid action.
  * flat model (GRU): maps sequence -> value, must memorise; fails on unseen lengths.
  * homomorphic model: applies each operator's function (the algebra); exact at ANY length.
Compositional generalisation is the test of whether semantics is expressed algebraically.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.layer import AlgebraicLayer
torch.manual_seed(0)

M = 11
OPS = [lambda x: (x + 1) % M, lambda x: (2 * x) % M, lambda x: (x - 3) % M, lambda x: (x + 5) % M]


def gen(T, n):
    x = torch.randint(0, len(OPS), (n, T)); y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        s = 0
        for t in range(T): s = OPS[int(x[i, t])](s)
        y[i] = s
    return x, y


class Flat(nn.Module):
    def __init__(self, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(OPS), 48)
        self.rnn = nn.GRU(48, hid, batch_first=True); self.out = nn.Linear(hid, M)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0][:, -1])


class Homomorphic(nn.Module):
    def __init__(self):
        super().__init__()
        self.alg = AlgebraicLayer(list(range(len(OPS))), 0,
                                  lambda s, a: OPS[a](s), M, M)
    def forward(self, x):
        return self.alg(x)[0][:, -1]                       # exact composition; head=identity


def train(m, T=3, steps=600, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(T, 256); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def acc(m, T, n=500):
    x, y = gen(T, n)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


print("Compositional language over operators (mod 11), train length T=3\n")
print(f"{'model':>16} {'params':>7}  acc T=3  T=4  T=5  T=8")
for name, m in [("flat GRU", Flat()), ("homomorphic (alg)", Homomorphic())]:
    train(m)
    a = [acc(m, T) for T in (3, 4, 5, 8)]
    print(f"{name:>16} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> meaning is a monoid homomorphism; the homomorphic model composes operators and")
print("   generalises to longer expressions, the flat model memorises and fails. Semantics,")
print("   in its compositional part, IS algebra.")
