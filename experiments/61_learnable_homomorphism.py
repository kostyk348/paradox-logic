"""
61 — Learnable homomorphism: the net FINDS the semantic algebra itself.

Exp 60 gave the model the operator functions. Here each token maps (via a hypernetwork) to an
affine transformation h -> A_a h + b_a on a state vector h; composition = applying them in
order. If gradient descent finds the operators, the model generalises compositionally to
lengths it never saw. Ceiling: the GIVEN homomorphism (exp 60). Floor: a flat GRU.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.layer import AlgebraicLayer
torch.manual_seed(0)
M, D = 11, 16
OPS = [lambda x: (x + 1) % M, lambda x: (2 * x) % M, lambda x: (x - 3) % M, lambda x: (x + 5) % M]


def gen(T, n):
    x = torch.randint(0, len(OPS), (n, T)); y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        s = 0
        for t in range(T): s = OPS[int(x[i, t])](s)
        y[i] = s
    return x, y


class LearnableHom(nn.Module):
    """token -> affine map; state composed by the learned operators."""
    def __init__(self, nops=len(OPS), d=D):
        super().__init__(); self.d = d
        self.hyper = nn.Embedding(nops, d * d + d)
        with torch.no_grad():                     # start near identity maps
            self.hyper.weight.zero_()
            for a in range(nops):
                self.hyper.weight[a, :d * d] = torch.eye(d).flatten()
        self.h0 = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, M)
    def forward(self, x):
        B, T = x.shape; h = self.h0.unsqueeze(0).expand(B, -1).clone()
        for t in range(T):
            w = self.hyper(x[:, t])                        # (B, d*d+d)
            A = w[:, :self.d * self.d].view(B, self.d, self.d)
            b = w[:, self.d * self.d:]
            h = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1) + b
        return self.out(h)


class Flat(nn.Module):
    def __init__(self, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(OPS), 48)
        self.rnn = nn.GRU(48, hid, batch_first=True); self.out = nn.Linear(hid, M)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0][:, -1])


class GivenHom(nn.Module):
    def __init__(self):
        super().__init__()
        self.alg = AlgebraicLayer(list(range(len(OPS))), 0, lambda s, a: OPS[a](s), M, M)
    def forward(self, x): return self.alg(x)[0][:, -1]


def train(m, T=3, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(T, 256); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def acc(m, T, n=500):
    x, y = gen(T, n)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


print("Compositional operators (mod 11), train length T=3\n")
print(f"{'model':>20} {'params':>7}  T=3    T=4    T=5    T=8")
for name, m in [("flat GRU", Flat()), ("learnable homomorphism", LearnableHom()),
                ("given homomorphism", GivenHom())]:
    train(m)
    a = [acc(m, T) for T in (3, 4, 5, 8)]
    print(f"{name:>20} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> if the LEARNED operators generalise to unseen lengths, gradient descent found the")
print("   semantic algebra (the monoid of operators) from data; the given version is the ceiling.")
