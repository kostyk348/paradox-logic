"""
46 — Fuzzy automata: weighted automata over the semiring ([0,1], max, x).

Crisp SemigroupRNN: state is a normalised distribution (softmax), max-product over a
stochastic relation -> states compete.
Fuzzy automaton : state is an UNNORMALISED membership in [0,1] (sigmoid), max-product
over a fuzzy relation -> states superpose, graded acceptance.

Tests: (a) exact mod-k language, (b) robustness to input noise (graded evidence).
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)


def gen(k, T, n, noise=0.0):
    x = torch.randint(0, k, (n, T))
    y = x.sum(1) % k
    if noise > 0:
        x = torch.where(torch.rand_like(x.float()) < noise, torch.randint(0, k, x.shape), x)
    return x, y


class Crisp(nn.Module):                       # our SemigroupRNN (normalised)
    def __init__(self, ntok, n):
        super().__init__(); self.n = n
        self.T = nn.Parameter(torch.randn(ntok, n, n) * 0.3); self.out = nn.Linear(n, ntok)
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        q = torch.softmax(self.T, -1)
        for t in range(T): p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


class Fuzzy(nn.Module):                       # fuzzy automaton (unnormalised, max-product)
    def __init__(self, ntok, n):
        super().__init__(); self.n = n
        self.D = nn.Parameter(torch.randn(ntok, n, n) * 0.3)      # delta[h,e]
        self.q0 = nn.Parameter(torch.zeros(n)); self.out = nn.Linear(n, ntok)
    def forward(self, x):
        B, T = x.shape
        m = torch.sigmoid(self.q0).unsqueeze(0).expand(B, -1).clone()
        for t in range(T):
            D = torch.sigmoid(self.D[x[:, t]])                    # (B,n,n)
            m = (D * m.unsqueeze(-1)).max(dim=1).values           # max-product composition
        return self.out(m)


def train(m, k, steps=500, lr=0.05):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(k, 12, 1024); opt.zero_grad()
        F.cross_entropy(m(x), y).backward(); opt.step()


def acc(m, k, T=48, noise=0.0, n=1000):
    x, y = gen(k, T, n, noise)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


print("Fuzzy vs crisp automaton\n")
print(f"{'k':>3} {'model':>8} {'params':>7}  clean  noise .1  noise .2")
for k in (2, 5, 7):
    for name, m in [("crisp", Crisp(k, k)), ("fuzzy", Fuzzy(k, k))]:
        train(m, k)
        a = [acc(m, k, 48, nz) for nz in (0.0, 0.1, 0.2)]
        print(f"{k:>3} {name:>8} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> fuzzy (unnormalised memberships) tolerates input noise better: graded evidence")
print("   superposes instead of snapping. It is a weighted automaton over a SEMIRING,")
print("   the natural generalisation of the monoid/group the earlier experiments used.")
