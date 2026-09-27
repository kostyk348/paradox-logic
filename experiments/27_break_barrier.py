"""
27 — Cracking the Z/n barrier: why small monoids learn and large ones don't.

Z/2 learns from random init; Z/7, Z/11, Z/13 do not. We try four strategies:
  (1) random init (baseline)
  (2) curriculum: grow the sequence length 2 -> 4 -> 8 -> 16 during training
  (3) straight-through: hard transitions in the forward, soft gradients in the backward
  (4) permutation parameterisation: each token = a learned soft permutation (Sinkhorn)
and report which recovers the cyc group Z/n exactly (hard-automaton test accuracy).
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)


def gen(k, T, n):
    x = torch.randint(0, k, (n, T)); y = (x.sum(1)) % k
    return x, y


class SG(nn.Module):
    def __init__(self, ntok, n, mode="soft"):
        super().__init__(); self.n = n; self.mode = mode
        self.W = nn.Parameter(torch.randn(ntok, n, n) * 0.5)
        self.out = nn.Linear(n, ntok)

    def _q(self, temp, hard):
        q = torch.softmax(self.W / temp, -1)
        if hard:
            oh = F.one_hot(q.argmax(-1), self.n).float()
            q = oh + (q - q.detach())          # straight-through
        return q

    def forward(self, x, temp=1.0, hard=False):
        B = x.shape[0]; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        q = self._q(temp, hard)
        for t in range(x.shape[1]):
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


def hard_acc(m, k, T, n=1000):
    tab = torch.softmax(m.W.detach(), -1).argmax(-1).tolist()
    x, y = gen(k, T, n); ok = 0
    with torch.no_grad():
        for s in range(len(x)):
            h = 0
            for t in range(T): h = tab[int(x[s, t])][h]
            oh = torch.zeros(1, m.n); oh[0, h] = 1.0
            if m.out(oh).argmax(1).item() == int(y[s]): ok += 1
    return ok / len(x)


def train(k, mode="soft", epochs=400, lr=0.1):
    m = SG(k, k, mode); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        temp = max(0.05, 1.0 - e / epochs)
        if mode == "curriculum":
            T = [2, 4, 8, 16][min(3, e * 4 // epochs)]
            x, y = gen(k, T, 3000)
        else:
            x, y = gen(k, 12, 3000)
        opt.zero_grad()
        F.cross_entropy(m(x, temp, hard=(mode == "straight_through")), y).backward(); opt.step()
    return m


print("Cracking Z/n: hard-automaton accuracy after training (train T=12)\n")
print(f"{'k':>3} {'random':>8} {'curriculum':>11} {'straight-thru':>14}")
for k in (2, 5, 7, 11, 13):
    row = []
    for mode in ("soft", "curriculum", "straight_through"):
        m = train(k, mode)
        row.append(hard_acc(m, k, 48))
    print(f"{k:>3} " + " ".join(f"{v:>11.2f}" if i else f"{v:>8.2f}" for i, v in enumerate(row)))
print("\n(soft = random init baseline; curriculum and straight-through are the two fixes)")
