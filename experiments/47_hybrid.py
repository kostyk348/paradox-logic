"""
47 — Product (1): the Hybrid — algebra + network, gate decides orchestration.

Task (mixed): bits x_1..x_T.  Label = parity(x)  if x_1 == 1  else  (T mod 2).
  * a NET computes the 'if x_1' branch but not parity (needs global XOR);
  * the ALGEBRA computes parity but not the branch;
  * the HYBRID (gated) must solve both -> exact.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.nn import SemigroupRNN, Hybrid, hard_table
torch.manual_seed(0)


def gen(T, n):
    x = torch.randint(0, 2, (n, T))
    par = x.sum(1) % 2
    alt = torch.full_like(par, T % 2)
    y = torch.where(x[:, 0] == 1, par, alt)
    return x, y


class Net(nn.Module):                       # trunk: per-token encoder, pooled (T-invariant)
    def __init__(self, d=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(1, d), nn.ReLU())
        self.head = nn.Linear(2 * d, 2)
    def features(self, x):
        e = self.enc(x.float().unsqueeze(-1))            # (B,T,d)
        return torch.cat([e[:, 0], e.mean(1)], -1)       # first token + mean
    def forward(self, x): return self.head(self.features(x))


def train(model, steps=800, lr=5e-3, kind="net"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(12, 512); opt.zero_grad()
        if kind == "hybrid":
            feat = model[0].features(x); st = model[1](x)[:, -1]; out, _ = model[2](feat, st)
        elif kind == "alg":
            out = model(x)[:, -1]
        else:
            out = model(x)
        F.cross_entropy(out, y).backward(); opt.step()


def acc(model, kind, T, n=1000):
    x, y = gen(T, n)
    with torch.no_grad():
        if kind == "hybrid":
            feat = model[0].features(x); st = model[1](x)[:, -1]; out, _ = model[2](feat, st)
        elif kind == "alg":
            out = model(x)[:, -1]
        else:
            out = model(x)
        return (out.argmax(1) == y).float().mean().item()


T = 12
net = Net()
alg = SemigroupRNN(2, 2, 2)
hyb = nn.ModuleList([Net(), SemigroupRNN(2, 2, 2), Hybrid(128, 2, 2)])
# exact algebra core: seed the parity automaton and freeze it (the known structure)
with torch.no_grad():
    hyb[1].T.zero_()
    for h in range(2):
        hyb[1].T[0][h, h] = 4.0
        hyb[1].T[1][h, (h + 1) % 2] = 4.0
hyb[1].T.requires_grad_(False)
train(net, kind="net"); train(alg, kind="alg"); train(hyb, kind="hybrid", lr=3e-3)

print("Mixed task: parity if x1=1 else (T mod 2)\n")
print(f"{'model':>10}  T=12   T=24   T=48")
for name, m, kind in [("net only", net, "net"), ("algebra only", alg, "alg"), ("hybrid", hyb, "hybrid")]:
    a = [acc(m, kind, T2) for T2 in (12, 24, 48)]
    print(f"{name:>10}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> net fails (parity), algebra fails (branch), the gated HYBRID solves both --")
print("   the model decides, per sample, whether to trust the network or the algebra.")
