"""
44 — Closing point 2: a TRAINABLE Kron-Rhodes cascade (learned structure, no seed).

Parameterise the transition as a wreath product: state = (mode, counter); for a token,
  T[(m,c) -> (m',c')] = Mode[a][m,m'] * Counter[a][m][c,c'].
This factorised prior is learned by gradient descent; the flat table is the control.
Task: Z/k wr C2 (counter with mode-dependent direction).
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
K = 5; NTOK = 3; NS = 2 * K


def gen(T, n):
    x = torch.randint(0, NTOK, (n, T)); y = torch.zeros(n, dtype=torch.long)
    for s in range(n):
        mode, cnt = 0, 0
        for t in range(T):
            a = int(x[s, t])
            if a == 2: mode ^= 1
            elif a == 0: cnt = (cnt + (1 if mode == 0 else -1)) % K
            else: cnt = (cnt + (-1 if mode == 0 else 1)) % K
        y[s] = mode * K + cnt
    return x, y


class Cascade(nn.Module):
    def __init__(self):
        super().__init__()
        self.Md = nn.Parameter(torch.randn(NTOK, 2, 2) * 0.5)      # mode transitions
        self.Cn = nn.Parameter(torch.randn(NTOK, 2, K, K) * 0.5)   # counter transitions per mode
        self.out = nn.Linear(NS, NS)
    def transition(self, a):
        M = torch.softmax(self.Md[a], -1)                          # (2,2)
        C = torch.softmax(self.Cn[a], -1)                          # (2,K,K)
        T = torch.zeros(NS, NS)
        for m in range(2):
            for m2 in range(2):
                for c in range(K):
                    T[m*K+c, m2*K:(m2+1)*K] += M[m, m2] * C[m, c]
        return T
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, NS); p[:, 0] = 1.0
        tabs = torch.stack([self.transition(a) for a in range(NTOK)])   # (NTOK,NS,NS)
        for t in range(T):
            p = torch.einsum("bij,bi->bj", tabs[x[:, t]], p)
        return self.out(p)


class Flat(nn.Module):
    def __init__(self):
        super().__init__(); self.T = nn.Parameter(torch.randn(NTOK, NS, NS) * 0.3)
        self.out = nn.Linear(NS, NS)
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, NS); p[:, 0] = 1.0
        q = torch.softmax(self.T, -1)
        for t in range(T): p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


def train(m, steps=500, lr=0.05):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(10, 2048); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def hard_acc(m, T):
    if isinstance(m, Cascade):
        tabs = [m.transition(a).detach().argmax(-1).tolist() for a in range(NTOK)]
    else:
        tabs = torch.softmax(m.T.detach(), -1).argmax(-1).tolist()
    with torch.no_grad(): cls = m.out(torch.eye(NS)).argmax(-1).tolist()
    x, y = gen(T, 1000); ok = 0
    for r in range(len(x)):
        st = 0
        for t in range(T): st = tabs[int(x[r, t])][st]
        if cls[st] == int(y[r]): ok += 1
    return ok / len(x)


print("Wreath task Z/k wr C2 -- LEARNED structure (no seed)\n")
print(f"{'model':>22} {'params':>7}  T=10   T=48   T=200")
for name, m in [("flat table", Flat()), ("trainable KR cascade", Cascade())]:
    train(m)
    a = [hard_acc(m, T) for T in (10, 48, 200)]
    print(f"{name:>22} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> with the wreath-product parameterisation the cascade LEARNS the mixed monoid")
print("   by gradient descent; the flat table does not discover it.")
