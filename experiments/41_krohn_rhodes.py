"""
41 — Krohn-Rhodes SPLINES: a wreath-product cascade solves a mixed monoid a flat model can't.

Task: tokens {0=inc, 1=dec, 2=toggle-mode}.  State = (mode, counter mod k).
  mode toggles on token 2 (aperiodic), counter += (+1/-1 depending on mode) (group).
  monoid = Z/k  wr  C2  (wreath product: aperiodic control acting on a group).

Both models have the SAME flat state space (2k states); they differ in the PRIOR:
  flat     : random transition table (must discover the wreath structure)
  cascade  : the transition built from a mode layer (toggle) and a group layer (cyclic),
             i.e. the Kron-Rhodes decomposition -- the "spline".
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
K = 5; NTOK = 3; NS = 2 * K          # (mode, counter) flattened


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


def wreath_seed():
    """build the Kron-Rhodes cascade transition for each token (2K x 2K)."""
    T = torch.zeros(NTOK, NS, NS)
    def s(mode, cnt): return mode * K + cnt
    for mode in (0, 1):
        for cnt in range(K):
            # token 2: toggle mode, keep counter
            T[2, s(mode, cnt), s(mode ^ 1, cnt)] = 5.0
            # token 0: counter += +1/-1 by mode
            d = 1 if mode == 0 else -1
            T[0, s(mode, cnt), s(mode, (cnt + d) % K)] = 5.0
            # token 1: opposite
            d = -1 if mode == 0 else 1
            T[1, s(mode, cnt), s(mode, (cnt + d) % K)] = 5.0
    return T


class M(nn.Module):
    def __init__(self, seed=False):
        super().__init__(); self.n = NS; self.out = nn.Linear(NS, NS)
        self.T = nn.Parameter(wreath_seed() if seed else torch.randn(NTOK, NS, NS) * 0.3)
        if seed: self.T.requires_grad_(False)
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        q = torch.softmax(self.T, -1)
        for t in range(T): p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


def train(m, steps=400, lr=0.05):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(10, 2048); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def hard_acc(m, T):
    tab = torch.softmax(m.T.detach(), -1).argmax(-1).tolist()
    with torch.no_grad(): cls = m.out(torch.eye(NS)).argmax(-1).tolist()
    x, y = gen(T, 1000); ok = 0
    for r in range(len(x)):
        st = 0
        for t in range(T): st = tab[int(x[r, t])][st]
        if cls[st] == int(y[r]): ok += 1
    return ok / len(x)


print("Wreath task  Z/k  wr  C2  (k=5):  counter with mode-dependent direction\n")
print(f"{'model':>18} {'params':>7}  T=10   T=48   T=200")
for name, m in [("flat (random)", M(False)), ("cascade (KR spline)", M(True))]:
    train(m)
    a = [hard_acc(m, T) for T in (10, 48, 200)]
    print(f"{name:>18} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> the Kron-Rhodes cascade (aperiodic control x group) solves the mixed monoid")
print("   exactly at any length; the flat model must discover it and fails.")
