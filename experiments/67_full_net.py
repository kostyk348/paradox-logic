"""
67 — A FULL network using our layers, on a task that needs BOTH perception and structure.

Task: an image holds 4 slots, each showing an operator symbol (a 4x4 codebook pattern).
The label is the COMPOSITION of the operators (mod 11). Train on images with <=2 non-identity
operators; test on 3-4 (compositional generalisation).

  A: end-to-end CNN (perception + composition learned together) -> memorises, fails
  B: perception CNN (reads the 4 symbols) + EXACT algebraic composition -> generalises
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)
M = 11
OPS = {0: lambda x: x, 1: lambda x: (x + 1) % M, 2: lambda x: (2 * x) % M,
       3: lambda x: (x - 3) % M, 4: lambda x: (x + 5) % M}
NS = 5                                                  # symbol alphabet (0 = identity)
# fixed 4x4 codebook patterns per symbol
CODEBOOK = np.array([rng.normal(0, 1, (4, 4)) for _ in range(NS)], dtype=np.float32)


def gen(n, kmax):
    imgs = np.zeros((n, 1, 16, 16), dtype=np.float32)
    syms = np.zeros((n, 4), dtype=int); y = np.zeros(n, dtype=int)
    for i in range(n):
        k = rng.integers(0, kmax + 1)
        s = np.zeros(4, dtype=int)
        pos = rng.choice(4, size=k, replace=False)
        for p in pos: s[p] = rng.integers(1, NS)
        for p in range(4):
            r, c = divmod(p, 2)
            patch = CODEBOOK[s[p]] + rng.normal(0, 0.5, (4, 4))
            imgs[i, 0, r * 8:r * 8 + 4, c * 8:c * 8 + 4] = patch
        val = 0
        for p in range(4): val = OPS[s[p]](val)
        syms[i] = s; y[i] = val
    return torch.tensor(imgs), torch.tensor(syms), torch.tensor(y)


class CNN(nn.Module):                                   # A: end-to-end
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Linear(32 * 16 * 16, M)
    def forward(self, x): return self.fc(self.c(x))


class Perception(nn.Module):                            # B: read the 4 slots
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(16 * 16 * 16, 4 * NS)
    def forward(self, x):
        return self.fc(self.c(x).flatten(1)).view(-1, 4, NS)


def train(perception, steps=800, lr=3e-3):
    opt = torch.optim.Adam(perception.parameters(), lr=lr)
    for e in range(steps):
        x, s, _ = gen(128, 2); opt.zero_grad()
        F.cross_entropy(perception(x).reshape(-1, NS), s.reshape(-1)).backward(); opt.step()


def train_A(m, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, _, y = gen(128, 2); opt.zero_grad()
        F.cross_entropy(m(x), y).backward(); opt.step()


def acc_A(m, kmax, n=500):
    x, _, y = gen(n, kmax)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def acc_B(p, kmax, n=500):
    x, s, y = gen(n, kmax)
    with torch.no_grad():
        pred = p(x).argmax(-1).numpy()                  # read symbols
    out = np.zeros(n, dtype=int)
    for i in range(n):
        v = 0
        for k in range(4): v = OPS[int(pred[i, k])](v)   # EXACT composition
        out[i] = v
    return float((torch.tensor(out) == y).float().mean())


A = CNN(); train_A(A)
P = Perception(); train(P)
print("Full net using our layers: perception + algebraic composition\n")
print(f"{'model':>34} {'params':>7}  k<=2   k<=3   k<=4   symbol-acc")
for name, ma, mb in [("A: end-to-end CNN", A, None), ("B: perception + algebra", None, P)]:
    if ma is not None:
        a = [acc_A(ma, k) for k in (2, 3, 4)]
        print(f"{name:>34} {sum(p.numel() for p in ma.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
    else:
        a = [acc_B(mb, k) for k in (2, 3, 4)]
        x, s, _ = gen(500, 4)
        with torch.no_grad(): sa = (mb(x).argmax(-1) == s).float().mean().item()
        print(f"{name:>34} {sum(p.numel() for p in mb.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a)
              + f"     {sa:.3f}")
print("\n=> a FULL network = learned perception + our algebraic core: it reads the image and")
print("   composes exactly, generalising to more operators; end-to-end fails at unseen counts.")
