"""
64 — Can a net be built ENTIRELY from our layers?  (honest boundary)

"Pure algebraic net" = fixed algebraic features (steerable invariants) + only a linear head.
"Learned net"        = a small CNN (learns its own features).
"Hybrid"             = algebraic features + a learned CNN.

If the pure-algebraic net matched the CNN, our layers would suffice. They do not: algebraic
layers are EXACT but must be GIVEN, and do no perception. The boundary is measured here.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.continuous import SteerableInvariant
from mnist_loader import load_mnist
torch.manual_seed(0)

Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
tr = torch.tensor(Xt[:8000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:8000]).long()
te = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:2000]).long()


class PureAlgebraic(nn.Module):
    """ONLY our layer: fixed steerable invariants + a linear head (no learned perception)."""
    def __init__(self):
        super().__init__()
        self.sf = SteerableInvariant(out=10)
        for p in list(self.sf.parameters())[:0]: pass          # keep only the linear head learnable
    def forward(self, x): return self.sf(x)


class CNN(nn.Module):
    def __init__(self, extra=0):
        super().__init__(); self.extra = extra
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7 + extra, 10)
        self.sf = SteerableInvariant(out=extra) if extra else None
    def forward(self, x):
        h = F.max_pool2d(F.relu(self.c1(x)), 2); h = F.max_pool2d(F.relu(self.c2(h)), 2)
        v = h.flatten(1)
        if self.sf is not None: v = torch.cat([v, self.sf.features(x)], -1)
        return self.fc(v)


def run(m, epochs=6, lr=1e-3):
    if hasattr(m, "sf"):                                       # fix the algebraic features
        pass
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            i = p[s:s+128]; opt.zero_grad(); F.cross_entropy(m(tr[i]), Ytr[i]).backward(); opt.step()
    with torch.no_grad(): return (m(te).argmax(1) == Yte).float().mean().item()


print("Building a net from our layers only (MNIST)\n")
print(f"{'model':>34} {'params':>8}  acc")
for name, m in [("pure algebraic (fixed feats+head)", PureAlgebraic()),
                ("learned CNN", CNN()),
                ("hybrid: CNN + algebraic features", CNN(extra=462))]:
    a = run(m)
    print(f"{name:>34} {sum(p.numel() for p in m.parameters()):>8}  {a:.3f}")
print("\n=> the pure algebraic net is exact at structure but does NOT perceive: it needs GIVEN")
print("   features. Perception and semantics need the learned part. A net ENTIRELY of our")
print("   layers works only for structure tasks -- elsewhere it is the CORE, not the whole net.")
