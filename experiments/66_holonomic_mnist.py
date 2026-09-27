"""
66 — Holonomic network on MNIST: perception (learned) + HOLONOMY bridge (perception through
     paradox) + a discrete topological core.

Bridge = the holonomy of the image's orientation field: for every 2x2 loop the winding of the
angle is an integer (a discrete, path-dependent defect). Perception is learned; the bridge is
the loop integral (NOT pointwise); the core is the discrete defect structure.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0)

Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
tr = torch.tensor(Xt[:8000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:8000]).long()
te = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:2000]).long()


def orientation_holonomy(x):
    """x:(B,1,H,W) -> discrete defect map from the winding of the orientation field."""
    B = x.shape[0]
    Sx = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]).view(1, 1, 3, 3)
    Sy = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, Sx, padding=1)[:, 0]; gy = F.conv2d(x, Sy, padding=1)[:, 0]
    th = torch.atan2(gy, gx + 1e-6)                          # (B,H,W)
    # plaquette winding: sum wrapped angle differences around each 2x2 loop
    d1 = wrap(th[:, :-1, 1:] - th[:, :-1, :-1])
    d2 = wrap(th[:, 1:, 1:] - th[:, :-1, 1:])
    d3 = wrap(th[:, 1:, :-1] - th[:, 1:, 1:])
    d4 = wrap(th[:, :-1, :-1] - th[:, 1:, :-1])
    w = (d1 + d2 + d3 + d4) / (2 * math.pi)                  # (B,H-1,W-1) ~ integer
    defc = (w.abs() > 0.5).float()
    return w, defc


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class CNN(nn.Module):
    def __init__(self, use_holo=False):
        super().__init__(); self.use_holo = use_holo
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        extra = 4 if use_holo else 0
        self.fc = nn.Linear(32 * 7 * 7 + extra, 10)
    def forward(self, x):
        h = F.max_pool2d(F.relu(self.c1(x)), 2); h = F.max_pool2d(F.relu(self.c2(h)), 2)
        v = h.flatten(1)
        if self.use_holo:
            w, defc = orientation_holonomy(x)
            feats = torch.stack([defc.sum((1, 2)), defc.mean((1, 2)),
                                 w.abs().sum((1, 2)), w.mean((1, 2))], -1)
            v = torch.cat([v, feats], -1)
        return self.fc(v)


def run(m, epochs=6, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            i = p[s:s+128]; opt.zero_grad(); F.cross_entropy(m(tr[i]), Ytr[i]).backward(); opt.step()
    with torch.no_grad(): return (m(te).argmax(1) == Yte).float().mean().item()


print("Holonomic network on MNIST (perception + loop-holonomy bridge + discrete core)\n")
print(f"{'model':>34} {'params':>8}  acc")
for name, m in [("CNN (baseline)", CNN(False)),
                ("CNN + holonomy defects", CNN(True))]:
    print(f"{name:>34} {sum(p.numel() for p in m.parameters()):>8}  {run(m):.3f}")

# do the holonomy features alone carry the digit? (discrete core only)
w, defc = orientation_holonomy(te)
feats = torch.stack([defc.sum((1, 2)), defc.mean((1, 2)), w.abs().sum((1, 2))], -1)
import numpy as np
from sklearn.linear_model import LogisticRegression
acc = LogisticRegression(max_iter=1000).fit(feats.numpy(), Yte.numpy()).score(feats.numpy(), Yte.numpy())
print(f"\n  holonomy defects alone (3 features) -> digit: {acc:.3f}")
print(f"  mean defects per digit 8: {defc[Yte==8].sum((1,2)).mean():.1f}   vs digit 1: {defc[Yte==1].sum((1,2)).mean():.1f}")
print("\n=> the loop-holonomy (a discrete, path-dependent integral of the orientation field) does")
print("   carry topological content (holes/defects), but on MNIST it barely adds to perception --")
print("   the texture is not algebraic. Perception stays learned; the bridge is the paradox.")
