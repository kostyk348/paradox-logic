"""
32 — Continuous symmetry (SO(2) approximated by C_n) for image classification.

A C4 prior is invariant only to multiples of 90. A C_n prior (n = 12, 36) is invariant to
a fine angular grid -> approximates a CONTINUOUS rotation symmetry. Train on UPRIGHT MNIST
only; test on arbitrary angles (0/15/30/45/60/90/180) via differentiable rotation.
"""
import sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0)


def rotate_batch(x, ang_deg):
    t = math.radians(ang_deg); c, s = math.cos(t), math.sin(t)
    A = torch.tensor([[c, -s, 0.0], [s, c, 0.0]]).unsqueeze(0).expand(x.size(0), 2, 3)
    grid = F.affine_grid(A, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class GnInvariant(nn.Module):
    """shared CNN, output averaged over n rotations -> C_n invariant (~SO(2) for large n)."""
    def __init__(self, n):
        super().__init__(); self.cnn = CNN(); self.angles = [360.0 * i / n for i in range(n)]
    def forward(self, x):
        outs = [self.cnn(rotate_batch(x, a)) for a in self.angles]
        return torch.stack(outs, 0).mean(0)


Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:6000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:6000]).long()
Xte = torch.tensor(Xe[:1000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:1000]).long()


def run(m, epochs=3, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 64):
            i = p[s:s+64]; opt.zero_grad(); F.cross_entropy(m(Xtr[i]), Ytr[i]).backward(); opt.step()


angles = [0, 15, 30, 45, 60, 90, 180]
print("Train on UPRIGHT only; test at arbitrary angles\n")
print(f"{'model':>16}  " + "  ".join(f"{a:>4}" for a in angles))
for name, m in [("CNN", CNN()), ("C4 (n=4)", GnInvariant(4)), ("C12", GnInvariant(12)), ("C36", GnInvariant(36))]:
    run(m)
    with torch.no_grad():
        acc = [(m(rotate_batch(Xte, a)).argmax(1) == Yte).float().mean().item() for a in angles]
    print(f"{name:>16}  " + "  ".join(f"{v:.2f}" for v in acc))
print("\n => C4 holds only multiples of 90; C36 is (near) continuously invariant --")
print("    a continuous symmetry prior, trained on upright digits alone.")
