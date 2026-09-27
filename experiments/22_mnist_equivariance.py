"""
22 — MNIST, done the group way: promote the TASK's symmetry group into the architecture.

The digit label is not a group, but the relevant algebra of MNIST is its ROTATION group:
a digit rotated by 90/180/270 is the same class. A standard CNN is equivariant to
translation only; it is NOT rotation-invariant, so it collapses on rotated test digits.
A C4-invariant model (shared CNN, pooled over the cyclic group C4) is, by construction.

Train on UPRIGHT digits only; test on upright + rot90/180/270.
=> algebra match: the group-matched model rides the rotation group for free.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist

torch.manual_seed(0)
G = 4  # cyclic group C4


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class C4Invariant(nn.Module):
    """Shared CNN, output averaged over the group C4 -> invariant to 90-degree rotations."""
    def __init__(self):
        super().__init__()
        self.cnn = CNN()

    def forward(self, x):
        outs = [self.cnn(torch.rot90(x, k, (2, 3))) for k in range(G)]
        return torch.stack(outs, 0).mean(0)


Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:12000]).float().reshape(-1, 1, 28, 28) / 255.0
Ytr = torch.tensor(Yt[:12000]).long()
Xte = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.0
Yte = torch.tensor(Ye[:2000]).long()


def rotate(x, k):
    return torch.rot90(x, k, (2, 3))


def train(model, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()


def evaluate(model):
    with torch.no_grad():
        return [(model(rotate(Xte, k)).argmax(1) == Yte).float().mean().item() for k in range(4)]


print("Train on UPRIGHT digits only; test on rot 0 / 90 / 180 / 270\n")
print(f"{'model':>18} {'params':>8}   acc @ rot 0 / 90 / 180 / 270")
for name, m in [("CNN (baseline)", CNN()), ("C4-invariant CNN", C4Invariant())]:
    train(m)
    a = evaluate(m)
    npar = sum(p.numel() for p in m.parameters())
    print(f"{name:>18} {npar:>8}   " + "  ".join(f"{v:.3f}" for v in a))

print("\n=> the baseline collapses off-axis; the group-matched model is invariant by")
print("   construction. Same theorem: match the model's algebra to the task's algebra.")
