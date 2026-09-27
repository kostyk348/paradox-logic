"""
30 — "Seed must match the algebra" on MNIST: the PRIOR is the symmetry group.

Same lesson as the cyclic scaffold for Z/n, now for images. Train on UPRIGHT digits only,
test on rot 0/90/180/270 with three priors:
  none  (CNN)                 -> collapses off-axis
  C2    (invariant to 180 only) -> holds 0/180, fails 90/270
  C4    (invariant to all 90s)  -> holds everywhere
The prior must match the task's symmetry group; capacity does not substitute.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class Prior(nn.Module):
    """shared CNN, output averaged over a chosen set of rotations (the prior group)."""
    def __init__(self, group):
        super().__init__(); self.cnn = CNN(); self.group = group
    def forward(self, x):
        outs = [self.cnn(torch.rot90(x, k, (2, 3))) for k in self.group]
        return torch.stack(outs, 0).mean(0)


Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:10000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:10000]).long()
Xte = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:2000]).long()
rot = lambda x, k: torch.rot90(x, k, (2, 3))


def run(model, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 64):
            i = p[s:s+64]; opt.zero_grad(); F.cross_entropy(model(Xtr[i]), Ytr[i]).backward(); opt.step()
    with torch.no_grad():
        return [(model(rot(Xte, k)).argmax(1) == Yte).float().mean().item() for k in range(4)]


print("Train on UPRIGHT only; test on rot 0/90/180/270\n")
print(f"{'prior (group)':>16}  acc @ 0 / 90 / 180 / 270")
for name, m in [("none (CNN)", CNN()),
                ("C2 {0,180}", Prior([0, 2])),
                ("C4 {0,90,180,270}", Prior([0, 1, 2, 3]))]:
    a = run(m)
    print(f"{name:>16}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> a prior invariant to the WRONG subgroup (C2) fixes only the axes it covers;")
print("   the prior must match the task's symmetry group (C4) -- like the cyclic seed for Z/n.")
