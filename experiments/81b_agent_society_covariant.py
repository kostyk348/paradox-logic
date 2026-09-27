"""
81b — Algebra-in-agents for TEXTURE: agent state (C4 lift) + local interaction.

Agents = full-resolution positions carrying a lifted C4 state; interactions are LOCAL (3x3).
Measured variants:
  (a) global mean message (exp 81)                     0.586  (degenerate interaction)
  (b) tied group-covariant interaction (algebra *is* the tie)   ~0.59 (underfits, constraints)
  (c) untied local interaction over lifted states (this file)   0.746 (underfits, 111k params)
  plain CNN                                            0.947  (20k params)
=> wrapping the algebra into the agents' INTERACTION does NOT crack texture: the group
   constraint / redundant lift costs capacity; plain local learned features are better.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)
from mnist_loader import load_mnist
Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:8000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:8000]).long()
Xte = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:2000]).long()


class GConvC4(nn.Module):
    """agents = positions with a C4 state; covariant local message passing (regular rep)."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, cout, 4 * cin, k, k) / ((4 * cin * k * k) ** 0.5))
        self.b = nn.Parameter(torch.zeros(cout)); self.cout = cout; self.k = k
    def forward(self, x):                       # x: (B, 4, cin, H, W)  -- UNTIED interaction
        B, G, cin, H, W = x.shape
        y = F.conv2d(x.reshape(B, 4 * cin, H, W), self.w.reshape(4 * self.cout, 4 * cin, self.k, self.k), padding=self.k // 2)
        return torch.relu(y.reshape(B, 4, self.cout, H, W) + self.b.view(1, 1, -1, 1, 1))


class SocietyC4(nn.Module):
    def __init__(self, c=16, k=3):
        super().__init__()
        self.perc = nn.Conv2d(1, c, k, padding=1)     # learned local perception per agent
        self.g1 = GConvC4(c, c); self.g2 = GConvC4(c, 2 * c)
        self.fc = nn.Linear(2 * c, 10)
    def forward(self, x):
        f = torch.relu(self.perc(x))                                       # (B, c, 28, 28)
        x = torch.stack([torch.rot90(f, kk, dims=(2, 3)) for kk in range(4)], 1)  # (B,4,c,28,28)
        x = self.g1(x)                                                     # (B,4,c,28,28)
        B, G, C, H, W = x.shape
        x = F.max_pool2d(x.reshape(B * G, C, H, W), 2).reshape(B, G, C, H // 2, W // 2)
        x = self.g2(x)
        x = x.mean(1)                                                      # INVARIANT over group
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def train(m, X, Y, epochs=6, lr=1e-3, tag=""):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(X))
        for s in range(0, len(X), 128):
            i = p[s:s + 128]; opt.zero_grad(); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()
        with torch.no_grad():
            print(f"    [{tag}] ep{ep} train {(m(X[:2000]).argmax(1) == Y[:2000]).float().mean().item():.3f}")


def acc(m, X, Y):
    with torch.no_grad(): return (m(X).argmax(1) == Y).float().mean().item()


soc = SocietyC4(c=16); train(soc, Xtr, Ytr, epochs=8, tag="soc")
cnn = CNN(); train(cnn, Xtr, Ytr, tag="cnn")
print("TEXTURE: agents with a lifted state + a LOCAL interaction (no algebraic tying)\n")
print(f"  equivariant agent society   acc = {acc(soc, Xte, Yte):.3f}   params {sum(p.numel() for p in soc.parameters())}")
print(f"  CNN baseline                acc = {acc(cnn, Xte, Yte):.3f}   params {sum(p.numel() for p in cnn.parameters())}")
print("\n(exp 81 global-mean 0.586; tied covariant ~0.59; untied 0.746 -- all below the CNN.)")
