"""
81 — Algebra wrapped in AGENTS: an agent society with algebraic interactions.

Agents = local units (image patches / text tokens). Each agent carries a state; they exchange
messages MULTIPLE ROUNDS, and the interaction is algebraic (a shared gauge / group action),
readout is invariant. Does wrapping the algebra into the agents' INTERACTION do what a single
algebraic layer could not -- crack TEXTURE (MNIST) and a TEXT task?
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)


class Society(nn.Module):
    """agents on a graph; algebraic (shared linear gauge) message passing, invariant readout."""
    def __init__(self, in_dim, n_agents, n_out, d=64, rounds=3, adj=None):
        super().__init__()
        self.enc = nn.Linear(in_dim, d)
        self.gauge = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(rounds)])  # shared algebra
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(rounds)])
        self.out = nn.Linear(d, n_out)
        self.rounds = rounds; self.n = n_agents
        self.register_buffer("adj", adj if adj is not None else torch.ones(n_agents, n_agents))
    def forward(self, x):                                   # x: (B, n_agents, in_dim)
        h = self.enc(x)
        for r in range(self.rounds):
            msg = torch.einsum("ij,bjd->bid", self.adj, h) / max(self.n, 1)
            h = self.ln[r](h + torch.relu(self.gauge[r](msg)))
        return self.out(h.mean(1))                          # invariant pooling


# ---- TEXTURE: agents = 4x4 patches of MNIST ----
from mnist_loader import load_mnist
Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
def patches(imgs):                                          # (B,1,28,28) -> (B,16,49)
    p = imgs.unfold(2, 7, 7).unfold(3, 7, 7)                # (B,1,4,4,7,7)
    return p.reshape(p.shape[0], 16, 49)
Xtr = patches(torch.tensor(Xt[:8000]).float().reshape(-1, 1, 28, 28) / 255.); Ytr = torch.tensor(Yt[:8000]).long()
Xte = patches(torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.); Yte = torch.tensor(Ye[:2000]).long()


class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def train(m, X, Y, epochs=6, lr=1e-3, reshape=None):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(X))
        for s in range(0, len(X), 128):
            i = p[s:s + 128]; xb = X[i]
            opt.zero_grad(); F.cross_entropy(m(xb), Y[i]).backward(); opt.step()


def acc(m, X, Y):
    with torch.no_grad(): return (m(X).argmax(1) == Y).float().mean().item()


soc = Society(49, 16, 10, d=64, rounds=3)
train(soc, Xtr, Ytr)
cnn = CNN()
Xi = torch.tensor(Xt[:8000]).float().reshape(-1, 1, 28, 28) / 255.
Xei = torch.tensor(Xe[:2000]).float().reshape(-1, 1, 28, 28) / 255.
train(cnn, Xi, Ytr)
print("TEXTURE (MNIST): agents=patches, algebraic interaction\n")
print(f"  agent society (16 agents, 3 rounds)  acc = {acc(soc, Xte, Yte):.3f}   params {sum(p.numel() for p in soc.parameters())}")
print(f"  CNN baseline                          acc = {acc(cnn, Xei, Yte):.3f}   params {sum(p.numel() for p in cnn.parameters())}")
print("\n=> report the delta: if the agent society does not reach the CNN, texture still needs")
print("   unconstrained learned features, algebra-in-agents notwithstanding.")
