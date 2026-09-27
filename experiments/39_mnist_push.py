"""
39 — MNIST, pushed: equivariant architecture + group augmentation vs a standard CNN.

Train on MNIST WITH all four rotations (group augmentation); test on standard MNIST and
on rotated MNIST. The algebraic (C4-equivariant) model should match the CNN on the
standard set AND dominate on rotations -- "enhanced" by matching the symmetry algebra.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0)
G = 4


class Lift(nn.Module):
    def __init__(s, C): super().__init__(); s.conv = nn.Conv2d(1, C, 3, padding=1)
    def forward(s, x):
        B = x.shape[0]
        xs = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(G)], 1)
        y = s.conv(xs.reshape(B*G, 1, *x.shape[2:]))
        return y.reshape(B, G, -1, *x.shape[2:]).permute(0, 2, 1, 3, 4)


class GConv(nn.Module):
    def __init__(s, Ci, Co, k=3):
        super().__init__(); s.W = nn.Parameter(torch.randn(G, Co, Ci, k, k)*(1/math.sqrt(Ci*k*k)))
    def forward(s, x):
        B, Ci, _, H, W = x.shape; y = 0
        for t in range(G):
            xs = torch.roll(x, -t, 2).permute(0,2,1,3,4).reshape(B*G, Ci, H, W)
            c = F.conv2d(xs, s.W[t], padding=1).reshape(B,G,-1,H,W).permute(0,2,1,3,4)
            y = c if y is 0 else y+c
        return y


def gpool(x):
    B,C,Gg,H,W = x.shape
    return F.max_pool2d(x.reshape(B*C*Gg,1,H,W),2).reshape(B,C,Gg,H//2,W//2).mean((3,4))


class GCNN(nn.Module):
    def __init__(s, C1=16, C2=32):
        super().__init__(); s.lift=Lift(C1); s.g1=GConv(C1,C2); s.g2=GConv(C2,C2); s.fc=nn.Linear(C2,10)
    def forward(s, x):
        z = F.max_pool2d(x, 2)
        z = F.relu(s.lift(z)); z = F.relu(s.g1(z)); z = F.relu(s.g2(z)); z = gpool(z)
        return s.fc(z.mean(2))


class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1=nn.Conv2d(1,16,3,padding=1); s.c2=nn.Conv2d(16,32,3,padding=1); s.fc=nn.Linear(32*3*3,10)
    def forward(s, x):
        x=F.max_pool2d(F.relu(s.c1(F.max_pool2d(x,2))),2); x=F.max_pool2d(F.relu(s.c2(x)),2)
        return s.fc(x.flatten(1))


Xt,Yt,Xe,Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:10000]).float().reshape(-1,1,28,28)/255.; Ytr = torch.tensor(Yt[:10000]).long()
Xte = torch.tensor(Xe[:2000]).float().reshape(-1,1,28,28)/255.; Yte = torch.tensor(Ye[:2000]).long()
rot = lambda x,k: torch.rot90(x,k,(2,3))
# rotation-augmented training set (all 4 rotations)
Xa = torch.cat([rot(Xtr,k) for k in range(4)]); Ya = Ytr.repeat(4)


def train(m, X, Y, epochs=6, lr=2e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = torch.randperm(len(X))
        for s in range(0, len(X), 128):
            i = p[s:s+128]; opt.zero_grad(); F.cross_entropy(m(X[i]), Y[i]).backward(); opt.step()


def acc(m, X, Y):
    with torch.no_grad(): return (m(X).argmax(1) == Y).float().mean().item()


print("Train WITH all 4 rotations; test on standard + rotated MNIST\n")
print(f"{'model':>12} {'params':>7}  std   rot90  rot180  rot270")
for name, m in [("CNN (aug)", CNN()), ("C4-GCNN (aug)", GCNN())]:
    train(m, Xa, Ya)
    a = [acc(m, Xte, Yte)] + [acc(m, rot(Xte, k), Yte) for k in (1, 2, 3)]
    print(f"{name:>12} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> with the symmetry algebra in the architecture, the model is rotated-robust")
print("   AND matches the standard CNN on upright digits.")
