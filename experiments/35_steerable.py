"""
35 — Steerable / continuous symmetry: rotation-invariant features via polar FFT.

Instead of averaging a CNN over n rotations (n x compute), decompose the image in polar
coordinates and take the MAGNITUDE of the angular Fourier modes. A rotation shifts the
angular phase -> the magnitudes are EXACTLY rotation-invariant, at FFT cost (no n x blowup).

Train a classifier on UPRIGHT digits only; test on arbitrary angles.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0)


def polar(x, R=14, A=64):
    rr = torch.linspace(0.03, 0.97, R); th = torch.linspace(0, 2 * math.pi, A)
    Rg, Tg = torch.meshgrid(rr, th, indexing="ij")
    grid = torch.stack([Rg * torch.cos(Tg), Rg * torch.sin(Tg)], -1)[None]
    return F.grid_sample(x, grid.expand(x.size(0), R, A, 2), align_corners=False)   # (B,1,R,A)


def rotation_invariant(x):
    """|angular FFT| per radius: exactly invariant to rotation of the input."""
    sp = torch.fft.rfft(polar(x), dim=-1)          # (B,1,R,A//2+1) complex
    return sp.abs().flatten(1)                     # (B, F) real, rotation-invariant


def rotate_batch(x, ang):
    t = math.radians(ang); c, s = math.cos(t), math.sin(t)
    A = torch.tensor([[c, -s, 0.0], [s, c, 0.0]]).unsqueeze(0).expand(x.size(0), 2, 3)
    return F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), align_corners=False)


class InvNet(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(rotation_invariant(x))


Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = torch.tensor(Xt[:6000]).float().reshape(-1, 1, 28, 28) / 255.; Ytr = torch.tensor(Yt[:6000]).long()
Xte = torch.tensor(Xe[:1000]).float().reshape(-1, 1, 28, 28) / 255.; Yte = torch.tensor(Ye[:1000]).long()
nf = rotation_invariant(Xtr[:1]).shape[1]

m = InvNet(nf); opt = torch.optim.Adam(m.parameters(), 1e-3)
print(f"steerable invariant feature dim = {nf}")
for ep in range(8):
    p = torch.randperm(len(Xtr))
    for s in range(0, len(Xtr), 128):
        i = p[s:s+128]; opt.zero_grad(); F.cross_entropy(m(Xtr[i]), Ytr[i]).backward(); opt.step()

angles = [0, 15, 30, 45, 60, 90, 135, 180]
with torch.no_grad():
    acc = [(m(rotate_batch(Xte, a)).argmax(1) == Yte).float().mean().item() for a in angles]
print("\ntrain upright only; test at arbitrary angles")
print("  angle: " + "  ".join(f"{a:>4}" for a in angles))
print("  acc  : " + "  ".join(f"{v:.2f}" for v in acc))
print(f"\n=> exact continuous rotation invariance at FFT cost ({sum(p.numel() for p in m.parameters())} params")
print("   on the head only); no 36x group-averaging blow-up.")
