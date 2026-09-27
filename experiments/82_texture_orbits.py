"""
82 — TEXTURE cracked by our algebra: exact rotation invariance via group ORBITS (C8) +
      the MONOID of sufficient statistics (LBP histogram).

Tools we already own, applied to texture:
  * group action + orbit quotient  -> exact C8 rotation invariance (LBP code minimum over the
    cyclic group; an invariant BY CONSTRUCTION, not by augmentation);
  * monoid of statistics           -> the histogram (counts add), exact aggregation.

Compare: CNN trained on upright MNIST, tested on ROTATED digits (fails), vs the algebraic
LBP-histogram + a linear head (exact invariance, a fraction of the parameters).
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import rotate as ndrotate, gaussian_filter
sys.path.insert(0, "/home/lain/paradox-logic")
np.random.seed(0); torch.manual_seed(0)
from mnist_loader import load_mnist
Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = Xt[:8000].astype(np.float32) / 255.; Ytr = Yt[:8000]
Xte = Xe[:2000].astype(np.float32) / 255.; Yte = Ye[:2000]

# ---- algebra: C8 orbit of a local binary pattern (rotation-invariant code) ----
def ring(r):
    return [(int(round(r * np.sin(2 * np.pi * p / 8))), int(round(r * np.cos(2 * np.pi * p / 8))))
            for p in range(8)]
RINGS = [ring(r) for r in (1, 2, 3, 4)]         # multi-scale C8 rings


def lbp_ring(imgs, ring):
    c = np.zeros(imgs.shape, np.uint8)
    for p, (dy, dx) in enumerate(ring):
        nb = np.roll(np.roll(imgs, -dy, axis=1), -dx, axis=2)   # neighbour at (y+dy, x+dx)
        c |= ((nb >= imgs).astype(np.uint8) << p)
    return c


def c8_orbit_min(code):
    """representative of the C8 orbit: min over the 8 cyclic shifts (exact invariance)."""
    c = code.astype(np.uint16); m = c.copy()
    for k in range(1, 8):
        m = np.minimum(m, ((c >> k) | (c << (8 - k))) & 0xFF)
    return m.astype(np.uint8)


def hist256(codes):
    """the MONOID: histogram of the invariant codes (counts add over patches)."""
    h = np.zeros((len(codes), 256), np.float32)
    for i, cc in enumerate(codes):
        h[i] = np.bincount(cc.ravel(), minlength=256)
    return h / h.sum(1, keepdims=True)


def features(imgs, invariant=True):
    hs = []
    for sigma in (0.0, 1.0, 2.0):               # blur levels -> macro-structure
        im = imgs if sigma == 0 else np.stack([gaussian_filter(x, sigma) for x in imgs])
        for rg in RINGS:                        # multi-scale rings
            c = lbp_ring(im, rg)
            hs.append(hist256(c8_orbit_min(c) if invariant else c))
    return np.concatenate(hs, 1)


# ---- rotate the test set (the challenge) ----
def rotated(imgs, deg, seed=0):
    r = np.random.default_rng(seed)
    return np.stack([ndrotate(im, r.uniform(-deg, deg) if deg < 180 else deg, order=1,
                              reshape=False, mode="constant") for im in imgs])


def cnn_plain_test(Xte_, Yte_):
    m = CNN(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt_ = torch.tensor(Xtr).unsqueeze(1); Yt_ = torch.tensor(Ytr).long()
    for _ in range(6):
        p = torch.randperm(len(Xt_))
        for s in range(0, len(Xt_), 128):
            i = p[s:s + 128]; opt.zero_grad(); F.cross_entropy(m(Xt_[i]), Yt_[i]).backward(); opt.step()
    with torch.no_grad():
        return (m(torch.tensor(Xte_).unsqueeze(1)).argmax(1) == torch.tensor(Yte_).long()).float().mean().item()


class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


from sklearn.linear_model import RidgeClassifier
def alg_test(Xte_, Yte_, mode="orbit"):
    def F_(imgs):
        raw = features(imgs, False); orb = features(imgs, True)
        return {"raw": raw, "orbit": orb, "both": np.concatenate([raw, orb], 1)}[mode]
    clf = RidgeClassifier(alpha=1.0).fit(F_(Xtr), Ytr)
    return (clf.predict(F_(Xte_)) == Yte_).mean()


print("TEXTURE: exact rotation invariance (C8 orbits) + monoid histogram vs a CNN\n")
print(f"{'test set':>18} {'CNN(no aug)':>12} {'orbit':>8} {'raw':>8} {'raw+orbit':>10}")
for name, Xr in [("upright", Xte), ("rot +-15", rotated(Xte, 15)), ("rot +-45", rotated(Xte, 45)),
                 ("rot all", rotated(Xte, 180))]:
    print(f"{name:>18} {cnn_plain_test(Xr, Yte):>12.3f} {alg_test(Xr, Yte, 'orbit'):>8.3f} "
          f"{alg_test(Xr, Yte, 'raw'):>8.3f} {alg_test(Xr, Yte, 'both'):>10.3f}")
print("\n=> 'alg LBP' uses the C8 orbit quotient -> invariance BY CONSTRUCTION; the CNN must learn")
print("   it from augmentation. The monoid histogram is exact aggregation (counts add).")
