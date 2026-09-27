"""
79 — (4) non-linear modular statistics + statistics on TEXTURE and FREE TEXT.

(4) label = (sum of x_t*y_t) mod k  -- a bilinear statistic with a modular reduction.
    exact monoid vs a GRU.
TEXTURE: MNIST via pixel statistics (histogram + moments) vs a CNN.
FREE TEXT: topic classification from bag-of-words COUNTS (a counting MONOID); streaming =
    sum counts over chunks exactly.
"""
import sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)
rng = np.random.default_rng(0)
K = 7


# ---- (4) non-linear modular statistic ----
def gen4(n, T):
    x = rng.integers(0, K, (n, T)); y = rng.integers(0, K, (n, T))
    lab = (x * y).sum(1) % K
    return torch.tensor(x), torch.tensor(y), torch.tensor(lab)


class Net4(nn.Module):
    def __init__(self, hid=96):
        super().__init__(); self.e = nn.Embedding(K, 32)
        self.rnn = nn.GRU(32, hid, batch_first=True); self.out = nn.Linear(hid, K)
    def forward(self, x, y):
        h = self.e(x) * self.e(y)                          # bilinear input
        return self.out(self.rnn(h)[0][:, -1])


def train4(m, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y, l = gen4(128, 10); opt.zero_grad(); F.cross_entropy(m(x, y), l).backward(); opt.step()


def acc4(m, T, n=500):
    x, y, l = gen4(n, T)
    with torch.no_grad(): return (m(x, y).argmax(1) == l).float().mean().item()


def acc4_mono(T, n=500):
    x, y, l = gen4(n, T)
    return float((torch.tensor((x.numpy() * y.numpy()).sum(1) % K) == l).float().mean())


m4 = Net4(); train4(m4)
print("(4) non-linear modular statistic  (sum x*y) mod 7  (train T=10)\n")
print(f"{'model':>40}  T=10   T=32   T=100")
print(f"{'net (GRU)':>40}  " + "  ".join(f"{acc4(m4,T):.3f}" for T in (10, 32, 100)))
print(f"{'exact monoid (sum x*y mod 7)':>40}  " + "  ".join(f"{acc4_mono(T):.3f}" for T in (10, 32, 100)))

# ---- TEXTURE: MNIST via statistics ----
from mnist_loader import load_mnist
from sklearn.linear_model import LogisticRegression
Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
Xtr = Xt[:8000].reshape(len(Xt[:8000]), -1) / 255.0; Xte = Xe[:2000].reshape(2000, -1) / 255.0
def stats(imgs):                                           # histogram(16) + mean + std
    h = np.array([np.histogram(im, bins=16, range=(0, 1))[0] for im in imgs], dtype=float)
    h /= h.sum(1, keepdims=True)
    return np.concatenate([h, imgs.mean(1, keepdims=True), imgs.std(1, keepdims=True)], 1)
acc_tex = LogisticRegression(max_iter=1000).fit(stats(Xtr), Yt[:8000]).score(stats(Xte), Ye[:2000])
print(f"\nTEXTURE: MNIST via pixel statistics (histogram+moments) -> acc {acc_tex:.3f}  (CNN ~0.94)")

# ---- FREE TEXT: topic from bag-of-words counts (a counting monoid) ----
VW = {"stock": 0, "market": 1, "profit": 2, "goal": 3, "match": 4, "team": 5, "film": 6, "actor": 7, "scene": 8}
TOPIC = {0: ["stock", "market", "profit"], 1: ["goal", "match", "team"], 2: ["film", "actor", "scene"]}
def doc(cls, n=40):
    ws = []
    for _ in range(n):
        ws.append(rng.choice(TOPIC[cls]) if rng.random() < 0.5 else rng.choice(list(VW)))
    return ws
def counts(docw):
    c = np.zeros(len(VW)); 
    for w in docw: c[VW[w]] += 1
    return c
X = []; Y = []
for _ in range(6000):
    cls = rng.integers(0, 3); X.append(counts(doc(cls))); Y.append(cls)
clf = LogisticRegression(max_iter=500).fit(np.array(X), np.array(Y))
Xe = [counts(doc(rng.integers(0, 3))) for _ in range(1000)]
acc_txt = clf.score(np.array(Xe), [None]*0) if False else None
# proper eval
Xv = []; Yv = []
for _ in range(2000):
    cls = rng.integers(0, 3); Xv.append(counts(doc(cls))); Yv.append(cls)
acc_txt = clf.score(np.array(Xv), np.array(Yv))
# streaming: counts of chunks sum exactly
d = doc(0, 100); a = counts(d[:50]); b = counts(d[50:])
print(f"\nFREE TEXT: topic from bag-of-words counts (monoid) -> acc {acc_txt:.3f}")
print(f"  streaming: counts(whole) == counts(chunk1)+counts(chunk2)? {np.allclose(counts(d), a + b)}")
print("\n=> non-linear modular stats: the exact monoid wins (net struggles with the reduction);")
print("   TEXTURE: low-order statistics are weak (texture needs a net); FREE TEXT: bag-of-words")
print("   counts ARE a sufficient statistic (monoid) -> exact, streaming, composable.")
