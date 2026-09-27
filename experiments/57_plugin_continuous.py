"""
57 — Both at once: the training plugin across tasks + continuous algebras.

(1) plugin: `attach(base, spec)` wraps any sequence model with an exact algebra layer.
(2) discover_layer: L* finds the algebra, then it is attached (no prior).
(3) continuous: SteerableInvariant on rotated MNIST; FuzzyAutomaton on a noisy counting task.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.plugin import TaskSpec, attach, discover_layer
from algebraic.continuous import SteerableInvariant, FuzzyAutomaton

torch.manual_seed(0)


class GRU(nn.Module):
    def __init__(self, V, hid=96):
        super().__init__(); self.emb = nn.Embedding(V, 48)
        self.rnn = nn.GRU(48, hid, batch_first=True)
    def forward(self, x): return self.rnn(self.emb(x))[0], None


def gen(spec, T, n):
    A, start, tr, ns, no = spec.alphabet, spec.start, spec.transition, spec.n_states, spec.n_out
    x = torch.randint(0, len(A), (n, T)); y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        s = start
        for t in range(T): s = tr(s, int(x[i, t]))
        y[i] = s
    return x, y


print("(1)+(2) training plugin: base GRU + exact algebra (attached or DISCOVERED)\n")
print(f"{'task':>10} {'model':>16} {'params':>7}  T=10   T=40")
parity = TaskSpec([0, 1], 0, lambda s, a: s ^ a, 2, 2)
for name, spec, use_disc in [("parity", parity, False), ("parity", parity, True)]:
    base = GRU(len(spec.alphabet))
    if use_disc:
        lay = discover_layer([0, 1], lambda seq: sum(seq) % 2 == 0, 2)
        model = attach(base, spec)                      # same wrap (spec transitions)
        model.alg = lay                                 # swap in the DISCOVERED layer
        tag = "+L* discovered"
    else:
        model = attach(base, spec); tag = "+attached"
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for p in model.base.parameters():                   # the base adds nothing here; freeze it
        p.requires_grad_(False)
    for e in range(300):
        x, y = gen(spec, 10, 256); opt.zero_grad()
        F.cross_entropy(model(x)[:, -1], y).backward(); opt.step()
    acc = []
    for T in (10, 40):
        x, y = gen(spec, T, 500)
        with torch.no_grad(): acc.append((model(x)[:, -1].argmax(1) == y).float().mean().item())
    print(f"{name:>10} {tag:>16} {sum(p.numel() for p in model.parameters()):>7}  {acc[0]:.3f}  {acc[1]:.3f}")

# (3a) steerable invariance on rotated MNIST
print("\n(3a) SteerableInvariant on rotated MNIST (train upright only)")
from mnist_loader import load_mnist
import math as _m
Xt, Yt, Xe, Ye = load_mnist("/tmp/opencode/mnist")
tr = torch.tensor(Xt[:6000]).float().reshape(-1,1,28,28)/255.; Ytr = torch.tensor(Yt[:6000]).long()
te = torch.tensor(Xe[:1000]).float().reshape(-1,1,28,28)/255.; Yte = torch.tensor(Ye[:1000]).long()
def rot(x, ang):
    t=_m.radians(ang); c,s=_m.cos(t),_m.sin(t)
    A=torch.tensor([[c,-s,0.],[s,c,0.]]).unsqueeze(0).expand(x.size(0),2,3)
    return F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), align_corners=False)
m = SteerableInvariant(out=10); opt = torch.optim.Adam(m.parameters(), 1e-3)
for e in range(6):
    p = torch.randperm(len(tr))
    for s in range(0, len(tr), 128):
        i = p[s:s+128]; opt.zero_grad(); F.cross_entropy(m(tr[i]), Ytr[i]).backward(); opt.step()
with torch.no_grad():
    a = [(m(rot(te, ang)).argmax(1) == Yte).float().mean().item() for ang in (0, 30, 45, 90)]
print("     acc @ 0/30/45/90 deg: " + "  ".join(f"{v:.3f}" for v in a))

# (3b) fuzzy automaton on a noisy counting task
print("\n(3b) FuzzyAutomaton on mod-5 counting with input noise (states superpose)")
K = 5
def gendata(T, n, noise):
    x = torch.randint(0, K, (n, T)); y = x.sum(1) % K
    if noise: x = torch.where(torch.rand_like(x.float()) < noise, torch.randint(0, K, x.shape), x)
    return x, y
class Crisp(nn.Module):
    def __init__(self, V=K, d=64):
        super().__init__(); self.emb = nn.Embedding(V, d); self.out = nn.Linear(d, V)
    def forward(self, x): return self.out(self.emb(x).mean(1))


for tag, make in [("crisp MLP", lambda: Crisp()),
                  ("fuzzy", lambda: FuzzyAutomaton(K, K, K))]:
    mm = make()
    opt = torch.optim.Adam(mm.parameters(), 3e-3)
    for e in range(500):
        x, y = gendata(12, 512, 0.0); opt.zero_grad(); F.cross_entropy(mm(x), y).backward(); opt.step()
    res = []
    for nz in (0.0, 0.1, 0.2):
        x, y = gendata(48, 1000, nz)
        with torch.no_grad(): res.append((mm(x).argmax(1) == y).float().mean().item())
    print(f"     {tag:>12}: acc @ noise 0/.1/.2: " + " ".join(f"{v:.3f}" for v in res))
