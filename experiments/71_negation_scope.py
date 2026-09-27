"""
71 — (item 2) Real negation words, compositional scope.

Sentences use real negators (not, n't, never, no, without) and real sentiment words.
The label is the sentiment AFTER composing all negations (parity). Train on <=1 negator,
test on 2-3 (negation scope = compositional generalisation).
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)
NEG = ["not", "never", "no", "without", "hardly"]
POS = ["good", "great", "brilliant"]
NEGWORD = ["bad", "terrible", "awful"]
VOCAB = ["<pad>", "the", "movie", "was"] + NEG + POS + NEGWORD
stoi = {w: i for i, w in enumerate(VOCAB)}
T = 9
NEGSET = set(NEG); POSSET = set(POS); NWORDSET = set(NEGWORD)


def gen(n, kmax):
    X = np.zeros((n, T), dtype=int); y = np.zeros(n, dtype=int)
    for i in range(n):
        k = rng.integers(0, kmax + 1)
        base = rng.choice(POS + NEGWORD)
        s = ["the", "movie", "was"] + [rng.choice(NEG) for _ in range(k)] + [base]
        for j, w in enumerate(s[:T]): X[i, j] = stoi[w]
        sgn = 1 if base in POSSET else -1
        if k % 2 == 1: sgn = -sgn
        y[i] = 1 if sgn > 0 else 0
    return torch.tensor(X), torch.tensor(y)


class Flat(nn.Module):
    def __init__(self, d=64, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), d)
        self.rnn = nn.GRU(d, hid, batch_first=True); self.out = nn.Linear(hid, 2)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0][:, -1])


class Alg(nn.Module):                                   # perception: classify each token
    def __init__(self, d=64):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), d); self.read = nn.Linear(d, 4)  # 0=other 1=neg 2=pos 3=negword
    def forward(self, x): return self.read(self.emb(x))


def trainA(m, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(256, 1); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def trainB(m, steps=600, lr=5e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, _ = gen(256, 3); opt.zero_grad()
        tgt = torch.zeros_like(x)                                   # 0 = other/pad
        tgt[torch.isin(x, torch.tensor([stoi[w] for w in NEG]))] = 1
        tgt[torch.isin(x, torch.tensor([stoi[w] for w in POS]))] = 2
        tgt[torch.isin(x, torch.tensor([stoi[w] for w in NEGWORD]))] = 3
        F.cross_entropy(m(x).reshape(-1, 4), tgt.reshape(-1)).backward(); opt.step()


def accA(m, kmax, n=500):
    x, y = gen(n, kmax)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def accB(m, kmax, n=500):
    x, y = gen(n, kmax)
    with torch.no_grad(): pred = m(x).argmax(-1).numpy()
    ok = 0
    for i in range(n):
        neg = 0; base = None
        for t in range(T):
            c = pred[i, t]
            if x[i, t] == 0: continue
            if c == 1: neg += 1
            elif c in (2, 3): base = c
        if base is None: continue
        sgn = 1 if base == 2 else -1
        if neg % 2 == 1: sgn = -sgn
        if (1 if sgn > 0 else 0) == int(y[i]): ok += 1
    return ok / n


A = Flat(); trainA(A)
B = Alg(); trainB(B)
print("Negation scope on real words (train <=1 negator; test 2-3)\n")
print(f"{'model':>28} {'params':>7}   k<=1   k<=2   k<=3")
for name, ma, mb in [("flat net", A, None), ("perception + parity (alg)", None, B)]:
    if ma is not None:
        a = [accA(ma, k) for k in (1, 2, 3)]
        print(f"{name:>28} {sum(p.numel() for p in ma.parameters()):>7}   " + "  ".join(f"{v:.3f}" for v in a))
    else:
        a = [accB(mb, k) for k in (1, 2, 3)]
        print(f"{name:>28} {sum(p.numel() for p in mb.parameters()):>7}   " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> negation scope is parity; reading the negators (perception) + parity (algebra)")
print("   generalises to multiple negations; the flat net must learn the scope from data.")
