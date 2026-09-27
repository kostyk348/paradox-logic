"""
68 — Text: compositional semantics as a MONOID ACTION (perception + algebraic core).

Text: zero or more modifiers followed by a base word. Modifiers are functions composed
(not: x -> -x, very: x -> clamp(2x), little: x -> clamp(x//2)); the base sets the start.
The MEANING is the composition. Train with <=2 modifiers; test with 3-4 (compositional
generalisation).

  A: flat text net (embed + GRU)      -> memorises, fails on unseen counts
  B: perception (read each word) + EXACT composition of modifiers -> generalises
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)
CLAMP = lambda x: max(-6, min(6, x))
MODS = {"not": lambda x: -x, "very": lambda x: CLAMP(2 * x), "little": lambda x: CLAMP(x // 2)}
BASES = {"good": 2, "bad": -2, "ok": 0}
VOCAB = ["<pad>", "good", "bad", "ok", "not", "very", "little"]
stoi = {w: i for i, w in enumerate(VOCAB)}
SYMS = ["<pad>", "good", "bad", "ok", "not", "very", "little"]   # what perception can read
def value(words):
    v = None
    for w in words:
        if w in BASES: v = BASES[w]
    for w in words:
        if w in MODS and v is not None: v = MODS[w](v)
    return v if v is not None else 0
def label(v): return v + 6                                   # 13 classes (-6..6)


def gen(n, kmax, T=8):
    x = np.zeros((n, T), dtype=int); y = np.zeros(n, dtype=int)
    for i in range(n):
        k = rng.integers(0, kmax + 1)
        ws = [rng.choice(list(MODS)) for _ in range(k)]
        base = rng.choice(list(BASES))
        ws = [base] + ws[:T - 1]
        for j, w in enumerate(ws): x[i, j] = stoi[w]
        y[i] = label(value(ws))
    return torch.tensor(x), torch.tensor(y)


class Flat(nn.Module):                                          # A
    def __init__(self, d=64, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), d)
        self.rnn = nn.GRU(d, hid, batch_first=True); self.out = nn.Linear(hid, 13)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0][:, -1])


class PerceptAlgebra(nn.Module):                                # B
    def __init__(self, d=64):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), d); self.read = nn.Linear(d, len(SYMS))
    def forward(self, x): return self.read(self.emb(x))         # (B,T,6) symbol logits


def train_A(m, steps=600, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(256, 2); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()


def train_B(m, steps=400, lr=5e-3):
    tgt = {s: i for i, s in enumerate([stoi[w] for w in SYMS])}
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, _ = gen(256, 2); opt.zero_grad()
        logits = m(x)                                            # (B,T,6)
        target = torch.zeros_like(x)
        for j, w in enumerate(SYMS): target[x == stoi[w]] = j
        F.cross_entropy(logits.reshape(-1, len(SYMS)), target.reshape(-1)).backward(); opt.step()


def acc_A(m, kmax, n=500):
    x, y = gen(n, kmax)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def acc_B(m, kmax, n=500):
    x, y = gen(n, kmax)
    with torch.no_grad(): pred = m(x).argmax(-1).numpy()
    ok = 0
    for i in range(n):
        v = None
        for t in range(x.shape[1]):
            w = SYMS[pred[i, t]]
            if w == "<pad>": continue
            if w in BASES: v = BASES[w]
            elif v is not None: v = MODS[w](v)
        if label(v if v is not None else 0) == int(y[i]): ok += 1
    return ok / n


A = Flat(); train_A(A)
B = PerceptAlgebra(); train_B(B)
print("Text compositional semantics: perception + monoid of modifiers\n")
print(f"{'model':>34} {'params':>7}  k<=2   k<=3   k<=4")
for name, ma, mb in [("A: flat net", A, None), ("B: perception + algebra", None, B)]:
    if ma is not None:
        a = [acc_A(ma, k) for k in (2, 3, 4)]
        print(f"{name:>34} {sum(p.numel() for p in ma.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
    else:
        a = [acc_B(mb, k) for k in (2, 3, 4)]
        print(f"{name:>34} {sum(p.numel() for p in mb.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> the meaning of the phrase is a monoid homomorphism; reading the words (perception)")
print("   + composing the modifiers (algebra) generalises to unseen modifier counts.")
