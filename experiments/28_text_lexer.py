"""
28 — Text, done the algebraic way: a differentiable lexer on real source code.

A lexer is exactly a finite-state machine -> its monoid is finite and known. Corpus =
the repository's own source files. Label = lexer state per character (NORMAL / "..."/ '...'
/ # comment / escapes). Task: recover that state stream.

A per-character SemigroupRNN (learned transition table) is compared with an LSTM.
The regular structure is exact, so the semigroup model should generalise to long windows;
the LSTM has to learn the same automaton from data.
"""
import sys, glob, os
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# ---------- corpus ----------
files = []
for pat in ("/home/lain/paradox-logic/*.py", "/home/lain/paradox-logic/*.c",
            "/home/lain/paradox-logic/experiments/*.py"):
    files += glob.glob(pat)
text = ""
for f in files:
    try: text += open(f, encoding="utf-8", errors="ignore").read() + "\n"
    except Exception: pass
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
print(f"corpus: {len(text)} chars, vocab {len(chars)}")

# ---------- reference lexer: 6 states ----------
NORMAL, DQ, SQ, COMMENT, ESC_DQ, ESC_SQ = range(6)

def lexer(seq):
    st = NORMAL; out = []
    for c in seq:
        if st == NORMAL:
            if c == '"': st = DQ
            elif c == "'": st = SQ
            elif c == '#': st = COMMENT
        elif st == DQ:
            if c == '\\': st = ESC_DQ
            elif c == '"': st = NORMAL
        elif st == SQ:
            if c == '\\': st = ESC_SQ
            elif c == "'": st = NORMAL
        elif st == COMMENT:
            if c == '\n': st = NORMAL
        elif st == ESC_DQ: st = DQ
        elif st == ESC_SQ: st = SQ
        out.append(st)
    return out

data = text[:200000]
X = torch.tensor([stoi[c] for c in data]); Yl = torch.tensor(lexer(data))
print("state distribution:", torch.bincount(Yl, minlength=6).tolist())

NV, NS = len(chars), 6

class SemigroupLexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(NV, NS, NS) * 0.3)
        self.out = nn.Linear(NS, NS)
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, NS); p[:, NORMAL] = 1.0
        q = torch.softmax(self.W, -1)
        outs = []
        for t in range(T):
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
            outs.append(self.out(p))
        return torch.stack(outs, 1)

class LSTMLexer(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.emb = nn.Embedding(NV, hid); self.rnn = nn.LSTM(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, NS)
    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])

def windows(T, n):
    i = torch.randint(0, len(data) - T - 1, (n,))
    xs = torch.stack([X[j:j+T] for j in i]); ys = torch.stack([Yl[j:j+T] for j in i])
    return xs, ys

def seqacc(model, T, n=200):
    xs, ys = windows(T, n)
    with torch.no_grad(): return (model(xs).argmax(-1) == ys).float().mean().item()


def hard_seqacc(m, T, n=200):
    """run the HARD learned automaton: exact transition table, classify via readout."""
    tab = torch.softmax(m.W.detach(), -1).argmax(-1)          # (NV, NS) -> next state
    xs, ys = windows(T, n); ok = tot = 0
    with torch.no_grad():
        for s in range(len(xs)):
            st = NORMAL
            for t in range(T):
                st = int(tab[int(xs[s, t]), st])
                oh = torch.zeros(1, NS); oh[0, st] = 1.0
                if m.out(oh).argmax(1).item() == int(ys[s, t]): ok += 1
                tot += 1
    return ok / tot


print("\nper-character lexer-state accuracy (train windows T=64)\n")
print(f"{'model':>14} {'params':>7}  soft T64/256/1024     hard T64/256/1024")
for name, m in [("SemigroupRNN", SemigroupLexer()), ("LSTM", LSTMLexer())]:
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for ep in range(8):
        for _ in range(60):
            xs, ys = windows(64, 128)
            opt.zero_grad(); F.cross_entropy(m(xs).reshape(-1, NS), ys.reshape(-1)).backward(); opt.step()
    soft = [seqacc(m, T) for T in (64, 256, 1024)]
    hard = [hard_seqacc(m, T) for T in (64, 256, 1024)] if name == "SemigroupRNN" else [float("nan")] * 3
    print(f"{name:>14} {sum(p.numel() for p in m.parameters()):>7}  " +
          " ".join(f"{a:.3f}" for a in soft) + "   " +
          " ".join("--" if a != a else f"{a:.3f}" for a in hard))

# show the recovered transition for a few key characters
m = SemigroupLexer()
print("\n(learned table is read as: for this char, which state -> which state)")

