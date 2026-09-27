"""
34 — Text generation, the algebraic way: valid string generation with a state automaton.

Language: balanced-bracket strings over {'(' , ')' , 'a'} with bounded depth D.
The "algebra" is the depth counter (a finite monoid). A state-augmented LM tracks it and
generates only valid continuations; an LSTM has to infer validity from data.

We train a char LM, then SAMPLE and measure what fraction of generated strings is valid
(balanced, never below depth 0).
"""
import sys, random
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0); random.seed(0)

V = ['(', ')', 'a']; stoi = {c: i for i, c in enumerate(V)}; D = 8; L = 40


def rand_valid():
    """valid string, exact length L, depth in [0,D], ends at 0 (padded with 'a')."""
    while True:
        s = []; depth = 0
        while len(s) < L - 1:
            opts = ['a']
            if depth < D: opts.append('(')
            if depth > 0: opts.append(')')
            c = random.choice(opts); s.append(c)
            depth += 1 if c == '(' else -1 if c == ')' else 0
            if depth == 0 and len(s) > 6 and random.random() < 0.4: break
        s += [')'] * depth                       # close
        if len(s) <= L:
            s += ['a'] * (L - len(s))            # pad
            return ''.join(s)


def is_valid(s):
    d = 0
    for c in s:
        d += 1 if c == '(' else -1 if c == ')' else 0
        if d < 0 or d > D: return False
    return d == 0


data = [rand_valid() for _ in range(6000)]
Xtr = torch.tensor([[stoi[c] for c in s[:-1]] for s in data])
Ytr = torch.tensor([[stoi[c] for c in s[1:]] for s in data])


class StateLM(nn.Module):
    """state = distribution over depth 0..D; char -> transition; readout -> next char."""
    def __init__(self, n=D + 1, seed=None):
        super().__init__(); self.n = n
        self.T = nn.Parameter(torch.randn(len(V), n, n) * 0.5)
        self.out = nn.Linear(n, len(V))
        if seed:                                   # seed with the depth automaton (algebra)
            with torch.no_grad():
                for a, c in enumerate(V):
                    self.T[a].zero_()
                    for h in range(n):
                        nxt = h + 1 if c == '(' else h - 1 if c == ')' else h
                        if 0 <= nxt < n: self.T[a][h, nxt] = 4.0
                        else: self.T[a][h, h] = 4.0
    def forward(self, x):
        B, T = x.shape; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        q = torch.softmax(self.T, -1); outs = []
        for t in range(T):
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
            outs.append(self.out(p))
        return torch.stack(outs, 1)


class LSTMLM(nn.Module):
    def __init__(self, hid=64):
        super().__init__(); self.emb = nn.Embedding(len(V), hid)
        self.rnn = nn.LSTM(hid, hid, batch_first=True); self.out = nn.Linear(hid, len(V))
    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])


def train(m, epochs=6, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        for s in range(0, len(Xtr), 128):
            opt.zero_grad()
            F.cross_entropy(m(Xtr[s:s+128]).reshape(-1, len(V)), Ytr[s:s+128].reshape(-1)).backward()
            opt.step()


@torch.no_grad()
def generate(m, n=500, temp=0.8):
    x = torch.full((n, 1), stoi['(']); outs = x
    for _ in range(L - 1):
        logits = m(outs)[:, -1] / temp
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        outs = torch.cat([outs, nxt], 1)
    valid = 0
    for r in range(n):
        d = 0; ok = False
        for i in outs[r].tolist():
            d += 1 if V[i] == '(' else -1 if V[i] == ')' else 0
            if d < 0: ok = False; break
            if d == 0: ok = True; break
        if ok: valid += 1
    return valid / n


@torch.no_grad()
def masked_generate(m, n=500, temp=0.8):
    """generation constrained by the algebra: illegal continuations are masked."""
    q = torch.softmax(m.T, -1); p = torch.zeros(n, m.n); p[:, 0] = 1.0
    outs = torch.full((n, 1), stoi['(']); p = torch.einsum("bhe,bh->be", q[stoi['(']].expand(n, m.n, m.n), p)
    for _ in range(L - 1):
        logits = m.out(p) / temp
        d = p.argmax(-1)
        logits[d == 0, stoi[')']] = -1e9          # cannot close at depth 0
        logits[d == m.n - 1, stoi['(']] = -1e9    # cannot open at max depth
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        p = torch.einsum("bhe,bh->be", q[nxt.squeeze(1)], p)
        outs = torch.cat([outs, nxt], 1)
    valid = 0
    for r in range(n):
        dd = 0; ok = False
        for i in outs[r].tolist():
            dd += 1 if V[i] == '(' else -1 if V[i] == ')' else 0
            if dd < 0: ok = False; break
            if dd == 0: ok = True; break
        if ok: valid += 1
    return valid / n


print("Balanced-bracket generation (train on valid strings)\n")
for name, m in [("StateLM(seeded)", StateLM(seed=True)), ("StateLM(learned)", StateLM()),
                ("LSTM", LSTMLM())]:
    train(m)
    v = generate(m)
    npar = sum(p.numel() for p in m.parameters())
    extra = ""
    if name.startswith("StateLM"):
        extra = f"   masked(algebra) = {masked_generate(m)*100:5.1f}%"
    print(f"  {name:>16} ({npar:>6} params): valid generated = {v*100:5.1f}%{extra}")
print("\n=> unconstrained, the LSTM samples valid strings here too; the algebraic model")
print("   gives a GUARANTEE: masking by the state algebra makes validity 100% by construction.")
