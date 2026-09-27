"""
70 — (item 1) Fluency + guaranteed meaning, and READ the sentences.

A flat LM generates a sentence for a target value (fluent, but often wrong meaning). The
algebraic corrector (BFS to the target over the modifier monoid) fixes the meaning while
keeping the sentence well-formed. Below we print the actual sentences.
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
rng = np.random.default_rng(0)
CL = lambda x: max(-8, min(8, x))
MODS = {"not": lambda x: -x, "very": lambda x: CL(2 * x), "little": lambda x: CL(x // 2)}
BASES = {"good": 2, "bad": -2, "ok": 0}
V = ["<pad>", "good", "bad", "ok", "not", "very", "little", "<eos>"]
stoi = {w: i for i, w in enumerate(V)}
T = 6


def phrase_value(words):
    val = None
    for w in words:
        if w in BASES: val = BASES[w]
        elif w in MODS and val is not None: val = MODS[w](val)
    return val


def rand_words():
    return [rng.choice(list(BASES))] + [rng.choice(list(MODS)) for _ in range(rng.integers(0, 4))]


def gen_data(n):
    X = np.zeros((n, 1 + T), dtype=int)
    for i in range(n):
        ws = rand_words(); v = phrase_value(ws)
        toks = [stoi[w] for w in ws] + [stoi["<eos>"]]
        X[i, 0] = v + 8; X[i, 1:1 + len(toks)] = toks
    return torch.tensor(X)


class CondLM(nn.Module):
    def __init__(self, d=64, hid=128):
        super().__init__(); self.emb = nn.Embedding(len(V) + 17, d)
        self.rnn = nn.GRU(d, hid, batch_first=True); self.out = nn.Linear(hid, len(V))
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0])


def train(m, steps=1500, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        X = gen_data(256); opt.zero_grad()
        F.cross_entropy(m(X)[:, :-1].reshape(-1, len(V)), X[:, 1:].reshape(-1),
                        ignore_index=stoi["<pad>"]).backward(); opt.step()


def sample_LM(m, v, temp=0.8):
    seq = [v + 8]; words = []
    with torch.no_grad():
        for _ in range(T):
            p = torch.softmax(m(torch.tensor([seq]))[0, -1] / temp, -1)
            nxt = int(torch.multinomial(p, 1))
            if V[nxt] == "<eos>": break
            seq.append(nxt); words.append(V[nxt])
    return words


def correct(v, budget=T - 1):
    from collections import deque
    for base in BASES:
        if BASES[base] == v: return [base]
        q = deque([(BASES[base], [base])]); seen = {BASES[base]}
        while q:
            cur, path = q.popleft()
            if len(path) - 1 >= budget: continue
            for w, f in MODS.items():
                nx = f(cur)
                if nx == v: return path + [w]
                if nx not in seen: seen.add(nx); q.append((nx, path + [w]))
    return None


def sentence(words): return "it is " + " ".join(words)

m = CondLM(); train(m)
print("Fluency + guaranteed meaning (reading the sentences)\n")
for v in (4, -2, 2, -4, 1):
    print(f"target value = {v}")
    for _ in range(3):
        lm = sample_LM(m, v)
        lmv = phrase_value(lm)
        mark = "OK " if lmv == v else "BAD"
        print(f"   LM   : '{sentence(lm)}'  (value {lmv})  {mark}")
    c = correct(v)
    if c is None:
        print("   fixed: (target unreachable in this monoid)\n")
    else:
        print(f"   fixed: '{sentence(c)}'  (value {phrase_value(c)})  OK\n")
print("=> the LM is fluent but often wrong; the algebraic corrector gives the target meaning")
print("   every time, and the sentence stays well-formed.")
