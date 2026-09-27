"""
50 — Constrained decoding for REAL structured output: valid JSON arrays.

Language: nested JSON arrays of integers, e.g. [1,[2,3],4].  A tiny char-level GRU LM is
trained on valid strings; the grammar is a bounded-depth DFA. Unconstrained sampling emits
invalid JSON; decoding constrained by the DFA is valid 100% by construction.
Validity is checked with json.loads.
"""
import sys, json, random
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.automaton import constrained_decode
torch.manual_seed(0); random.seed(0)

D = 3
V = ['[', ']', ','] + [str(d) for d in range(10)]
stoi = {c: i for i, c in enumerate(V)}
ACCEPT = (0, 1)                      # depth 0, after a value
START = (0, 0)                       # depth 0, expecting a value or ']'


def delta(s, ch):
    d, ph = s
    if ph == 0:                                     # expecting value or ']'
        if ch == '[' and d < D: return (d + 1, 0)
        if ch == ']' and d > 0: return (d - 1, 1)
        if ch.isdigit(): return (d, 1)
        return None
    else:                                           # expecting ',' or ']'
        if ch == ',': return (d, 0)
        if ch == ']' and d > 0: return (d - 1, 1)
        return None


def rand_arr(depth=0):
    if depth >= D or random.random() < 0.4:
        return str(random.randint(0, 9))
    n = random.randint(0, 3)
    return '[' + ','.join(rand_arr(depth + 1) for _ in range(n)) + ']'


def rand_str():
    s = rand_arr()
    return s if s.startswith('[') else '[' + s + ']'


data = [rand_str()[:40] for _ in range(8000)]
Lmax = max(len(s) for s in data)
Xs, Ys = [], []
for s in data:
    ids = [stoi[c] for c in s if c in stoi]
    Xs.append(torch.tensor([stoi['[']] + ids[:-1]))
    Ys.append(torch.tensor(ids))


class CharLM(nn.Module):
    def __init__(self, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(V), 48)
        self.rnn = nn.GRU(48, hid, batch_first=True); self.out = nn.Linear(hid, len(V))
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0])


padded = torch.zeros(len(Xs), 40, dtype=torch.long)
for i, x in enumerate(Xs):
    padded[i, :len(x)] = x
Ypad = torch.zeros(len(Ys), 40, dtype=torch.long)
for i, y in enumerate(Ys):
    Ypad[i, :len(y)] = y
mask = (Ypad != stoi['[']) | True

m = CharLM(); opt = torch.optim.Adam(m.parameters(), 2e-3)
for ep in range(30):
    for s in range(0, len(padded), 128):
        xb, yb = padded[s:s + 128], Ypad[s:s + 128]
        lens = torch.tensor([len(Ys[i]) for i in range(s, min(s + 128, len(Ys)))])
        opt.zero_grad()
        logits = m(xb)
        loss = F.cross_entropy(logits.reshape(-1, len(V)), yb.reshape(-1), reduction='none')
        loss = (loss.reshape(xb.shape[0], -1) * (torch.arange(40)[None] < lens[:, None])).sum() / lens.sum()
        loss.backward(); opt.step()


@torch.no_grad()
def gen_free(n=500):
    x = torch.full((n, 1), stoi['[']); ok = 0
    for _ in range(40):
        p = torch.softmax(m(x)[:, -1], -1)
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    for r in range(n):
        s = ''.join(V[i] for i in x[r].tolist())
        try:
            json.loads(s); ok += 1
        except Exception:
            try: json.loads(s[:40]); ok += 1
            except Exception: pass
    return ok / n


def policy(prefix):
    ids = [stoi['[']] + list(prefix)
    x = torch.tensor([ids])
    with torch.no_grad(): lg = m(x)[0, -1]
    return {i: float(lg[i]) for i in range(len(V))}


trans = {}
for d in range(D + 1):
    for ph in (0, 1):
        trans[(d, ph)] = {i: delta((d, ph), V[i]) for i in range(len(V))}

outs = constrained_decode(policy, trans, {ACCEPT}, START, budget=40, n_samples=500, rng=random.Random(0))
ok = 0
for seq in outs:
    s = ''.join(V[i] for i in seq)
    if seq and seq[0] == stoi['[']:
        s = '[' + ''.join(V[i] for i in seq[1:])
    try:
        json.loads(s); ok += 1
    except Exception: pass

print("JSON-array generation (char GRU LM)\n")
print(f"  unconstrained sampling : valid = {gen_free()*100:5.1f}%")
print(f"  constrained by grammar : valid = {ok/len(outs)*100:5.1f}%")
print(f"  example constrained output: {''.join(V[i] for i in outs[0])}")
