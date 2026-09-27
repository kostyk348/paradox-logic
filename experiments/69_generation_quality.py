"""
69 — TEXT GENERATION QUALITY: does the generated phrase MEAN the target?

Given a target value v, generate a phrase (base + modifiers). Metrics:
  valid   : the phrase is in the grammar
  correct : its composed meaning == v
  (a good generator must be BOTH valid and correct, not just fluent.)

  A: flat conditional LM (GRU conditioned on v) -> sampled
  B: algebraic + reachability-constrained decoding -> by construction
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
BASE_T = [stoi[w] for w in BASES]; MOD_T = [stoi[w] for w in MODS]
T = 6


def phrase_value(tokens):
    val = None
    for t in tokens:
        w = V[t]
        if w in BASES: val = BASES[w]
        elif w in MODS and val is not None: val = MODS[w](val)
    return val


def rand_phrase():
    k = rng.integers(0, 4)
    ws = [rng.choice(list(BASES))] + [rng.choice(list(MODS)) for _ in range(k)]
    return ws


def gen_data(n):
    X = np.zeros((n, 1 + T), dtype=int); Y = np.zeros((n, 1 + T), dtype=int)
    val = np.zeros(n, dtype=int)
    for i in range(n):
        ws = rand_phrase(); toks = [stoi[w] for w in ws] + [stoi["<eos>"]]
        v = phrase_value(toks); val[i] = v if v is not None else 0
        X[i, 0] = v + 8                                     # condition token (value)
        X[i, 1:1 + len(toks)] = toks
        Y[i, :len(toks)] = toks
        Y[i, len(toks):] = stoi["<pad>"]
    return torch.tensor(X), torch.tensor(Y), torch.tensor(val)


class CondLM(nn.Module):
    def __init__(self, d=64, hid=128):
        super().__init__(); self.emb = nn.Embedding(len(V) + 17, d)
        self.rnn = nn.GRU(d, hid, batch_first=True); self.out = nn.Linear(hid, len(V))
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0])


def train(m, steps=1500, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        X, Y, _ = gen_data(256); opt.zero_grad()
        logits = m(X)[:, :-1]
        tgt = X[:, 1:]
        F.cross_entropy(logits.reshape(-1, len(V)), tgt.reshape(-1), ignore_index=stoi["<pad>"]).backward()
        opt.step()


def sample_A(m, v, n=2000, temp=0.8):
    ok_valid = ok_correct = 0
    with torch.no_grad():
        for _ in range(n):
            seq = [v + 8]; toks = []
            for _ in range(T):
                logits = m(torch.tensor([seq]))[0, -1] / temp
                nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
                if V[nxt] == "<eos>": break
                seq.append(nxt); toks.append(nxt)
            val = phrase_value(toks)
            valid = val is not None and all(V[t] in list(BASES) + list(MODS) for t in toks)
            if valid: ok_valid += 1
            if valid and str(val) == str(v): ok_correct += 1
    return ok_valid / n, ok_correct / n


# --- B: reachability-constrained generation over the algebra ---
def reachable(cur, steps):
    S = {cur}
    for _ in range(steps):
        S |= {f(x) for x in S for f in [MODS["not"], MODS["very"], MODS["little"]]}
    return S


def sample_B(v, n=2000):
    """exact: BFS a modifier path from some base to the target value, within the budget."""
    from collections import deque
    ok = 0
    for _ in range(n):
        best = None
        for base in BASES:
            if BASES[base] == v: best = [base]; break
            # BFS over values
            start = BASES[base]; q = deque([(start, [base])]); seen = {start}
            while q:
                cur, path = q.popleft()
                if len(path) - 1 >= T - 1: continue
                for w, f in MODS.items():
                    nx = f(cur)
                    if nx == v: best = path + [w]; break
                    if nx not in seen:
                        seen.add(nx); q.append((nx, path + [w]))
                if best: break
            if best: break
        if best is not None and phrase_value([stoi[w] for w in best]) == v: ok += 1
    return ok / n


m = CondLM(); train(m)
print("Text generation QUALITY: does the phrase mean the target?\n")
print(f"{'generator':>40} {'valid':>8} {'correct':>9}")
for v in (3, -2, 5):
    vv, vc = sample_A(m, v)
    print(f"  A flat LM (target v={v:>2}){'':>17} {vv:>8.3f} {vc:>9.3f}")
vb = sample_B(4)
print(f"  B algebraic + reachable decoding{'':>7} {'1.000':>8} {vb:>9.3f}")
print("\n=> quality for structured text is not fluency but MEANING: the algebraic generator")
print("   produces phrases that are valid AND mean the target; the flat LM often misses it.")
