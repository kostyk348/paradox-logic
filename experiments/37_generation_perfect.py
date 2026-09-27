"""
37 — Generation, done perfectly: automaton-constrained decoding.

Language: over {'(' , ')' , 'a'} valid iff balanced (depth in [0,D], ends 0) AND the number
of 'a' is EVEN  -> state = (depth, parity) = a wreath product (aperiodic x Z/2).

An LSTM LM is trained on valid strings; unconstrained sampling leaks invalid strings.
Constrained decoding (only symbols from which ACCEPT is still reachable within the budget)
yields valid strings 100% BY CONSTRUCTION.
"""
import sys, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0)

V = ['(', ')', 'a']; stoi = {c: i for i, c in enumerate(V)}; D = 6; L = 48
NA = 2 * (D + 1)                                   # states (depth, parity)


def enc(d, p): return d * 2 + p
def delta(s, c):
    d, p = divmod(s, 2)
    if c == '(':
        if d + 1 > D: return None
        return enc(d + 1, p)
    if c == ')':
        if d == 0: return None
        return enc(d - 1, p)
    return enc(d, 1 - p)                            # 'a' flips parity
ACCEPT = enc(0, 0)


def valid(s):
    d = p = 0
    for c in s:
        if c == '(':
            d += 1
            if d > D: return False
        elif c == ')':
            d -= 1
            if d < 0: return False
        else:
            p ^= 1
    return d == 0 and p == 0


# reachability: reach[s][r] = accept reachable from s within r steps
reach = [[False] * (L + 1) for _ in range(NA)]
for r in range(L + 1): reach[ACCEPT][r] = True
for r in range(1, L + 1):
    for s in range(NA):
        for c in V:
            n = delta(s, c)
            if n is not None and reach[n][r - 1]: reach[s][r] = True


def rand_valid():
    while True:
        s = []; st = ACCEPT
        for _ in range(L):
            opts = [c for c in V if delta(st, c) is not None]
            c = random.choice(opts); s.append(c); st = delta(st, c)
            if st == ACCEPT and len(s) > 6 and random.random() < 0.35: break
        if valid(''.join(s)) and len(s) >= L - 1:
            return ''.join(s)


data = [rand_valid() for _ in range(8000)]
Xtr = torch.tensor([[stoi[c] for c in s[:-1]] for s in data])
Ytr = torch.tensor([[stoi[c] for c in s[1:]] for s in data])


class LSTMLM(nn.Module):
    def __init__(self, hid=64):
        super().__init__(); self.emb = nn.Embedding(len(V), hid)
        self.rnn = nn.LSTM(hid, hid, batch_first=True); self.out = nn.Linear(hid, len(V))
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0])


m = LSTMLM(); opt = torch.optim.Adam(m.parameters(), 3e-3)
for ep in range(8):
    for s in range(0, len(Xtr), 128):
        opt.zero_grad()
        F.cross_entropy(m(Xtr[s:s+128]).reshape(-1, len(V)), Ytr[s:s+128].reshape(-1)).backward(); opt.step()


@torch.no_grad()
def gen_free(n=500, temp=0.8):
    x = torch.full((n, 1), stoi['('])
    for _ in range(L - 1):
        p = torch.softmax(m(x)[:, -1] / temp, -1)
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    return sum(valid(''.join(V[i] for i in x[r].tolist())) for r in range(n)) / n


@torch.no_grad()
def gen_constrained(n=500, temp=0.8):
    ok = 0
    for _ in range(n):
        seq = [stoi['(']]; st = delta(ACCEPT, '(')
        for step in range(1, L):
            rem = L - step - 1
            logits = m(torch.tensor([seq]))[0, -1] / temp
            mask = torch.full((len(V),), -1e9)
            for ci, c in enumerate(V):
                nx = delta(st, c)
                if nx is not None and reach[nx][rem]:
                    mask[ci] = logits[ci]
            if float(mask.max()) < -1e8:                              # forced
                ci = stoi[')'] if delta(st, ')') is not None else stoi['a']
            else:
                ci = int(torch.multinomial(torch.softmax(mask, -1), 1))
            seq.append(ci); st = delta(st, V[ci])
            if st == ACCEPT:
                break
        if valid(''.join(V[i] for i in seq)):
            ok += 1
    return ok / n


print("Language: balanced brackets AND even #a   (state = depth x parity, 14 states)\n")
print(f"  LSTM unconstrained      : valid = {gen_free()*100:5.1f}%")
print(f"  LSTM + automaton-decode : valid = {gen_constrained()*100:5.1f}%")
print("\n=> constrained decoding by the grammar gives 100% valid generation BY CONSTRUCTION;")
print("   the model supplies the distribution, the algebra supplies the guarantee.")
