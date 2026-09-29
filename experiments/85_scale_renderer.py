"""
85 — EXTEND the algorithm (interpolated orders 1..8) and SCALE the renderer (an LSTM).

Algorithm: recursive Jelinek-Mercer interpolation of char orders 1..8 (exact counts).
Renderer : an LSTM whose input is emb(char) (+ a learned projection of the algorithmic log-probs),
           output = a RESIDUAL on the algorithmic logits:  p = softmax(log p_alg + f_net(h_t)).

  (a) algorithm alone
  (b) LSTM alone (no algorithmic logits)
  (c) LSTM + algorithmic logits   <- "the net only writes the result"
"""
import sys, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)

text = open("/tmp/opencode/shakespeare.txt").read()
text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], np.int64)
K = 4; TR = data[:260000]; TE = data[260000:320000]
D = 0.9                                     # PPM-C discount

tabs = [collections.defaultdict(collections.Counter) for _ in range(K + 1)]
for i in range(K, len(TR)):
    for k in range(1, K + 1):
        tabs[k][tuple(TR[i - k:i])][TR[i]] += 1
UNI = collections.Counter(TR); tot = sum(UNI.values())
uni = (np.array([UNI.get(c, 0) + 1 for c in range(V)], np.float64) / (tot + V))


def alg_p(ctx):
    """PPM-C: longest matching context first, with escape down to the uniform order-0."""
    p = np.zeros(V, np.float64); mass = 1.0
    for k in range(min(K, len(ctx)), 0, -1):
        c = tabs[k].get(tuple(ctx[-k:]))
        if not c: continue
        s = sum(c.values()); esc = D * len(c) / s
        for ch, cnt in c.items(): p[ch] += mass * (cnt - D) / s
        mass *= esc
        if mass < 1e-7: break
    p += mass * uni
    return p


def precompute(d):
    """P[j] = the algorithm's log-prob for the character AT position j, from d[j-K:j] (past only).
    At input position t the renderer is fed P[t+1]: the logit for the TARGET (no future leak,
    since P[t+1] uses only d[t+1-K : t+1], i.e. x up to and including t)."""
    P = np.tile(np.log(uni + 1e-12).astype(np.float32), (len(d), 1))
    for j in range(K, len(d)):
        P[j] = np.log(alg_p(d[j - K:j]) + 1e-12)
    return P


print("algorithm: PPM-C (orders 1..4, discount 0.9) ...")
Ptr = precompute(TR); Pte = precompute(TE)
Ytr = torch.tensor(TR[K:]); Yte = torch.tensor(TE[K:])
alg = float(-Pte[K:][np.arange(len(TE) - K), TE[K:]].mean() / math.log(2))
print(f"(a) algorithm alone                         bits/char {alg:.3f}\n")

Ptr_t = torch.tensor(Ptr); Pte_t = torch.tensor(Pte)


class Rend(nn.Module):
    def __init__(self, h=256, use_alg=True):
        super().__init__(); self.use_alg = use_alg
        self.e = nn.Embedding(V, 64)
        self.proj = nn.Linear(V, 64) if use_alg else None
        self.l = nn.LSTM(64 * (2 if use_alg else 1), h, 1, batch_first=True)
        self.o = nn.Linear(h, V)
    def forward(self, x, palg=None):
        z = [self.e(x)]
        if self.use_alg: z.append(self.proj(palg))
        return self.o(self.l(torch.cat(z, -1))[0])


def run(use_alg, epochs=4):
    m = Rend(use_alg=use_alg); opt = torch.optim.Adam(m.parameters(), 2e-3)
    L = 128
    for ep in range(epochs):
        idx = torch.randperm(len(Ytr) - L - 1)
        for s in range(0, len(idx[:12000]), 64):
            i = idx[s:s + 64]
            xs = torch.stack([torch.arange(t, t + L) for t in i])
            x = torch.tensor(TR)[xs]; y = torch.tensor(TR)[xs + 1]
            pa = Ptr_t[xs + 1]                      # aligned to the TARGET
            opt.zero_grad()
            r = m(x, pa if use_alg else None)
            logits = r + pa if use_alg else r
            F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        i = torch.stack([torch.arange(t, t + 128) for t in range(0, 4000, 64)])
        x = torch.tensor(TE)[i]; y = torch.tensor(TE)[i + 1]; pa = Pte_t[i + 1]
        logits = m(x, pa if use_alg else None)
        if use_alg: logits = logits + pa
        return float(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)) / math.log(2))


print(f"(b) LSTM alone (scaled renderer)             bits/char {run(False):.3f}")
print(f"(c) LSTM + algorithmic logits                bits/char {run(True):.3f}")
print(f"(a) algorithm alone                          bits/char {alg:.3f}")
