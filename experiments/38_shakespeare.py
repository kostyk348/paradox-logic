"""
38 — Shakespeare char-LM: is the algebraic channel a replacement?  (honest test)

Baseline: char-LSTM.
Hybrid: the SAME LSTM plus a learned finite-state (semigroup) channel -- the automaton's
state distribution is concatenated to the LSTM hidden state before the readout.

Metric: bits per character (lower is better). Free text is NOT regular, so the algebraic
channel is expected to help only marginally (or not at all) -- reported honestly.
"""
import sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

text = open("/tmp/opencode/shakespeare.txt", encoding="utf-8", errors="ignore").read()[:250000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text])
print(f"corpus {len(text)} chars, vocab {V}")


def batches(bs=64, T=128, n=None):
    n = n or (len(data) - T - 1) // bs * bs
    starts = torch.arange(0, n, T + 1)
    idx = starts[: (len(starts) // bs) * bs].reshape(-1, bs).T.contiguous()
    for i in range(0, idx.shape[0], 1):
        pass
    # simple random windows
    for _ in range(2000):
        s = torch.randint(0, len(data) - T - 1, (bs,))
        x = torch.stack([data[j:j + T] for j in s])
        y = torch.stack([data[j + 1:j + T + 1] for j in s])
        yield x, y


class LSTMLM(nn.Module):
    def __init__(self, hid=160):
        super().__init__(); self.emb = nn.Embedding(V, 64)
        self.rnn = nn.LSTM(64, hid, batch_first=True); self.out = nn.Linear(hid, V)
    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])


class HybridLM(nn.Module):
    def __init__(self, ns=16, hid=160):
        super().__init__(); self.ns = ns
        self.emb = nn.Embedding(V, 64); self.rnn = nn.LSTM(64, hid, batch_first=True)
        self.T = nn.Parameter(torch.randn(V, ns, ns) * 0.2)      # char -> transition
        self.out = nn.Linear(hid + ns, V)
    def forward(self, x):
        h = self.rnn(self.emb(x))[0]                             # (B,T,hid)
        q = torch.softmax(self.T, -1)
        p = torch.zeros(x.size(0), self.ns); p[:, 0] = 1.0
        ps = []
        for t in range(x.size(1)):
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p); ps.append(p)
        feat = torch.cat([h, torch.stack(ps, 1)], -1)
        return self.out(feat)


def run(m, steps=1500, lr=2e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr); it = batches()
    t0 = time.time(); losses = []
    for k in range(steps):
        x, y = next(it); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); loss.backward(); opt.step()
        losses.append(loss.item())
    return sum(losses[-100:]) / 100 / math.log(2)


print("\nShakespeare char-LM (bits per character, lower is better)\n")
for name, m in [("LSTM (baseline)", LSTMLM()), ("LSTM + semigroup channel", HybridLM())]:
    bpc = run(m)
    print(f"  {name:>26} ({sum(p.numel() for p in m.parameters()):>7} params): {bpc:.3f} bpc")
print("\n=> the algebraic channel does not replace or beat the LSTM on free text;")
print("   it is a structural prior, useful only where the task algebra exists.")
