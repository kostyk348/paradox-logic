"""76b — multi-seed controlled text-quality comparison (same steps, matched params)."""
import sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:80000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
V = len(chars); data = torch.tensor([stoi[c] for c in text])


def batch(T=128, n=64):
    i = torch.randint(0, len(data) - T - 1, (n,))
    return (torch.stack([data[j:j+T] for j in i]), torch.stack([data[j+1:j+T+1] for j in i]))


def feats(x):
    B, T = x.shape; out = torch.zeros(B, T, 14)
    for b in range(B):
        st = 0; pos = 0
        for t in range(T):
            out[b, t, st] = 1.0; out[b, t, 6 + (pos % 8)] = 1.0
            c = itos[int(x[b, t])]
            if st == 0: st = 1 if c == '"' else 2 if c == "'" else 3 if c == '#' else 0
            elif st == 1: st = 4 if c == '\\' else (0 if c == '"' else 1)
            elif st == 2: st = 4 if c == '\\' else (0 if c == "'" else 2)
            elif st == 3: st = 0 if c == '\n' else 3
            elif st == 4: st = 1
            pos = 0 if c == '\n' else pos + 1
    return out


class LM(nn.Module):
    def __init__(self, hid, uf):
        super().__init__(); self.uf = uf
        self.emb = nn.Embedding(V, 128); self.rnn = nn.LSTM(128, hid, batch_first=True)
        self.out = nn.Linear(hid + (14 if uf else 0), V)
    def forward(self, x):
        h = self.rnn(self.emb(x))[0]
        if self.uf: h = torch.cat([h, feats(x)], -1)
        return self.out(h)


def run(hid, uf, seed, steps=500):
    torch.manual_seed(seed)
    m = LM(hid, uf); opt = torch.optim.Adam(m.parameters(), 2e-3); L = []
    for s in range(steps):
        x, y = batch(); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
    return sum(L[-150:]) / 150 / math.log(2), sum(p.numel() for p in m.parameters())


hA, hB = 176, 173
A = [run(hA, False, s) for s in (0, 1, 2)]
B = [run(hB, True, s) for s in (0, 1, 2)]
bA = np.array([a[0] for a in A]); bB = np.array([b[0] for b in B])
print("Controlled multi-seed (Shakespeare char-LM; same 500 steps)\n")
print(f"  A plain char-LSTM     params {A[0][1]}   bits/char {bA.mean():.3f} ± {bA.std():.3f}   {np.round(bA,3)}")
print(f"  B + exact algebra     params {B[0][1]}   bits/char {bB.mean():.3f} ± {bB.std():.3f}   {np.round(bB,3)}")
d = bB.mean() - bA.mean()
print(f"\n  delta (B - A) = {d:+.3f} bits/char  ({'B better' if d<0 else 'A better'})")
print("  => if |delta| < std, the algebra gives NO reliable text-quality gain at equal params/time.")
