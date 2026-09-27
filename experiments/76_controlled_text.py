"""
76 — CONTROLLED: same time, same parameters — does our (algebraic) char-LM produce
     better text than a plain char-LM?

A: plain char-LSTM.
B: char-LSTM + exact algebraic features (lexer state + position-in-line) fed to the readout.
Both trained the SAME number of steps on the SAME data; hidden sizes chosen so the parameter
counts match. Metric: bits/char (lower is better).
"""
import sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:120000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
V = len(chars); data = torch.tensor([stoi[c] for c in text])
print(f"corpus {len(text)} chars, vocab {V}, same steps for both\n")


def batch(T=128, n=64):
    i = torch.randint(0, len(data) - T - 1, (n,))
    return (torch.stack([data[j:j + T] for j in i]), torch.stack([data[j + 1:j + T + 1] for j in i]))


def feats(x):                                        # exact prefix features (B,T,14)
    B, T = x.shape
    out = torch.zeros(B, T, 14)
    for b in range(B):
        st = 0; pos = 0
        for t in range(T):
            out[b, t, st] = 1.0
            out[b, t, 6 + (pos % 8)] = 1.0
            c = itos[int(x[b, t])]
            if st == 0: st = 1 if c == '"' else 2 if c == "'" else 3 if c == '#' else 0
            elif st == 1: st = 4 if c == '\\' else (0 if c == '"' else 1)
            elif st == 2: st = 4 if c == '\\' else (0 if c == "'" else 2)
            elif st == 3: st = 0 if c == '\n' else 3
            elif st == 4: st = 1
            pos = 0 if c == '\n' else pos + 1
    return out


class LM(nn.Module):
    def __init__(self, hid, use_feat):
        super().__init__(); self.use_feat = use_feat
        self.emb = nn.Embedding(V, 128)
        self.rnn = nn.LSTM(128, hid, batch_first=True)
        self.out = nn.Linear(hid + (14 if use_feat else 0), V)
    def forward(self, x):
        h = self.rnn(self.emb(x))[0]
        if self.use_feat: h = torch.cat([h, feats(x)], -1)
        return self.out(h)


def nparams(hid, use_feat):
    return V * 128 + 4 * hid * (128 + hid + 1) + (hid + (14 if use_feat else 0)) * V + V


# match parameters: pick hid_A, hid_B so |params| are close
hA = 224; pA = nparams(hA, False)
hB = 221; pB = nparams(hB, True)
print(f"A (plain) params ~ {pA}   B (algebraic) params ~ {pB}   (diff {abs(pA-pB)})\n")


def train(hid, use_feat, steps=700):
    m = LM(hid, use_feat); opt = torch.optim.Adam(m.parameters(), 2e-3); t0 = time.time(); L = []
    for s in range(steps):
        x, y = batch(); opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
    return m, sum(L[-150:]) / 150 / math.log(2), time.time() - t0


print(f"{'model':>26} {'params':>8} {'time s':>7}  bits/char")
for name, h, uf in [("A plain char-LSTM", hA, False), ("B + exact algebra feats", hB, True)]:
    m, bpc, secs = train(h, uf)
    print(f"{name:>26} {nparams(h, uf):>8} {secs:>7.0f}  {bpc:.3f}")
print("\n=> same time, same parameters: report the delta above (this answers the question")
print("   directly -- if B is not lower, the algebra does not improve free-text quality).")
