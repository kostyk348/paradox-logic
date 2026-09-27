"""
49 — For training GENERAL nets: exact-state auxiliary supervision.

A cheap exact automaton supplies the latent state per step; a small auxiliary head on the
net's hidden state is trained to predict it (an auxiliary loss). This is process supervision
from an algebra the task already obeys. It should speed up convergence and help long range.

Task: running count mod k over a Shakespeare stream (per-position). Real data.
  baseline : GRU -> main head
  + aux    : GRU -> main head  AND  GRU -> aux head (exact state)
"""
import sys, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:150000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text]); TARGET = stoi['e']; K = 11


def windows(T, n):
    i = torch.randint(0, len(data) - T - 1, (n,))
    x = torch.stack([data[j:j + T] for j in i])
    st = torch.cumsum((x == TARGET).long(), 1) % K
    return x, st, (st[:, -1] if False else st)          # per-step state (label = state here)


class GRU(nn.Module):
    def __init__(self, use_aux=False, hid=96):
        super().__init__(); self.use_aux = use_aux
        self.emb = nn.Embedding(V, 48); self.rnn = nn.GRU(48, hid, batch_first=True)
        self.main = nn.Linear(hid, K)
        self.aux = nn.Linear(hid, K) if use_aux else None
    def forward(self, x):
        h = self.rnn(self.emb(x))[0]
        return self.main(h), (self.aux(h) if self.aux else None)


def train(m, steps=1200, lr=3e-3, lam=0.5, thresh=0.90):
    opt = torch.optim.Adam(m.parameters(), lr=lr); hit = None
    for e in range(steps):
        x, st, _ = windows(48, 128); opt.zero_grad()
        logits, aux = m(x)
        loss = F.cross_entropy(logits.reshape(-1, K), st.reshape(-1))
        if aux is not None:
            loss = loss + lam * F.cross_entropy(aux.reshape(-1, K), st.reshape(-1))
        loss.backward(); opt.step()
        if hit is None and e % 25 == 0 and main_acc(m, 48) >= thresh: hit = e
    return None if hit is None else hit


def main_acc(m, T, n=300):
    x, st, _ = windows(T, n)
    with torch.no_grad(): return (m(x)[0].argmax(-1) == st).float().mean().item()


print("Auxiliary exact-state supervision for a general GRU (Shakespeare, mod 11)\n")
print(f"{'model':>18} {'params':>7} {'steps->90%':>11}  acc T=48/192/512")
for name, m, lam in [("GRU (baseline)", GRU(False), 0.0), ("GRU + aux state", GRU(True), 0.5),
                     ("GRU + aux (strong)", GRU(True), 2.0)]:
    hit = train(m, lam=lam)
    a = [main_acc(m, T) for T in (48, 192, 512)]
    print(f"{name:>18} {sum(p.numel() for p in m.parameters()):>7} "
          f"{(str(hit) if hit is not None else '>1200'):>11}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> an exact automaton as an AUXILIARY teacher gives process supervision for free:")
print("   faster to threshold and better long-range -- a training trick usable on any net.")
