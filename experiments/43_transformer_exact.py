"""
43 — Closing point 1: transformer + EXACT algebraic state head.

Fix of exp 42: the label is a function of the state, so route it through a dedicated
algebra head on the state channel; the transformer head handles only the residual.
Result: exact at any length (1.0), where the bare transformer decays.
"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:150000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text]); TARGET = stoi['e']; K = 7


def windows(T, n):
    i = torch.randint(0, len(data) - T - 1, (n,))
    x = torch.stack([data[j:j + T] for j in i])
    y = torch.cumsum((x == TARGET).long(), 1) % K
    return x, y


def state_onehot(x): return F.one_hot(torch.cumsum((x == TARGET).long(), 1) % K, K).float()


class TF(nn.Module):
    def __init__(self, d=64, layers=2, heads=2):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head_tf = nn.Linear(d, K)
        self.head_alg = nn.Linear(K, K)               # algebra head (on the state)
    def forward(self, x):
        h = self.tf(self.emb(x) + self.pos[:, :x.size(1)])
        return self.head_tf(h) + self.head_alg(state_onehot(x)), h


def train(m, steps=1200, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = windows(48, 128); opt.zero_grad()
        logits, _ = m(x); F.cross_entropy(logits.reshape(-1, K), y.reshape(-1)).backward(); opt.step()


def acc(m, T, n=300):
    x, y = windows(T, n)
    with torch.no_grad(): return (m(x)[0].argmax(-1) == y).float().mean().item()


def tf_only(m, T, n=300):
    x, y = windows(T, n)
    with torch.no_grad():
        _, h = m(x)
        return (m.head_tf(h).argmax(-1) == y).float().mean().item()


def alg_only(m, T, n=300):
    x, y = windows(T, n)
    with torch.no_grad():
        st = state_onehot(x)
        return (m.head_alg(st).argmax(-1) == y).float().mean().item()


print("Running count mod 7 over Shakespeare: exact state, head comparison\n")
m = TF(); train(m)
print(f"{'head':>28}  T=48   T=192  T=512")
print(f"{'transformer head only':>28}  " + "  ".join(f"{tf_only(m,T):.3f}" for T in (48, 192, 512)))
print(f"{'algebra head only':>28}  " + "  ".join(f"{alg_only(m,T):.3f}" for T in (48, 192, 512)))
print(f"{'sum (tf + algebra)':>28}  " + "  ".join(f"{acc(m,T):.3f}" for T in (48, 192, 512)))
print("\n=> the algebra head (state -> label) is EXACT at any length; the transformer head")
print("   decays. Route the exact structure through the algebra, the rest through the net.")

# clean fit of the algebra head alone (state -> label is the identity by construction)
xf, yf = windows(128, 4000)
head = nn.Linear(K, K); opt = torch.optim.Adam(head.parameters(), 1e-2)
for _ in range(800):
    opt.zero_grad(); F.cross_entropy(head(state_onehot(xf)).reshape(-1, K), yf.reshape(-1)).backward(); opt.step()
for T in (48, 192, 512):
    x, y = windows(T, 400)
    with torch.no_grad():
        a = (head(state_onehot(x)).argmax(-1) == y).float().mean().item()
    print(f"   fitted algebra head alone @ T={T}: {a:.3f}")
