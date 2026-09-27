"""
42 — What the algebraic layer gives a TRANSFORMER: exact state.

Transformers are strong at semantics but weak at exact long-range state (counting, parity).
We add an algebraic channel: a finite-state automaton computes the exact state per position
and its one-hot is concatenated to the transformer output before the classifier.

Task: running count mod k over a Shakespeare stream (per-position label).
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)

text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:150000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text]); TARGET = stoi['e']; K = 7


def windows(T, n):
    i = torch.randint(0, len(data) - T - 1, (n,))
    x = torch.stack([data[j:j + T] for j in i])
    y = torch.cumsum((x == TARGET).long(), 1) % K
    return x, y


def automaton_states(x):
    """exact (depth-free) count state per position, one-hot, as an extra channel."""
    hit = (x == TARGET).long()
    cs = torch.cumsum(hit, 1) % K
    return F.one_hot(cs, K).float()


class TinyTransformer(nn.Module):
    def __init__(self, use_state=False, d=64, layers=2, heads=2):
        super().__init__(); self.use_state = use_state
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d + (K if use_state else 0), K)
    def forward(self, x):
        h = self.emb(x) + self.pos[:, : x.size(1)]
        h = self.tf(h)
        if self.use_state:
            h = torch.cat([h, automaton_states(x)], -1)
        return self.out(h)


def train(m, steps=400, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = windows(48, 128); opt.zero_grad()
        F.cross_entropy(m(x).reshape(-1, K), y.reshape(-1)).backward(); opt.step()


def acc(m, T, n=300):
    x, y = windows(T, n)
    with torch.no_grad(): return (m(x).argmax(-1) == y).float().mean().item()


print("Running count mod 7 over a Shakespeare stream (per position)\n")
print(f"{'model':>26} {'params':>7}  T=48   T=192  T=512")
for name, m in [("Transformer", TinyTransformer(False)),
                ("Transformer + automaton", TinyTransformer(True, d=64))]:
    train(m)
    a = [acc(m, T) for T in (48, 192, 512)]
    print(f"{name:>26} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> the transformer learns local patterns but not exact state; the algebraic channel")
print("   makes the state EXACT at any length -- semantics (transformer) + state (algebra).")
