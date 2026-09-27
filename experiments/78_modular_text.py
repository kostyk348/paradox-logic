"""
78 — Modular statistics + streaming composition, on TEXT and on numbers.

(2) MODULAR statistic: label = (count of a target char) mod k over a text window.
    The counter is a monoid (exact, any length); a char-net must learn it and drifts.
(1) STREAMING: split a long stream into chunks; the monoid combines chunk statistics
    associatively and EXACTLY (S(A++B) = S(A).S(B)); the net's carried state drifts.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0)
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:200000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); TARGET = stoi['e']; K = 7


def windows(T, n):
    i = torch.randint(0, len(data) - T - 1, (n,))
    x = torch.stack([data[j:j + T] for j in i])
    y = torch.cumsum((x == TARGET).long(), 1) % K
    return x, y


class Net(nn.Module):
    def __init__(self, hid=96):
        super().__init__(); self.emb = nn.Embedding(len(chars), 48)
        self.rnn = nn.GRU(48, hid, batch_first=True); self.out = nn.Linear(hid, K)
    def forward(self, x): return self.out(self.rnn(self.emb(x))[0])


def train(m, T=64, steps=800, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = windows(T, 128); opt.zero_grad()
        F.cross_entropy(m(x).reshape(-1, K), y.reshape(-1)).backward(); opt.step()


def acc_net(m, T, n=200):
    x, y = windows(T, n)
    with torch.no_grad(): return (m(x).argmax(-1) == y).float().mean().item()


def acc_mono(T, n=200):
    x, y = windows(T, n)                              # exact monoid counter
    return (torch.cumsum((x == TARGET).long(), 1) % K == y).float().mean().item()
    # (identical by construction -> 1.000 at any length)


m = Net(); train(m)
print("Modular statistic over TEXT: (count of 'e') mod 7 (train T=64)\n")
print(f"{'model':>40}  T=64   T=256  T=1024")
print(f"{'net (char-GRU, learns the counter)':>40}  " + "  ".join(f"{acc_net(m,T):.3f}" for T in (64, 256, 1024)))
print(f"{'exact monoid counter (mod 7)':>40}  " + "  ".join(f"{acc_mono(T):.3f}" for T in (64, 256, 1024)))

# (1) streaming: chunk statistics combine exactly (associative monoid)
print("\nStreaming composition (a 1000-char stream, 10 chunks of 100):")
x = data[:1000]
full = int((x == TARGET).sum()) % K
chunks = [int((x[i:i + 100] == TARGET).sum()) for i in range(0, 1000, 100)]
combined = sum(chunks) % K
print(f"  whole-stream statistic  = {full}")
print(f"  combined chunk stats    = {combined}   equal? {full == combined}")
print(f"  (the monoid composes EXACTLY; a net carrying state across chunks would drift)")
print("\n=> modular statistics are a monoid: exact on text, any length, and chunk-composable.")
print("   On such statistics the algebra beats the net; on smooth statistics the net is fine.")
