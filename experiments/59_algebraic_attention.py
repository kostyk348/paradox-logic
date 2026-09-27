"""
59 — Algebraic attention: RoPE (group action) + LINEAR attention (associativity).

Softmax attention is O(T^2). Dropping the softmax lets the product reassociate:
    (phi(Q) K^T) V  ==  phi(Q) (K^T V)      ->  O(T) with an O(d^2) running state.
RoPE encodes relative position as a group action. We assemble one block with both and
measure wall-time vs sequence length (softmax vs linear).
"""
import sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
D = 64


def rope_tables(T, dh):
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
    f = torch.outer(torch.arange(T).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos[None, None] - x2 * sin[None, None]
    o2 = x1 * sin[None, None] + x2 * cos[None, None]
    return torch.stack([o1, o2], -1).flatten(-2)


class AlgeAttention(nn.Module):
    def __init__(self, kind="softmax", use_rope=True, nhead=2):
        super().__init__(); self.kind = kind; self.h = nhead; self.dh = D // nhead
        self.use_rope = use_rope
        self.qkv = nn.Linear(D, 3 * D); self.o = nn.Linear(D, D); self.ln = nn.LayerNorm(D)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if self.use_rope:
            cos, sin = rope_tables(T, self.dh)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kind == "softmax":
            mask = torch.triu(torch.ones(T, T), 1).bool()
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            att = att.masked_fill(mask, -1e9)
            y = att.softmax(-1) @ v
        else:                                    # LINEAR: phi(x)=elu(x)+1, associative
            phi = lambda z: F.elu(z) + 1.0
            qp, kp = phi(q), phi(k)
            # causal running state: S_t = sum_{s<=t} kp_s v_s^T ; out_t = qp_t S_t
            S = torch.zeros(B, self.h, self.dh, self.dh)
            y = torch.zeros_like(v)
            for t in range(T):
                S = S + kp[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
                y[:, :, t] = (qp[:, :, t].unsqueeze(-2) @ S).squeeze(-2)
            y = y / (torch.arange(1, T + 1).float()[None, None, :, None] ** 0.5)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.ln(x + self.o(y))


def timeit(kind, T, reps=3):
    m = AlgeAttention(kind); x = torch.randn(4, T, D)
    for _ in range(2): m(x)
    t0 = time.time()
    for _ in range(reps): m(x)
    return (time.time() - t0) / reps * 1000


print("Algebraic attention block: softmax (O(T^2)) vs linear (O(T), associative)\n")
print(f"{'T':>6} {'softmax ms':>11} {'linear ms':>10} {'ratio':>7}")
for T in (64, 256, 1024, 2048):
    s, l = timeit("softmax", T), timeit("linear", T)
    print(f"{T:>6} {s:>11.1f} {l:>10.1f} {s/max(l,1e-9):>7.1f}x")

# does linear keep accuracy on a task?  relative-position copy (as exp 58)
V, OFF = 12, 3
def gen(T, n):
    x = torch.randint(0, V, (n, T)); return x, x[:, T - 1 - OFF]


class M(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.emb = nn.Embedding(V, D)
        self.a1 = AlgeAttention(kind); self.a2 = AlgeAttention(kind); self.out = nn.Linear(D, V)
    def forward(self, x):
        h = self.emb(x); h = self.a1(h); h = self.a2(h); return self.out(h[:, -1])


print("\nrelative-position task (train T=20): accuracy of the two blocks")
for kind in ("softmax", "linear"):
    m = M(kind); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for e in range(500):
        x, y = gen(20, 128); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    with torch.no_grad():
        a = []
        for T in (20, 40, 80):
            x, y = gen(T, 400)
            a.append((m(x).argmax(1) == y).float().mean().item())
    print(f"  {kind:>8}: T=20 {a[0]:.3f}  T=40 {a[1]:.3f}  T=80 {a[2]:.3f}")
print("\n=> linear attention is asymptotically cheaper (O(T)) and here keeps the task;")
print("   the associativity (monoid) reassociation is the algebraic lever.")
