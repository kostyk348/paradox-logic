"""
58 — Improving attention with algebra: relative position = a GROUP ACTION (RoPE).

Attention is a bilinear form; relative position m-n is the action of SO(2) on (Q,K) by a
position-dependent rotation. We test a relative-position task (copy the token d steps before
the end): absolute learned PE must memorise positions (fails past the training length);
RoPE depends only on m-n and should generalise to any length.
"""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
V, D, D_OFF = 12, 64, 3


def gen(T, n):
    x = torch.randint(0, V, (n, T))
    y = x[:, T - 1 - D_OFF]                     # token d steps back from the end
    return x, y


def rope_tables(T):
    dh = D // 2                                  # per-head dimension (nhead=2)
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
    pos = torch.arange(T).float()
    freqs = torch.outer(pos, inv)               # (T, dh/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):                    # x: (B,H,T,D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, use_rope, nhead=2):
        super().__init__(); self.h = nhead; self.dh = D // nhead; self.use_rope = use_rope
        self.qkv = nn.Linear(D, 3 * D); self.o = nn.Linear(D, D)
        self.ln = nn.LayerNorm(D)
    def forward(self, x, cos=None, sin=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T), 1).bool()
        att = att.masked_fill(mask, -1e9)
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        return self.ln(x + self.o(y))


class Tiny(nn.Module):
    def __init__(self, use_rope):
        super().__init__(); self.use_rope = use_rope
        self.emb = nn.Embedding(V, D)
        self.pos = None if use_rope else nn.Parameter(torch.randn(1, 256, D) * 0.02)
        self.b1 = Block(use_rope); self.b2 = Block(use_rope); self.out = nn.Linear(D, V)
    def forward(self, x):
        h = self.emb(x)
        if not self.use_rope: h = h + self.pos[:, :x.size(1)]
        cos, sin = rope_tables(x.size(1))
        h = self.b1(h, cos, sin); h = self.b2(h, cos, sin)
        return self.out(h[:, -1])


def train(m, T=20, steps=600, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(steps):
        x, y = gen(T, 128); opt.zero_grad()
        F.cross_entropy(m(x), y).backward(); opt.step()


def acc(m, T, n=500):
    x, y = gen(T, n)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


print("Relative-position task: copy the token 3 steps before the end (train T=20)\n")
print(f"{'positional encoding':>22} {'params':>7}  acc T=20  T=40  T=80")
for name, m in [("absolute learned", Tiny(False)), ("RoPE (group action)", Tiny(True))]:
    train(m)
    a = [acc(m, T) for T in (20, 40, 80)]
    print(f"{name:>22} {sum(p.numel() for p in m.parameters()):>7}  " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> relative position is a group action; RoPE generalises to lengths it never saw,")
print("   absolute learned PE memorises positions and degrades. Algebra in attention works.")
