"""25 (fast) — semigroup RNN for a NON-group (non-invertible) sequence task.

Task: tokens {0=inc, 1=reset}; state = counter since last reset, saturating at K.
reset is not invertible => no group models it. A learnable finite SEMIGROUP does.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from nn_group import GroupRNN
torch.manual_seed(0)
K = 4; N = K + 1          # states 0..K


def gen(T, n):
    x = torch.randint(0, 2, (n, T))
    y = torch.zeros(n, dtype=torch.long)
    for s in range(n):
        st = 0
        for t in range(T):
            st = 0 if x[s, t] == 1 else min(st + 1, K)
        y[s] = st
    return x, y


class SemigroupRNN(nn.Module):
    def __init__(self, ntok, n, nout):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(torch.randn(ntok, n, n) * 0.1)   # logits over next state
        self.out = nn.Linear(n, nout)

    def forward(self, x):
        B, T = x.shape
        p = torch.zeros(B, self.n); p[:, 0] = 1.0
        for t in range(T):
            q = torch.softmax(self.W, -1)                      # (ntok, cur, next)
            qa = q[x[:, t]]                                    # (B, cur, next)
            p = torch.einsum("bhe,bh->be", qa, p)
        return self.out(p)


class LSTMBase(nn.Module):
    def __init__(self, ntok, nout, hid=64):
        super().__init__(); self.emb = nn.Embedding(ntok, hid)
        self.rnn = nn.LSTM(hid, hid, batch_first=True); self.out = nn.Linear(2*hid, nout)
    def forward(self, x):
        o, _ = self.rnn(self.emb(x)); return self.out(torch.cat([o[:, -1], o.mean(1)], -1))


def train(m, epochs=60, lr=0.02):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        x, y = gen(12, 3000); opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    accs = []
    for T in (12, 24, 48, 96):
        x, y = gen(T, 1000)
        with torch.no_grad(): accs.append((m(x).argmax(1) == y).float().mean().item())
    return accs


print("reset-counter task (non-invertible): state = count since reset, saturate at %d\n" % K)
print(f"{'model':>14} {'params':>7}   acc @ T 12/24/48/96")
for name, m, k in [("GroupRNN(Z/%d)" % (K+1), GroupRNN('cyclic', K+1, n_tokens=2, k=K+1), 'g'),
                   ("SemigroupRNN", SemigroupRNN(2, N, N), 's'),
                   ("LSTM", LSTMBase(2, N), 'l')]:
    a = train(m)
    print(f"{name:>14} {sum(p.numel() for p in m.parameters()):>7}   " + "  ".join(f"{v:.3f}" for v in a))
print("\n=> reset is not invertible: no group can represent it. The learnable semigroup can.")
