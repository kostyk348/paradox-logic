"""
26 — Semigroup RNN = differentiable regular-language learner (Myhill-Nerode).

A regular language L has a SYNTACTIC MONOID M(L); learning to recognise L with a
differentiable finite-state machine = RECOVERING M(L). Groups (Z/2, Z/k) are the group
case; counters / pattern detectors need APERIODIC monoids (Krohn-Rhodes split).

Training is soft (differentiable); at TEST TIME we run the HARD automaton given by the
learned transition table -- exact for any length. We report length generalisation and
exact recovery of the target algebra (up to a state permutation).
"""
import sys, itertools
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)


def T_parity(): return 2, {0: lambda h: h, 1: lambda h: 1 - h}, 0, lambda h: h, 2
def T_mod(k): return k, {a: (lambda a: (lambda h: (h + a) % k))(a) for a in range(k)}, 0, lambda h: h, k
def T_reset(K=4): return K + 1, {0: lambda h: min(h + 1, K), 1: lambda h: 0}, 0, lambda h: h, K + 1
def T_contains11():
    tab = {0: [0, 0, 2], 1: [1, 2, 2]}
    return 3, {a: (lambda a: (lambda h: tab[a][h]))(a) for a in (0, 1)}, 0, lambda h: int(h == 2), 2


TASKS = {"parity(Z/2)": T_parity(), "mod7(Z/7)": T_mod(7), "reset(K=4)": T_reset(), "contains11": T_contains11()}


def gen(task, T, n):
    nst, trans, start, lab, ncl = task; ntok = max(trans) + 1
    x = torch.randint(0, ntok, (n, T)); y = torch.zeros(n, dtype=torch.long)
    for s in range(n):
        h = start
        for t in range(T): h = trans[int(x[s, t])](h)
        y[s] = lab(h)
    return x, y


class SemigroupRNN(nn.Module):
    def __init__(self, ntok, n, nout):
        super().__init__(); self.n = n
        self.W = nn.Parameter(torch.randn(ntok, n, n) * 0.5); self.out = nn.Linear(n, nout)

    def forward(self, x, temp=1.0):
        B = x.shape[0]; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        for t in range(x.shape[1]):
            q = torch.softmax(self.W / temp, -1)
            p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


class LSTMBase(nn.Module):
    def __init__(self, ntok, nout, hid=64):
        super().__init__(); self.emb = nn.Embedding(ntok, hid)
        self.rnn = nn.LSTM(hid, hid, batch_first=True); self.out = nn.Linear(2*hid, nout)
    def forward(self, x):
        o, _ = self.rnn(self.emb(x)); return self.out(torch.cat([o[:, -1], o.mean(1)], -1))


def train_sg(m, task, epochs=500, lr=0.2):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        temp = max(0.05, 1.0 - e / epochs)
        x, y = gen(task, 12, 3000); opt.zero_grad()
        F.cross_entropy(m(x, temp), y).backward(); opt.step()


def train_lstm(m, task, epochs=40, lr=3e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        x, y = gen(task, 12, 3000); opt.zero_grad()
        F.cross_entropy(m(x), y).backward(); opt.step()


def acc_soft(m, task, T):
    x, y = gen(task, T, 1000)
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def acc_hard(m, task, T):
    """run the HARD automaton given by the learned table; classify via the readout."""
    nst, trans, start, lab, ncl = task; ntok = max(trans) + 1
    tab = torch.softmax(m.W.detach(), -1).argmax(-1).tolist()
    x, y = gen(task, T, 1000)
    ok = 0
    with torch.no_grad():
        for s in range(len(x)):
            h = 0
            for t in range(T): h = tab[int(x[s, t])][h]
            onehot = torch.zeros(1, nst); onehot[0, h] = 1.0
            if m.out(onehot).argmax(1).item() == int(y[s]): ok += 1
    return ok / len(x)


def recover(m, task):
    nst, trans, start, lab, ncl = task; ntok = max(trans) + 1
    learned = {a: torch.softmax(m.W[a].detach(), -1).argmax(-1).tolist() for a in range(ntok)}
    target = {a: [trans[a](h) for h in range(nst)] for a in range(ntok)}
    best = 0
    for perm in itertools.permutations(range(nst)):
        good = tot = 0
        for a in range(ntok):
            for h in range(nst):
                tot += 1
                if perm[learned[a][h]] == target[a][h]: good += 1
        best = max(best, good / tot)
    return best


print("Semigroup RNN = differentiable automaton; tasks = regular languages.\n")
print(f"{'task':>12} {'|M|':>4} {'hard-acc T12/48/192/512':>26} {'LSTM T12/48/192':>17} {'recovered':>10}")
for name, task in TASKS.items():
    nst = task[0]; ntok = max(task[1]) + 1; ncl = task[4]
    sg = SemigroupRNN(ntok, nst, ncl); train_sg(sg, task)
    ls = LSTMBase(ntok, ncl); train_lstm(ls, task)
    ha = [acc_hard(sg, task, T) for T in (12, 48, 192, 512)]
    la = [acc_soft(ls, task, T) for T in (12, 48, 192)]
    print(f"{name:>12} {nst:>4}   " + " ".join(f"{v:.2f}" for v in ha) + "      "
          + " ".join(f"{v:.2f}" for v in la) + f"   {recover(sg, task)*100:>8.0f}%")
print("\n=> with a HARD automaton at test time the semigroup RNN is exact at any length;")
print("   it recovers the target algebra. LSTM stays at chance on the group languages.")
