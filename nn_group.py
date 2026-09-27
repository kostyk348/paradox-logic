"""
Algorithmic Whorf — group-valued RNN vs standard sequence models (PyTorch).

A GroupRNN keeps its recurrent state as a distribution over a finite group Γ and composes
it through the group's regular representation (permutation matrices of right multiplication).
It therefore exactly implements the holonomy of the token sequence.

Hypothesis: it wins on tasks whose ground-truth algebra IS the group
(parity = Z/2, running sum = Z/k, permutations = S_3) and gives no advantage on MNIST,
whose algebra is not a group.

Run:  python3 nn_group.py
"""
import math, argparse
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"


# ------------------------------------------------------------------ groups
def make_group(kind, k=None):
    if kind == "cyclic":
        els = list(range(k))
        mul = lambda a, b: (a + b) % k
        return els, mul, 0
    if kind == "symmetric3":
        import itertools
        els = [tuple(p) for p in itertools.permutations(range(3))]
        mul = lambda a, b: tuple(a[b[i]] for i in range(3))
        return els, mul, (0, 1, 2)
    raise ValueError(kind)


def regular_rep(els, mul):
    idx = {e: i for i, e in enumerate(els)}
    n = len(els)
    P = torch.zeros(n, n, n)
    for gi, g in enumerate(els):
        for ki, kk in enumerate(els):
            P[gi, idx[mul(kk, g)], ki] = 1.0        # right multiplication
    return P


# ------------------------------------------------------------------ model
class GroupRNN(nn.Module):
    def __init__(self, kind, n_out, n_tokens=None, in_dim=None, k=None, hid=0, pool=True):
        super().__init__()
        els, mul, ident = make_group(kind, k)
        self.register_buffer("P", regular_rep(els, mul))
        self.order = len(els); self.ident = ident
        self.discrete = n_tokens is not None
        self.pool = pool
        if self.discrete:
            self.emb = nn.Embedding(n_tokens, self.order)
        else:
            self.emb = nn.Linear(in_dim, self.order)
        f = (2 if pool else 1) * self.order + hid
        self.extra = nn.GRUCell(self.order, hid) if hid else None
        self.out = nn.Linear(f, n_out)

    def forward(self, x):                       # x: (B,T) ids  or (B,T,D)
        B, T = x.shape[0], x.shape[1]
        p = torch.zeros(B, self.order, device=x.device)
        p[:, self.ident] = 1.0
        pool = torch.zeros_like(p)
        hh = None
        for t in range(T):
            xt = x[:, t]
            q = torch.softmax(self.emb(xt), dim=-1)
            p = torch.einsum("bg,ghk,bk->bh", q, self.P, p)
            pool = pool + p
            if self.extra is not None:
                hh = self.extra(q, hh) if hh is not None else self.extra(q)
        feat = [p, pool / T] if self.pool else [p]
        if self.extra is not None:
            feat.append(hh)
        return self.out(torch.cat(feat, -1))


class SeqBaseline(nn.Module):
    def __init__(self, cell, n_in, n_out, hid=64, n_tokens=None):
        super().__init__()
        self.discrete = n_tokens is not None
        if self.discrete:
            self.emb = nn.Embedding(n_tokens, hid)
            self.rnn = cell(hid, hid, batch_first=True)
        else:
            self.rnn = cell(n_in, hid, batch_first=True)
        self.out = nn.Linear(2 * hid, n_out)

    def forward(self, x):
        if self.discrete:
            h0 = self.emb(x)
        else:
            h0 = x
        o, _ = self.rnn(h0)
        return self.out(torch.cat([o[:, -1], o.mean(1)], -1))


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hid=256, ntok=None):
        super().__init__()
        self.ntok = ntok
        self.net = nn.Sequential(nn.Linear(n_in, hid), nn.ReLU(), nn.Linear(hid, n_out))

    def forward(self, x):
        if self.ntok is not None:
            x = nn.functional.one_hot(x.long(), self.ntok).float().reshape(x.shape[0], -1)
        else:
            x = x.float().reshape(x.shape[0], -1)
        return self.net(x)


# ------------------------------------------------------------------ tasks
def gen(kind, T, N):
    if kind == "parity":
        x = np.random.randint(0, 2, size=(N, T))
        y = x.sum(1) % 2
        return torch.tensor(x), torch.tensor(y)
    if kind == "sum7":
        x = np.random.randint(0, 7, size=(N, T))
        y = x.sum(1) % 7
        return torch.tensor(x), torch.tensor(y)
    if kind == "s3":
        # generators a=(1,0,2), b=(0,2,1); label = resulting permutation index
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        idx = {p: i for i, p in enumerate(perms)}
        x = np.random.randint(0, 2, size=(N, T)); y = np.zeros(N, dtype=int)
        A, B = (1,0,2), (0,2,1)
        for s in range(N):
            cur = (0,1,2)
            for t in range(T):
                g = B if x[s, t] else A
                cur = tuple(g[cur[i]] for i in range(3))
            y[s] = idx[cur]
        return torch.tensor(x), torch.tensor(y)
    raise ValueError(kind)


def task_dims(kind):
    return {"parity": (2, 2, "cyclic", 2),
            "sum7": (7, 7, "cyclic", 7),
            "s3": (2, 6, "symmetric3", None)}[kind]


def train_eval(model, make_batch, epochs, T, N, test_Ts, opt_lr=1e-2, bs=64, is_mlp=False, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=opt_lr)
    for ep in range(epochs):
        x, y = make_batch(T, N)
        for s in range(0, N, bs):
            xb, yb = x[s:s+bs], y[s:s+bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        if verbose and ep % 20 == 0:
            with torch.no_grad():
                tr = (model(x).argmax(1) == y).float().mean().item()
            print(f"        [train] ep{ep:3d} acc={tr:.3f} loss={loss.item():.3f}")
    accs = []
    for T2 in test_Ts:
        if is_mlp and T2 != T:
            accs.append(float("nan")); continue
        xe, ye = make_batch(T2, 2000)
        with torch.no_grad():
            accs.append((model(xe).argmax(1) == ye).float().mean().item())
    return accs


# ------------------------------------------------------------------ run
def main():
    print("=== algorithmic Whorf: group-valued RNN vs baselines (PyTorch) ===\n")
    for kind in ["parity", "sum7", "s3"]:
        ntok, nout, gk, kk = task_dims(kind)
        print(f"[{kind}]  group = {gk}{'' if kk is None else ' ' + str(kk)};  train T=12, N=6000")
        models = {
            "GroupRNN": GroupRNN(gk, nout, n_tokens=ntok, k=kk),
            "Group+GRU": GroupRNN(gk, nout, n_tokens=ntok, k=kk, in_dim=None, hid=48),
            "GRU": SeqBaseline(nn.GRU, None, nout, n_tokens=ntok),
            "LSTM": SeqBaseline(nn.LSTM, None, nout, n_tokens=ntok),
            "MLP": MLP(12 * ntok, nout, ntok=ntok),
        }
        for name, m in models.items():
            is_mlp = name == "MLP"
            accs = train_eval(m, (lambda T, N: gen(kind, T, N)),
                              150, 12, 4000, [12, 24, 48, 96],
                              opt_lr=0.05 if name.startswith("Group") else (3e-3 if name == "MLP" else 0.01),
                              is_mlp=is_mlp, verbose=(name == "GroupRNN"))
            npar = sum(p.numel() for p in m.parameters())
            print(f"    {name:10s} {npar:6d} params : " +
                  "  ".join(f"T={T2}:{'--' if math.isnan(a) else f'{a:.2f}'}"
                            for T2, a in zip([12, 24, 48, 96], accs)))
        print()

    # ---------------- MNIST control ----------------
    print("[MNIST control]  sequences = 28 rows, group state = Z/8")
    from mnist_loader import load_mnist
    Xtr, Ytr, Xte, Yte = load_mnist("/tmp/opencode/mnist")
    Xtr = torch.tensor(Xtr[:12000]).float().reshape(-1, 28, 28) / 255.0
    Ytr = torch.tensor(Ytr[:12000]).long()
    Xte = torch.tensor(Xte[:2000]).float().reshape(-1, 28, 28) / 255.0
    Yte = torch.tensor(Yte[:2000]).long()
    NT = len(Xtr)

    def mrun(model, is_mlp=False, epochs=6, lr=2e-3):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(NT)
            for s in range(0, NT, 64):
                idx = perm[s:s+64]
                xb, yb = Xtr[idx], Ytr[idx]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward(); opt.step()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == Yte).float().mean().item()
        return acc, sum(p.numel() for p in model.parameters())

    acc, np_ = mrun(GroupRNN("cyclic", 10, in_dim=28, k=8))
    print(f"    GroupRNN(Z/8, {np_} params):  test acc = {acc:.3f}")
    acc, np_ = mrun(MLP(784, 10, 128), is_mlp=True)
    print(f"    MLP(784-128-10, {np_} params):  test acc = {acc:.3f}")
    acc, np_ = mrun(SeqBaseline(nn.LSTM, 28, 10, hid=64))
    print(f"    LSTM(28->64, {np_} params):     test acc = {acc:.3f}")
    acc, np_ = mrun(GroupRNN("cyclic", 10, in_dim=28, k=8, hid=64))
    print(f"    GroupRNN+GRU({np_} params):     test acc = {acc:.3f}")


if __name__ == "__main__":
    main()
