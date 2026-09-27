"""
31 — Learning to SEARCH the algebra (no oracle).

(A) Size discovery: train an over-complete automaton (n_max states) and MINIMISE it
    (Moore partition refinement, = Myhill-Nerode).  The number of surviving classes
    is the discovered size of the syntactic monoid.
(B) Order discovery: for an unknown cyclic group, SEARCH over the order k, seeding each
    with the cyclic scaffold, and pick the k that fits -- no oracle, just search.
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)


# ---------------- tasks ----------------
def T_parity(): return 2, {0: lambda h: h, 1: lambda h: 1 - h}, 0, lambda h: h, 2
def T_reset(K=4): return K + 1, {0: lambda h: min(h + 1, K), 1: lambda h: 0}, 0, lambda h: h, K + 1
def T_contains11():
    tab = {0: [0, 0, 2], 1: [1, 2, 2]}
    return 3, {a: (lambda a: (lambda h: tab[a][h]))(a) for a in (0, 1)}, 0, lambda h: int(h == 2), 2


def gen(task, T, n):
    nst, trans, start, lab, ncl = task; ntok = max(trans) + 1
    x = torch.randint(0, ntok, (n, T)); y = torch.zeros(n, dtype=torch.long)
    for s in range(n):
        h = start
        for t in range(T): h = trans[int(x[s, t])](h)
        y[s] = lab(h)
    return x, y


class SG(nn.Module):
    def __init__(self, ntok, n, seed=None, scale=3.0):
        super().__init__(); self.n = n
        self.W = nn.Parameter(torch.randn(ntok, n, n) * 0.4); self.out = nn.Linear(n, ntok if False else n)
        if seed is not None:
            with torch.no_grad():
                for a in range(ntok):
                    s = seed[a % len(seed)]
                    self.W[a].zero_()
                    for h in range(n): self.W[a][h, (h + s) % n] = scale
    def forward(self, x):
        B = x.shape[0]; p = torch.zeros(B, self.n); p[:, 0] = 1.0
        q = torch.softmax(self.W, -1)
        for t in range(x.shape[1]): p = torch.einsum("bhe,bh->be", q[x[:, t]], p)
        return self.out(p)


def train(m, task, epochs=200, lr=0.1):
    ncl = task[4]; m.out = nn.Linear(m.n, ncl)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for e in range(epochs):
        x, y = gen(task, 12, 3000); opt.zero_grad()
        F.cross_entropy(m(x), y).backward(); opt.step()
    return m


# ---------------- (A) Moore minimisation ----------------
def minimise(m, task, T=200, n=400):
    """build the trained DFA, keep reachable states, merge Myhill-Nerode classes."""
    ntok = max(task[1]) + 1
    tab = torch.softmax(m.W.detach(), -1).argmax(-1).tolist()          # tab[c][s] -> next
    with torch.no_grad():
        out = m.out(torch.eye(m.n)).argmax(-1).tolist()                # class per state
    # reachable
    reach = {0}; st = [0]
    while st:
        s = st.pop()
        for c in range(ntok):
            t = tab[c][s]
            if t not in reach: reach.add(t); st.append(t)
    reach = sorted(reach)
    # refine partition by (output, transitions)
    part = {s: out[s] for s in reach}
    changed = True
    while changed:
        changed = False
        sig = {}
        for s in reach:
            key = (out[s], tuple(part[tab[c][s]] for c in range(ntok)))
            sig.setdefault(key, len(sig))
        newpart = {s: sig[(out[s], tuple(part[tab[c][s]] for c in range(ntok)))] for s in reach}
        if len(set(newpart.values())) != len(set(part.values())): changed = True
        part = newpart
    return len(set(part.values()))


print("(A) discovered monoid size by minimising the learned automaton (n_max=8)\n")
print(f"{'task':>12} {'true |M|':>9} {'discovered':>11} {'train-task acc':>15}")
for name, task in [("parity", T_parity()), ("reset(K=4)", T_reset()), ("contains11", T_contains11())]:
    m = SG(max(task[1]) + 1, 8)              # over-complete
    train(m, task)
    x, y = gen(task, 64, 500)
    with torch.no_grad(): acc = (m(x).argmax(1) == y).float().mean().item()
    print(f"{name:>12} {task[0]:>9} {minimise(m, task):>11} {acc:>15.2f}")

print("\n(B) order discovery for an unknown cyclic group (search k = 2..12, cyclic seed)\n")
true_k = 7
x, y = (torch.randint(0, true_k, (3000, 12)), None)
y = (x.sum(1)) % true_k
best = []
for k in range(2, 13):
    m = SG(true_k, k, seed=list(range(k)))   # seed uses k, tokens still 0..true_k-1
    m.out = nn.Linear(k, true_k)             # must predict true_k classes
    opt = torch.optim.Adam(m.parameters(), 0.1)
    for e in range(150):
        opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    with torch.no_grad(): a = (m(x).argmax(1) == y).float().mean().item()
    best.append((a, k))
best.sort(reverse=True)
print(f"    true group order = {true_k}; search results (acc, k): " +
      ", ".join(f"({a:.2f},{k})" for a, k in best[:4]))
print(f"    discovered k = {best[0][1]}  (acc {best[0][0]:.2f})")
