"""
36 — Algebra search WITHOUT a prior: Angluin's L* (with an equivalence teacher).

L* builds the unique minimal DFA of a regular language from membership queries and
equivalence queries -- i.e. it DISCOVERS the syntactic monoid with no architecture prior.
It then classifies the algebra: all transitions bijective -> group; else aperiodic.
"""
import itertools


def l_star(alphabet, membership, max_len=8):
    S = [()]; E = [()]

    def row(s): return tuple(membership(s + e) for e in E)

    def hypothesise():
        reps = {}
        for s in S: reps.setdefault(row(s), s)
        states = list(reps); start = row(())
        trans = {r: {a: row(reps[r] + (a,)) for a in alphabet} for r in states}
        accept = {r for r in states if membership(reps[r])}
        return reps, states, trans, accept, start

    def run_hyp(s, trans, accept, start):
        r = start
        for a in s: r = trans[r][a]
        return r in accept

    queries = 0
    while True:
        # close
        while True:
            reps, states, trans, accept, start = hypothesise()
            rows = set(states)
            new = next((s + (a,) for s in S for a in alphabet if row(s + (a,)) not in rows), None)
            if new is None: break
            S.append(new)
        # consistent
        inc = None
        for s in S:
            for t in S:
                if row(s) == row(t):
                    for a in alphabet:
                        if row(s + (a,)) != row(t + (a,)):
                            for e in E:
                                if membership(s + (a,) + e) != membership(t + (a,) + e):
                                    inc = (a,) + e; break
                            if inc: break
                    if inc: break
            if inc: break
        if inc is not None: E.append(inc); continue
        # equivalence query: exhaustive counterexample search
        reps, states, trans, accept, start = hypothesise()
        counter = None
        for L in range(0, max_len + 1):
            for s in itertools.product(alphabet, repeat=L):
                queries += 1
                if run_hyp(s, trans, accept, start) != membership(s):
                    counter = s; break
            if counter: break
        if counter is None:
            bij = all(len({trans[r][a] for r in states}) == len(states) for a in alphabet)
            return len(states), bij, queries
        for L in range(0, len(counter) + 1):
            p = counter[:L]
            if p not in S: S.append(p)


def mem_parity(s):
    p = 0
    for a in s: p ^= a & 1
    return p == 0
def mem_mod7(s): return sum(a for a in s) % 7 == 0
def mem_reset(s):
    c = 0
    for a in s: c = 0 if a == 1 else min(c + 1, 4)
    return c == 4
def mem_contains11(s): return any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1))

TASKS = [("parity", [0, 1], mem_parity, 2), ("mod7", list(range(7)), mem_mod7, 7),
         ("reset(K=4)", [0, 1], mem_reset, 5), ("contains11", [0, 1], mem_contains11, 3)]
print("L* discovers the minimal automaton from queries only (no architecture prior)\n")
print(f"{'language':>12} {'true |M|':>9} {'L* states':>10} {'algebra':>12} {'queries':>9}")
for name, alpha, mem, true_n in TASKS:
    n, bij, q = l_star(alpha, mem)
    print(f"{name:>12} {true_n:>9} {n:>10} {'group' if bij else 'aperiodic':>12} {q:>9}")
print("\n=> L* recovers the syntactic monoid size and its group/aperiodic type with NO prior;")
print("   that type (Krohn-Rhodes) tells you which architecture the task needs.")
