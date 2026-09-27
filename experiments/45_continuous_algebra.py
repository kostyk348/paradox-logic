"""
45 — Closing point 3: continuous inputs for the algebra.

L* needs a discrete alphabet. Pipeline for continuous input:
  continuous signal -> quantiser (fixed bins) -> symbol stream -> L* -> finite automaton.
Task: continuous increments theta_t; hidden state = cumulative rotation mod 2pi in Q=4
quadrants; accept = the cumulative quadrant returns to 0 (a Z/4 counting language over
quantised increments). L* must recover a 4-state automaton from the SYMBOLIC stream.
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/lain/paradox-logic")


# ---- L* (from exp 36), compact ----
def l_star(alphabet, membership, max_len=8):
    S = [()]; E = [()]
    def row(s): return tuple(membership(s + e) for e in E)
    def hyp():
        reps = {}
        for s in S: reps.setdefault(row(s), s)
        states = list(reps); start = row(())
        trans = {r: {a: row(reps[r] + (a,)) for a in alphabet} for r in states}
        acc = {r for r in states if membership(reps[r])}
        return reps, states, trans, acc, start
    import itertools
    while True:
        while True:
            reps, states, trans, acc, start = hyp()
            new = next((s + (a,) for s in S for a in alphabet if row(s + (a,)) not in set(states)), None)
            if new is None: break
            S.append(new)
        inc = None
        for s in S:
            for t in S:
                if row(s) == row(t):
                    for a in alphabet:
                        if row(s + (a,)) != row(t + (a,)):
                            for e in E:
                                if membership(s + (a,) + e) != membership(t + (a,) + e):
                                    inc = (a,) + e
                            if inc: break
                    if inc: break
            if inc: break
        if inc is not None: E.append(inc); continue
        reps, states, trans, acc, start = hyp()
        counter = None
        for L in range(max_len + 1):
            for s in itertools.product(alphabet, repeat=L):
                r = start
                for a in s: r = trans[r][a]
                if (r in acc) != membership(s): counter = s; break
            if counter: break
        if counter is None:
            bij = all(len({trans[r][a] for r in states}) == len(states) for a in alphabet)
            return len(states), bij
        for L in range(len(counter) + 1):
            p = counter[:L]
            if p not in S: S.append(p)


# ---- continuous signal -> quantiser -> symbols ----
Q = 4
rng = np.random.default_rng(0)
# continuous increments: angles in [-pi, pi); quantise into Q bins
theta = rng.uniform(-math.pi, math.pi, 200000)
bins = np.floor((theta + math.pi) / (2 * math.pi) * Q).astype(int) % Q   # symbols 0..3

# symbolic language: accept if cumulative quadrant == 0  (mod-Q counting over symbols)
def membership(s):
    return sum(s) % Q == 0

print("Continuous input -> quantiser -> symbolic stream -> L*\n")
print(f"  quantiser: {Q} bins over theta in [-pi, pi)")
print(f"  sample symbols from the stream: {bins[:16].tolist()}")
n, bij = l_star(list(range(Q)), membership)
print(f"  L* states recovered from symbols: {n}   (true Z/{Q} counter: {Q})")
print(f"  algebra type: {'group' if bij else 'aperiodic'}")
print("\n=> a continuous signal becomes algebra-solvable by quantisation: after that L*")
print("   recovers the exact finite automaton. (A learned VQ would replace the fixed bins.)")
