"""
12 — Holonomy code: detection, localisation, distance.

A consistent group-labelled graph is a codeword. A corrupted edge is an error.
  * detection:  the holonomy of any cycle through the edge becomes non-trivial
  * distance:   the minimum undetectable error is a non-zero coboundary, whose
                minimum weight is the GIRTH (shortest cycle); a bridge carries no
                cycle, so a single error on a bridge is undetectable
  * localisation: find the unique edge whose removal restores consistency

Measured on random S_3-labelled graphs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from groupholonomy import symmetric_group, consistency, find_bad_edge, sign

S3 = symmetric_group(3)
ID = S3.id
ERR = next(e for e in S3.els if e != ID)          # any non-identity corruption

def rd_graph(V, p, rng):
    E = [(v, int(rng.integers(0, v))) for v in range(1, V)]
    for a in range(V):
        for b in range(a + 1, V):
            if rng.random() < p:
                E.append((a, b))
    return E

def connected(V, E):
    adj = {v: [] for v in range(V)}
    for a, b in E:
        adj[a].append(b); adj[b].append(a)
    seen = {0}; st = [0]
    while st:
        u = st.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w); st.append(w)
    return len(seen) == V

def is_bridge(V, E, i):
    return not connected(V, E[:i] + E[i + 1:])

def consistent_labels(V, E, rng):
    x = {v: S3.els[int(rng.integers(0, len(S3.els)))] for v in range(V)}
    return [(a, b, S3.mul(x[a], S3.inv(x[b]))) for (a, b) in E]

rng = np.random.default_rng(0)
print("Holonomy code on random S_3-labelled graphs\n")
print(f"{'V':>3} {'p':>4} {'cycle-rank':>11} {'detect':>8} {'localise':>9} {'bridge-err undetect':>20}")
for V, p in [(7, 0.2), (7, 0.4), (9, 0.25), (9, 0.5), (12, 0.3)]:
    det = loc = brg = tot = 0
    crsum = 0
    for _ in range(300):
        E = rd_graph(V, p, rng)
        if not connected(V, E):
            continue
        crsum += len(E) - V + 1
        labels = consistent_labels(V, E, rng)
        i = int(rng.integers(0, len(labels)))
        a, b, g = labels[i]
        labels[i] = (a, b, S3.mul(ERR, g))
        ok, _, _ = consistency(list(range(V)), labels, S3)
        bridge = is_bridge(V, E, i)
        if not ok: det += 1
        if not ok and find_bad_edge(list(range(V)), labels, S3) == i: loc += 1
        if bridge and ok: brg += 1
        tot += 1
    print(f"{V:>3} {p:>4} {crsum/max(tot,1):>11.1f} {100*det/tot:>7.0f}% {100*loc/tot:>8.0f}% {brg:>20}")

print("\n'bridge-err undetect' = single errors placed on a bridge that parity/holonomy")
print("cannot see at all (no cycle through them) -- the code's minimum-weight undetectable error.")
print("Detectable errors localise exactly whenever the corrupted edge is the unique edge whose")
print("removal restores consistency; larger cycle rank makes that uniqueness generic.")
