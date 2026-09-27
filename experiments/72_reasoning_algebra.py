"""
72 — (item 3) Reasoning as ALGEBRA: a knowledge base is a monoid of parity constraints.

Facts are x_u XOR x_v = c over GF(2). With absolute assumptions (grounds) we can decide:
  * entailment: Q follows iff in EVERY satisfying assignment x_Q is the same value
  * contradiction: the KB is inconsistent iff holonomy_dim > 0 (the paradox)
  * repair: the minimum facts to drop
All exact (enumerate small; the consistency/holonomy checks are polynomial).
"""
import sys, itertools
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic import consistency, holonomy_dim, min_repair

NAMES = {0: "rains", 1: "wet", 2: "umbrella", 3: "happy", 4: "indoors"}
NV = 6
KB = [(0, 1, 0),      # rains    <=> wet
      (1, 2, 0),      # wet      <=> umbrella
      (2, 4, 0)]      # umbrella <=> indoors


def satisfied(a, edges):
    return all((a[u] ^ a[v]) == c for (u, v, c) in edges)


def entails(nv, edges, assump, q):
    vals = set()
    for a in itertools.product((0, 1), repeat=nv):
        if all(a[k] == v for k, v in assump.items()) and satisfied(a, edges):
            vals.add(a[q])
    return (len(vals) == 1 and (q in NAMES)), vals


print("Reasoning as algebra (a knowledge base = parity constraints)\n")
assump = {0: 1}                                        # assume it RAINS
for q, name in [(1, "wet"), (2, "umbrella"), (4, "indoors"), (3, "happy")]:
    ent, vals = entails(NV, KB, assump, q)
    print(f"  given 'rains', does '{name}' follow?  entail={ent}  (possible values {sorted(vals)})")

BAD = KB + [(4, 0, 1)]                                 # rains != indoors -> odd cycle
print(f"\n  add a contradicting fact (indoors != rains):")
print(f"    consistent = {consistency(NV, BAD)}   contradictions = {holonomy_dim(NV, BAD)}"
      f"   min facts to drop = {min_repair(NV, BAD)}")
print("\n=> reasoning is algebra: entailment = the KB forces the value; contradiction = a")
print("   non-trivial holonomy (the paradox); repair = min_repair. Exact, no generation.")
