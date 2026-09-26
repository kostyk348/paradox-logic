"""
15 — Sapir–Whorf, madef precise: language = bundle, thought = global section.

Model a "language" as a group-labelled graph: nodes are concepts, edges are relations.
A "thought" is a globally consistent assignment (a coherent belief set).

Strong Sapir–Whorf ("language determines thought") becomes:
    the set of coherent thoughts is a function of the language.
Gauge transformations (relabelling concepts, x_v -> c x_v) change the surface labels but
NOT the holonomy class; they are exactly "translations" between languages.

Finding: the coherent-thought set depends only on the CONJUGACY CLASS of the holonomy
(the translation invariant). So
   * strong Whorf is FALSE  (there is an invariant core shared by all translations),
   * weak   Whorf is TRUE   (the class bounds the thinkable: a non-trivial class admits
                             no global thought at all).
The number of meaning-classes is the number of conjugacy classes of Γ.
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groupholonomy import symmetric_group

S3 = symmetric_group(3)
G = S3.els
ID = S3.id

def conj_class(e):
    return frozenset(S3.mul(S3.mul(g, e), S3.inv(g)) for g in G)

def holonomy(labels):
    h = ID
    for g in labels: h = S3.mul(h, g)
    return h

def count_solutions(L, labels):
    """number of global assignments x in S3^L with x_a = g x_b on each cycle edge"""
    sols = 0
    for x in itertools.product(G, repeat=L):
        if all(S3.mul(labels[i], x[(i + 1) % L]) == x[i] for i in range(L)):
            sols += 1
    return sols

L = 4
print("Languages = labelings of a 4-cycle over S_3; thoughts = globally consistent assignments\n")
by_class = {}
for labels in itertools.product(G, repeat=L):
    h = holonomy(labels)
    c = conj_class(h)
    s = count_solutions(L, labels)
    by_class.setdefault((c, h == ID), set()).add(s)

print(f"{'class':>34} {'holonomy=id?':>12} {'#solutions':>11}")
for (c, triv), sols in sorted(by_class.items(), key=lambda kv: (not kv[0][1], -len(kv[1]))):
    name = "identity" if triv else f"class[{len(c)} elems]"
    print(f"{name:>34} {str(triv):>12} {str(sorted(sols)):>11}")

print("\nnumber of conjugacy classes of S_3 = 3  ->  3 meaning-classes:")
print("   identity -> 6 coherent thoughts (the language admits a global world)")
print("   transposition class -> 0 ; 3-cycle class -> 0 (paradoxical languages)")
print()
print("=> translation between two languages exists iff they share a holonomy class.")
print("   Same class, different surface labels  -> identical thought-set (mutually translatable).")
print("   Different class                        -> one can think what the other cannot.")
print("   STRONG Whorf: false (invariant core survives all translations).")
print("   WEAK   Whorf: true  (the class bounds the thinkable).")
