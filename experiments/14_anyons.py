"""
14 — Anyons: holonomy *is* computation (the one place it computes).

Non-abelian braiding of anyons is a holonomy of the configuration space: moving
particles around a loop acts on the state by the holonomy of the braid. The braid
group B_n maps into a non-abelian group; a braid word = a sequence of crossings;
the resulting group element = the "gate".

This toy uses the surjection B_n -> S_n (σ_i -> transposition (i, i+1)).
  * order matters  -> genuinely non-abelian gates (a qubit needs non-commuting ops)
  * topology only  -> the gate depends on the braid class, not the geometry
  * any permutation is reachable -> the representation is surjective (a gate set)
Non-abelian anyons (e.g. Fibonacci) are the real thing; S_n is the shadow, but it
shows the principle: holonomy over a path space is a computational primitive.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groupholonomy import symmetric_group

for n in (3, 4):
    S = symmetric_group(n)
    def sigma(i):                                   # braid generator σ_i  -> transposition (i, i+1)
        p = list(range(n)); p[i], p[i + 1] = p[i + 1], p[i]; return tuple(p)
    def braid(word):
        g = S.id
        for i in word:
            g = S.mul(g, sigma(i))
        return g
    words = [(0, 1), (1, 0), (0, 1, 0), (0, 0)] if n == 3 else \
            [(0, 1), (1, 0), (0, 1, 2), (2, 1, 0), (0, 1, 0, 1)]
    print(f"B_{n} -> S_{n}")
    for w in words:
        print(f"    braid {str(w):12s} -> {braid(w)}")
    print(f"    order matters? σ0σ1 = {braid((0,1))}  vs  σ1σ0 = {braid((1,0))}  "
          f"difference = {braid((0,1)) != braid((1,0))}")
    print()

print("=> braiding a non-abelian group produces non-commuting operations (gates),")
print("   determined by the braid CLASS only -> the computation is topologically protected.")
print("   This is why non-abelian anyons are the physical realisation of a holonomy computer;")
print("   our classical labelled graphs are its abelian / finite shadow.")
