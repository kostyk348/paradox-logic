"""
16 — Incommensurable languages: content that exists in one language and cannot be
expressed in another AT ALL.

Two S_3-labelled 4-cycles:
    L_id : holonomy = identity      -> consistent (6 coherent thoughts)
    L_3c : holonomy = a 3-cycle     -> paradoxical (0 coherent thoughts)
Both labelings use only EVEN permutations, so their ℤ/2 images are identical ("even",
hence "consistent").  ℤ/2 therefore assigns the two the SAME meaning, while their truth
is opposite.  The proposition "L_3c is paradoxical" is expressible in S_3 and inexpressible
in ℤ/2: the languages are incommensurable (no translation carries the content across).
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groupholonomy import symmetric_group, cyclic_group, consistency

S3 = symmetric_group(3)
Z2 = cyclic_group(2)
G = S3.els
ID = S3.id

def sign(a): return sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j]) % 2
A3 = [e for e in G if sign(e) == 0]                 # even permutations (order 3, abelian)
C = next(e for e in A3 if e != ID)                  # a 3-cycle
def prod(seq):
    h = ID
    for g in seq: h = S3.mul(h, g)
    return h

L_id = [C, C, S3.inv(C), S3.inv(C)]                 # product = identity
L_3c = [C, C, C, C]                                  # product = C  (order 3)

def cycle_edges(labels): return [(i, (i + 1) % 4, labels[i]) for i in range(4)]

print("Two S_3-languages with the SAME ℤ/2 image but opposite truth\n")
for name, labels in (("L_id", L_id), ("L_3c", L_3c)):
    h = prod(labels)
    ok, _, _ = consistency(list(range(4)), cycle_edges(labels), S3)
    zimg = [sign(g) for g in labels]
    zok, _, _ = consistency(list(range(4)), [(u, v, z) for (u, v, _), z in zip(cycle_edges(labels), zimg)], Z2)
    print(f"  {name}: labels={labels}")
    print(f"        S_3 holonomy = {h}  ->  consistent = {ok}")
    print(f"        ℤ/2 image    = {zimg}  ->  consistent = {zok}")

print("\n  both have identical ℤ/2 image (all 'even' -> ℤ/2 says 'consistent')")
print("  but L_3c is PARADOXICAL in S_3.  ℤ/2 cannot state this.")
print("  => a proposition expressible in S_3 is INEXPRESSIBLE in ℤ/2: the languages are")
print("     incommensurable -- not a translation gap but an expressivity gap.")
print("  (ℤ/2 has 2 classes; S_3 has 3: the third class is the untranslatable content.)")
