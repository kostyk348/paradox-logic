"""
07 — Non-abelian holonomy: paradoxes invisible to parity, and path dependence.

Replace labels {0,1} by a non-abelian group G (here S_3). An edge label g acts by
x_u = g * x_v. Holonomy around a cycle = the ordered product of labels; a global
assignment exists iff every cycle holonomy is the identity.

Two consequences parity (GF(2)) cannot see:
  * a commutator labeling [a, b, a^-1, b^-1] has TRIVIAL abelianization (even number of
    odd permutations) yet a NON-trivial group holonomy  =>  a parity-invisible paradox;
  * the holonomy of a path depends on the path  =>  hysteresis / real memory.
"""
from itertools import permutations, product
from functools import reduce

def mul(a, b): return tuple(a[b[i]] for i in range(3))       # (a*b)(i) = a(b(i))
def inv(a):
    r = [0, 0, 0]
    for i, v in enumerate(a):
        r[v] = i
    return tuple(r)
def sign(a):
    invs = sum(1 for i in range(3) for j in range(i + 1, 3) if a[i] > a[j])
    return (-1) ** invs

G = [tuple(p) for p in permutations(range(3))]
E = (0, 1, 2)
A = (1, 0, 2)          # transposition (01)
B = (0, 2, 1)          # transposition (12)

print("(1) parity-invisible paradox (commutator labeling on a 4-cycle)")
labels = [A, B, inv(A), inv(B)]                    # a, b, a^-1, b^-1
prod = reduce(mul, labels, E)
ab = reduce(lambda x, g: x * sign(g), labels, 1)
print(f"    abelianization (product of signs) = {ab:+d}   -> GF(2) says: CONSISTENT")
print(f"    non-abelian holonomy = {prod}   (identity? {prod == E})")

def satisfiable(labels):
    n = len(labels)
    for x in product(G, repeat=n):
        if all(mul(labels[i], x[(i + 1) % n]) == x[i] for i in range(n)):
            return True
    return False

print(f"    global assignment exists? {satisfiable(labels)}")
print("    => S_3 says PARADOX where parity sees none: non-abelian holonomy is strictly finer.\n")

print("(2) path dependence (hysteresis)")
edge = {(0, 1): A, (1, 2): B, (0, 2): E}
path_a = mul(edge[(0, 1)], edge[(1, 2)])           # 0 -> 1 -> 2
path_b = edge[(0, 2)]                              # 0 -> 2
print(f"    holonomy 0->1->2 = {path_a}    holonomy 0->2 = {path_b}    equal? {path_a == path_b}")
print("    => a node's value depends on HOW you reached it: path-dependent state = memory.")
