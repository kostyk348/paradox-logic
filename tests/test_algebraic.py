"""Tests for the algebraic library (plain asserts, no framework needed)."""
import sys
from algebraic import (consistency, holonomy_dim, min_groundings, xor_sat,
                       l_star, DFA, verify, min_repair, product, cascade)


def test_mermin():
    # a 3-cycle of XOR=1 constraints: no global assignment -> 1 contradiction
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
    assert consistency(3, edges) is False
    assert holonomy_dim(3, edges) == 1


def test_consistent_cycle():
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 0)]           # even number of 1s
    assert consistency(3, edges) is True
    assert holonomy_dim(3, edges) == 0


def test_min_repair():
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
    assert min_repair(3, edges) == 1                     # drop one edge -> consistent


def test_xor_sat():
    ok, free = xor_sat(3, [[0, 1], [1, 2], [0, 2]], [1, 1, 1])
    assert ok is False
    ok, free = xor_sat(3, [[0, 1], [1, 2]], [1, 1])
    assert ok is True and free == 1


def test_l_star_parity():
    dfa = l_star([0, 1], lambda s: sum(s) % 2 == 0)
    assert dfa.n == 2 and dfa.is_group()


def test_l_star_contains11():
    dfa = l_star([0, 1], lambda s: any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1)))
    assert dfa.n == 3 and not dfa.is_group()


def test_verify():
    r = verify(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)])
    assert r["consistent"] is False and r["contradictions"] == 1


def test_product():
    A = l_star([0, 1], lambda s: sum(s) % 2 == 0)                 # even parity
    B = l_star([0, 1], lambda s: len(s) % 2 == 0)                 # even length
    P = product(A, B)
    assert P.run((0, 0)) is True and P.run((1,)) is False


def test_cascade():
    A = l_star([0, 1], lambda s: True)                            # any
    B = l_star([0, 1], lambda s: sum(s) % 2 == 0)
    C = cascade(A, B, lambda a, x: x)
    assert C.run((1, 1)) is True and C.run((1,)) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all tests passed")
