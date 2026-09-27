"""
algebraic — exact algebra as a layer for neural networks (LEVEL 1) and a verification
primitive for process networks (LEVEL 2).

LEVEL 1: automaton.SemigroupRNN / nn.Hybrid / automaton.constrained_decode
LEVEL 2: core.consistency / core.holonomy_dim / verify.verify / verify.min_repair
"""
from .core import consistency, holonomy_dim, min_groundings, xor_sat, gf2_rank
from .automaton import DFA, l_star, constrained_decode
from .verify import verify, min_repair, product, cascade

__all__ = ["consistency", "holonomy_dim", "min_groundings", "xor_sat", "gf2_rank",
           "DFA", "l_star", "constrained_decode",
           "verify", "min_repair", "product", "cascade"]

try:                                    # torch is optional
    from .nn import SemigroupRNN, AlgebraHead, Hybrid, policy_from_dfa
    __all__ += ["SemigroupRNN", "AlgebraHead", "Hybrid", "policy_from_dfa"]
except Exception:
    pass
