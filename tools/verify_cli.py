#!/usr/bin/env python3
"""
verify_cli — LEVEL 2 tool: verify a process / constraint network by cohomology.

Input: a JSON spec {"nodes": N, "edges": [[u, v, label], ...]} where each edge means
    x_u XOR x_v = label   (over GF(2); label 0 = "must agree", 1 = "must differ").
This models replicated state, protocols, parity constraints, XOR-SAT.

Output: consistency, number of independent contradictions (holonomy dimension), how many
groundings pin a solution, and the minimum repair (which constraints to drop).

Usage:  python3 tools/verify_cli.py tools/example_replication.json
"""
from __future__ import annotations
import json
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algebraic import consistency, holonomy_dim, min_groundings


def report(spec: dict) -> str:
    nv = spec["nodes"]
    edges = [tuple(e) for e in spec["edges"]]
    cons = consistency(nv, edges)
    out = [f"nodes={nv}  constraints={len(edges)}",
           f"consistent={cons}  contradictions={holonomy_dim(nv, edges)}  "
           f"groundings_to_pin={min_groundings(nv, edges)}"]
    if not cons:
        for k in range(len(edges) + 1):
            found = None
            for comb in itertools.combinations(range(len(edges)), k):
                drop = set(comb)
                if consistency(nv, [e for i, e in enumerate(edges) if i not in drop]):
                    found = comb
                    break
            if found is not None:
                out.append(f"minimum repair: drop {k} constraint(s): "
                           f"{[edges[i] for i in found]}")
                break
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tools/example_replication.json"
    with open(path) as f:
        print(report(json.load(f)))
