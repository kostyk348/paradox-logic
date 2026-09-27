#!/usr/bin/env python3
"""
algebra_mcp — a tiny MCP (stdio JSON-RPC) server that makes structured output 'almost free'.

Tools:
  verify_network   {nodes, edges}  -> consistency / contradictions / groundings
  min_repair       {nodes, edges}  -> minimum constraints to drop
  xor_sat          {nvars, clauses, rhs}
  json_repair      {text}          -> always-valid JSON (repairs brackets/quotes)
  balance_repair   {text}          -> balanced () [] {} and quotes for free text
  lexer_state      {text}          -> exact state of free text (normal/string/comment)

Run:  python3 mcp/algebra_mcp.py     (reads one JSON-RPC request per line on stdin)
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algebraic import consistency, holonomy_dim, min_groundings, xor_sat


def _balance(s: str):
    """append the closers needed to balance (), [], {} and quotes (respecting strings)."""
    stack = []; in_str = False; esc = False; q = ''
    for ch in s:
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == q: in_str = False
        else:
            if ch in '"\'': in_str = True; q = ch
            elif ch in '([{': stack.append(ch)
            elif ch in ')]}':
                pairs = {')': '(', ']': '[', '}': '{'}
                if stack and stack[-1] == pairs[ch]: stack.pop()
    out = s
    if in_str: out += q
    close = {'(': ')', '[': ']', '{': '}'}
    for o in reversed(stack):
        out += close[o]
    return out


def json_repair(text: str):
    try:
        json.loads(text); return text, True
    except Exception:
        pass
    s = text.strip()
    repaired = _balance(s)
    for _ in range(4):
        try:
            json.loads(repaired); return repaired, True
        except Exception:
            repaired = repaired[:-1]              # drop a trailing incomplete token
            repaired = _balance(repaired)
    return repaired, False


def balance_repair(text: str):
    r = _balance(text)
    d = 0; ok = True
    for ch in r:
        d += ch in '([{'; d -= ch in ')]}'
        if d < 0: ok = False
    return r, ok


def lexer_state(text: str):
    st = "normal"
    for c in text:
        if st == "normal":
            st = "dq" if c == '"' else "sq" if c == "'" else "comment" if c == '#' else "normal"
        elif st == "dq":
            st = "esc" if c == '\\' else ("normal" if c == '"' else "dq")
        elif st == "sq":
            st = "esc" if c == '\\' else ("normal" if c == "'" else "sq")
        elif st == "comment":
            st = "normal" if c == '\n' else "comment"
        elif st == "esc":
            st = "dq"
    return st


TOOLS = {
    "verify_network": {
        "description": "consistency / contradictions / groundings of a GF(2) constraint network",
        "input": {"nodes": "int", "edges": "[[u,v,label],...]"},
        "fn": lambda a: {"consistent": consistency(a["nodes"], [tuple(e) for e in a["edges"]]),
                         "contradictions": holonomy_dim(a["nodes"], [tuple(e) for e in a["edges"]]),
                         "groundings": min_groundings(a["nodes"], [tuple(e) for e in a["edges"]])},
    },
    "min_repair": {
        "description": "minimum number of constraints to drop to restore consistency",
        "input": {"nodes": "int", "edges": "[[u,v,label],...]"},
        "fn": lambda a: {"min_edges_to_drop": _minrep(a["nodes"], [tuple(e) for e in a["edges"]])},
    },
    "xor_sat": {
        "description": "exact k-XOR-SAT: satisfiable? + free variables",
        "input": {"nvars": "int", "clauses": "[[var,...],...]", "rhs": "[0/1,...]"},
        "fn": lambda a: {"satisfiable": xor_sat(a["nvars"], a["clauses"], a["rhs"])[0],
                         "free": xor_sat(a["nvars"], a["clauses"], a["rhs"])[1]},
    },
    "json_repair": {
        "description": "always-valid JSON: repairs unbalanced brackets/quotes",
        "input": {"text": "str"},
        "fn": lambda a: (lambda r: {"valid": r[1], "json": r[0]})(json_repair(a["text"])),
    },
    "balance_repair": {
        "description": "balance () [] {} and quotes in free text",
        "input": {"text": "str"},
        "fn": lambda a: (lambda r: {"balanced": r[1], "text": r[0]})(balance_repair(a["text"])),
    },
    "lexer_state": {
        "description": "exact state of free text (normal / dq / sq / comment / esc)",
        "input": {"text": "str"},
        "fn": lambda a: {"state": lexer_state(a["text"])},
    },
}


def _minrep(nv, edges):
    import itertools
    for k in range(len(edges) + 1):
        for comb in itertools.combinations(range(len(edges)), k):
            drop = set(comb)
            if consistency(nv, [e for i, e in enumerate(edges) if i not in drop]):
                return k
    return len(edges)


def handle(req):
    method = req.get("method"); rid = req.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "algebra_mcp", "version": "0.1"}}}
    if method == "tools/list":
        tools = [{"name": k, "description": v["description"],
                  "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in v["input"]}}}
                 for k, v in TOOLS.items()]
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}}
    if method == "tools/call":
        name = req["params"]["name"]; args = req["params"].get("arguments", {})
        try:
            res = TOOLS[name]["fn"](args)
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": json.dumps(res)}]}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle(req)
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
