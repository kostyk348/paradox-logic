#!/usr/bin/env python3
"""
algebra_mcp — a tiny MCP (stdio JSON-RPC) server that makes structured output 'almost free'.

Tools:
  verify_network   {nodes, edges}  -> consistency / contradictions / groundings
  min_repair       {nodes, edges}  -> minimum constraints to drop
  xor_sat          {nvars, clauses, rhs}
  json_repair      {text}          -> longest complete prefix, closed; `valid` is json.loads
  balance_repair   {text}          -> balanced () [] {} and quotes for free text
  lexer_state      {text}          -> exact state of free text (normal/string/comment)
  json_generate    {schema, ...}   -> schema-valid JSON by typed decisions (Jev-style)
  json_validate    {instance, schema} -> JSON-Schema subset -> errors
  json_plan        {schema}        -> the static typed-decision program of a schema

Run:  python3 mcp/algebra_mcp.py     (reads one JSON-RPC request per line on stdin)
"""
from __future__ import annotations
import json
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algebraic import consistency, holonomy_dim, min_groundings, xor_sat
from algebraic.repair import json_repair, balance_repair, lexer_state
from algebraic.jev_json import (generate as jev_generate, validate as jev_validate,
                               plan as jev_plan, emit_logits as jev_emit_logits,
                               check_schema as jev_check_schema, SchemaError)


def _as_obj(x, default=None) -> Any:
    """Accept a JSON string or an already-decoded value (MCP parameters arrive as text)."""
    if x is None or x == "":
        return {} if default is None else default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {} if default is None else default


def _dict_arg(x) -> dict:
    v = _as_obj(x)
    return v if isinstance(v, dict) else {}


def _logits_arg(a):
    """logits: a JSON dict, or the literal "self" (module's own logits) / "none" (RNG only)."""
    mode = str(a.get("logits_mode") or "").strip().lower()
    raw = a.get("logits")
    if isinstance(raw, str) and raw.strip().lower() in ("self", "none"):
        return raw.strip().lower()
    if mode == "self":
        return "self"
    if mode == "none":
        return None
    return _dict_arg(raw) or None


def _seed(a):
    raw = a.get("seed")
    return int(raw) if str(raw).strip() not in ("", "None", "none") else 0


def _gen_report(a):
    schema = _dict_arg(a.get("schema"))
    if not schema:
        return {"valid": False, "json": "", "errors": ["schema is required"], "stats": {}}
    try:
        obj, rep = jev_generate(schema, context=a.get("context", ""),
                                values=_dict_arg(a.get("values")), logits=_logits_arg(a),
                                rng=random.Random(_seed(a)))
    except SchemaError as e:
        return {"valid": False, "json": "", "errors": [str(e)],
                "schema_errors": jev_check_schema(schema), "stats": {}}
    return {
        "json": json.dumps(obj, ensure_ascii=False),
        "valid": rep["valid"],
        "errors": rep["errors"],
        "stats": {k: rep[k] for k in ("read", "decided", "given", "generated",
                                      "decode_steps", "baseline_steps", "savings")},
        "decisions": rep["decisions"],
    }


def _logits_report(a):
    schema = _dict_arg(a.get("schema"))
    if not schema:
        return {"error": "schema is required", "logits": {}, "fields": []}
    logits = _logits_arg(a)
    if logits is None:
        logits = "self"
    try:
        res = jev_emit_logits(schema, context=a.get("context", ""), logits=logits,
                              rng=random.Random(_seed(a)))
    except SchemaError as e:
        return {"error": str(e), "schema_errors": jev_check_schema(schema),
                "logits": {}, "fields": []}
    res["source"] = "self (lexical heuristic)" if logits == "self" else "provided"
    return res


def _validate_report(a):
    schema = _dict_arg(a.get("schema"))
    if not schema:
        return {"valid": False, "errors": ["schema is required"]}
    errs = jev_validate(_as_obj(a.get("instance")), schema)
    out = {"valid": not errs, "errors": errs}
    lint = jev_check_schema(schema)
    if lint:
        out["schema_errors"] = lint          # fail-closed: we never vouch for what we ignore
    return out


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
        "description": ("repair malformed JSON: keeps the longest prefix ending after a complete "
                        "value, closes open brackets/quotes, drops trailing commas and dangling "
                        "escapes, unwraps ```json fences and single-quoted JSON. `valid` is the "
                        "result of json.loads on the returned string; on failure the ORIGINAL "
                        "text is returned unchanged"),
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
    "json_generate": {
        "description": ("generate schema-valid JSON by typed decisions (Jev-style). CLOSED choices "
                        "(enum / boolean / key presence / array length / oneOf-anyOf branch) are read "
                        "from logits at ZERO decode steps; OPEN slots (string/integer/number) come from "
                        "`values` or are synthesised under min/max/length. Supports $defs/$ref. "
                        "`logits` is a JSON dict {field: {letter: score}}, the string 'self' (the "
                        "module's own deterministic lexical logits over `context`), or 'none'. "
                        "Fail-closed: a schema with unsupported keywords or a recursive $ref is "
                        "refused with schema_errors instead of being silently ignored"),
        "input": {"schema": "json", "context": "str", "values": "json",
                  "logits": "json", "logits_mode": "str", "seed": "int"},
        "fn": _gen_report,
    },
    "json_logits": {
        "description": ("the logits request/answer for a schema: for EVERY closed decision (enum, "
                        "boolean, key presence, array length, oneOf/anyOf branch) it returns the "
                        "letter options, the scores, the normalised shares, the softmax gap and the "
                        "pick. mode='self' uses the module's own deterministic lexical logits over "
                        "`context`; a provided `logits` dict is echoed back. The `logits` table can be "
                        "edited and fed straight into json_generate"),
        "input": {"schema": "json", "context": "str", "logits": "json", "mode": "str", "seed": "int"},
        "fn": _logits_report,
    },
    "json_check_schema": {
        "description": ("lint a schema against the implemented subset: unknown keywords, pattern/"
                        "format, tuple items, unresolvable or recursive $ref. Empty list == fully "
                        "supported. validate() reports these instead of claiming a document is valid"),
        "input": {"schema": "json"},
        "fn": lambda a: {"errors": jev_check_schema(_dict_arg(a.get("schema"))),
                         "supported": not jev_check_schema(_dict_arg(a.get("schema")))},
    },
    "json_validate": {
        "description": "validate a document against a JSON-Schema subset (type/properties/required/"
                        "additionalProperties/items/min-max/MinLength-maxLength/enum/null)",
        "input": {"instance": "json", "schema": "json"},
        "fn": _validate_report,
    },
    "json_plan": {
        "description": "static typed-decision program of a schema: which fields are closed choices, "
                        "which are open slots, and their options",
        "input": {"schema": "json"},
        "fn": lambda a: {"steps": jev_plan(_dict_arg(a.get("schema")))},
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
                           "serverInfo": {"name": "algebra_mcp", "version": "0.2"}}}
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
