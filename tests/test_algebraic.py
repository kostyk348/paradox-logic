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


def test_verify_report():
    from tools.verify_cli import report
    spec = {"nodes": 5, "edges": [[0, 1, 0], [1, 2, 0], [2, 0, 1], [2, 3, 0], [3, 4, 0], [4, 0, 0]]}
    out = report(spec)
    assert "contradictions=1" in out and "drop 1 constraint" in out


def test_hybrid_shapes():
    try:
        import torch
        from algebraic.nn import Hybrid
    except Exception:
        return
    h = Hybrid(8, 2, 2)
    out, _ = h(torch.randn(4, 8), torch.randn(4, 2))
    assert tuple(out.shape) == (4, 2)


def test_jev_read_choice():
    from algebraic.jev import read_choice
    best, dist, gap = read_choice({"A": 4.0, "B": 0.0, "C": 0.0})
    assert best == "A" and abs(sum(dist.values()) - 1.0) < 1e-9 and gap == 4.0


def test_jev_typed_fill():
    from algebraic.jev import typed_fill
    schema = {"intent": {"type": "enum", "options": ["refund", "cancel"]},
              "note": {"type": "string"}}
    obj, meta = typed_fill(schema, "ctx",
                           lambda q: {"A": 5.0, "B": 0.0},
                           lambda q: "hello")
    assert obj["intent"] == "refund" and obj["note"] == "hello"
    assert meta["read"] == 1 and meta["generated"] == 1


def test_runtime_discover_execute():
    from algebraic.runtime import Runtime
    rt = Runtime([0, 1], lambda s: sum(s) % 2 == 0)          # L* finds parity
    assert rt.accepts((1, 1)) and not rt.accepts((1,))
    assert rt.type() == "group"
    assert Runtime.quantize([-3.0, 0.0, 3.0], -3.0, 3.0, 2) == [0, 1, 1]


# --------------------------------------------------------------- repair (algebraic.repair)
def test_repair_keeps_longest_complete_prefix():
    import json
    from algebraic.repair import json_repair
    for bad, want in [('[1,2,', '[1,2]'),
                      ('{"a": 1,', '{"a": 1}'),
                      ('{"a": 1, "b":', '{"a": 1}'),
                      ('{"a":1,}', '{"a":1}'),
                      ('{"a": 1}}', '{"a": 1}'),
                      ('{', '{}'),
                      ('```json\n{"a":1,\n```', '{"a":1}')]:
        r, v = json_repair(bad)
        assert v and json.loads(r) == json.loads(want), (bad, r)
    r, v = json_repair("{'a': 1, 'b': [2, 3]}")
    assert v and json.loads(r) == {"a": 1, "b": [2, 3]}


def test_repair_is_honest_on_failure():
    from algebraic.repair import json_repair
    r, v = json_repair("tru")
    assert (r, v) == ("tru", False)          # original returned, never a mutated string


def test_repair_never_emits_invalid_on_every_truncation():
    import json
    from algebraic.repair import json_repair
    J = '{"a": 1, "b": [2, 3], "c": {"d": "e\\"f"}, "n": -1.5e10}'
    for i in range(1, len(J) + 1):
        r, v = json_repair(J[:i])
        assert v, J[:i]
        json.loads(r)                        # valid:true must always parse


def test_lexer_escape_returns_to_its_own_quote():
    from algebraic.repair import lexer_state
    assert lexer_state("'it\\'s'") == "normal"
    assert lexer_state('"a\\"') == "dq"
    assert lexer_state("'a\\'") == "sq"
    assert lexer_state("# open") == "comment"
    assert lexer_state("x # c\n") == "normal"


# ------------------------------------------------------------- typed JSON (Jev-style)
TICKET = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "priority", "escalate", "tags", "customer"],
    "properties": {
        "intent": {"type": "enum", "options": ["refund", "cancel", "info", "complaint"]},
        "priority": {"type": "enum", "options": ["low", "medium", "high", "urgent"]},
        "escalate": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 12},
                 "minItems": 1, "maxItems": 3},
        "customer": {"type": "object", "required": ["name", "age"],
                     "properties": {"name": {"type": "string", "maxLength": 20},
                                    "age": {"type": "integer", "minimum": 18, "maximum": 99}},
                     "additionalProperties": False},
        "note": {"type": "string", "maxLength": 40},
    },
}


def test_jev_json_valid_by_construction():
    import json, random
    from algebraic.jev_json import generate, validate
    obj, rep = generate(TICKET, logits={"intent": {"A": 5.0, "B": 1.0},
                                        "priority": {"A": 0.0, "B": 3.0},
                                        "escalate": {"A": 2.0, "B": 0.0}},
                        rng=random.Random(0))
    assert rep["valid"] and validate(obj, TICKET) == []
    assert obj["intent"] == "refund" and obj["escalate"] is True
    json.loads(json.dumps(obj))
    assert set(obj) <= set(TICKET["properties"])
    for t in obj["tags"]:
        assert isinstance(t, str) and len(t) <= 12
    assert 18 <= obj["customer"]["age"] <= 99


def test_jev_json_closed_choices_cost_zero_decode_steps():
    import random
    from algebraic.jev_json import generate
    schema = {"type": "object", "required": ["a", "b"],
              "properties": {"a": {"type": "boolean"},
                             "b": {"type": "enum", "options": ["x", "y", "z"]}}}
    for i in range(50):
        obj, rep = generate(schema, rng=random.Random(i))
        assert rep["valid"] and rep["decode_steps"] == 0
        assert obj["a"] in (True, False) and obj["b"] in ("x", "y", "z")


def test_jev_json_reports_schema_violations():
    from algebraic.jev_json import validate
    errs = validate({"kind": "z", "n": 0}, {"type": "object", "required": ["kind", "n"],
                    "properties": {"kind": {"type": "enum", "options": ["a", "b"]},
                                   "n": {"type": "integer", "minimum": 1}}})
    assert len(errs) == 2 and any("not in enum" in e for e in errs)


def test_jev_json_values_reach_the_output():
    """Regression: `values` (real content from the model) must land in the document."""
    import random
    from algebraic.jev_json import generate
    schema = {"type": "object", "required": ["c", "t"],
              "properties": {"c": {"type": "object", "required": ["name", "age"],
                                   "properties": {"name": {"type": "string"},
                                                  "age": {"type": "integer"}}},
                             "t": {"type": "array", "items": {"type": "string"}}}}
    obj, rep = generate(schema, values={"c": {"name": "Ann", "age": 31}, "t": ["a", "b", "c"]},
                        rng=random.Random(0))
    assert obj == {"c": {"name": "Ann", "age": 31}, "t": ["a", "b", "c"]}
    assert rep["valid"]


def test_jev_json_fail_closed():
    """A schema we do not fully implement must never be reported as valid."""
    from algebraic.jev_json import check_schema, validate, generate, SchemaError
    assert check_schema({"type": "object", "properties": {"f": {"type": "string",
                                                               "pattern": "^\\d+$"}}})
    assert check_schema({"type": "object", "properties": {"f": {"type": "string",
                                                               "format": "email"}}})
    assert check_schema({"type": "object", "properties": {"f": {"type": "array",
                                                               "items": [{"type": "integer"}]}}})
    assert check_schema({"type": "object", "properties": {"f": {"$ref": "#/$defs/nope"}}})
    assert check_schema({"type": "object", "properties": {"f": {"type": "integer",
                                                               "title": "n", "default": 1}}}) == []
    assert validate("abc", {"type": "string", "pattern": "^\\d+$"})[0].startswith("schema:")
    try:
        generate({"type": "object", "properties": {"f": {"type": "string", "format": "email"}}})
    except SchemaError:
        pass
    else:
        raise AssertionError("generate must refuse an unsupported schema")


def test_jev_json_ref_union_and_recursion():
    import random
    from algebraic.jev_json import generate, validate, check_schema
    schema = {"type": "object", "required": ["home", "work"], "additionalProperties": False,
              "properties": {"home": {"$ref": "#/$defs/addr"},
                             "work": {"$ref": "#/$defs/addr"}},
              "$defs": {"addr": {"type": "object", "required": ["city", "mode"],
                                 "additionalProperties": False,
                                 "properties": {"city": {"type": "string", "maxLength": 10},
                                                "mode": {"type": "enum",
                                                         "options": ["urban", "rural"]},
                                                "note": {"oneOf": [{"type": "string",
                                                                    "maxLength": 6},
                                                                   {"type": "boolean"}]}}}}}
    assert check_schema(schema) == []
    obj, rep = generate(schema, values={"home": {"city": "Berlin"}, "work": {"city": "Prague"}},
                        logits={"home.mode": {"A": 5.0, "B": 0.0},
                                "work.mode": {"A": 0.0, "B": 5.0},
                                "home.note?": {"A": 5.0, "B": 0.0},
                                "home.note@branch": {"A": 0.0, "B": 3.0},
                                "home.note": {"A": 0.0, "B": 3.0}},
                        rng=random.Random(0))
    # paths keep same-named fields apart, and $ref expands twice
    assert obj["home"]["city"] == "Berlin" and obj["work"]["city"] == "Prague"
    assert obj["home"]["mode"] == "urban" and obj["work"]["mode"] == "rural"
    assert obj["home"]["note"] is False and rep["valid"] and validate(obj, schema) == []
    # recursion is detected, not crashed on
    rec = {"type": "object", "properties": {}}
    rec["properties"]["f"] = rec
    assert any("recursive" in e for e in check_schema(rec))
    cyc = {"type": "object", "properties": {"f": {"$ref": "#/$defs/X"}},
           "$defs": {"X": {"$ref": "#/$defs/X"}}}
    assert any("recursive $ref" in e for e in check_schema(cyc))


def test_jev_json_logits_interface():
    """The module can emit its OWN logits; they are deterministic and reusable."""
    import random
    from algebraic.jev_json import generate, emit_logits
    schema = {"type": "object", "required": ["intent", "urgent"],
              "properties": {"intent": {"type": "enum", "options": ["refund", "cancel", "info"]},
                             "urgent": {"type": "boolean"}}}
    ctx = "Customer wants a refund, the charge was duplicated"
    em = emit_logits(schema, context=ctx)
    assert set(em["logits"]) == {"intent", "urgent"}
    assert em["logits"]["intent"]["A"] > em["logits"]["intent"]["B"]      # lexical: refund wins

    obj, rep = generate(schema, context=ctx, logits="self", rng=random.Random(0))
    assert obj["intent"] == "refund" and rep["read"] == 2 and rep["valid"]
    a = generate(schema, context=ctx, logits="self", rng=random.Random(1))[0]
    b = generate(schema, context=ctx, logits="self", rng=random.Random(1))[0]
    assert a == b                                                        # deterministic

    obj2, _ = generate(schema, context=ctx, logits=em["logits"], rng=random.Random(2))
    assert obj2 == obj                                                   # emitted table round-trips

    def src(question, field, options):                                   # a real model hook
        return {"B": 9.0, "A": 0.0, "C": 0.0} if field == "intent" else {}

    obj3, rep3 = generate(schema, logits=src, rng=random.Random(0))
    assert obj3["intent"] == "cancel" and rep3["decided"] == 1           # boolean used the RNG


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all tests passed")
