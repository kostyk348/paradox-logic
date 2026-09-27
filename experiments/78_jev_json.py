"""
78 — Jev-style JSON: TYPED DECISIONS instead of token decoding.

Claim under test: a schema-valid JSON document does not have to be decoded token by token.
Closed choices (enum branch, boolean, key presence, array length) are DECISIONS — read from
letter logits (or taken by an RNG) at ZERO decode steps; only the open slots (string /
integer / number) have to be produced.  Combined with a repair that turns any prefix back
into valid JSON, structured output becomes 'almost free'.

Measured here:
  1. the generated document, the decision trace, and read/decided/generated accounting;
  2. zero decode steps for a closed-choice-only schema;
  3. by construction: random schemas -> 100% valid (strict json.loads + the validator);
  4. contrast with unconstrained character sampling of the same alphabet;
  5. determinism: the same seed gives the same bytes;
  6. repair: EVERY truncation of the generated document returns valid JSON;
  7. mutation testing: every injected schema violation is caught.
"""
import json
import random
import sys

sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.jev_json import (generate, validate, plan, emit_logits,
                                check_schema, SchemaError)
from algebraic.repair import json_repair

SCHEMA = {
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

# mock frozen model: it "votes" for one option's letter (as in experiment 55)
LOGITS = {
    "intent":   {"A": 5.0, "B": 1.0, "C": 0.0, "D": 0.0},     # -> refund
    "priority": {"A": 0.0, "B": 3.0, "C": 1.0, "D": 0.0},     # -> medium
    "escalate": {"A": 2.0, "B": 0.5},                         # -> true
}

print("Jev-style structured output: typed decisions, not decoding\n")
obj, rep = generate(SCHEMA, logits=LOGITS, rng=random.Random(0))
doc = json.dumps(obj, ensure_ascii=False)
print("1. generated document")
print("   " + doc)
print(f"   valid (re-checked) : {rep['valid']}   errors: {rep['errors']}")
print(f"   decisions: read={rep['read']} (logits)  decided={rep['decided']} (rng)"
      f"  generated={rep['generated']} (open slots)")
print(f"   decode steps = {rep['decode_steps']}   token-by-token baseline = {rep['baseline_steps']}"
      f"   savings = {rep['savings'] * 100:.1f}%")
print("   closed decisions (0 decode):")
for d in rep["decisions"]:
    if d["mode"] in ("read", "decide"):
        gap = f" gap={d['gap']:.2f}" if "gap" in d else ""
        print(f"     {d['field']:>12}: {str(d['pick']):>10}  [{d['mode']}]{gap}")

print("\n2. a closed-choice-only schema costs ZERO decode steps")
SC2 = {"type": "object", "required": ["a", "b", "c"],
       "properties": {"a": {"type": "boolean"},
                      "b": {"type": "enum", "options": ["x", "y", "z"]},
                      "c": {"type": "enum", "options": ["p", "q"]}}}
o2, r2 = generate(SC2, logits={"a": {"A": 2.0, "B": 0.0}}, rng=random.Random(1))
print(f"   {json.dumps(o2)}   decode_steps={r2['decode_steps']}  valid={r2['valid']}")

print("\n3. by construction: random schemas -> validity")
def rand_schema(rng, d=0):
    kinds = (["object", "array", "enum", "boolean", "integer", "number", "string"]
             if d < 3 else ["enum", "boolean", "integer", "string"])
    t = rng.choice(kinds)
    if t == "object":
        props = {f"k{i}": rand_schema(rng, d + 1) for i in range(rng.randint(1, 4))}
        return {"type": "object", "properties": props,
                "required": [k for k in props if rng.random() < 0.6],
                "additionalProperties": False}
    if t == "array":
        return {"type": "array", "items": rand_schema(rng, d + 1),
                "minItems": rng.randint(0, 2), "maxItems": rng.randint(2, 4)}
    if t == "enum":
        return {"type": "enum", "options": [f"o{i}" for i in range(rng.randint(2, 5))]}
    if t == "boolean":
        return {"type": "boolean"}
    if t == "integer":
        return {"type": "integer", "minimum": rng.randint(-50, 0), "maximum": rng.randint(1, 50)}
    if t == "number":
        return {"type": "number", "minimum": 0.0, "maximum": rng.randint(1, 10)}
    return {"type": "string", "minLength": 1, "maxLength": rng.randint(3, 20)}

rng = random.Random(42)
N = 5000
good = 0
for _ in range(N):
    sc = rand_schema(rng)
    if sc.get("type") == "array":
        sc = {"type": "object", "required": ["v"], "properties": {"v": sc},
              "additionalProperties": False}
    o, r = generate(sc, rng=rng)
    try:
        json.loads(json.dumps(o))
        if r["valid"] and not validate(o, sc):
            good += 1
    except Exception:
        pass
print(f"   {good}/{N} = {100 * good / N:.1f}% valid (strict json.loads + validator)")

print("\n4. contrast: unconstrained sampling instead of typed decisions")
ALPHA = list('{}[]",:0123456789abcdefghijklmnopqrstuvwxyz ')
free = 0
for _ in range(N):
    s = "".join(rng.choice(ALPHA) for _ in range(rng.randint(5, len(doc))))
    try:
        json.loads(s)
        free += 1
    except Exception:
        pass
print(f"   character sampling : {100 * free / N:.1f}% valid")
print(f"   typed decisions    : {100 * good / N:.1f}% valid")

print("\n5. determinism (same seed -> same bytes)")
a = json.dumps(generate(SCHEMA, logits=LOGITS, rng=random.Random(7))[0], sort_keys=False)
b = json.dumps(generate(SCHEMA, logits=LOGITS, rng=random.Random(7))[0], sort_keys=False)
print(f"   identical: {a == b}")

print("\n6. repair closes the loop: EVERY truncation of the document is valid JSON")
ok = 0
for i in range(1, len(doc) + 1):
    r, v = json_repair(doc[:i])
    try:
        json.loads(r)
        ok += int(bool(v))
    except Exception:
        pass
print(f"   truncated prefixes recovered: {ok}/{len(doc)} = {100 * ok / len(doc):.1f}%")

print("\n7. mutation testing: injected violations caught by the validator")
def mutations(o):
    out = []
    m = json.loads(json.dumps(o)); m.pop("intent", None); out.append(("drop required", m))
    m = json.loads(json.dumps(o)); m["intent"] = "nope"; out.append(("bad enum", m))
    m = json.loads(json.dumps(o)); m["escalate"] = 1; out.append(("bool as int", m))
    m = json.loads(json.dumps(o)); m["customer"]["age"] = 5; out.append(("age < min", m))
    m = json.loads(json.dumps(o)); m["customer"]["name"] = 123; out.append(("string as int", m))
    m = json.loads(json.dumps(o)); m["extra"] = 1; out.append(("extra property", m))
    m = json.loads(json.dumps(o)); m["tags"] = []; out.append(("below minItems", m))
    return out

caught = 0
muts = mutations(obj)
for name, m in muts:
    errs = validate(m, SCHEMA)
    caught += int(bool(errs))
    print(f"   {name:>16} -> {len(errs)} error(s): {errs[0] if errs else 'MISSED'}")
print(f"   caught {caught}/{len(muts)} = {100 * caught / len(muts):.0f}%")

print("\n8. the static typed-decision program (plan)")
for s in plan(SCHEMA):
    print(f"   {s['field']:>16}  {s['kind']:<7} {s['options']}")

print("\n9. the logits interface: the module can EMIT its own logits")
CTX = "The customer wants a refund because the charge was duplicated"
SC3 = {"type": "object", "required": ["intent", "urgent"],
       "properties": {"intent": {"type": "enum", "options": ["refund", "cancel", "info"]},
                      "urgent": {"type": "boolean"}}}
em = emit_logits(SC3, context=CTX)
print(f"   context: {CTX}")
for f in em["fields"]:
    print(f"     {f['field']:>8} [{f['kind']}] options={f['options']} pick={f['pick']} gap={f['gap']:.3f}")
    if f["scores"]:
        print(f"              scores={f['scores']}  shares={ {k: round(v,3) for k,v in f['shares'].items()} }")
o3, r3 = generate(SC3, context=CTX, logits="self", rng=random.Random(0))
o4, _ = generate(SC3, context=CTX, logits=em["logits"], rng=random.Random(9))
print(f"   self  -> {json.dumps(o3)}  (read={r3['read']}, decode={r3['decode_steps']})")
print(f"   fed back -> {json.dumps(o4)}   identical: {o3 == o4}")

def fake_model(question, field, options):
    return {"C": 8.0, "A": 0.0, "B": 0.0} if field == "intent" else {}

o5, r5 = generate(SC3, logits=fake_model, rng=random.Random(0))
print(f"   callable model -> {json.dumps(o5)}  (read={r5['read']}, decided={r5['decided']})")
print("   NOTE: 'self' is a deterministic LEXICAL scorer, not a language model; its gap shows")
print("         when a decision is really unreliable (gap~0 on booleans without cue words).")

print("\n10. fail-closed: schemas outside the subset are refused, never silently 'valid'")
BAD = [("pattern", {"type": "string", "pattern": "^\\d+$"}),
       ("format", {"type": "string", "format": "email"}),
       ("tuple items", {"type": "array", "items": [{"type": "integer"}]}),
       ("unknown kw", {"type": "string", "minLen": 3}),
       ("bad $ref", {"$ref": "#/$defs/nope"})]
for name, sub in BAD:
    sc = {"type": "object", "properties": {"f": sub}}
    lint = check_schema(sc)
    print(f"   {name:12} lint={lint[0] if lint else 'none'}")
    try:
        generate(sc, rng=random.Random(0))
        print("                -> GENERATED (bad!)")
    except SchemaError as e:
        print(f"                -> refused: {str(e)[:58]}")
print(f"   validate('abc', pattern) -> {validate('abc', {'type': 'string', 'pattern': '^\\\\d+$'})}")

print("\n11. $defs/$ref and oneOf/anyOf as typed decisions (paths keep names apart)")
SC4 = {"type": "object", "required": ["home", "work"], "additionalProperties": False,
       "properties": {"home": {"$ref": "#/$defs/addr"}, "work": {"$ref": "#/$defs/addr"}},
       "$defs": {"addr": {"type": "object", "required": ["city", "mode"],
                          "additionalProperties": False,
                          "properties": {"city": {"type": "string", "maxLength": 10},
                                         "mode": {"enum": ["urban", "rural"]},
                                         "label": {"oneOf": [{"type": "string", "maxLength": 6},
                                                             {"type": "boolean"}]}}}}}
o6, r6 = generate(SC4, values={"home": {"city": "Berlin"}, "work": {"city": "Prague"}},
                  logits={"home.mode": {"A": 5.0, "B": 0.0}, "work.mode": {"A": 0.0, "B": 5.0},
                          "home.label?": {"A": 5.0, "B": 0.0},
                          "home.label@branch": {"A": 0.0, "B": 3.0},
                          "home.label": {"A": 0.0, "B": 3.0}},
                  rng=random.Random(0))
print(f"   {json.dumps(o6)}")
print(f"   valid={r6['valid']}  paths used: {[d['field'] for d in r6['decisions'] if d['mode']=='read']}")
