"""
algebraic.jev_json — schema-valid JSON by TYPED DECISIONS (Jev-style).

A document is never decoded token by token.  It is assembled from typed decisions taken
against a compiled schema:

  CLOSED choices  (which enum branch, true/false, include-an-optional-key, array length)
                  are READ from letter logits -- one forward pass, ZERO decoded tokens
                  (algebraic.jev.read_choice).  Without logits they are decided by an RNG
                  and still cost ZERO decode steps: they are decisions, not tokens.
  OPEN slots      (string / integer / number) are produced under the schema's constraints,
                  from `values` when the caller supplies them.

The result is valid BY CONSTRUCTION: the generator can only emit declared properties,
always emits `required`, encodes each type exactly, and respects enum / range / length /
item bounds.  Everything is then re-checked with `validate` (a JSON-Schema subset) and the
errors travel back in the report, so a caller never has to trust the construction claim.

    obj, rep = generate(SCHEMA, logits=next_logits, rng=random.Random(0))
    rep["valid"] -> True     rep["decode_steps"] -> 0 for closed-choice-only schemas
"""
from __future__ import annotations
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence

from .jev import LETTERS, read_choice

_MISSING = object()
_OBJECT_TYPES = ("object", "array")


# ------------------------------------------------------------------ schema handling
def schema_type(spec: dict) -> str:
    """Resolve the effective type of a subschema (enum wins over type)."""
    if not isinstance(spec, dict):
        return "any"
    if "enum" in spec or spec.get("type") == "enum":
        return "enum"
    return spec.get("type", "any")


def normalize(schema: dict) -> dict:
    """Shallow, non-destructive normalisation: enum options into `options`."""
    if not isinstance(schema, dict):
        raise TypeError("schema must be an object")
    out = dict(schema)
    if "enum" in out and "options" not in out:
        out["options"] = list(out["enum"])
    return out


def _options(spec: dict) -> List[Any]:
    opts = spec.get("options", spec.get("enum"))
    if opts is None:
        raise ValueError("enum subschema without options/enum")
    return list(opts)


# ------------------------------------------------------------------------- decisions
def _join(field: str, name: str) -> str:
    return f"{field}.{name}" if field else name


def _pick(options: Sequence[Any], field: str, logits, rng: Optional[random.Random]):
    """A closed typed decision: `read` from logits (0 decode) or `decide` by RNG (0 decode)."""
    letters = LETTERS[:len(options)]
    if logits and field in logits:
        scores = {l: float(logits[field].get(l, -1e9)) for l in letters}
        best, dist, gap = read_choice(scores)
        pick = options[letters.index(best)]
        return pick, {"field": field, "mode": "read", "options": list(options), "pick": pick,
                      "shares": {k: round(v, 6) for k, v in dist.items()}, "gap": gap,
                      "cost": 0}
    idx = rng.randrange(len(options)) if rng is not None else 0
    return options[idx], {"field": field, "mode": "decide", "options": list(options),
                          "pick": options[idx], "cost": 0}


def _synth_string(field: str, spec: dict, rng: Optional[random.Random]) -> str:
    field = re.sub(r"\[\d+\]$", "", field)          # tags[0] -> tags
    mx = spec.get("maxLength")
    mn = int(spec.get("minLength", 1) or 0)
    r = rng.randrange(1000) if rng is not None else 0
    base = f"{field}_{r:03d}"
    if mx is not None and len(base) > int(mx):
        base = base[: max(1, int(mx))]
    while len(base) < mn:
        base += "x"
    return base


def _synth_number(spec: dict, rng: Optional[random.Random], integer: bool):
    lo = spec.get("minimum")
    hi = spec.get("maximum")
    if integer:
        lo = int(lo) if lo is not None else 0
        hi = int(hi) if hi is not None else 100
        if hi < lo:
            hi = lo
        return rng.randint(lo, hi) if rng is not None else lo
    lo = float(lo) if lo is not None else 0.0
    hi = float(hi) if hi is not None else 1.0
    if hi < lo:
        hi = lo
    return round(rng.uniform(lo, hi), 6) if rng is not None else lo


def _len_options(spec: dict, rng: Optional[random.Random]) -> List[int]:
    mn = int(spec.get("minItems", 0) or 0)
    mx = spec.get("maxItems")
    if mx is None:
        mx = max(mn, 3)
    mx = int(mx)
    if mx < mn:
        mx = mn
    return list(range(mn, mx + 1))


def generate(schema: dict, *, context: str = "", values: Optional[dict] = None,
             logits: Optional[dict] = None, rng: Optional[random.Random] = None,
             include: Optional[Sequence[str]] = None) -> tuple:
    """Build a schema-valid document through typed decisions.

    values  : field -> value (also used to force-include optional keys)
    logits  : field -> {letter: score} for closed choices (0-decode read)
    rng     : random.Random for the decisions the caller does not supply
    include : optional property names to force into the document
    Returns (obj, report); report["valid"] is the result of validate(obj, schema).
    """
    schema = normalize(schema)
    values = values or {}
    include_set = set(include or ())
    decisions: List[dict] = []
    stat = {"read": 0, "decided": 0, "generated": 0, "decode_steps": 0}

    def closed(field, options):
        pick, info = _pick(options, field, logits, rng)
        decisions.append(info)
        stat["read" if info["mode"] == "read" else "decided"] += 1
        return pick

    def gen(spec: dict, field: str, local: dict) -> Any:
        t = schema_type(spec)
        given = local.get(field, _MISSING)
        if given is not _MISSING and t not in _OBJECT_TYPES:
            decisions.append({"field": field, "mode": "given", "pick": given, "cost": 0})
            return given
        if t == "enum":
            return closed(field, _options(spec))
        if t == "boolean":
            return closed(field, [True, False])
        if t == "null":
            return None
        if t == "object":
            props = spec.get("properties", {})
            required = set(spec.get("required", list(props)))
            sub = given if isinstance(given, dict) else {}
            out = {}
            for name, ps in props.items():
                if name not in required and name not in include_set and name not in values \
                        and name not in sub:
                    if not closed(_join(field, name), [True, False]):
                        continue
                out[name] = gen(normalize(ps), name, sub)
            return out
        if t == "array":
            n = closed(f"{field}[]", _len_options(spec, rng))
            item_spec = normalize(spec.get("items", {}))
            seq = given if isinstance(given, list) else None
            out = [gen(item_spec, f"{field}[{i}]", {})
                   for i in range(len(seq) if seq is not None else n)]
            for i, v in enumerate(seq or []):
                out[i] = v
            return out
        if t == "integer":
            v = _synth_number(spec, rng, True)
            stat["generated"] += 1
            stat["decode_steps"] += len(str(v))
            return v
        if t == "number":
            v = _synth_number(spec, rng, False)
            stat["generated"] += 1
            stat["decode_steps"] += len(str(v))
            return v
        if t == "string":
            v = _synth_string(field, spec, rng)
            stat["generated"] += 1
            stat["decode_steps"] += len(v)
            return v
        raise ValueError(f"unsupported subschema type: {t!r}")

    root = normalize(schema)
    obj = gen(root, "", {}) if schema_type(root) == "object" else gen(root, "value", {})
    errors = validate(obj, schema)
    baseline = len(json.dumps(obj, ensure_ascii=False))     # token-by-token writer cost
    report = {
        "valid": not errors,
        "errors": errors,
        "read": stat["read"],
        "decided": stat["decided"],
        "generated": stat["generated"],
        "decode_steps": stat["decode_steps"],
        "baseline_steps": baseline,
        "savings": round(1.0 - stat["decode_steps"] / baseline, 4) if baseline else 0.0,
        "decisions": decisions,
    }
    return obj, report


# -------------------------------------------------------------------------- validator
def validate(instance: Any, schema: dict) -> List[str]:
    """Minimal JSON-Schema subset validator -> list of human-readable errors."""
    errs: List[str] = []
    _validate(instance, normalize(schema), "", errs)
    return errs


def _validate(x: Any, spec: dict, path: str, errs: List[str]) -> None:
    p = path or "$"
    t = schema_type(spec)
    if t == "any":
        return
    if t == "enum":
        if x not in _options(spec):
            errs.append(f"{p}: {x!r} not in enum {_options(spec)!r}")
        return
    if t == "null":
        if x is not None:
            errs.append(f"{p}: expected null")
        return
    if t == "boolean":
        if not isinstance(x, bool):
            errs.append(f"{p}: expected boolean")
        return
    if t == "integer":
        if isinstance(x, bool) or not isinstance(x, int):
            errs.append(f"{p}: expected integer")
            return
    elif t == "number":
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            errs.append(f"{p}: expected number")
            return
    elif t == "string":
        if not isinstance(x, str):
            errs.append(f"{p}: expected string")
            return
    if t in ("integer", "number"):
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None and x < lo:
            errs.append(f"{p}: {x} < minimum {lo}")
        if hi is not None and x > hi:
            errs.append(f"{p}: {x} > maximum {hi}")
        return
    if t == "string":
        if not isinstance(x, str):
            errs.append(f"{p}: expected string")
            return
        if spec.get("minLength") is not None and len(x) < spec["minLength"]:
            errs.append(f"{p}: shorter than minLength {spec['minLength']}")
        if spec.get("maxLength") is not None and len(x) > spec["maxLength"]:
            errs.append(f"{p}: longer than maxLength {spec['maxLength']}")
        return
    if t == "array":
        if not isinstance(x, list):
            errs.append(f"{p}: expected array")
            return
        if spec.get("minItems") is not None and len(x) < spec["minItems"]:
            errs.append(f"{p}: fewer than minItems {spec['minItems']}")
        if spec.get("maxItems") is not None and len(x) > spec["maxItems"]:
            errs.append(f"{p}: more than maxItems {spec['maxItems']}")
        items = spec.get("items")
        if items is not None:
            for i, v in enumerate(x):
                _validate(v, normalize(items), f"{p}[{i}]", errs)
        return
    if t == "object":
        if not isinstance(x, dict):
            errs.append(f"{p}: expected object")
            return
        props = spec.get("properties", {})
        for name in spec.get("required", []):
            if name not in x:
                errs.append(f"{p}.{name}: required property missing")
        if spec.get("additionalProperties") is False:
            for name in x:
                if name not in props:
                    errs.append(f"{p}.{name}: additional property not allowed")
        for name, sub in props.items():
            if name in x:
                _validate(x[name], normalize(sub), f"{p}.{name}", errs)


# ------------------------------------------------------------------------------- plan
def plan(schema: dict, _field: str = "", _depth: int = 0) -> List[dict]:
    """The static decision program a document will take (kind + options per field)."""
    spec = normalize(schema)
    t = schema_type(spec)
    steps: List[dict] = []
    if t == "object":
        props = spec.get("properties", {})
        required = set(spec.get("required", list(props)))
        for name, ps in props.items():
            steps.append({"field": name, "kind": "key",
                          "options": ["present", "absent"] if name not in required else ["required"]})
            steps.extend(plan(ps, name, _depth + 1))
    elif t == "enum":
        steps.append({"field": _field, "kind": "closed", "options": _options(spec)})
    elif t == "boolean":
        steps.append({"field": _field, "kind": "closed", "options": [True, False]})
    elif t == "array":
        steps.append({"field": _field, "kind": "closed",
                      "options": ["length in %s" % (_len_options(spec, None),)]})
        if "items" in spec:
            steps.extend(plan(spec["items"], _field + "[]", _depth + 1))
    elif t in ("string", "integer", "number"):
        steps.append({"field": _field, "kind": "open", "options": [t]})
    return steps
