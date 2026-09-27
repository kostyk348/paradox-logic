"""
algebraic.jev_json — schema-valid JSON by TYPED DECISIONS (Jev-style), with a logits interface.

A document is never decoded token by token.  It is assembled from typed decisions:

  CLOSED choices  (which enum branch, true/false, include-an-optional-key, array length,
                  which oneOf/anyOf branch) are READ from letter logits -- one forward pass,
                  ZERO decoded tokens (algebraic.jev.read_choice).  Without a logits source
                  they are taken by an RNG and still cost ZERO decode steps.
  OPEN slots      (string / integer / number) come from `values` or are synthesised under the
                  schema's constraints.

The logits interface is the point of this module:
  * `logits` may be a dict  ({field: {letter: score}}), a CALLABLE
    (question, field, options) -> {letter: score}, or the string "self";
  * "self" attaches `SelfLogits`, a deterministic lexical scorer -- the module's OWN logits
    when no language model is connected;
  * `emit_logits(schema, context=...)` returns the logits for every closed field of a schema
    (a logits request/answer the caller can inspect or override);
  * every decision in the report carries the raw `scores`, the normalised `shares`, the
    softmax `gap` and its `source`.

Correctness is fail-closed: `check_schema` rejects anything outside the implemented subset
(unknown keywords, pattern, format, tuple items, unresolvable or recursive $ref), and
`validate` reports those schema errors instead of claiming a document is valid.

    obj, rep = generate(SCHEMA, context=user_text, logits="self")     # its own logits
    obj, rep = generate(SCHEMA, logits=lambda q, f, o: model.logits(q))
"""
from __future__ import annotations
import hashlib
import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .jev import LETTERS, build_question, read_choice

_MISSING = object()
_OBJECT_TYPES = ("object", "array")

SUPPORTED_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null",
                   "enum", "choice", "any"}
# `default` / `title` / `description` are annotation-only: accepted, never misleading.
SUPPORTED_KEYWORDS = {
    "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "minimum", "maximum", "minLength", "maxLength", "enum", "options", "const",
    "default", "title", "description", "oneOf", "anyOf", "$defs", "definitions", "$ref",
}


class SchemaError(ValueError):
    """Raised when a schema leaves the implemented subset (fail-closed, never silent)."""


# --------------------------------------------------------------- schema introspection
def _deref(spec: dict, root: dict, _seen=()) -> dict:
    """Follow $ref chains (#/$defs/X, #/definitions/X). Cycles are a check_schema error."""
    while isinstance(spec, dict) and "$ref" in spec:
        ref = spec["$ref"]
        if ref in _seen:
            raise SchemaError(f"recursive $ref '{ref}'")
        target = _pointer(root, ref)
        if target is None:
            raise SchemaError(f"unresolvable $ref '{ref}'")
        spec, _seen = target, _seen + (ref,)
    return spec


def _pointer(root: Any, ref: str) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#"):
        return None
    node = root
    for part in ref[1:].strip("/").split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _branch(spec: dict):
    for key in ("oneOf", "anyOf"):
        if key in spec:
            return key, list(spec[key])
    return None, None


def _branches(spec: dict) -> List[dict]:
    """The union branches of a schema (empty list if it is not a union)."""
    for key in ("oneOf", "anyOf"):
        if isinstance(spec.get(key), list):
            return list(spec[key])
    return []


def schema_type(spec: dict) -> str:
    """Effective type: a oneOf/anyOf union is a 'choice', an enum is an 'enum'."""
    if not isinstance(spec, dict):
        return "any"
    if _branch(spec)[0] is not None:
        return "choice"
    if "const" in spec:
        return "const"
    if "enum" in spec or spec.get("type") == "enum":
        return "enum"
    return spec.get("type", "any")


def normalize(schema: dict) -> dict:
    """Shallow, non-destructive normalisation: enum options into `options`."""
    if not isinstance(schema, dict):
        raise SchemaError("schema must be an object")
    out = dict(schema)
    if "enum" in out and "options" not in out:
        out["options"] = list(out["enum"])
    return out


def _options(spec: dict) -> List[Any]:
    opts = spec.get("options", spec.get("enum"))
    if opts is None:
        raise SchemaError("enum subschema without options/enum")
    return list(opts)


# ------------------------------------------------------------------ fail-closed linting
def check_schema(schema: dict) -> List[str]:
    """Static lint of a schema against the IMPLEMENTED subset. Empty list == fully supported.

    Anything not understood is reported here, so `validate` can never answer 'valid' for a
    constraint we silently ignore (the pattern/format hole in the first version).
    """
    if not isinstance(schema, dict):
        return ["$: schema must be an object"]
    errs: List[str] = []
    _check(schema, schema, "$", errs, (), set())
    return errs


def _check(spec, root, path, errs, active_refs, visited):
    if not isinstance(spec, dict):
        errs.append(f"{path}: subschema must be an object")
        return
    if id(spec) in visited:
        errs.append(f"{path}: recursive schema (object cycle) — bound the recursion explicitly")
        return
    visited = visited | {id(spec)}
    if "$ref" in spec:
        ref = spec["$ref"]
        if not isinstance(ref, str):
            errs.append(f"{path}: $ref must be a string")
            return
        if ref in active_refs:
            errs.append(f"{path}: recursive $ref '{ref}' — unsupported (bound the recursion explicitly)")
            return
        target = _pointer(root, ref)
        if target is None:
            errs.append(f"{path}: unresolvable $ref '{ref}'")
            return
        _check(target, root, path, errs, active_refs + (ref,), visited)
        return
    for k in spec:
        if k not in SUPPORTED_KEYWORDS:
            errs.append(f"{path}: unsupported keyword '{k}'")
    t = spec.get("type")
    if t is not None and t not in SUPPORTED_TYPES:
        errs.append(f"{path}: unsupported type '{t!r}'")
    if "items" in spec and not isinstance(spec["items"], dict):
        errs.append(f"{path}: tuple 'items' arrays are unsupported")
    for key in ("enum", "options"):
        if key in spec and not isinstance(spec[key], list):
            errs.append(f"{path}: '{key}' must be an array")
    for key in ("oneOf", "anyOf"):
        if key in spec:
            if not isinstance(spec[key], list) or not spec[key]:
                errs.append(f"{path}: '{key}' must be a non-empty array")
            else:
                for i, b in enumerate(spec[key]):
                    _check(b, root, f"{path}.{key}[{i}]", errs, active_refs, visited)
    if "properties" in spec:
        if not isinstance(spec["properties"], dict):
            errs.append(f"{path}: 'properties' must be an object")
        else:
            for n, p in spec["properties"].items():
                _check(p, root, f"{path}.properties.{n}", errs, active_refs, visited)
    if isinstance(spec.get("items"), dict):
        _check(spec["items"], root, f"{path}.items", errs, active_refs, visited)
    for key in ("$defs", "definitions"):
        defs = spec.get(key)
        if defs is not None:
            if not isinstance(defs, dict):
                errs.append(f"{path}: '{key}' must be an object")
            else:
                for n, d in defs.items():
                    _check(d, root, f"{path}.{key}.{n}", errs, active_refs, visited)


# ------------------------------------------------------------------------- logits source
class SelfLogits:
    """Its own logits when no model is attached: deterministic lexical scoring.

    Honest scope: this is token overlap between the decision's options and the context plus a
    fixed length prior and a blake2s tiebreak.  It is a reproducible heuristic, NOT a language
    model — use it when the decision really is lexical, or as a stable placeholder.
    """

    _WORD = re.compile(r"[0-9a-zA-Z\u0400-\u04ff]+")

    def __init__(self, context: str = ""):
        self.ctx = set(self._tokens(context))

    @classmethod
    def _tokens(cls, s) -> List[str]:
        return cls._WORD.findall(str(s).lower())

    def __call__(self, question: str, field: str, options: Sequence[Any]) -> Dict[str, float]:
        words = self.ctx | set(self._tokens(field))
        out = {}
        for i, o in enumerate(options):
            ot = set(self._tokens(o))
            overlap = float(len(words & ot))
            prior = 0.05 / (1.0 + len(ot))
            tie = int(hashlib.blake2s(f"{field}|{o}".encode(), digest_size=2).hexdigest(), 16)
            out[LETTERS[i]] = overlap + prior + 1e-4 * (tie / 65535.0)
        return out


class _DictSource:
    """Static {field: {letter: score}}; a field absent from the dict falls back to the RNG."""

    def __init__(self, d: dict):
        self.d = d

    def __call__(self, question, field, options):
        if field in self.d:
            return self.d[field]
        bare = field.rsplit(".", 1)[-1]          # "customer.city" -> "city"
        return self.d.get(bare)


def make_source(logits, context: str = ""):
    """None | callable(question, field, options) -> {letter: score}."""
    if logits is None or logits == "none":
        return None
    if logits == "self":
        return SelfLogits(context)
    if callable(logits):
        return logits
    if isinstance(logits, dict):
        if logits.get("mode") == "self":
            return SelfLogits(context)
        return _DictSource(logits)
    return None


# ------------------------------------------------------------------------- decisions
def _pick(options: Sequence[Any], field: str, source, rng, question: str):
    """A closed typed decision: read from logits (0 decode) or taken by the RNG (0 decode)."""
    letters = LETTERS[:len(options)]
    scores = None
    if source is not None:
        try:
            got = source(question, field, list(options))
        except Exception:
            got = None
        if isinstance(got, dict) and got:
            scores = got
    if scores is not None:
        s = {l: float(scores.get(l, -1e9)) for l in letters}
        best, dist, gap = read_choice(s)
        j = letters.index(best)
        return options[j], {"field": field, "mode": "read", "options": list(options),
                            "pick": options[j], "scores": {l: round(s[l], 6) for l in letters},
                            "shares": {k: round(v, 6) for k, v in dist.items()},
                            "gap": gap, "cost": 0}
    idx = rng.randrange(len(options)) if rng is not None else 0
    return options[idx], {"field": field, "mode": "decide", "options": list(options),
                          "pick": options[idx], "cost": 0}


def _synth_string(field: str, spec: dict, rng: Optional[random.Random]) -> str:
    field = re.sub(r"[\[\].]", "_", re.sub(r"\[\d+\]$", "", field)) or "v"
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
    lo, hi = spec.get("minimum"), spec.get("maximum")
    if integer:
        lo = int(lo) if lo is not None else 0
        hi = int(hi) if hi is not None else 100
        return (rng.randint(lo, max(lo, hi)) if rng is not None else lo)
    lo = float(lo) if lo is not None else 0.0
    hi = float(hi) if hi is not None else 1.0
    return (round(rng.uniform(lo, max(lo, hi)), 6) if rng is not None else lo)


def _len_options(spec: dict, rng: Optional[random.Random]) -> List[int]:
    mn = int(spec.get("minItems", 0) or 0)
    mx = spec.get("maxItems")
    mx = int(mx) if mx is not None else max(mn, 3)
    return list(range(mn, max(mn, mx) + 1))


def _join(field: str, name: str) -> str:
    return f"{field}.{name}" if field else name


# --------------------------------------------------------------------------- generate
def generate(schema: dict, *, context: str = "", values: Optional[dict] = None,
             logits=None, rng: Optional[random.Random] = None,
             include: Optional[Sequence[str]] = None, max_depth: int = 32) -> tuple:
    """Build a schema-valid document through typed decisions.

    logits  : dict {key: {letter: score}} | callable(question, key, options) | "self".
              Decision keys: `path` (value), `path?` (key presence), `path[]` (array length),
              `path@branch` (oneOf/anyOf branch). `path` is dotted, e.g. "customer.city";
              a bare name also matches. A callable that returns {} falls back to the RNG.
    values  : key -> value (also forces optional keys in)
    rng     : random.Random for decisions no source supplies
    Raises SchemaError if the schema leaves the supported subset (fail-closed).
    Returns (obj, report); report["valid"] is the result of validate(obj, schema).
    """
    errs = check_schema(schema)
    if errs:
        raise SchemaError("; ".join(errs))
    root = schema
    values = values or {}
    include_set = set(include or ())
    source = make_source(logits, context)
    decisions: List[dict] = []
    stat = {"read": 0, "decided": 0, "given": 0, "generated": 0, "decode_steps": 0}

    def label(path):
        return path or "$"

    def branch_key(path):
        return label(path) + "@branch"

    def closed(path, options):
        q = build_question(context, label(path), options) if source is not None else ""
        pick, info = _pick(options, label(path), source, rng, q)
        decisions.append(info)
        stat["read" if info["mode"] == "read" else "decided"] += 1
        return pick

    def gen(spec, key, path, local, depth):
        """key: name inside `local` for `values`;  path: dotted path for logits/decisions."""
        if depth > max_depth:
            raise SchemaError(f"max_depth {max_depth} exceeded at {label(path)!r}")
        spec = normalize(_deref(spec, root))
        t = schema_type(spec)
        given = local.get(key, _MISSING) if key else _MISSING
        if t == "choice":
            branches = _branches(spec)
            if given is not _MISSING:
                decisions.append({"field": label(path), "mode": "given", "pick": given, "cost": 0})
                stat["given"] += 1
                return given
            labels = [(b.get("title") or schema_type(normalize(b)) or f"branch{i}")
                      if isinstance(b, dict) else f"branch{i}" for i, b in enumerate(branches)]
            if len(set(labels)) != len(labels):                 # keep the letters unambiguous
                labels = [f"{l}#{i}" for i, l in enumerate(labels)]
            pick = closed(branch_key(path), labels)
            return gen(branches[labels.index(pick)], key, path, local, depth + 1)
        if given is not _MISSING and t not in _OBJECT_TYPES:
            decisions.append({"field": label(path), "mode": "given", "pick": given, "cost": 0})
            stat["given"] += 1
            return given
        if t == "enum":
            return closed(path, _options(spec))
        if t == "const":
            return spec["const"]
        if t == "boolean":
            return closed(path, [True, False])
        if t == "null":
            return None
        if t == "object":
            props = spec.get("properties", {})
            required = set(spec.get("required", list(props)))
            # `local` is the values-subtree for THIS container (root: `values`).
            sub = given if isinstance(given, dict) else local
            out = {}
            for name, ps in props.items():
                if name not in required and name not in include_set and name not in sub:
                    if not closed(_join(path, name) + "?", [True, False]):
                        continue
                out[name] = gen(ps, name, _join(path, name), sub, depth + 1)
            return out
        if t == "array":
            item_spec = normalize(_deref(spec.get("items", {}), root))
            seq = given if isinstance(given, list) else None
            apath = f"{path}[]" if path else "[]"
            if seq is not None:
                n = len(seq)
                decisions.append({"field": apath, "mode": "given", "pick": n, "cost": 0})
                stat["given"] += 1
            else:
                n = closed(apath, _len_options(spec, rng))
            out = []
            for i in range(n):
                if seq is not None:
                    v = seq[i]
                    if schema_type(item_spec) == "object" and isinstance(v, dict):
                        out.append(gen(item_spec, "", f"{path}[{i}]", v, depth + 1))
                    else:
                        out.append(v)
                else:
                    out.append(gen(item_spec, "", f"{path}[{i}]", {}, depth + 1))
            return out
        if t == "integer":
            v = _synth_number(spec, rng, True)
        elif t == "number":
            v = _synth_number(spec, rng, False)
        elif t == "string":
            v = _synth_string(path, spec, rng)
        elif t == "any":
            raise SchemaError(f"{label(path)}: subschema has no usable type "
                              "(add type/enum/const/oneOf/anyOf)")
        else:
            raise SchemaError(f"{label(path)}: unsupported subschema type {t!r}")
        stat["generated"] += 1
        stat["decode_steps"] += len(str(v))
        return v

    obj = gen(root, "", "", values, 0)
    errors = validate(obj, schema)
    baseline = len(json.dumps(obj, ensure_ascii=False))
    return obj, {
        "valid": not errors,
        "errors": errors,
        "read": stat["read"], "decided": stat["decided"], "given": stat["given"],
        "generated": stat["generated"], "decode_steps": stat["decode_steps"],
        "baseline_steps": baseline,
        "savings": round(1.0 - stat["decode_steps"] / baseline, 4) if baseline else 0.0,
        "decisions": decisions,
    }


# ------------------------------------------------------------------------ emit logits
def _closed_sites(spec, root, field="", depth=0, out=None):
    """Every closed decision of a schema: (field, kind, options) -- the logits request."""
    out = [] if out is None else out
    if depth > 64:
        return out
    spec = normalize(_deref(spec, root))
    t = schema_type(spec)
    if t == "choice":
        branches = _branches(spec)
        labels = [(b.get("title") or schema_type(normalize(b))) if isinstance(b, dict) else None
                  for i, b in enumerate(branches)]
        labels = [l or f"branch{i}" for i, l in enumerate(labels)]
        if len(set(labels)) != len(labels):
            labels = [f"{l}#{i}" for i, l in enumerate(labels)]
        out.append(((field or "$") + "@branch", "choice", labels))
        for b in branches:
            _closed_sites(b, root, field, depth + 1, out)
    elif t == "object":
        props = spec.get("properties", {})
        required = set(spec.get("required", list(props)))
        for name, ps in props.items():
            if name not in required:
                out.append((_join(field, name) + "?", "include", [True, False]))
            _closed_sites(ps, root, _join(field, name), depth + 1, out)
    elif t == "array":
        out.append((f"{field}[]" if field else "[]", "length", _len_options(spec, None)))
        if isinstance(spec.get("items"), dict):
            _closed_sites(spec["items"], root, field, depth + 1, out)
    elif t == "enum":
        out.append((field or "$", "enum", _options(spec)))
    elif t == "boolean":
        out.append((field or "$", "boolean", [True, False]))
    return out


def emit_logits(schema: dict, *, context: str = "", logits=None,
                rng: Optional[random.Random] = None) -> dict:
    """The logits for EVERY closed field of a schema -- the module's own when logits='self'.

    Returns {"logits": {field: {letter: score}}, "fields": [{field, kind, options, scores,
    pick, gap}]}.  Feed `logits` back into `generate` to accept them, or override any entry.
    """
    errs = check_schema(schema)
    if errs:
        raise SchemaError("; ".join(errs))
    source = make_source("self" if logits is None else logits, context)
    fields = []
    table = {}
    for field, kind, options in _closed_sites(schema, schema):
        q = build_question(context, field, options)
        pick, info = _pick(options, field, source, rng, q)
        fields.append({"field": field, "kind": kind, "options": list(options),
                       "scores": info.get("scores"), "shares": info.get("shares"),
                       "gap": info.get("gap"), "pick": pick, "mode": info["mode"]})
        if info.get("scores"):
            table[field] = info["scores"]
    return {"logits": table, "fields": fields}


# -------------------------------------------------------------------------- validator
def validate(instance: Any, schema: dict) -> List[str]:
    """Fail-closed validation: schema errors are reported, never ignored."""
    errs = check_schema(schema)
    if errs:
        return [f"schema: {e}" for e in errs]
    out: List[str] = []
    _validate(instance, schema, "$", out, schema)
    return out


def _validate(x: Any, spec: dict, path: str, errs: List[str], root: dict) -> None:
    try:
        spec = normalize(_deref(spec, root))
    except SchemaError as e:
        errs.append(f"{path}: {e}")
        return
    t = schema_type(spec)
    if t == "any":
        return
    if t == "choice":
        key = _branch(spec)[0] or "anyOf"
        branches = _branches(spec)
        ok = []
        for i, b in enumerate(branches):
            sub: List[str] = []
            _validate(x, b, path, sub, root)
            ok.append(not sub)
        if key == "anyOf":
            if not any(ok):
                errs.append(f"{path}: matches none of anyOf {list(range(len(branches)))}")
        else:
            if sum(ok) != 1:
                errs.append(f"{path}: oneOf expects exactly one matching branch, got {sum(ok)}")
        return
    if t == "const":
        if x != spec["const"]:
            errs.append(f"{path}: {x!r} != const {spec['const']!r}")
        return
    if t == "enum":
        if x not in _options(spec):
            errs.append(f"{path}: {x!r} not in enum {_options(spec)!r}")
        return
    if t == "null":
        if x is not None:
            errs.append(f"{path}: expected null")
        return
    if t == "boolean":
        if not isinstance(x, bool):
            errs.append(f"{path}: expected boolean")
        return
    if t == "integer":
        if isinstance(x, bool) or not isinstance(x, int):
            errs.append(f"{path}: expected integer")
            return
    elif t == "number":
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            errs.append(f"{path}: expected number")
            return
    if t in ("integer", "number"):
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None and x < lo:
            errs.append(f"{path}: {x} < minimum {lo}")
        if hi is not None and x > hi:
            errs.append(f"{path}: {x} > maximum {hi}")
        return
    if t == "string":
        if not isinstance(x, str):
            errs.append(f"{path}: expected string")
            return
        if spec.get("minLength") is not None and len(x) < spec["minLength"]:
            errs.append(f"{path}: shorter than minLength {spec['minLength']}")
        if spec.get("maxLength") is not None and len(x) > spec["maxLength"]:
            errs.append(f"{path}: longer than maxLength {spec['maxLength']}")
        return
    if t == "array":
        if not isinstance(x, list):
            errs.append(f"{path}: expected array")
            return
        if spec.get("minItems") is not None and len(x) < spec["minItems"]:
            errs.append(f"{path}: fewer than minItems {spec['minItems']}")
        if spec.get("maxItems") is not None and len(x) > spec["maxItems"]:
            errs.append(f"{path}: more than maxItems {spec['maxItems']}")
        items = spec.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(x):
                _validate(v, items, f"{path}[{i}]", errs, root)
        return
    if t == "object":
        if not isinstance(x, dict):
            errs.append(f"{path}: expected object")
            return
        props = spec.get("properties", {})
        for name in spec.get("required", []):
            if name not in x:
                errs.append(f"{path}.{name}: required property missing")
        if spec.get("additionalProperties") is False:
            for name in x:
                if name not in props:
                    errs.append(f"{path}.{name}: additional property not allowed")
        for name, sub_spec in props.items():
            if name in x:
                _validate(x[name], sub_spec, f"{path}.{name}", errs, root)


# ------------------------------------------------------------------------------- plan
def plan(schema: dict, _field: str = "", _depth: int = 0) -> List[dict]:
    """The static typed-decision program a document will take (kind + options per field)."""
    errs = check_schema(schema)
    if errs:
        raise SchemaError("; ".join(errs))
    steps: List[dict] = []
    for field, kind, options in _closed_sites(schema, schema):
        steps.append({"field": field, "kind": "closed" if kind != "include" else "key",
                      "options": options if not isinstance(options[0], int) or kind != "length"
                      else [f"length in {options}"]})
    for field, spec in _open_sites(schema, schema):
        steps.append({"field": field, "kind": "open", "options": [schema_type(normalize(spec))]})
    return steps


def _open_sites(spec, root, field="", depth=0, out=None):
    out = [] if out is None else out
    if depth > 64:
        return out
    spec = normalize(_deref(spec, root))
    t = schema_type(spec)
    if t in ("string", "integer", "number"):
        out.append((field, spec))
    elif t == "object":
        for name, ps in spec.get("properties", {}).items():
            _open_sites(ps, root, _join(field, name), depth + 1, out)
    elif t == "array" and isinstance(spec.get("items"), dict):
        _open_sites(spec["items"], root, field, depth + 1, out)
    elif t == "choice":
        for b in _branches(spec):
            _open_sites(b, root, field, depth + 1, out)
    return out
