"""
algebraic.jev — typed decisions (Jev-style): CLOSED-CHOICE fields are READ from logits,
not generated.  Strings/numbers are still generated under a grammar.

Why it is cheap: for an enum/boolean field the model answers with a single letter; we read
that letter's next-token logits at the answer position (one forward pass, ZERO decoded
tokens) and get a full ranking + a confidence gap for free.  The question is appended AFTER
the context, so the context KV cache is shared across all questions (same prefix).
"""
from __future__ import annotations
import math
from typing import Callable, Dict, List, Sequence

LETTERS = [chr(ord('A') + i) for i in range(26)]


def read_choice(letter_scores: Dict[str, float]):
    """scores = {letter: logit}. Returns (choice, normalised_shares, confidence_gap)."""
    best = max(letter_scores, key=letter_scores.get)
    m = max(letter_scores.values())
    exps = {k: math.exp(v - m) for k, v in letter_scores.items()}
    tot = sum(exps.values())
    dist = {k: v / tot for k, v in exps.items()}
    srt = sorted(letter_scores.values(), reverse=True)
    gap = (srt[0] - srt[1]) if len(srt) > 1 else float("inf")
    return best, dist, gap


def build_question(context: str, field: str, options: Sequence[str]) -> str:
    """closed-choice field -> a lettered question appended AFTER the (cacheable) context."""
    if options:
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
        return f"{context}\n\nQuestion: what is `{field}`?\n{opts}\nAnswer:"
    return f"{context}\n\nWrite the value of `{field}`:"


def typed_fill(schema: dict, context: str,
               next_logits: Callable[[str], Dict[str, float]],
               generate: Callable[[str], str]):
    """Fill a flat JSON schema. Closed choices are READ (letters); the rest generated.

    next_logits(question) -> {letter: logit}  (one forward pass, no decoding)
    generate(question)    -> str              (grammar-constrained generation)
    Returns (obj, stats) where stats counts read vs generated fields and decode steps.
    """
    out, meta = {}, {"read": 0, "generated": 0, "decode_steps": 0, "choices": {}}
    for field, spec in schema.items():
        if spec["type"] in ("enum", "boolean"):
            options = spec["options"] if spec["type"] == "enum" else ["true", "false"]
            q = build_question(context, field, options)
            scores = next_logits(q)
            letters = LETTERS[:len(options)]
            best, dist, gap = read_choice({l: scores.get(l, -1e9) for l in letters})
            out[field] = options[letters.index(best)]
            meta["read"] += 1
            meta["choices"][field] = {"pick": out[field], "shares": dist, "gap": gap}
        else:
            val = generate(build_question(context, field, []))
            out[field] = val
            meta["generated"] += 1
            meta["decode_steps"] += max(1, len(str(val)))
    return out, meta


def json_baseline_cost(schema: dict, obj: dict) -> int:
    """decode steps a token-by-token JSON writer would spend (rough token estimate)."""
    return len(json_dumps_estimate(obj))


def json_dumps_estimate(obj: dict) -> str:
    import json
    return json.dumps(obj)
