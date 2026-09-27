"""
55 — Full Jev interface (typed decisions) on our stack, and the KV-cache answer.

Schema: intent (enum, 4 options) + subscribed (boolean) + note (string).
Closed choices are READ from letter logits (0 decoded tokens); the string is generated.
We compare decode steps vs the JSON-baseline and show the context prefix is shared.
"""
import sys, json, math
sys.path.insert(0, "/home/lain/paradox-logic")
from algebraic.jev import typed_fill, build_question, LETTERS


# a mock "model": prefers an option's letter (simulates a frozen LM's next-token logits)
TRUTH = {"intent": "refund", "subscribed": "true"}
def mock_logits(question):
    scores = {}
    for l in LETTERS[:6]:
        scores[l] = 0.0
    # the model "votes": stronger logit for the right option's letter
    for field, val in TRUTH.items():
        if f"`{field}`" in question:
            opts = ["refund", "cancel", "info", "complaint"] if field == "intent" else ["true", "false"]
            if val in opts:
                scores[LETTERS[opts.index(val)]] = 4.0
    return scores


def mock_generate(question):
    return "customer asked for a refund"          # string field (would be generated)


SCHEMA = {
    "intent": {"type": "enum", "options": ["refund", "cancel", "info", "complaint"]},
    "subscribed": {"type": "boolean"},
    "note": {"type": "string"},
}
CONTEXT = "User: I want my money back for last month's charge."

obj, meta = typed_fill(SCHEMA, CONTEXT, mock_logits, mock_generate)
print("Typed-decision fill (Jev-style)\n")
print("  JSON:", json.dumps(obj))
print(f"  closed-choice fields READ (no decode): {meta['read']}")
print(f"  fields GENERATED (grammar):            {meta['generated']}  ({meta['decode_steps']} decode steps)")
for f, c in meta["choices"].items():
    print(f"    {f}: pick={c['pick']:>8}  gap={c['gap']:.2f}  shares={ {k: round(v,2) for k,v in c['shares'].items()} }")

# JSON baseline: token-by-token writer must decode EVERY field
import json as _j
baseline_steps = len(_j.dumps(obj)) + 8
print(f"\n  JSON baseline decode steps (every field): ~{baseline_steps}")
print(f"  typed decode steps (strings/numbers only): {meta['decode_steps']}")
print(f"  => decode work cut to {meta['decode_steps']/baseline_steps*100:.0f}% for this schema")

# KV cache: every question shares the same prefix (the context) -> cacheable
q1 = build_question(CONTEXT, "intent", ["refund", "cancel", "info", "complaint"])
q2 = build_question(CONTEXT, "subscribed", ["true", "false"])
shared = len(CONTEXT)
print(f"\n  prompt 1 prefix == context: {q1.startswith(CONTEXT)}   prompt 2: {q2.startswith(CONTEXT)}")
print(f"  shared prefix length = {shared} chars -> identical KV across questions (prefix cache OK)")
print("  CAVEAT: put the question AFTER the text; re-reading the text per field defeats the cache.")
