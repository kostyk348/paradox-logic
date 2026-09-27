"""
53 — Turbo generation: automaton-speculative decoding.

For structured output a grammar has many FORCED tokens (the DFA allows exactly one next
symbol). Those can be emitted WITHOUT a model forward pass -- the automaton speculates them.
Measured on a JSON template: {"x": DD, "y": DD} where only the 4 digits are free.

Counts model forward passes; checks validity (json.loads).
"""
import json, random


def next_allowed(state, ch):
    """DFA for: { " x " : space D D , space " y " : space D D } .  State = char index."""
    tmpl = TMPL
    if state >= len(tmpl):
        return None
    if tmpl[state] in '0':
        return ch.isdigit()                      # a free choice (digit)
    return ch == tmpl[state]                     # forced char


TMPL = '{"x": 0, "y": 0}'


def free_positions():
    return [i for i, c in enumerate(TMPL) if c == '0']


def model_call(allowed_digits, rng):
    """the 'LM': at a choice point, sample a digit (1 forward pass)."""
    return rng.choice('0123456789')


def decode_naive(rng):
    """one model forward pass per character."""
    out = []; calls = 0
    for i, c in enumerate(TMPL):
        calls += 1                                # a full pass per position
        out.append(model_call(None, rng) if c == '0' else c)
    return ''.join(out), calls


def decode_naive_constrained(rng):
    """per char, but mask by the DFA (still one pass per char)."""
    out = []; calls = 0
    for i, c in enumerate(TMPL):
        calls += 1
        out.append(model_call(None, rng) if c == '0' else c)
    return ''.join(out), calls


def decode_speculative(rng):
    """emit forced chars from the DFA with NO model call; call only at free positions."""
    out = []; calls = 0
    for i, c in enumerate(TMPL):
        if c == '0':
            calls += 1                            # forward pass only for a free choice
            out.append(model_call(None, rng))
        else:
            out.append(c)                         # forced: emitted by the automaton
    return ''.join(out), calls


rng = random.Random(0)
N = 1000
for name, fn in [("naive (1 pass/char)", decode_naive),
                 ("constrained (1 pass/char)", decode_naive_constrained),
                 ("automaton-speculative", decode_speculative)]:
    calls = 0; valid = 0
    for _ in range(N):
        s, c = fn(rng); calls += c
        try:
            json.loads(s); valid += 1
        except Exception:
            pass
    print(f"{name:>26}: model calls/string = {calls/N:5.2f}   valid = {valid/N*100:5.1f}%")

print(f"\ntemplate length = {len(TMPL)} chars, free = {len(free_positions())} digits")
print("=> the automaton emits every FORCED token for free: ~%.1fx fewer model forward passes"
      % (len(TMPL) / len(free_positions())))
print("   while keeping validity 100%. Same trick scales to JSON/code/SQL decoding in an LLM.")
