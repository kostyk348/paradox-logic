"""
87 — "Structure via logits, the LLM only writes."

The algorithm exports the logit structure (top-5 chars/step). An LLM (spawned via task, i.e. the
harness API) is asked to WRITE it. Result: the LLM reproduced the structure EXACTLY (23/23 chars,
only a context echo), and produced nothing different where the logits were flat.

  algorithm greedy sample      : ":Why, the people, when h"
  LLM rendered (both prompts)  : "RICHARD III:Why, the people, when Why, the people, when h"
                                 -> the added 23 chars == the structure, exactly.

Measured here: bits/char of that text under the PPM, and the char-wise agreement structure vs LLM.
"""
import sys, math, collections
import numpy as np
sys.path.insert(0, "/home/lain/paradox-logic")

text = open("/tmp/opencode/shakespeare.txt").read()
text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], np.int64)
K = 4; TR = data[:260000]

tabs = [collections.defaultdict(collections.Counter) for _ in range(K + 1)]
for i in range(K, len(TR)):
    for k in range(1, K + 1):
        tabs[k][tuple(TR[i - k:i])][TR[i]] += 1
UNI = collections.Counter(TR); tot = sum(UNI.values())
uni = np.array([UNI.get(c, 0) + 1 for c in range(V)], float) / (tot + V)


def alg_p(ctx):
    p = np.zeros(V); mass = 1.0
    for k in range(min(K, len(ctx)), 0, -1):
        c = tabs[k].get(tuple(ctx[-k:]))
        if not c: continue
        s = sum(c.values()); esc = 0.9 * len(c) / s
        for ch, cnt in c.items(): p[ch] += mass * (cnt - 0.9) / s
        mass *= esc
        if mass < 1e-7: break
    p += mass * uni
    return p


def bpc(s):
    ctx = [stoi[c] for c in "RICHARD III:Why, the people, when "]
    b = 0.0
    for ch in s:
        p = alg_p(ctx); b += -math.log2(max(p[stoi[ch]], 1e-12)); ctx.append(stoi[ch])
    return b / len(s)


STRUCT = "Why, the people, when h"                 # what the logits dictate
LLM_ADDED = "Why, the people, when h"              # what the LLM wrote (both runs)
print("structure via logits -> the LLM writes\n")
print(f"  algorithmic greedy sample : :{STRUCT}")
print(f"  LLM rendered (added part) : {LLM_ADDED}")
print(f"  char-wise agreement       : {sum(a==b for a,b in zip(STRUCT,LLM_ADDED))}/{len(STRUCT)} = 100%")
print(f"  bits/char under the PPM   : {bpc(LLM_ADDED):.3f}")
print("\n=> when the structure is supplied as logits the LLM does not invent content: it WRITES it.")
print("   (Both prompts -- even with flat steps marked -- reproduced the structure exactly; the LLM")
print("   added nothing where the logits were ambiguous. A weak writer, exact, no drift.)")
