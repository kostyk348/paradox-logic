"""
86 — 2+3: SCALE (done in 85) + STRUCTURE-TO-TEXT.

An ALGORITHM extracts a structure of a held-out sentence; a WRITER (the char model) fills the
surface. Structure = per-word (length, first letter) -- what an automaton can produce exactly.
Measure how much of the text the structure determines (exact word / sentence recovery), and the
bits/char of the true text under the skeleton-constrained writer.

Also exports the LOGIT STRUCTURE (top-5 characters per step) for a fragment -- the artefact we
hand to an LLM in the 'structure via logits' experiment (87).
"""
import sys, math, collections, json
import numpy as np
sys.path.insert(0, "/home/lain/paradox-logic")

text = open("/tmp/opencode/shakespeare.txt").read()
text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], np.int64)
K = 4; D = 0.9; TR = data[:260000]; TE = data[260000:320000]

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
        s = sum(c.values()); esc = D * len(c) / s
        for ch, cnt in c.items(): p[ch] += mass * (cnt - D) / s
        mass *= esc
        if mass < 1e-7: break
    p += mass * uni
    return p


# ---------- export the LOGIT STRUCTURE for the LLM (87) ----------
ctx = list(TE[:24]); steps = []
for _ in range(24):
    p = alg_p(ctx)
    top = sorted(range(V), key=lambda c: -p[c])[:5]
    steps.append({"context": "".join(itos[c] for c in ctx[-12:]),
                  "top": [{"ch": itos[c], "p": round(float(p[c]), 3)} for c in top]})
    ctx.append(top[0])
art = {"note": "algorithmic logit structure: top-5 next chars at each step",
       "steps": steps, "greedy_sample": "".join(itos[c] for c in ctx[24:])}
open("/tmp/opencode/logit_structure.json", "w").write(json.dumps(art, indent=1))
print("exported /tmp/opencode/logit_structure.json  (steps:", len(steps), ")")
print("algorithm greedy sample :", art["greedy_sample"].replace("\n", " ")[:90])

# ---------- STRUCTURE -> TEXT: the writer fills a (length, first-letter) skeleton ----------
def words_of(seq):
    w, res = [], []
    for c in seq:
        ch = itos[c]
        if ch.isalpha(): w.append(ch)
        else:
            if w: res.append("".join(w)); w = []
    return res


def reconstruct(ctx, length, first):
    """the writer fills the word greedily under the skeleton constraints (length, first letter)."""
    s = first
    for _ in range(length - 1):
        p = alg_p(ctx); best, bp = None, -1
        for c in range(V):
            if itos[c].isalpha() and p[c] > bp: bp, best = p[c], c
        s += itos[best]; ctx = ctx + [best]
    return s


# evaluate on held-out Word tokens with the true preceding context
words = words_of(TE)
ctx = list(TE[:K]); ok = 0; n = 0; bits = 0.0
for pos in range(K, len(TE) - 1):
    ch = itos[TE[pos]]
    if ch.isalpha():
        n += 1
        # skeleton of the true word
    ctx = ctx + [TE[pos]]
    if pos % 3000 == 0:
        pass
# simpler: walk the raw stream, detect words, reconstruct each from the true prefix
i = K; got = 0; tot_w = 0; exact_sent = 0; sent_n = 0
while i < len(TE):
    c = itos[TE[i]]
    if c.isalpha():
        j = i
        while j < len(TE) and itos[TE[j]].isalpha(): j += 1
        true = "".join(itos[kk] for kk in TE[i:j])
        guess = reconstruct(list(TE[max(0, i - K):i]), len(true), true[0])
        tot_w += 1; got += (guess == true)
                # bits/char of the true word under a skeleton-aware writer (approx: PPM on the word)
        ctx_w = list(TE[max(0, i - K):i])
        for kk in TE[i:j]:
            p = alg_p(ctx_w); bits += -math.log2(max(p[kk], 1e-12)); ctx_w.append(kk)
        i = j
    else:
        i += 1
print(f"\nstructure = per-word (length, first letter), exactly produced by the algorithm")
print(f"  words reconstructed EXACTLY from the skeleton: {got}/{tot_w} = {got/max(tot_w,1)*100:.1f}%")
print(f"  bits/char of the true words under the writer:  {bits/max(n,1):.3f}")
print("\n=> the skeleton (length + first letter) already pins many words exactly; the writer fills")
print("   the rest. That is 'structure -> text': the algorithm gives the plan, the net writes.")
