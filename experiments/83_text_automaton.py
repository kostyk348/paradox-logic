"""
83 — FREE TEXT cracked the same way: an exact lexical AUTOMATON (trie) + learned fluency.

MIRROR of the texture result. Texture: exact invariance via the C8 orbit (a group quotient).
Text:     exact VALIDITY via the lexicon trie (a finite automaton = a discrete algebra).
The learned part supplies fluency; the algebra supplies exactness at zero learned cost.

Measure: % of generated words that are real, and bits/char, unconstrained vs trie-constrained.
"""
import sys, random, collections, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
torch.manual_seed(0); random.seed(0)

text = open("/tmp/opencode/shakespeare.txt").read()
text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
words = [w for w in text.lower().split() if w.isalpha()]
vocab = {w for w, c in collections.Counter(words).items() if c >= 2}
print(f"corpus {len(text)} chars; vocabulary {len(vocab)} words")

trie = {}
for w in vocab:
    d = trie
    for ch in w:
        d = d.setdefault(ch, {})
    d["$"] = True                      # word end


def trie_allowed(node, ch):
    return ch in node                    # a real continuation (or '$' end)


chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
V = len(chars)


class LM(nn.Module):
    def __init__(self, h=256):
        super().__init__(); self.e = nn.Embedding(V, 64); self.l = nn.LSTM(64, h, 1, batch_first=True)
        self.o = nn.Linear(h, V)
    def forward(self, x, s=None):
        y, s = self.l(self.e(x), s); return self.o(y), s


def batches(data, L=96):
    i = 0
    while i + L + 1 < len(data):
        yield data[i:i + L], data[i + 1:i + L + 1]; i += L


data = [stoi[c] for c in text[:200000]] if False else [stoi[c] for c in text[:300000]]
m = LM(); opt = torch.optim.Adam(m.parameters(), 2e-3); sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)
for ep in range(8):
    tot = n = 0.0
    for x, y in batches(data):
        x = torch.tensor(x)[None]; y = torch.tensor(y)[None]
        opt.zero_grad(); out, _ = m(x); loss = F.cross_entropy(out.view(-1, V), y.view(-1)); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tot += loss.item(); n += 1
    sched.step()
    if ep % 2 == 1: print(f"  ep{ep} train bpc {(tot / n) / math.log(2):.3f}")


def sample(n=1200, constraint=False, temp=0.8):
    x = torch.tensor([[stoi[" "]]]); s = None; out_chars = []
    node = trie; in_word = False
    for _ in range(n):
        logits, s = m(x, s)
        logits = logits[0, -1] / temp
        if constraint:
            if in_word:
                cont = [stoi[c] for c in chars if c.isalpha() and c in node]
                end = [stoi[c] for c in chars if not c.isalpha()] if "$" in node else []
                allowed = cont + end
                if not allowed: allowed = [stoi[c] for c in chars if not c.isalpha()] or list(range(V))
            else:
                allowed = [stoi[c] for c in chars if c.isalpha() and c in trie] + \
                          [stoi[c] for c in chars if not c.isalpha()]
            mask = torch.full_like(logits, -1e9); mask[[a for a in set(allowed)]] = 0; logits = logits + mask
        p = torch.softmax(logits, -1); nx = torch.multinomial(p, 1)
        ch = itos[nx.item()]; out_chars.append(ch)
        if constraint:
            if in_word:
                if ch.isalpha(): node = node.get(ch, node)
                else: in_word = False; node = trie
            elif ch.isalpha():
                in_word = True; node = trie.get(ch, trie)
            else:
                node = trie
        x = nx[None]
    return "".join(out_chars)


def valid_frac(s):
    ws = [w for w in s.lower().split() if w.isalpha()]
    return sum(w in vocab for w in ws) / max(len(ws), 1)


for tag, con in [("unconstrained ", False), ("trie-constrained", True)]:
    txt = sample(1200, con)
    print(f"\n{tag}: valid-word {valid_frac(txt)*100:5.1f}%   |  sample: {txt[:110].strip()}")
print("\n=> the trie is an exact finite automaton (a discrete algebra): under it every emitted word")
print("   is real BY CONSTRUCTION; the learned part only supplies fluency. Texture: orbit=C8,")
print("   text: trie=automaton. Same architecture: perception + an exact discrete core.")
