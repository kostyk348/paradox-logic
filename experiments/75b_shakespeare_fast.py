"""
75b — FAST long generation on Shakespeare (readable, ~1 min).

Small char-LSTM (1 layer, hid 256), 150k chars, 1200 steps. Prints a long passage.
Honest note: our algebra accelerates STRUCTURED tasks, not free-text LM (exp 38/51).
"""
import sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
text = open("/tmp/opencode/shakespeare.txt", errors="ignore").read()[:150000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
V = len(chars); data = torch.tensor([stoi[c] for c in text])
print(f"corpus {len(text)} chars, vocab {V}")


class LM(nn.Module):
    def __init__(s, d=128, hid=256):
        super().__init__(); s.emb = nn.Embedding(V, d)
        s.rnn = nn.LSTM(d, hid, batch_first=True); s.out = nn.Linear(hid, V)
    def forward(s, x, h=None):
        o, h = s.rnn(s.emb(x), h); return s.out(o), h


def batch(T=128, n=64):
    i = torch.randint(0, len(data) - T - 1, (n,))
    return (torch.stack([data[j:j+T] for j in i]), torch.stack([data[j+1:j+T+1] for j in i]))


m = LM(); opt = torch.optim.Adam(m.parameters(), 2e-3); t0 = time.time(); L = []
for step in range(1200):
    x, y = batch(); opt.zero_grad()
    loss = F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
bpc = sum(L[-200:]) / 200 / math.log(2)
print(f"trained {time.time()-t0:.0f}s   bits/char = {bpc:.3f}\n")


@torch.no_grad()
def gen(prompt, n=1200, temp=0.65):
    out = [stoi.get(c, stoi[' ']) for c in prompt]; h = None
    for c in out[:-1]: _, h = m(torch.tensor([[c]]), h)
    for _ in range(n):
        logits, h = m(torch.tensor([[out[-1]]]), h)
        out.append(int(torch.multinomial(torch.softmax(logits[0, -1] / temp, -1), 1)))
    return "".join(itos[i] for i in out)


print("=" * 72)
print("GENERATED SHAKESPEARE (prompt 'ROMEO: ', temp 0.65)")
print("=" * 72)
print(gen("ROMEO: "))
print("=" * 72)
s = gen("JULIET: ", 500)
print(f"second sample (JULIET:): {s[:300]}...")
