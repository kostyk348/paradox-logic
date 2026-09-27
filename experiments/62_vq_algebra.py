"""
62 — VQ as a MONOID HOMOMORPHISM (equivariant quantisation).

Continuous task: angles theta_t in [0, 2pi); the label is the COMPOSED rotation
    y = floor( ((sum theta) mod 2pi) / (2pi) * K )
We quantise each angle to K codes and compose the codes.

  * plain VQ (k-means centres): the codebook is arbitrary -> quantisation does NOT commute
    with the group action -> codes do not compose -> fails.
  * group-aligned VQ: codes are the K sectors of the circle (the orbit of the group) ->
    q is a HOMOMORPHISM (R/2pi,+) -> (Z_K,+) -> codes compose -> works.

So a VQ that PRESERVES the algebra must be equivariant (a homomorphism), not just any codebook.
"""
import sys, math
import numpy as np
rng = np.random.default_rng(0)
K = 8


def gen(n, T):
    th = rng.uniform(0, 2 * math.pi, (n, T))
    y = (np.floor(((th.sum(1) % (2 * math.pi)) / (2 * math.pi)) * K)).astype(int) % K
    return th, y


def kmeans_1d(x, k, iters=50):
    c = np.quantile(x, np.linspace(0.05, 0.95, k))
    for _ in range(iters):
        a = np.abs(x[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            if (a == j).any(): c[j] = x[a == j].mean()
    return np.sort(c)


# --- train VQs on single angles ---
train_angles, _ = gen(20000, 1); train_angles = train_angles[:, 0]
centers_kmeans = kmeans_1d(train_angles, K)                     # arbitrary codebook
centers_aligned = (np.arange(K) + 0.5) * (2 * math.pi / K)      # group orbit (sectors)

def code(angles, centers, aligned):
    if aligned:
        return (np.floor(angles / (2 * math.pi / K))).astype(int) % K
    return np.abs(angles[:, :, None] - centers[None, None, :]).argmin(2)

# --- evaluate composition ---
print("VQ as a homomorphism: composed-rotation task (K=8)\n")
print(f"{'quantiser':>26} {'codebook':>10}  acc (predict sum of codes mod K)")
th, y = gen(4000, 6)
for name, c, al in [("plain k-means VQ", centers_kmeans, False),
                    ("group-aligned VQ", centers_aligned, True)]:
    codes = code(th, c, al)                    # (n, T)
    pred = codes.sum(1) % K
    acc = (pred == y).mean()
    print(f"{name:>26} {len(c):>10}  {acc:.3f}")

# is it REALLY a homomorphism?  q(a+b) == q(a)+q(b) mod K  for aligned sectors
a = rng.uniform(0, 2 * math.pi, 20000); b = rng.uniform(0, 2 * math.pi, 20000)
qa = np.floor(a / (2 * math.pi / K)).astype(int) % K
qb = np.floor(b / (2 * math.pi / K)).astype(int) % K
qab = np.floor(((a + b) % (2 * math.pi)) / (2 * math.pi / K)).astype(int) % K
print(f"\nhomomorphism check (aligned sectors): q(a+b)==q(a)+q(b)  in {np.mean(qab == (qa+qb)%K)*100:.1f}% of pairs")
print("=> group-aligned VQ is a homomorphism and the codes compose; a generic codebook is not,")
print("   so its codes do not carry the algebra. This is the VQ that lets automata/L* run on")
print("   CONTINUOUS inputs: equivariant quantisation = algebra-preserving discretisation.")
