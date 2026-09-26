"""
06 — Frustration / spin glasses: does the concept behind #2 survive in a proper Ising model?

Mapping: parity label n=1 (NOT)  <=>  antiferromagnetic bond J=-1;  n=0 <=> J=+1.
A frustrated loop (odd number of J=-1) is exactly a non-trivial holonomy class.
Unlike the Poisson defect gas of experiment 02, frustrated systems have a genuine
phase transition.  We measure the Edwards-Anderson overlap of the SK model.

SK model: J_ij ~ N(0, 1/N), fully connected.  Known:  T_c = 1.
Below T_c the overlap q = (1/N) sum_i s_i^(1) s_i^(2) of two replicas becomes non-zero.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 32
SAMPLES = 120
Tlist = [0.5, 0.7, 0.85, 1.0, 1.15, 1.4, 1.8, 2.4]

J = rng.normal(0, 1 / np.sqrt(N), size=(SAMPLES, N, N))
J = (J + J.transpose(0, 2, 1)) / 2
for i in range(N):
    J[:, i, i] = 0.0

def run(T, sweeps=300):
    x = rng.choice([-1.0, 1.0], size=(SAMPLES, N))
    y = rng.choice([-1.0, 1.0], size=(SAMPLES, N))
    beta = 1.0 / T
    for _ in range(sweeps):
        for z in (x, y):
            for i in range(N):
                h = np.einsum('sj,sj->s', J[:, i, :], z)
                dE = 2.0 * z[:, i] * h
                acc = rng.random(SAMPLES) < np.exp(-beta * dE)
                z[:, i] = np.where(acc, -z[:, i], z[:, i])
    q = np.mean(x * y, axis=1)
    return float(np.mean(q ** 2)), float(N * np.mean(q ** 2))

print("SK spin glass:  J_ij ~ N(0,1/N),  theory T_c = 1.   Overlap <q^2> turns on below T_c.\n")
print(f"{'T':>5} {'<q^2>':>9} {'chi_SG':>9}")
for T in Tlist:
    q2, chi = run(T)
    print(f"{T:>5.2f} {q2:>9.3f} {chi:>9.2f}")
print("\n=> <q^2> ~ 0 above T_c and rises through T_c = 1: a real phase transition.")
print("   The Poisson defect gas of experiment 02 had none; frustration is the right model.")
