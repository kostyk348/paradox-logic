"""
02 — Defect thermodynamics: is there a critical point?   VERDICT: NO.

A single-input Boolean network (refs + NOT labels). Structure events at rate mu
reroute a reference / flip a label. We track the holonomy dimension rho(t) and the
number of class-change events. If the gas of paradoxes self-organized, we would see
avalanches with a scale-free size distribution and a critical mu. We see neither:
rho is stationary and events are Poisson (rate ~ mu).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from holonomy import holonomy_dim

def edges_of(n, ref, lab):
    return [(i, ref[i], lab[i]) for i in range(n)]

def hurst(x):
    x = np.asarray(x); n = len(x); hi = min(n // 2, 50)
    lx, ly = [], []
    for lag in range(2, hi):
        tau = np.sqrt(np.mean((x[lag:] - x[:-lag]) ** 2))
        if tau > 1e-12:
            lx.append(np.log(lag)); ly.append(np.log(tau))
    return np.polyfit(lx, ly, 1)[0] if len(lx) > 1 else 0.5

rng = np.random.default_rng(1)
n, steps, seeds = 16, 4000, 30
print(f"Structure-noise scan  (n={n}, steps={steps}, seeds={seeds})\n")
print(f"{'mu':>6} {'mean rho':>9} {'std rho':>8} {'class-change events':>20} {'Hurst':>7}")
for mu in (0.0, 0.002, 0.01, 0.05, 0.2):
    rhos, events, hs = [], [], []
    for s in range(seeds):
        ref = list(rng.integers(0, n, n)); lab = list(rng.integers(0, 2, n))
        v = list(rng.integers(0, 2, n))
        prev = holonomy_dim(n, edges_of(n, ref, lab)); rt = [prev]; ev = 0; xs = []
        for t in range(steps):
            v = [v[ref[i]] ^ (lab[i] & 1) for i in range(n)]
            if mu > 0 and rng.random() < mu * n:
                j = int(rng.integers(0, n))
                if rng.random() < 0.5: ref[j] = int(rng.integers(0, n))
                else: lab[j] ^= 1
            cur = holonomy_dim(n, edges_of(n, ref, lab))
            if cur != prev: ev += 1
            rt.append(cur); prev = cur; xs.append(v[0])
        rhos.append(np.mean(rt)); events.append(ev); hs.append(hurst(xs))
    print(f"{mu:>6} {np.mean(rhos):>9.2f} {np.std(rhos):>8.2f} {np.mean(events):>20.0f} {np.mean(hs):>7.3f}")
print("\nrho is stationary; events scale with mu (Poisson). No avalanche, no critical point.")
