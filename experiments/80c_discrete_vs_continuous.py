"""
80c — The paradox is DISCRETE. Continuous relaxation dissolves it.

Matching pennies: A wants to MATCH, B wants to MISMATCH.
  * DISCRETE strategies {0,1}: no pure Nash equilibrium -> best-response CYCLEs (period 2).
  * CONTINUOUS strategies: the quadratic game has a critical point (0,0) -> GD converges.
So representing a network as agents of a game works, and the (in)stability (paradox) lives in
the DISCRETE structure -- exactly where our holonomy/parity certificate applies.
"""
import numpy as np


def discrete_bp(steps=8):
    a, b = 0, 0; hist = [(a, b)]
    for _ in range(steps):
        b = 1 - a          # B best-responds (mismatch)
        a = b              # A best-responds (match)
        hist.append((a, b))
    period = next(p for p in range(1, 5) if all(hist[i] == hist[i + p] for i in range(len(hist) - p)))
    return hist, period


def continuous_gd(steps=400, lr=0.1, seed=0):
    r = np.random.default_rng(seed); x = r.normal(0, 1, 2); hist = []
    for _ in range(steps):
        # A: (x0-x1)^2 ; B: (x0+x1)^2  (different losses)
        x = x - lr * np.array([2 * (x[0] - x[1]) + 2 * (x[0] + x[1]), -2 * (x[0] - x[1]) + 2 * (x[0] + x[1])])
        hist.append(x.copy())
    osc = float(np.std(np.array(hist[-50:]), 0).max())
    return float(np.linalg.norm(x)), osc


h, per = discrete_bp()
print("Matching pennies: discrete vs continuous\n")
print(f"  DISCRETE best-response : no pure NE -> cycle, period = {per}   trajectory {h[:6]}")
conv, osc = continuous_gd()
print(f"  CONTINUOUS gradient    : critical point (0,0) -> converges, |x|={conv:.3f}, osc={osc:.3f}")
print("\n=> the PARADOX (no pure Nash equilibrium) is a DISCRETE phenomenon. A network as a")
print("   continuous game converges (critical point exists); as DISCRETE agents it can cycle,")
print("   and there the holonomy/cycle-parity certificate predicts instability (exp 21, 80).")
