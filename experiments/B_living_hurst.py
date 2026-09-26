"""
B — Is the "criticality" (Hurst ~ 0.6) real?   Null model calibration.

Their estimator is the structure-function Hurst: slope of log RMS-increment vs log lag.
We calibrate it on known processes at the same series length. If the estimator is sane,
the heavy tail of the Living system is NOT an estimator artifact -> it must be checked
against a proper null before being called criticality.

(The Living system itself is ported to C in experiments/living.c, because the reference
Python implementation composes meta-rules recursively -> evaluation cost 2^depth, and its
own long-run cell aborted with KeyboardInterrupt.)
"""
import numpy as np

def hurst(x):
    x = np.asarray(x); n = len(x); hi = min(n // 2, 50); lx, ly = [], []
    for lag in range(2, hi):
        tau = np.sqrt(np.mean((x[lag:] - x[:-lag]) ** 2))
        if tau > 1e-12: lx.append(np.log(lag)); ly.append(np.log(tau))
    return np.polyfit(lx, ly, 1)[0] if len(lx) > 1 else 0.5

rng = np.random.default_rng(0)
for n in (1000, 5000):
    print(f"Null distribution at series length n={n}  (2000 runs)")
    for name, gen in (("i.i.d.  (H_true=0)", lambda: rng.standard_normal(n)),
                      ("random walk (H=.5)", lambda: np.cumsum(rng.standard_normal(n)))):
        hs = np.array([hurst(gen()) for _ in range(2000)])
        print(f"    {name:20s} H={hs.mean():.3f}+-{hs.std():.3f}  >0.6: {100*np.mean(hs>0.6):.0f}%  max={hs.max():.3f}")
    x = np.zeros(n)
    for i in range(1, n): x[i] = 0.9 * x[i-1] + rng.standard_normal()
    print(f"    {'AR(1, .9)':20s} H={hurst(x):.3f}")
    print()
