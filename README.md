# paradox-logic

**Parity holonomy of Boolean constraint networks — and why "paradox logic" is a coordinate, not a new class of computation.**

A test-driven study of one thesis: *can a system escape the "state + rule" form by treating paradox as a foundation rather than an error?* This repository formalizes the claim over GF(2), turns every assertion into a checkable experiment, and reports which parts survive.

> **Short answer.** The only non-trivial structure in the whole construction is **parity**. A "paradox" is exactly a non-trivial class in `H¹(G; ℤ/2)` of the network's negation labeling. That makes the invariant a useful *coordinate* (consistency checking, error detection, contextuality, memory) — but **not** a new class of computation, and **not** a self-organizing dynamical engine.

---

## TL;DR

| # | Direction | Verdict |
|---|-----------|---------|
| 1 | Paradox = (Mermin/GHZ) contextuality | ✅ confirmed |
| 2 | Defect thermodynamics — criticality from paradox birth/death | ❌ not found |
| 3 | Topological (parity) memory | ✅ confirmed (it is a checksum, not a memory) |
| 4 | Novelty = change of topological class | ◐ core yes, criticality signal no |
| 5 | Cohomological diagonal lemma (self-reference ⇒ paradox) | ❌ refuted |

Companion results: **XOR is the exact reduction barrier to Hopfield** (LP-verified), **Goles–Olivos period ≤ 2** for symmetric coupling, and the reference "criticality" (Hurst ≈ 0.6) is **not** established — its own long run aborted.

---

## The model

A Boolean constraint network is a graph whose edges carry a parity label `n ∈ {0,1}`:

```
edge (u, v, n)   ⇔   x_u XOR x_v = n          x_i ∈ {0,1}
```

This is exactly the *linear layer* of "paradox logic": a `NOT` node flips the value along its reference edge (`n = 1`), a `PURE` node copies it (`n = 0`). Writing the system as `A x = n` over GF(2):

> **Central fact.** A global assignment exists ⟺ `n` is a coboundary ⟺ **every cycle has an even number of 1-labels**.
> When it does not, the number of independent paradoxes is `dim coker(A)` — the **holonomy dimension**.

Two consequences drive everything below:

- **The "act" (ground a node) is a gauge fixing.** It cannot create a solution for an inconsistent system; it only selects a representative of a coset. This is why a parity-based act and a random act are *observationally identical* (verified: they agree in 100% of trials).
- **Paradox ≠ oscillation.** Inconsistency *implies* oscillation, but oscillation also occurs in consistent systems (an affine map can have both fixed points and longer cycles). Conflating the two is the single most common error in the source material.

---

## What it gives you

All of it is exact GF(2) linear algebra on `holonomy.py`:

| Function | Question it answers |
|---|---|
| `consistency(nv, edges)` | Is the network satisfiable at all? |
| `holonomy_dim(nv, edges)` | How many *independent* contradictions are there? |
| `min_groundings(nv, edges)` | How many nodes must be pinned (a minimum feedback vertex set) to fix a solution? |
| `contextual_fraction(nv, edges)` | How contextual (Mermin-like) is the model? |
| `cycle_holonomy(labels)` | The parity-check bit of a cycle |

**#3 restated.** The holonomy of a cycle is a *parity-check bit*. It is invariant under any change of *values* (the labels are structural), so it is robust on the value axis and fails only under structural edits. That is **an error-detecting code on the network**, not a memory: any graph carries a fixed holonomy, and an edge edit changes it. Useful as a **checksum / contradiction detector**, honest as nothing more.

---

## Results

### 1 — Paradox is contextuality ✅

```
cycle L=3, #NOT=2:  max-sat 3/3  holonomy 0  non-contextual
cycle L=3, #NOT=3:  max-sat 2/3  holonomy 1  CONTEXTUAL
Mermin triangle (a⊕b=1, b⊕c=1, c⊕a=1): max-sat 2/3 → contextuality
```

`max-sat < L ⟺ holonomy > 0` for all cycles `L = 3..6`. The GF(2) invariant coincides with the standard contextuality criterion, so the "paradox" is the classical shadow of the GHZ / Mermin parity proof — the sheaf-theoretic apparatus (Abramsky–Brandenburger) transfers directly.

### 2 — Defect thermodynamics ❌

Structure-noise scan (`n=16`, 4000 steps, 30 seeds): `rho` is **stationary** (0.69–0.73 at every rate), class-change events scale **linearly** with the edit rate (Poisson, not a power law), Hurst stays low (0.02–0.09). There is no avalanche, no critical point, no self-organization in the gas of paradoxes.

### 3 — Topological memory ✅ (caveat)

```
noise    parity, VALUE    parity, STRUCTURE    Hopfield, VALUE
0.05          0.000              0.138               0.377
0.10          0.000              0.246               0.897
0.20          0.000              0.385               1.000
```

The parity bit is exactly invariant to value noise and only pays under structural edits — see the caveat above.

### 4 — Novelty ≠ noise ◐

Random **value** noise (the reference's `p_bif`) produces **zero** class changes; structural edits produce many. So novelty is provably orthogonal to randomness. But the novelty rate is Poisson in the edit rate, so it is not a criticality signal.

### 5 — Diagonal lemma ❌

Self-gauge map `n'[i] = c[i] ⊕ holonomy(n)`: exhaustive search finds a **trivial-class fixed point for 60/120** base vectors. Self-reference does **not** force paradox. The "silence beyond formalization" thesis is not a theorem in this form.

### Companion A — Reduction barrier = XOR

The LP classifier (validated against OEIS A000609: 4, 14, 104 threshold functions for `n = 1, 2, 3`) shows `AND, OR, NAND, MAJ, IMPL` are LTFs and fold into one Hopfield layer; **`XOR` / `XNOR` are the only gates that escape**. Directional coupling escapes to the RNN class; Goles–Olivos gives the exact invariant:

```
SYMMETRIC  (Hopfield)  max cycle length = 2
ASYMMETRIC (RNN)       max cycle length = 6   (n = 3)
```

### Companion B — "Criticality" is not established

The Hurst estimator is sane (i.i.d. → H = 0.000, 0% > 0.6), so the Living system's heavy tail is model-side, not estimator-side. But the reference implementation composes meta-rules recursively (evaluation cost `2^depth`), and **its own long-run cell aborted** (`KeyboardInterrupt`). A bounded C port (`experiments/living.c`) gives a mean that is implementation-sensitive (0.23 → 0.58 depending on one knob). Not a regime — a parameter.

---

## Quickstart

```python
from holonomy import cycle_edges, label_cycle, consistency, holonomy_dim, min_groundings

# a pure-NOT triangle: x0⊕x1=1, x1⊕x2=1, x2⊕x0=1
edges = label_cycle(cycle_edges(3), [1, 1, 1])

assert consistency(3, edges) is False        # no global assignment
assert holonomy_dim(3, edges) == 1           # exactly one independent paradox
assert min_groundings(3, edges) == 1         # one act pins a concrete solution
```

## Reproduce

```bash
./run_all.sh            # writes every table to results/ and the figure to figures/
```

Requires Python 3 with `numpy` (+ `scipy` and `matplotlib` for `A_conjugacy.py` and the figure), and optionally `gcc` for the C experiments.

## Layout

```
holonomy.py                     the library (exact GF(2), numpy only)
experiments/
  01_contextuality.py           paradox = contextuality
  02_defect_thermodynamics.py   the failed criticality claim
  03_topological_memory.py      parity checksum vs Hopfield
  04_novelty.py                 novelty != noise
  05_diagonal_lemma.py          the refuted self-reference lemma
  A_conjugacy.py                XOR barrier + Goles-Olivos
  B_living_hurst.py             Hurst estimator null model
  living.c, hurst_null.c        fast C ports
  plot_results.py               figure
results/                        generated tables
figures/                        generated figure
```

---

## Limitations & honest negatives

- The invariant describes only the **linear layer** (`NOT`/`PURE`). As soon as `AND`/`OR` is involved, `A x = n` stops being an affine system and `H¹` no longer tracks the dynamics. This is a boundary of the model, not a missing technique.
- Directions 2 and 5 fail; direction 4 succeeds only in its core. Two of five proposed directions did not survive — reported rather than buried.
- #3 is a checksum, not a memory: it lives in the wiring, and wiring edits destroy it.
- No new class of computation was found. Everything here is `state + rule`; the contribution is a **coordinate**, not computability.

## Related work

- Contextuality as sheaf cohomology — Abramsky & Brandenburger (2011); Mermin parity proofs.
- Limit cycles of threshold networks — Goles & Olivos (1980).
- Boolean threshold functions — OEIS A000609.
- Frustration in spin glasses (see Roadmap) — Toulouse (1977).

## Roadmap

Where this line can go next, in the order I would pursue it:

1. **Classification, not dynamics.** Since the class is a coordinate, use `([n], dim coker)` to *separate* networks into regimes that cannot be reached from one another. Purely static, cheap, no criticality needed.
2. **Frustration / spin glasses.** Parity paradoxes are exactly *frustrated loops* in an Ising model. The thermodynamics that failed in #2 may succeed here, because frustrated systems have a genuine phase transition (replica symmetry breaking) that a Poisson defect gas does not.
3. **Non-abelian holonomy.** Replace `{0,1}` by a non-abelian group `G`. The obstruction becomes a path-dependent word — real hysteresis and memory, not a static bit.
4. **A real diagnostic library.** `consistency` / `holonomy_dim` / `min_groundings` as tooling for XOR-SAT-like constraint systems, replicated-state reconciliation, and parity-fault diagnosis.

## License

MIT — see [LICENSE](LICENSE).
