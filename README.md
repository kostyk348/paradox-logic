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

## Library — the two levels

```python
from algebraic import verify, min_repair, consistency, holonomy_dim, l_star, DFA, constrained_decode, product, cascade
from algebraic.nn import SemigroupRNN, AlgebraHead, Hybrid          # torch, optional

# ---- LEVEL 2: programs as monoids (verify a process network by cohomology) ----
verify(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)])   # -> {'consistent': False, 'contradictions': 1, ...}
min_repair(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)])   # minimum clauses to drop (NP-hard in general)
P = product(dfaA, dfaB)                          # compose processes;  cascade(A, B, couple) for wreath

# ---- LEVEL 1: the algebra as a drop-in layer for a network ----
dfa = l_star([0, 1], lambda s: sum(s) % 2 == 0)  # discovers parity -> 2 states, type=group
model = SemigroupRNN(n_tokens=7, n_states=7)     # learnable finite automaton (exact at test time)
head = AlgebraHead(7, 7)                         # exact state -> label; Hybrid = trunk + head
```

Tests: `PYTHONPATH=. python3 tests/test_algebraic.py` (25/25 — Mermin, min-repair, L*, product, cascade, repair, Jev JSON).

### Structured output — `algebraic.repair` and `algebraic.jev_json`

Two pure modules that make an LLM's structured output cheap and trustworthy, exposed over the
algebra MCP (`mcp/algebra_mcp.py`):

```python
from algebraic.repair import json_repair, lexer_state
json_repair('{"a": 1, "b": [2, 3')      # -> ('{"a": 1, "b": [2, 3]}', True)

from algebraic.jev_json import generate, validate, emit_logits, check_schema
obj, rep = generate(schema, context=user_text, logits="self")          # its own logits
obj, rep = generate(schema, logits=lambda q, k, opts: model.logits(q)) # a real model
obj, rep = generate(schema, values=extracted)                          # content already known
```

**Typed decisions, not decoding.** `enum` / `boolean` / key presence / array length / the
`oneOf`-`anyOf` branch are CLOSED choices: they are read from letter logits in one forward pass
with ZERO decoded tokens (`algebraic.jev.read_choice`). Only `string` / `integer` / `number`
slots are produced — from `values`, or synthesised under `min`/`max`/length bounds. A
closed-choice-only schema costs literally 0 decode steps.

**The logits interface.** `logits` may be a dict, a callable `(question, key, options)`, or the
string `"self"` — the module's own deterministic lexical scorer (a heuristic, explicitly not a
language model; its softmax `gap` tells you when a decision is unreliable). `emit_logits(schema,
context=...)` returns the logits for every closed decision (a logits request/answer you can
inspect or override and feed straight back). Decision keys are `path` (value), `path?` (key
presence), `path[]` (array length), `path@branch` (union branch); paths are dotted, so
`home.city` and `work.city` never collide.

**Fail-closed.** `check_schema` lints the schema against the implemented subset
(`type`/`properties`/`required`/`additionalProperties`/`items`/`minItems`-`maxItems`/
`minimum`-`maximum`/`minLength`-`maxLength`/`enum`/`const`/`oneOf`/`anyOf`/`$defs`/`$ref`).
Anything else — `pattern`, `format`, tuple `items`, unknown keywords, unresolvable or recursive
`$ref` — is refused with a path-labelled error, and `validate` reports those schema errors
instead of claiming a document is valid. `$ref` chains are followed, so a definition reused
twice generates correctly twice.

`json_repair` keeps the longest prefix ending after a *complete* JSON value and closes it
(100% of truncation points of the test corpus become valid JSON), is string-aware, drops
trailing commas, unwraps ```` ```json ```` fences and single-quoted JSON — and on failure
returns the **original** text with `valid=False`, never a mutated non-JSON string.
MCP tools: `json_generate`, `json_logits`, `json_validate`, `json_check_schema`, `json_plan`.
`experiments/78_jev_json.py` measures the whole thing: 100% valid over 5000 random schemas,
0% for unconstrained sampling, 0 decode steps for closed-choice-only schemas, and every
injected schema violation caught.



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
groupholonomy.py                group-valued holonomy (any finite group Γ)
experiments/
  01_contextuality.py           paradox = contextuality
  02_defect_thermodynamics.py   the failed criticality claim
  03_topological_memory.py      parity checksum vs Hopfield
  04_novelty.py                 novelty != noise
  05_diagonal_lemma.py          the refuted self-reference lemma
  06_frustration.py             spin-glass order parameter (criticality rescued)
  07_nonabelian.py              non-abelian holonomy, path dependence
  08_classification.py          invariant vs behaviour
  09_diagnostics.py             exact k-XOR-SAT, contradiction localisation
  10_group_holonomy.py          detection ladder + error localisation
  11_repair_hardness.py         consistency is easy, optimal repair is hard
  12_holonomy_code.py           detection / localisation / distance
  13_pose_graph.py              SO(3) loop-closure error localisation
  14_anyons.py                  braiding as computation (topological protection)
  15_sapir_whorf.py             language = bundle, thought = global section
  16_incommensurable.py         expressivity gap between languages
  17_expressiveness.py          expressiveness law: |Γ|^(-cycle rank)
  18_meaning_gauge.py           meaning = gauge invariance
  19_nn_breakthrough.py         holonomy fixes the message-passing blind spot
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

## Extended checks — the four roadmap items, now tested

| Roadmap item | Experiment | Outcome |
|---|---|---|
| Frustration / spin glasses | `06_frustration.py` | concept rescued: a real order parameter turns on |
| Non-abelian holonomy | `07_nonabelian.py` | finer than parity; path-dependent → memory |
| Classification by invariant | `08_classification.py` | exact for paradox, incomplete for behaviour |
| Diagnostic library | `09_diagnostics.py` | exact k-XOR-SAT, packed GF(2) rank |

**06 — Frustration.** Parity labels are Ising couplings (`n=1 ⇔ J=-1`); a frustrated loop is a holonomy. In the SK model the Edwards–Anderson overlap turns on through `T_c = 1`:

```
 T     <q^2>   chi_SG
 0.50  0.204    6.53
 1.00  0.062    1.99
 2.40  0.040    1.28
```

`<q²>` rises ~5× as `T` crosses `T_c`. Unlike the Poisson defect gas of experiment 02, frustration has a genuine order parameter — the concept behind the failed criticality claim is not empty. (Caveat: a crude Metropolis smears the transition; a sharp `T_c` needs parallel tempering.)

**07 — Non-abelian holonomy.** With `G = S₃`, the commutator labeling `[a, b, a⁻¹, b⁻¹]` on a 4-cycle has trivial abelianization (so **parity says "consistent"**) yet a non-trivial group holonomy — no global assignment exists. Non-abelian holonomy is strictly finer than parity. Path holonomies differ between paths between the same nodes: state is path-dependent, which is the memory that #3 lacked.

**08 — Classification.** Over random single-input networks the invariant decides **paradox exactly** (`dim coker = 0 ⇔ a fixed point exists`), but it does **not** decide behaviour: at `dim = 0` a large share of systems still oscillate (multistability). Near-perfect classifier of *paradox*, poor classifier of *dynamics*.

**09 — Diagnostics.** k-XOR-SAT (any arity) solved exactly by one packed GF(2) rank computation; reports satisfiability, free variables (= minimum groundings), and localises contradictory clauses. Exact; the Python reference is `O(n²)` in practice (see timings in `results/09_diagnostics.txt`).

## The non-abelian layer — new mathematics

Replace the labels `{0,1}` by a general (possibly non-abelian) group `Γ`. An edge `(a, b, g)` means `x_a = g · x_b`. A global assignment exists iff the labelling defines a **trivial flat Γ-bundle**, i.e. iff the holonomy representation `ρ : π₁(G) → Γ` is trivial. When `Γ = ℤ/2` this is exactly the parity invariant above. For non-abelian `Γ` it is **strictly finer**, and it yields three results, each verified in `experiments/10,11`:

**Detection ladder.** `abelian ⊂ non-abelian`, strictly. An even corruption (a 3-cycle) leaves the abelianization untouched, so a ℤ/2 check is blind — while the S₃ holonomy finds it:

```
random S₃-labelled graphs, one EVEN corruption per trial (400 trials)
  ℤ/2 (abelianization) detects:   0/400   (0%)
  S₃  (non-abelian)    detects: 398/400   (100%)
  corrupted edge localised:     377/400   (94%)
```

Parity is the shadow of the real invariant.

**Localisation.** The holonomy does not merely detect a corrupted label — it *localises* the edge (94% exact recovery). This is loop-closure error localisation in pose-graph SLAM (with `SO(3)` rotations in place of group elements), and single-error correction in a group-labelled network.

**Complexity ladder.** *Consistency* is polynomial for any fixed finite `Γ` (`O(E)` group operations). *Optimal repair* is not: the minimum number of labels to change is the **frustration index** `= m − MAX-CUT`, NP-hard already for `ℤ/2`.

```
signed graphs (ℤ/2)                 consistent   max-sat  frustration
 V=8   m=16   poly (µs)                no          13          3
 V=14  m=40   poly (µs)                no          31          9
```

So: **count the paradoxes in polynomial time; fix them minimally — hard.**

**Frontier.** Non-abelian holonomy *is* braiding. The one setting where holonomy is known to *compute* is topological quantum computation (non-abelian anyons); the classical labelled graphs here are its shadow.

## Computation, errors, and language

### Holonomy code — detection, localisation, distance (`12`)

A consistent labelling is a codeword; a corrupted edge is an error. The **distance** of the code is the girth: the minimum undetectable error is a non-zero coboundary, whose least weight is the shortest cycle — so a single error on a **bridge** is invisible (no cycle runs through it); larger cycle rank makes single-error localisation generic.

```
 V    p  cycle-rank   detect  localise   bridge-err invisible
 7  0.2    4.3          87%     64%            40
 9  0.5   17.8          99%     99%             2
12  0.3   19.9         100%     97%             0
```

### Real SO(3) pose graph (`13`)

The applied face: nodes are orientations in `SO(3)`, edges are measured relative rotations, one loop closure is corrupted.

```
trials 200:  detected 200/200 (100%)   localised 196/200 (98%)
```

The non-abelian invariant localises a single bad loop closure exactly. An abelian check cannot: rotations do not commute.

### Anyons — holonomy computes (`14`)

Braiding non-abelian anyons *is* holonomy over the configuration space. Mapping `B_n → S_n`, a braid word becomes a group element = a gate:

```
σ0σ1 = (1,2,0)   vs   σ1σ0 = (2,0,1)     -> order matters (non-commuting gates)
```

The gate depends only on the braid class, not the geometry — the computation is topologically protected. This is the one setting where holonomy provably *computes* (topological quantum computation); the classical graphs here are its shadow.

### Sapir–Whorf, made precise (`15`)

Read a **language** as a group-labelled graph: concepts are nodes, relations are edges; a **thought** is a globally consistent assignment (a coherent belief-set). A **translation** is a gauge transformation (`x_v ↦ c·x_v`), which changes the surface labels but not the holonomy class.

```
languages = labellings of a 4-cycle over S_3
  holonomy = identity        -> 6 coherent thoughts
  holonomy = transposition   -> 0   (paradoxical language)
  holonomy = 3-cycle         -> 0
```

The coherent-thought set depends **only on the conjugacy class of the holonomy**. Hence: **strong Whorf is false** — there is a translation-invariant core (the holonomy class) shared by all languages in one gauge orbit — and **weak Whorf is true** — the class bounds the thinkable, and a non-trivial class admits no global thought at all. The number of "meaning-classes" of a language is the number of conjugacy classes of its group.

Three further results sharpen this (`16`–`18`):

**Incommensurable languages (`16`).** The strong Whorf hypothesis survives only at the level of the *relation algebra* (the group), not the labels. Two S₃-languages — one consistent, one paradoxical — have **identical ℤ/2 images**, so a parity language assigns them the same meaning while their truth is opposite. A proposition expressible in S₃ is *inexpressible* in ℤ/2: an expressivity gap, not a translation gap. (ℤ/2 has 2 conjugacy classes, S₃ has 3; the third class is the untranslatable content.)

**Expressiveness law (`17`).** The fraction of labelings that are globally coherent is `≈ |Γ|^(−cycle_rank)`, `cycle_rank = E − V + 1`:

```
 Γ     rank   measured   |Γ|^-rank
Z/2      5     0.0315     0.0312
Z/2     14     0.0001     0.0001
S_3      2     0.0285     0.0278
```

A tree (rank 0) constrains nothing; each independent cycle multiplies the constraint by `|Γ|`. **Thought is bound in proportion to the cycle structure of the grammar.**

**Meaning = gauge invariance (`18`).** Under `x_v ↦ c·x_v` the labels conjugate and the coherent-thought set maps bijectively to itself: number of thoughts, holonomy conjugacy class, and the conjugation law are all preserved (2000/2000 trials). The invariant content of a language is exactly its gauge orbit — a working definition of meaning as invariance, not reference.

## What this gives neural networks

The XOR barrier of this repository is exactly the **1-WL / message-passing barrier**: a global parity property is *not* a function of any permutation-invariant local statistic. `19` makes it concrete on a signed-graph **balance** task (is every cycle even?):

```
local statistics (7 features):   test acc = 0.508     <- chance
holonomy only (1 feature, O(E)): test acc = 1.000
```

The breakthrough is narrow but provable: **a global topological invariant, computed in `O(E)`, closes an architectural blind spot that more depth and more data cannot.** Concrete directions:

- **Holonomy-augmented GNNs.** Add an `O(E)` cycle-holonomy readout (spanning tree + fundamental cycles). Message passing over-squashes and cannot count cycles / detect frustration; this supplies the missing *global* invariant at negligible cost. Testable on tasks with topological ground truth (molecules, circuits, constraint systems, pose graphs).
- **Group-valued sequence models.** Let the state be a group element and update `h_t = g_t · h_{t−1}` with non-abelian `Γ`. This is path-dependent memory that does not saturate — the non-abelian holonomy of experiment `07`, used as a sequence-model prior.
- **Meaning = invariance as a training principle.** Learn on gauge-invariant features only; the invariant content is the gauge orbit, so gauge-fixing removes a large weight-space redundancy.
- **Topologically protected memory.** The holonomy class is discrete and cannot be moved by value perturbations — a candidate carrier for continual-learning memory that gradients do not overwrite.

## Open questions

- The invariant describes only the linear layer; the `AND`/`OR` boundary is where it stops tracking dynamics. Is there a natural *non-linear* refinement (e.g. matroid / tropical) that extends the coordinate past that boundary?
- The non-abelian case gives path dependence but was only checked on toy cycles. What is the right `G` for real constraint systems, and does `H¹(G; G)` classify their regimes?
- If frustration is the right dynamical picture, does the frustrated Ising model of the *actual* reference network reproduce its measured Hurst — or is even that still a parameter, not a regime?

## The full arc — experiments 01–31

**Parity holonomy (01–11).** paradox = contextuality (`01`); defect thermodynamics fails (`02`); parity checksum (`03`); novelty ≠ noise (`04`); self-reference lemma refuted (`05`); frustration (SK spin glass) (`06`); non-abelian holonomy + path dependence (`07`); classification (`08`); exact k-XOR-SAT (`09`); detection + localisation (`10`); consistency P vs optimal repair NP-hard (`11`).

**Computation, errors, language (12–19).** holonomy code, distance = girth (`12`); SO(3) loop-closure localisation, 98% (`13`); anyons / braiding (`14`); Sapir–Whorf: language = bundle, thought = section (`15`); incommensurable languages (`16`); expressiveness law `|Γ|^(−cycle rank)` (`17`); meaning = gauge invariance (`18`); holonomy fixes the message-passing blind spot (`19`).

**Neural networks (20–31).**
- `20` group-valued RNN: parity with **14 params** at 1.00 vs GRU/LSTM (25–34k) at chance; MNIST control shows no algebraic prior helps there.
- `21` game theory: paradox = no pure Nash equilibrium; the act = a mixed strategy.
- `22` MNIST the group way: train upright, C4-invariance generalises to all rotations.
- `23` real C4 G-CNN (group convolutions); rotation classification = group task (0.98).
- `24` topological invariant of MNIST: holes b1 (8 → 2 holes).
- `25–26` semigroup RNN = differentiable automaton (Myhill–Nerode); recovers the syntactic monoid; exact at any length.
- `27` the `Z/n` barrier is robust (curriculum, straight-through do not crack it).
- `28` real text (lexer on source code): SemigroupRNN 0.70 vs LSTM 0.99 — the honest failure.
- `29` **decisive:** a cyclic scaffold init solves `Z/2..Z/13` exactly; a *wrong* scaffold gives no help.
- `30` MNIST prior = the symmetry group; the wrong subgroup (C2) fixes only the axes it covers.
- `31` algebra search: automaton minimisation discovers the monoid size; order search finds the group.
- `32` continuous symmetry: train upright, `C_n` (n=12,36) is flat across all angles (~SO(2)).
- `35` **steerable:** polar-FFT magnitude features are *exactly* rotation-invariant at FFT cost (0.75 flat over 0–180°, single pass, no n× averaging).
- `36` **algebra search without a prior:** Angluin's `L*` recovers the syntactic monoid size and type (group / aperiodic) from queries alone — parity 2, mod7 7, reset 5, contains11 3, all exact.
- `37` **generation, perfect:** on a mixed language (balanced brackets ∧ even #a), an LSTM samples 59% valid; automaton-constrained decoding gives **100% valid by construction**.
- `34` text generation (plain balanced brackets) is the honest counter-example: unconstrained, the LSTM beats the small state model.

## Headline result: the prior must match the algebra

| domain | algebra | right prior | result |
|---|---|---|---|
| sequences | `Z/n` | cyclic scaffold | random 0.21/0.16/0.09 → **1.00** (k=5/7/11/13) |
| images | symmetry `C4` | C4-equivariance | CNN 0.17 → **0.88** on all rotations |
| wrong prior | another algebra | — | `Z/n`: 0.00; MNIST 90°: 0.13 |

**Representation is cheap and exact; the wall is search. A matched prior removes the wall; capacity does not substitute for the right algebra.** This is Whorf's thesis in machine form: the grammar determines what is learnable.

## What it gives neural networks (fewer parameters)

Where the algebra matches, the parameter count collapses:

```
task        algebraic model        LSTM / GRU            ratio
parity        14 params            33 666 params        ~2400x
sum mod 7    154 params            34 631 params         ~225x
```

and for images the win is *zero-shot invariance* (train upright only) with the **same** parameter count as a CNN — no rotation-augmentation data needed. The honest limits: it only helps where the algebra matches; on real text the finite-state model lost to an LSTM, and discovering larger algebras (Z≥5) by gradient descent remains the open problem that a scaffold solves.

## License

MIT — see [LICENSE](LICENSE).
