"""
89 — HOLONOMY of an allocator.

Definition used here (the algebra made measurable):
  the free space has a CANONICAL normal form -- the maximal free intervals (a merge-semilattice
  of address intervals). An allocator's free space carries a HOLONOMY = its distance from the
  normal form = how many adjacent free intervals are left UNMERGED. Canonical == 0.

Consequence: with holonomy > 0 the allocator cannot serve a request that the merged free space
would satisfy -> fragmentation -> failures. The algebraic fix is to keep the state in normal form
(coalesce on free); then the free space is path-independent and allocability is maximal.

Second invariant: PHASE-ESCAPE holonomy -- O(1) counter, zero iff no block outlives its phase
(the cross-frame use-after-free of a phase arena).
"""
import random

N = 256
SIZES = [8, 16, 24, 32]


class Alloc:
    """first-fit free list. coalesce=False ~ galloc (no merging); True = canonical normal form."""
    def __init__(self, n, coalesce):
        self.n = n; self.occ = [False] * n; self.coalesce = coalesce
        self.freelist = [(0, n)]; self.live = {}

    def _insert(self, s, l):
        self.freelist.append((s, l))
        if self.coalesce:
            self.freelist.sort()
            merged = []
            for st, ln in self.freelist:
                if merged and merged[-1][0] + merged[-1][1] == st:
                    merged[-1] = (merged[-1][0], merged[-1][1] + ln)
                else:
                    merged.append((st, ln))
            self.freelist = merged

    def alloc(self, size, tag):
        for i, (s, l) in enumerate(self.freelist):
            if l >= size:
                self.freelist.pop(i)
                if l > size:
                    self._insert(s + size, l - size)
                for x in range(s, s + size):
                    self.occ[x] = True
                self.live[tag] = (s, size)
                return True
        return False

    def free(self, tag):
        s, l = self.live.pop(tag)
        for x in range(s, s + l):
            self.occ[x] = False
        self._insert(s, l)

    def holonomy(self):
        """distance from the canonical normal form = mergeable adjacent free pairs."""
        fl = sorted(self.freelist)
        return sum(1 for i in range(len(fl) - 1) if fl[i][0] + fl[i][1] == fl[i + 1][0])

    def max_interval(self):
        return max((l for _, l in self.freelist), default=0)

    def total_free(self):
        return sum(l for _, l in self.freelist)


def workload(coalesce, steps=3000, seed=0):
    rng = random.Random(seed); a = Alloc(N, coalesce); live = []
    fails = 0; holo = []; wasted = 0
    for t in range(steps):
        if live and rng.random() < 0.5:
            tag = rng.choice(live); live.remove(tag); a.free(tag)
        else:
            size = rng.choice(SIZES)
            if a.alloc(size, t):
                live.append(t)
            else:
                fails += 1
                if a.total_free() >= size:      # failure though the merged space would fit
                    wasted += 1
        holo.append(a.holonomy())
    return fails, wasted, sum(holo) / len(holo), max(holo)


print("HOLONOMY of an allocator: fragmentation = distance from the canonical free-space form\n")
print(f"{'allocator':>34} {'failures':>9} {'avoidable':>10} {'mean_holonomy':>14} {'max_holonomy':>13}")
for coalesce, name in [(False, "free-list, NO coalescing (galloc)"), (True, "canonical (coalescing)")]:
    f, w, mh, xh = workload(coalesce)
    print(f"{name:>34} {f:>9} {w:>10} {mh:>14.2f} {xh:>13}")

# the invariant, stated precisely: with coalescing the free space is the canonical form of the
# live set; without it the mergeable pairs persist (holonomy > 0).
def after_running(coalesce):
    a = Alloc(N, coalesce)
    for i in range(6):
        a.alloc(16, i)          # 6 contiguous blocks (0..95)
    a.free(1); a.free(0)        # two adjacent freed
    # the merged span 0..31 is free; can a 32-byte request be served?
    ok = a.alloc(32, "big")
    return a.holonomy(), ok
print("\nfree blocks 1 and 0 (adjacent 16+16), then request 32:")
h0, ok0 = after_running(False); h1, ok1 = after_running(True)
print(f"  no coalescing: holonomy {h0}  request 32 served: {ok0}")
print(f"  canonical    : holonomy {h1}  request 32 served: {ok1}")

# phase-escape holonomy: zero iff no block outlives its phase
def phase_escape(seq, ph):
    born, esc = {}, 0
    for p, (op, tag) in zip(ph, seq):
        if op == "alloc":
            born[tag] = p
        elif p != born.pop(tag, p):
            esc += 1
    return esc, len(born)
seq = [("alloc", 0), ("alloc", 1), ("free", 0), ("alloc", 2), ("free", 2)]
esc, leak = phase_escape(seq, [0, 0, 1, 1, 1])
print(f"\nphase-escape holonomy: cross-phase frees {esc}, leaked blocks {leak}  (zero iff clean)")
print("\n=> holonomy>0 == unmerged adjacent free space == fragmentation; the canonical form drives it")
print("   to 0 and makes allocability path-independent. The phase-escape counter is an exact O(1)")
print("   invariant flagging a block that outlives its phase.")
