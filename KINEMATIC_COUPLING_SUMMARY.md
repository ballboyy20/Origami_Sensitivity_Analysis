# Kinematic Coupling Placement — Findings & Roadmap

Branch: `Kinematic-Coupling-Placement`. This document is unrelated to
`README.md` (which describes a different, earlier sensitivity-analysis
framework in this repo — `SensitivityModel`, `HingeElement`, etc.). This
document tracks the coupling-*placement*-and-*orientation* optimization
work: `RigidBodyModel.py`, `coupling_optimizer.py`, and everything built
on top of them.

**Goal**: optimize the placement (which of *m* candidate kinematic-coupling
sites are active) and orientation (groove angle θ per active coupling) of
the *minimal* set of couplings joining panels in an origami array, to
maximize λ_min(K), K = CᵀC — in a way that survives peer review: built on
established optimization methods, validated soundly, honestly reported.

Last updated: 2026-09-10.

---

## 1. What We've Learned About the Problem

### The mathematical structure is unusually clean

For a fixed active-coupling set, exactly (validated to ~1e-15):

```
K(theta) = K0 + sum_i [ cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i ]
```

with **no cross terms** between couplings, and each coupling's own
contribution separately affine and additive (isolated per-coupling
calibration matches joint calibration to the same precision). This is
provable from the constraint-row physics (K = CᵀC is a Gram matrix over
row-blocks that are physically independent per coupling), not a
curve-fit — it's the foundation everything below relies on.

Two structural facts followed from taking this seriously:

- **θ lives on a circle of period π, not an interval.** `u(θ+π) = -u(θ)`,
  the groove faces swap, K is exactly unchanged. This is the literal
  cause of the boundary-pinning failure documented in `optimize_theta`'s
  own docstring (DE/Nelder-Mead getting stuck near 0°/180°) — a category
  error (box-constrained optimizer on a manifold-valued variable), not a
  tuning problem.
- **The "locked eigenvalue" gauge null space is topology-independent.**
  The 6-dimensional rigid-body null space of K is the *same fixed
  subspace* for any coupling subset that reaches full rank, since it
  comes from whole-assembly rigid motion, not from which couplings are
  active. This is what makes the SDP bound's gauge projection valid
  across every node of a branch-and-bound tree without recomputation.

### λ_min is nonsmooth exactly where it matters

Optimal designs generically sit at eigenvalue multiplicities (established
theory: Cox & Overton proved the analogous strongest-column problem has a
provably double least eigenvalue at its optimum). This is why DE/Nelder-Mead
struggle here — documented, well-understood behavior in the broader
eigenvalue-optimization literature, not specific to kinematic couplings.

**Most surprising finding of the whole exploration**: the two-panel
system's λ_min landscape has a *wide plateau of global maximizers* —
roughly two-thirds of randomly sampled θ triples already sit at the exact
same maximal λ_min with a numerically-zero gradient, cross-checked
directly against the real system. "The optimal design" is not a unique
point; it's a large set of equally-good designs. Report the optimal
*value*, not a single θ point, as the primary finding.

### What each optimization approach taught us

| Piece | Status | Key finding |
|---|---|---|
| Per-topology SDP relaxation (`sdp_theta_optimizer.py`) | Bound: sound. Decode: unreliable. | Zero violations across 220 subsets, but decoding the relaxed solution is fundamentally unreliable for many topologies — the relaxed optimum often sits *inside* the unit disk (no real θ behind it), confirmed via two different secondary tie-break objectives converging to the same poor decode. |
| Topology-level SDP bound (`sdp_topology_bound.py`) | Sound, but loose mid-tree. | Zero violations (root, 98 random intermediate nodes). Bound does **not** tighten monotonically with B&B depth — stays loose from root through ~10 of 12 decisions, only collapsing near leaves. For m=12, naive B&B likely won't beat brute force; may still help at much larger scale. |
| Riemannian gradient sampling (`riemannian_gradient_sampling.py`) | Validated, now integrated. | Literature-grounded (Hosseini & Uschmajew's Riemannian generalization of Burke-Lewis-Overton). Crossed the 0/π seam cleanly in tests (178.00° → 3.46° in one step). Matched/beat DE on real subsets. |
| Split-responsibility integration (`bound_and_evaluate.py`) | **Done, validated at full scale (this session).** | See §3. |

---

## 2. Literature Grounding

The problem splits into two established literatures (full citations in
session history / on request):

- **Continuous stage (θ optimization on the torus)**: nonsmooth eigenvalue
  optimization (Overton, Cox & Overton, Lewis on partly-smooth functions)
  + Riemannian gradient sampling (Hosseini & Uschmajew, SIAM J. Optim.
  2017 — the direct manifold generalization of Burke-Lewis-Overton).
  Established software: HANSO/GRANSO/PyGRANSO (independent cross-check
  option, not required).
- **Discrete stage (topology/placement selection)**: cardinality-
  constrained SDP relaxation (Frangioni & Gentile's perspective
  reformulation — literally the `a²+b² ≤ z²` construction used in
  `sdp_topology_bound.py`), sensor selection (Joshi & Boyd 2009 — same
  "choose k of m to optimize a function of a sum of per-candidate
  matrices" structure, with an established rounding+local-search fix for
  exactly the decode-unreliability problem found here), robust truss
  topology design (Ben-Tal & Nemirovski), and branch-and-bound for
  optimal experimental design (Ahipaşaoğlu — variable-bound-tightening
  branching order, directly transferable given our own bound's
  measured looseness).

This means the architecture below isn't ad hoc — each piece has a named,
citable precedent.

---

## 3. Split-Responsibility Architecture (implemented this session)

**Decision**: stop asking the SDP relaxation to produce a candidate
design (it's unreliable at that). Use it *only* for certification
(a sound upper bound); use Riemannian gradient sampling *only* for
evaluation (the actual achieved λ_min).

- `sdp_certified_bound()` — SDP **stage 1 only** (skips the stage-2
  decode entirely, since that decode is exactly the unreliable part
  being routed around — no point paying for a QP solve whose result is
  discarded).
- `riemannian_multistart()` — best-of-N Riemannian gradient sampling
  restarts (gradient sampling only certifies *local* Clarke-stationarity,
  not global optimality, so multiple independent restarts are necessary,
  not decorative — one from the topology's reference θ, the rest random).
- `bound_and_evaluate()` — combines both: certified bound `t`, achieved
  `lambda_min`, and the gap between them.

### Two real bugs found and fixed during validation (not glossed over)

1. **Multiplicity guard gap**: `optimize_theta_riemannian`'s very first
   evaluation of `theta_init` wasn't wrapped in the same multiplicity
   retry logic every *other* evaluation in the function already had.
   Invisible in the original small-scale test suite; surfaced
   immediately at full-sweep scale (hundreds of random restarts). Fixed
   at the source in `riemannian_gradient_sampling.py`, re-verified
   against the full existing test suite (all 7 checks still pass,
   identical results).
2. **No per-subset error isolation**: `bound_and_evaluate_sweep.py`'s
   worker function let any exception (including the above, and a
   separate "persistent multiplicity across a whole restart budget" edge
   case) crash the entire 220-subset `ProcessPoolExecutor` run, losing
   all progress. Fixed to mirror `exhaustive_subset_baseline.py`'s own
   established convention: catch and record per-subset failures, never
   let one bad subset take down the pool. Also added a queue-based
   restart-replacement scheme in `riemannian_multistart()` — a start
   that can't escape multiplicity is discarded and replaced with a
   fresh, unrelated random draw rather than aborting.

### Full-scale validation result (206/220 birds-foot size-9 subsets; 14 skipped — no full-rank reference found, consistent with prior known structural-infeasibility signal)

- **Soundness**: PASSED for all 206 — zero violations.
- **Certification gap** (t − achieved): median **8.5e-6**, max 6.7e-3
  (the max is a known relaxation-looseness case, not a decode failure).
- **Of the 56 subsets where DE itself failed to converge, Riemannian
  multi-start succeeded on all 56 (100%)** — resolved λ_min from 0.0012
  to 0.0093. This is the headline practical result: not just matching DE
  where DE works, but recovering every case DE couldn't handle.
- vs. DE where DE converged (150 comparable): 20 better, 19 worse, 111
  ties — statistically equivalent.
- vs. the old SDP decode: Riemannian better in 143/206 (69%).
- Best subset found matches the existing champion (0.009263 vs. 0.009259
  previously reported), with a certification gap at that topology of
  just 8e-6 — i.e., provably within 8e-6 of optimal for that specific
  design.
- **Caveat, not hidden**: 29/206 subsets (14%) showed restart spread
  > 1e-4 — meaningful disagreement across the 5 restarts. Not a
  correctness problem (soundness held throughout), but a signal that 5
  restarts may not always be enough if this becomes the primary method
  rather than a validation exercise.

Raw results: `results/bound_and_evaluate_sweep_results.{json,csv}`,
`results/bound_and_evaluate_sweep_summary.txt`.

### Infrastructure lesson (worth remembering, not just for this task)

This machine has only 7.6GB RAM. The default `cpu_count()-1` worker
convention used elsewhere in this repo (`exhaustive_subset_baseline.py`)
gave 7 workers here, each fully loading numpy/scipy/cvxpy — enough to
exhaust available memory and cause a real `MemoryError` inside a worker's
import. `bound_and_evaluate_sweep.py` now defaults to 3 workers
(overridable via `N_WORKERS` env var). Also: an empty *buffered* log file
is not evidence a background process has died — check actual process
state (or use `python -u`) before concluding a long-running job crashed.

---

## 4. Updated Next Steps

1. **(Just completed)** ~~Split responsibilities: SDP bounds for
   certification, Riemannian sampling for evaluation~~ — done, validated
   at full 220-subset scale, two real bugs found and fixed in the
   process.

2. **(Just completed)** ~~Decide whether 5 restarts is enough~~ — reran
   the 29 disagreeing subsets at n_restarts=7 (`bound_and_evaluate_restart_check.py`).
   **27/29 (93%) had already converged at n_restarts=5** (improvement
   < 1e-6); only 2/29 improved, and only slightly (max 1.1e-5). The
   restart *spread* itself stays wide even at n=7 (23/29 still > 1e-4),
   but this does **not** translate into a better best-found value — it
   reflects genuinely different local optima of varying quality on a
   nonconvex landscape, not undersampling. **Conclusion: n_restarts=5 is
   already sufficient to trust the achieved value; no need to raise the
   default.** Soundness held throughout (0 violations).
   One methodology lesson from building this check: `riemannian_multistart`
   shares one RNG object across all restarts, so requesting more restarts
   shifts every restart's internal sampling path, not just appending new
   ones — caught and corrected the experiment's framing before trusting
   results (verified directly on one subset). Worth fixing properly
   (independent child RNG per restart) if this pipeline becomes more
   central later — not urgent now given the result above.
   Raw results: `results/bound_and_evaluate_restart_check_results.{json,csv}`,
   `results/bound_and_evaluate_restart_check_summary.txt`.

3. **(Just completed — strong result)** ~~Reconsider whether full custom
   B&B is worth building at m=12~~ — tried **forward (bound-guided)
   greedy** instead (`forward_greedy_topology.py`) as an alternative to
   both brute force and B&B. Backward elimination (`prune_couplings()`,
   already in this codebase) only reached 48.8% of the known champion
   (0.004521 vs 0.009259) — but that's a genuinely different algorithm
   from forward (additive) greedy, and the classical (1−1/e)
   approximation guarantee (Nemhauser, Wolsey & Fisher 1978) applies to
   forward greedy, not backward elimination. **Forward greedy found the
   *exact* known champion topology** (`[0,1,2,3,5,6,7,8,11]`, spokes
   O-M:3,O-B:2,O-N:3,O-A:1) — **100.0% of the exhaustively-verified
   optimum**, in 7 seconds and 72 bound evaluations, vs. hours of
   exhaustive search. Method: since the true objective is identically
   zero for any subset smaller than k=9 (target_rank=18 needs ≥9
   couplings to reach full rank at all), each step scores candidates by
   `sdp_topology_bound.py`'s certified bound (best-case completion) —
   an admissible look-ahead heuristic, not a literal textbook application
   of the (1−1/e) theorem's hypotheses (stated honestly in the script's
   own docstring). Also the first real exercise of `sdp_topology_bound`'s
   shared-`registry` parameter (built earlier, never tested until now) —
   verified to give identical bounds to the non-shared path before being
   trusted here.
   **Caveat**: this is one data point (one m=12, k=9 instance) — a
   striking first result, not yet evidence this generalizes to larger
   patterns. The natural next check is whether forward greedy still
   tracks close to optimal as m grows, where exhaustive ground truth
   won't be available to confirm against directly.
   Raw results: `results/forward_greedy_topology_results.json`,
   `results/forward_greedy_topology_summary.txt`.

4. **Investigate the plateau finding as a result, not a curiosity.**
   If many θ configurations share the optimal λ_min, the real design
   question shifts to the secondary (log_product) criterion's behavior
   across that plateau — already computed elsewhere in this codebase,
   never characterized against this finding. Possible genuine
   contribution (connects to Lewis's partly-smooth-function theory). The
   forward-greedy trace above is a small supporting data point: many
   candidates were within numerical noise of each other's bound score in
   the early steps (a tight cluster at ~0.013297), consistent with a wide
   region of near-equally-good partial choices.

5. **Test forward greedy at larger scale, without exhaustive ground
   truth to check against.** Since brute-force comparison won't be
   available once m grows, this means either (a) trusting the certified
   bound's gap at the final greedy answer as the optimality certificate
   (already reported: 7.5e-6 here), and/or (b) layering local search
   (swap-based refinement, e.g. Fedorov-exchange-style) on top of the
   greedy result as a cheap correctness booster before trusting it fully
   at scale.

6. **(Ambitious/optional)** Tighten the SDP bound with a higher moment-
   relaxation level (Lasserre-hierarchy-style) if bound looseness becomes
   a real blocker at larger scale — the forward-greedy bound-guidance
   above would only get MORE useful from a tighter bound, not less.

**Recommendation if picking one**: forward bound-guided greedy just
produced the strongest practical result of this whole project (exact
champion, seconds not hours) — the highest-value next step is #5, testing
whether this holds up at larger scale where the answer isn't already
known, since that's the actual target application.
