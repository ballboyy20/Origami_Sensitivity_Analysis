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

2. **Decide whether 5 restarts is enough.** The 14% meaningful-disagreement
   rate found above is the natural next thing to pin down before trusting
   this as the primary evaluation method — e.g. rerun the 29 disagreeing
   subsets with more restarts and see whether the achieved value keeps
   improving or has genuinely converged.

3. **Reconsider whether full custom B&B is worth building at m=12.** The
   depth-tightness data says the bound doesn't prune well until near the
   leaves at this problem size. With Riemannian evaluation now validated
   and fast, "near-brute-force with Riemannian replacing DE at each leaf"
   is probably simpler and competitive with a B&B tree whose bound won't
   cut much here. Revisit B&B only if targeting larger patterns where
   brute force stops being an option at all — and if so, use
   Ahipaşaoğlu's variable-bound-tightening branching order, not a naive
   branch order.

4. **Investigate the plateau finding as a result, not a curiosity.**
   If many θ configurations share the optimal λ_min, the real design
   question shifts to the secondary (log_product) criterion's behavior
   across that plateau — already computed elsewhere in this codebase,
   never characterized against this finding. Possible genuine
   contribution (connects to Lewis's partly-smooth-function theory).

5. **(Ambitious/optional)** Tighten the SDP bound with a higher moment-
   relaxation level (Lasserre-hierarchy-style) if the 44%-loose root
   bound becomes a real blocker for a future B&B effort.

**Recommendation if picking one**: #2 (restart-count validation) is
cheap and directly de-risks trusting today's integration as the primary
method going forward. #3 is the next architecturally significant
decision, but should wait until #2 gives a clearer picture of how
reliable "achieved" values from this pipeline really are.
