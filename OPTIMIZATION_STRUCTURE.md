# Optimization Structure

Branch: `Kinematic-Coupling-Placement`. Written 2026-10-26 from the code
as it stands (including uncommitted changes). Companion to
[KINEMATIC_COUPLING_SUMMARY.md](KINEMATIC_COUPLING_SUMMARY.md), which has
the findings and history; this document only describes how the pieces fit
together.

**Problem.** Choose which candidate couplings are active (and, with mixed
types, which kind sits at each site), then the groove angle θ of each
active V-groove, to maximize λ_min = smallest *locked* eigenvalue of
K = CᵀC (the 6 rigid-body modes skipped). Designs must be exactly
constrained: total rows = target_rank = total_dofs − 6, C at full rank.

---

## 1. The whole pipeline at a glance

```
SETUP
├─ Build the system: panels + candidate couplings                 (RigidBodyModel)
│   └─ [mixed types] expand each site into flat / V / tet candidates (coupling_types)
├─ SetEvaluator: one per system, caches every leaf bound + evaluation (swap_search)
│   └─ per-coupling registry {K0_i, Kc_i, Ks_i}, built once        (sdp_topology_bound)
└─ verify_symmetry for each proposed mirror permutation            (branch_and_bound)

STAGE A — starting incumbent (heuristic, no guarantee)
├─ forward_greedy: add the coupling with the best node bound, k times
└─ swap_search: best-improvement one-swap local search on complete sets

STAGE B — branch-and-bound (BranchAndBound.run)
└─ LOOP: pop node (forced_active, forced_inactive) from a depth-first stack
    ├─ 1. cardinality: rows forced vs. rows reachable vs. budget
    ├─ if the node has only one possible completion → LEAF:
    │   ├─ symmetry check
    │   ├─ leaf bound        SetEvaluator.bound    → bound_node → SDP (disk relaxation)
    │   ├─ skip if bound ≤ incumbent·(1+tol)
    │   └─ leaf evaluation   SetEvaluator.evaluate → bound_and_evaluate
    │        ├─ no V-grooves  → exact eigenvalue, no optimization
    │        └─ otherwise     → SDP stage-1 bound + Riemannian multistart (5 restarts)
    │                              └─ gradient sampling loop on the θ-torus
    ├─ 2. cut counting
    ├─ 3. symmetry
    ├─ 4. generic rank (random-θ draws)
    ├─ 5. node bound: topology SDP (perspective relaxation of z, disk relaxation of θ)
    └─ branch on most-fractional z: push "off" and "on" children
   END: incumbent + certified upper bound + gap

OPTIONAL / STANDALONE (not called by the B&B)
├─ polish_tied_restarts: stage-2 log_product tie-break among equal-λ_min angles
├─ arc_refine.LeafRefiner: arc branching to tighten one loose leaf bound
└─ moment_leaf_bound: order-1/2 moment relaxations of one leaf
```

---

## 2. Setup

- **System model** — [RigidBodyModel.py](RigidBodyModel.py). Each panel
  has 6 DOFs; each `KinematicCoupling` adds one constraint row per contact
  normal (flat 1, V-groove 2, tetrahedral socket 3), and
  `build_constraint_matrix` stacks the rows of the active couplings into C.
  Only V-grooves depend on θ; a flat and the evenly spaced socket give the
  same K at every θ.
- **Mixed-type candidates** — `TypedCandidates`
  ([coupling_types.py:34](coupling_types.py#L34)). Expands every site into
  one candidate per kind (index = site × n_kinds + kind), with `rows` and
  `site_of` arrays. A valid design has at most one candidate per site and
  exactly target_rank rows. Also lifts site symmetries to candidate
  permutations (`lift_perm`).
- **Affine calibration** — `calibrate`
  ([test_K_theta_structure.py:64](test_K_theta_structure.py#L64)). Finds
  K(θ) = K0 + Σ [cos 2θᵢ Kcᵢ + sin 2θᵢ Ksᵢ] exactly from three evaluations
  of K per coupling (θᵢ = 0, π/2, π/4). Every bound and the Riemannian
  gradient are built on this.
- **Per-coupling registry** — `build_coupling_registry`
  ([sdp_topology_bound.py:115](sdp_topology_bound.py#L115)). Calibrates
  each candidate alone, once, so any node's K can be assembled by adding
  registry entries instead of recalibrating.
- **Gauge projection** — `_locked_subspace_projection`
  ([sdp_theta_optimizer.py:171](sdp_theta_optimizer.py#L171)). From K at a
  full-rank reference θ, takes the eigenvectors above the 6 rigid-body
  modes as a basis P; PᵀKP then has exactly the locked eigenvalues, so the
  SDPs can maximize its smallest eigenvalue directly.
- **Leaf cache** — `SetEvaluator`
  ([swap_search.py:53](swap_search.py#L53)). Holds the system, registry
  and target_rank, and caches `bound(set)` and `evaluate(set)` per set,
  seeded from the set itself, so a set gets the same answer whichever
  search reaches it. Shared by swap search and B&B.
- **Symmetry verification** — `verify_symmetry`
  ([branch_and_bound.py:94](branch_and_bound.py#L94)). Accepts a
  permutation only if reflecting the contact points and face normals maps
  each candidate exactly onto its partner, and sampled designs and their
  images have equal certified bounds and feasibility. Riemannian values
  are deliberately not used (restart noise can reject a valid symmetry).

---

## 3. Stage A — starting incumbent

- **Forward bound-guided greedy** — `forward_greedy`
  ([forward_greedy_topology.py:82](forward_greedy_topology.py#L82)).
  Starts from an empty set (or `forced_active_start`, e.g. leaf-panel
  couplings from `mandatory_leaf_couplings`) and repeatedly adds the
  candidate whose node bound (best-case completion) is highest, until k
  are active. V-grooves only: it does not pass `rows`/`sites`.
- **One-swap local search** — `swap_search`
  ([swap_search.py:134](swap_search.py#L134)). Each pass bounds every
  swap (one active out, one inactive in), discards infeasible swaps and
  those whose bound can't beat the current λ_min, evaluates survivors in
  descending bound order until no remaining bound can beat the best found,
  and accepts the best swap if it improves by more than 1e-4 relative.
  Stops at a one-swap local optimum. With mixed types, a swap that changes
  the row count or doubles up a site is rejected as infeasible by the
  bound, so only like-for-like swaps survive.

In `bnb_validation.py` the incumbent is greedy → swap search. In
`mixed_types_validation.py` it is the all-V champion (birds-foot) or none
(two-panel); greedy and swap search are not run there.

---

## 4. Stage B — branch-and-bound

`BranchAndBound.run` ([branch_and_bound.py:289](branch_and_bound.py#L289)).
Two modes: V-grooves only (budget = 2k rows) or mixed types (pass `rows`
and `site_of`; budget = target_rank rows, at most one candidate per site).

**Start.** The incumbent is evaluated once (if given). The root node has
`fixed_active` forced on and their same-site siblings forced off. Search is
depth-first from a stack, up to `max_nodes` (200,000).

**Per node, in this order:**

1. **Cardinality** — `_completion`
   ([branch_and_bound.py:223](branch_and_bound.py#L223)). Prunes the node
   if the forced-on rows exceed the budget, or if even the largest
   candidate at every free site can't reach it.
2. **Leaf normalization** ([branch_and_bound.py:319](branch_and_bound.py#L319)).
   If the forced rows already equal the budget, or the only way to reach
   it is to take the largest candidate at every free site, the node has
   exactly one completion and is handled as a leaf (below); steps 3–6 are
   skipped.
3. **Cut counting** — `_cut_ok`
   ([branch_and_bound.py:235](branch_and_bound.py#L235)). Applied only
   when budget = target_rank. Prunes if forced-on rows on any one crease
   exceed 6; or, for any split of the panels into two groups, the
   still-possible rows crossing the split are fewer than 6, or the forced-on
   rows inside a group exceed 6 × (group size − 1).
4. **Symmetry** — `_symmetry_ok`
   ([branch_and_bound.py:264](branch_and_bound.py#L264)). Keeps only the
   lexicographically largest design of each symmetric family, comparing the
   on/off/undecided vector with its image and pruning only once the
   decided entries make it certain the image is larger.
5. **Generic rank** — `_reference`
   ([branch_and_bound.py:279](branch_and_bound.py#L279)). Draws up to 3
   random θ vectors; if forced-on ∪ undecided never reaches full rank, the
   node is pruned (rank is generic in θ, so failure is structural). The
   first full-rank θ found becomes the reference for the SDP's gauge
   projection.
6. **Node bound** — `sdp_topology_bound`
   ([sdp_topology_bound.py:214](sdp_topology_bound.py#L214)). Maximizes t
   with PᵀK(z, x)P − tI ⪰ 0, where each forced-on coupling's
   (cos 2θ, sin 2θ) is relaxed to the unit disk and each undecided
   coupling gets a relaxed on/off z ∈ [0, 1] with the perspective cone
   a² + b² ≤ z² (Frangioni–Gentile). The row budget is an equality on Σ
   rows·z, with Σ z ≤ 1 per site. Prunes if t ≤ incumbent·(1 + 1e-4). If
   the SDP raises an error, the node is not pruned (t = ∞, all z = 0.5).
7. **Branch** ([branch_and_bound.py:371](branch_and_bound.py#L371)). Picks
   the undecided candidate whose z is closest to 0.5. The "on" child
   forces it on and its same-site siblings off; the "off" child forces it
   off. The child the relaxation leans toward (on if z ≥ 0.5) is pushed
   last so it is explored first.

**Leaf processing** ([branch_and_bound.py:319-343](branch_and_bound.py#L319-L343)):

1. Symmetry check on the complete design.
2. **Leaf bound** — `SetEvaluator.bound` → `bound_node`
   ([sdp_topology_bound.py:524](sdp_topology_bound.py#L524)). Searches up
   to 300 reference θ vectors (`find_full_rank_reference`: base angles,
   then ±30° jitter, then uniform random), then solves the same SDP with
   nothing undecided (pure disk relaxation). A leaf with no full-rank
   reference, or two candidates at one site, or the wrong row count, is
   counted as infeasible.
3. Skip the leaf if its bound ≤ incumbent·(1 + tol).
4. **Leaf evaluation** — `SetEvaluator.evaluate` → `bound_and_evaluate`
   (Section 5). If the achieved λ_min beats the incumbent, it becomes the
   new incumbent. The leaf's bound and achieved value are recorded either
   way.

**Result and certificate**
([branch_and_bound.py:379-382](branch_and_bound.py#L379-L382)). Every
pruned node or leaf has bound ≤ incumbent·(1 + tol), and every evaluated
leaf's true optimum ≤ its own bound, so

```
certified upper bound = max( incumbent·(1 + tol), largest bound among evaluated leaves )
certified gap         = upper bound − incumbent
```

If the node limit stops the search early, the upper bound is reported as
∞. The certificate is relative to leaf evaluation: Riemannian is a local
method, and the gap is what covers that.

---

## 5. Leaf evaluation (one fixed design)

`bound_and_evaluate`
([bound_and_evaluate.py:271](bound_and_evaluate.py#L271)), called with the
reference θ found by the leaf bound and a per-set RNG.

- **Angle-free shortcut** — `_evaluate_angle_free`
  ([bound_and_evaluate.py:253](bound_and_evaluate.py#L253)). If the
  design has no V-grooves (only flats and sockets), K is fixed: its
  locked λ_min is computed directly and is both the achieved value and an
  exact bound (gap 0).
- **Certified bound** — `sdp_certified_bound`
  ([bound_and_evaluate.py:63](bound_and_evaluate.py#L63)). Joint
  calibration of the design, projection onto the locked subspace
  (`_calibrate_reduced`), then the stage-1 SDP: maximize t with each
  coupling's (cos 2θ, sin 2θ) relaxed to the disk
  (`_solve_stage1_bound`,
  [sdp_theta_optimizer.py:276](sdp_theta_optimizer.py#L276)). The
  stage-2 decode is never run. This is the same relaxation as the leaf
  bound above, computed again through a different code path.
- **Achieved value** — `riemannian_multistart`
  ([bound_and_evaluate.py:106](bound_and_evaluate.py#L106)). Runs
  `optimize_theta_riemannian` from the reference θ plus 4 uniform-random
  starts and keeps the best λ_min. A start that stays stuck on an
  eigenvalue multiplicity is replaced by a fresh random start; if 5
  successful runs can't be collected, it raises `MultiplicityError`, which
  `SetEvaluator` records as an infeasible leaf (λ_min 0).
- **Riemannian gradient sampling** — `optimize_theta_riemannian`
  ([riemannian_gradient_sampling.py:278](riemannian_gradient_sampling.py#L278)).
  Burke–Lewis–Overton gradient sampling on the torus of θ (period π), per
  Hosseini & Uschmajew. Each iteration:
  1. Evaluate λ_min and its exact gradient vᵀ(∂K/∂θᵢ)v at the current
     point and n random nearby points (`eval_lambda_min_and_grad`,
     [riemannian_gradient_sampling.py:151](riemannian_gradient_sampling.py#L151));
     resample any point at an eigenvalue multiplicity.
  2. Find the minimum-norm element of the convex hull of those gradients,
     a small QP (`_min_norm_hull_element`,
     [riemannian_gradient_sampling.py:240](riemannian_gradient_sampling.py#L240)).
  3. If its norm is below tolerance, shrink the sampling radius (or stop
     at the floor); otherwise step along it with a backtracking line
     search on λ_min, wrapping with θ ← (θ + step) mod π.

  Every active candidate gets a θ variable, including flats and sockets in
  a mixed design; theirs simply have zero gradient.

- **Stage-2 polish (optional, off in the B&B)** — `polish_tied_restarts`
  ([bound_and_evaluate.py:208](bound_and_evaluate.py#L208)) →
  `polish_log_product`
  ([log_product_polish.py:180](log_product_polish.py#L180)). For every
  restart tied with the best λ_min, maximizes log det(PᵀKP) (the sum of
  log locked eigenvalues) while holding λ_min above a threshold, using
  trust-region SDP steps on a first-order model of the eigenvalues near
  that threshold. Enabled with `bound_and_evaluate(..., polish=True)`.

---

## 6. Standalone leaf-bound experiments

Neither of these is called by `BranchAndBound`; both target the loose
leaf bounds behind the birds-foot certified gap (summary §7.11–§7.12).

- **Arc branching** — `LeafRefiner` ([arc_refine.py:91](arc_refine.py#L91)).
  Restricts each coupling's (cos 2θ, sin 2θ) to an arc (one linear cut
  added through `sdp_topology_bound(..., arcs=...)`), tightens arcs per
  coupling, then runs best-first branching that bisects arcs. Sound, but
  doesn't tighten the bound here because the relaxation has no product
  terms.
- **Moment relaxations** — `MomentLeafRelaxation`,
  `LinearMultiplierRelaxation`
  ([moment_leaf_bound.py:54](moment_leaf_bound.py#L54)). Henrion–Lasserre
  moment relaxation of "is there θ with K(θ) − τI ⪰ 0?", bisecting on τ.
  Order 1 equals the disk bound; order 2 adds products xᵢxⱼ but is too
  large for this machine's solvers.

---

## 7. Where the code differs from, or adds to, the summary document

- **Mixed types are now in the B&B, bounds and evaluator** (`rows` /
  `site_of` / `sites` arguments throughout, `coupling_types.py`,
  `mixed_types_validation.py`). The summary still describes the B&B as
  V-grooves only (§7.10, Phase 2 status) and its code map (§6.2) doesn't
  list the two new files. The two-panel mixed validation has run and
  passed (`results/mixed_types_validation_two_panel_summary.txt`: optimum
  0.017142, a V + flat + socket design, 128.6% of the best all-V design).
- **The leaf bound is computed twice** per evaluated leaf: once by
  `bound_node` (registry assembly) to decide whether to evaluate, and
  again by `sdp_certified_bound` (joint calibration) inside
  `bound_and_evaluate`. They are the same relaxation.
- **"Unresolved" leaves count as infeasible.** `bound_node` reports
  "no full-rank reference after 300 tries" as unresolved and says not to
  prune on it; the B&B prunes such leaves anyway. That relies on summary
  §2.7 (rank is generic in θ), which is well supported but is the step
  that makes this pruning valid.
- **Leaves skip the cut and rank tests**: a forced completion goes
  straight to the leaf bound, whose own reference search does the rank
  check.
- **Angle-free candidates still get angle variables in Riemannian** for
  a mixed design containing at least one V-groove (zero gradient, so
  harmless). Summary §8.3 says the angle stage "runs only over V-grooves",
  which is true in effect but not in the variable list.
- **Greedy is V-groove only**, and swap search in mixed mode only finds
  like-for-like swaps, so neither yet produces a mixed-type incumbent.
