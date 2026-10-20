# Kinematic Coupling Placement — Findings & Roadmap

Branch: `Kinematic-Coupling-Placement`. Last updated: 2026-10-20.

This document tracks the coupling *placement* and *orientation*
optimization work (`RigidBodyModel.py`, `coupling_optimizer.py`, and
everything built on them). It is unrelated to `README.md`, which
describes an earlier sensitivity-analysis framework on another branch
(`SensitivityModel`, `HingeElement`, etc.).

**Goal**: choose which of *m* candidate kinematic-coupling sites are
active, and the groove angle θ of each active coupling, for the
*minimal* set of couplings joining the panels of an origami array, to
maximize λ_min(K), K = CᵀC — using established optimization methods,
validated soundly, and reported honestly.

---

## Status at a Glance

- **Angle optimization for a fixed set of couplings: solved.**
  Riemannian gradient sampling (best of 5 restarts) gives the achieved
  λ_min; the SDP relaxation gives a certified upper bound. Validated on
  all 220 birds-foot size-9 subsets with zero soundness violations
  (§7.1).
- **Choosing *which* couplings: partly solved.** Forward bound-guided
  greedy found the exact birds-foot champion, but fails when candidate
  sites are near-duplicates (55–73% of optimum, §7.6–§7.7). One-swap local
  search fixes every two-panel case but only 48% of birds-foot starts:
  birds-foot has real one-swap traps (96.8% and 90.6% of the champion,
  §7.8–§7.9). Neither method alone is reliable; branch-and-bound is the
  candidate for a guarantee.
- **Choosing among equally good angles: tool built, policy open.**
  Stage-2 log_product polish works, but on birds-foot it only helps if a
  small λ_min sacrifice is allowed (§7.5).
- **Physical answer for the assembled flasher: not yet available.** Gore
  tests are algorithm stress tests with one arbitrary boundary
  condition, and the full-gore result is over-constrained (46 rows for
  42 DOFs), so it isn't a valid design under the exact-constraint
  requirement (§8.2).

**Roadmap** (§8): swap search after greedy (built; needs a two-swap
escape or B&B behind it) → generalize the coupling
model (flat / V-groove / tetrahedral, any face orientation) →
branch-and-bound selection under an exact row budget → couplings on
arms.

---

## 1. Problem and Test Systems

**Model.** Each panel is rigid (6 DOFs). Each `KinematicCoupling` is a
sphere in a 90° V-groove on the shared mating face; it contributes 2
constraint rows (one per groove wall, normals n1, n2) to C. The groove's
sliding axis lies in the mating face, rotated by θ. K = CᵀC.

**Objective.** λ_min is the smallest *locked* eigenvalue of K: the
(d − rank C)-th eigenvalue, skipping the 6 whole-assembly rigid-body
modes. Any set below the target rank (`total_dofs − 6`) scores 0, so
under-constrained designs can never beat fully constrained ones. Each
coupling gives 2 rows, so at least k = target_rank / 2 couplings are
needed.

**Exact constraint.** Designs must be exactly constrained: total
rows = target_rank with C at full rank, so no row is redundant
(decided 2026-10-13, §8.1). Over-constrained designs are out of scope.

**Standard candidate sites.** 3 per crease: 20% and 80% along the crease
on the top face, and the midpoint on the **bottom face** (z = top − t).
(Earlier docstrings called the third one "mid-thickness"; that was
wrong and has been fixed, see §7.6.)

| System | Panels | Candidates m | Minimum k | Source | Ground truth |
|---|---|---|---|---|---|
| Two-panel | 2 | 3 | 3 | `interactive_optimizer.build_system` | trivial |
| Two-panel, 6 sites | 2 | 6 | 3 | `two_panel_six_sites.py` | exhaustive (42 subsets) |
| Birds-foot | 4 (star around one hub) | 12 | 9 | `interactive_optimizer.build_birdsfoot_system` | exhaustive (220 size-9 subsets) |
| Birds-foot, 6 sites/spoke | 4 | 24 | 9 | `birdsfoot_six_sites.py` | none (≈1.3M subsets) |
| Single flasher gore | 6 (two-hub tree) | 18 | 15 | `flasher_gore_system.py` (DXF) | none |
| Full gore + neighbor panels | 8 | 27 | 21 (greedy's set needed 23: over-constrained, §8.2) | `full_flasher_gore_system.py` (DXF) | none |

Best known values: two-panel 0.013330 (`[0,1,2]` and its mirror image
`[3,4,5]` in the 6-site system); birds-foot 0.009263, reached by **two**
designs: `[0,1,2,3,5,6,7,8,11]` (O-M:3, O-B:2, O-N:3, O-A:1) and
`[0,1,2,5,6,7,8,9,11]` (O-M:3, O-B:1, O-N:3, O-A:2), equal to 6e-7. The
second is in the exhaustive sweep but was never reported (§7.9). It looks
like a mirror image of the first; the geometry hasn't been checked.

---

## 2. What We Know About the Math

### 2.1 K(θ) is exactly affine in (cos 2θᵢ, sin 2θᵢ), with no cross terms

For a fixed active set, exactly (validated to ~1e-15):

```
K(theta) = K0 + sum_i [ cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i ]
```

Each coupling's contribution is separately affine and additive:
calibrating one coupling in isolation gives the same Kc_i, Ks_i as a
joint calibration. This follows from the physics (K is a Gram matrix
over row blocks that belong to one coupling each), not a curve fit.
Everything below relies on it: exact gradients, the per-coupling SDP
registry, and the reduced model used by the polish.

### 2.2 θ lives on a circle of period π

u(θ + π) = −u(θ), the groove walls swap, and K is exactly unchanged.
Treating θ as a box-bounded interval is what caused DE/Nelder-Mead to
pin solutions near 0°/180° (documented in `optimize_theta`'s
docstring): a modeling error, not a tuning problem. The fixes are the
periodic embedding in `coupling_optimizer.py` and the mod-π retraction
in the Riemannian code.

### 2.3 The rigid-body null space is the same for every full-rank design

The 6-dim gauge null space of K comes from whole-assembly rigid motion,
so it is the same fixed subspace for any coupling subset and any θ that
reaches full rank. One projection basis P is therefore valid across a
whole branch-and-bound tree, and PᵀK(θ)P has exactly the locked
eigenvalues as its spectrum everywhere.

### 2.4 λ_min is nonsmooth exactly where optima sit

Optimal designs generically sit at eigenvalue multiplicities (Cox &
Overton proved the analogous strongest-column problem has a double least
eigenvalue at its optimum). This is why DE/Nelder-Mead struggle and why
gradient sampling is the right tool.

### 2.5 The set of optimal angles can be a plateau or a ridge

- **Two-panel: a wide plateau.** About two-thirds of random θ triples sit
  at the same maximal λ_min with numerically zero gradient. "The optimal
  design" is a large set; report the optimal *value*.
- **Birds-foot champion: a double eigenvalue on a ~1-D ridge** (§7.5).
  The two smallest locked eigenvalues coincide (relative gap 4.9e-6), and
  the near-optimal set is essentially one coupling's angle sliding
  (O-M bottom-face midpoint, 19° of spread vs. ≤3° for the others).
  λ_min and log_product trade off along it.

Which of these the gore-scale systems show is untested.

---

## 3. What We Know About Good Designs

- **Leaf panels force couplings.** A panel attached by only one crease
  needs all 3 of that crease's couplings (exactly 6 rows for 6 relative
  DOFs). Detected automatically by
  `gore_system_common.mandatory_leaf_couplings()` (§7.4).
- **The structural minimum k isn't always enough.** On the full gore,
  k = 21 and 22 could not reach full rank at any θ; 23 was needed
  (§7.4). The leaf rule is a special case of a cut condition (couplings
  crossing any cut must supply ≥ 6 independent rows), which may explain
  this but hasn't been checked.
- **Three collinear contacts can't constrain rotation about their line.**
  Such subsets are rank-deficient at every θ (§7.6).
- **Spread along the crease beats depth through the panel.** Mirroring
  sites through the thickness gave nothing at minimum k on two-panel, and
  greedy's worst choices on birds-foot were "stacked pairs" (top and
  bottom contacts at the same point along the crease) (§7.6–§7.7).
- **More couplings help a lot — but that's out of scope** under the
  exact-constraint requirement (§8.2). Two-panel, best λ_min by k: 0.0133
  (k=3), 0.0200 (k=4), 0.0240 (k=5), 0.0300 (k=6) (§7.6).
- **Good designs can be several swaps apart.** On birds-foot, the most
  common one-swap trap (96.8%) is exactly 2 swaps from the champion, with
  every intermediate step worse. The design space is genuinely
  multi-modal at the topology level, not just in θ (§7.9).
- **Pure λ_min maximization can pick a fragile design.** On the
  birds-foot ridge, the strict λ_min maximizer had the 2nd-worst
  log_product of 20 restarts; giving up 0.1% of λ_min buys +14% in the
  product of locked eigenvalues (§7.5).

---

## 4. What We Know About the Methods

| Method | Status | Key finding |
|---|---|---|
| DE / Nelder-Mead (`coupling_optimizer.py`) | Superseded for evaluation. | Boundary pinning (§2.2) and nonsmoothness (§2.4); failed to converge on 56 of 206 birds-foot subsets. |
| Backward greedy pruning (`CouplingOptimizer.prune_couplings`) | Superseded. | Reached 48.8% of the birds-foot champion. |
| Per-topology SDP relaxation (`sdp_theta_optimizer.py`) | Bound sound; decode unreliable. | Zero violations on 220 subsets, but the relaxed optimum often sits inside the unit disk (no real θ behind it); two different tie-break objectives gave the same poor decode. Now used for the bound only. |
| Topology-level SDP bound (`sdp_topology_bound.py`) | Sound; loose for partial sets. | Zero violations (root + 98 random nodes). Stays loose through ~10 of 12 decisions, tight only near leaves; nearly exact at complete subsets. Looseness flattens greedy's scores (§7.6–§7.7). |
| Riemannian gradient sampling (`riemannian_gradient_sampling.py`) | Validated; the achieved-value method. | Hosseini & Uschmajew's manifold version of Burke–Lewis–Overton. Crosses the 0/π seam cleanly; recovered all 56 subsets DE failed on; statistically tied with DE elsewhere. |
| Split-responsibility pipeline (`bound_and_evaluate.py`) | Validated at full scale. | SDP for the certified bound, Riemannian for the achieved value (§6, §7.1). |
| Forward bound-guided greedy (`forward_greedy_topology.py`) | Works on birds-foot; unreliable in general. | Exact birds-foot champion in seconds (§7.3). Misses with near-duplicate sites: 72.8% (two-panel) and 55% (birds-foot 24 sites), because tied bound scores leave early choices to tie order. Can't see mandatory couplings without forcing (§7.4). |
| One-swap local search (`swap_search.py`) | Validated on two scales; reliable only on two-panel. | Two-panel: 20/20 starts reach the optimum (1 swap each). Birds-foot: 105/220 (48%); the rest stop at one-swap local optima (96.8%, 90.6%). Bound screening prunes most swaps; ~5 min per birds-foot run (§7.8–§7.9). |
| Stage-2 log_product polish (`log_product_polish.py`) | Validated (12/12 checks vs. real system); opt-in. | Big gain on the two-panel plateau; no-op on birds-foot at strict tolerance, +4–14% with a 0.01–0.1% λ_min tolerance (§7.5). |

---

## 5. Literature Grounding

The problem splits into two established literatures (full citations on
request):

- **Continuous stage (θ on the torus)**: nonsmooth eigenvalue
  optimization (Overton; Cox & Overton; Lewis on partly-smooth
  functions) and Riemannian gradient sampling (Hosseini & Uschmajew,
  SIAM J. Optim. 2017 — the manifold generalization of
  Burke–Lewis–Overton). Established software: HANSO/GRANSO/PyGRANSO
  (optional independent cross-check).
- **Discrete stage (which couplings)**: cardinality-constrained SDP
  relaxation (Frangioni & Gentile's perspective reformulation — the
  `a²+b² ≤ z²` construction in `sdp_topology_bound.py`); sensor
  selection (Joshi & Boyd 2009 — same "choose k of m to optimize a
  function of a sum of per-candidate matrices" structure, with an
  established rounding + local-search fix); robust truss topology design
  (Ben-Tal & Nemirovski); branch-and-bound for optimal experimental
  design (Ahipaşaoğlu — bound-tightening branching order); greedy
  approximation for submodular maximization (Nemhauser, Wolsey & Fisher
  1978 — its (1−1/e) guarantee does not strictly apply here, since greedy
  scores by a bound rather than the objective).
- **Stage-2 polish**: SQP / trust-region treatment of a semidefinite
  (eigenvalue) constraint, with Overton's first-order model of a multiple
  eigenvalue.
- **Periodic structures** (for §8.5): cyclic-symmetry / Bloch-wave
  reduction (Thomas 1979 and successors).

Each piece of the pipeline has a named, citable precedent.

---

## 6. Pipeline and Code Map

### 6.1 How a design is evaluated

For a fixed active set:

1. **Certify** — `sdp_certified_bound()`: SDP stage 1 only, giving a
   sound upper bound t. The stage-2 decode is skipped, since that is the
   unreliable part.
2. **Evaluate** — `riemannian_multistart()`: best of N Riemannian
   gradient-sampling runs (one from the reference θ, the rest random).
   Multiple restarts are needed because gradient sampling only
   guarantees *local* Clarke stationarity. A start stuck at an eigenvalue
   multiplicity is replaced with a fresh random one.
3. **Polish (optional)** — `polish_tied_restarts()`: stage-2 log_product
   polish of every restart within tolerance of the best λ_min.
4. **Report** — `bound_and_evaluate()` returns t, the achieved λ_min, the
   gap, and (with `polish=True`) the polished design.

To choose the active set: `forward_greedy()` scores each candidate
addition by the certified bound of its best-case completion, with
optional `forced_active_start` for mandatory couplings and k-bumps when
the final set can't reach full rank.

### 6.2 Code map

| File | Role |
|---|---|
| `RigidBodyModel.py` | Panels, couplings, constraint matrix C |
| `coupling_optimizer.py` | Original DE/Nelder-Mead two-stage optimizer and backward pruning |
| `interactive_optimizer.py` | Two-panel and birds-foot geometry; GUI |
| `test_K_theta_structure.py` | `calibrate()` for the affine K(θ) decomposition (§2.1) |
| `sdp_theta_optimizer.py` | Per-topology SDP relaxation |
| `sdp_topology_bound.py` | Topology-level bound, per-coupling registry, `bound_node`, `find_full_rank_reference` |
| `riemannian_gradient_sampling.py` | Riemannian gradient sampling |
| `bound_and_evaluate.py` | Split-responsibility pipeline, multistart, polish hook |
| `log_product_polish.py` | Stage-2 polish |
| `forward_greedy_topology.py` | Forward greedy; birds-foot run |
| `swap_search.py` | One-swap local search (`swap_search`) and the cached, per-set-seeded `SetEvaluator` |
| `dxf_to_panels.py`, `gore_system_common.py`, `flasher_gore_system.py`, `full_flasher_gore_system.py` | DXF geometry loading and gore systems |
| `forward_greedy_gore.py`, `forward_greedy_full_gore.py` | Gore greedy runs |
| `polish_birdsfoot_champion.py`, `two_panel_six_sites.py`, `birdsfoot_six_sites.py` | 2026-10-13 experiments |
| `two_panel_swap_validation.py`, `birdsfoot_swap_validation.py` | 2026-10-20 swap-search validations |
| `test_*.py` | Validation suites (one per major module) |

### 6.3 Engineering notes

- **Memory**: this machine has 7.6 GB RAM. The `cpu_count() − 1` worker
  default (7 workers, each loading numpy/scipy/cvxpy) caused a real
  `MemoryError`. `bound_and_evaluate_sweep.py` defaults to 3 workers
  (`N_WORKERS` env var to override).
- **Background jobs**: an empty *buffered* log isn't evidence a job died;
  check process state or run with `python -u`.
- **Shared RNG**: `riemannian_multistart` shares one RNG across restarts,
  so changing `n_restarts` changes every restart's path, not just adds
  new ones. Fix with an independent child RNG per restart if this
  matters.
- **Dependencies**: `ezdxf` and `shapely` were added for DXF loading.
  There is no `requirements.txt` yet.
- **Bugs fixed during the full sweep**: (1) the first evaluation of
  `theta_init` in `optimize_theta_riemannian` lacked the multiplicity
  retry every other evaluation had; (2) `bound_and_evaluate_sweep.py`
  let one failing subset crash the whole process pool — now failures are
  recorded per subset, matching `exhaustive_subset_baseline.py`.

---

## 7. Experiment Log

### 7.1 Full validation of the pipeline on birds-foot (2026-09-10)

206 of 220 birds-foot size-9 subsets (14 skipped: no full-rank reference
found, consistent with known structural infeasibility).

- **Soundness**: passed for all 206 — zero violations.
- **Certification gap** (t − achieved): median 8.5e-6, max 6.7e-3 (a
  known relaxation-looseness case, not a decode failure).
- **DE failures recovered**: on the 56 subsets where DE failed to
  converge, Riemannian multistart succeeded on all 56 (λ_min 0.0012 to
  0.0093).
- vs. DE where DE converged (150): 20 better, 19 worse, 111 ties.
- vs. the old SDP decode: Riemannian better on 143/206 (69%).
- Best subset matches the champion (0.009263 vs. 0.009259 previously),
  with a gap of 8e-6 at that topology.
- 29/206 subsets (14%) had restart spread > 1e-4 — see §7.2.

Results: `results/bound_and_evaluate_sweep_{results.json,results.csv,summary.txt}`.

### 7.2 Are 5 restarts enough? (2026-09-10)

Reran the 29 high-spread subsets at 7 restarts
(`bound_and_evaluate_restart_check.py`). 27/29 improved by < 1e-6; the
other 2 by at most 1.1e-5. Spread stayed wide (23/29 still > 1e-4), but
that reflects genuinely different local optima, not undersampling.
**Conclusion: 5 restarts is enough.** Zero soundness violations.

Results: `results/bound_and_evaluate_restart_check_{results.json,results.csv,summary.txt}`.

### 7.3 Forward greedy on birds-foot (2026-09-11)

Tried forward bound-guided greedy as an alternative to brute force and
to a full branch-and-bound. Because the true objective is zero for any
set smaller than k = 9, each step scores candidates by the certified
bound of the best-case completion (an admissible lookahead heuristic,
not a literal use of the (1−1/e) theorem).

- **Found the exact champion** `[0,1,2,3,5,6,7,8,11]` — 100% of the
  exhaustive optimum, in seconds (19 s in the saved run; 72 bound
  evaluations) vs. hours of exhaustive search.
- Backward pruning reached only 48.8% (0.004521).
- First real use of the shared per-coupling `registry`; verified to give
  bounds identical to the non-shared path.
- Caveat at the time: one data point. §7.6–§7.7 later showed it doesn't
  generalize reliably.

Results: `results/forward_greedy_topology_{results.json,summary.txt}`,
`results/forward_greedy_topology_3d_before_after.png`.

### 7.4 Forward greedy on flasher gores (2026-09-14)

**Geometry from DXF.** `dxf_to_panels.load_gore_geometry()` treats the
DXF's line segments as a planar graph and recovers panels as its faces
(`shapely.ops.polygonize`); an edge shared by two faces is a crease,
one face is boundary. `Flasher_gore_reference_cleaned.dxf`: 17 lines →
6 panels, 6 creases, every line accounted for. Gore systems use the
standard 3 sites per crease (shared code in `gore_system_common.py`).

**Single gore (m=18): leaf panels.** Panels 2 and 5 attach by one crease
each, so those creases' couplings are mandatory (§3). Unmodified greedy
at k = 15 and 16 left the P1-P2 crease one coupling short, giving sets
that can't reach full rank at any θ. Fix: `forced_active_start`
(backward-compatible) seeded with the automatically detected leaf
couplings.

- Result: `[0,1,2,4,5,6,7,8,9,10,11,12,13,14,17]` (P0-P1:3, P0-P4:2,
  P0-P5:3, P1-P2:3, P1-P3:3, P3-P4:1). t = 0.012966, achieved λ_min =
  **0.012351**, gap 4.98% (much looser than birds-foot). Greedy 46.7 s.
- Results: `results/forward_greedy_gore_{results.json,summary.txt}`.

**Missing inter-gore couplings.** The single-gore DXF omitted the
couplings to neighboring gores, so the result above is for an isolated
wedge with free radial edges. `Full_Flasher_gore_reference.dxf` adds the
neighbor's boundary panels. Two loader generalizations were needed:
block `INSERT` support (via `ezdxf`'s `virtual_entities()`), and
opt-in vertex snapping (`snap_tol`; endpoints were off by up to 0.45
units vs. ~80 units between genuinely distinct vertices).

**Full gore (m=27).** 8 panels, 9 creases; only panel 5 is still a leaf
(3 forced couplings).

- k = 21 and 22 couldn't reach full rank; each retry extended the same
  set by one greedy step. Settled at **k = 23 of 27**. Note: 23
  V-grooves = 46 rows for target_rank 42, so this design is
  **over-constrained** and invalid under the exact-constraint
  requirement adopted later (§8.2).
- t = 0.005242, achieved λ_min = **0.005185**, gap 1.10%. Spokes P0-P1:3,
  P0-P4:1, P0-P5:3, P1-P6:3, P1-P2:2, P1-P3:2, P2-P7:3, P3-P4:3, P6-P7:3.
- Greedy 1068.6 s (vs. 46.7 s at m=18), partly from the two extra
  k-bump rounds.
- Results: `results/forward_greedy_full_gore_{results.json,summary.txt}`;
  figures `..._3d_before_after.png`, `..._hinges.png`,
  `..._convergence.png`.

**Limitation**: one neighbor gore's panels are one arbitrary boundary
condition on a rotationally periodic problem. Adding more neighbor
detail doesn't converge to the assembled λ_min. These are algorithm
stress tests, not physical answers (§8.5).

### 7.5 Stage-2 log_product polish (2026-10-13)

**Why.** The DE pipeline had a stage 2 (maximize log_product =
Σ log(locked eigenvalues) subject to λ_min ≥ λ* − tol) to choose among
λ_min-tied designs. The Riemannian pipeline kept whichever restart had
the best λ_min, so every reported θ was an arbitrary point of the optimal
set. Topology choice is unaffected: log_product is −∞ for partial sets.

**Method** (`log_product_polish.py`):
- Smooth objective via the fixed gauge subspace (§2.3): log_product =
  log det(PᵀK(θ)P), with exact gradient tr(Kr⁻¹ ∂Kr/∂θᵢ) from the
  existing calibration.
- Each step solves a small trust-region SDP: maximize the first-order
  log_product gain, keeping a first-order model of the eigenvalues near
  the threshold above it (Overton's model of a multiple eigenvalue;
  cluster chosen by a Weyl bound), ‖d‖ ≤ Δ. A step is accepted only if
  the true λ_min and log_product check out; retraction is θ + d mod π.
- Opt-in via `bound_and_evaluate(..., polish=True)`; existing outputs
  unchanged.
- Validation (`test_log_product_polish.py`, 12/12, checked against the
  real system): reduced spectrum matches to 2e-15; gradient matches
  finite differences to 5e-7; every step feasible on the real system.

**Two-panel**: the plateau is real and stage 2 uses it — log_product
−4.88 → −1.99 (product ×18) at unchanged λ_min, θ moving up to 84°.

**Birds-foot champion** (`polish_birdsfoot_champion.py`, λ* = 0.00926316):
- **Double eigenvalue, not a plateau**: smallest two locked eigenvalues
  0.00926316 / 0.00926321; the third is 23% higher. At strict tolerance
  (relative 1e-7, as in `coupling_optimizer.py`) the polish moves θ by
  0.000°: **strict stage 2 is a no-op here.**
- **Steep trade-off, ~100:1**: from the best restart, giving up 0.01% /
  0.1% / 1% of λ_min multiplies the product of locked eigenvalues by
  1.011 / 1.106 / 2.38, and separates the second-weakest mode (λ₂
  0.009263 → 0.010524 at 1%).
- **Restarts don't merge.** With a looser tie tolerance (1e-4: 4/5 tied;
  1e-3: 5/5), each polish moves ≤ 0.1° / ≤ 0.9° and ends exactly on the
  λ_min constraint; restarts stay 2.5–20° apart. The gain is λ_min budget
  being spent, not travel along the ridge.
- **20 restarts** (`N_RESTARTS=20`): the ridge is ~1-D (§2.5); the strict
  λ_min maximizer has the 2nd-worst log_product of 20. Best polished
  design vs. strict design: ×1.039 at 1e-4 (5 restarts: ×1.022), ×1.140
  at 1e-3 (×1.124). Cost ~4.5 min vs. ~40 s.
- **Caveat**: every birds-foot polish stopped on the trust-region floor,
  not a certified KKT point (the linear model of the coalesced pair keeps
  over-predicting gain; some runs took 400–500 iterations). Final points
  are feasible and on the constraint.

**Interpretation**: on birds-foot, the useful stage 2 is choosing among
ridge points, not polishing one. Since the ridge is ~1-D and its
direction is known, a 1-D continuation along it would map the whole
trade-off more efficiently than random restarts (§8.5).

Results: `results/polish_birdsfoot_champion_{results.json,summary.txt}`
and `..._r20_{results.json,summary.txt}`.

### 7.6 Contact-location choice: two panels, 6 sites (2026-10-13)

**Question**: do more candidate *locations* per crease improve λ_min?
Tested on two-panel, where exhaustive ground truth is cheap.

**Setup**: the 3 standard sites plus each mirrored through the
mid-thickness plane z = −t/2 (panels span z ∈ [−t, 0], t = 0.1):

| site | point | | site | mirror |
|---|---|---|---|---|
| 0 | p1 (y=.2, top) | | 3 | p1′ (y=.2, bottom) |
| 1 | p2 (y=.8, top) | | 4 | p2′ (y=.8, bottom) |
| 2 | p3 (y=.5, bottom) | | 5 | p3′ (y=.5, top) |

Setting this up revealed the "mid-thickness" mislabel (§1): p3 is on the
bottom face, so its mirror is a new top-face point. Docstrings in
`interactive_optimizer.py`, `gore_system_common.py` and
`sdp_theta_optimizer.py` were fixed; no numbers changed.

**Results** (exhaustive, all 42 subsets for k = 3–6, ~28 s):
- **k=3: the original design is still optimal.** {p1,p2,p3} and its
  mirror {p1′,p2′,p3′} tie at 0.013330; every mixed top/bottom set is
  worse (next 0.00992, worst feasible 0.00324).
- The two collinear triples ({p1,p2,p3′}, {p3,p1′,p2′}) are rank-deficient,
  as predicted.
- Best λ_min by k: 0.0133 / 0.0200 / 0.0240 / 0.0300 for k = 3 / 4 / 5 / 6
  (k=4 best: {p1,p3,p1′,p3′}).
- Soundness: max (achieved − bound) = 1.1e-9; the bound is essentially
  exact at complete subsets.
- **Forward greedy missed: 72.8%** ({p3,p3′,p2}, 0.009707). At steps 1
  and 2 the bound scored candidates at exactly 0.0150 (6/6, then 3/5
  tied), so it committed by tie order to p3 and p3′, which stack at the
  same y.

Results: `results/two_panel_six_sites_{results.json,summary.txt,3d_before_after.png}`.
The figure shows all 6 sites at start angles (λ_min 0.0300) vs. the
chosen design (= the original optimum) with unused sites marked.

### 7.7 Contact-location choice: birds-foot, 6 sites per spoke (2026-10-13)

**Question**: with more options, does greedy match or beat its 3-site
result?

**Setup**: §7.6's construction on every spoke; m = 24, k = 9. Index
6s + j (j = 0–2 original, 3–5 mirror of j − 3). Mirrors start at the
original's angle + 47° (mod 180). The old champion maps to
`[0,1,2,6,8,12,13,14,20]`, so it is one of the candidate subsets and the
24-site optimum is ≥ 0.009263. Both designs evaluated identically
(certified bound + 5 restarts + strict polish, seed 17).

| design | λ_min | own bound t | gap | log_product |
|---|---|---|---|---|
| old 3-site champion (re-evaluated in the 24-site system) | 0.009263 | 0.009271 | 0.08% | −19.98 |
| forward greedy over 24 sites | **0.005106** | 0.008397 | 64% | −22.19 |

- **Greedy got worse: 55% of the old champion.** This is certified: the
  greedy set's own upper bound (0.008397) is below what the old champion
  achieves, so no choice of angles could rescue that set.
- The old champion reproduces exactly, so greedy simply didn't find it.
- **Root bound for any 9-of-24 design: 0.014676**, so the 24-site optimum
  lies between 0.009263 and 0.014676. Whether mirrored sites can beat the
  old champion is still unknown.
- Greedy took 19 s, no k-bump; 3 of 9 chosen sites are mirrors.
- One SDP solve warned "solution may be inaccurate"; the ~10% margin
  makes it very unlikely to matter.

**Why it failed** (same mechanism as §7.6):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| tied | 12/24 | 7/23 | 4/22 | 1/21 | 4/20 | 1/19 | 2/18 | 1/17 | 1/16 |
| bound | 0.014676 | 0.014676 | 0.014676 | 0.014676 | 0.014476 | 0.014196 | 0.013704 | 0.012887 | 0.008397 |

The bound sat at the root value for 4 steps; greedy committed by tie
order to stacked pairs at the 50% position on O-M (sites 2, 5) and O-N
(14, 17). The cost only showed up at the last step (0.0129 → 0.0084).

**Lesson**: forward greedy is only as good as its bound is informative.
More options made the *search* worse even though they can't make the
*optimum* worse.

Results: `results/birdsfoot_six_sites_{results.json,summary.txt}`.
Figures: `..._3d_before_after.png` (all 24 / old champion / greedy, weakest
mode overlaid), `..._hinges.png` (one hinge layout per spoke, old vs.
new), `..._convergence.png` (bound per step vs. achieved, old champion,
root bound).

### 7.8 One-swap local search: two panels, 6 sites (2026-10-20)

**Method** (`swap_search.py`, Phase 0 of §8.4). From a starting set of k
couplings, each pass considers every swap (one active out, one inactive
in; set size and row count unchanged):

1. Screen each swapped set with its certified bound (complete set, so
   nothing undecided). Drop sets that can't reach full rank, and sets
   whose bound can't beat the current achieved λ_min — a rigorous rule.
2. Evaluate survivors (certified bound + 5-restart Riemannian) in
   descending bound order, stopping once no remaining bound can beat the
   best found. This finds the true best swap while evaluating few sets.
3. Accept it if λ_min improves by > 1e-4 relative (above the restart
   noise measured in §7.2); otherwise stop at a one-swap local optimum.

Each set's evaluation is seeded from the set itself and cached, so a set
always gets the same value whichever path reaches it.

**Validation** (`two_panel_swap_validation.py`): swap search from all 20
three-site starts, against the exhaustive ground truth of §7.6.

- **20/20 starts reached the optimum** (0.013330), each ending at the
  nearer of the two mirror-image optima. That includes greedy's
  {p2, p3, p3′} (fixed by swapping p3′ → p1) and both rank-deficient
  starts.
- Every non-optimal start needed exactly 1 swap: on two-panel every set
  is one swap from an optimum, so this is an easy case.
- Bound screening did most of the work: 342 swaps considered, 36
  infeasible, 288 pruned by bound, 18 evaluated in full (one per feasible
  set). 21 s total.
- Cross-check against §7.6's saved values: max difference 5.6e-13, 0
  feasibility disagreements.

Results: `results/two_panel_swap_validation_{results.json,summary.txt,starts.png}`
(figure: start vs. final λ_min per starting set).

### 7.9 One-swap local search: birds-foot, 12 sites (2026-10-20)

**Validation** (`birdsfoot_swap_validation.py`): swap search from all 220
size-9 starts, against the exhaustive sweep of §7.1.

**Result: 105/220 starts (47.7%) reached the champion value.**

| final design | λ_min | % of champion | starts |
|---|---|---|---|
| `[0,1,2,3,5,6,7,8,11]` (the known champion) | 0.009263 | 100% | 80 |
| `[0,1,2,5,6,7,8,9,11]` (O-B:1, O-A:2) | 0.009263 | 100% | 25 |
| `[0,1,2,4,5,7,8,10,11]` | 0.008969 | 96.8% | 89 |
| `[0,2,3,4,5,6,8,10,11]`, `[0,2,4,5,6,8,9,10,11]` (tie) | 0.008392 | 90.6% | 26 |

- **A second design ties the champion** (equal to 6e-7). It was in the
  §7.1 sweep all along; only one was ever reported.
- **The traps are genuine one-swap local optima**, confirmed with the
  sweep's independent values: the 96.8% design's best neighbor is 0.008657,
  lower than its own value. The champion is exactly 2 swaps from it.
  The two 90.6% designs are each other's best neighbors.
- **The bound can't certify the traps**: some neighbors' bounds (up to
  0.0098) exceed the trap's value, so they had to be evaluated in full.
- Every run ended at ≥ 90.6% of the champion. Runs are short: median 2
  accepted swaps, max 3.
- **Cost per run (as if nothing were cached)**: median 59 sets bounded and
  11 evaluated in full, about 5 minutes. The full 220-start run took
  96 min (at most one evaluation per feasible set, thanks to the cache).
- Cross-check against §7.1's saved values: median difference 3e-5, all
  206 sets agree on feasibility. The largest difference (3%) is one set
  at λ ≈ 0.005 where one seed's restarts found a better local optimum;
  it doesn't affect any conclusion.

**Lesson**: one-swap search fixes greedy's tie-order mistakes when the
optimum is one swap away, but birds-foot's topology landscape has real
local optima. On its own it isn't reliable; its job is to supply a strong
starting incumbent.

Results: `results/birdsfoot_swap_validation_{results.json,summary.txt,starts.png}`
(figure: start vs. final λ_min for all 220 starts; three horizontal bands
are the champion and the two traps).

---

## 8. Roadmap (agreed 2026-10-13)

### 8.1 Goals and requirements

Three goals:

- **A. Reliable active-set selection** that works at every scale
  (two-panel, birds-foot, gore). Preferably a branch-and-bound (B&B) that
  prunes efficiently.
- **B. Multiple coupling types**: the optimizer chooses, per site, among
  a flat (1 constraint row), a V-groove (2 rows) and a tetrahedral
  socket (3 rows), as in a Kelvin coupling (1 + 2 + 3 = 6).
- **C. Couplings in other planes, on arms**: still a fixed list of
  candidate locations, but contacts may sit on rigid arms reaching across
  to the neighboring panel, with mating faces at any orientation.

Decisions (2026-10-13):

- **Exactly constrained designs only.** Total constraint rows =
  target_rank and C has full rank (so no redundant rows). This replaces
  "choose k couplings": with mixed types, the budget is rows, not
  couplings.
- **Every site may host every type** (for now).
- **Arms only ever connect adjacent panels**, so candidate panel pairs
  are still the crease graph's edges.
- **Mating faces may be at any orientation**, including parallel to the
  panel (e.g. an arm landing on the neighbor's top face).
- **Arm length is a fixed list of candidates for now.** A continuous
  length variable is a later possibility (see 8.6).

### 8.2 Consequences for existing results

- **The full-gore result (§7.4) is over-constrained.** 8 panels give
  target_rank 42; the k-bumps settled at 23 V-grooves = 46 rows, i.e. 4
  redundant rows. It is not a valid design under the exact-constraint
  requirement. Open question: does any exactly constrained 21-V-groove
  set reach full rank, or does the full gore need mixed types?
  (Greedy's particular 21-set failing doesn't settle it.)
- **"More couplings help" (§3, §7.6, k = 4–6)** is out of scope:
  those designs are over-constrained.
- Still valid: two-panel (3 V = 6 rows), birds-foot (9 V = 18 rows),
  single gore (15 V = 30 rows), and both 6-site experiments (k at the
  minimum).

### 8.3 Structural facts the plan relies on

- **Row counting under exact constraint.** For any cut of the crease
  graph splitting the panels into sides with n₁ and n₂ panels, rows
  inside side 1 can't exceed 6(n₁ − 1) (more would be redundant) and must
  be at least that (bridge rows can't see side 1's internal motion). So
  **rows crossing every cut ≥ 6**. This is necessary, not sufficient.
  The leaf rule (§3) is the special case of a one-panel side.
- **Bridge creases decouple.** A crease whose removal disconnects the
  graph gets *exactly* 6 rows, and those rows alone must lock that panel
  pair. So row allocation splits into independent subproblems per bridge
  and per loop-containing component. (λ_min itself still couples them,
  since it's an eigenvalue of the whole K.)
- **Flats and symmetric tetrahedrals have no angle.** A V-groove's K
  depends on θ through w(θ)w(θ)ᵀ. A flat contributes ffᵀ; a tetrahedral
  socket with three evenly spaced normals is isotropic in the face plane.
  Both are constant (Kc = Ks = 0), so the affine structure (§2.1), the
  fixed gauge subspace (§2.3), the per-coupling registry and the SDP all
  carry over unchanged, and the angle stage runs only over V-grooves.
- **Arms already work kinematically.** `get_interpolation_matrix` uses
  r = p − centroid, so a contact on a rigid arm is just a point far from
  the centroid. What's hard-coded is `_compute_groove_normals`, which
  assumes a vertical mating face and uses +Z as the reference direction
  (it breaks outright for a face parallel to the panel).

### 8.4 Plan, in order

**Phase 0 — swap search after greedy.** *Built and validated
(2026-10-20, §7.8–§7.9); not sufficient on its own.*
- Done: one-swap, best-improvement search with rigorous bound screening
  (`swap_search.py`). Two-panel: 20/20 starts reach the optimum.
  Birds-foot: 105/220; the rest stop at one-swap local optima.
- Options still open (can be combined):
  - *Two-swap escape when stuck*: the main birds-foot trap is exactly 2
    swaps from the champion. 108 extra sets per escape on birds-foot,
    ~3,800 at 24 sites (bound screening carries the load there).
  - *Multiple random starts*: each start reaches the champion with
    probability ~0.48, so ~5 starts would very likely find it. Cheap,
    but only a statistical guarantee.
- Not yet run: birds-foot 24 sites from greedy's set (§7.7; 4 swaps from
  the old champion), and from the old champion itself.
- Role going forward: supplies B&B's starting incumbent. Every birds-foot
  run ended at ≥ 90.6% of the champion, which is what B&B pruning needs.
- For mixed coupling types, a swap must keep the row count exact.

**Phase 1 — generalize the coupling model (goal B + groundwork for C).**
- A contact = a set of unit normals at a point, built in a local frame
  (face normal + explicit in-plane reference vector), with a type:
  flat (1 normal), V-groove (2, rotated by θ), tetrahedral (3, evenly
  spaced, fixed tilt).
- Replace the +Z assumption with an explicit reference vector so faces
  can have any orientation.
- **Open decision: stiffness per row.** Today every row has unit weight
  (K = CᵀC), which implicitly says a flat contact is as stiff as a groove
  wall, and treats arms as perfectly rigid. Use K = CᵀWC with a weight per
  row (contact stiffness, and later arm compliance in series:
  w = 1/(1/k_contact + 1/k_arm)). Decide before comparing types or arm
  geometries, or the optimizer will just reflect the unit-weight
  assumption.
- Validation: per-type rows against hand-derived values; the affine
  structure test (`test_K_theta_structure.py`) extended to mixed types;
  rank and gauge checks on a Kelvin coupling (should give exactly 6
  independent rows).

**Phase 2 — reliable selection by branch-and-bound (goal A).**
- Formulation: one choice per site from {none, flat, V, tet}; exact row
  budget Σ rows = target_rank. In the SDP relaxation: z_flat + z_V +
  z_tet ≤ 1 per site, Σ (rows × z) = target_rank.
- Pruning levers, in expected order of payoff:
  1. Incumbent from greedy + swap (Phase 0).
  2. Cheap combinatorial pruning before any SDP solve: the cut
     condition and bridge decomposition (8.3). Also resolves today's
     "unresolved node" ambiguity.
  3. Symmetry breaking (two-panel and birds-foot have mirror
     symmetries; §7.6 found tied mirror-image optima).
  4. Depth-first search branching on the most fractional z, to find good
     incumbents early.
  5. At leaves, use the certified bound (near-exact for complete sets) as
     a filter; run Riemannian only where the bound beats the incumbent.
  6. Only if still too slow: tighter relaxations (valid inequalities;
     Lasserre-style).
- Design the bound to accept a *list* of K matrices (objective = min
  λ_min over them): it costs nothing now and is exactly what the
  cyclic-symmetry reduction needs later (8.5).
- Validation ladder, with exhaustive ground truth where feasible:
  two-panel 6 sites (20 subsets) → birds-foot 12 sites (220 subsets
  known) → single gore (6 leaf couplings forced, choose 9 of 12: also 220
  subsets) → the same systems with mixed types. The full gore under exact
  constraint (k = 21, 3 forced, choose 18 of 24) is C(24,6) = 134,596
  subsets: too many to evaluate with Riemannian, but bound-only screening
  of all of them (~0.2 s each, ~7 h) is feasible as a partial check.
  (Corrected 2026-10-20: an earlier version said 17,550, which counted
  the over-constrained k = 23.)

**Phase 3 — arms and other planes (goal C).**
- Mostly candidate generation once Phases 1–2 exist: arm candidates are
  just more sites with their own points and face frames.
- Needs `draw_hinge_layout` generalized beyond vertical faces.
- Watch-outs: with rigid arms, rotational stiffness grows roughly with
  arm length², so the optimizer will always prefer the longest arm
  unless arm compliance is in the row weights; and `length_scale`
  (rotation vs. translation weighting) becomes a first-order effect.

### 8.5 Still open from earlier work

- **Physical λ_min for the assembled flasher**: full assembly vs.
  cyclic-symmetry (Bloch-wave) reduction — solve one gore under a swept
  inter-sector phase; global λ_min = minimum over the sweep. Phase 2's
  list-of-K bound prepares for this.
- **Birds-foot ridge trade-off**: 1-D continuation along the O-M
  bottom-face coupling's angle, then choose a stage-2 λ_min tolerance as
  design policy (§7.5).
- **Gore plateau vs. ridge**: run `polish_tied_restarts` on the gore
  results (re-run once an exactly constrained full-gore design exists).
- **Housekeeping**: independent child RNG per restart (§6.3); a
  `requirements.txt`; the polish's trust-region-floor stopping on
  birds-foot (§7.5).

### 8.6 Later

- **Continuous arm length / contact position.** Note that K is quadratic
  in contact position (C is linear in r), so the affine-in-θ structure
  and the per-coupling SDP registry don't cover it. It would need
  gradient-based optimization over positions, separate from the current
  machinery.
