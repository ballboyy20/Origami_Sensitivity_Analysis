# Kinematic Coupling Placement — Findings & Roadmap

Branch: `Kinematic-Coupling-Placement`. Last updated: 2026-11-25.

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
  §7.8–§7.9). **A two-swap escape fixes them: all 220 birds-foot starts
  now reach the champion**, and at 24 sites it climbs from the old
  champion to within 0.04% of the best known design (§7.18). A branch-and-bound prototype now finds the exhaustive
  optimum every time on two-panel and birds-foot, with 11 Riemannian
  evaluations instead of 206 (§7.10). **Its certified gap is 17.6% on
  birds-foot**, set by about 15 leaves whose own bound is loose. Arc
  branching (§7.11) and moment lifting (§7.12) both failed to tighten
  them: per-coupling methods can't, and full order-2 lifting (solved
  definitively with MOSEK) only moves the worst leaf from 17.6% to 15.4%.
  **Decided 2026-11-13** (§8.4): accept the gap and report it. At 24
  sites B&B certifies a new best design, 0.009630, within 15.7% (§7.14).
  **A one-restart screen** (stop a leaf after one restart if it is
  below 80% of the incumbent) never changes the certificate and cuts
  B&B time 1.4–2.6×. With it, the 24-site run finishes from a weak
  incumbent in 2.9 h. The cost is that the incumbent becomes "best
  found": a replay showed it can skip a leaf that would have raised the
  incumbent (§7.16). **On flasher gores the bound is nearly tight**:
  B&B matches the single-gore exhaustive optimum within 0.15% using 6
  evaluations instead of 216, and certifies the full gore within 1.74%
  (§7.17).
- **Coupling types (flat / V-groove / tetrahedral): built, validated on
  two-panel.** Every site may host any type under an exact row budget;
  the bound, branch-and-bound and evaluator all support it. On two-panel
  a V + flat + socket design reaches 0.017142, 128.6% of the best
  all-V-groove design (with unit row weights, §7.13). **On birds-foot
  (12 sites), mixed types beat the all-V champion by 11.8%**: 0.010358
  vs. 0.009263, certified within 33.0% (§7.15). But that B&B took 9.6 h
  and 1,977 leaf evaluations (all-V: 7 s, 11). **Decision (2026-11-18):
  equal row weights (K = CᵀC) for the foreseeable future**, so these
  comparisons stand as reported (§8.4 Phase 1). Mixed-type incumbents are
  still to do.
- **Choosing among equally good angles: tool built, policy open.**
  Stage-2 log_product polish works, but on birds-foot it only helps if a
  small λ_min sacrifice is allowed (§7.5).
- **Physical answer for the assembled flasher: not yet available.** Gore
  tests are algorithm stress tests with one arbitrary boundary
  condition. **The full gore now has a valid, exactly constrained
  baseline**: 21 V-grooves (42 rows for 42 DOFs), λ_min 0.006767,
  certified within 1.74%, 130.5% of the earlier over-constrained 46-row
  design (§7.17). Mixed-type 42-row designs exist, but the first search
  (no incumbent) stopped at the 24 h cap below the all-V result, so
  mixed vs. all-V on the gore is still open. **Whole flasher (hub + 5
  gores, 26 bodies, cyclic designs) is built**; one-swap search gives an
  all-V incumbent of 6.56e-4 (§7.19). **The 19 h all-V B&B (MOSEK stall
  fixed) raised it 20.5% to 7.904e-4**, uncertified. The mixed-type run
  only reached 6.70e-4 and never got to the all-V design that lies
  inside its own search space, so mixed vs. all-V is open here too
  (§7.20). **Caveat**: on cyclic designs λ_min is often an exact
  symmetry-forced pair, which the evaluator wrongly rejects as
  degenerate. 5 leaves were scored 0.0 that way, and flasher λ_min
  values may be underestimates until that's fixed (§7.20).
- **Soundness flag resolved: solver tolerance, not a modelling error**
  (§7.18). On the single gore, default Clarabel returns
  "optimal_inaccurate" on 212 of 216 sets and underestimates the bound by
  up to 2.0e-6 (1.5e-4 of the optimum). MOSEK solves every flagged set to
  "optimal" with the bound above λ_min. B&B's leaf bounds carry the same
  small error, so they need MOSEK or a safety margin (§8.5).

**Roadmap** (§8): swap search after greedy (built) → branch-and-bound
(prototype built; needs tighter leaf bounds) → generalize the coupling
model (flat / V-groove / tetrahedral, any face orientation: built) →
branch-and-bound selection under an exact row budget (built, two-panel
validated) → flasher → couplings on arms (moved after the flasher
2026-11-18, §8.6). Day-by-day schedule to Thanksgiving: §8.7.

---

## 1. Problem and Test Systems

**Model.** Each panel is rigid (6 DOFs). Each `KinematicCoupling` is a
sphere in a 90° V-groove on the shared mating face; it contributes 2
constraint rows (one per groove wall, normals n1, n2) to C. The groove's
sliding axis lies in the mating face, rotated by θ from an in-plane
reference direction `ref_dir` (default +Z). K = CᵀC. Since 2026-10-26 a
coupling can also be a flat (1 row, the face normal) or a tetrahedral
socket (3 rows, normals 120° apart at 45° from the face normal); neither
depends on θ (§7.13).

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

### 2.6 Why the certified bound can be loose, and what can't fix it

The bound relaxes each coupling's point x_i = (cos 2θᵢ, sin 2θᵢ) from the
unit circle to the filled disk. The disk is the smallest convex region
containing the circle, which an SDP needs; a point inside the disk is a
fake groove (the center = the average over all orientations). Because K
is linear in each x_i (§2.1), this disk relaxation is exactly the
Lagrangian dual of the real problem:

- real problem: max over angles of min over modes w of wᵀK(x)w
- bound: min over *mixtures* of modes of max over angles

They differ when the optimum balances two or more weak modes (the
double-eigenvalue structure of §2.4). The relaxation then blends several
couplings at once to fake a stiffness no real combination of angles has:
the looseness is **collective**.

Consequences (verified in §7.11–§7.12):
- Restricting any *one* coupling's angle doesn't help; the others
  compensate.
- No per-coupling constraint can fix it, in any solver: linear cuts on
  each x_i separately, LP or SOCP reformulations (which drop the SDP's
  coupling between modes and are only looser), or the SDP itself.
- Fixing it needs **product terms** xᵢxⱼ (a lifted relaxation such as
  the moment/Lasserre hierarchy), which carry information about which
  angle combinations are jointly achievable.

An equivalent way to see it: a point inside a disk is the average of a
random spread of groove angles, and K is linear, so **the disk bound is
the best λ_min of the expected stiffness matrix when each groove's angle
is drawn independently at random**. Anything that only constrains one
coupling at a time (arc cuts, linear multipliers) is also satisfied by
independent random grooves, so it can't separate them from real designs.
Only constraints that couple different grooves' angles can, and in the
moment hierarchy that first happens in the full order-2 localizing
matrix, which is the expensive part (§7.12).

The bound is still tight for most complete designs (median gap 8.5e-6 on
birds-foot); it's loose for a minority (up to 55% above achieved).

### 2.7 Whether a design can be fully constrained doesn't depend on luck

For a given set of couplings, the minors of C are trigonometric
polynomials in the angles, so either almost every angle choice gives full
rank, or none does. One random angle draw (a few, for numerical safety)
therefore decides feasibility exactly. Verified on birds-foot: all 206
feasible sets reached full rank on every one of 20 random draws (4,120
total); the 14 "no reference" sets on none. Failures are structural, and
not always caught by row counting: one of the 14 has spokes 3/2/2/2,
which passes every cut.

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
- **Birds-foot's two champions are exact mirror images** across the
  line through O, M and N (O-B ↔ O-A). Verified by geometry and by
  certified bounds of all 220 sets; their Riemannian values differ only
  by restart noise. 38 of the 220 sets are their own mirror image, so
  there are 129 distinct designs (§7.10).
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
| Topology-level SDP bound (`sdp_topology_bound.py`) | Sound; loose for partial sets. | Zero violations (root + 98 random nodes). Stays loose through ~10 of 12 decisions, tight only near leaves. Nearly exact for most complete subsets, but loose for some (up to 55% above achieved), for a structural reason (§2.6). Looseness flattens greedy's scores (§7.6–§7.7). |
| Riemannian gradient sampling (`riemannian_gradient_sampling.py`) | Validated; the achieved-value method. | Hosseini & Uschmajew's manifold version of Burke–Lewis–Overton. Crosses the 0/π seam cleanly; recovered all 56 subsets DE failed on; statistically tied with DE elsewhere. |
| Split-responsibility pipeline (`bound_and_evaluate.py`) | Validated at full scale. | SDP for the certified bound, Riemannian for the achieved value (§6, §7.1). |
| Forward bound-guided greedy (`forward_greedy_topology.py`) | Works on birds-foot; unreliable in general. | Exact birds-foot champion in seconds (§7.3). Misses with near-duplicate sites: 72.8% (two-panel) and 55% (birds-foot 24 sites), because tied bound scores leave early choices to tie order. Can't see mandatory couplings without forcing (§7.4). |
| One-swap local search (`swap_search.py`) | Validated on two scales; reliable only on two-panel. | Two-panel: 20/20 starts reach the optimum (1 swap each). Birds-foot: 105/220 (48%); the rest stop at one-swap local optima (96.8%, 90.6%). Bound screening prunes most swaps; ~5 min per birds-foot run (§7.8–§7.9). |
| Two-swap escape (`swap_search(two_swap=True)`) | Validated on birds-foot 12 and 24 sites; heuristic. | Birds-foot: 220/220 starts reach the champion, each stuck start with one escape (~15 extra evaluations). 24 sites from the old champion: 3 escapes to 0.009626, 99.96% of the best known; ~45 min per 2-swap search, dominated by 3,780 bound solves (§7.18). |
| Branch-and-bound (`branch_and_bound.py`) | Prototype validated on two-panel and birds-foot; mixed-type mode validated on two-panel. | Finds the optimum every time; birds-foot with 11 Riemannian evaluations instead of 206. Certified gap 0.01% (two-panel), 17.6% (birds-foot), limited by loose leaf bounds (§7.10). Mixed types (exact row budget, ≤ 1 type per site): two-panel 6 sites, exhaustive optimum found, gap 0.01% (§7.13). |
| Mixed coupling types (`coupling_types.py`, `RigidBodyModel.py`) | Validated (`test_coupling_types.py`, all checks pass). | Flat / V-groove / socket per site. Kelvin couplings reach rank 6 in every placement; two sockets leave rank 5. Flats and sockets have Kc = Ks = 0. V-groove-only designs reproduce the original C exactly (§7.13). |
| Arc branching on loose leaves (`arc_refine.py`) | Sound; ineffective here. | 3,000 nodes moved the worst leaf's bound 0.010890 → 0.010881. Our relaxation has no product terms, so per-coupling arc cuts can't reach the collective looseness (§2.6, §7.11). |
| Moment lifting on loose leaves (`moment_leaf_bound.py`) | Order 1 verified = disk bound; order 2 impractical here. | Linear-multiplier lifting: worst leaf 0.010890 → 0.010880. Full order 2: Clarabel needs 27.5 GB; SCS unconverged after 50 min, margin still positive at the champion (inconclusive, leans no) (§7.12). |
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
- **Leaf bounds (unit-modulus structure)**: each x_i on a circle is the
  "unit-modulus" constraint of phase-only beamforming and radar code
  design. Argument-division branch-and-bound with arc convex-hull cuts:
  Lu & Liu (IEEE Trans. Signal Process. 2017); Lu, Deng, Zhang & Fang
  (J. Glob. Optim. 70, 2018) — works there because their SDP is lifted
  (§7.11). Bound tightening: Coffrin, Hijazi & Van Hentenryck (QC
  relaxation for power flow, CP 2015 / IEEE TPS 2016). Moment-SOS
  hierarchy for polynomial matrix inequalities: Henrion & Lasserre (IEEE
  TAC 2006); sparse variants TSSOS / CS-TSSOS (Wang, Magron & Lasserre).
  Mixed-integer SDP for truss topology: Gally, Pfetsch & Ulbrich (SCIP-SDP).
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
| `swap_search.py` | One-swap local search with optional two-swap escape (`swap_search`, `two_swap=True`) and the cached, per-set-seeded `SetEvaluator` |
| `single_gore_soundness.py` | Bound-only soundness audit of the single-gore sweep: both bounds on every feasible set, exact re-solves (Clarabel / SCS / MOSEK) and a three-way λ_min check on violators; `--reverdict` (§7.18) |
| `dxf_to_panels.py`, `gore_system_common.py`, `flasher_gore_system.py`, `full_flasher_gore_system.py` | DXF geometry loading and gore systems |
| `forward_greedy_gore.py`, `forward_greedy_full_gore.py` | Gore greedy runs |
| `full_flasher_system.py` | Whole flasher: hub + 5 gores (26 bodies, target rank 150), 24 cyclic orbit candidates with tied angles |
| `flasher_timing.py`, `flasher_bnb.py` | Full-flasher per-solve timing check; full-flasher B&B (`allv`, `mixed`), MOSEK only (§7.19) |
| `polish_birdsfoot_champion.py`, `two_panel_six_sites.py`, `birdsfoot_six_sites.py` | 2026-10-13 experiments |
| `two_panel_swap_validation.py`, `birdsfoot_swap_validation.py` | 2026-10-20 swap-search validations |
| `branch_and_bound.py` | B&B prototype (`BranchAndBound`), `verify_symmetry`, `reflect` |
| `bnb_validation.py` | B&B validation against exhaustive ground truth |
| `arc_refine.py`, `test_arc_refine.py` | Arc-restricted leaf bounds and refinement (`LeafRefiner`); soundness tests |
| `moment_leaf_bound.py`, `moment_leaf_test.py` | Moment relaxations of a leaf: `MomentLeafRelaxation` (order 1/2), `LinearMultiplierRelaxation`; worst-leaf test |
| `coupling_types.py` | `TypedCandidates`: expands each site into one candidate per type (index = site × n_kinds + kind), row counts, design rules, exhaustive enumeration, symmetry lifting (`lift_perm`) |
| `test_coupling_types.py` | Coupling-type validation: rows vs. hand derivation, Kelvin/Maxwell rank, affine structure, regression, `ref_dir` |
| `mixed_types_validation.py` | Mixed-type bound and B&B vs. exhaustive ground truth (`two-panel`, `birds-foot`, `all`) |
| `gore_bnb_full_gore_allv_figures.py`, `birdsfoot_mixed_figures.py` | Design figures: full-gore all-V best (§7.17); birds-foot all-V vs. mixed best (§7.15) |
| `OPTIMIZATION_STRUCTURE.md` | How the pipeline's pieces fit together (call graph, B&B node steps, certificate); companion to this document |
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
- **Large semidefinite blocks and Clarabel**: an interior-point method
  needs memory ~ (packed block size)², so a 342×342 block needs ~27.5 GB.
  SCS fits but converges slowly; MOSEK works in the (smaller) dual space.
  Also: piping a long run through `grep` without `--line-buffered` hides
  its output until it exits.
- **Never verify a symmetry with Riemannian λ_min values**: restart
  noise (up to ~1% between mirror pairs) can reject a valid symmetry.
  Use `verify_symmetry` (exact geometry + certified bounds).
- **B&B implementation details** (from `OPTIMIZATION_STRUCTURE.md` §7):
  each evaluated leaf's bound is computed twice (`bound_node` to decide,
  then `sdp_certified_bound` inside `bound_and_evaluate`; same
  relaxation). Leaves with no full-rank reference after 300 tries are
  treated as infeasible, which is valid by §2.7. Forced completions skip
  the cut and rank tests (the leaf bound's reference search does the rank
  check). In a mixed design with at least one V-groove, flats and sockets
  still get (zero-gradient) θ variables in Riemannian.
- **MOSEK stalls on full-flasher B&B nodes** (§7.19): status UNKNOWN,
  which cvxpy raises as `SolverError`. `_solve` now retries once at
  relaxed tolerances and reports the result as "optimal_inaccurate"
  (B&B pads those node bounds by 1%); if the retry fails it raises
  `RuntimeError`, which every caller already handles.
- **False multiplicity alarms on cyclic designs** (§7.20, open): with
  tied angles, λ_min is often an exact symmetry-forced pair. The 1e-8
  simplicity check rejects it even though the gradient is well defined.
  Restarts burn their 30-attempt budget, and sets can be scored 0.0
  (infeasible) by `SetEvaluator`; a screen recheck then overwrites the
  leaf's known first-restart value with that 0.0.
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

### 7.10 Branch-and-bound prototype (2026-10-20)

**What it is** (`branch_and_bound.py`, Phase 2 of §8.4, V-grooves only).
A depth-first search over which couplings are on, off or undecided. At
each branch, cheapest test first:

1. Cardinality (enough undecided couplings left to reach k).
2. Cut counting (§8.3): ≥ 6 possible rows across every split of the
   panels; forced-on rows inside a group ≤ 6 × (group size − 1). Only
   applied when the design is exactly constrained.
3. Symmetry: keep only the lexicographically largest design of each
   mirror family, from user-supplied permutations.
4. Feasibility by random-angle rank checks (§2.7) — no "unresolved"
   status.
5. Certified bound: discard if bound ≤ incumbent × (1 + 1e-4).

Survivors branch on the most-fractional relaxed z. At complete designs
(leaves), the bound is checked first and Riemannian runs only if the
bound beats the incumbent. The incumbent starts from greedy + swap search
and updates whenever a leaf beats it. Output: the best design plus a
certified upper bound = max(incumbent × (1 + tol), largest bound among
evaluated leaves).

**Validation** (`bnb_validation.py`), against exhaustive ground truth.
Configurations: A no starting incumbent, no symmetry; B + symmetry;
C + greedy/swap incumbent (the planned setup); D start from the 96.8%
one-swap trap.

| system / config | found | certified gap | branches | Riemannian evals | time |
|---|---|---|---|---|---|
| two-panel A / B / C | 0.013330 ✓ | 0.010% | 23 / 25 / 17 | 3 / 2 / 0 | ≤ 2 s |
| birds-foot A | 0.009263 ✓ | 17.6% | 215 | 24 | 648 s (cold cache) |
| birds-foot B | 0.009263 ✓ | 17.6% | 213 | 14 | 52 s |
| birds-foot C | 0.009263 ✓ | 17.6% | 189 | 11 | 15 s |
| birds-foot D | 0.009263 ✓ | 17.6% | 193 | 11 | 13 s |

(Times after A are with leaf evaluations cached from earlier configs;
swap search for C's incumbent took another ~5 min.)

- **Finds the optimum every time**, including from the trap that stopped
  swap search, with **11 Riemannian evaluations instead of 206**.
- **Symmetry breaking nearly halves leaf evaluations** (24 → 14) but
  barely changes the branch count (215 → 213): most-fractional branching
  rarely decides the mirrored O-B/O-A sites early, so symmetry mostly
  prunes near the leaves.
- **Cut counting and the rank test pruned nothing** on these systems
  (every branch passed). They exist for the gores and mixed types.
- **The certified gap is 17.6% on birds-foot**, and it does not come from
  the tree search. 15 evaluated leaves have their own bound above the
  champion while achieving less (bound/achieved 1.14–1.55). The worst is
  the 96.8% trap `[0,1,2,4,5,7,8,10,11]`: bound 0.010890, achieved
  0.008969. It is its own mirror image, so symmetry can't remove it.
  §7.11 is the attempt to tighten these leaf bounds.

**Branching on mirror pairs first (2026-11-12).** `BranchAndBound(...,
branch_rule='symmetry_first')` picks the most-fractional coupling among
the pairs the symmetry test is still waiting on (birds-foot: 3/9, then
4/10, then 5/11), and falls back to most-fractional otherwise. Rerun of
`bnb_validation.py` with configs Bs/Cs/Ds (= B/C/D with this rule); the
old configs reproduced their earlier counts exactly:

| birds-foot | B → Bs | C → Cs | D → Ds |
|---|---|---|---|
| branches | 213 → **97** | 189 → **97** | 193 → **97** |
| Riemannian evals | 14 → 11 | 11 → 11 | 11 → 11 |
| pruned by symmetry | 37 → 6 | 28 → 6 | 29 → 6 |

- **Branches drop 54%** (213 → 97) and all three find 0.009263. Fewer
  symmetry prunes but higher in the tree: 6 cuts near the root remove what
  37 cuts near the leaves did before, and more.
- **The starting incumbent no longer matters**: Bs, Cs and Ds visit the
  same 97 nodes, because Bs reaches the champion early anyway.
- Two-panel: 25 → 17 (B), 17 → 13 (C).
- **The certified gap is unchanged** (17.6%), as expected: it is set by
  leaf bounds, not by the search.

**Lesson about symmetry verification.** The first run rejected the
birds-foot mirror because it compared Riemannian λ_min values of each set
and its mirror image, and those carry restart noise (mirror pairs differ
by up to 0.94%). `verify_symmetry` now checks exact geometry (reflected
points and face normals) and that each set and its mirror have the same
*certified bound* (deterministic): birds-foot mirror verified to 9e-7
over all 220 sets; a deliberately wrong permutation fails both checks.

Results: `results/bnb_validation_{results.json,summary.txt,counts.png}`
(figure: branches visited and Riemannian evaluations per configuration).

### 7.11 Tightening loose leaf bounds with arc branching: tried, ineffective (2026-10-20)

**Method** (`arc_refine.py`; arc cuts added to `sdp_topology_bound` as
an optional `arcs` argument, off by default). From the unit-modulus
literature (§5): restrict each coupling's point x = (cos 2θ, sin 2θ) to an
arc of the circle; the arc's convex hull is the disk plus one chord cut,
⟨x, m⟩ ≥ cos(w/2), with m the arc's mid-direction and w its width
(Lu, Deng, Zhang & Fang 2018, sec. 2). Two pieces:

- **B. Bound tightening** (Coffrin, Hijazi & Van Hentenryck): for each
  coupling, bound each half of its arc; discard halves that can't beat
  the incumbent.
- **A. Best-first branching on arcs** (Lu–Deng Algorithm 1): branch on
  the coupling whose relaxed point is furthest from its arc, bisect that
  arc; round each node's relaxed solution onto the arcs as a candidate
  design, polished with Riemannian if promising.

**Soundness** (`test_arc_refine.py`, all pass, checked on the real
system): no arcs = identical bound; 2,700 angle vectors sampled inside
random arcs, 0 above their bound (min margin 1.6e-3); halving an arc never
raised the bound (24 cases); refined bounds never below achieved values.

**Result on the worst leaf** (bound 0.010890; true optimum 0.008969,
confirmed by the best 5 of 40 Riemannian restarts):

- **B removed nothing.** Restricting any *single* coupling, even to an
  eighth of its circle, leaves the bound at 0.01089: the other couplings
  rearrange to compensate.
- **A: 3,000 nodes, 6,000 SDP solves, 11.5 min → bound 0.010881.** Not
  one node was pruned (every node produced two surviving children).
  Rounding gave designs at λ ≈ 0.0002: the relaxed points sit too deep
  inside their disks to round to anything useful.
- **The bound only reaches the incumbent when all 9 couplings are pinned
  together.** Arcs of equal width w on every coupling, centred on the
  optimum:

  | w (in 2θ) | 360° | 90° | 45° | 22.5° | 11.25° | 5.6° | 2.8° |
  |---|---|---|---|---|---|---|---|
  | bound above achieved | 21.4% | 20.8% | 16.9% | 9.3% | 3.3% | 0.9% | 0.2% |

  Proving this leaf needs ~11° on every coupling at once (≈ five halvings
  each, ~45 levels deep near the optimum).

**Why it worked for Lu & Deng but not here.** Their relaxation is
*lifted*: a matrix variable C ⪰ ccᴴ holds every product cᵢcⱼ*, so
narrowing one variable's arc tightens its interactions with all the
others. Ours has no product terms — K is linear in x — so an arc cut
only constrains each coupling on its own, while the looseness is
*collective* (§2.6). Arc branching is the right tool only on top of a
lifted relaxation.

The arc-width table, the 40-restart check and the 3,000-node run were
one-off diagnostics (not saved as scripts); each is a few lines with
`arc_refine.LeafRefiner`.

### 7.12 Tightening loose leaf bounds with moment lifting: tried, not practical (2026-10-20)

**Question**: does adding product terms xᵢxⱼ (§2.6) prove the worst loose
leaf `[0,1,2,4,5,7,8,10,11]` (disk bound 0.010890, true optimum 0.008969)
can't beat the champion (0.009263)?

**Method** (`moment_leaf_bound.py`, `moment_leaf_test.py`). The leaf
question "is there any x on the 9 circles with K(x) − τI ⪰ 0?" is a
polynomial matrix inequality. Its moment relaxation (Henrion & Lasserre
2006) replaces x by moments y (y₀ = 1) with a PSD moment matrix, a PSD
localizing matrix for K(x) − τI, and the circle equalities. For
robustness the code maximizes a margin ε (localizing matrix ⪰ εI):
ε < 0 proves no design reaches τ; every real design gives ε ≥ 0.

- **Order 1 = the disk bound**, as expected: the margin changes sign
  exactly at 0.010890 (a correctness check of the construction).
- **Order 2** (all degree-2 products): 7,315 moments, PSD blocks 190×190
  and 342×342.

**Results:**

| relaxation | worst-leaf result | proves the leaf? |
|---|---|---|
| disk (order 1) | bound 0.010890 | no |
| linear-multiplier lifting (`LinearMultiplierRelaxation`), 4 / 8 / 16 directions per coupling | bound 0.010884 / 0.010880 / 0.010880 | no |
| full order 2, Clarabel | ran out of memory: tried to allocate 27.5 GB (7.6 GB machine) | — |
| full order 2, SCS (first-order solver) | margin **+4.6e-4** at τ = 0.009264; +5.6e-4 at τ = 0.008968 (validity). Both stopped at the 50-min limit, `optimal_inaccurate` | inconclusive, leans no |
| **full order 2, MOSEK** (2026-11-13, lab machine, ~30 s per solve, all `optimal`) | margin **+4.3e-4** at τ = 0.009264; +5.3e-4 at τ = 0.008968 (valid). Bisection: **order-2 bound 0.010694** | **no** (definitive) |
| needed | bound ≤ 0.009264 (margin < 0) | |

- **Linear multipliers** ((1 + ⟨u, xᵢ⟩)·(K(x) − τI) ⪰ 0, valid on the
  circle; only degree-2 moments, solves in seconds) **don't help.** They
  only add information about each coupling on its own (see the
  interpretation below).
- **Full order 2 is impractical here.** Clarabel (interior point) forms
  a dense matrix over the 342×342 block's 58,653 packed entries →
  58,653² × 8 bytes = 27.5 GB. SCS fits in memory but didn't converge in
  50 min per solve. With ~15 loose leaves on birds-foot and larger gores,
  it isn't usable routinely even if it worked.
- **The SCS margin at the champion stayed clearly positive** (+4.6e-4).
  An unconverged solve can't certify anything, so this isn't proof that
  order 2 fails, but it suggests order 2 also can't rule this leaf out.

**Interpretation (why per-coupling information can't help).** A point
inside a disk is the average of some random spread of groove angles. K is
linear in x, so the disk bound equals the best λ_min of the *expected*
stiffness matrix when each groove's angle is drawn **independently at
random**. Any constraint that only involves one coupling at a time (arc
cuts, linear multipliers) is also satisfied by such independent random
grooves, so it can't separate them from real designs. Only constraints
that couple different grooves' angles (the full order-2 localizing
matrix, with every E[xₐx_b(K(x) − τI)] in one PSD block) can, and that is
exactly the expensive part.

**MOSEK settles it (2026-11-13): order 2 does not prove the leaf.**
(`moment_leaf_test.py --solver MOSEK`, 16 GB lab machine.) MOSEK works in
the moment space, so each order-2 solve took ~30 s, not the hours SCS
needed, and every solve returned `optimal`. The order-2 bound is
0.010694: 19.2% above the leaf's achieved value (disk: 21.4%) and 15.4%
above the champion (disk: 17.6%). Order 2 tightens this leaf by only
1.8%. Even applied to every loose leaf, the birds-foot gap could fall at
most from 17.6% to 15.4%, since this leaf would still set it. Order 3 is
far larger. This confirms the interpretation above in practice: the
looseness is collective, and the first lifted level barely touches it.
Results: `results/moment_leaf_test_mosek_summary.txt`.

**Possible next attempts, if this is revisited:**
- **MOSEK** (free academic license via BYU) for one definitive order-2
  answer on this leaf. It works in the 7,315-moment space (dense Schur
  complement ~0.4 GB instead of 27.5 GB), so it should solve accurately
  in minutes to tens of minutes. Settles whether order 2 proves the
  leaf, but likely still too slow for routine use.
- **Shrink the moment problem**: on the circles, x_c² + x_s² = 1 makes
  many moments redundant (trigonometric / complex-variable reformulation),
  and the birds-foot mirror symmetry can block-diagonalize the matrices.
  Sparse hierarchies (TSSOS / CS-TSSOS) are another route.
- Combining order 2 with arc branching (the Lu–Deng structure transferred
  correctly) only makes sense once a single order-2 solve is affordable.

### 7.13 Mixed coupling types: model, B&B, two-panel validation (2026-10-26)

**What was built** (Phase 1 of §8.4, plus the mixed-type form of Phase 2):

- **Coupling model** (`RigidBodyModel.KinematicCoupling`): a `kind` of
  `'flat'` (1 row: the face normal), `'vgroove'` (2 rows, as before) or
  `'tet'` (3 rows: normals 120° apart about the face normal, tilted 45°
  like the groove walls). Each row has unit weight. A flat's K is fixed,
  and the socket's Σ nnᵀ is the same at every θ, so both have
  Kc = Ks = 0 and the affine structure (§2.1) still holds.
- **Any face orientation**: θ is now measured from an explicit in-plane
  `ref_dir` (default +Z, which reproduces the old frame exactly). A face
  parallel to the panel must supply its own `ref_dir`, and raises without
  one. This removes the +Z limitation noted in §8.3.
- **Candidates** (`coupling_types.TypedCandidates`): each site becomes
  one candidate per type, so the registry, calibration, rank tests and
  Riemannian all work unchanged on candidate indices. A valid design has
  at most one candidate per site and exactly target_rank rows.
- **Bound** (`sdp_topology_bound(..., rows=, sites=)`): Σ rows·z equals
  the remaining row budget, Σ z ≤ 1 per site. Angle-free candidates
  contribute z·K0 only.
- **B&B mixed mode** (`BranchAndBound(..., rows=, site_of=)`): the budget
  is rows, not k. Turning a candidate on turns off its siblings at that
  site. Cut counting uses rows, plus a new rule: at most 6 forced-on rows
  per crease (so a bridge crease gets exactly 6).
- **Evaluator**: a design with no V-grooves has a fixed K, so its λ_min
  is computed exactly (gap 0) with no optimization.

**Unit tests** (`test_coupling_types.py`, all pass, rerun 2026-11-09):
rows and Σ nnᵀ identities match hand derivations; on two-panel, all 6
Kelvin placements (T+V+F) and Maxwell (V+V+V) reach rank 6, and every
two-socket pair leaves rank 5 (rotation about the line joining them); the
affine model holds for mixed designs to 4e-15 with Kc = Ks = 0 (max
9e-16) for flats and sockets; V-only designs in the expanded birds-foot
system give exactly the original C; `ref_dir` behaves as specified.

**Two-panel validation** (`mixed_types_validation.py two-panel`, against
exhaustive ground truth):

| system | designs (feasible) | exhaustive optimum | B&B |
|---|---|---|---|
| 3 sites | 10 (7) | **0.017142**, [V F T] and its mirror [F V T] | found, gap 0.010%, 7 nodes, 1 leaf evaluated |
| 6 sites (+ thickness mirrors) | 336 (126) | **0.017142**, [V F T] on top or bottom set, 4 tied | A (no symmetry): found, 835 nodes, 24 s. B (3 verified symmetries): found, 583 nodes, 14 s. 1 leaf evaluated in each |

- **Mixed types beat all V-grooves**: 0.017142 vs. 0.013330 (128.6%).
  The socket goes at the bottom-face midpoint p3; the V-groove and flat
  at the two top-face ends. Next best: socket at an end (0.014106); all-V
  is third (0.013330).
- **Assumes unit row weights**: a flat row is as stiff as a groove wall.
  Equal weights were adopted on 2026-11-18 (§8.4 Phase 1); the ranking
  could change under non-uniform weights.
- Symmetries (thickness mirror, y-mirror, both) verified by exact
  geometry and certified bounds over 300 designs (max relative bound
  difference 2.3e-7).
- Soundness: 120 random partial nodes, bound ≥ best completion in every
  case (min margin −8e-13, i.e. rounding), 0 violations.
- **The rank test prunes in mixed mode** (it pruned nothing on the
  V-only runs, §7.10): 21 nodes in A and 10 in B, and most leaves were
  rank-infeasible (118 of 145 in A, 38 of 44 in B), e.g. two sockets
  leave rank 5. Cut counting still pruned nothing.
- Exhaustive 6-site search: 342 s and 126 Riemannian evaluations, vs. 14 s
  for B&B with symmetry.

**Not yet done** (as of 2026-10-26): the `birds-foot` part of the script
(B&B with the all-V champion as incumbent, plus a 300-design
random-sample check) had not been run; it ran 2026-11-13/14 (§7.15). Greedy is still V-groove only, and swap search in mixed
mode only finds like-for-like swaps, so neither can produce a mixed-type
incumbent yet.

Results: `results/mixed_types_validation_two_panel_{results.json,summary.txt}`.

### 7.14 Scale test: birds-foot 24 sites (2026-11-12)

**Question** (Day 2 of §8.7): does B&B still work at m = 24, k = 9 (§7.7,
C(24,9) ≈ 1.3M designs, no exhaustive ground truth)?

**Setup** (`birdsfoot24_scale_test.py`, three modes run as separate
processes, 2 BLAS threads each, seed 31): one-swap search from greedy's
set and from the old champion; B&B with mirror pairs first (§7.10) and
three verified symmetries: the O-B ↔ O-A spoke mirror, the mid-thickness
mirror (site j ↔ j+3 on every spoke) and both (exact geometry; bounds over
300 designs each, max relative difference 7e-6, 0 feasibility
mismatches). Each B&B was capped at 8 h.

**Swap search found a better design than the old champion:**

| start | final | λ_min | swaps | time |
|---|---|---|---|---|
| old champion `[0,1,2,6,8,12,13,14,20]` (0.009263) | unchanged: a one-swap local optimum | 0.009263 | 0 | 13 min |
| greedy `[0,2,5,11,12,14,17,19,20]` (0.005106) | `[0,1,2,7,9,13,14,19,21]` | **0.009630** (104.0%) | 6 | 40 min |

The new design keeps spoke 0's original sites and spoke 2's 80% top + 50%
bottom; on spokes 1 and 3 it uses the 80% top site plus the *mirrored*
20% site (on the bottom face). It is mirror-symmetric (O-B and O-A use
the same layout). This settles §7.7's open question: mirrored sites do
beat the 3-site champion. The old champion is a one-swap trap, 2 swaps
from the new design (as in §7.9).

**Branch-and-bound:**

| | B&B from 0.009630 | B&B from old champion |
|---|---|---|
| complete | **yes**, 6.2 h (+13 min symmetry checks) | no, stopped at the 8 h cap |
| best found | 0.009630 (no leaf beat it) | 0.009630, found itself at node 19,597 (via 0.009310, 0.009392, 0.009597) |
| certified upper bound | 0.011143 (**gap 15.7%**) | none (incomplete) |
| nodes | 23,725 | 26,694 |
| pruned by bound / symmetry / rank / cut | 7,762 / 364 / 39 / 152 | 8,425 / 373 / 56 / 172 |
| leaves: pruned by bound / evaluated | 2,887 / **633** | 3,406 / 873 |

- **B&B works at 24 sites and certifies 0.009630 within 15.7%**: every
  design either has a bound ≤ the incumbent or was evaluated, and none of
  633 evaluated leaves beat it. As at 12 sites, the certificate is
  relative to leaf evaluation.
- **But it is ~3,000× slower than at 12 sites** (6.2 h vs. 7 s; 633
  Riemannian evaluations vs. 11), and **without a strong starting
  incumbent it doesn't finish in 8 h**. At 24 sites the incumbent matters,
  unlike at 12 (§7.10).
- **The cost is loose leaf bounds, not the tree.** Every evaluated leaf
  had a bound above 0.009630 while achieving less: median bound/achieved
  2.0, max 6.3 (vs. 1.14–1.55 at 12 sites). The loosest, bound 0.011143,
  achieves only 0.005215. Each such leaf needs a full Riemannian run.
- **Cut counting and the rank test pruned for the first time** on an
  all-V-groove system (152 and 39 nodes), though bound pruning dominates.
- Nearest rivals (all < 0.009630): 0.009597 `[0,2,7,9,12,13,16,19,21]`,
  0.009392, 0.009391, 0.009380.

**Verdict (Day 2 go/no-go): go at 24 sites, no-go as-is for the
flasher.** B&B with a swap-search incumbent completes and certifies, but
the 6 h cost comes from ~600 leaves whose bounds are about 2× their true
value. Larger systems need either a cheaper screen for loose leaves
(e.g. one Riemannian restart first, full evaluation only if close to
the incumbent), enforced cyclic symmetry (Day 10), or the swap-based
fallback (Day 3).

Results: `results/birdsfoot24_{swap_greedy,swap_champion,bnb,bnb_inc}_{results.json,summary.txt}`.

### 7.15 Mixed coupling types: birds-foot, 12 sites (2026-11-14)

**Question** (Day 4 of §8.7): with flat / V-groove / tetrahedral socket
allowed at every site, does any design beat the all-V champion 0.009263?

**Setup** (`mixed_types_validation.py birds-foot`, lab machine, started
11/13): the 12 birds-foot sites (§7.1) × 3 types = 36 candidates, exact
budget of 18 rows, ≤ 1 type per site, unit row weights, socket walls at
45°. About 1.7M valid designs, so no exhaustive ground truth. B&B started
from the all-V champion as incumbent (its λ_min reproduced exactly), with
mirror pairs first and the O-B ↔ O-A spoke symmetry (verified: bounds
over 60 random designs, max relative difference 1.5e-7, 0 feasibility
mismatches). 12 h cap; then a 300-design random-sample check. Design
notation: one letter per site, 3 sites per spoke, `-` = no coupling.

**Result: mixed types win, by 11.8%.**

| | value |
|---|---|
| best found | **0.010358** `[FFT V-V V-T V-V]` (111.8% of the all-V champion) |
| certified upper bound | 0.013780 (**gap 33.0%**; all-V at 12 sites: 17.6%) |
| complete | yes, 9.6 h (34,470 s) |
| nodes | 151,645 |
| pruned by bound / symmetry / rank / cut / cardinality | 42,704 / 693 / 1,289 / 2,662 / 10,126 |
| leaves: pruned by bound / infeasible / evaluated | 11,076 / 5,296 / **1,977** |
| SDP errors | 0 (2 "solution may be inaccurate" solver warnings in the log) |
| random-sample check | 300 designs, 83 full rank, none beat 0.010358 or had a bound above it: PASS |

- **The best design is a near-tie, not a single winner.** Within 0.5%:
  0.010358 (two distinct designs tie), 0.010350 `[FFT V-V T-V V-V]`, 0.010324, 0.010308, and the
  mirror-symmetric `[F-T VFV F-T VFV]` at 0.010304.
- **The symmetric design is nearly proven at its own angles**: its leaf
  bound is 0.010305 vs. achieved 0.010304 (and two close relatives are
  tight to within 1e-6). It is a strong design of record if a
  defensible, symmetric design matters more than the last 0.5%.
- **The win is broad**: 81 of the 1,977 evaluated leaves beat 0.009263.
  It also beats the 24-site all-V best (0.009630, §7.14) using only 12
  sites.
- **Incumbent history**: 0.009298 (194 s) → 0.009875 → 0.010299
  (3,064 s) → 0.010350 → 0.010358 at 11,602 s (3.2 h). The remaining
  6.4 h was spent closing the tree.
- **The gap comes from very loose leaf bounds, as in §7.10 and §7.14 but
  worse.** Evaluated leaves had median bound/achieved 2.3 (90th
  percentile 5.7, max 9.2; all-V at 12 sites: 1.14–1.55). The loosest,
  `[FVF -VV VFT -VV]`, has bound 0.013780 but achieves 0.002514; its true
  optimum is almost certainly near the latter. Per the 2026-11-13
  decision (§8.4) the gap is accepted and reported.
- **Cost: ~5,000× the all-V run** (9.6 h vs. 7 s; 1,977 evaluations vs.
  11). From the progress log, roughly three-quarters of the wall time was
  in intervals containing leaf evaluations (about 13–14 s per leaf at 5
  restarts), the rest in node bounding.

**Caveat: unit row weights.** The 11.8% assumes a flat contact row is as
stiff as one groove wall or socket wall. Under Hertz contact, stiffness
grows as F^(1/3), and one preload splits differently across a flat, a
groove and a socket, so realistic weights could shrink or reverse the
win. *2026-11-18: equal row weights adopted for the foreseeable future
(§8.4 Phase 1), so the result stands as reported under that assumption.*

**Implications**:

- If non-uniform weights are revisited later: re-running a 10 h B&B per
  weight choice is too costly. Re-score the top ~100 evaluated leaves
  under 2–3 candidate W as a screen (not a certificate: leaves pruned
  under W = I aren't pruned under other W), then one full B&B for the
  chosen W.
- **Cheaper leaf evaluation is now needed.** B&B's certified bound is
  max(threshold, evaluated leaf bounds) and never uses a leaf's achieved
  value, so evaluation only searches for a better incumbent. A 1-restart
  screen (all 5 restarts only if the first comes within ~20% of the
  incumbent) would keep the certificate unchanged. Even using best-of-5
  as a stand-in for the first restart, 91% of evaluated leaves were below
  80% of the running incumbent, so the screen would have cut evaluation
  time by roughly 3–4×.
- Mixed-type B&B on the flasher is out of reach without enforced cyclic
  symmetry (Day 10), which makes the Day 3 two-swap fallback a
  requirement rather than a backup.

Figures (`birdsfoot_mixed_figures.py`, 2026-11-17): the all-V champion
`[VVV V-V VVV --V]` next to the mixed best `[FFT V-V V-T V-V]`, as 3-D
views at the achieved angles (contacts marked by type with their normals,
weakest eigenmode overlaid) and as per-spoke hinge slices. Both designs
were re-evaluated with the same `SetEvaluator` settings (seed 31) and
reproduce the saved λ_min (0.009263 and 0.010358). Their own leaf bounds
are 0.009271 (all-V) and 0.012215 (mixed). Saved:
`results/birdsfoot_mixed_vs_allv_{3d,hinges}.png` and the angles in
`results/birdsfoot_mixed_vs_allv_designs.json`.

Results: `results/mixed_types_validation_birdsfoot_{results.json,summary.txt}`;
full log `mixed_bf.log` (UTF-16).

### 7.16 One-restart screen: validation (2026-11-16)

**Question**: does the one-restart screen proposed in §7.15 leave the
search and certificate unchanged, and how much time does it save?

**Method** (`branch_and_bound.py` `screen_frac`, `screen_validation.py`;
lab machine, weekend of 11/14, 8 runs in parallel at 2 BLAS threads each,
see `LAB_QUEUE.md`). With screen fraction f, an evaluated leaf runs one
Riemannian restart first. If that restart is below f × the running
incumbent, the leaf stops there; otherwise it gets all 5 restarts and
the same result it would get unscreened. Random starts are drawn the
same way in both cases. The certified bound never uses a leaf's achieved
value, so the certificate can't change. Only the incumbent can be worse.
After the search, the 20 screened leaves with the highest first-restart
values get a full evaluation (the "recheck"). All runs used f = 0.8.

**Results**:

| run | unscreened | screened | same certificate | same tree |
|---|---|---|---|---|
| small systems (two-panel 3 and 6 sites, birds-foot all-V 12 sites) | optimum each | optimum each | yes | yes (PASS) |
| birds-foot mixed, 12 sites (§7.15) | 34,470 s | 25,244 s (**73%**) | yes, 0.013780 | yes, identical incumbent and best 0.010358 |
| birds-foot 24, from best 0.009630 (§7.14) | 22,489 s | 8,474 s (**38%**) | yes, 0.011143 | nearly (see below) |
| birds-foot 24, from old champion (§7.14) | incomplete at 8 h | **complete in 10,311 s (2.9 h)** | now certified, 0.011143 | n/a |

- **The certificate was identical in every run, and so was the best
  design.** The small differences in tree counts on the 24-site run from
  0.009630 come from re-evaluating the starting incumbent: it came out
  1e-8 lower than in the saved run, which shifted a handful of pruning
  decisions. The 633 evaluated leaves and the bound are the same.
- **Most leaves stop after one restart**: 1,806 of 1,998 evaluations
  (mixed), 541 of 654 and 779 of 905 (24 sites).
- **The 24-site run from the old champion now finishes.** It finds
  0.009630 by itself (via 0.009310, 0.009392, 0.009597) and certifies it
  within 15.7%. Unscreened, it hit the 8 h cap with no certificate.
  B&B is therefore much less dependent on a strong starting incumbent
  than §7.14 concluded.
- **The mixed-run saving (27%) is well short of the 3–4× predicted in
  §7.15**, even though the screen cut restarts from 9,885 to 2,661 (−73%).
  Either node bounding is a bigger share of that run than §7.15
  estimated, or the 8 parallel runs slowed it. The 24-site runs did save
  2.6×.
- **The recheck never changed anything**: 0 incumbent updates in any
  run. On birds-foot, a rechecked leaf's full value was at most 1.4%
  above its first restart.

**Replay check: the screen can lose an incumbent.** The `replay` mode
re-ran the first restart for every leaf of the saved 9.6 h mixed run
(§7.15) whose best-of-5 was ≥ 0.6 × the running incumbent (570 leaves),
and checked whether any incumbent-raising leaf would have been screened.
Re-evaluating the 9 incumbent-raising leaves reproduced them exactly.

- Restart 1 vs. best of 5: median ratio 1.000, 10th percentile 0.889,
  min 0.218. Restart 1 was the best of the 5 in only 125 of 570 leaves.
- Screened share: f = 0.7: 87%; f = 0.8: 91%; f = 0.9: 97%.
- **At every f, one incumbent-raising leaf is lost**:
  `[0,5,7,13,15,22,26,31,34]`, best-of-5 0.009968 but restart 1 only
  0.003835, at a time when the incumbent was 0.009875. Lowering f to 0.7
  doesn't save it; it would need f < 0.39.
- In the actual screened run this cost nothing: 0.009984 turned up 162
  nodes later and the tree was unchanged. On a less forgiving problem
  it could lose the best design.

**Verdict: adopt the screen (f = 0.8) for long runs.** It never changes
the certificate and saves 1.4–2.6× here. The tradeoff is that the
incumbent is now a heuristic, as with greedy and swap search: a
screened run's best design is "best found", not "best among evaluated
leaves". For a design of record, either re-run the final top leaves
unscreened or run without the screen when time allows. The 20-leaf recheck
selects by first-restart value, so it can't catch a leaf like the one
above (first restart 38% of its best). A recheck ranked by leaf bound
would.

Results: `results/screen_validation_{small,replay,birdsfoot_mixed,birdsfoot24_best,birdsfoot24_champion}_{results.json,summary.txt}` and `.log`.

### 7.17 Gore B&B: single gore vs. exhaustive, full gore under exact constraint (2026-11-16)

**Questions** (Days 3 and 6 of §8.7): does B&B find the exhaustive
optimum on the single gore? Is there an exactly constrained (42-row)
full-gore design, all V-grooves or mixed, and how good is it?

**Setup** (`gore_bnb.py`; same lab-machine batch as §7.16). Geometry,
sites and forced leaf couplings as in §7.4. Every B&B here started with
no incumbent and no symmetry, and the full-gore runs used the screen at
f = 0.8 (they started before §7.16's validation finished). Design
notation for mixed runs: one letter per site, 3 sites per crease.

**Single gore** (m = 18, k = 15, target rank 30; forced 6, choose 9 of
12 = 220 sets):

| | value |
|---|---|
| exhaustive | 216 of 220 feasible; optimum **0.013044** `[0,1,4,5,6,7,8,9,10,11,12,13,14,16,17]`; 12,741 s (~59 s per evaluation) |
| runner-up | 0.012925 (99.1%) |
| greedy (§7.4) | 0.012351 (94.7% of optimum) |
| B&B, unscreened | finds 0.013044; certified UB 0.013064 (**gap 0.15%**); 89 nodes, **6** evaluations, 339 s |
| B&B, screened | identical tree and result, 0 leaves screened, 395 s |

- **B&B matches the exhaustive optimum** with 6 Riemannian evaluations
  instead of 216, and the gap is far tighter than on birds-foot.
- **The run reports OVERALL: FAIL, but only because of the soundness
  check on the exhaustive sweep.** One set's bound was **1.68e-6 below its
  achieved value**. The check allows 1e-8 relative error (1.3e-10 here).
  The violation is about 1e-4 relative to the optimum; earlier systems
  stayed under about 1e-9 (§7.1, §7.6). The likely cause is SDP solver
  accuracy on the larger gore SDPs, but this is **unconfirmed**: the
  exhaustive results store only λ_min per set, so the offending set is
  unknown. A bound that low could in principle prune a leaf that is
  within 1.7e-6 of the incumbent. Here it didn't matter, since B&B's
  result matches ground truth.
- *Resolved 2026-11-17 (§7.18)*: solver tolerance. Default Clarabel
  underestimates the bound by up to 2.0e-6; MOSEK puts every flagged
  set's bound above its achieved value.

**Full gore, all V-grooves, exactly constrained** (m = 27, k = 21,
target rank 42; panel 5's 3 couplings forced, choose 18 of 24 = 134,596
sets):

- **Rank sweep (17 s): 46,656 sets (34.7%) reach full rank at generic
  angles.** So the 23-coupling result in §7.4 wasn't necessary: greedy
  just missed the 21-coupling sets. There are 400 distinct per-crease
  count patterns. The smallest relative σ₄₂ among feasible sets is 4.9e-12
  (median 8.1e-5), close to the ~1e-13 level where rank decisions get
  suspect, so a few sets are only marginally full rank.
- **B&B: best 0.006767**, `[0,1,4,5,6,7,8,9,11,12,14,15,17,18,19,20,21,22,23,25,26]`
  (couplings per crease 2,2,3,2,2,2,3,3,2), **certified UB 0.006885 (gap
  1.74%), complete in 32,448 s (9.0 h)**.
- **130.5% of §7.4's over-constrained greedy design** (0.005185, 23 V =
  46 rows). Using fewer couplings and the exact row count gives a better
  design.
- Tree: 7,625 nodes; pruned by bound / rank / cut 2,736 / 180 / 117;
  leaves 780: 289 pruned by bound, 329 infeasible, **162 evaluated**, 0
  screened.
- **Leaf bounds are nearly tight on the gore**: evaluated leaves had
  median bound/achieved 1.02 (90th percentile 1.03, max 1.06), vs.
  2.0–2.4 on birds-foot (§7.14–§7.15). That's why the gap is small and
  why the screen never fired: every evaluated leaf had a bound above the
  incumbent, so it also achieved close to it. The 9 h goes to larger SDPs
  per node and slower evaluations, not to loose leaves.
- Incumbent: 0.005243 at 2 min, 0.006685 (98.8%) at 37 min, final
  0.006767 at 7.5 h (18 updates).
- **Nearly as stiff as using every candidate**: all 27 couplings active
  (54 rows, starting angles) give λ_min 0.006829. The exactly
  constrained design reaches 99.1% of that with 6 fewer couplings and
  12 fewer rows.
- Couplings per crease: P0-P1 2, P0-P4 2, P0-P5 3 (forced leaf), P1-P6 2,
  P1-P2 2, P1-P3 2, P2-P7 3, P3-P4 3, P6-P7 2.
- Figures (`gore_bnb_full_gore_allv_figures.py`, 2026-11-16):
  re-evaluating the best set on the laptop with the same seed reproduces
  λ_min 0.006767 to within 4.6e-6. Its own leaf bound is 0.006768, so
  the design is essentially proven optimal at its angles. Clarabel
  reported "solution may be inaccurate" on one SDP during that
  re-evaluation, which is relevant to the single-gore soundness flag
  above. Saved: `results/gore_bnb_full_gore_allv_{3d_before_after,hinges,incumbent}.png`
  and the angles in `results/gore_bnb_full_gore_allv_design.json`.

**Full gore, mixed types, exactly 42 rows** (27 sites × 3 types = 81
candidates; 24 h cap):

- **A valid exactly constrained mixed design exists**: the first full-rank
  design appeared at 178 s.
- **Incomplete at the cap** (87,599 s), no certificate. Best found
  **0.006644** `[-VT F-T VFT -VT VVF FVV FFT FFV --T]` (22 couplings),
  reached at 10.3 h. Nothing better turned up in the remaining 14 h.
- **That is 98.2% of the all-V result**, but it says nothing about mixed
  vs. all-V. Every 21-V design is also a valid mixed design, so the
  mixed optimum is at least 0.006767. Without an incumbent or symmetry,
  this search simply never reached that region.
- Tree: 81,414 nodes; pruned by bound / rank / cut / cardinality
  15,461 / 4,901 / 17,032 / 422; leaves 2,875: 282 pruned by bound,
  2,233 infeasible, 360 evaluated (61 screened). Bounds are looser than
  all-V but still far tighter than birds-foot mixed: median
  bound/achieved 1.13, 90th percentile 1.64, max 3.28.

**Implications**:

- **Day 6 answered: yes, a valid full-gore baseline exists for Day 10**:
  all-V 0.006767, certified within 1.7%. This replaces §7.4's
  over-constrained design.
- **On gores, B&B is limited by tree size and per-node cost, not by loose
  bounds.** That changes the outlook for the flasher. The certificate
  should stay tight, and the work is in keeping the tree small (cyclic
  symmetry, a strong incumbent).
- **Next run: mixed full gore seeded with the all-V incumbent** (and
  mirror symmetry if the gore has one). Whether mixed types beat all-V on
  the gore is still open (equal row weights, §8.4 Phase 1).

Results: `results/gore_bnb_{single_gore,full_gore_allv,full_gore_mixed}_{results.json,summary.txt}` and `.log`.

### 7.18 Soundness audit and two-swap escape (2026-11-17)

**Questions** (Day 3 of §8.7): is the single-gore soundness flag (§7.17)
solver tolerance or a real error? Does a two-swap escape get swap search
out of the birds-foot traps (§7.9), and what does it cost at 24 sites?

**Soundness audit** (`single_gore_soundness.py`, lab machine; bound sweep 218 s).
Bound-only: both bounds recomputed for all 216 feasible single-gore sets
with the sweep's seed (same references), compared with the saved λ_min.
Every violating set, plus the 3 tightest, was re-solved with tight
Clarabel, tight SCS and MOSEK, and its λ_min was recomputed at the
achieved angles three independent ways.

| check | result |
|---|---|
| t (`sdp_certified_bound`, default Clarabel) | **18 of 216 below achieved**; worst −1.68e-6 at `[0,1,2,3,4,5,6,7,8,9,10,11,15,16,17]`: **the §7.17 flag, reproduced** |
| b_node (`bound_node`, what B&B prunes with) | **22 of 216 below achieved**; worst −2.01e-6 at `[0,…,14]` |
| default solver status | "optimal_inaccurate" on **212 of 216** sets |
| MOSEK on the 23 flagged sets | status "optimal" on all 23; bound **above** λ_min by +1.8e-7 to +1.2e-6 |
| Clarabel, tolerances at 1e-12 | still "optimal_inaccurate", same values: tighter settings don't help |
| SCS, tolerances at 1e-10 | doesn't converge (values 0.06–2.0, meaningless) |
| Riemannian re-run (same seed) | reproduces every saved λ_min exactly |
| λ_min at the achieved angles: Riemannian convention / SDP reduced model / CᵀC from geometry | agree to 9 digits on every set |

- **Verdict: solver tolerance, not a modelling error.** The bound and the
  evaluator measure the same matrix. Default Clarabel stops short on the
  30×30 gore SDPs and underestimates by up to 2.0e-6, i.e. **1.5e-4 of the
  optimum**. Earlier systems stayed under about 1e-9 (§7.1, §7.6).
- **B&B is affected the same way.** A leaf within about 1.5e-4 of the
  incumbent could be pruned wrongly, and a certified upper bound can be
  low by about as much. That's negligible next to the gaps reported here
  (0.15% to 33%), and the single-gore B&B still matched ground truth, but
  it should be fixed before the flasher: MOSEK for leaf bounds, or a
  safety margin of about 3e-4 relative (§8.5).
- The log's last line reads "REAL VIOLATION": a bug in the first version
  of the verdict, which counted "optimal_inaccurate" re-solves as exact.
  Fixed and recomputed from the saved results (`--reverdict`); the
  summary and JSON have the correct verdict.

**Two-swap escape** (`swap_search(two_swap=True)`). When no 1-swap
improves, try every 2-swap (remove 2 non-fixed couplings, add 2) with the
same bound screening and best-bound-first evaluation; accept the best and
return to 1-swaps. With `two_swap=False` the search is unchanged.

- **Birds-foot 12 sites, all 220 starts** (`birdsfoot_swap_validation.py
  --two-swap`, 5,886 s with a shared cache vs. 5,786 s for one-swap only):

  | | one-swap only (§7.9) | with two-swap escape |
  |---|---|---|
  | starts reaching the champion 0.009263 | 105 / 220 | **220 / 220** |
  | escapes needed | n/a | exactly 1 in each of the 115 previously stuck starts |
  | swaps accepted per run | median 2, max 3 | median 2, max 4 |
  | distinct sets evaluated per run (cold cache) | median 11, max 19 | median 19, max 25 |
  | distinct sets bounded per run | median 59, max 74 | median 185, max 198 (of 220) |

  Runs end at the champion (43) or its mirror twin `[0,1,2,5,6,7,8,9,11]`
  (177), both 0.009263. From the main trap `[0,1,2,4,5,7,8,10,11]`
  (96.8%), one 2-swap `[4,10] → [6,9]` reaches the champion: 108 moves,
  98 pruned by bound, 7 evaluated, 125 s.
- **Birds-foot 24 sites from the old champion** (`birdsfoot24_scale_test.py
  swap-champion --two-swap`; the old champion is a one-swap local
  optimum, §7.14):

  | pass | accepted 2-swap | λ_min |
  |---|---|---|
  | 1 | `[0,20] → [18,22]` | 0.009263 → 0.009392 |
  | 2 | `[8,13] → [0,10]` | → 0.009623 |
  | 3 | `[12,14] → [16,17]` | → 0.009626 |
  | 4 | none: 2-swap local optimum | 0.009626 |

  Final `[0,1,2,6,10,16,17,18,22]`, **0.009626 = 99.96% of B&B's best
  0.009630** `[0,1,2,7,9,13,14,19,21]`, and not an image of it under any
  of the three verified symmetries. No 1-swap ever improved; every gain
  came from an escape. Each 2-swap search: 3,780 moves, 1,165–1,600
  infeasible, 2,109–2,578 pruned by bound, 37–71 evaluated. Total 13,924
  bound solves, 190 Riemannian evaluations, 10,970 s (3.0 h), so bound
  solves dominate the cost at this size.

**Implications**:

- **The escape removes every birds-foot trap**, at the cost of about 15
  extra evaluations per stuck start. Swap search with the escape is a
  reliable incumbent supplier for B&B.
- **It's still a heuristic.** At 24 sites it stops at a 2-swap local
  optimum 0.04% below the best known design. B&B (with the one-restart
  screen, §7.16) stays the selection method; the escape is the fallback
  if the flasher tree is too large, and a strong incumbent either way.
- **Cost grows with C(k, 2) · C(m − k, 2)**: 108 moves at 12 sites, 3,780
  at 24. At flasher size, cyclic symmetry (choosing per gore) is what
  keeps this affordable.

Results: `results/single_gore_soundness_{summary.txt,results.json}`,
`.log` and `.err` (stderr: the Clarabel "may be inaccurate" warnings); `results/birdsfoot_swap_validation_two_swap_{results.json,summary.txt,starts.png}`
and `.log`; `results/birdsfoot24_swap_champion_two_swap_{results.json,summary.txt}`
and `.log` (the `.log` files from the two-swap runs are UTF-16).

### 7.19 Full flasher B&B: first overnight run and a MOSEK stall (2026-11-19)

**Question** (Day 10 of §8.7): can B&B select an exactly constrained,
cyclically symmetric all-V design for the whole flasher
(`full_flasher_system.py`: hub + 5 gores, 26 bodies, target rank 150; 24
orbit candidates with tied angles, choose 15, 1.3M sets)?

**Setup.** Clarabel fails on these 150×150 SDPs and its SCS fallback is
slow and badly wrong, so every SDP uses MOSEK (`flasher_timing.py`,
2026-11-18: root bound 3.0 s, leaf bound ~1.4 s, one evaluation ~140 s).
`flasher_bnb.py allv --threads 7 --swap-passes 2 --max-hours 14`: random
full-rank start, one-swap search for the incumbent, then B&B (screen f =
0.8, bound margin 3e-4).

**Incumbent (finished, 2.2 h).**

| stage | λ_min |
|---|---|
| random full-rank start `[0,1,4,5,6,8,9,10,11,12,13,16,18,20,22]` | 9.45e-5 |
| pass 1: swap 0 → 23 | 5.56e-4 |
| pass 2: swap 16 → 15 | **6.56e-4** |

Final set `[1,4,5,6,8,9,10,11,12,13,15,18,20,22,23]` (per crease
P0-P1:1 P0-P4:2 P0-H:2 P1-P4':3 P1-P2:2 P1-P3:1 P2-P3':2 P3-P4:2), 6.9×
the start. Each pass: 135 swaps, ~22 infeasible, ~85 pruned by bound,
~28 evaluated; 248 bound solves and 57 evaluations in all. Pass 2 still
improved, so this isn't a one-swap local optimum yet.

**B&B (crashed before its first progress line).** Diagnosis by replaying
the tree from the incumbent (scratch scripts, not committed):

- The root solves cleanly: t = 1.358e-3, "optimal", 2.7 s.
- The 4th node (`fa=[13]`, `fi=[10,22]`) and 13 of the next 46 node SDPs
  ended with MOSEK status **UNKNOWN**: it converges to ~1.35e-3, then
  stalls a few iterations short of its default 1e-8 tolerances (primal
  −1.3511e-3, dual −1.3161e-3, identical last three iterations). cvxpy
  raises `SolverError` on that, and the node loop only caught
  `RuntimeError`/`ValueError`, so the run died. The swap search had the
  same gap but never hit a failure (its SDPs are fully decided sets).
- Likely cause: poor scaling. K entries reach 2.3e4 while t ≈ 1e-3.
  Multiplying K by 100 or 1000 doesn't help.
- Relaxed tolerances do: at relative gap 1e-6 (feasibility 1e-7) all 6
  failing nodes tested return "optimal", within ~0.3% of the stalled
  values (and above them on 5 of 6). `accept_unknown=True` also returns a
  value, but it's the stalled primal and isn't guaranteed to be an upper
  bound.
- Side effect if failures are only caught: a failed node gets z = 0.5
  everywhere, so B&B branches on orbits 0, 1, 2, … in index order and
  loses the relaxation's guidance.

**Fix** (`sdp_theta_optimizer._solve`, `branch_and_bound.py`): on a
MOSEK `SolverError`, retry once at relaxed tolerances and report
"optimal_inaccurate"; B&B multiplies such node bounds by an extra 1.01
(~3× the scatter seen) and counts them as `sdp_relaxed`. If the retry
fails too, `_solve` raises `RuntimeError`: B&B branches without pruning,
swap search records the set and moves on. Re-running the 80-node replay:
**0 failures**, 18 node bounds from the retry, 3 nodes and 6 leaves
pruned by bound, 19 leaves evaluated (all screened), no new incumbent.

**Implications**:

- **The 14 h run won't finish.** About 13 s per node (node bound plus
  leaf work) gives roughly 4,000 nodes; node bounds at depth ~20 are
  still 8–9e-4 against the 6.56e-4 incumbent, so pruning is rare.
  Expect an incomplete run that may raise the incumbent but won't
  certify it.
- **The end-of-run screen recheck is expensive here.** At the 80-node
  cap it re-evaluated all 19 screened leaves in full: ~50 min, no
  improvement. Cap `screen_recheck` before a long run.
- **Before rerunning**: pass the known incumbent instead of repeating the
  2.2 h swap search. **Done 2026-11-23**: `flasher_bnb.py` now takes
  `--incumbent 1,4,5,...` (evaluated once, ~160 s, and reused by B&B) and
  `--screen-recheck R` (default now 5). A 0.03 h test reproduced
  λ_min 6.560721e-4 exactly; 14 of its 22 node bounds needed the relaxed
  retry, with no SDP errors.

Results: the crashed run's `results/flasher_bnb_allv.log` (incumbent
search only) was overwritten by the §7.20 rerun; the swap-search numbers
above are its record. Timing check:
`results/flasher_timing_{summary.txt,results.json}`.

### 7.20 Full flasher B&B: overnight rerun, all-V and mixed (2026-11-24)

**Question**: with the MOSEK stall fixed (§7.19) and the swap-search
incumbent passed in, how far do all-V and mixed-type B&B get on the
whole flasher in one long run?

**Setup** (lab machine, `flasher_bnb.py {allv,mixed} --incumbent
1,4,5,6,8,9,10,11,12,13,15,18,20,22,23 --max-hours 19`): MOSEK, 7
threads, screen f = 0.8, screen recheck 5, bound margin 3e-4. The given
incumbent re-evaluated to 6.560721e-4 (292 s), matching §7.19. Mixed
mode lifts it to V-grooves at the same 15 orbit sites (72 typed
candidates, 24 sites × 3 types, exact row budget) and re-evaluates it at
6.637441e-4 there: same set, but a different per-set RNG key (typed
indices), hence different restarts.

**Results** (both stopped at the 19 h cap, incomplete, no certificate):

| | all-V | mixed |
|---|---|---|
| start → best found | 6.561e-4 → **7.904e-4 (+20.5%)** | 6.637e-4 → 6.701e-4 (+1.0%) |
| incumbent updates | 3 (1.1 h, 2.1 h, 9.0 h) | 1 (4.8 h) |
| nodes (s/node) | 2,964 (23.8) | 9,048 (7.8) |
| depth on the stack | ~20–23 | ~60–72 |
| leaves: evaluated / pruned by bound / infeasible | 677 / 373 / 154 | 279 / 893 / 667 |
| nodes pruned: bound / rank / cardinality | 219 / 52 / 0 | 1,412 / 515 / 737 |
| node bounds from the relaxed retry | 456 | 3,331 |
| SDP errors (both solves failed) | 0 | 46 nodes |

- **All-V best**: `[0,3,5,6,7,8,10,11,12,14,17,19,21,22,23]`, per crease
  P0-P1:1 P0-P4:2 P0-H:3 P1-P4':2 P1-P2:2 P1-P3:1 P2-P3':1 P3-P4:3. From
  the swap incumbent it moves one coupling P2-P3' → P0-H and one P1-P4'
  → P3-P4. 12.0× the random start of §7.19. Nothing new in the last
  10.5 h.
- **Mixed best**: `F-T VF- V-V -VV T-- T-F VT- V-F` (30 rows per gore),
  found at node 3162; nothing new in the last 14.6 h.
- **Mixed vs. all-V on the flasher is still open.** Every all-V design
  is also a mixed design, so the all-V 7.904e-4 set (lifted) beats the
  mixed best inside the mixed space; the mixed run just never reached
  it. The mixed tree is ~3× deeper, so most of its 19 h went to node
  bounds, not leaves.
- **No certificate, as §7.19 predicted.** Bounds stay loose: the leaf
  that gave 7.904e-4 had bound 1.10e-3 (39% above). Of 677 all-V leaf
  evaluations, 667 were screened; median achieved 1.2e-4.
- **The relaxed-tolerance retry held up.** No all-V SDP errors in 19 h;
  in mixed mode both solves failed on 46 node SDPs, which B&B branched
  without pruning (sound, but they cost time).
- **Restart 1 is a weaker proxy here than on birds-foot.** In the
  recheck, `[0,4,5,6,7,8,10,11,12,13,14,17,21,22,23]` was stopped at
  5.15e-4 after restart 1, but its best of 5 is 6.704e-4 (+30%;
  birds-foot's largest recheck gap was 1.4%, §7.16). No loss this time:
  it was evaluated (leaf 83) after 6.987e-4 was found (leaf 65), so it
  could not have raised the incumbent. Two other rechecks rose 9% and
  7%. More reason to rank the recheck by leaf bound (§7.16) or rerun
  the top leaves unscreened before reporting a design of record.
- **One mixed recheck returned 0.0 after a feasible first restart**
  (4.697e-4): see the check below.

**Check: the 0.0 recheck is a false multiplicity alarm from cyclic
symmetry** (2026-11-25, this laptop, no MOSEK; scratch scripts, not
committed). The leaf is `[0,8,10,12,15,19,25,27,32,33,37,47,51,55,58,64,69]`
= `F-T VFF V-V FTF V-- T-F VV- V-F` (150 rows, bound 7.72e-4).

- **Reproduction.** Rebuilding its reference and restart RNG exactly as
  `SetEvaluator` does, the screened call needs 6 attempts for its first
  restart (5 `MultiplicityError`s) and returns 4.696579e-4, matching the
  run. The full call needs **28 of its 30 allowed attempts** (23
  multiplicity failures) to collect 5 restarts: 4.697e-4, 1.2e-6,
  8.3e-5, 1.1e-4, 1.8e-4. On the lab machine it evidently ran out of
  attempts; `riemannian_multistart` then raises `MultiplicityError`,
  `SetEvaluator.evaluate` records the set as infeasible at 0.0, and the
  recheck overwrites the leaf's known 4.697e-4. (The error isn't saved in
  the JSON, so the exception type on the lab machine is inferred; the
  other path to 0.0, a Clarabel failure in the min-norm QP, raises
  `RuntimeError`, but nothing points to it.)
- **Cause: exactly doubled λ_min, not a degeneracy.** Designs are
  cyclic with tied angles, so K(θ) commutes with the 5-fold rotation and
  every eigenvalue outside the rotation-invariant modes comes in an exact
  pair. Using the optimizer's own K(θ) at random starts, λ_min equals
  its upper neighbor to 1e-8 at 37 of 40 starts for this leaf, 4 of 40
  for the mixed best, and **24 of 40 for the all-V best**. At such
  starts the pair gap is 1e-13 to 1e-12 (roundoff), stays at roundoff
  along random tied-angle steps up to 0.3 rad, and the directional
  derivative is identical from either eigenvector or any rotation of
  them (cross term ~1e-14). So λ_min is smooth there and the envelope
  gradient is valid; `eval_lambda_min_and_grad`'s simplicity check
  (relative 1e-8) is a false alarm for symmetry-forced pairs.
- **Consequences.**
  - 5 leaves were scored 0.0 this way: the 3 mixed ones (bounds
    6.81e-4, 6.82e-4, 7.72e-4) and **3 all-V leaves with bounds 7.10e-4,
    9.48e-4 and 1.04e-3**. Two of the all-V ones could still beat
    7.904e-4. Not re-evaluated yet.
  - Starts and steps where a doubled mode is the minimum are thrown away
    or resampled, so the Riemannian search keeps to the θ region where a
    rotation-invariant mode is lowest. If the max-min optimum lies where
    a doubled mode is lowest (a 24-of-40 start rate for the all-V best
    suggests that region is large), **flasher λ_min values could be
    underestimates.** Untested.
  - Any set whose λ_min is doubled over most of θ-space can come out
    as 0.0 (infeasible) instead of its real value.
- **Fix options** (not implemented): treat a pair as simple when it is
  symmetry-forced (gap at roundoff and the gradient agrees across the
  eigenspace, as checked above); or solve per irrep with the
  cyclic-symmetry (Bloch-wave) block reduction of §8.5, which removes the
  pairs entirely; and in any case make `SetEvaluator` keep a recheck's
  first-restart value instead of replacing it with 0.0. Then re-evaluate
  the 5 zero leaves and the two best designs.

Results: `results/flasher_bnb_{allv,mixed}_{results.json,summary.txt}`
and `.log`.

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
  carry over unchanged, and the angle stage runs only over V-grooves (in
  effect: flats and sockets keep a θ variable with zero gradient).
  Implemented and verified (§7.13).
- **Arms already work kinematically.** `get_interpolation_matrix` uses
  r = p − centroid, so a contact on a rigid arm is just a point far from
  the centroid. The old +Z hard-coding in `_compute_groove_normals` is
  gone: each coupling now takes an explicit `ref_dir`, and a face
  parallel to the panel works when one is given (§7.13).

### 8.4 Plan, in order

**Phase 0 — swap search after greedy.** *Built and validated
(2026-10-20, §7.8–§7.9); not sufficient on its own.*
- Done: one-swap, best-improvement search with rigorous bound screening
  (`swap_search.py`). Two-panel: 20/20 starts reach the optimum.
  Birds-foot: 105/220; the rest stop at one-swap local optima.
- Done 2026-11-17 (§7.18): *two-swap escape when stuck*
  (`two_swap=True`). Birds-foot: 220/220 starts reach the champion. 24
  sites from the old champion: 0.009626, 99.96% of the best known.
- Still open:
  - *Multiple random starts*: each start reaches the champion with
    probability ~0.48, so ~5 starts would very likely find it. Cheap,
    but only a statistical guarantee.
- Birds-foot 24 sites (2026-11-12, §7.14): from greedy's set, 6 swaps to
  a new best 0.009630 (104% of the old champion); the old champion is a
  one-swap local optimum.
- Role going forward: supplies B&B's starting incumbent. Every birds-foot
  run ended at ≥ 90.6% of the champion, which is what B&B pruning needs.
- For mixed coupling types, a swap must keep the row count exact.

**Phase 1 — generalize the coupling model (goal B + groundwork for C).**
*Built and validated (2026-10-26, §7.13): all three types, explicit
`ref_dir`, unit tests passing. Row weights decided 2026-11-18 (below).
Still open: a mixed-type incumbent (row-aware greedy; swaps that trade
types at equal row count, e.g. V+V ↔ flat+socket).*
- A contact = a set of unit normals at a point, built in a local frame
  (face normal + explicit in-plane reference vector), with a type:
  flat (1 normal), V-groove (2, rotated by θ), tetrahedral (3, evenly
  spaced, fixed tilt).
- Replace the +Z assumption with an explicit reference vector so faces
  can have any orientation.
- **Decided 2026-11-18: equal row weights (K = CᵀC) for the foreseeable
  future.** All type comparisons are reported under that assumption; the
  W plumbing below is deferred. Original note:
- *Stiffness per row.* Today every row has unit weight
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
*Prototype built and validated (2026-10-20, §7.10): V-grooves only,
depth-first, most-fractional branching, cut counting, generic-rank
feasibility, manual symmetry breaking, greedy + swap incumbent. Finds the
optimum on two-panel and birds-foot; birds-foot certified gap 17.6%.
Mixed-type mode (exact row budget, ≤ 1 type per site) added 2026-10-26
and validated on two-panel against exhaustive ground truth (§7.13);
birds-foot mixed run done 2026-11-14: 0.010358, gap 33.0%, 9.6 h (§7.15).*

**Where this was left (2026-10-20): an open decision on the certified
gap.** Arc branching (§7.11) and moment lifting (§7.12) both failed to
tighten the ~15 loose birds-foot leaves. Options:

1. **Accept the gap and report it honestly** (recommended). B&B finds the
   optimum reliably; the 17.6% certified gap is confined to named leaves,
   and on each of them independent methods agree on the achieved value
   (exhaustive sweep, DE, swap search, B&B; 40 restarts on the worst one).
   Claim: "certified relative to leaf evaluation, with a quantified gap".
   Unblocks the steps below.
2. **One definitive order-2 answer with MOSEK** (free academic license):
   settles whether order 2 proves the worst leaf; likely too slow for
   routine use (§7.12).
3. **Keep researching tighter leaf bounds**: reduced/symmetric moment
   formulations, sparse hierarchies, interval methods. Open-ended.

**Decision (2026-11-13): option 1, accept the gap and report it.**
Option 2 was run first: MOSEK solves order 2 accurately in ~30 s, and
order 2 does *not* prove the worst leaf (bound 0.010890 → 0.010694, still
15.4% above the champion; §7.12). So no affordable lifting closes the gap.
Report B&B results as "certified relative to leaf evaluation, with a
quantified gap": 17.6% at 12 sites, 15.7% at 24 sites (§7.14). Option 3
stays open as future work, not on the critical path.

Then:
- Branch on mirror pairs first, so symmetry prunes higher in the tree.
  *Done 2026-11-12: birds-foot 213 → 97 branches (§7.10).*
- Run birds-foot 24 sites (§7.7): the go/no-go test for scale.
  *Done 2026-11-12 (§7.14): completes in 6.2 h with a swap-search
  incumbent, gap 15.7%; doesn't finish in 8 h without one.*
- Phase 1 (coupling types): since built (§7.13); remaining items are
  scheduled in §8.7.

Original Phase 2 design notes:
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
  6. Tighter relaxations — needed for the certified gap, not just speed;
     arc branching and moment lifting tried without success (§7.11–§7.12).
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
*Moved to Later (§8.6) on 2026-11-18, after the flasher: with equal row
weights, arm compliance can't be modelled (watch-outs below). Planned
for the week of 11/30.*
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
- **Leaf-bound solver accuracy** (§7.18): default Clarabel returns
  "optimal_inaccurate" on the gore SDPs and underestimates bounds by up
  to 1.5e-4 relative. Before the flasher runs, switch leaf bounds to
  MOSEK or add a safety margin of about 3e-4 relative. **Done for the
  flasher (2026-11-19)**: `flasher_bnb.py` uses MOSEK for every SDP plus
  the 3e-4 margin; MOSEK stalls are retried at relaxed tolerances with an
  extra 1% margin on node bounds (§7.19).
- **Flasher B&B throughput** (§7.19, §7.20): measured 23.8 s per node
  (all-V) and 7.8 s (mixed) over 19 h, with weak pruning; neither run
  finishes. First, fix the false multiplicity alarm on cyclic designs
  (§7.20) and re-evaluate the zero-scored leaves and both best designs,
  since flasher λ_min values may be underestimates. Next: swap search (one- and two-swap) from the all-V 7.904e-4
  design, and a mixed run seeded with that design lifted, so mixed has to
  beat the best known all-V design rather than the old one. Report the
  best design found, uncertified, with an unscreened re-evaluation.
- **Housekeeping**: independent child RNG per restart (§6.3; deferred
  from Day 3 of §8.7); the polish's trust-region-floor stopping on
  birds-foot (§7.5).

### 8.6 Later

- **Arms (Phase 3 of §8.4; former Days 7–8 of §8.7).** Moved here
  2026-11-18; planned for the week of 11/30. Candidate generation (fixed
  arm lengths, contacts on the neighbor, any face orientation via
  `ref_dir`); then two-panel and birds-foot with and without arms, and
  the `length_scale` sweep. Decide first how to handle arm compliance
  under equal row weights: rigid arms favour the longest arm (§8.4
  Phase 3), so either report arms as an upper bound or make arm
  compliance the one exception to W = I. Also: arm sites in the flasher
  system, and `draw_hinge_layout` generalized for arms.
- **Continuous arm length / contact position.** Note that K is quadratic
  in contact position (C is linear in r), so the affine-in-θ structure
  and the per-coupling SDP registry don't cover it. It would need
  gradient-based optimization over positions, separate from the current
  machinery.

### 8.7 Schedule to Thanksgiving (drafted 2026-11-09)

Twelve business days, Mon 11/9 – Tue 11/24; Wed 11/25 is buffer. Each day
ends with a check that says it's done. Status (updated 2026-11-19):
✅ done · ⏳ running · 🟡 partial · ⬜ not started · ➡️ moved.

| Day | Date | Task | Done when | Status |
|---|---|---|---|---|
| 1 | Mon 11/9 | Bring this doc up to date (§7.13). Close the gap decision (§8.4): one order-2 MOSEK solve on the worst leaf, stopped after 3 h, then adopt option 1 (accept and report the gap). Branch on mirror pairs first. | Decision recorded in §7.12/§8.4; birds-foot B&B still finds 0.009263 with fewer than 213 branches. | ✅ Done 11/13. MOSEK: order 2 doesn't prove the leaf (bound 0.010694); option 1 adopted (§7.12, §8.4). Mirror pairs first: 213 → 97 branches (§7.10). |
| 2 | Tue 11/10 | Scale test: birds-foot 24 sites (§7.7). Swap search from greedy's set and from the old champion for the incumbent; B&B in the background (`python -u`, 3 workers). | Go/no-go verdict on B&B at scale: nodes, evaluations, gap, time. | ✅ Done 11/13. Go at 24 sites with a swap incumbent (6.2 h, gap 15.7%, new best 0.009630); no-go as-is for the flasher (§7.14). |
| 3 | Wed 11/11 | Two-swap escape in `swap_search.py` (fallback if B&B didn't scale). B&B on the single gore (6 forced + 9 of 12 = 220 sets) vs. an exhaustive sweep. Child RNG per restart; `requirements.txt`. | Single-gore B&B matches ground truth; selection method for the rest of the plan chosen. | ✅ Done 11/17. `requirements.txt` done 11/13. Single-gore B&B done 11/15: matches the exhaustive optimum 0.013044 with 6 evaluations vs. 216, gap 0.15% (§7.17). Soundness flag resolved 11/17: solver tolerance (default Clarabel low by up to 2.0e-6; MOSEK clears every set), §7.18. Two-swap escape done 11/17: birds-foot 220/220 starts reach the champion; 24 sites 0.009626 from the old champion (§7.18). Selection method: B&B with the one-restart screen (§7.16), two-swap escape as fallback and incumbent supplier. Child RNG deferred to §8.5 housekeeping. |
| 4 | Thu 11/12 | Run `mixed_types_validation.py birds-foot`. Does a mixed design beat 0.009263? | Results saved and logged as §7.15. | ✅ Done 11/14. Yes: 0.010358 `[FFT V-V V-T V-V]`, 111.8% of the all-V champion (unit weights), gap 33.0%, 9.6 h, 1,977 leaf evaluations (§7.15). |
| 5 | Fri 11/13 | Decide row weights: K = CᵀWC, plumbed through with W = I default. Mixed-type incumbents: row-aware greedy, and swaps that trade types at equal row count. | Weight convention documented; greedy + swap produce a valid mixed design on birds-foot. | 🟡 Weights decided 11/18: equal row weights (W = I) for the foreseeable future; no plumbing needed. Mixed-type incumbents not started. |
| 6 | Mon 11/16 | Full gore under exact constraint (§8.2): is there a 42-row design, all V-grooves or mixed? | Answer logged; a valid full-gore baseline exists for Day 10. | ✅ Done 11/16 (ran over the weekend). Yes for both. All-V: 21 couplings, best 0.006767, certified within 1.74%, 9.0 h, 130.5% of §7.4's over-constrained design. Mixed: valid designs exist, but the 24 h run without an incumbent stopped incomplete at 0.006644 (§7.17). Follow-up: mixed run seeded with the all-V incumbent. |
| 7 | Tue 11/17 | Arms: candidate generation (fixed arm lengths, contacts on the neighbor, any face orientation via `ref_dir`); arm compliance in the row weights, w = 1/(1/k_contact + 1/k_arm). | Arm contact on two-panel matches a hand calculation; rank and gauge checks pass; arm sites are ordinary candidates. | ➡️ Moved to §8.6 (11/18). |
| 8 | Wed 11/18 | Arm experiments on two-panel and birds-foot, with and without arms; sweep `length_scale` and k_arm. | When arms win, and by how much, logged. | ➡️ Moved to §8.6 (11/18). |
| 9 | Thu 11/19 | Build the flasher test system from the DXF: a repeated subset (2–3 assembled gores) with all candidate types (no arm sites; §8.6). | System loads; panel and crease counts checked; rotational symmetry permutations verified. | ✅ Done 11/18, as the whole flasher (hub + 5 gores, `full_flasher_system.py`) rather than 2–3 gores; timing check: MOSEK required (§7.19). |
| 10 | Fri 11/20 | Flasher selection with cyclically symmetric designs enforced (the same choice in every gore): greedy + swap incumbent, then B&B or the Day 3 fallback. | Exactly constrained design with certified bound and gap, or a record of where it stopped scaling. | 🟡 All-V incumbent 6.56e-4 from swap search (2 passes, 2.2 h). First B&B run crashed on a MOSEK stall; fixed 11/19. Rerun 11/23–24 (19 h cap, incomplete): all-V best **7.904e-4** (+20.5%), mixed best 6.701e-4; no certificate (§7.20). |
| 11 | Mon 11/23 | Scale the flasher up: more gores, or the cyclic-symmetry (Bloch-wave) reduction of §8.5 if time allows. Final evaluation (5 restarts, optional polish). | Best flasher design evaluated; trend with gore count recorded. | ⬜ Not started |
| 12 | Tue 11/24 | Figures and write-up: Status, §4, §7, roadmap. | Results committed; this doc reflects what's done and what's open. | ⬜ Not started |

**Replan 2026-11-18** (equal row weights; arms moved to §8.6):

| Date | Work |
|---|---|
| Wed 11/18 | Day 5 remainder: mixed-type incumbents (row-aware greedy, type-trading swaps). Leaf-bound safety margin or MOSEK (§8.5). Lab machine overnight: mixed full gore seeded with the all-V incumbent (§7.17). |
| Thu 11/19 | Day 9: flasher test system. |
| Fri 11/20 | Day 10: flasher selection (lab machine over the weekend if long). |
| Mon 11/23 | Day 11: scale up the flasher, final evaluation. |
| Tue 11/24 | Day 12: figures and write-up. |
| Wed 11/25 | Buffer. |

**If behind, cut in this order**: flasher size (Day 11: stay at 2
gores); mixed-type incumbents (use all-V incumbents for mixed B&B).
(Original list: MOSEK (Day 1); the `length_scale` sweep (Day 8);
generalizing `draw_hinge_layout`; flasher size. The first three are
done or moved to §8.6.)

**Biggest risk**: Day 2. If B&B can't handle 24 sites it won't handle the
flasher. That is why Day 3 builds the two-swap fallback and Day 10
enforces cyclic symmetry to keep the search small.
