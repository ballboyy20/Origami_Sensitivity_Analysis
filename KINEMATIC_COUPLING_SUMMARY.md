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

Last updated: 2026-09-14.

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

5. **(Done this session)** ~~Test forward greedy at larger scale, without
   exhaustive ground truth to check against~~ — done for both a single
   gore (m=18) and the corrected full gore with inter-gore couplings
   (m=27, k=23, gap 1.10%). **Real result found and real fix applied —
   see §5.** Still open: (a) whether to trust the certified-bound gap as
   the sole optimality evidence at this scale, and/or (b) layering local
   search (swap-based refinement, e.g. Fedorov-exchange-style) on top of
   the greedy result as a cheap correctness booster — not yet done.

6. **(Ambitious/optional)** Tighten the SDP bound with a higher moment-
   relaxation level (Lasserre-hierarchy-style) if bound looseness becomes
   a real blocker at larger scale — the forward-greedy bound-guidance
   above would only get MORE useful from a tighter bound, not less.

7. **(Still open — one gore-scale test done, decision not yet made)**
   Decide how to get a trustworthy λ_min for the actual assembled
   (rotationally-tiled) flasher pattern, not just a gore-scale test. One
   gore wired to real duplicate neighbor panels (now done for both single
   and inter-gore-coupled versions — see §5.5) is still only ONE
   arbitrary boundary condition on the periodic problem — it does not
   converge to the true full-assembly λ_min just by adding more neighbor
   detail, no matter how clean the m=27 result looked. Two options, not
   mutually exclusive:
   - **Evaluate the full assembly directly** (all N gores, real m) —
     conceptually the simplest, correct by construction, but scales
     linearly (or worse — greedy construction time already grew from
     46.7s at m=18 to 1068.6s at m=27, a >20x slowdown for a 1.5x
     candidate-count increase, driven partly by two extra k-bump rounds
     — see §5.5) in cost with N.
   - **Cyclic-symmetry / Bloch-wave reduction** — the established
     technique for exactly this situation in structural dynamics
     (rotationally periodic structures: bladed disks/rotors; Thomas 1979
     and the literature it spawned; standard cyclic-symmetry solvers in
     commercial FEA). Solve ONE sector's (gore's) eigenvalue problem
     repeatedly under a swept inter-sector phase condition (N conditions,
     one per "nodal diameter" 0..N-1); the true global λ_min is the
     minimum over that sweep. Correct without approximation error, and
     each sector solve stays cheap (m=18-ish), but requires generalizing
     this codebase's real-valued constraint-row machinery to complex
     (phase-shifted) boundary couplings — not yet attempted, and a
     nontrivial implementation step.
   Not yet decided which to pursue; see §5.4/§5.5 for the concrete
   trigger (an early single-gore design that turned out to silently omit
   inter-gore couplings entirely) and the gore-scale test's own greedy-
   runtime growth, which is itself evidence for taking the full-assembly
   option's cost seriously before committing to it over Bloch reduction.

**Recommendation if picking one**: forward bound-guided greedy just
produced the strongest practical result of this whole project (exact
champion, seconds not hours) — the highest-value next step is finishing
#7's decision (full assembly vs. Bloch reduction), since without it the
gore-scale numbers in §5 aren't yet known to represent the actual target
application's λ_min, only a lower or upper bound on it of unknown size.

---

## 5. Larger-Scale Forward-Greedy Test: Flasher Gore (this session)

Step 5 from the previous session's roadmap — testing forward greedy on a
topology bigger than birds-foot's m=12, with no exhaustive ground truth
available to check against.

### 5.1 Geometry source: DXF, not hand-coded

Unlike birds-foot (hand-coded vertices in `interactive_optimizer.py`),
this test's geometry came from a DXF crease-pattern drawing (a flat set of
LINE entities meeting at shared vertices — no layers, no pre-closed panel
polygons, no crease/boundary distinction). New reusable module:

- **`dxf_to_panels.py`** — `load_gore_geometry(dxf_path)` treats the
  DXF's line segments as a planar straight-line graph and recovers panels
  as its enclosed faces via `shapely.ops.polygonize`; an edge shared by
  exactly two resulting faces is an interior crease (a candidate coupling
  spoke), an edge bordering only one face is outer boundary. New
  dependencies added to the environment: `ezdxf`, `shapely` (no
  `requirements.txt` exists yet in this repo to record them in).
- Verified clean on `Flasher_gore_reference_cleaned.dxf`: all 17 raw
  lines resolved into exactly 6 panels and 6 interior creases, every
  input line accounted for in exactly one polygon edge (no gaps, no
  dangling segments) — not degenerate, despite the user's initial worry
  that it might be.
- **`flasher_gore_system.py`** — builds the `CouplingSystem` from that
  extracted geometry the same way `build_birdsfoot_system` does: 3
  KinematicCouplings per interior crease (20%/80% along the top face, 1
  at mid-thickness), giving **m=18** candidates (6 creases x 3) vs.
  birds-foot's m=12. Panel connectivity is a two-hub tree (panels 0 and 1
  each border 3 other panels), structurally different from birds-foot's
  single-hub star.

### 5.2 Real finding: leaf panels make some couplings structurally mandatory

Two panels in this gore (panel 2, panel 5) connect to the rest of the
assembly through exactly ONE crease each. A panel with only one crease
needs all 6 rows that crease can offer (3 couplings x 2 rows each) just
to pin its own 6 relative degrees of freedom — there is no other path
through the graph to constrain it, and exactly 6 rows are available, so
all 3 of that crease's couplings are mandatory in any full-rank topology,
not merely "good candidates."

The bound-guided forward greedy (`forward_greedy_topology.py`'s
`forward_greedy()`, reused unchanged) cannot see this: its per-step score
is a best-case-completion SDP bound that treats every candidate as
freely replaceable by a better one later, so it has no way to recognize
"this one is non-optional." **Confirmed empirically, not just in theory**:
unforced runs at both k=15 (target_rank/2 = 30/2, the exact structural
minimum, same convention that worked cleanly for birds-foot's k=9) and
k=16 left the panel-2 leaf crease (P1-P2) one coupling short of its
required 3, and `find_full_rank_reference` failed for both final active
sets — i.e. plain forward greedy, unmodified, produced an active set that
provably cannot reach full rank at ANY theta, a real failure mode birds-
foot's star topology never exposed.

**Fix applied**: `forward_greedy()` gained an optional
`forced_active_start` parameter (backward-compatible, default `None` —
birds-foot's existing calls are unaffected) to pre-seed structurally
mandatory couplings before the greedy loop starts.
`forward_greedy_gore.py` computes which creases are mandatory
automatically (panel degree == 1 in the crease graph), forces those 6
couplings (P0-P5, P1-P2) active, and lets greedy allocate the remaining
9 of the k=15 budget across the other 12 candidates (the P0-P1/P0-P4/
P1-P3/P3-P4 four-crease core loop — structurally its own rank-18 sub-
problem, the same size as birds-foot's whole problem).

### 5.3 Result (with the fix)

- Final active set: `[0,1,2,4,5,6,7,8,9,10,11,12,13,14,17]` — spokes
  `P0-P1:3, P0-P4:2, P0-P5:3, P1-P2:3, P1-P3:3, P3-P4:1`.
- Certified bound: t = 0.012966. Achieved λ_min (Riemannian, 5 restarts):
  **0.012351**. Certification gap: 6.15e-4 (**4.98%** of the achieved
  value) — looser than birds-foot's typical gaps (median 8.5e-6), a real
  sign the SDP bound loosens at this scale/topology and worth watching if
  this becomes the primary evaluation method rather than a one-off test.
- Greedy construction: 46.7s, 0 unresolved-reference events, once the
  mandatory couplings were forced.
- Raw results: `results/forward_greedy_gore_results.json`,
  `results/forward_greedy_gore_summary.txt`.

### 5.4 Caveat that reopened the scope of this test (not glossed over)

This whole test evaluates ONE gore (wedge) of the full rotationally-
symmetric flasher pattern, chosen originally to save computation while
(it was assumed) still giving a viable answer. Partway through, the user
identified that the initial single-gore DXF silently omitted the
couplings that exist between adjacent gores in the real assembly — i.e.
every result in §5.3 above is for an isolated wedge with free (unconnected)
radial boundaries, not a physically complete boundary condition. A
corrected DXF (`Full_Flasher_gore_reference.dxf`) has since been created
with full duplicate neighbor-gore panels so the inter-gore couplings can
be represented — **not yet evaluated as of this writing**.

Discussed and NOT resolved by just wiring in one neighbor on each side
with real (non-phase-shifted) couplings: that is only one arbitrary
boundary condition on what is properly a periodic problem, and doesn't
converge to the true assembled λ_min just by adding more neighbor detail.
The correct reduction (cyclic-symmetry / Bloch-wave analysis, sweeping
inter-sector phase — see roadmap item 7 above) or full-assembly
evaluation are the two options on the table; single-gore-plus-real-
neighbor-couplings is useful only as an algorithm-scale stress test, not
as evidence about the physical assembly's true λ_min.

### 5.5 Full-gore test (with inter-gore couplings) — run this session

`Full_Flasher_gore_reference.dxf` evaluated. Two loader fixes were needed
before it would parse, both generalizations of `dxf_to_panels.py` rather
than one-off workarounds:

- **Block INSERT support**: the corrected DXF adds the neighbor gore's
  boundary panels via a mirrored block reference (`INSERT` entity, 2
  `LINE`s inside `Block_0`), which `_read_segments` didn't previously
  walk into. Fixed via `ezdxf`'s `virtual_entities()` (recursing through
  nested blocks, transform-correct by construction) rather than
  special-casing this file's specific insert.
- **Vertex snapping**: the inserted copy's endpoints landed close to but
  not bit-identical with the original gore's vertices they're meant to
  coincide with — off by 0.0106 and 0.4547 units against a geometry scale
  where the next-nearest genuinely distinct vertex pair is ~80 units away
  (a 175x safety margin). Without merging these, `polygonize()` sees a
  gap, not a shared vertex, and silently fails to close the new face.
  Added `load_gore_geometry(..., snap_tol=...)` (opt-in, default `None` —
  the original single-gore file needed no snapping and still gets none)
  using simple O(n²) union-find clustering (`_snap_points`) — plenty fast
  at these point counts. `full_flasher_gore_system.py` uses
  `snap_tol=1.0`.

Result: **8 panels, 9 interior creases → m=27** candidates. Panel 2
(a leaf in the single-gore version) now has a second crease (to new panel
7) — exactly the connection that was previously missing. Panel 5 remains
the only leaf (still only reachable via P0-P5), so
`gore_system_common.mandatory_leaf_couplings()` (the single-gore fix's
leaf-detection logic, factored out for reuse — see below) forces only 3
couplings this time, not 6.

**Code factored out, not duplicated**: `gore_system_common.py` is new —
`flasher_gore_system.py` and `full_flasher_gore_system.py` both build on
its shared `spokes_from_creases`/`build_gore_system`/
`mandatory_leaf_couplings`/`default_start_thetas_deg` instead of each
reimplementing the recipe. Confirmed the refactor changed nothing:
`flasher_gore_system.py` reproduces its pre-refactor output exactly
(m=18, same spoke labels, λ_min=0.013203 at full-active) before the new
full-gore module was written against the shared code.

**Forward-greedy result** (`forward_greedy_full_gore.py`, k starting at
the structural minimum target_rank/2 = 21):

- k=21 and k=22 both failed to reach full rank at any theta (same
  phenomenon as §5.2/§5.3's single-gore case, not a new failure mode) —
  rather than restart from scratch, each retry appended exactly one more
  greedy step onto the existing active set at k+1 (no wasted computation,
  one continuous trace). Settled at **k=23 of 27**.
- **Certified bound t = 0.005242. Achieved λ_min (Riemannian, 5
  restarts) = 0.005185.** Certification gap 5.72e-5 (**1.10%** of
  achieved value) — tighter than the single-gore test's 4.98%, despite
  m growing from 18 to 27; bound looseness doesn't appear to scale
  simply with m alone.
- Final active set (23 of 27): spokes `P0-P1:3, P0-P4:1, P0-P5:3,
  P1-P6:3, P1-P2:2, P1-P3:2, P2-P7:3, P3-P4:3, P6-P7:3`.
- Greedy construction: 1068.6s (~18 min) — substantially slower than the
  single-gore case's 46.7s, both from the larger m and from needing two
  extra k-bump rounds; worth watching if this pipeline needs to run
  routinely rather than as a one-off test.
- Raw results: `results/forward_greedy_full_gore_results.json`,
  `results/forward_greedy_full_gore_summary.txt`.

**Visualizations generated** (new: `forward_greedy_full_gore.py` also
produces the same figure types birds-foot analyses used elsewhere in this
repo — `visualization_rigid.py`'s `draw_3d_config`/`draw_hinge_layout`,
unmodified):
- `results/forward_greedy_full_gore_3d_before_after.png` — 3-D config,
  all 27 candidates active vs. the greedy result at its achieved thetas,
  each with the weakest constrained eigenmode overlaid.
- `results/forward_greedy_full_gore_hinges.png` — one 2-D hinge-layout
  panel per crease (9 total), showing each crease's 3 candidate contact
  points (active: θ label + n1/n2 vectors; inactive: removed marker).
- `results/forward_greedy_full_gore_convergence.png` — certified bound
  chosen at each greedy step vs. active-coupling count, with the final
  achieved λ_min and certified bound marked for comparison.

**Status unchanged from §5.4's conclusion**: this is still only a
larger algorithm-scale stress test, not a trustworthy physical λ_min for
the assembled pattern — one neighbor gore's real duplicate panels are one
arbitrary boundary condition, not the periodic-problem-correct answer.
Roadmap item 7 (full-assembly evaluation vs. cyclic-symmetry/Bloch-wave
reduction) is still open and still the recommended next step for a
trustworthy physical number.
