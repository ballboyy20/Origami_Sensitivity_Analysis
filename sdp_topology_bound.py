"""
sdp_topology_bound.py

Convex (SDP) relaxation of a PARTIALLY-decided branch-and-bound node over
coupling topology: some candidate coupling sites forced active (z_i = 1),
some forced inactive (z_i = 0), some undecided (z_i relaxed to [0, 1]).
This is a DIFFERENT, UNVALIDATED-UNTIL-test_sdp_topology_bound.py-passes
relaxation from sdp_theta_optimizer.py's sdp_lambda_min(), which only
relaxes theta for an ALREADY-FIXED active-coupling set. Do not treat this
module's bound as trustworthy for real branch-and-bound use until that
test script passes with ZERO violations -- see its own docstring and the
results/sdp_topology_bound_summary.txt it produces.

Builds on two already-validated pieces, reused (imported) here, not
reimplemented:

  - calibrate(system, ref_thetas, active_idx, length_scale), from
    test_K_theta_structure.py: solves K(theta) = K0 + sum_i [cos(2*theta_i)
    *Kc_i + sin(2*theta_i)*Ks_i] exactly (to ~1e-15) for a FIXED active
    set. Does not touch .active flags itself -- only reads them, indirectly,
    via CouplingSystem.build_constraint_matrix.

  - _locked_subspace_projection / _symmetrize, from sdp_theta_optimizer.py:
    project K onto the target_rank-dim "locked" subspace (complement of
    the theta-independent whole-assembly rigid-body gauge null space) --
    see that module's own module-level comment for the full argument for
    why this projection, not raw K - t*I, is the correct SDP. The key fact
    this module leans on again: that gauge null space is the SAME fixed
    subspace for ANY topology (any subset of couplings, any theta) that
    itself reaches target_rank -- a structural fact about the panel
    assembly staying one connected rigid group, not specific to which
    couplings or angles are involved. So a projection P derived from one
    node's own full-rank reference is valid for that node's whole SDP.

Per-coupling decomposition (Part A -- see test_sdp_topology_bound.py's own
Part A check for the validation that this holds, to ~1e-9, before trusting
anything below): because C's rows are partitioned one-per-coupling with no
cross terms (already the substance of test_K_theta_structure.py's zero-
cross-term result), each coupling's OWN contribution is separately affine:

    K_i(theta_i) = K0_i + cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i

with K(theta) = sum over active i of K_i(theta_i) for ANY active set
containing i -- K0_i/Kc_i/Ks_i computed with coupling i ISOLATED (every
other coupling deactivated) are IDENTICAL to what a joint calibration
would produce for that same coupling. This lets a per-coupling registry
be computed ONCE (build_coupling_registry, below) for every candidate
coupling site, and reused verbatim across every node in a B&B tree --
never recalibrating from scratch per node.

The (z, angle) relaxation itself (Part B) is the standard "perspective
reformulation" for convexifying an on/off indicator coupled to a
continuous variable living in a convex set C containing the origin: the
convex hull of {(z, x): z in {0,1}, x in z*C} is exactly {(z, x): 0 <= z
<= 1, x in z*C}, and for C = the unit disk {(u,v): u^2+v^2<=1}, "x in z*C"
is precisely the 3-dimensional second-order (Lorentz) cone a^2+b^2 <= z^2
(with a=z*u, b=z*v) -- the tightest possible convex relaxation of
{(z,u,v): z in {0,1}, if z=1 then u^2+v^2=1, if z=0 then u=v=0} for z
relaxed to [0,1]. Forced-active couplings get the SAME disk relaxation as
the existing per-topology sdp_lambda_min (z=1 fixed, theta still free);
forced-inactive couplings contribute nothing at all (already confirmed
inert by test_K_theta_structure.py's inactive-coupling-invariance check).

Run directly: python sdp_topology_bound.py runs a couple of quick
standalone sanity prints (NOT the full validation battery -- see
test_sdp_topology_bound.py for that).
"""

import sys, os
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from test_K_theta_structure import calibrate
from sdp_theta_optimizer import _locked_subspace_projection, _symmetrize, _solve


# ══════════════════════════════════════════════════════════════════════
# Part A: per-coupling decomposition registry
# ══════════════════════════════════════════════════════════════════════

def calibrate_single_coupling(system, ref_thetas, i, length_scale):
    """
    K0_i, Kc_i, Ks_i for coupling i ALONE: calibrate(system, ref_thetas,
    active_idx=[i], length_scale) with every OTHER coupling temporarily
    deactivated. calibrate() itself never touches .active (only
    build_constraint_matrix reads it) -- this function does that part,
    saving and restoring every coupling's .active flag around the call so
    it has no lasting side effect on the caller's system state.

    ref_thetas must be a full-length array (one entry per system.coupling)
    -- calibrate()/get_K() zip over ALL of system.couplings regardless of
    which one is being isolated here; only ref_thetas[i] actually
    influences the result (every other coupling contributes zero rows
    while deactivated, so its own reference angle is irrelevant), but the
    array itself still needs to be full length.

    Returns (K0_i, Kc_i, Ks_i), each a (total_dofs, total_dofs) symmetric
    ndarray -- raw/full-dimension, NOT yet gauge-projected (that happens
    per node in sdp_topology_bound, since the projection depends on which
    OTHER couplings share the node).
    """
    saved_active = [getattr(c, 'active', True) for c in system.couplings]
    for j, c in enumerate(system.couplings):
        c.active = (j == i)
    try:
        K0, Kc, Ks = calibrate(system, np.asarray(ref_thetas, dtype=float).copy(),
                                [i], length_scale)
    finally:
        for c, act in zip(system.couplings, saved_active):
            c.active = act
    return _symmetrize(K0), _symmetrize(Kc[i]), _symmetrize(Ks[i])


def build_coupling_registry(system, ref_thetas, coupling_indices, length_scale):
    """
    {i: (K0_i, Kc_i, Ks_i)} for each i in coupling_indices, via
    calibrate_single_coupling(). Computed ONCE, independent of any
    particular node's forced_active/forced_inactive/undecided split (see
    module docstring's Part A argument), then reused verbatim across every
    node sdp_topology_bound() is asked to bound -- pass a precomputed
    registry in via that function's `registry` parameter to skip
    recomputing it per node.
    """
    ref_thetas = np.asarray(ref_thetas, dtype=float)
    return {i: calibrate_single_coupling(system, ref_thetas, i, length_scale)
            for i in coupling_indices}


# ══════════════════════════════════════════════════════════════════════
# Reference-theta search (orchestration-layer utility, not part of the
# core relaxation -- mirrors sdp_size9_sweep.py's _find_full_rank_
# reference, generalized from a fixed size-9 subset to an arbitrary
# index set, since a node's forced_active UNION undecided is frequently
# NOT full-rank at any single naive default reference (already observed
# repeatedly for birds-foot subsets in prior work on this repo). Kept
# here rather than imported from sdp_size9_sweep.py, which is a script
# meant to be run directly and is off-limits to modify.
# ══════════════════════════════════════════════════════════════════════

def find_full_rank_reference(system, active_indices, target_rank, length_scale=1.0,
                              base_thetas_deg=None, rng=None, max_tries=300):
    """
    Search for a reference theta vector (radians, one entry per
    system.coupling) for which build_constraint_matrix, restricted to
    `active_indices`, reaches target_rank. Tries the shared default first,
    then jittered variants, then fully random -- cheapest/most-plausible
    candidates first (same strategy as sdp_size9_sweep.py's
    _find_full_rank_reference).

    Returns the reference (radians ndarray) on success, or None if no
    candidate among max_tries reached target_rank -- callers should treat
    None as evidence (not proof) that this particular index set may be
    structurally rank-deficient for ANY theta, not merely that this
    search got unlucky (see sdp_topology_bound's own docstring on this
    point). This function does NOT raise -- sdp_topology_bound itself
    stays simple/deterministic (one attempt with the given or default
    ref_thetas, raises RuntimeError on a rank-deficient reference); this
    retry logic belongs in the calling/orchestration layer, exactly the
    pattern already established by sdp_size9_sweep.py on top of
    sdp_theta_optimizer.py's sdp_lambda_min.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(system.couplings)
    if base_thetas_deg is None:
        base_thetas_deg = np.degrees([c.theta for c in system.couplings])
    base_thetas_deg = np.asarray(base_thetas_deg, dtype=float)

    saved_active = [getattr(c, 'active', True) for c in system.couplings]
    saved_theta = [c.theta for c in system.couplings]
    active_set = set(int(i) for i in active_indices)
    for i, c in enumerate(system.couplings):
        c.active = (i in active_set)

    try:
        candidates = [np.radians(base_thetas_deg)]
        for _ in range(max_tries // 2):
            jitter = rng.uniform(-30, 30, size=n)
            candidates.append(np.radians(base_thetas_deg + jitter))
        for _ in range(max_tries - len(candidates)):
            candidates.append(rng.uniform(0, np.pi, size=n))

        for ref in candidates:
            for c, th in zip(system.couplings, ref):
                c.set_theta(float(th))
            C = system.build_constraint_matrix(length_scale=length_scale)
            if np.linalg.matrix_rank(C) >= target_rank:
                return ref
        return None
    finally:
        for c, act, th in zip(system.couplings, saved_active, saved_theta):
            c.active = act
            c.set_theta(th)


# ══════════════════════════════════════════════════════════════════════
# Part B: the combined (z, angle) topology-bound SDP
# ══════════════════════════════════════════════════════════════════════

def _row_need(rows, sites, forced_active, undecided, target_rank):
    """Mixed-type cardinality: (rows still needed, most rows the undecided
    candidates can still supply, one per free site)."""
    need = target_rank - int(sum(rows[i] for i in forced_active))
    taken = {int(sites[i]) for i in forced_active}
    best = {}
    for i in undecided:
        s = int(sites[i])
        if s not in taken:
            best[s] = max(best.get(s, 0), int(rows[i]))
    return need, sum(best.values())


def sdp_topology_bound(system, forced_active, forced_inactive, undecided, k,
                        length_scale=1.0, target_rank=None, ref_thetas=None,
                        solver=None, registry=None, arcs=None, rows=None,
                        sites=None, **solver_kwargs):
    """
    Upper-bounds lambda_min over EVERY completion of a partially-decided
    branch-and-bound node: forced_active couplings fixed on, forced_
    inactive fixed off, undecided couplings' on/off choice relaxed to
    z_i in [0, 1] (perspective relaxation of the on/off/angle set -- see
    module docstring), with the angle for both forced_active and
    undecided couplings relaxed to the unit disk exactly as in the
    existing per-topology sdp_lambda_min. Solves:

        maximize t
        s.t.  K_reduced(z, u, v, a, b) - t*I >> 0
              u_i^2 + v_i^2 <= 1                for i in forced_active
              a_i^2 + b_i^2 <= z_i^2, 0<=z_i<=1  for i in undecided
              sum(z_i for i in undecided) == k - len(forced_active)

    where K_reduced is built additively from a per-coupling registry (see
    build_coupling_registry) projected onto the target_rank-dim "locked"
    subspace for THIS node's reference topology (forced_active UNION
    undecided, its largest possible completion).

    UNLIKE sdp_lambda_min, this function does NOT decode or apply any
    result to the system -- there is no single real theta/topology behind
    a fractional z, so "decoding" a node bound isn't meaningful (that's
    the job of the B&B branching step, which is out of scope here). The
    system's .active flags and thetas are restored to whatever they were
    before this call on return (success or failure) -- this is purely a
    bound computation, not a "leaves the system at the result" call like
    sdp_lambda_min.

    Parameters
    ----------
    system : CouplingSystem
        Not left mutated -- see above.
    forced_active, forced_inactive, undecided : sequence of int
        Must be pairwise disjoint and together cover every index
        0..len(system.couplings)-1 exactly once (every candidate coupling
        site must be classified into one of the three buckets).
    k : int
        Required total active-coupling count (e.g. 9 for the birds-foot
        system's target_rank=18, ceil(18/2)=9).
    length_scale : float
        Same meaning as elsewhere in this codebase.
    target_rank : int or None
        Defaults to system.total_dofs - 6.
    ref_thetas : (len(system.couplings),) array or None
        Reference angles used both to build the per-coupling registry (if
        not supplied via `registry`) and to check this node's largest
        possible topology (forced_active UNION undecided) reaches
        target_rank. Defaults to each coupling's current theta. Many
        nodes are NOT full-rank at any single naive default -- see
        find_full_rank_reference() for a random-search utility to locate
        one; this function itself makes exactly ONE attempt and raises
        RuntimeError if it's rank-deficient, deliberately NOT catching or
        working around that (see module docstring: a node whose largest
        possible topology can't reach target_rank is likely infeasible/
        prunable in a real B&B and should be surfaced as such, not
        silently papered over -- though "no valid reference found after
        one try" is also consistent with simply an unlucky reference; use
        find_full_rank_reference() to distinguish the two before
        concluding a node is truly infeasible).
    solver : cvxpy solver constant or None
        Defaults to trying cp.CLARABEL, falling back to cp.SCS -- same as
        sdp_theta_optimizer.py's _solve().
    registry : dict or None
        Precomputed {i: (K0_i, Kc_i, Ks_i)} (see build_coupling_registry),
        covering at least every index in forced_active UNION undecided.
        If None, built internally via `ref_thetas` for exactly the
        indices this node needs. Passing a shared registry in (built ONCE
        over all m candidate sites) avoids recomputing it for every node
        in a B&B tree.
    arcs : dict or None
        Optional per-coupling angle restrictions, {i: (phi, w)}: coupling
        i's point x_i = (cos 2theta_i, sin 2theta_i) is restricted to the
        arc of the unit circle centred at angle phi with width w (both in
        2*theta space, radians, 0 < w <= 2*pi). Each adds the arc's
        convex-hull cut (Lu, Deng, Zhang & Fang, J. Glob. Optim. 2018,
        sec. 2): <x_i, (cos phi, sin phi)> >= cos(w/2), on top of the
        existing disk constraint -- in perspective form,
        a*cos phi + b*sin phi >= z*cos(w/2), for undecided couplings.
        Default None: no cuts, identical to the original relaxation.
    rows, sites : (len(system.couplings),) int arrays or None
        Mixed coupling types (coupling_types.TypedCandidates.rows /
        .site_of). When `rows` is given, the cardinality constraint
        becomes the exact row budget
            sum(rows_i * z_i for undecided) == target_rank - rows(forced_active)
        and `k` is ignored; with `sites`, at most one candidate per site
        may be on (sum of z over a site's undecided candidates <= 1, or
        == 0 if the site already has a forced-active candidate). `sites`
        defaults to one site per candidate. Default None for both: the
        original count constraint.
        Candidates whose K has no angle dependence (flats, sockets:
        Kc_i = Ks_i = 0 in the registry) contribute z_i * K0_i only.
    solver_kwargs : forwarded to Problem.solve().

    Returns
    -------
    dict with keys:
        't'              : float -- the relaxed upper bound on lambda_min
                            over every completion of this node.
        'solver_status'  : str -- cvxpy problem.status.
        'z'              : (len(undecided),) ndarray -- relaxed on/off
                            values, index-aligned to sorted(undecided).
        'u', 'v'         : (len(forced_active),) ndarrays -- relaxed
                            (cos 2*theta, sin 2*theta) for forced_active,
                            index-aligned to sorted(forced_active).
        'a', 'b'         : (len(undecided),) ndarrays -- z_i*(cos, sin)
                            for undecided, index-aligned to
                            sorted(undecided).
        'forced_active', 'forced_inactive', 'undecided' : list[int],
                            echoed back sorted.
    """
    m = len(system.couplings)
    forced_active   = sorted(int(i) for i in forced_active)
    forced_inactive = sorted(int(i) for i in forced_inactive)
    undecided       = sorted(int(i) for i in undecided)

    all_idx = set(forced_active) | set(forced_inactive) | set(undecided)
    if (len(forced_active) + len(forced_inactive) + len(undecided) != len(all_idx)
            or all_idx != set(range(m))):
        raise ValueError(
            "forced_active, forced_inactive, undecided must be pairwise "
            "disjoint and together cover every coupling index 0.."
            f"{m - 1} exactly once (got {len(forced_active)} + "
            f"{len(forced_inactive)} + {len(undecided)} covering "
            f"{sorted(all_idx)} of {m} total).")

    if target_rank is None:
        target_rank = system.total_dofs - 6

    n_fa = len(forced_active)
    if rows is None:
        n_free_needed = k - n_fa
        if n_free_needed < 0 or n_free_needed > len(undecided):
            raise ValueError(
                f"Node is infeasible by cardinality alone: need {n_free_needed} "
                f"of {len(undecided)} undecided couplings active to reach k={k} "
                f"given {n_fa} already forced active.")
    else:
        rows = np.asarray(rows, dtype=int)
        sites = np.arange(m) if sites is None else np.asarray(sites, dtype=int)
        n_free_needed, most = _row_need(rows, sites, forced_active, undecided,
                                        target_rank)
        if n_free_needed < 0 or n_free_needed > most:
            raise ValueError(
                f"Node is infeasible by row count alone: need {n_free_needed} "
                f"more rows, undecided candidates can supply at most {most}.")

    reference_indices = sorted(set(forced_active) | set(undecided))
    if not reference_indices:
        raise ValueError("Node has no forced_active or undecided couplings to bound.")

    if ref_thetas is None:
        ref_thetas = np.array([c.theta for c in system.couplings], dtype=float)
    else:
        ref_thetas = np.asarray(ref_thetas, dtype=float).copy()

    saved_active = [getattr(c, 'active', True) for c in system.couplings]
    saved_theta  = [c.theta for c in system.couplings]
    try:
        if registry is None:
            registry = build_coupling_registry(system, ref_thetas, reference_indices,
                                                length_scale)
        else:
            missing = [i for i in reference_indices if i not in registry]
            if missing:
                raise ValueError(f"registry is missing entries for coupling "
                                  f"indices {missing}")

        # ── Gauge projection, computed from THIS node's largest possible
        #    topology (forced_active UNION undecided) -- see module
        #    docstring and _locked_subspace_projection's own comment in
        #    sdp_theta_optimizer.py for why this must reach target_rank. ──
        for j, c in enumerate(system.couplings):
            c.active = (j in reference_indices)
        for j, c in enumerate(system.couplings):
            c.set_theta(float(ref_thetas[j]))
        C_ref = system.build_constraint_matrix(length_scale=length_scale)
        rank_ref = np.linalg.matrix_rank(C_ref)
        if rank_ref < target_rank:
            raise RuntimeError(
                f"Node's largest possible topology (forced_active UNION "
                f"undecided = {reference_indices}) is rank-deficient at the "
                f"given reference thetas (rank={rank_ref} < target_rank="
                f"{target_rank}). NOT caught or worked around here -- this "
                f"node may be structurally infeasible/prunable, or the "
                f"reference theta was simply unlucky; use "
                f"find_full_rank_reference() to tell the two apart before "
                f"concluding infeasibility.")
        K_ref = _symmetrize(C_ref.T @ C_ref)
        P, gauge_eigvals = _locked_subspace_projection(K_ref, target_rank)
        d = K_ref.shape[0]
        n_free = d - target_rank
        max_gauge_eig = float(np.max(gauge_eigvals[:n_free])) if n_free > 0 else 0.0
        if max_gauge_eig > 1e-6 * max(1.0, float(np.max(gauge_eigvals))):
            raise RuntimeError(
                f"This node's reference topology's 'gauge' eigenvalues "
                f"aren't close to zero (max={max_gauge_eig:.3e}) -- the "
                f"assumed theta-independent null space doesn't hold "
                f"cleanly here; pass different ref_thetas.")

        K0p = {i: P.T @ registry[i][0] @ P for i in reference_indices}
        Kcp = {i: P.T @ registry[i][1] @ P for i in reference_indices}
        Ksp = {i: P.T @ registry[i][2] @ P for i in reference_indices}
        # Flats and sockets: Kc_i = Ks_i = 0 up to rounding; leave their
        # angle terms out rather than carry ~1e-16 noise into the SDP.
        has_angle = {i: max(np.abs(registry[i][1]).max(), np.abs(registry[i][2]).max())
                        > 1e-12 * max(1.0, np.abs(registry[i][0]).max())
                     for i in reference_indices}

        # ── Build the perspective-relaxed SDP ───────────────────────────
        z = cp.Variable(len(undecided)) if undecided else None
        a = cp.Variable(len(undecided)) if undecided else None
        b = cp.Variable(len(undecided)) if undecided else None
        u = cp.Variable(len(forced_active)) if forced_active else None
        v = cp.Variable(len(forced_active)) if forced_active else None
        t = cp.Variable()

        K_expr = np.zeros((target_rank, target_rank))
        constraints = []

        arcs = arcs or {}
        for kk, i in enumerate(forced_active):
            K_expr = K_expr + K0p[i]
            if has_angle[i]:
                K_expr = K_expr + u[kk] * Kcp[i] + v[kk] * Ksp[i]
            constraints.append(cp.SOC(1.0, cp.hstack([u[kk], v[kk]])))
            if i in arcs and arcs[i][1] < 2 * np.pi:
                phi, w = arcs[i]
                constraints.append(np.cos(phi) * u[kk] + np.sin(phi) * v[kk]
                                   >= np.cos(w / 2))

        for kk, i in enumerate(undecided):
            K_expr = K_expr + z[kk] * K0p[i]
            if has_angle[i]:
                K_expr = K_expr + a[kk] * Kcp[i] + b[kk] * Ksp[i]
            constraints.append(cp.SOC(z[kk], cp.hstack([a[kk], b[kk]])))
            constraints.append(z[kk] <= 1)
            if i in arcs and arcs[i][1] < 2 * np.pi:
                phi, w = arcs[i]
                constraints.append(np.cos(phi) * a[kk] + np.sin(phi) * b[kk]
                                   >= np.cos(w / 2) * z[kk])

        if undecided and rows is not None:
            constraints.append(rows[undecided] @ z == n_free_needed)
            taken = {int(sites[i]) for i in forced_active}
            by_site = {}
            for kk, i in enumerate(undecided):
                by_site.setdefault(int(sites[i]), []).append(kk)
            for s, kks in by_site.items():
                cap = 0 if s in taken else 1
                if cap == 0 or len(kks) > 1:
                    constraints.append(cp.sum(z[kks]) <= cap)
        elif undecided:
            constraints.append(cp.sum(z) == n_free_needed)
        elif n_free_needed != 0:
            raise ValueError(
                f"No undecided couplings but forced_active "
                f"{'rows' if rows is not None else 'count'} doesn't match "
                f"the requirement ({n_free_needed} short).")

        constraints.append(K_expr - t * np.eye(target_rank) >> 0)

        problem = cp.Problem(cp.Maximize(t), constraints)
        status = _solve(problem, solver, solver_kwargs)
        if status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(
                f"Topology-bound SDP failed to solve (status={status}) for "
                f"forced_active={forced_active}, "
                f"forced_inactive={forced_inactive}, undecided={undecided}, "
                f"k={k}")

        return dict(
            t=float(t.value),
            solver_status=status,
            z=(np.asarray(z.value, dtype=float) if z is not None else np.array([])),
            u=(np.asarray(u.value, dtype=float) if u is not None else np.array([])),
            v=(np.asarray(v.value, dtype=float) if v is not None else np.array([])),
            a=(np.asarray(a.value, dtype=float) if a is not None else np.array([])),
            b=(np.asarray(b.value, dtype=float) if b is not None else np.array([])),
            forced_active=forced_active,
            forced_inactive=forced_inactive,
            undecided=undecided,
        )
    finally:
        for c, act, th in zip(system.couplings, saved_active, saved_theta):
            c.active = act
            c.set_theta(th)


# ══════════════════════════════════════════════════════════════════════
# Policy wrapper: reference-search-failure-safe node evaluation
# ══════════════════════════════════════════════════════════════════════
#
# find_full_rank_reference() returning None is only an EMPIRICAL correlate
# of a node being infeasible (see results/sdp_size9_summary.txt: 14/220
# birds-foot size-9 subsets never found a reference in 300 random tries,
# and those also independently correlated with DE's own failures there) --
# it is NOT a proof. A real branch-and-bound search that treats "no
# reference found" as "prune this node" risks silently discarding the true
# optimum whenever the search simply got unlucky rather than the node
# actually being infeasible -- exactly the failure mode this whole SDP-
# bound effort exists to avoid (see this module's own docstring and
# sdp_theta_optimizer.py's module docstring). bound_node() is the single
# entry point that encodes the agreed policy: such a node is reported
# UNRESOLVED, never pruned.

def bound_node(system, forced_active, forced_inactive, undecided, k,
                length_scale=1.0, target_rank=None, base_thetas_deg=None,
                registry=None, solver=None, max_ref_tries=300, rng=None,
                rows=None, sites=None, **solver_kwargs):
    """
    Policy wrapper around find_full_rank_reference() + sdp_topology_bound()
    for use by a real branch-and-bound search. Never raises for a node
    that merely couldn't be resolved -- returns a status instead, so the
    caller is forced to make an explicit decision rather than have an
    exception implicitly treated as "infeasible, prune".

    Returns a dict with a 'status' key, one of:

      'infeasible'  -- cardinality ALONE already rules out this node
                       (need more active couplings than there are
                       undecided slots for, or negative headroom). This
                       is a hard, decidable fact independent of any SDP
                       or reference search -- always safe to prune.

      'unresolved'  -- no full-rank reference was found for this node's
                       largest possible topology (forced_active UNION
                       undecided) within max_ref_tries random tries, OR
                       sdp_topology_bound() itself raised once a
                       reference WAS found (e.g. its gauge-eigenvalue
                       sanity check failed). 'bound' is None.
                       *** DO NOT PRUNE a node in this state. *** It must
                       be revisited (e.g. with a bigger reference-search
                       budget, or a smarter feasibility search) before a
                       real B&B can safely decide its fate -- see the
                       module-level comment above.

      'bounded'     -- a valid reference was found and the SDP solved.
                       'bound' holds the relaxed upper bound t, safe to
                       use for pruning (bound <= incumbent => prune).

    Also returns 'reference' (the theta vector used, or None),
    'forced_active'/'forced_inactive'/'undecided' (echoed back sorted),
    and for non-'bounded' statuses, 'reason' (a short machine-readable
    string: 'cardinality', 'no_reference', or the RuntimeError message
    from sdp_topology_bound for the rarer gauge-eigenvalue-check failure).
    For 'bounded', also returns 'solver_status'.

    Parameters mirror sdp_topology_bound()'s, plus:
    base_thetas_deg : reference-search starting point forwarded to
        find_full_rank_reference (defaults to each coupling's current
        theta, in degrees, if not given).
    max_ref_tries : forwarded to find_full_rank_reference as max_tries.
    rng : numpy.random.Generator or None, forwarded to
        find_full_rank_reference (defaults to a fresh, unseeded
        Generator -- pass your own for reproducibility).
    rows, sites : mixed coupling types, forwarded to sdp_topology_bound
        (cardinality then means the exact row budget; `k` is ignored).
    """
    forced_active   = sorted(int(i) for i in forced_active)
    forced_inactive = sorted(int(i) for i in forced_inactive)
    undecided       = sorted(int(i) for i in undecided)

    if target_rank is None:
        target_rank = system.total_dofs - 6

    if rows is None:
        n_free_needed, most = k - len(forced_active), len(undecided)
    else:
        site_arr = (np.arange(len(system.couplings)) if sites is None
                    else np.asarray(sites, dtype=int))
        n_free_needed, most = _row_need(np.asarray(rows, dtype=int), site_arr,
                                        forced_active, undecided, target_rank)
    if n_free_needed < 0 or n_free_needed > most:
        return dict(status='infeasible', bound=None, reference=None,
                    reason='cardinality', forced_active=forced_active,
                    forced_inactive=forced_inactive, undecided=undecided)
    if base_thetas_deg is None:
        base_thetas_deg = np.degrees([c.theta for c in system.couplings])
    if rng is None:
        rng = np.random.default_rng()

    reference_indices = sorted(set(forced_active) | set(undecided))
    ref = find_full_rank_reference(
        system, reference_indices, target_rank, length_scale=length_scale,
        base_thetas_deg=base_thetas_deg, rng=rng, max_tries=max_ref_tries)
    if ref is None:
        return dict(status='unresolved', bound=None, reference=None,
                    reason='no_reference', forced_active=forced_active,
                    forced_inactive=forced_inactive, undecided=undecided)

    try:
        result = sdp_topology_bound(
            system, forced_active, forced_inactive, undecided, k,
            length_scale=length_scale, target_rank=target_rank,
            ref_thetas=ref, solver=solver, registry=registry,
            rows=rows, sites=sites, **solver_kwargs)
    except RuntimeError as e:
        return dict(status='unresolved', bound=None, reference=ref,
                    reason=str(e), forced_active=forced_active,
                    forced_inactive=forced_inactive, undecided=undecided)

    return dict(status='bounded', bound=result['t'], reference=ref,
                reason=None, forced_active=forced_active,
                forced_inactive=forced_inactive, undecided=undecided,
                solver_status=result['solver_status'])


# ══════════════════════════════════════════════════════════════════════
# Standalone smoke check (run directly) -- NOT the full validation
# battery, see test_sdp_topology_bound.py for that.
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG

    print("=" * 72)
    print("sdp_topology_bound.py standalone smoke check")
    print("=" * 72)

    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    target_rank = system.total_dofs - 6
    m = len(system.couplings)

    ref = find_full_rank_reference(
        system, list(range(m)), target_rank, length_scale=1.0,
        base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
        rng=np.random.default_rng(0), max_tries=50)
    assert ref is not None, "full 12-coupling birds-foot system should be full-rank at its own start angles"

    result = sdp_topology_bound(system, forced_active=[], forced_inactive=[],
                                 undecided=list(range(m)), k=9, length_scale=1.0,
                                 target_rank=target_rank, ref_thetas=ref)
    print(f"Root-node bound (0 decided, {m} undecided, k=9): t = {result['t']:.6f}  "
          f"status={result['solver_status']}")
    print("(Run test_sdp_topology_bound.py for the actual validation battery.)")
