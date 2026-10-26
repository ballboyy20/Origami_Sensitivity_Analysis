"""
bound_and_evaluate.py

Splits responsibilities per this project's own conclusion (see
results/riemannian_gradient_sampling_summary.txt and the architecture
discussion that followed it): the per-topology SDP relaxation
(sdp_theta_optimizer.py) is a SOUND upper bound on lambda_min, but its
DECODE is UNRELIABLE -- verified two independent ways during that
project's own validation: the relaxed optimum frequently sits strictly
INSIDE the unit disk (no real theta behind it, for any secondary
objective used to pick the exact point), and two different secondary
tie-break objectives (log_det, tr_inv) converged to the identical poor
decode on the same subset. Riemannian gradient sampling
(riemannian_gradient_sampling.py) is a validated, literature-grounded
LOCAL method (Hosseini & Uschmajew's Riemannian generalization of
Burke-Lewis-Overton gradient sampling) that matched/slightly beat
differential_evolution on the one birds-foot subset tried.

This module is the integration point for a FIXED active-coupling
topology:

  CERTIFICATION -- sdp_certified_bound(): the SDP relaxation's STAGE-1
    bound (t*) ONLY. Stage 2 (the log_det tie-break + decode) is never
    run here -- it costs an extra QP solve to produce a theta this
    module doesn't use, since the decode is exactly the unreliable part
    being routed around. Reuses (imports) sdp_theta_optimizer.py's own
    stage-1 pieces (_calibrate_reduced, _solve_stage1_bound) rather than
    reimplementing them -- same already-validated math, just skipping
    the wasted stage-2 step.

  EVALUATION -- riemannian_multistart() / bound_and_evaluate(): the
    achieved lambda_min comes ONLY from Riemannian gradient sampling,
    run from MULTIPLE independent starting points and keeping the best.
    Gradient sampling gives a Clarke-stationarity guarantee -- a LOCAL
    property -- not global optimality, so a single run from one
    arbitrary starting point is not enough to trust as "the achieved
    value" for comparison against a global upper bound. Neither DE nor
    the SDP's own stage-2 decode is used anywhere in this module to
    produce an achieved value.

Does not modify sdp_theta_optimizer.py or riemannian_gradient_sampling.py
-- only imports from them.

Run directly: python bound_and_evaluate.py runs one quick smoke check on
the two-panel system (see test_bound_and_evaluate.py for the actual
validation battery).
"""

import sys, os
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sdp_theta_optimizer import _calibrate_reduced, _solve_stage1_bound
from riemannian_gradient_sampling import optimize_theta_riemannian, MultiplicityError


# ══════════════════════════════════════════════════════════════════════
# Certification: SDP stage-1 bound only
# ══════════════════════════════════════════════════════════════════════

def sdp_certified_bound(system, active_indices, length_scale=1.0,
                         target_rank=None, ref_thetas=None, solver=None,
                         **solver_kwargs):
    """
    The SDP relaxation's stage-1 bound t* for a FIXED active-coupling
    topology -- a valid upper bound on the true optimum lambda_min (see
    sdp_theta_optimizer.py's module docstring: relaxing u_i^2+v_i^2=1 to
    <=1 only enlarges the feasible region, so the relaxed maximum can
    only be >= the true one). Deliberately does NOT run stage 2 or
    decode any theta -- see this module's own docstring for why.

    Mutates system: sets every coupling's .active to match
    active_indices (same convention _calibrate_reduced/calibrate()
    themselves rely on -- they only READ .active, via
    build_constraint_matrix, so the caller -- this function -- has to
    set it). Leaves coupling thetas at ref_thetas (calibrate()'s own
    "leaves system at reference" convention).

    Returns dict: {'t': float, 'stage1_status': str}.
    """
    active_indices = sorted(int(i) for i in active_indices)
    for i, c in enumerate(system.couplings):
        c.active = (i in active_indices)

    if target_rank is None:
        target_rank = system.total_dofs - 6
    if ref_thetas is None:
        ref_thetas = np.array([c.theta for c in system.couplings], dtype=float)
    else:
        ref_thetas = np.asarray(ref_thetas, dtype=float).copy()

    K0_reduced, Kc_reduced, Ks_reduced = _calibrate_reduced(
        system, active_indices, length_scale, target_rank, ref_thetas)
    t_star, status = _solve_stage1_bound(
        K0_reduced, Kc_reduced, Ks_reduced, active_indices, target_rank,
        solver, solver_kwargs)
    return dict(t=t_star, stage1_status=status)


# ══════════════════════════════════════════════════════════════════════
# Evaluation: multi-start Riemannian gradient sampling
# ══════════════════════════════════════════════════════════════════════

def riemannian_multistart(system, active_indices, length_scale=1.0,
                           target_rank=None, ref_thetas=None,
                           n_restarts=5, rng=None, riemannian_kwargs=None):
    """
    Best-of-n_restarts Riemannian gradient sampling: the achieved
    lambda_min for this topology (see module docstring for why this,
    not the SDP decode or DE, is the achieved-value source of truth
    going forward).

    One restart begins at `ref_thetas` (or the system's current thetas
    if not given) -- the topology's designated non-degenerate starting
    configuration, same as used throughout this codebase, NOT anything
    derived from the SDP. The remaining n_restarts-1 begin from
    independent uniform-random points in [0, pi) per active coupling.
    Multi-start is necessary, not decorative: gradient sampling only
    certifies Clarke stationarity of whatever point it converges to,
    which need not be the global maximizer of a nonconcave objective
    like lambda_min.

    A given start can, rarely, land somewhere optimize_theta_riemannian
    cannot escape via its own small-perturbation retry (it raises
    MultiplicityError after exhausting that budget -- observed in
    practice on a handful of birds-foot topologies out of 220 during
    this project's own validation sweep, not purely theoretical). Rather
    than let that abort the whole topology's evaluation, such a start is
    discarded and replaced with a FRESH, unrelated random draw (up to
    max_start_attempts total draws) -- there is no reason to keep
    retrying near a start that is already known to be problematic when
    n_restarts-1 of the starts are randomly chosen anyway. If even that
    cannot produce n_restarts successful starts, this raises rather than
    silently returning fewer restarts than requested.

    Returns dict:
        'lambda_min'      : float -- best achieved value across restarts
        'theta'           : ndarray -- the theta achieving it
        'all_results'     : list of the successful optimize_theta_
                             riemannian() return dicts (may be longer
                             than n_restarts if some starts were
                             discarded and replaced)
        'n_start_failures': int -- how many starts were discarded for
                             persistent multiplicity (diagnostic --
                             should be rare; see
                             results/bound_and_evaluate_sweep_summary.txt
                             for the observed rate)
    """
    active_indices = sorted(int(i) for i in active_indices)
    n = len(active_indices)
    if target_rank is None:
        target_rank = system.total_dofs - 6
    if rng is None:
        rng = np.random.default_rng()
    if riemannian_kwargs is None:
        riemannian_kwargs = {}

    if ref_thetas is None:
        theta0 = np.array([system.couplings[i].theta for i in active_indices],
                           dtype=float)
        ref_thetas_full = None
    else:
        ref_thetas_full = np.asarray(ref_thetas, dtype=float)
        theta0 = ref_thetas_full[active_indices].copy()

    pending = deque([theta0] + [rng.uniform(0, np.pi, size=n)
                                 for _ in range(max(0, n_restarts - 1))])
    n_wanted = len(pending)

    max_start_attempts = n_restarts * 4 + 10
    results = []
    n_start_failures = 0
    attempts = 0
    while pending and attempts < max_start_attempts:
        start = pending.popleft()
        attempts += 1
        try:
            r = optimize_theta_riemannian(
                system, active_indices, length_scale=length_scale,
                theta_init=start, target_rank=target_rank,
                ref_thetas=ref_thetas_full, rng=rng, **riemannian_kwargs)
        except MultiplicityError:
            n_start_failures += 1
            pending.append(rng.uniform(0, np.pi, size=n))  # fresh, unrelated draw
            continue
        results.append(r)

    if len(results) < n_wanted:
        raise MultiplicityError(
            f"Only {len(results)}/{n_wanted} restarts succeeded after "
            f"{attempts} attempts ({n_start_failures} discarded for "
            f"persistent multiplicity) -- this active set may be "
            f"structurally prone to eigenvalue degeneracy across a wide "
            f"region of theta-space, not just isolated unlucky points; "
            f"worth investigating rather than raising this budget further.")

    best = max(results, key=lambda r: r['lambda_min'])
    return dict(lambda_min=best['lambda_min'], theta=best['theta'],
                all_results=results, n_start_failures=n_start_failures)


# ══════════════════════════════════════════════════════════════════════
# The split-responsibility entry point
# ══════════════════════════════════════════════════════════════════════

def polish_tied_restarts(system, active_indices, all_results, length_scale=1.0,
                         target_rank=None, polish_kwargs=None):
    """
    Stage-2 log_product polish (log_product_polish.py) applied to EVERY
    restart tied with the best lambda_min (within the polish threshold),
    not just the best one -- tied restarts can sit in different parts of
    the optimal set with different log_product, and comparing them is
    part of what the polish is for. One shared ReducedKModel (calibration
    + gauge basis) is reused across all of them.

    Returns dict:
        'best'      : the polish result with the highest final log_product
        'per_start' : list of (restart_index, polish result) for each tied
                      restart, in restart order
        'threshold' : the lambda_min floor every polish was held to
    """
    from log_product_polish import polish_log_product, ReducedKModel

    polish_kwargs = dict(polish_kwargs or {})
    lam_best = max(r['lambda_min'] for r in all_results)
    rel_tol = polish_kwargs.get('rel_tol', 1e-7)
    abs_tol = polish_kwargs.get('abs_tol', 1e-9)
    threshold = lam_best - max(abs_tol, rel_tol * abs(lam_best))

    best_start = max(all_results, key=lambda r: r['lambda_min'])
    model = ReducedKModel(system, active_indices, best_start['theta'],
                          target_rank, length_scale)

    per_start = []
    for idx, r in enumerate(all_results):
        if r['lambda_min'] < threshold:
            continue
        p = polish_log_product(system, active_indices, r['theta'],
                               lambda_star=lam_best, target_rank=target_rank,
                               length_scale=length_scale, model=model,
                               **polish_kwargs)
        per_start.append((idx, p))

    best = max((p for _, p in per_start), key=lambda p: p['log_product'])
    active = sorted(int(i) for i in active_indices)
    for k, i in enumerate(active):   # leave system at the reported winner
        system.couplings[i].set_theta(float(best['theta'][k]))
    return dict(best=best, per_start=per_start, threshold=threshold)


def _evaluate_angle_free(system, active_indices, length_scale, target_rank):
    """A design of only flats and sockets has no angles: K is fixed, so its
    locked lambda_min is both the achieved value and an exact bound. Same
    return keys as bound_and_evaluate (theta is the active couplings'
    current, irrelevant, angles)."""
    for i, c in enumerate(system.couplings):
        c.active = (i in active_indices)
    C = system.build_constraint_matrix(length_scale=length_scale)
    if np.linalg.matrix_rank(C) < target_rank:
        raise ValueError(f"angle-free design {active_indices} is rank-deficient")
    eig = np.linalg.eigvalsh(C.T @ C)
    lam = float(eig[C.shape[1] - target_rank])
    theta = np.array([system.couplings[i].theta for i in active_indices])
    return dict(t=lam, stage1_status='exact (no angles)', lambda_min=lam,
                theta=theta, gap=0.0, all_riemannian_results=[],
                n_start_failures=0, active_indices=active_indices)


def bound_and_evaluate(system, active_indices, length_scale=1.0,
                        target_rank=None, ref_thetas=None, n_restarts=5,
                        rng=None, solver=None, riemannian_kwargs=None,
                        polish=False, polish_kwargs=None):
    """
    Certified upper bound (SDP stage 1 only) + achieved value
    (Riemannian multi-start) for one FIXED topology -- the full split-
    responsibility evaluation described in this module's docstring.

    polish=True additionally runs the stage-2 log_product tie-break
    (polish_tied_restarts, above) on every restart tied with the best
    lambda_min, and adds a 'polish' key (polish_tied_restarts' return
    dict). Off by default -- 'lambda_min', 'theta' and 'gap' always
    report the stage-1 (pre-polish) result either way, so existing
    callers and saved results are unaffected.

    Returns dict:
        't'                      : float -- certified upper bound
        'stage1_status'          : str -- cvxpy status for the bound solve
        'lambda_min'             : float -- best achieved value
        'theta'                  : ndarray -- theta achieving it
        'gap'                    : float -- t - lambda_min (>= 0 expected;
                                    a negative gap here would be a
                                    soundness violation -- see
                                    test_bound_and_evaluate.py)
        'all_riemannian_results' : list -- see riemannian_multistart()
        'n_start_failures'       : int -- see riemannian_multistart()
        'active_indices'         : list[int], sorted
    """
    active_indices = sorted(int(i) for i in active_indices)
    if target_rank is None:
        target_rank = system.total_dofs - 6

    if all(getattr(system.couplings[i], 'kind', 'vgroove') != 'vgroove'
           for i in active_indices):
        return _evaluate_angle_free(system, active_indices, length_scale, target_rank)

    bound = sdp_certified_bound(system, active_indices, length_scale,
                                 target_rank, ref_thetas, solver)
    evald = riemannian_multistart(system, active_indices, length_scale,
                                   target_rank, ref_thetas, n_restarts, rng,
                                   riemannian_kwargs)

    out = dict(
        t=bound['t'],
        stage1_status=bound['stage1_status'],
        lambda_min=evald['lambda_min'],
        theta=evald['theta'],
        gap=bound['t'] - evald['lambda_min'],
        all_riemannian_results=evald['all_results'],
        n_start_failures=evald['n_start_failures'],
        active_indices=active_indices,
    )
    if polish:
        out['polish'] = polish_tied_restarts(
            system, active_indices, evald['all_results'], length_scale,
            target_rank, polish_kwargs)
    return out


if __name__ == "__main__":
    from interactive_optimizer import build_system, ARBITRARY_START_THETAS_DEG

    print("=" * 72)
    print("bound_and_evaluate.py standalone smoke check (two-panel system)")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    ref = np.radians(np.array(ARBITRARY_START_THETAS_DEG))

    result = bound_and_evaluate(system, active, length_scale=1.0,
                                 ref_thetas=ref, n_restarts=5,
                                 rng=np.random.default_rng(0))
    print(f"  certified bound t*     = {result['t']:.6f}  "
          f"(status={result['stage1_status']})")
    print(f"  achieved lambda_min    = {result['lambda_min']:.6f}")
    print(f"  gap (t - achieved)     = {result['gap']:.3e}")
    print(f"  restart lambda_min values: "
          f"{[f'{r['lambda_min']:.6f}' for r in result['all_riemannian_results']]}")
    print("\n(Run test_bound_and_evaluate.py for the actual validation battery.)")
