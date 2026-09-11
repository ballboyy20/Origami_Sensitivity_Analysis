"""
riemannian_gradient_sampling.py

Riemannian gradient sampling for MAXIMIZING lambda_min(theta) over a FIXED,
already-chosen active-coupling topology -- a drop-in alternative to
CouplingOptimizer.optimize_theta()'s stage-1 search (maximize lambda_min)
for one active set. Does NOT touch topology search, stage-2 (log_product
tie-break), or coupling on/off selection -- those are separate, unrelated
pieces of this codebase.

WHY "RIEMANNIAN": each active coupling's theta_i doesn't live on a bounded
interval of R -- it lives on a CIRCLE of period pi. RigidBodyModel.py's
KinematicCoupling._compute_groove_normals builds the groove axis
u(theta) = cos(theta)*Z_hat + sin(theta)*Y_hat; at theta+pi, u flips sign
and n1/n2 swap, leaving K = C^T C EXACTLY unchanged (see
test_riemannian_gradient_sampling.py's Part B checks for a direct
numerical confirmation, not just this claim). So theta lives on a flat
torus (product of circles), not a box in R^n -- and running a Euclidean-
box-constrained optimizer on it is exactly what produces the boundary-
pinning failure mode documented in optimize_theta's own docstring.

This particular manifold is about as simple as they get, which is why this
module does NOT depend on a general-purpose manifold-optimization library
(no Pymanopt, no dependency added for this):
  - The retraction retract(theta, xi) = (theta + xi) mod pi is the EXACT
    exponential map on a flat circle, not a first-order approximation.
  - Vector transport is the IDENTITY: tangent spaces at every point of a
    flat circle canonically identify with each other (a tangent vector is
    just "a number of radians"), so there is no curvature correction
    needed when combining gradients sampled at different nearby points.

Net effect: this reduces almost exactly to Burke-Lewis-Overton gradient
sampling in flat space, with ONE change -- steps wrap via retract (mod pi)
instead of getting boxed/pinned at an artificial edge at 0 or pi.

PART A -- closed-form gradient of lambda_min
=============================================
lambda_min(theta) is, away from eigenvalue multiplicities, a simple
eigenvalue of K(theta), so the envelope theorem gives:

    d(lambda_min)/d(theta_i) = v^T (dK/d theta_i) v

where v is the unit eigenvector of K(theta) at lambda_min, and (reusing
the affine decomposition validated to ~1e-15 by test_K_theta_structure.py,
K(theta) = K0 + sum_i [cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i], with NO
cross terms between different couplings' angles):

    dK/d theta_i = -2*sin(2*theta_i)*Kc_i + 2*cos(2*theta_i)*Ks_i

K0/Kc_i/Ks_i are TRUE CONSTANTS for a given active-coupling set (not a
local linearization -- valid at every theta), calibrated ONCE via
calibrate() (test_K_theta_structure.py) and reused for an entire run. This
means computing lambda_min AND its full gradient at any theta costs
exactly ONE eigendecomposition of K(theta) plus cheap matrix arithmetic --
not one eigendecomposition per coordinate via finite differences.

lambda_min here means exactly what CouplingOptimizer._locked_eigenvalues
means by it: the smallest of the "locked" eigenvalues (eigs[n_free:],
skipping the target_rank-complement of theta-independent rigid-body gauge
modes), NOT the literal smallest eigenvalue of the full K. rank(C) (and
hence n_free) is recomputed at every theta via the REAL system's
build_constraint_matrix + np.linalg.matrix_rank, exactly matching
_locked_eigenvalues' own convention -- not assumed constant, since a
degenerate (rank-deficient) theta changing which eigenvalue is "the" one
being tracked is exactly the kind of silent error this module cannot
afford. K itself, once rank is known, is built from the cheap affine
formula, not recomputed as C^T C.

Multiplicity: if the two eigenvalues bracketing lambda_min in the sorted
spectrum are closer to it than a relative tolerance, the envelope-theorem
gradient is not well-defined (which eigenvector "belongs to" lambda_min
is ambiguous there) -- MultiplicityError is raised rather than returning
a misleading gradient. This should be RARE for individually-sampled
points (gradient sampling only ever evaluates at randomly perturbed
points, which are simple eigenvalues with probability 1) -- see this
module's test file for how often it actually fires in practice.

PART B -- manifold primitives
==============================
retract(theta, xi) = (theta + xi) mod pi, elementwise. See module
docstring above and this module's test file for the exact group-law and
period identities this satisfies.

PART C -- the gradient sampling iteration (Burke-Lewis-Overton, adapted)
==========================================================================
For a fixed active-coupling topology:
  1. Sample m = dim+1 nearby points (current point + dim random
     perturbations of a shrinkable radius, wrapped via retract).
  2. Evaluate (lambda_min, gradient) at each via Part A; resample any
     point that trips the multiplicity guard (don't abort the iteration).
  3. Find the minimum-norm element of the convex hull of the sampled
     gradients: minimize ||G @ lam||^2 s.t. lam >= 0, sum(lam) = 1 -- a
     small QP, solved via cvxpy (already a dependency in this repo).
  4. If that minimum-norm element's norm is below a stopping tolerance:
     stop if the sampling radius is already at its floor, else shrink the
     radius and continue.
  5. Otherwise, step in the ASCENT direction of that minimum-norm element
     (see this module's docstring for the sign derivation: this module
     MAXIMIZES lambda_min, the classical Burke-Lewis-Overton algorithm
     MINIMIZES a function by stepping in the NEGATIVE of the min-norm
     element of its sampled gradients -- negating "minimize lambda_min's
     negative" cancels that sign, so stepping in the RAW (un-negated)
     min-norm element ascends lambda_min; see
     test_riemannian_gradient_sampling.py's explicit sign-convention
     check), with backtracking line search on lambda_min itself, then
     retract.
"""

import sys, os
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from test_K_theta_structure import calibrate


class MultiplicityError(Exception):
    """Raised when lambda_min is not (numerically) a simple eigenvalue of
    K(theta) -- the envelope-theorem gradient is not well-defined there.
    See eval_lambda_min_and_grad's docstring."""
    pass


# ══════════════════════════════════════════════════════════════════════
# Part A: calibration + closed-form (lambda_min, gradient, eigenvector)
# ══════════════════════════════════════════════════════════════════════

def calibrate_topology(system, active_indices, ref_thetas, length_scale=1.0):
    """
    Thin wrapper around calibrate() (test_K_theta_structure.py): sets
    every coupling's `.active` flag to match active_indices (calibrate()
    itself only READS .active, indirectly via build_constraint_matrix --
    it never sets it), then calibrates K0/Kc/Ks for the whole active set
    in one call. Leaves the system with .active set to active_indices and
    every coupling's theta at ref_thetas on return (calibrate()'s own
    "leaves system at reference" convention).

    Returns (K0, Kc, Ks): K0 is (total_dofs, total_dofs); Kc, Ks are
    {coupling_index: (total_dofs, total_dofs)} dicts keyed by
    active_indices. TRUE CONSTANTS for this active set -- compute once,
    reuse for the whole optimization run (see module docstring).
    """
    active_indices = sorted(int(i) for i in active_indices)
    for i, c in enumerate(system.couplings):
        c.active = (i in active_indices)
    ref_thetas = np.asarray(ref_thetas, dtype=float).copy()
    K0, Kc, Ks = calibrate(system, ref_thetas, active_indices, length_scale)
    return K0, Kc, Ks


def eval_lambda_min_and_grad(system, active_indices, theta, K0, Kc, Ks,
                              target_rank, length_scale=1.0,
                              simplicity_tol=1e-8):
    """
    (lambda_min, gradient, eigenvector) at `theta` (radians, length =
    len(active_indices), index-aligned to sorted(active_indices)).

    Mutates system: sets each active coupling's theta to the given value
    (via set_theta) -- same "leaves the system at whatever was last
    evaluated" convention CouplingOptimizer itself uses; not restored.

    Raises ValueError if theta is rank-deficient (rank(C) < target_rank --
    lambda_min would be 0 by this codebase's own convention, and there's
    no well-defined "locked" eigenvector there). Raises MultiplicityError
    if lambda_min isn't a simple eigenvalue within simplicity_tol
    (relative) of its neighbors -- see module docstring.
    """
    active_indices = sorted(int(i) for i in active_indices)
    theta = np.asarray(theta, dtype=float)
    if len(theta) != len(active_indices):
        raise ValueError(
            f"theta has length {len(theta)}, expected {len(active_indices)} "
            f"(one per active coupling)")

    for k, i in enumerate(active_indices):
        system.couplings[i].set_theta(float(theta[k]))
    C = system.build_constraint_matrix(length_scale=length_scale)
    rank = np.linalg.matrix_rank(C)
    d = C.shape[1]
    n_free = d - rank
    if rank < target_rank:
        raise ValueError(
            f"theta is rank-deficient (rank={rank} < target_rank="
            f"{target_rank}) -- lambda_min is 0 by this codebase's own "
            f"convention (CouplingOptimizer._locked_eigenvalues); no "
            f"well-defined gradient there.")

    K = K0.copy()
    for k, i in enumerate(active_indices):
        K = K + np.cos(2. * theta[k]) * Kc[i] + np.sin(2. * theta[k]) * Ks[i]
    K = 0.5 * (K + K.T)   # guard tiny floating-point asymmetry

    eigvals, eigvecs = np.linalg.eigh(K)   # ascending
    lam = float(eigvals[n_free])
    v = eigvecs[:, n_free]

    scale = max(1.0, abs(lam))
    if n_free > 0 and (lam - eigvals[n_free - 1]) < simplicity_tol * scale:
        raise MultiplicityError(
            f"lambda_min={lam:.6g} is within {simplicity_tol:.1e} (relative) "
            f"of its neighbor below (eigs[n_free-1]={eigvals[n_free - 1]:.6g}) "
            f"-- not a simple eigenvalue; envelope-theorem gradient is "
            f"undefined here.")
    if n_free + 1 < d and (eigvals[n_free + 1] - lam) < simplicity_tol * scale:
        raise MultiplicityError(
            f"lambda_min={lam:.6g} is within {simplicity_tol:.1e} (relative) "
            f"of its neighbor above (eigs[n_free+1]={eigvals[n_free + 1]:.6g}) "
            f"-- not a simple eigenvalue; envelope-theorem gradient is "
            f"undefined here.")

    grad = np.empty(len(active_indices))
    for k, i in enumerate(active_indices):
        dK_dtheta = (-2. * np.sin(2. * theta[k]) * Kc[i]
                     + 2. * np.cos(2. * theta[k]) * Ks[i])
        grad[k] = float(v @ (dK_dtheta @ v))

    return lam, grad, v


# ══════════════════════════════════════════════════════════════════════
# Part B: manifold primitives
# ══════════════════════════════════════════════════════════════════════

def retract(theta, xi):
    """
    R_theta(xi) = (theta + xi) mod pi, elementwise.

    This is the EXACT exponential map on the flat circle each active
    coupling's theta lives on (period pi -- see module docstring and this
    module's test file for a direct numerical confirmation that theta and
    theta+pi give the identical K), not a first-order stand-in for it.
    """
    return (np.asarray(theta, dtype=float) + np.asarray(xi, dtype=float)) % np.pi


# ══════════════════════════════════════════════════════════════════════
# Part C: minimum-norm element of the convex hull of sampled gradients
# ══════════════════════════════════════════════════════════════════════

def _min_norm_hull_element(G, solver=None):
    """
    Minimum-norm element of the convex hull of G's columns (each a
    sampled lambda_min gradient): minimize ||G @ lam||^2 subject to
    lam >= 0, sum(lam) = 1. A small QP (dimension = G.shape[1], the
    number of sampled points), solved via cvxpy.

    Returns (d, norm_d): d = G @ lam* (an (n,) ndarray, n = number of
    active couplings), norm_d = ||d||_2.
    """
    n, m = G.shape
    lam = cp.Variable(m)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(G @ lam)),
                          [lam >= 0, cp.sum(lam) == 1])
    if solver is not None:
        problem.solve(solver=solver)
    else:
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            problem.solve(solver=cp.SCS)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            problem.solve(solver=cp.SCS)
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"min-norm-element QP failed to solve "
                            f"(status={problem.status})")

    lam_val = np.clip(np.asarray(lam.value, dtype=float), 0., None)
    total = lam_val.sum()
    lam_val = lam_val / total if total > 0 else np.full(m, 1.0 / m)
    d = G @ lam_val
    return d, float(np.linalg.norm(d))


# ══════════════════════════════════════════════════════════════════════
# Part C: the gradient-sampling iteration
# ══════════════════════════════════════════════════════════════════════

def optimize_theta_riemannian(system, active_indices, length_scale=1.0,
                               theta_init=None, target_rank=None,
                               ref_thetas=None,
                               sampling_radius_init=0.3,
                               sampling_radius_min=1e-6,
                               radius_shrink_factor=0.5,
                               stationarity_tol=1e-6,
                               max_outer_iters=200,
                               max_line_search_iters=30,
                               simplicity_tol=1e-8,
                               max_resample_tries=20,
                               rng=None):
    """
    Riemannian gradient sampling (Burke-Lewis-Overton, adapted -- see
    module docstring) maximizing lambda_min(theta) for a FIXED active-
    coupling set. A drop-in alternative to CouplingOptimizer.
    optimize_theta()'s stage-1 search for one already-chosen active set --
    does not touch topology/on-off selection or stage-2 tie-breaking.

    Parameters
    ----------
    system : CouplingSystem
        Mutated in place: left with active couplings' thetas at the final
        (or last-evaluated) point, same convention as
        CouplingOptimizer.optimize_theta().
    active_indices : sequence of int
        Coupling indices to activate (every other coupling deactivated).
    length_scale : float
        Forwarded to build_constraint_matrix / calibrate() -- same
        meaning as elsewhere in this codebase.
    theta_init : (len(active_indices),) array or None
        Starting point. Defaults to each active coupling's current theta.
    target_rank : int or None
        Defaults to system.total_dofs - 6.
    ref_thetas : (len(system.couplings),) array or None
        Reference angles calibrate() linearizes K around (the affine
        decomposition is exact for any reference -- this choice cannot
        bias the result, it only needs to not be rank-deficient at
        calibration time, which is a much weaker requirement than being
        rank-deficient at theta_init specifically). Defaults to every
        coupling's current theta, with active ones overridden to
        theta_init.
    sampling_radius_init, sampling_radius_min, radius_shrink_factor :
        Standard gradient-sampling outer-loop radius schedule: start at
        sampling_radius_init, shrink by radius_shrink_factor whenever the
        minimum-norm sampled-gradient element is already small, floor at
        sampling_radius_min.
    stationarity_tol : float
        Stop (once at the radius floor) once the minimum-norm element's
        norm falls below this.
    max_outer_iters, max_line_search_iters : int
        Iteration caps.
    simplicity_tol : float
        Forwarded to eval_lambda_min_and_grad.
    max_resample_tries : int
        Per sampled point, how many fresh random perturbations to try
        before giving up if each trips the multiplicity guard (astronomically
        unlikely to ever matter -- multiplicities should be rare, see
        module docstring).
    rng : numpy.random.Generator or None
        Defaults to an unseeded Generator -- pass your own for
        reproducibility.

    Returns
    -------
    dict with keys:
        'theta'                  : (n,) ndarray -- final angles (radians)
        'lambda_min'             : float -- final lambda_min
        'converged'              : bool
        'n_iterations'           : int -- outer iterations actually run
        'n_multiplicity_events'  : int -- total resample-on-multiplicity
                                    count across the whole run (diagnostic)
        'active_indices'         : list[int], sorted
        'history'                : dict with 'theta', 'lambda_min',
                                    'radius', 'min_norm_grad' lists, one
                                    entry per ACCEPTED point (mirrors
                                    CouplingOptimizer's own self._history
                                    convention)
    """
    active_indices = sorted(int(i) for i in active_indices)
    n = len(active_indices)
    if n == 0:
        raise ValueError("active_indices is empty")

    if target_rank is None:
        target_rank = system.total_dofs - 6
    if rng is None:
        rng = np.random.default_rng()

    if theta_init is None:
        theta = np.array([system.couplings[i].theta for i in active_indices],
                          dtype=float)
    else:
        theta = np.asarray(theta_init, dtype=float).copy()

    if ref_thetas is None:
        ref_thetas = np.array([c.theta for c in system.couplings], dtype=float)
        for k, i in enumerate(active_indices):
            ref_thetas[i] = theta[k]

    K0, Kc, Ks = calibrate_topology(system, active_indices, ref_thetas, length_scale)

    def evaluate(th):
        return eval_lambda_min_and_grad(system, active_indices, th, K0, Kc, Ks,
                                         target_rank, length_scale, simplicity_tol)

    history = {'theta': [], 'lambda_min': [], 'radius': [], 'min_norm_grad': []}
    n_multiplicity_events = 0

    # The initial point is not exempt from the same rare degeneracy every
    # OTHER evaluation in this function already resamples around (sampling
    # loop, line search below) -- a caller-supplied or randomly-drawn
    # theta_init can land exactly on a multiplicity too (observed in
    # practice running many random multi-start restarts across many
    # topologies -- rare per point, but not rare enough to skip handling
    # over hundreds of restarts). Perturb slightly and retry rather than
    # letting this propagate as an unhandled crash.
    for _attempt in range(max_resample_tries):
        try:
            lam0, grad0, _ = evaluate(theta)
            break
        except MultiplicityError:
            n_multiplicity_events += 1
            theta = retract(theta, rng.uniform(-1e-3, 1e-3, size=n))
    else:
        raise MultiplicityError(
            f"theta_init remained at an eigenvalue multiplicity after "
            f"{max_resample_tries} small perturbations -- this is far "
            f"beyond the 'rare' rate documented in this module's docstring "
            f"and warrants investigation, not a larger retry budget.")

    history['theta'].append(theta.copy())
    history['lambda_min'].append(lam0)

    radius = sampling_radius_init
    converged = False
    it = 0
    while it < max_outer_iters:
        it += 1

        grads = [grad0]
        for _ in range(n):
            sampled = False
            for _attempt in range(max_resample_tries):
                xi = rng.uniform(-radius, radius, size=n)
                theta_s = retract(theta, xi)
                try:
                    _, grad_s, _ = evaluate(theta_s)
                except MultiplicityError:
                    n_multiplicity_events += 1
                    continue
                grads.append(grad_s)
                sampled = True
                break
            if not sampled:
                # Astronomically unlikely (see docstring) -- fall back to
                # the current point's own gradient rather than crash the
                # whole run over one persistently-degenerate direction.
                grads.append(grad0)

        G = np.array(grads).T   # (n, m)
        d_star, min_norm = _min_norm_hull_element(G)
        history['radius'].append(radius)
        history['min_norm_grad'].append(min_norm)

        if min_norm < stationarity_tol:
            if radius <= sampling_radius_min:
                converged = True
                break
            radius = max(radius * radius_shrink_factor, sampling_radius_min)
            continue

        # ── Ascent step (see module docstring for the sign derivation) ──
        direction = d_star / min_norm
        alpha = radius
        accepted = False
        for _ in range(max_line_search_iters):
            theta_try = retract(theta, alpha * direction)
            try:
                lam_try, grad_try, _ = evaluate(theta_try)
            except MultiplicityError:
                n_multiplicity_events += 1
                alpha *= 0.5
                continue
            if lam_try > lam0:
                theta, lam0, grad0 = theta_try, lam_try, grad_try
                accepted = True
                break
            alpha *= 0.5

        if accepted:
            history['theta'].append(theta.copy())
            history['lambda_min'].append(lam0)
        elif radius <= sampling_radius_min:
            converged = True
            break
        else:
            radius = max(radius * radius_shrink_factor, sampling_radius_min)

    return dict(
        theta=theta,
        lambda_min=lam0,
        converged=converged,
        n_iterations=it,
        n_multiplicity_events=n_multiplicity_events,
        active_indices=active_indices,
        history=history,
    )
