"""
log_product_polish.py

Stage-2 (secondary, tie-break) polish for a FIXED active-coupling topology:
starting from a lambda_min-optimal theta* (e.g. riemannian_multistart's
best), maximize

    log_product(theta) = sum(log(locked eigenvalues of K(theta)))

subject to  lambda_min(theta) >= lambda_min* - tol

-- the same lexicographic stage-2 objective CouplingOptimizer.
_optimize_two_stage already uses (coupling_optimizer.py), re-implemented
here on the torus with exact gradients instead of a penalized
differential_evolution search over the (cos 2theta, sin 2theta) box.

WHY IT'S CHEAP AND SMOOTH HERE
==============================
The whole-assembly rigid-body gauge null space of K is the SAME fixed
6-dim subspace for every full-rank theta (see KINEMATIC_COUPLING_SUMMARY.md
sec 1), so ONE orthonormal basis P (d x target_rank) for its complement,
computed at theta*, is valid everywhere along the polish. In that basis

    Kr(theta) = P^T K(theta) P,     K(theta) = K0 + sum_i [cos 2theta_i Kc_i
                                                        + sin 2theta_i Ks_i]

has exactly the locked eigenvalues as its spectrum, and

    log_product(theta) = log det Kr(theta)

is SMOOTH wherever Kr is positive definite (unlike lambda_min), with exact
gradient

    d/dtheta_i log det Kr = tr( Kr^{-1} dKr_i ),
    dKr_i = P^T (-2 sin 2theta_i Kc_i + 2 cos 2theta_i Ks_i) P.

THE STEP: trust-region SDP on the eigenvalue constraint
=======================================================
lambda_min is nonsmooth exactly where optima sit (eigenvalue
multiplicity -- Cox & Overton), so a plain gradient projection onto
"don't decrease lambda_min" is ill-defined at a multiple eigenvalue.
Instead each iteration linearizes the CLUSTER of locked eigenvalues near
the threshold (Overton's first-order model of a multiple eigenvalue:
the cluster's eigenvalues move like the eigenvalues of
V_c^T (sum_i d_i dKr_i) V_c to first order), and solves

    maximize   g^T d
    s.t.       diag(mu_c - threshold) + sum_i d_i V_c^T dKr_i V_c  >>  0
               ||d||_2 <= Delta

a small SDP (cluster size x cluster size LMI, n variables). The cluster
is every eigenvalue mu_j with mu_j - threshold <= Delta * sqrt(sum_i
||dKr_i||_2^2): by Weyl's inequality, an eigenvalue farther than that
from the threshold cannot reach it to first order under a step of norm
<= Delta, so leaving it out loses nothing at first order. The step is
then accepted only if the TRUE log_product increases and the TRUE
lambda_min stays >= threshold (second-order effects are caught here, not
assumed away); otherwise Delta is halved. This is a standard SQP /
trust-region treatment of a semidefinite (eigenvalue) constraint, with
the retraction (theta + d) mod pi from riemannian_gradient_sampling.py.

Stops when the subproblem's predicted gain falls below gain_tol (a
first-order KKT point of the constrained problem: no feasible direction
improves log_product) or Delta falls below delta_min.

Does not modify riemannian_gradient_sampling.py or coupling_optimizer.py
-- only imports from the former.
"""

import sys, os
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from riemannian_gradient_sampling import calibrate_topology, retract


def wrapped_delta(theta_a, theta_b):
    """Signed elementwise difference theta_a - theta_b on the period-pi
    circle, in [-pi/2, pi/2)."""
    return (np.asarray(theta_a) - np.asarray(theta_b) + np.pi / 2) % np.pi - np.pi / 2


class ReducedKModel:
    """
    Kr(theta) = P^T K(theta) P for one fixed active set, P fixed at a
    full-rank reference theta (see module docstring for why one P is
    valid everywhere). theta arrays are index-aligned to
    sorted(active_indices), same convention as riemannian_gradient_
    sampling.py.
    """

    def __init__(self, system, active_indices, theta_ref, target_rank=None,
                 length_scale=1.0, gauge_tol=1e-8):
        self.active = sorted(int(i) for i in active_indices)
        if target_rank is None:
            target_rank = system.total_dofs - 6
        self.target_rank = target_rank

        theta_ref = np.asarray(theta_ref, dtype=float)
        ref_full = np.array([c.theta for c in system.couplings], dtype=float)
        ref_full[self.active] = theta_ref
        K0, Kc, Ks = calibrate_topology(system, self.active, ref_full, length_scale)

        K_ref = self._assemble(K0, [Kc[i] for i in self.active],
                               [Ks[i] for i in self.active], theta_ref)
        eigvals, eigvecs = np.linalg.eigh(K_ref)
        n_free = K_ref.shape[0] - target_rank
        scale = max(1.0, abs(eigvals[-1]))
        if n_free > 0 and abs(eigvals[n_free - 1]) > gauge_tol * scale:
            raise RuntimeError(
                f"theta_ref does not leave a clean {n_free}-dim gauge null "
                f"space (largest 'gauge' eigenvalue {eigvals[n_free - 1]:.3e}) "
                f"-- theta_ref must be a full-rank (rank={target_rank}) point.")
        P = eigvecs[:, n_free:]
        self.P = P

        sym = lambda A: 0.5 * (A + A.T)
        self.K0r = sym(P.T @ K0 @ P)
        self.Kcr = [sym(P.T @ Kc[i] @ P) for i in self.active]
        self.Ksr = [sym(P.T @ Ks[i] @ P) for i in self.active]

    @staticmethod
    def _assemble(K0, Kc_list, Ks_list, theta):
        K = K0.copy()
        for th, Kc_i, Ks_i in zip(theta, Kc_list, Ks_list):
            K = K + np.cos(2. * th) * Kc_i + np.sin(2. * th) * Ks_i
        return 0.5 * (K + K.T)

    def Kr(self, theta):
        return self._assemble(self.K0r, self.Kcr, self.Ksr, theta)

    def dKr(self, theta):
        return [-2. * np.sin(2. * th) * Kc_i + 2. * np.cos(2. * th) * Ks_i
                for th, Kc_i, Ks_i in zip(theta, self.Kcr, self.Ksr)]

    def spectrum(self, theta):
        """(ascending locked eigenvalues, eigenvectors) of Kr(theta)."""
        return np.linalg.eigh(self.Kr(theta))

    def log_product(self, theta):
        mu = np.linalg.eigvalsh(self.Kr(theta))
        return float(np.sum(np.log(mu))) if mu[0] > 0 else float('-inf')

    def log_product_grad(self, theta, mu=None, V=None):
        if mu is None:
            mu, V = self.spectrum(theta)
        Kinv = (V / mu) @ V.T
        return np.array([float(np.sum(Kinv * D)) for D in self.dKr(theta)])


def _direction_subproblem(g, mu, V, dK, threshold, delta, solver=None):
    """Trust-region SDP step (see module docstring). Returns (d, predicted
    gain g^T d, cluster size)."""
    n = len(g)
    weyl = delta * np.sqrt(sum(np.linalg.norm(D, 2) ** 2 for D in dK))
    cluster = np.where(mu - threshold <= weyl)[0]

    d = cp.Variable(n)
    constraints = [cp.norm(d, 2) <= delta]
    if len(cluster) > 0:
        Vc = V[:, cluster]
        M = [0.5 * (Vc.T @ D @ Vc + (Vc.T @ D @ Vc).T) for D in dK]
        S = np.diag(mu[cluster] - threshold) + sum(d[i] * M[i] for i in range(n))
        constraints.append(0.5 * (S + S.T) >> 0)
    prob = cp.Problem(cp.Maximize(g @ d), constraints)
    if solver is not None:
        prob.solve(solver=solver)
    else:
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS)
    if prob.status not in ('optimal', 'optimal_inaccurate') or d.value is None:
        return np.zeros(n), 0.0, len(cluster)
    d_val = np.asarray(d.value, dtype=float)
    return d_val, float(g @ d_val), len(cluster)


def polish_log_product(system, active_indices, theta_star, lambda_star=None,
                       target_rank=None, length_scale=1.0,
                       rel_tol=1e-7, abs_tol=1e-9,
                       delta_init=0.2, delta_min=1e-6, delta_max=0.5,
                       gain_tol=1e-9, max_iters=300, solver=None, model=None):
    """
    Maximize log_product over theta on the torus, holding lambda_min >=
    lambda_star - max(abs_tol, rel_tol*lambda_star) -- the same threshold
    convention as CouplingOptimizer._optimize_two_stage. See module
    docstring for the method.

    Parameters
    ----------
    theta_star : (n,) radians, index-aligned to sorted(active_indices)
        Must be full-rank. Normally riemannian_multistart's best theta.
    lambda_star : float or None
        Stage-1 optimum the threshold is anchored to. Defaults to
        lambda_min(theta_star).
    rel_tol, abs_tol : float
        How much lambda_min may be traded away. Defaults match
        coupling_optimizer.py's stage 2 (essentially none).
    model : ReducedKModel or None
        Pass one in to reuse its calibration across several polishes of
        the same topology (e.g. several tied restarts).

    Mutates system (thetas/active flags) via calibration, same as the
    rest of this pipeline; leaves the active couplings at the polished
    theta on return.

    Returns dict:
        'theta', 'lambda_min', 'log_product'          -- polished point
        'theta_start', 'lambda_min_start', 'log_product_start'
        'threshold'
        'max_abs_move_deg', 'rms_move_deg'           -- wrapped distance moved
        'locked_eigs_start', 'locked_eigs'           -- full locked spectra
        'converged' (bool), 'stop_reason', 'n_iterations'
        'history' : {'log_product', 'lambda_min', 'delta', 'cluster_size'}
                    per accepted step
    """
    active = sorted(int(i) for i in active_indices)
    theta = np.asarray(theta_star, dtype=float).copy() % np.pi
    if model is None:
        model = ReducedKModel(system, active, theta, target_rank, length_scale)

    mu, V = model.spectrum(theta)
    lam_start = float(mu[0])
    if lambda_star is None:
        lambda_star = lam_start
    threshold = lambda_star - max(abs_tol, rel_tol * abs(lambda_star))
    if lam_start < threshold:
        raise ValueError(f"theta_star is infeasible: lambda_min={lam_start:.6g} "
                         f"< threshold={threshold:.6g}")

    f = float(np.sum(np.log(mu)))
    theta0, f0, mu0 = theta.copy(), f, mu.copy()
    delta = delta_init
    history = {'log_product': [f], 'lambda_min': [lam_start], 'delta': [delta],
               'cluster_size': [0]}
    converged, stop_reason, it = False, 'max_iters', 0

    while it < max_iters:
        it += 1
        g = model.log_product_grad(theta, mu, V)
        dK = model.dKr(theta)
        d, pred, n_cluster = _direction_subproblem(g, mu, V, dK, threshold, delta, solver)

        if pred < gain_tol:
            if delta <= delta_min:
                converged, stop_reason = True, 'kkt_point'
                break
            delta = max(0.5 * delta, delta_min)
            continue

        theta_try = retract(theta, d)
        mu_try, V_try = model.spectrum(theta_try)
        if mu_try[0] >= threshold and mu_try[0] > 0:
            f_try = float(np.sum(np.log(mu_try)))
            if f_try > f:
                ratio = (f_try - f) / pred
                theta, mu, V, f = theta_try, mu_try, V_try, f_try
                history['log_product'].append(f)
                history['lambda_min'].append(float(mu[0]))
                history['delta'].append(delta)
                history['cluster_size'].append(n_cluster)
                if ratio > 0.75 and np.linalg.norm(d) > 0.9 * delta:
                    delta = min(2. * delta, delta_max)
                continue
        if delta <= delta_min:
            converged, stop_reason = True, 'trust_region_floor'
            break
        delta = max(0.5 * delta, delta_min)

    for k, i in enumerate(active):
        system.couplings[i].set_theta(float(theta[k]))

    move = np.degrees(wrapped_delta(theta, theta0))
    return dict(
        theta=theta, lambda_min=float(mu[0]), log_product=f,
        theta_start=theta0, lambda_min_start=lam_start, log_product_start=f0,
        threshold=threshold,
        max_abs_move_deg=float(np.max(np.abs(move))),
        rms_move_deg=float(np.sqrt(np.mean(move ** 2))),
        locked_eigs_start=mu0, locked_eigs=mu,
        converged=converged, stop_reason=stop_reason, n_iterations=it,
        history=history, active_indices=active,
    )
