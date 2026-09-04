"""
sdp_theta_optimizer.py

Convex (SDP) relaxation of STAGE 1 ONLY (maximize lambda_min) of
CouplingOptimizer.optimize_theta(), for a FIXED, already-chosen active-
coupling topology. See coupling_optimizer.py's module docstring, item 2
"Coupling on/off selection" — that discrete on/off choice is NOT relaxed
here; this module treats the active set as given. Relaxing it too (for
branch-and-bound topology search) is a separate, riskier follow-up that
needs its own soundness validation against this one first.

Relies on the affine decomposition test_K_theta_structure.py already
validated to ~1e-15 (floating point noise, not an approximation):

    K(theta) = K0 + sum_i [ cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i ]     (*)

over active couplings i, with no cross terms. Substituting u_i =
cos(2*theta_i), v_i = sin(2*theta_i) (u_i**2 + v_i**2 = 1) makes K affine
in (u, v), so stage 1

    maximize_theta  lambda_min( K(theta) )

relaxes to the SDP

    maximize t
    s.t.     K0 + sum_i (u_i*Kc_i + v_i*Ks_i) - t*I  >>  0
             u_i**2 + v_i**2 <= 1     for each active coupling i

Dropping "== 1" to "<= 1" enlarges the feasible set (it's the convex hull
of the circle, not the circle itself), so the SDP's optimal t is a valid
UPPER BOUND on the true stage-1 optimum lambda_min — never a lower bound.
The whole point of validating against ground truth (two-panel, and the
birds-foot size-9 sweep) is to see how tight that bound is in practice,
and whether decoding (u_i, v_i) back to a real theta_i and re-evaluating
the TRUE lambda_min through CouplingOptimizer's own eigendecomposition
finds solutions competitive with (or better than) differential_evolution.

Run directly: python sdp_theta_optimizer.py runs (a) a standalone decode
round-trip unit test and (b) the two-panel sanity check.
"""

import sys, os
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from test_K_theta_structure import calibrate
from coupling_optimizer import CouplingOptimizer


# ══════════════════════════════════════════════════════════════════════
# theta <-> (u, v) = (cos 2*theta, sin 2*theta) encode/decode
# ══════════════════════════════════════════════════════════════════════

def encode_theta_to_uv(theta):
    """theta (radians) -> (u, v) = (cos(2*theta), sin(2*theta))."""
    return np.cos(2. * theta), np.sin(2. * theta)


def decode_uv_to_theta(u, v):
    """
    (u, v) -> theta in [0, pi), radians.

    theta = 0.5*atan2(v, u). Only the ANGLE of (u, v) matters, not its
    magnitude, so this is well-defined even when the SDP's relaxed
    (u_i, v_i) lands strictly inside the unit disk (u_i**2+v_i**2 < 1) —
    exactly the same convention CouplingOptimizer._decode_thetas already
    uses for its periodic-embedding search (see that method's docstring).

    Taken mod pi because theta and theta+pi are the same physical V-groove
    (n1/n2 swap, K unchanged) — atan2 alone returns a value in
    (-pi/2, pi/2] after the 0.5x scaling, so the mod also folds in the
    other half of atan2's (-pi, pi] range. The only degenerate input is
    (u, v) = (0, 0) (atan2(0,0) = 0 by convention) — this can only occur
    at a measure-zero point of the relaxed feasible region, not a wall a
    solver gets pinned against.
    """
    return (0.5 * np.arctan2(v, u)) % np.pi


def _symmetrize(M):
    return 0.5 * (M + M.T)


# ══════════════════════════════════════════════════════════════════════
# Gauge-mode projection
# ══════════════════════════════════════════════════════════════════════
#
# K = C^T C always has (at least) a d_free = total_dofs - target_rank
# dimensional null space from the whole-assembly rigid-body gauge modes:
# constraint rows measure RELATIVE motion between two panels at a shared
# point, which is exactly zero whenever every panel undergoes the SAME
# global rigid motion -- true regardless of groove orientation theta,
# regardless of which couplings are active, as long as the active set
# keeps the panels in one connected rigid group. So this subspace is
# fixed (theta-independent) for a given connected topology.
#
# A naive "K(theta) - t*I >> 0, maximize t" SDP is therefore trivially
# capped at t=0 (K is PSD with a nonempty null space for EVERY theta),
# which is NOT what lambda_min() means here -- CouplingOptimizer defines
# lambda_min as the smallest *locked* eigenvalue (eigs[n_free:], skipping
# exactly those always-zero gauge directions), not literally the
# smallest eigenvalue of the whole matrix. Getting this projection right
# is what makes the SDP relaxation match the real objective; see
# sdp_lambda_min's docstring.
#
# If the active topology reaches target_rank at the reference thetas,
# rank-nullity forces its null space to be EXACTLY this fixed gauge
# subspace (dimension target_rank's complement, contains it, same
# dimension count => equal) -- so P, computed once from K at any single
# full-rank reference, correctly captures the "locked" subspace for
# every other theta of the SAME topology too. If rank drops below
# target_rank at some other theta (a genuinely degenerate configuration,
# e.g. parallel grooves), the extra null direction still shows up as a
# near-zero eigenvalue of P^T K P -- P doesn't need to change to still
# catch it.

def _locked_subspace_projection(K_ref, target_rank):
    """
    Orthonormal basis P (d x target_rank) for the "locked" complement of
    K_ref's smallest (d - target_rank) eigenvalues (expected ~0 gauge
    modes if K_ref truly reaches target_rank -- checked by the caller).
    """
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(K_ref))  # ascending
    d = K_ref.shape[0]
    n_free = d - target_rank
    return eigvecs[:, n_free:], eigvals


# ══════════════════════════════════════════════════════════════════════
# Core SDP relaxation
# ══════════════════════════════════════════════════════════════════════

def sdp_lambda_min(system, active_indices, length_scale=1.0,
                    target_rank=None, ref_thetas=None, solver=None,
                    **solver_kwargs):
    """
    Solve the SDP relaxation of stage 1 (maximize lambda_min) for a fixed
    active-coupling topology, then decode back to real thetas and report
    the TRUE achieved lambda_min alongside the relaxed bound.

    Parameters
    ----------
    system : CouplingSystem
        Modified in place: every coupling's `active` flag is set to
        match active_indices, and (on return) active couplings are left
        at the decoded optimal thetas — same "leaves system at the
        result" convention as CouplingOptimizer.optimize_theta() +
        apply_result().
    active_indices : sequence of int
        Indices into system.couplings to activate. Every other coupling
        is deactivated. Must be nonempty and reach target_rank (checked
        after the fact via the true re-evaluation, not assumed).
    length_scale : float
        Forwarded to build_constraint_matrix / CouplingOptimizer — same
        meaning as elsewhere in this codebase.
    target_rank : int or None
        Defaults to system.total_dofs - 6, matching CouplingOptimizer's
        own default.
    ref_thetas : (len(system.couplings),) array or None
        Reference angles calibrate() linearizes K around. Defaults to
        each coupling's current theta. The decomposition (*) is EXACT
        (not a local linearization) for any reference, so this choice
        cannot bias the SDP's answer — it only has to avoid an
        accidentally rank-deficient reference config, which calibrate()
        itself doesn't require either.
    solver : cvxpy solver constant or None
        Defaults to trying cp.CLARABEL, falling back to cp.SCS if
        CLARABEL fails or is unavailable — both open-source, no MOSEK
        license assumed.
    solver_kwargs : forwarded to Problem.solve().

    Returns
    -------
    dict with keys:
        't'                 : float — SDP-relaxed upper bound on lambda_min
        'lambda_min'        : float — TRUE lambda_min after decode + apply
                               (via CouplingOptimizer.lambda_min())
        'log_product'       : float — TRUE log_product at the same thetas
                               (-inf if rank-deficient)
        'thetas'            : (n,) ndarray — decoded angles (radians),
                               index-aligned to active_indices
        'active_indices'    : list[int] — echoed back, sorted
        'gap'               : float — t - lambda_min (>= 0 expected; see
                               module docstring — never negative in
                               principle since t is a valid upper bound)
        'solver_status'     : str — cvxpy problem.status
        'u', 'v'            : (n,) ndarrays — raw SDP solution
    """
    active_indices = sorted(int(i) for i in active_indices)
    n = len(active_indices)
    if n == 0:
        raise ValueError("active_indices is empty")

    for i, c in enumerate(system.couplings):
        c.active = (i in active_indices)

    if target_rank is None:
        target_rank = system.total_dofs - 6

    if ref_thetas is None:
        ref_thetas = np.array([c.theta for c in system.couplings], dtype=float)
    else:
        ref_thetas = np.asarray(ref_thetas, dtype=float).copy()

    K0, Kc, Ks = calibrate(system, ref_thetas, active_indices, length_scale)
    d = K0.shape[0]
    K0 = _symmetrize(K0)
    Kc = {i: _symmetrize(Kc[i]) for i in active_indices}
    Ks = {i: _symmetrize(Ks[i]) for i in active_indices}

    # calibrate() leaves the system's active couplings set to ref_thetas
    # (see its docstring) -- build C/K there directly to get the gauge
    # projection P (see _locked_subspace_projection's module-level
    # comment for why this, not raw K - t*I, is the right SDP).
    C_ref = system.build_constraint_matrix(length_scale=length_scale)
    rank_ref = np.linalg.matrix_rank(C_ref)
    if rank_ref < target_rank:
        raise RuntimeError(
            f"Reference thetas are rank-deficient for this active topology "
            f"(rank={rank_ref} < target_rank={target_rank}) -- the gauge-"
            f"mode projection needs a full-rank reference. Pass different "
            f"ref_thetas (e.g. the topology's known non-degenerate start "
            f"angles).")
    K_ref = _symmetrize(C_ref.T @ C_ref)
    P, gauge_eigvals = _locked_subspace_projection(K_ref, target_rank)
    n_free = d - target_rank
    max_gauge_eig = float(np.max(gauge_eigvals[:n_free])) if n_free > 0 else 0.0
    if max_gauge_eig > 1e-6 * max(1.0, float(np.max(gauge_eigvals))):
        raise RuntimeError(
            f"Reference thetas' 'gauge' eigenvalues aren't close to zero "
            f"(max={max_gauge_eig:.3e}) -- the assumed theta-independent "
            f"null space doesn't hold cleanly here; pass different "
            f"ref_thetas.")

    K0 = P.T @ K0 @ P
    Kc = {i: P.T @ Kc[i] @ P for i in active_indices}
    Ks = {i: P.T @ Ks[i] @ P for i in active_indices}

    # ── Build and solve the SDP (over the target_rank-dim locked subspace,
    #    not the full d-dim space -- see the projection comment above) ───
    u = cp.Variable(n)
    v = cp.Variable(n)
    t = cp.Variable()

    K_expr = K0
    for k, i in enumerate(active_indices):
        K_expr = K_expr + u[k] * Kc[i] + v[k] * Ks[i]

    constraints = [K_expr - t * np.eye(target_rank) >> 0]
    for k in range(n):
        constraints.append(cp.SOC(1.0, cp.hstack([u[k], v[k]])))

    problem = cp.Problem(cp.Maximize(t), constraints)

    if solver is not None:
        problem.solve(solver=solver, **solver_kwargs)
    else:
        try:
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        except cp.error.SolverError:
            problem.solve(solver=cp.SCS, **solver_kwargs)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            problem.solve(solver=cp.SCS, **solver_kwargs)

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(
            f"SDP relaxation failed to solve (status={problem.status}) "
            f"for active_indices={active_indices}")

    t_val = float(t.value)
    u_val = np.asarray(u.value, dtype=float)
    v_val = np.asarray(v.value, dtype=float)
    thetas = decode_uv_to_theta(u_val, v_val)

    # ── Apply decoded thetas and get the TRUE lambda_min/log_product ────
    for i, theta in zip(active_indices, thetas):
        system.couplings[i].set_theta(float(theta))

    optimizer = CouplingOptimizer(system, target_rank=target_rank,
                                   length_scale=length_scale)
    lam  = optimizer.lambda_min()
    logp = optimizer.log_product()

    return dict(
        t=t_val,
        lambda_min=lam,
        log_product=logp,
        thetas=thetas,
        active_indices=active_indices,
        gap=t_val - lam,
        solver_status=problem.status,
        u=u_val,
        v=v_val,
    )


# ══════════════════════════════════════════════════════════════════════
# Standalone checks (run directly)
# ══════════════════════════════════════════════════════════════════════

def _test_decode_roundtrip():
    """theta -> (u,v) -> theta over a grid, checking the physical (mod pi)
    identity, not raw equality — see decode_uv_to_theta's docstring for
    why theta and theta+pi must map to the same result."""
    print("=" * 72)
    print("decode_uv_to_theta round-trip unit test")
    print("=" * 72)

    grid = np.linspace(0., np.pi, 4001, endpoint=False)  # avoid pi itself: pi%pi==0==0%pi trivially, fine either way
    u, v = encode_theta_to_uv(grid)
    decoded = decode_uv_to_theta(u, v)

    # Compare on the circle (cos, sin of 2*theta), not raw theta, so the
    # 0/pi wraparound point doesn't false-positive as a big error.
    err = np.abs(np.angle(np.exp(1j * 2 * (decoded - grid))))
    max_err = float(np.max(err))
    print(f"  {len(grid)} grid points, max angular round-trip error = {max_err:.3e}")
    assert max_err < 1e-9, f"round-trip failed: max_err={max_err:.3e}"

    # theta and theta+pi must decode identically (same physical groove).
    # Compared via the same circular (mod-pi) metric as above, since a
    # raw subtraction would false-positive right at decode's own [0, pi)
    # wraparound edge (e.g. 1e-16 vs pi-1e-16 are the same groove, but
    # differ by ~pi as raw floats).
    d1 = decode_uv_to_theta(*encode_theta_to_uv(grid))
    d2 = decode_uv_to_theta(*encode_theta_to_uv(grid + np.pi))
    max_period_err = float(np.max(np.abs(np.angle(np.exp(1j * 2 * (d1 - d2))))))
    print(f"  theta vs theta+pi decode difference (should be ~0): "
          f"{max_period_err:.3e}")
    assert max_period_err < 1e-9

    # Interior-of-disk points (relaxed SDP solutions, not on the unit
    # circle) must still decode by angle only.
    rng = np.random.default_rng(0)
    r = rng.uniform(0.01, 1.0, size=200)
    th = rng.uniform(0, np.pi, size=200)
    u2, v2 = r * np.cos(2 * th), r * np.sin(2 * th)
    decoded2 = decode_uv_to_theta(u2, v2)
    err2 = np.max(np.abs(np.angle(np.exp(1j * 2 * (decoded2 - th)))))
    print(f"  interior-of-disk (magnitude != 1) decode error: {err2:.3e}")
    assert err2 < 1e-9

    print("  -> OK\n")


def _test_two_panel_sanity():
    """Two-panel, 3-coupling sanity check against the hand-confirmed
    Slocum bisector angles and lambda_min from the task instructions."""
    from interactive_optimizer import build_system, ARBITRARY_START_THETAS_DEG

    print("=" * 72)
    print("Two-panel sanity check")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    result = sdp_lambda_min(system, active_indices=[0, 1, 2], length_scale=1.0)

    thetas_deg = np.degrees(result['thetas']) % 180.0
    # Coupling construction order is CONTACT_POINTS = (p1, p2, p3) =
    # (y=0.2, y=0.8, y=0.5-through-thickness) -- NOT sorted by the
    # "x=0.2, 0.5, 0.8" phrasing used to describe the bisector angles, so
    # the expected values are reordered to match index order 0, 1, 2.
    expected_deg = np.array([99.22, 80.78, 180.00]) % 180.0

    print(f"  SDP relaxed bound t:       {result['t']:.6f}")
    print(f"  achieved lambda_min:       {result['lambda_min']:.6f}")
    print(f"  gap (t - lambda_min):      {result['gap']:.6e}")
    print(f"  decoded thetas (deg):      {np.round(thetas_deg, 2).tolist()}")
    print(f"  expected (Slocum) (deg):   {expected_deg.tolist()}")

    # Angular distance mod 180 deg (0 and 180 are the same groove).
    diff = np.abs(((thetas_deg - expected_deg + 90) % 180) - 90)
    print(f"  angular error per coupling (deg): {np.round(diff, 3).tolist()}")

    assert np.all(diff < 2.0), \
        f"decoded thetas too far from expected Slocum angles: {diff}"
    assert abs(result['lambda_min'] - 0.0133) < 2e-3, \
        f"achieved lambda_min {result['lambda_min']:.6f} too far from 0.0133"
    print("  -> OK\n")


if __name__ == "__main__":
    _test_decode_roundtrip()
    _test_two_panel_sanity()
    print("All standalone sdp_theta_optimizer.py checks passed.")
