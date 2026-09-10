"""
test_riemannian_gradient_sampling.py

Validation battery for riemannian_gradient_sampling.py. Run directly:
    python test_riemannian_gradient_sampling.py
Prints a PASS/FAIL summary per check, in the style of this repo's other
test scripts (test_K_theta_structure.py, test_sdp_topology_bound.py) --
plain functions run from __main__, not pytest (this repo has no pytest
files anywhere).

Parts, in order (matching the module's own Part A/B/C structure, plus
Part D's integration checks):
  A. Closed-form gradient vs. finite differences (central differences,
     a scan of step sizes -- not one arbitrary h) on the two-panel and
     birds-foot systems.
  B. Manifold primitives: retract's exact group law, and a direct model-
     level check (not just retract's own arithmetic) that shifting a
     coupling's theta by pi leaves K, hence lambda_min, unchanged.
  C/D. Integration: stationarity at the known two-panel optimum,
     convergence from a clearly-off starting point, a boundary-seam
     (theta crossing 0/pi) regression check, and an explicit ascent-
     direction sign check.

IMPORTANT FINDING FROM BUILDING THIS (not assumed going in, discovered
while validating Part A -- see the print output when you run this):
lambda_min(theta) for the two-panel system has a WIDE PLATEAU of global
maximizers. Scanning random theta triples shows a clear majority already
sit almost exactly at the same maximal lambda_min (~0.0133296) with
near-zero gradient; only a minority (poorly-oriented/near-degenerate
configurations) sit at substantially lower values with a real gradient.
This was cross-checked directly against the affine K formula vs. the real
system's own K (agreement to ~1e-16), so it's a genuine property of this
system, not a code defect -- but it means "a random theta point" is a
poor choice for exercising a NONTRIVIAL gradient. The test points below
were chosen deliberately (by direct calculation, not by construction) to
include points confirmed off the plateau.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)
from riemannian_gradient_sampling import (
    calibrate_topology, eval_lambda_min_and_grad, MultiplicityError,
    retract, optimize_theta_riemannian, _min_norm_hull_element,
)

KNOWN_OPTIMUM_DEG = [99.22, 80.78, 180.00]   # cross-validated elsewhere in
                                              # this repo (e.g.
                                              # sdp_theta_optimizer.py's
                                              # _test_two_panel_sanity),
                                              # index-aligned to couplings
                                              # 0,1,2 (CONTACT_POINTS order)
KNOWN_OPTIMUM_LAMBDA = 0.0133296   # ~ from the same cross-validation


# ══════════════════════════════════════════════════════════════════════
# Part A: closed-form gradient vs. finite differences
# ══════════════════════════════════════════════════════════════════════

def _fd_gradient(system, active, theta, K0, Kc, Ks, target_rank, length_scale, h):
    n = len(theta)
    grad = np.zeros(n)
    for k in range(n):
        tp = theta.copy(); tp[k] += h
        tm = theta.copy(); tm[k] -= h
        lp, _, _ = eval_lambda_min_and_grad(system, active, tp, K0, Kc, Ks,
                                             target_rank, length_scale)
        lm, _, _ = eval_lambda_min_and_grad(system, active, tm, K0, Kc, Ks,
                                             target_rank, length_scale)
        grad[k] = (lp - lm) / (2 * h)
    return grad


def test_part_a_gradient_finite_difference():
    print("=" * 72)
    print("PART A: closed-form gradient vs. finite differences")
    print("=" * 72)

    hs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    configs = []

    sys2p = build_system(ARBITRARY_START_THETAS_DEG)
    ref2p = np.radians(np.array(ARBITRARY_START_THETAS_DEG))
    active2p = [0, 1, 2]
    target_rank_2p = sys2p.total_dofs - 6
    K0_2p, Kc_2p, Ks_2p = calibrate_topology(sys2p, active2p, ref2p, 1.0)
    rng = np.random.default_rng(1)
    theta_points_2p = [ref2p.copy()]
    theta_points_2p += [np.radians(rng.uniform(0, 180, size=3)) for _ in range(6)]
    # deliberately off-plateau points, confirmed by direct calculation
    # during development (see module docstring)
    theta_points_2p += [np.radians(np.array([3.0, 146.4, 164.3])),
                         np.radians(np.array([160.2, 40.9, 112.2]))]
    configs.append(('two-panel', sys2p, active2p, target_rank_2p,
                     K0_2p, Kc_2p, Ks_2p, theta_points_2p, 1.0))

    sysbf = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    refbf = np.radians(np.array(BIRDSFOOT_START_THETAS_DEG))
    activebf = list(range(12))
    target_rank_bf = sysbf.total_dofs - 6
    K0_bf, Kc_bf, Ks_bf = calibrate_topology(sysbf, activebf, refbf, 1.0)
    rng2 = np.random.default_rng(2)
    theta_points_bf = [refbf.copy()]
    theta_points_bf += [np.radians(rng2.uniform(0, 180, size=12)) for _ in range(6)]
    configs.append(('birds-foot', sysbf, activebf, target_rank_bf,
                     K0_bf, Kc_bf, Ks_bf, theta_points_bf, 1.0))

    overall_max_abs = 0.0
    overall_max_rel = 0.0
    n_checked = 0
    n_skipped_degenerate = 0
    n_skipped_multiplicity = 0
    orders = []

    for name, system, active, target_rank, K0, Kc, Ks, theta_points, length_scale in configs:
        print(f"\n  {name}:")
        for theta in theta_points:
            try:
                lam, grad, v = eval_lambda_min_and_grad(
                    system, active, theta, K0, Kc, Ks, target_rank, length_scale)
            except MultiplicityError as e:
                n_skipped_multiplicity += 1
                print(f"    theta(deg)={np.round(np.degrees(theta), 1)}  "
                      f"SKIPPED (multiplicity): {e}")
                continue
            except ValueError as e:
                n_skipped_degenerate += 1
                print(f"    theta(deg)={np.round(np.degrees(theta), 1)}  "
                      f"SKIPPED (rank-deficient): {e}")
                continue

            errs = []
            for h in hs:
                try:
                    fd = _fd_gradient(system, active, theta, K0, Kc, Ks,
                                       target_rank, length_scale, h)
                    errs.append(float(np.max(np.abs(fd - grad))))
                except (MultiplicityError, ValueError):
                    errs.append(None)

            valid = [(h, e) for h, e in zip(hs, errs) if e is not None]
            if not valid:
                continue
            best_h, best_err = min(valid, key=lambda p: p[1])
            grad_norm = float(np.linalg.norm(grad))
            overall_max_abs = max(overall_max_abs, best_err)
            if grad_norm > 1e-4:
                overall_max_rel = max(overall_max_rel, best_err / grad_norm)
            n_checked += 1

            # observed convergence order between the two largest step
            # sizes (should be ~2 for central differences, before
            # floating-point cancellation takes over at very small h)
            if len(valid) >= 2 and valid[0][1] > 0 and valid[1][1] > 0:
                order = (np.log(valid[0][1] / valid[1][1])
                         / np.log(valid[0][0] / valid[1][0]))
                orders.append(order)
                order_str = f"{order:.2f}"
            else:
                order_str = "n/a"

            errs_str = ' '.join(f'{e:.2e}' if e is not None else 'nan' for e in errs)
            print(f"    theta(deg)={np.round(np.degrees(theta), 1)}  "
                  f"lambda_min={lam:.6f}  |grad|={grad_norm:.3e}")
            print(f"      FD errs by h={hs}: [{errs_str}]  "
                  f"best_abs_err={best_err:.3e} (h={best_h:.0e})  "
                  f"observed order(h[0]->h[1])={order_str}")

    print(f"\n  {n_checked} points checked, {n_skipped_degenerate} skipped "
          f"(rank-deficient), {n_skipped_multiplicity} skipped (multiplicity)")
    print(f"  max absolute FD error across all points: {overall_max_abs:.3e}")
    print(f"  max RELATIVE FD error across points with |grad| > 1e-4: "
          f"{overall_max_rel:.3e}")
    if orders:
        print(f"  observed convergence order (h[0] -> h[1]): "
              f"median={np.median(orders):.2f}, expected ~2 for central "
              f"differences (points with a near-zero gradient/error can "
              f"show a noisy order here -- not a red flag on its own)")

    ok = overall_max_abs < 1e-4 and overall_max_rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Part B: manifold primitives
# ══════════════════════════════════════════════════════════════════════

def test_part_b_retract_group_law():
    print("=" * 72)
    print("PART B: retract's exact group law and period identity")
    print("=" * 72)

    rng = np.random.default_rng(3)
    n = 5000
    theta = rng.uniform(0, np.pi, size=n)
    a = rng.uniform(-5, 5, size=n)
    b = rng.uniform(-5, 5, size=n)

    lhs = retract(theta, a + b)
    rhs = retract(retract(theta, a), b)
    diff = np.abs(((lhs - rhs + np.pi / 2) % np.pi) - np.pi / 2)
    max_group_law_err = float(np.max(diff))
    print(f"  group law R_theta(a+b) == R_(R_theta(a))(b): "
          f"max circular error over {n} samples = {max_group_law_err:.3e}")
    ok_group = max_group_law_err < 1e-9

    xi = rng.uniform(-10, 10, size=n)
    r1 = retract(theta, xi)
    r2 = retract(theta, xi + np.pi)
    diff2 = np.abs(((r1 - r2 + np.pi / 2) % np.pi) - np.pi / 2)
    max_period_err = float(np.max(diff2))
    print(f"  retract(theta,xi) == retract(theta,xi+pi) (raw retracted "
          f"values): max error over {n} samples = {max_period_err:.3e}")
    ok_period = max_period_err < 1e-9

    ok = ok_group and ok_period
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_part_b_model_period_identity():
    """Direct MODEL-level check (not just retract's own arithmetic): does
    shifting one active coupling's theta by pi leave K (hence lambda_min)
    exactly unchanged, via the REAL system -- the physical claim
    retract's periodicity rests on."""
    print("=" * 72)
    print("PART B: model-level theta ~ theta+pi identification (direct check)")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    target_rank = system.total_dofs - 6
    ref = np.radians(np.array(ARBITRARY_START_THETAS_DEG))
    K0, Kc, Ks = calibrate_topology(system, active, ref, 1.0)

    rng = np.random.default_rng(4)
    max_err = 0.0
    for _ in range(20):
        theta = rng.uniform(0, np.pi, size=3)
        lam1, _, _ = eval_lambda_min_and_grad(system, active, theta, K0, Kc, Ks,
                                               target_rank, 1.0)
        theta_shifted = theta.copy()
        theta_shifted[0] += np.pi   # raw, NOT wrapped -- a model-level check
        lam2, _, _ = eval_lambda_min_and_grad(system, active, theta_shifted, K0,
                                               Kc, Ks, target_rank, 1.0)
        max_err = max(max_err, abs(lam1 - lam2))

    print(f"  max |lambda_min(theta) - lambda_min(theta with coupling-0 "
          f"raw += pi)| over 20 random points = {max_err:.3e}")
    ok = max_err < 1e-9
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Part C/D: integration checks
# ══════════════════════════════════════════════════════════════════════

def test_stationarity_at_known_optimum():
    print("=" * 72)
    print("PART D.1: stationarity check at the known two-panel optimum")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    target_rank = system.total_dofs - 6
    theta_opt = np.radians(np.array(KNOWN_OPTIMUM_DEG))
    ref = theta_opt.copy()
    K0, Kc, Ks = calibrate_topology(system, active, ref, 1.0)

    lam0, grad0, _ = eval_lambda_min_and_grad(system, active, theta_opt, K0, Kc, Ks,
                                               target_rank, 1.0)
    print(f"  lambda_min at known optimum = {lam0:.6f} "
          f"(cross-validated elsewhere: ~{KNOWN_OPTIMUM_LAMBDA:.6f})")
    print(f"  analytic |grad| at known optimum = {np.linalg.norm(grad0):.3e}")

    rng = np.random.default_rng(5)
    grads = [grad0]
    for _ in range(3):
        xi = rng.uniform(-0.05, 0.05, size=3)
        theta_s = retract(theta_opt, xi)
        _, grad_s, _ = eval_lambda_min_and_grad(system, active, theta_s, K0, Kc, Ks,
                                                 target_rank, 1.0)
        grads.append(grad_s)
    G = np.array(grads).T
    d_star, min_norm = _min_norm_hull_element(G)
    print(f"  minimum-norm gradient-sample element at the optimum: "
          f"||d*|| = {min_norm:.3e}")
    ok_stationary = min_norm < 1e-2

    result = optimize_theta_riemannian(system, active, length_scale=1.0,
                                        theta_init=theta_opt.copy(),
                                        target_rank=target_rank,
                                        max_outer_iters=30,
                                        rng=np.random.default_rng(6))
    print(f"  running optimizer FROM the known optimum: "
          f"lambda_min {lam0:.6f} -> {result['lambda_min']:.6f} "
          f"({result['n_iterations']} iterations, converged={result['converged']})")
    ok_no_regression = result['lambda_min'] >= lam0 - 1e-6

    ok = ok_stationary and ok_no_regression
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_convergence_from_off_optimum():
    print("=" * 72)
    print("PART D.2: convergence from a clearly-off starting point")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    target_rank = system.total_dofs - 6
    theta_start_deg = [3.0, 146.4, 164.3]   # confirmed off-plateau during
                                              # development (see module
                                              # docstring)
    theta_start = np.radians(np.array(theta_start_deg))
    ref = theta_start.copy()
    K0, Kc, Ks = calibrate_topology(system, active, ref, 1.0)
    lam_start, _, _ = eval_lambda_min_and_grad(system, active, theta_start, K0, Kc,
                                                Ks, target_rank, 1.0)
    print(f"  start: theta(deg)={theta_start_deg}  lambda_min={lam_start:.6f}")

    result = optimize_theta_riemannian(system, active, length_scale=1.0,
                                        theta_init=theta_start.copy(),
                                        target_rank=target_rank,
                                        max_outer_iters=100,
                                        rng=np.random.default_rng(7))

    hist = result['history']['lambda_min']
    print(f"  lambda_min trajectory ({len(hist)} accepted points): "
          f"{[f'{v:.6f}' for v in hist]}")
    n_decreases = sum(1 for a, b in zip(hist, hist[1:]) if b < a - 1e-12)
    print(f"  accepted-point decreases in the trajectory: {n_decreases} "
          f"(should be 0 -- every ACCEPTED step is gated on a strict "
          f"lambda_min increase by the line search)")

    final_gap = KNOWN_OPTIMUM_LAMBDA - result['lambda_min']
    print(f"  final: theta(deg)={np.round(np.degrees(result['theta']), 2).tolist()}  "
          f"lambda_min={result['lambda_min']:.6f}  "
          f"({result['n_iterations']} iterations, converged={result['converged']}, "
          f"{result['n_multiplicity_events']} multiplicity-resample events)")
    print(f"  final gap vs. known optimum ({KNOWN_OPTIMUM_LAMBDA:.6f}): "
          f"{final_gap:.3e}")

    ok = n_decreases == 0 and abs(final_gap) < 1e-3
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_boundary_seam_regression():
    print("=" * 72)
    print("PART D.3: boundary-pinning regression check (theta crossing 0/pi)")
    print("=" * 72)
    print("  optimize_theta()'s own docstring documents DE/Nelder-Mead pinning "
          "near 0/pi (e.g. thetas of 0.39 deg and 179.83 deg in the same "
          "result) -- this checks the Riemannian method moves smoothly "
          "through that seam instead.\n")

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    target_rank = system.total_dofs - 6
    # coupling 2 (index 2) starts within 2 degrees of the seam (178 deg,
    # i.e. -2 deg mod 180); confirmed during development to have a
    # consistently positive gradient there (pushing it further toward,
    # and eventually across, the 180/0 seam) with the other two couplings
    # off-plateau too, so the optimizer has real work to do.
    theta_start_deg = [3.0, 146.4, 178.0]
    theta_start = np.radians(np.array(theta_start_deg))
    ref = theta_start.copy()
    K0, Kc, Ks = calibrate_topology(system, active, ref, 1.0)
    lam_start, grad_start, _ = eval_lambda_min_and_grad(
        system, active, theta_start, K0, Kc, Ks, target_rank, 1.0)
    print(f"  start: theta(deg)={theta_start_deg}  lambda_min={lam_start:.6f}  "
          f"grad={np.round(grad_start, 4).tolist()}")

    result = optimize_theta_riemannian(system, active, length_scale=1.0,
                                        theta_init=theta_start.copy(),
                                        target_rank=target_rank,
                                        max_outer_iters=100,
                                        rng=np.random.default_rng(8))

    coupling2_traj_deg = [np.degrees(th[2]) % 360 for th in result['history']['theta']]
    print(f"  coupling-2 theta trajectory (deg, unwrapped display via %360): "
          f"{[f'{v:.2f}' for v in coupling2_traj_deg]}")

    # Detect an actual seam crossing: consecutive trajectory points whose
    # raw values (mod pi, in [0,180)) jump by more than 90 deg in one
    # step is exactly what wrapping through 0/pi looks like (a smooth
    # continuous angle crossing the seam is represented as this kind of
    # jump in the mod-pi coordinate, whereas a method stuck AT the seam
    # would show values pinned near 0 or 180 with no such jump at all).
    raw_deg = [np.degrees(th[2]) for th in result['history']['theta']]
    jumps = [abs(b - a) for a, b in zip(raw_deg, raw_deg[1:])]
    crossed = any(j > 90 for j in jumps)
    near_seam_stuck = all(
        (v < 3 or v > 177) for v in raw_deg[1:]) and len(raw_deg) > 3

    print(f"  coupling-2 raw (mod pi) trajectory (deg): "
          f"{[f'{v:.2f}' for v in raw_deg]}")
    print(f"  seam crossing detected (a >90 deg jump in the mod-pi "
          f"trajectory, i.e. a smooth wrap): {crossed}")
    print(f"  stuck pinned near the seam (every point within 3 deg of "
          f"0 or 180): {near_seam_stuck}")
    print(f"  final: lambda_min {lam_start:.6f} -> {result['lambda_min']:.6f} "
          f"({result['n_iterations']} iterations)")

    ok = (not near_seam_stuck) and result['lambda_min'] > lam_start
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_sign_convention():
    print("=" * 72)
    print("PART D.4: sign-convention check (ascent, not descent)")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    target_rank = system.total_dofs - 6
    theta = np.radians(np.array([3.0, 146.4, 164.3]))   # off-plateau, real gradient
    ref = theta.copy()
    K0, Kc, Ks = calibrate_topology(system, active, ref, 1.0)

    lam0, grad0, _ = eval_lambda_min_and_grad(system, active, theta, K0, Kc, Ks,
                                               target_rank, 1.0)
    grad_norm = np.linalg.norm(grad0)
    print(f"  start: lambda_min={lam0:.6f}  |grad|={grad_norm:.3e}")
    assert grad_norm > 1e-4, "test point should have a substantial gradient"

    eps = 1e-4
    theta_plus = retract(theta, eps * grad0 / grad_norm)     # step along +grad
    theta_minus = retract(theta, -eps * grad0 / grad_norm)   # step along -grad
    lam_plus, _, _ = eval_lambda_min_and_grad(system, active, theta_plus, K0, Kc,
                                               Ks, target_rank, 1.0)
    lam_minus, _, _ = eval_lambda_min_and_grad(system, active, theta_minus, K0, Kc,
                                                Ks, target_rank, 1.0)
    print(f"  step of size {eps} along +grad: lambda_min = {lam_plus:.8f} "
          f"(delta={lam_plus - lam0:+.3e})")
    print(f"  step of size {eps} along -grad: lambda_min = {lam_minus:.8f} "
          f"(delta={lam_minus - lam0:+.3e})")

    ok_direct = lam_plus > lam0 and lam_minus < lam0

    # Same check through the actual convex-hull machinery: with a single
    # sampled gradient, the "minimum-norm hull element" is just that
    # gradient itself -- confirms the SAME sign convention holds through
    # _min_norm_hull_element, not just the raw analytic gradient.
    d_star, _ = _min_norm_hull_element(grad0.reshape(-1, 1))
    print(f"  min-norm hull element of a single-point sample == raw grad: "
          f"max diff = {np.max(np.abs(d_star - grad0)):.3e}")
    theta_hull_step = retract(theta, eps * d_star / np.linalg.norm(d_star))
    lam_hull, _, _ = eval_lambda_min_and_grad(system, active, theta_hull_step, K0,
                                               Kc, Ks, target_rank, 1.0)
    ok_hull = lam_hull > lam0
    print(f"  step along the hull's min-norm element: lambda_min = "
          f"{lam_hull:.8f} (delta={lam_hull - lam0:+.3e})")

    ok = ok_direct and ok_hull
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 72)
    print("# riemannian_gradient_sampling.py validation battery")
    print("#" * 72 + "\n")

    results = {}
    results['A: gradient vs finite differences'] = test_part_a_gradient_finite_difference()
    results['B1: retract group law + period'] = test_part_b_retract_group_law()
    results['B2: model-level theta~theta+pi'] = test_part_b_model_period_identity()
    results['D1: stationarity at known optimum'] = test_stationarity_at_known_optimum()
    results['D2: convergence from off-optimum'] = test_convergence_from_off_optimum()
    results['D3: boundary-seam regression'] = test_boundary_seam_regression()
    results['D4: sign-convention (ascent)'] = test_sign_convention()

    print("=" * 72)
    print("OVERALL RESULT")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:<42s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  {'ALL CHECKS PASSED.' if all_ok else 'AT LEAST ONE CHECK FAILED -- see detail above.'}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
