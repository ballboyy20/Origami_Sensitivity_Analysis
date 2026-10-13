"""
test_log_product_polish.py

Validation for log_product_polish.py, checked against the REAL system
(CouplingOptimizer._locked_eigenvalues: C built from scratch, rank via
matrix_rank, eigs[n_free:]) rather than against the reduced model itself:

  1. Reduced spectrum: eig(P^T K(theta) P) equals the real locked
     eigenvalues at random thetas FAR from the reference theta P was
     built at (the fixed-gauge-subspace claim the whole module rests on).
  2. log_product: matches CouplingOptimizer.log_product() at those thetas.
  3. Gradient: closed-form d(log det Kr)/dtheta vs. central finite
     differences of the real log_product.
  4. Polish (two-panel, where a wide lambda_min plateau is known to
     exist): log_product does not decrease, final point is feasible
     (real lambda_min >= threshold), every accepted step is feasible.
  5. Polish (birds-foot champion topology): same feasibility/monotonicity
     checks, verified on the real system.

Run directly: python test_log_product_polish.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)
from coupling_optimizer import CouplingOptimizer
from sdp_topology_bound import find_full_rank_reference
from bound_and_evaluate import riemannian_multistart
from log_product_polish import ReducedKModel, polish_log_product

CHAMPION = [0, 1, 2, 3, 5, 6, 7, 8, 11]
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def real_locked(system, active, theta):
    for i, c in enumerate(system.couplings):
        c.active = i in active
    opt = CouplingOptimizer(system)
    _, locked = opt._locked_eigenvalues(theta)
    return opt, locked


def birdsfoot_champion_setup(seed=0):
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    target_rank = system.total_dofs - 6
    ref = find_full_rank_reference(system, CHAMPION, target_rank,
                                   base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
                                   rng=np.random.default_rng(seed))
    return system, target_rank, ref


def test_model_against_real_system():
    print("=" * 72)
    print("Parts 1-3: reduced model vs. real system (birds-foot champion topology)")
    print("=" * 72)
    system, target_rank, ref = birdsfoot_champion_setup()
    model = ReducedKModel(system, CHAMPION, ref[CHAMPION], target_rank)
    rng = np.random.default_rng(1)

    worst_eig, worst_lp, n_checked = 0.0, 0.0, 0
    for _ in range(20):
        th = rng.uniform(0, np.pi, size=len(CHAMPION))
        opt, locked = real_locked(system, CHAMPION, th)
        if locked is None:
            continue
        n_checked += 1
        mu = np.linalg.eigvalsh(model.Kr(th))
        worst_eig = max(worst_eig, np.max(np.abs(mu - locked)) / np.max(locked))
        worst_lp = max(worst_lp, abs(model.log_product(th) - opt.log_product()))
    check("reduced spectrum == real locked eigenvalues", worst_eig < 1e-9,
          f"(max rel err {worst_eig:.2e} over {n_checked} random thetas)")
    check("log_product == CouplingOptimizer.log_product()", worst_lp < 1e-7,
          f"(max abs err {worst_lp:.2e})")

    worst_g = 0.0
    h = 1e-6
    for _ in range(5):
        th = rng.uniform(0, np.pi, size=len(CHAMPION))
        g = model.log_product_grad(th)
        fd = np.empty_like(g)
        for k in range(len(th)):
            e = np.zeros_like(th); e[k] = h
            fd[k] = (real_locked(system, CHAMPION, th + e)[0].log_product()
                     - real_locked(system, CHAMPION, th - e)[0].log_product()) / (2 * h)
        worst_g = max(worst_g, np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd))))
    check("gradient vs. central finite differences", worst_g < 1e-5,
          f"(max rel err {worst_g:.2e})")


def _check_polish(label, system, active, p):
    hist = p['history']['log_product']
    check(f"{label}: log_product non-decreasing along accepted steps",
          all(b >= a for a, b in zip(hist, hist[1:])))
    check(f"{label}: every accepted step feasible (reduced model)",
          min(p['history']['lambda_min']) >= p['threshold'])
    opt, locked = real_locked(system, active, p['theta'])
    check(f"{label}: final point feasible on REAL system",
          locked is not None and locked[0] >= p['threshold'],
          f"(real lambda_min {locked[0]:.8f} >= threshold {p['threshold']:.8f})")
    check(f"{label}: final log_product matches REAL system",
          abs(opt.log_product() - p['log_product']) < 1e-7)
    print(f"      log_product {p['log_product_start']:.6f} -> {p['log_product']:.6f}"
          f"  lambda_min {p['lambda_min_start']:.8f} -> {p['lambda_min']:.8f}"
          f"  max move {p['max_abs_move_deg']:.2f} deg  ({p['stop_reason']}, "
          f"{p['n_iterations']} iters)")


def test_polish_two_panel():
    print("=" * 72)
    print("Part 4: polish on the two-panel system (known wide plateau)")
    print("=" * 72)
    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    ref = np.radians(np.array(ARBITRARY_START_THETAS_DEG))
    ms = riemannian_multistart(system, active, ref_thetas=ref, n_restarts=3,
                               rng=np.random.default_rng(0))
    p = polish_log_product(system, active, ms['theta'], lambda_star=ms['lambda_min'])
    _check_polish("two-panel", system, active, p)


def test_polish_birdsfoot_champion():
    print("=" * 72)
    print("Part 5: polish on the birds-foot champion topology")
    print("=" * 72)
    system, target_rank, ref = birdsfoot_champion_setup()
    ms = riemannian_multistart(system, CHAMPION, target_rank=target_rank,
                               ref_thetas=ref, n_restarts=2,
                               rng=np.random.default_rng(0))
    p = polish_log_product(system, CHAMPION, ms['theta'],
                           lambda_star=ms['lambda_min'], target_rank=target_rank)
    _check_polish("birds-foot champion", system, CHAMPION, p)


if __name__ == "__main__":
    test_model_against_real_system()
    test_polish_two_panel()
    test_polish_birdsfoot_champion()
    print("=" * 72)
    print(f"{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)
