"""
test_optimizer.py
Tests for CouplingOptimizer — groove angle optimisation for two panels.
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from RigidBodyModel    import RigidPanel, KinematicCoupling, CouplingSystem
from coupling_optimizer import CouplingOptimizer
from visualization_rigid import (
    figure_optimization_result,
    figure_optimization_heatmaps,
)


# ── Shared geometry ────────────────────────────────────────────────────
t = 0.1

panel_A = RigidPanel(0, vertices=np.array([
    [0., 0., 0.], [1., 0., 0.],
    [1., 1., 0.], [0., 1., 0.]]), thickness=t)

panel_B = RigidPanel(1, vertices=np.array([
    [1., 0., 0.], [2., 0., 0.],
    [2., 1., 0.], [1., 1., 0.]]), thickness=t)

face_normal = np.array([1., 0., 0.])

p1 = np.array([1., 0.2,  0.0])
p2 = np.array([1., 0.8,  0.0])
p3 = np.array([1., 0.5, -t  ])

def make_system(thetas=(0., 0., 0.)):
    """Fresh CouplingSystem with 3 grooves at the given thetas."""
    sys = CouplingSystem([panel_A, panel_B])
    for p, th in zip([p1, p2, p3], thetas):
        sys.add_coupling(KinematicCoupling(
            panel_A, panel_B, p, face_normal, theta=th))
    return sys


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Baseline checks before optimisation
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Baseline checks")
print("=" * 60)

# theta=(0,0,0) is 3 PARALLEL grooves — only rank 5 (see RigidBodyModel
# fix notes: parallel grooves always leave one shared slide DOF free).
# Use a spread-out, non-parallel baseline for the "clean" sanity checks
# in this section so they aren't entangled with the rank-deficient case,
# which Test 1d below tests on purpose.
NONPARALLEL_THETAS = (0., np.radians(60), np.radians(120))

print("\nTest 1a: optimizer.lambda_min() matches direct (rank-aware) computation")
system = make_system(thetas=NONPARALLEL_THETAS)
opt    = CouplingOptimizer(system)
lam_opt = opt.lambda_min()
C       = system.build_constraint_matrix()
rank    = np.linalg.matrix_rank(C)
assert rank == opt.target_rank, \
    f"Expected this spread-out config to be full rank ({opt.target_rank}), got {rank}"
eigs       = np.sort(np.linalg.eigvalsh(C.T @ C))
lam_direct = float(eigs[C.shape[1] - rank])
assert abs(lam_opt - lam_direct) < 1e-12, \
    f"lambda_min mismatch: {lam_opt} vs {lam_direct}"
print(f"  lambda_min = {lam_opt:.6f} ✓")

print("\nTest 1b: active flag defaults to True for all couplings")
system = make_system()
opt    = CouplingOptimizer(system)
active = opt._active_couplings()
assert len(active) == 3, f"Expected 3 active, got {len(active)}"
print(f"  {len(active)} active couplings ✓")

print("\nTest 1c: setting coupling.active=False excludes it")
system = make_system()
system.couplings[1].active = False
opt    = CouplingOptimizer(system)
active = opt._active_couplings()
assert len(active) == 2, f"Expected 2 active, got {len(active)}"
system.couplings[1].active = True   # restore
print(f"  2 active after disabling one ✓")

print("\nTest 1d: lambda_min == 0 at theta=(0,0,0) — parallel grooves are")
print("          rank-deficient (rank 5 < target 6), NOT a valid baseline")
system = make_system(thetas=(0., 0., 0.))
opt    = CouplingOptimizer(system)
lam    = opt.lambda_min()
assert np.linalg.matrix_rank(system.build_constraint_matrix()) == 5
assert lam == 0., f"lambda_min should be exactly 0 (rank-deficient), got {lam}"
print(f"  lambda_min = {lam:.6f} (rank-deficient, correctly reported as 0) ✓")

print("\nTest 1e: lambda_min > 0 for a non-parallel, fully-locked configuration")
system = make_system(thetas=NONPARALLEL_THETAS)
opt    = CouplingOptimizer(system)
lam    = opt.lambda_min()
assert lam > 0, f"lambda_min should be > 0, got {lam}"
print(f"  lambda_min = {lam:.6f} > 0 ✓")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: optimize_theta() — differential evolution
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: optimize_theta() — differential_evolution")
print("=" * 60)

print("\nTest 2a: result.lambda_min >= result.lambda_min_initial")
system = make_system(thetas=(0., 0., 0.))
opt    = CouplingOptimizer(system)
result = opt.optimize_theta(method='differential_evolution', seed=42)
print(f"  λ_min: {result.lambda_min_initial:.6f} → {result.lambda_min:.6f}")
print(f"  improvement: {result.improvement:+.6f}")
assert result.lambda_min >= result.lambda_min_initial - 1e-10, \
    "Optimisation made lambda_min worse"
print("  ✓")

print("\nTest 2b: result.optimal_thetas has correct shape")
assert result.optimal_thetas.shape == (3,), \
    f"Expected shape (3,), got {result.optimal_thetas.shape}"
print(f"  shape: {result.optimal_thetas.shape} ✓")
print(f"  θ_optimal (deg): {np.round(np.degrees(result.optimal_thetas), 2)}")

print("\nTest 2c: history is populated")
assert len(result.history['lambda_min']) > 0
assert len(result.history['thetas'])     == len(result.history['lambda_min'])
print(f"  {result.n_evaluations} evaluations recorded ✓")

print("\nTest 2d: result.lambda_min matches direct re-computation at optimal thetas")
system2 = make_system(thetas=result.optimal_thetas)
opt2    = CouplingOptimizer(system2)
lam_check = opt2.lambda_min()
assert abs(lam_check - result.lambda_min) < 1e-8, \
    f"Mismatch: result says {result.lambda_min:.8f}, direct gives {lam_check:.8f}"
print(f"  Direct recompute: {lam_check:.6f} ✓")

# Save system state before apply for comparison figure
system_before = make_system(thetas=(0., 0., 0.))
lam_before    = CouplingOptimizer(system_before).lambda_min()


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: apply_result()
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: apply_result()")
print("=" * 60)

print("\nTest 3a: after apply_result, system thetas match optimal_thetas")
system = make_system(thetas=(0., 0., 0.))
opt    = CouplingOptimizer(system)
result = opt.optimize_theta(method='differential_evolution', seed=42)
opt.apply_result(result)
for i, coupling in enumerate(system.couplings):
    assert abs(coupling.theta - result.optimal_thetas[i]) < 1e-12, \
        f"Coupling {i} theta mismatch after apply_result"
print("  All coupling thetas updated correctly ✓")

print("\nTest 3b: after apply_result, optimizer.lambda_min() == result.lambda_min")
lam_after = opt.lambda_min()
assert abs(lam_after - result.lambda_min) < 1e-8, \
    f"After apply: {lam_after:.8f} vs result: {result.lambda_min:.8f}"
print(f"  lambda_min after apply: {lam_after:.6f} ✓")

# Save for figure
system_after = system


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: optimize_theta() — nelder_mead
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: optimize_theta() — nelder_mead")
print("=" * 60)

print("\nTest 4a: nelder_mead also improves lambda_min from theta=0")
system = make_system(thetas=(0., 0., 0.))
opt    = CouplingOptimizer(system)
result_nm = opt.optimize_theta(method='nelder_mead')
print(f"  λ_min: {result_nm.lambda_min_initial:.6f} → {result_nm.lambda_min:.6f}")
assert result_nm.lambda_min >= result_nm.lambda_min_initial - 1e-10
print("  ✓")

print("\nTest 4b: nelder_mead seeded from optimal DE result finds at least as good")
system_warm = make_system(thetas=result.optimal_thetas)
opt_warm    = CouplingOptimizer(system_warm)
result_warm = opt_warm.optimize_theta(method='nelder_mead')
print(f"  Warm start λ_min: {result_warm.lambda_min:.6f}")
print(f"  DE result λ_min:  {result.lambda_min:.6f}")
assert result_warm.lambda_min >= result.lambda_min - 1e-6, \
    "Warm Nelder-Mead gave worse result than DE"
print("  ✓")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: active flag (future on/off infrastructure)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: active flag behaviour")
print("=" * 60)

print("\nTest 5a: disabling a coupling reduces lambda_min")
system = make_system(thetas=result.optimal_thetas)
lam_all_on = CouplingOptimizer(system).lambda_min()
system.couplings[0].active = False
lam_one_off = CouplingOptimizer(system).lambda_min()
system.couplings[0].active = True
print(f"  All on:      λ_min = {lam_all_on:.6f}")
print(f"  One off:     λ_min = {lam_one_off:.6f}")
assert lam_one_off <= lam_all_on + 1e-10, \
    "Disabling a coupling should not increase lambda_min"
print("  ✓")

print("\nTest 5b: optimising with one coupling disabled uses 2 thetas only")
system = make_system(thetas=(0., 0., 0.))
system.couplings[2].active = False
opt    = CouplingOptimizer(system)
result_2 = opt.optimize_theta(method='differential_evolution', seed=0)
assert result_2.optimal_thetas.shape == (2,), \
    f"Expected 2 thetas, got {result_2.optimal_thetas.shape}"
system.couplings[2].active = True
print(f"  Optimised 2 thetas: {np.round(np.degrees(result_2.optimal_thetas), 2)} deg ✓")

print("\n✓ All tests passed.")
print(repr(result))


# ══════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════
figure_optimization_result(
    system_before = system_before,
    system_after  = system_after,
    result        = result,
)
plt.show()

figure_optimization_heatmaps(
    system_before = system_before,
    system_after  = system_after,
    result        = result,
)
plt.show()
