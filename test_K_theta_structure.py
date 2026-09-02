"""
test_K_theta_structure.py

Checks whether K(theta) = C(theta)^T C(theta), as actually built by
RigidBodyModel.CouplingSystem, decomposes into a per-coupling sum

    K(theta) = K0 + sum_i [ cos(2*theta_i)*Kc_i + sin(2*theta_i)*Ks_i ]     (*)

over the ACTIVE couplings i -- i.e. an affine function of (cos 2*theta_i,
sin 2*theta_i) for each coupling separately, with no theta_i/theta_j
(i != j) cross terms and no higher harmonics. (*) is exactly the structure
an SDP relaxation over theta would need, so how close this gets to "zero to
machine precision" is the actual feasibility signal.

Method (black-box: only RigidBodyModel.CouplingSystem.build_constraint_matrix
is used -- this deliberately does NOT reuse the u/w/n1/n2 formulas from
RigidBodyModel, so a bug shared between the model and this script can't
cancel out):

  Calibration, per active coupling i, holding every other active coupling at
  a fixed reference angle:
    - evaluate K at theta_i in {0, pi/4, pi/2}. cos(2*theta_i)/sin(2*theta_i)
      take the clean values (1,0) / (0,1) / (-1,0) there, which is enough to
      solve (*) for Kc_i, Ks_i exactly (3 equations, 3 unknowns, no fitting).
    - K0 (summed over ALL active couplings) then falls out of one more
      identity using K at the full reference angle vector.

  Verification (the actual test): rebuild K directly from the CouplingSystem
  at many theta combinations -- both a single-axis grid sweep per active
  coupling (holding the rest at reference, as requested) and a joint random
  sweep across all active couplings at once (to catch cross terms a
  single-axis sweep alone can't) -- and compare against K0 + sum cos/sin
  Kc_i/Ks_i from calibration. Also checks that wiggling an INACTIVE
  coupling's theta leaves K unchanged, since which couplings are "active"
  is part of what K(theta) means here.

Run directly: python test_K_theta_structure.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)

GRID_N          = 41     # points per single-axis theta sweep
N_RANDOM_JOINT  = 200     # joint random theta combinations
TOL             = 1e-9    # failure threshold on max|K_actual - K_predicted|
SEED            = 0


def get_K(system, thetas_rad, length_scale):
    """Set every coupling's theta (index-aligned to system.couplings) and
    rebuild K = C^T C through the real CouplingSystem code path."""
    for c, th in zip(system.couplings, thetas_rad):
        c.set_theta(th)
    C = system.build_constraint_matrix(length_scale=length_scale)
    return C.T @ C


def calibrate(system, ref_thetas, active_idx, length_scale):
    """Solve (*) exactly for Kc_i, Ks_i (all i in active_idx) and the
    combined K0, using only calls to get_K."""
    K_ref = get_K(system, ref_thetas, length_scale)

    Kc, Ks = {}, {}
    trial = ref_thetas.copy()
    for i in active_idx:
        trial[i] = 0.0
        K_at_0 = get_K(system, trial, length_scale)
        trial[i] = np.pi / 2
        K_at_half = get_K(system, trial, length_scale)
        trial[i] = np.pi / 4
        K_at_quarter = get_K(system, trial, length_scale)
        trial[i] = ref_thetas[i]

        Kc[i] = (K_at_0 - K_at_half) / 2.0
        Ks[i] = K_at_quarter - (K_at_0 + K_at_half) / 2.0

    correction = np.zeros_like(K_ref)
    for i in active_idx:
        correction = correction + np.cos(2 * ref_thetas[i]) * Kc[i] \
                                 + np.sin(2 * ref_thetas[i]) * Ks[i]
    K0 = K_ref - correction

    get_K(system, ref_thetas, length_scale)   # leave system at reference
    return K0, Kc, Ks


def predict_K(K0, Kc, Ks, thetas_rad, active_idx):
    K = K0.copy()
    for i in active_idx:
        K = K + np.cos(2 * thetas_rad[i]) * Kc[i] + np.sin(2 * thetas_rad[i]) * Ks[i]
    return K


def verify_config(name, system, ref_thetas_deg, active_idx, length_scale=1.0,
                   grid_n=GRID_N, n_random=N_RANDOM_JOINT, seed=SEED):
    print(f"\n{name}")
    print(f"  {len(active_idx)} active / {len(system.couplings)} total couplings, "
          f"length_scale={length_scale}")

    ref = np.radians(np.array(ref_thetas_deg, dtype=float))
    K0, Kc, Ks = calibrate(system, ref, active_idx, length_scale)

    # ── single-axis grid sweep: vary one active theta_i at a time ──────
    grid = np.linspace(0, 2 * np.pi, grid_n)
    trial = ref.copy()
    max_err_axis = 0.0
    for i in active_idx:
        for th in grid:
            trial[i] = th
            err = np.max(np.abs(get_K(system, trial, length_scale)
                                 - predict_K(K0, Kc, Ks, trial, active_idx)))
            max_err_axis = max(max_err_axis, err)
        trial[i] = ref[i]
    print(f"  single-axis sweep  ({len(active_idx)} couplings x {grid_n} pts): "
          f"max|K_actual - K_pred| = {max_err_axis:.3e}")

    # ── joint random sweep: vary all active thetas simultaneously ──────
    rng = np.random.default_rng(seed)
    max_err_joint = 0.0
    for _ in range(n_random):
        trial = ref.copy()
        for i in active_idx:
            trial[i] = rng.uniform(0, 2 * np.pi)
        err = np.max(np.abs(get_K(system, trial, length_scale)
                             - predict_K(K0, Kc, Ks, trial, active_idx)))
        max_err_joint = max(max_err_joint, err)
    print(f"  joint random sweep ({n_random} combos):              "
          f"max|K_actual - K_pred| = {max_err_joint:.3e}")

    # ── inactive couplings must not move K at all ───────────────────────
    inactive_idx = [i for i in range(len(system.couplings)) if i not in active_idx]
    max_err_inactive = 0.0
    if inactive_idx:
        K_base = get_K(system, ref, length_scale)
        for i in inactive_idx:
            trial = ref.copy()
            for th in grid[::4]:
                trial[i] = th
                max_err_inactive = max(max_err_inactive,
                    np.max(np.abs(get_K(system, trial, length_scale) - K_base)))
            trial[i] = ref[i]
        print(f"  inactive-coupling invariance ({len(inactive_idx)} couplings): "
              f"max|K(theta) - K(ref)| = {max_err_inactive:.3e}")
        get_K(system, ref, length_scale)

    worst = max(max_err_axis, max_err_joint, max_err_inactive)
    status = "OK" if worst < TOL else "FAIL"
    print(f"  -> {status} (worst = {worst:.3e}, tol = {TOL:.0e})")

    assert max_err_axis < TOL, \
        f"{name}: single-axis sweep residual {max_err_axis:.3e} exceeds tol {TOL:.0e}"
    assert max_err_joint < TOL, \
        f"{name}: joint random sweep residual {max_err_joint:.3e} exceeds tol {TOL:.0e}"
    assert max_err_inactive < TOL, \
        f"{name}: inactive coupling moved K by {max_err_inactive:.3e} (should be exactly inert)"

    return dict(name=name, max_err_axis=max_err_axis, max_err_joint=max_err_joint,
                max_err_inactive=max_err_inactive)


# ══════════════════════════════════════════════════════════════════════
# Configurations -- a handful of distinct active-coupling setups
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("K(theta) structure check: K(theta) =?= K0 + sum_i [cos(2*ti)Kc_i + sin(2*ti)Ks_i]")
print("=" * 72)

configs = []

sys1 = build_system(ARBITRARY_START_THETAS_DEG)
configs.append(("Two-panel, 3/3 couplings active",
                 sys1, ARBITRARY_START_THETAS_DEG, [0, 1, 2], 1.0))

sys2 = build_system(ARBITRARY_START_THETAS_DEG)
sys2.couplings[2].active = False
configs.append(("Two-panel, 2/3 active (1 pruned)",
                 sys2, ARBITRARY_START_THETAS_DEG, [0, 1], 1.0))

sys3 = build_system(ARBITRARY_START_THETAS_DEG)
configs.append(("Two-panel, 3/3 active, length_scale=0.35",
                 sys3, ARBITRARY_START_THETAS_DEG, [0, 1, 2], 0.35))

sys4 = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
configs.append(("Birds-foot, 12/12 couplings active",
                 sys4, BIRDSFOOT_START_THETAS_DEG, list(range(12)), 1.0))

sys5 = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
pruned = [2, 5, 9]   # one coupling off of three different spokes
for i in pruned:
    sys5.couplings[i].active = False
active5 = [i for i in range(12) if i not in pruned]
configs.append(("Birds-foot, 9/12 active (3 pruned across spokes)",
                 sys5, BIRDSFOOT_START_THETAS_DEG, active5, 1.0))

results = [verify_config(*cfg) for cfg in configs]

print("\n" + "=" * 72)
print("Summary")
print("=" * 72)
for r in results:
    worst = max(r['max_err_axis'], r['max_err_joint'], r['max_err_inactive'])
    print(f"  {r['name']:<48s} worst = {worst:.3e}")
print(f"\nAll {len(results)} configurations satisfy (*) to within {TOL:.0e} -- "
      f"K(theta) is exactly affine in (cos 2*theta_i, sin 2*theta_i) per active "
      f"coupling, with no cross terms. SDP over theta has a well-founded "
      f"quadratic/trigonometric structure to build on.")
