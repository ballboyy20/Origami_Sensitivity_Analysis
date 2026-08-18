"""
test_prune_couplings.py
Tests for CouplingOptimizer.prune_couplings() — stage 3, greedy leave-
one-out coupling removal on top of the stage-1/2 theta optimizer.

Run against the birds-foot system (12 couplings) rather than the 2-panel
system: the 2-panel system's structural floor (ceil(6/2)=3) equals its
starting coupling count, so it can't prune at all and isn't a useful test
case. Birds-foot has real room to prune (12 -> floor 9).
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from coupling_optimizer  import CouplingOptimizer
from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG

# Reduced maxiter vs. the library default (1000) — prune_couplings() runs a
# full stage-1/2 DE search every round, and this script runs several full
# prune_couplings() calls, so keeping each round's search modest keeps the
# whole script's runtime practical (mirrors the same tradeoff
# interactive_optimizer.py makes for GUI responsiveness — these are the
# same maxiter/n_solutions values used there). n_solutions=1 was tried
# first but occasionally failed to clear stage 2's tight (1e-7-relative)
# convergence tolerance within only 150 generations; n_solutions=2 gives
# stage 2 two independent attempts per round instead.
FAST_KW = dict(maxiter=300, n_solutions=2)


def fresh_optimizer():
    """Fresh birds-foot CouplingSystem + CouplingOptimizer, all 12
    couplings active at the standard non-degenerate starting thetas."""
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    return CouplingOptimizer(system)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: prune_couplings() runs to completion with a sane trajectory
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: prune_couplings() basic trajectory")
print("=" * 60)

opt = fresh_optimizer()
structural_floor = int(np.ceil(opt.target_rank / 2))
print(f"\ntarget_rank = {opt.target_rank}, structural floor = {structural_floor}")

result = opt.prune_couplings(**FAST_KW)
print(repr(result))

print("\nTest 1a: non-empty steps, valid stop_reason")
assert len(result.steps) > 0, "Expected at least one step"
assert result.stop_reason in ('min_couplings_reached', 'rank_floor_reached'), \
    f"Unexpected stop_reason: {result.stop_reason}"
print(f"  {len(result.steps)} round(s), stop_reason={result.stop_reason} ✓")

print("\nTest 1b: n_active_before strictly decreases round to round")
counts = [s.n_active_before for s in result.steps]
print(f"  active-count trajectory: {counts}")
assert all(counts[i] > counts[i + 1] for i in range(len(counts) - 1)), \
    "Active coupling count should strictly decrease between rounds"
print("  ✓")

print("\nTest 1c: final active count respects both floors")
final_active = result.steps[-1].n_active_before
assert result.min_couplings <= final_active <= 12, \
    f"final_active={final_active} outside [{result.min_couplings}, 12]"
print(f"  final active = {final_active} (min_couplings={result.min_couplings}) ✓")

print("\nTest 1d: every round (including the final, non-removing one) is "
      "still full rank")
for i, step in enumerate(result.steps):
    assert step.theta_result.lambda_min > 0, \
        f"Round {i} lost full rank (lambda_min={step.theta_result.lambda_min})"
print(f"  all {len(result.steps)} round(s) have lambda_min > 0 ✓")

print("\nTest 1e: every non-final step actually removed a coupling; the "
      "final step removed nothing")
for step in result.steps[:-1]:
    assert step.removed_coupling is not None
assert result.steps[-1].removed_coupling is None
print("  ✓")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: second_choice_prob actually changes which coupling is removed
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: second_choice_prob engages the randomness path")
print("=" * 60)

def removed_point_sequence(prune_result):
    """Identify removed couplings by contact point (coupling objects
    differ across independently-built systems, so compare by geometry)."""
    return [tuple(np.round(step.removed_coupling.point, 6))
            for step in prune_result.steps if step.removed_coupling is not None]

opt_worst  = fresh_optimizer()
opt_second = fresh_optimizer()

# min_couplings=11 forces exactly one removal round (12 active -> 11,
# which then meets the floor) — enough to see whether the two policies
# choose differently, without running a full trajectory down to 9 twice.
result_worst  = opt_worst.prune_couplings(
    min_couplings=11, second_choice_prob=0.0, rng_seed=7, theta_seed=7, **FAST_KW)
result_second = opt_second.prune_couplings(
    min_couplings=11, second_choice_prob=1.0, rng_seed=7, theta_seed=7, **FAST_KW)

seq_worst  = removed_point_sequence(result_worst)
seq_second = removed_point_sequence(result_second)
print(f"\n  always-worst removal sequence:  {seq_worst}")
print(f"  always-second removal sequence: {seq_second}")

n_compare = min(len(seq_worst), len(seq_second))
assert n_compare > 0, "Expected at least one removal round to compare"
diverged = any(seq_worst[i] != seq_second[i] for i in range(n_compare))
assert diverged, \
    "second_choice_prob=0 and =1 removed the exact same couplings in the " \
    "exact same order — randomness path doesn't appear to engage"
print("  removal choices diverge between second_choice_prob=0 and =1 ✓")


# ══════════════════════════════════════════════════════════════════════
# FIGURE: pruning trajectory — stiffness vs. active coupling count
# ══════════════════════════════════════════════════════════════════════
lambda_traj = result.lambda_min_trajectory
counts_plot = [c for c, _ in lambda_traj]
lam_plot    = [lam for _, lam in lambda_traj]
logp_plot   = [step.theta_result.log_product for step in result.steps]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Stage 3: greedy coupling pruning (birds-foot)', fontweight='bold')

axes[0].plot(counts_plot, lam_plot, marker='o', color='steelblue')
axes[0].invert_xaxis()
axes[0].set_xlabel('Active couplings')
axes[0].set_ylabel('lambda_min (after each round\'s reoptimization)')
axes[0].set_title('Stiffness vs. coupling count')
axes[0].grid(True, alpha=0.3)

axes[1].plot(counts_plot, logp_plot, marker='o', color='#E74C3C')
axes[1].invert_xaxis()
axes[1].set_xlabel('Active couplings')
axes[1].set_ylabel('log_product')
axes[1].set_title('Overall stiffness proxy vs. coupling count')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ All tests passed.")
