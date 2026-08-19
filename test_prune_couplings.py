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
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name)
from visualization_rigid import draw_3d_config, draw_eigenvalue_bar

RESULT_LOG_PATH = os.path.join(os.path.dirname(__file__), 'prune_couplings_result.txt')

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

# Kept separate from opt.system (which prune_couplings mutates in place)
# so the "before" state stays available for a before/after comparison.
system_before = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))

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
# REPORT: which couplings got removed, round by round — printed AND
# saved to RESULT_LOG_PATH so this doesn't have to be re-derived from
# terminal scrollback (or a re-run) later.
# ══════════════════════════════════════════════════════════════════════
def build_removal_report(prune_result):
    lines = [
        "Coupling pruning report (birds-foot)",
        "=" * 78,
        f"stop_reason:   {prune_result.stop_reason}",
        f"min_couplings: {prune_result.min_couplings}",
        "",
    ]
    header = (f"{'Round':>5}  {'Active->':>8}  {'Spoke':>5}  "
              f"{'Removed point (x,y,z)':>26}  {'theta(deg)':>10}  "
              f"{'2nd choice':>10}  {'lambda_min after':>16}  {'log_prod after':>14}")
    lines.append(header)
    lines.append("-" * len(header))
    for i, step in enumerate(prune_result.steps):
        if step.removed_coupling is None:
            lines.append(f"{i:>5}  {step.n_active_before:>8}  "
                         f"(final round — nothing removed, search stopped)")
            continue
        c     = step.removed_coupling
        spoke = birdsfoot_spoke_name(c)
        pt    = tuple(np.round(c.point, 4))
        lines.append(
            f"{i:>5}  {step.n_active_before:>8}  {spoke:>5}  {str(pt):>26}  "
            f"{np.degrees(c.theta):>10.2f}  {str(step.removed_was_second_choice):>10}  "
            f"{step.lambda_min_after_removal:>16.6f}  {step.log_product_after_removal:>14.6f}")

    lines.append("")
    lines.append("Final active couplings per spoke:")
    active_by_spoke = {}
    for c in opt.system.couplings:
        if getattr(c, 'active', True):
            active_by_spoke.setdefault(birdsfoot_spoke_name(c), []).append(c)
    for spoke_name in ('O-M', 'O-B', 'O-N', 'O-A'):
        n = len(active_by_spoke.get(spoke_name, []))
        lines.append(f"  {spoke_name}: {n} active")

    return "\n".join(lines)


report_text = build_removal_report(result)
print("\n" + report_text)
with open(RESULT_LOG_PATH, "w", encoding="utf-8") as f:
    f.write(report_text + "\n")
print(f"\n(saved to {RESULT_LOG_PATH})")


# ══════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════

# -- Trajectory: stiffness vs. active coupling count ---------------------
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

# -- Before vs. after 3-D configuration -----------------------------------
final_active = result.steps[-1].n_active_before
fig2 = plt.figure(figsize=(14, 7))
fig2.suptitle('Birds-foot configuration: before vs. after pruning', fontweight='bold')
ax_before = fig2.add_subplot(1, 2, 1, projection='3d')
ax_after  = fig2.add_subplot(1, 2, 2, projection='3d')
draw_3d_config(ax_before, system_before,
              title=f'Before pruning ({len(system_before.couplings)} couplings, all active)')
draw_3d_config(ax_after, opt.system,
              title=f'After pruning ({final_active} active, '
                    f'{len(opt.system.couplings) - final_active} removed)')
plt.tight_layout()
plt.show()

# -- Before vs. after eigenvalue spectrum ----------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Eigenvalue spectrum: before vs. after pruning', fontweight='bold')
draw_eigenvalue_bar(axes3[0], system_before, title='Before pruning')
draw_eigenvalue_bar(axes3[1], opt.system, title='After pruning')
plt.tight_layout()
plt.show()

print("\n✓ All tests passed.")
