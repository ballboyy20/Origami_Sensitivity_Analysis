"""
test_prune_baselines.py
Compares CouplingOptimizer.prune_couplings()'s greedy result against four
naive baselines on the birds-foot system: each baseline keeps full
(3-groove) coupling on three spoke edges and removes all 3 couplings from
the fourth, so every baseline has the same total coupling budget (9) as
the greedy result's default stopping point — a fair same-budget comparison
of "prune unevenly, guided by leave-one-out impact" vs. "just drop one
whole hinge".
"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from coupling_optimizer  import CouplingOptimizer
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name)
from visualization_rigid import draw_3d_config

# Same DE settings used throughout test_prune_couplings.py, for continuity.
FAST_KW = dict(maxiter=300, n_solutions=2)
SPOKE_NAMES = ('O-M', 'O-B', 'O-N', 'O-A')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_LOG_PATH = os.path.join(RESULTS_DIR, 'prune_baseline_comparison.txt')
FIG_PATH        = os.path.join(RESULTS_DIR, 'prune_baseline_comparison.png')
CONFIG_FIG_PATH = os.path.join(RESULTS_DIR, 'prune_baseline_configs.png')


def build_baseline_system(removed_spoke):
    """Birds-foot system with all 3 couplings on `removed_spoke`
    deactivated, all other 9 couplings active at the standard starting
    thetas."""
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    for c in system.couplings:
        if birdsfoot_spoke_name(c) == removed_spoke:
            c.active = False
    return system


# ══════════════════════════════════════════════════════════════════════
# Greedy pruned result — re-run fresh here (same default seeds as
# test_prune_couplings.py) so the comparison is apples-to-apples within
# one script execution rather than against numbers from a prior run.
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Running greedy prune_couplings() for comparison")
print("=" * 60)

greedy_system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
greedy_opt    = CouplingOptimizer(greedy_system)
greedy_result = greedy_opt.prune_couplings(**FAST_KW)

greedy_final_active = greedy_result.steps[-1].n_active_before
greedy_lambda_min    = greedy_result.steps[-1].theta_result.lambda_min
greedy_log_product   = greedy_result.steps[-1].theta_result.log_product
print(f"\nGreedy result: {greedy_final_active} active couplings, "
      f"lambda_min={greedy_lambda_min:.6f}, log_product={greedy_log_product:.6f}")
if greedy_final_active != 9:
    print(f"  NOTE: greedy stopped at {greedy_final_active}, not 9 — baselines "
          f"below are still exactly 9 by construction (4 spokes x 3, minus "
          f"one full spoke), so this run's comparison isn't perfectly "
          f"budget-matched.")


# ══════════════════════════════════════════════════════════════════════
# Four naive baselines: full coupling on 3 spokes, zero on the 4th
# ══════════════════════════════════════════════════════════════════════
baseline_results = {}
baseline_systems = {}
for spoke in SPOKE_NAMES:
    print("\n" + "=" * 60)
    print(f"Baseline: full coupling on all spokes except {spoke}")
    print("=" * 60)

    system = build_baseline_system(spoke)
    opt    = CouplingOptimizer(system)
    n_active = len(opt._active_couplings())
    assert n_active == 9, f"Expected 9 active couplings, got {n_active}"

    results = opt._optimize_theta_with_retries(
        theta_seed=42, tol=1e-8, max_retries=5, **FAST_KW)
    best = results[0]
    opt.apply_result(best)

    print(f"  lambda_min={best.lambda_min:.6f}  log_product={best.log_product:.6f}")
    baseline_results[spoke] = best
    baseline_systems[spoke] = system


# ══════════════════════════════════════════════════════════════════════
# Comparison report — printed and saved
# ══════════════════════════════════════════════════════════════════════
lines = [
    "Greedy pruning vs. naive single-spoke-removal baselines (birds-foot)",
    "=" * 78,
    f"Greedy final active couplings: {greedy_final_active}  "
    f"(distribution: see prune_couplings_result.txt)",
    "Baseline active couplings: 9 (3 spokes x 3, 1 spoke x 0) for all four",
    "",
    f"{'Configuration':>28}  {'lambda_min':>12}  {'log_product':>12}",
    "-" * 58,
    f"{'Greedy pruned (uneven)':>28}  {greedy_lambda_min:>12.6f}  {greedy_log_product:>12.6f}",
]
for spoke in SPOKE_NAMES:
    r = baseline_results[spoke]
    lines.append(f"{'Baseline: no ' + spoke:>28}  {r.lambda_min:>12.6f}  {r.log_product:>12.6f}")

best_baseline_spoke = max(
    SPOKE_NAMES,
    key=lambda s: (baseline_results[s].lambda_min, baseline_results[s].log_product))
best_baseline = baseline_results[best_baseline_spoke]

lines.append("")
lines.append(f"Best baseline: no {best_baseline_spoke} "
             f"(lambda_min={best_baseline.lambda_min:.6f}, "
             f"log_product={best_baseline.log_product:.6f})")
lines.append(
    f"Greedy {'beats' if greedy_lambda_min > best_baseline.lambda_min else 'does NOT beat'} "
    f"the best baseline on lambda_min "
    f"({greedy_lambda_min:.6f} vs {best_baseline.lambda_min:.6f}).")
lines.append(
    f"Greedy {'beats' if greedy_log_product > best_baseline.log_product else 'does NOT beat'} "
    f"the best baseline on log_product "
    f"({greedy_log_product:.6f} vs {best_baseline.log_product:.6f}).")

report = "\n".join(lines)
print("\n" + report)
with open(RESULT_LOG_PATH, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"\n(saved to {RESULT_LOG_PATH})")


# ══════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════

# -- Bar chart: lambda_min and log_product across all 5 configurations ---
labels    = ['Greedy\n(uneven)'] + [f'No {s}' for s in SPOKE_NAMES]
lam_vals  = [greedy_lambda_min]  + [baseline_results[s].lambda_min  for s in SPOKE_NAMES]
logp_vals = [greedy_log_product] + [baseline_results[s].log_product for s in SPOKE_NAMES]
colors    = ['#3F8EFC'] + ['#BDC3C7'] * 4

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Greedy pruning vs. naive single-spoke-removal baselines', fontweight='bold')

axes[0].bar(labels, lam_vals, color=colors, edgecolor='white')
axes[0].set_ylabel('lambda_min')
axes[0].set_title('Worst-case stiffness')
axes[0].grid(True, axis='y', alpha=0.3)

axes[1].bar(labels, logp_vals, color=colors, edgecolor='white')
axes[1].set_ylabel('log_product')
axes[1].set_title('Overall stiffness proxy')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_PATH, dpi=150)
plt.show()

# -- 3-D configs: 4 baselines side by side --------------------------------
fig2 = plt.figure(figsize=(16, 8))
fig2.suptitle('Naive baselines: one spoke fully removed', fontweight='bold')
for i, spoke in enumerate(SPOKE_NAMES):
    ax = fig2.add_subplot(2, 2, i + 1, projection='3d')
    draw_3d_config(ax, baseline_systems[spoke], title=f'No {spoke} (9 active)')
plt.tight_layout()
fig2.savefig(CONFIG_FIG_PATH, dpi=150)
plt.show()

print("\nFigures saved:")
print(f"  {FIG_PATH}")
print(f"  {CONFIG_FIG_PATH}")

print("\n✓ Baseline comparison complete.")
