"""
two_panel_optimizer_baseline.py
Runs the basic two-panel / 3-coupling assembly through the current
two-stage theta optimizer (CouplingOptimizer.optimize_theta(),
method='differential_evolution') and persists the result as a reference
baseline: a lambda_min / log_product before-vs-after summary (console +
results/two_panel_optimizer_baseline.txt) and a 2-D hinge-layout figure
showing the groove arrangement before vs after
(results/two_panel_hinge_before_after.png).

This is meant as a fixed comparison point for the two-panel system --
e.g. for judging whether an SDP relaxation over theta recovers the same
lambda_min the two-stage search finds here (or proves a better one).

Run directly: python two_panel_optimizer_baseline.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import build_system, ARBITRARY_START_THETAS_DEG, t, face_normal
from coupling_optimizer     import CouplingOptimizer
from visualization_rigid    import draw_hinge_layout

SEED = 42

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
LOG_PATH      = os.path.join(RESULTS_DIR, 'two_panel_optimizer_baseline.txt')
FIG_PATH      = os.path.join(RESULTS_DIR, 'two_panel_hinge_before_after.png')

# Two-panel system's single hinge: mating face at x=1, spanning y in [0, 1].
EDGE_START = np.array([1., 0., 0.])
EDGE_END   = np.array([1., 1., 0.])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # optimize_theta() leaves the system's coupling thetas at whatever it
    # last evaluated internally (not necessarily the start or the optimum --
    # see apply_result()'s docstring), so drive the search on its own
    # throwaway system and rebuild fresh, untouched systems below for the
    # actual "before" (start) and "after" (optimal) theta vectors.
    opt = CouplingOptimizer(build_system(ARBITRARY_START_THETAS_DEG))
    result = opt.optimize_theta(method='differential_evolution', seed=SEED)[0]

    system_before = build_system(ARBITRARY_START_THETAS_DEG)
    system_after  = build_system(np.degrees(result.optimal_thetas))

    lines = [
        "Two-panel / 3-coupling assembly -- two-stage theta optimizer baseline",
        "=" * 72,
        f"method: differential_evolution, seed={SEED}",
        f"start thetas (deg): {ARBITRARY_START_THETAS_DEG}",
        "",
        f"lambda_min:   {result.lambda_min_initial:.6f}  ->  {result.lambda_min:.6f}  "
        f"(primary stage-1 optimum: {result.primary_lambda_min:.6f})",
        f"log_product:  {result.log_product_initial:.6f}  ->  {result.log_product:.6f}",
        f"converged:    {result.converged}",
        f"evaluations:  {result.n_evaluations}",
        f"optimal thetas (deg): {np.round(np.degrees(result.optimal_thetas), 3).tolist()}",
    ]
    print("\n".join(lines))

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {LOG_PATH}")

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Two-panel hinge: groove arrangement before vs after optimization',
                 fontweight='bold', fontsize=12)

    draw_hinge_layout(ax_before, system_before.couplings, EDGE_START, EDGE_END,
                       face_normal, t,
                       title=f'Before  (lambda_min={result.lambda_min_initial:.4f}, '
                             f'log-vol={result.log_product_initial:.3f})')
    draw_hinge_layout(ax_after, system_after.couplings, EDGE_START, EDGE_END,
                       face_normal, t,
                       title=f'After  (lambda_min={result.lambda_min:.4f}, '
                             f'log-vol={result.log_product:.3f})')

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
