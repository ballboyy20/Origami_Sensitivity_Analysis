"""
forward_greedy_topology_figure.py

Before/after 3-D figure for forward_greedy_topology.py's birds-foot result
-- mirrors test_prune_couplings.py's prune_before_after_3d.png (backward
greedy) and forward_greedy_full_gore.py's *_3d_before_after.png (forward
greedy, full gore), but for forward greedy on the birds-foot system.

"Before" = all 12 candidate couplings active at the standard starting
thetas. "After" = only the 9 couplings forward_greedy_topology.py chose,
each set to the theta bound_and_evaluate() found for it (Riemannian
multi-start), reusing results/forward_greedy_topology_results.json.

Run directly: python forward_greedy_topology_figure.py
"""

import os, json

import numpy as np
import matplotlib.pyplot as plt

from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
from visualization_rigid import draw_3d_config
from forward_greedy_topology import (
    forward_greedy, K, SEED, MAX_REF_TRIES, N_RESTARTS_FINAL, RESULTS_DIR, OUT_JSON,
)
from sdp_topology_bound import build_coupling_registry, find_full_rank_reference
from bound_and_evaluate import bound_and_evaluate

FIG_3D = os.path.join(RESULTS_DIR, 'forward_greedy_topology_3d_before_after.png')


def main():
    # Reuse the saved run rather than re-deriving it, so this figure always
    # matches whatever forward_greedy_topology.py last reported -- rerun
    # that script first if a fresh construction is wanted.
    with open(OUT_JSON, 'r', encoding='utf-8') as f:
        saved = json.load(f)
    active = sorted(saved['active'])
    lambda_min = saved['lambda_min']
    t = saved['t']

    base_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG)
    system_before = build_birdsfoot_system(base_thetas_deg)
    m = len(system_before.couplings)
    target_rank = system_before.total_dofs - 6
    lam_before = system_before.get_rigidity_eigenvalue(length_scale=1.0)

    # Re-evaluate the final active set to recover per-coupling thetas
    # (not saved in the JSON trace) via the same validated pipeline
    # forward_greedy_topology.py itself uses for its final report.
    system_after = build_birdsfoot_system(base_thetas_deg)
    ref_rng = np.random.default_rng(SEED + 1000)
    ref = find_full_rank_reference(
        system_after, active, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)
    final = bound_and_evaluate(system_after, active, length_scale=1.0,
                                target_rank=target_rank, ref_thetas=ref,
                                n_restarts=N_RESTARTS_FINAL,
                                rng=np.random.default_rng(SEED + 2000))

    active_set = set(active)
    theta_by_active_idx = dict(zip(sorted(active), final['theta']))
    for i, c in enumerate(system_after.couplings):
        if i in active_set:
            c.active = True
            c.set_theta(theta_by_active_idx[i])
        else:
            c.active = False

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Birds-foot: all candidates vs. forward-greedy result',
                 fontweight='bold', fontsize=12)
    ax_before = fig.add_subplot(1, 2, 1, projection='3d')
    ax_after  = fig.add_subplot(1, 2, 2, projection='3d')
    draw_3d_config(ax_before, system_before,
                   title=f'Before: all {m} candidates active\n'
                         f'(lambda_min={lam_before:.6f})',
                   show_lambda_min_mode=False)
    draw_3d_config(ax_after, system_after,
                   title=f'After: forward-greedy result, {len(active)} of {m} active\n'
                         f'(lambda_min={final["lambda_min"]:.6f}, cert. bound={final["t"]:.6f})',
                   show_lambda_min_mode=False)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(FIG_3D, dpi=150)
    print(f"Saved: {FIG_3D}")
    plt.show()


if __name__ == "__main__":
    main()
