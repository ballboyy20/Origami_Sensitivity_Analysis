"""
gore_bnb_full_gore_allv_figures.py

Figures for the exactly constrained all-V-groove full-gore design found by
B&B (gore_bnb.py full-gore-allv; KINEMATIC_COUPLING_SUMMARY.md sec 7.17).
The B&B results JSON stores the chosen set but not its angles, so this
re-evaluates that set with the same SetEvaluator settings (seed 31, 5
restarts, per-set seeded rng) and checks that lambda_min reproduces.

Same figure styles as forward_greedy_full_gore.py (sec 7.4), so the two
designs can be compared directly:
  1. 3-D config, all 27 candidates vs. the B&B design at its achieved
     angles, each with the weakest constrained eigenmode overlaid.
  2. Hinge layout per crease (active: theta + groove normals; inactive:
     removed marker).
  3. B&B incumbent trace over time, with the certified upper bound and the
     sec 7.4 over-constrained greedy design for reference.

Saves: results/gore_bnb_full_gore_allv_{3d_before_after,hinges,incumbent}.png
and results/gore_bnb_full_gore_allv_design.json (set, angles, lambda_min, bound).

Run: python gore_bnb_full_gore_allv_figures.py
"""
import sys, os, json, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from full_flasher_gore_system import (
    build_full_gore_system, FULL_GORE_START_THETAS_DEG, full_gore_spoke_name,
    FULL_GORE_CREASES, FULL_GORE_THICKNESS,
)
from swap_search import SetEvaluator
from visualization_rigid import draw_3d_config, draw_hinge_layout

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
IN_JSON = os.path.join(RESULTS_DIR, 'gore_bnb_full_gore_allv_results.json')
OUT_BASE = os.path.join(RESULTS_DIR, 'gore_bnb_full_gore_allv')
SEED = 31                     # gore_bnb.py's SEED
GREEDY_74 = 0.005185          # sec 7.4: 23 V-grooves, 46 rows (over-constrained)


def main():
    with open(IN_JSON, encoding='utf-8') as f:
        bnb = json.load(f)['bnb']
    active = sorted(bnb['active'])
    saved_lam, ub = bnb['lambda_min'], bnb['certified_upper_bound']

    system = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    m, tr = len(system.couplings), system.total_dofs - 6
    print(f"Re-evaluating B&B design ({len(active)} of {m}): {active}", flush=True)
    ev = SetEvaluator(system, tr, FULL_GORE_START_THETAS_DEG, seed=SEED)
    t0 = time.perf_counter()
    e = ev.evaluate(active)
    print(f"   lambda_min {e['lambda_min']:.6f} (saved {saved_lam:.6f}, rel diff "
          f"{abs(e['lambda_min'] - saved_lam) / saved_lam:.1e}); leaf bound t = {e['t']:.6f}; "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    theta = np.asarray(e['theta'], dtype=float)

    with open(OUT_BASE + '_design.json', 'w', encoding='utf-8') as f:
        json.dump(dict(active=active, theta_rad=theta.tolist(),
                       theta_deg=np.degrees(theta).tolist(),
                       lambda_min=float(e['lambda_min']), leaf_bound=float(e['t']),
                       certified_upper_bound=ub, saved_lambda_min=saved_lam), f, indent=1)

    # after: B&B design at its achieved angles; before: all candidates active
    after = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    th = dict(zip(active, theta))
    for i, c in enumerate(after.couplings):
        c.active = i in th
        if c.active:
            c.set_theta(th[i])
    before = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    lam_before = before.get_rigidity_eigenvalue(length_scale=1.0)

    # 1. 3-D before/after
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Full flasher gore: all candidates vs. exactly constrained B&B design '
                 '(21 V-grooves, 42 rows)', fontweight='bold', fontsize=12)
    draw_3d_config(fig.add_subplot(1, 2, 1, projection='3d'), before,
                   title=f'All {m} candidates active\n(lambda_min={lam_before:.4f})',
                   show_lambda_min_mode=True)
    draw_3d_config(fig.add_subplot(1, 2, 2, projection='3d'), after,
                   title=f'B&B design, {len(active)} of {m} active\n'
                         f'(lambda_min={e["lambda_min"]:.4f}, certified UB={ub:.4f})',
                   show_lambda_min_mode=True)
    plt.tight_layout()
    fig.savefig(OUT_BASE + '_3d_before_after.png', dpi=150)
    plt.close(fig)

    # 2. hinge layouts, one per crease (3 candidates per crease, in order)
    n = len(FULL_GORE_CREASES)
    ncols, nrows = 3, -(-n // 3)
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    fig.suptitle('Full flasher gore: groove arrangement per crease (B&B design)',
                 fontweight='bold', fontsize=12)
    for ci, crease in enumerate(FULL_GORE_CREASES):
        cc = after.couplings[3 * ci: 3 * ci + 3]
        act = [c for c in cc if c.active]
        draw_hinge_layout(fig.add_subplot(nrows, ncols, ci + 1), act,
                          crease['p_start'], crease['p_end'], cc[0].face_normal,
                          FULL_GORE_THICKNESS,
                          title=f'Crease {full_gore_spoke_name(cc[0])} ({len(act)}/3 active)',
                          inactive_couplings=[c for c in cc if not c.active])
    plt.tight_layout()
    fig.savefig(OUT_BASE + '_hinges.png', dpi=150)
    plt.close(fig)

    # 3. incumbent trace
    tr_ = bnb['incumbent_trace']
    hrs = [u['elapsed_sec'] / 3600 for u in tr_] + [bnb['elapsed_sec'] / 3600]
    lams = [u['lambda_min'] for u in tr_] + [tr_[-1]['lambda_min']]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(hrs, lams, where='post', color='steelblue', linewidth=1.8,
            label='B&B incumbent (achieved lambda_min)')
    ax.plot(hrs[:-1], lams[:-1], 'o', color='steelblue', markersize=4)
    ax.axhline(ub, color='#2ECC71', linestyle=':', linewidth=1.6,
               label=f'Certified upper bound ({ub:.6f}, gap {100 * (ub / saved_lam - 1):.2f}%)')
    ax.axhline(GREEDY_74, color='#E74C3C', linestyle='--', linewidth=1.5,
               label=f'Sec 7.4 greedy, 23 V = 46 rows, over-constrained ({GREEDY_74:.6f})')
    ax.axvline(bnb['elapsed_sec'] / 3600, color='gray', linestyle=':', linewidth=1,
               label=f'Search complete ({bnb["elapsed_sec"] / 3600:.1f} h)')
    ax.set_xlabel('Wall time (h)')
    ax.set_ylabel('lambda_min')
    ax.set_title('Full gore, all V-grooves, exactly constrained: B&B progress', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_BASE + '_incumbent.png', dpi=150)
    plt.close(fig)
    print("Saved: " + ", ".join(OUT_BASE + s for s in
          ('_3d_before_after.png', '_hinges.png', '_incumbent.png', '_design.json')))


if __name__ == "__main__":
    main()
