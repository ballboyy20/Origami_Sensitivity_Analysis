"""
two_panel_swap_validation.py

Validates swap_search.py against exhaustive ground truth: the two-panel,
6-candidate-site system (two_panel_six_sites.py, KINEMATIC_COUPLING_SUMMARY.md
sec 7.6), k = 3. Runs swap search from EVERY one of the C(6,3) = 20 starting
sets -- including the 2 rank-deficient (collinear) ones and forward greedy's
{1,2,5} -- and checks where each one ends up against the exhaustive optimum
(0.013330, reached by {0,1,2} and its mirror image {3,4,5}).

Also cross-checks the evaluator's per-set lambda_min against the values
saved by two_panel_six_sites.py's exhaustive run (different seeds, so they
should agree to within restart noise).

Saves: results/two_panel_swap_validation_{results.json,summary.txt,starts.png}

Run directly: python two_panel_swap_validation.py
"""

import sys, os, json, time
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from two_panel_six_sites import (
    build_six_site_system, START_THETAS_DEG, SITE_LABELS,
    OUT_JSON as EXHAUSTIVE_JSON,
)
from swap_search import SetEvaluator, swap_search

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, 'two_panel_swap_validation_results.json')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'two_panel_swap_validation_summary.txt')
OUT_FIG     = os.path.join(RESULTS_DIR, 'two_panel_swap_validation_starts.png')

K               = 3
SEED            = 31
IMPROVE_REL_TOL = 1e-4
OPT_REL_TOL     = 1e-4      # "reached the optimum" = within this of the best
GREEDY_START    = [1, 2, 5]

COLOR_START = '#9ec5f4'     # blue 200 (start)
COLOR_FINAL = '#2a78d6'     # blue 450 (after swap search)
INK_MUTED   = '#6b6b6b'


def short(active):
    return '{' + ', '.join(SITE_LABELS[i].split(' ')[0] for i in active) + '}'


def main():
    system = build_six_site_system()
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    ev = SetEvaluator(system, target_rank, START_THETAS_DEG, seed=SEED)

    with open(EXHAUSTIVE_JSON, encoding='utf-8') as f:
        exhaustive = {tuple(r['active']): r for r in json.load(f)['exhaustive'][str(K)]}
    opt = max(r['lambda_min'] for r in exhaustive.values() if r['feasible'])
    optimal_sets = sorted(k for k, r in exhaustive.items()
                          if r['feasible'] and r['lambda_min'] >= opt * (1 - OPT_REL_TOL))

    t0 = time.perf_counter()
    runs = []
    for start in combinations(range(m), K):
        start = list(start)
        print(f"start {start} {short(start)}")
        r = swap_search(ev, start, improve_rel_tol=IMPROVE_REL_TOL)
        r['reached_opt'] = r['lambda_min'] >= opt * (1 - OPT_REL_TOL)
        r['n_swaps_accepted'] = sum(1 for p in r['passes'] if p['accepted'])
        runs.append(r)
    elapsed = time.perf_counter() - t0

    # Evaluator vs. saved exhaustive values (independent seeds)
    diffs = []
    for key, r in exhaustive.items():
        e = ev.evaluate(list(key))
        if r['feasible'] and e['feasible']:
            diffs.append(abs(e['lambda_min'] - r['lambda_min']) / r['lambda_min'])
        elif r['feasible'] != e['feasible']:
            diffs.append(float('inf'))

    # ── Summary ───────────────────────────────────────────────────────
    n_ok = sum(r['reached_opt'] for r in runs)
    L = []
    L.append("Swap search validation -- two panels, 6 candidate sites, k=3 "
             "(exhaustive ground truth)")
    L.append("=" * 78)
    L.append(f"Exhaustive optimum: {opt:.6f}, reached by {optimal_sets}")
    L.append(f"Improvement threshold {IMPROVE_REL_TOL:g} (relative); best-improvement "
             f"passes with bound screening.")
    L.append("")
    L.append(f"RESULT: {n_ok}/{len(runs)} starts reached the optimum.  "
             f"Total time {elapsed:.1f}s; {ev.n_bound_calls} bound solves, "
             f"{ev.n_eval_calls} Riemannian evaluations (cached across starts).")
    L.append("")
    L.append(f"   {'start':10s} {'':22s} {'start lam':>10s} -> {'final':10s} "
             f"{'final set':22s} {'swaps':>5s}  {'reached?':8s}")
    for r in sorted(runs, key=lambda r: -r['start_lambda_min']):
        tag = '  <- forward greedy' if r['start_active'] == GREEDY_START else ''
        L.append(f"   {str(r['start_active']):10s} {short(r['start_active']):22s} "
                 f"{r['start_lambda_min']:10.6f} -> {r['lambda_min']:.6f} "
                 f"{short(r['active']):22s} {r['n_swaps_accepted']:5d}  "
                 f"{'yes' if r['reached_opt'] else 'NO':8s}{tag}")
    L.append("")
    tot = lambda key: sum(p[key] for r in runs for p in r['passes'])
    L.append(f"Screening across all passes: {tot('n_swaps')} swaps considered, "
             f"{tot('n_infeasible')} infeasible, {tot('n_pruned_by_bound')} pruned by "
             f"bound, {tot('n_evaluated')} needed evaluation.")
    finite = [d for d in diffs if np.isfinite(d)]
    L.append(f"Evaluator vs. saved exhaustive values (independent seeds): "
             f"max relative difference {max(finite):.2e} over {len(finite)} feasible sets; "
             f"{sum(1 for d in diffs if not np.isfinite(d))} feasibility disagreements.")
    summary = "\n".join(L)
    print("\n" + summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(
            optimum=opt, optimal_sets=[list(s) for s in optimal_sets],
            n_reached=n_ok, n_starts=len(runs), elapsed_sec=elapsed,
            runs=[dict(start=r['start_active'], start_lambda_min=r['start_lambda_min'],
                       final=r['active'], lambda_min=r['lambda_min'],
                       reached_opt=bool(r['reached_opt']), passes=r['passes'])
                  for r in runs]), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")

    # ── Figure: dumbbell, start -> final per starting set ─────────────
    order = sorted(runs, key=lambda r: r['start_lambda_min'])
    fig, ax = plt.subplots(figsize=(9, 7))
    ys = np.arange(len(order))
    for y, r in zip(ys, order):
        ax.plot([r['start_lambda_min'], r['lambda_min']], [y, y], color=COLOR_START,
                linewidth=2, zorder=1, solid_capstyle='round')
    ax.scatter([r['start_lambda_min'] for r in order], ys, s=64, color=COLOR_START,
               edgecolor='white', linewidth=2, zorder=2, label='Start')
    ax.scatter([r['lambda_min'] for r in order], ys, s=64, color=COLOR_FINAL,
               edgecolor='white', linewidth=2, zorder=3, label='After swap search')
    ax.axvline(opt, color=INK_MUTED, linestyle='--', linewidth=1, zorder=0)
    ax.text(opt, len(order) - 0.3, f' exhaustive optimum {opt:.4f}', color=INK_MUTED,
            fontsize=8, va='bottom', ha='right')
    labels = []
    for r in order:
        lab = f"{short(r['start_active'])}"
        if r['start_active'] == GREEDY_START:
            lab += '  (greedy)'
        if r['start_lambda_min'] == 0:
            lab += '  (rank-deficient)'
        labels.append(lab)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8, color='#333333')
    ax.set_xlabel('λ_min', color='#333333')
    ax.set_title(f'Swap search from all {len(runs)} starting sets: '
                 f'{n_ok}/{len(runs)} reach the optimum', fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
