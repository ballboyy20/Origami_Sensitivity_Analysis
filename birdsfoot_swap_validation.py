"""
birdsfoot_swap_validation.py

Validates swap_search.py on birds-foot (12 candidate sites, k = 9) against
the exhaustive ground truth from bound_and_evaluate_sweep.py (all 220
size-9 subsets; champion [0,1,2,3,5,6,7,8,11], lambda_min 0.009263).

Runs swap search from EVERY one of the 220 subsets, including the 14 with
no full-rank reference. A shared, per-set-seeded evaluator caches results,
so the whole run costs at most one evaluation per feasible set; each run
also reports how many distinct sets IT needed (its cold-cache cost).

Questions answered:
  - Basin of attraction: how many starts reach the champion?
  - Where do the others stop (1-swap local optima), and how far below?
  - Cost per run: distinct sets bounded / evaluated, number of swaps.
  - Cross-check: evaluator values vs. the sweep's saved values.

With --two-swap, every run uses the two-swap escape (swap_search
two_swap=True; sec 8.7 Day 3): does it carry the 115 stuck starts to the
champion, and at what cost? Output names then end in _two_swap, so the
one-swap results are kept.

Saves (incrementally, after every start):
  results/birdsfoot_swap_validation[_two_swap]_{results.json,summary.txt,starts.png}

Run directly: python -u birdsfoot_swap_validation.py [--two-swap]
"""

import sys, os, json, time, argparse
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name,
)
from swap_search import SetEvaluator, swap_search

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SWEEP_JSON  = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_results.json')
OUT_BASE    = os.path.join(RESULTS_DIR, 'birdsfoot_swap_validation')
OUT_JSON    = OUT_BASE + '_results.json'
OUT_SUMMARY = OUT_BASE + '_summary.txt'
OUT_FIG     = OUT_BASE + '_starts.png'
TWO_SWAP    = False


def set_two_swap(on):
    """Point the outputs at the _two_swap files (keeps the 1-swap results)."""
    global OUT_JSON, OUT_SUMMARY, OUT_FIG, TWO_SWAP
    TWO_SWAP = on
    sfx = '_two_swap' if on else ''
    OUT_JSON, OUT_SUMMARY, OUT_FIG = (OUT_BASE + sfx + '_results.json',
                                      OUT_BASE + sfx + '_summary.txt',
                                      OUT_BASE + sfx + '_starts.png')

K               = 9
SEED            = 31
IMPROVE_REL_TOL = 1e-4
OPT_REL_TOL     = 1e-4
CHAMPION        = [0, 1, 2, 3, 5, 6, 7, 8, 11]

COLOR_MARK = '#2a78d6'      # blue 450
INK_MUTED  = '#6b6b6b'


def spokes(system, active):
    c = {s: 0 for s in ('O-M', 'O-B', 'O-N', 'O-A')}
    for i in active:
        c[birdsfoot_spoke_name(system.couplings[i])] += 1
    return ','.join(f'{s}:{n}' for s, n in c.items())


def write_outputs(system, runs, opt, ev, sweep, elapsed, done):
    reached = [r for r in runs if r['reached_opt']]
    finals = {}
    for r in runs:
        finals.setdefault(tuple(r['active']), []).append(r)

    L = []
    L.append("Swap search validation -- birds-foot, 12 candidate sites, k=9 "
             "(exhaustive ground truth: all 220 subsets)"
             + ("; WITH two-swap escape" if TWO_SWAP else ""))
    L.append("=" * 78)
    L.append(f"Champion: {CHAMPION} ({spokes(system, CHAMPION)}), lambda_min {opt:.6f}")
    L.append(f"Improvement threshold {IMPROVE_REL_TOL:g} relative; 'reached' = within "
             f"{OPT_REL_TOL:g} of the champion.")
    L.append(f"Progress: {len(runs)}/{done} starts.  Elapsed {elapsed:.0f}s; shared cache: "
             f"{ev.n_bound_calls} bound solves, {ev.n_eval_calls} Riemannian evaluations.")
    L.append("")
    L.append(f"RESULT: {len(reached)}/{len(runs)} starts reached the champion "
             f"({100 * len(reached) / max(1, len(runs)):.1f}%).")
    L.append("")
    L.append("Where runs stopped (distinct final sets):")
    for key, rs in sorted(finals.items(), key=lambda kv: -kv[1][0]['lambda_min']):
        lam = rs[0]['lambda_min']
        L.append(f"   {str(list(key)):32s} {spokes(system, key):24s} lambda_min {lam:.6f} "
                 f"({100 * lam / opt:5.1f}% of champion)  <- {len(rs)} start(s)")
    L.append("")
    ns = np.array([r['n_swaps_accepted'] for r in runs])
    ne = np.array([r['n_sets_evaluated'] for r in runs])
    nb = np.array([r['n_sets_bounded'] for r in runs])
    L.append("Cost per run (cold-cache equivalent):")
    L.append(f"   swaps accepted:        median {np.median(ns):.0f}, max {ns.max()}")
    if TWO_SWAP:
        n2s = np.array([r['n_two_swap_searches'] for r in runs])
        n2e = np.array([r['n_two_swap_escapes'] for r in runs])
        n2ev = np.array([sum(p['two_swap']['n_evaluated'] for p in r['passes']
                             if p['two_swap']) for r in runs])
        L.append(f"   2-swap neighborhoods searched: median {np.median(n2s):.0f}, max {n2s.max()}; "
                 f"escapes accepted: {int((n2e > 0).sum())} runs, {int(n2e.sum())} total")
        L.append(f"   Riemannian evaluations inside 2-swap searches (warm cache counts too): "
                 f"median {np.median(n2ev):.0f}, max {n2ev.max()}")
    L.append(f"   distinct sets bounded: median {np.median(nb):.0f}, max {nb.max()} (of 220)")
    L.append(f"   distinct sets evaluated (Riemannian): median {np.median(ne):.0f}, "
             f"max {ne.max()} (of 206 feasible)")
    fails = [r for r in runs if not r['reached_opt']]
    if fails:
        L.append("")
        L.append(f"Starts that did NOT reach the champion ({len(fails)}):")
        for r in sorted(fails, key=lambda r: -r['lambda_min'])[:40]:
            L.append(f"   {str(r['start_active']):32s} {r['start_lambda_min']:.6f} -> "
                     f"{r['lambda_min']:.6f} at {r['active']}  ({r['n_swaps_accepted']} swaps)")
    diffs = []
    for key, sr in sweep.items():
        e = ev._eval_cache.get(key)
        if e is None:
            continue
        if sr['status'] == 'ok' and e['feasible']:
            diffs.append(abs(e['lambda_min'] - sr['lambda_min']) / sr['lambda_min'])
        elif (sr['status'] == 'ok') != e['feasible']:
            diffs.append(float('inf'))
    finite = [d for d in diffs if np.isfinite(d)]
    if finite:
        L.append("")
        L.append(f"Evaluator vs. sweep's saved values (independent seeds): over "
                 f"{len(finite)} sets, median relative difference {np.median(finite):.1e}, "
                 f"max {max(finite):.1e}; {sum(1 for d in diffs if not np.isfinite(d))} "
                 f"feasibility disagreements.")
    summary = "\n".join(L)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(champion=CHAMPION, optimum=opt, n_reached=len(reached),
                       n_starts=len(runs), elapsed_sec=elapsed,
                       runs=[dict(start=r['start_active'],
                                  start_lambda_min=r['start_lambda_min'],
                                  final=r['active'], lambda_min=r['lambda_min'],
                                  reached_opt=bool(r['reached_opt']),
                                  n_swaps_accepted=r['n_swaps_accepted'],
                                  n_sets_evaluated=r['n_sets_evaluated'],
                                  n_sets_bounded=r['n_sets_bounded'],
                                  n_two_swap_searches=r.get('n_two_swap_searches', 0),
                                  n_two_swap_escapes=r.get('n_two_swap_escapes', 0),
                                  passes=r['passes']) for r in runs]),
                  f, indent=1, default=int)
    return summary


def make_figure(runs, opt):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = [r['start_lambda_min'] for r in runs]
    y = [r['lambda_min'] for r in runs]
    ax.scatter(x, y, s=64, color=COLOR_MARK, alpha=0.6, edgecolor='white',
               linewidth=1.5, zorder=3)
    lim = opt * 1.08
    ax.plot([0, lim], [0, lim], color=INK_MUTED, linewidth=1, linestyle=':', zorder=1)
    ax.axhline(opt, color=INK_MUTED, linewidth=1, linestyle='--', zorder=1)
    ax.text(lim, opt, f'champion {opt:.4f} ', color=INK_MUTED, fontsize=8,
            va='bottom', ha='right')
    ax.text(lim * 0.62, lim * 0.56, 'no change (y = x)', color=INK_MUTED, fontsize=8,
            ha='left', va='top')
    n_ok = sum(r['reached_opt'] for r in runs)
    ax.set_xlim(-lim * 0.02, lim)
    ax.set_ylim(-lim * 0.02, lim)
    ax.set_xlabel('λ_min of starting set (0 = rank-deficient)', color='#333333')
    ax.set_ylabel('λ_min after swap search', color='#333333')
    ax.set_title(f'Birds-foot {"1+2" if TWO_SWAP else "1"}-swap search from all {len(runs)} starting sets: '
                 f'{n_ok}/{len(runs)} reach the champion', fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--two-swap', action='store_true',
                    help='use the two-swap escape; outputs end in _two_swap')
    args = ap.parse_args()
    set_two_swap(args.two_swap)

    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    ev = SetEvaluator(system, target_rank, BIRDSFOOT_START_THETAS_DEG, seed=SEED)

    with open(SWEEP_JSON, encoding='utf-8') as f:
        sweep = {tuple(r['subset']): r for r in json.load(f)}
    opt = max(r['lambda_min'] for r in sweep.values() if r['status'] == 'ok')

    starts = [list(s) for s in combinations(range(m), K)]
    t0 = time.perf_counter()
    runs = []
    for n, start in enumerate(starts, 1):
        r = swap_search(ev, start, improve_rel_tol=IMPROVE_REL_TOL, verbose=False,
                        two_swap=TWO_SWAP)
        r['reached_opt'] = r['lambda_min'] >= opt * (1 - OPT_REL_TOL)
        r['n_swaps_accepted'] = sum(1 for p in r['passes'] if p['accepted'])
        runs.append(r)
        print(f"[{n}/{len(starts)}] {start} {r['start_lambda_min']:.6f} -> "
              f"{r['lambda_min']:.6f} ({r['n_swaps_accepted']} swaps"
              f"{', ' + str(r['n_two_swap_escapes']) + ' via 2-swap' if TWO_SWAP else ''}, "
              f"{'ok' if r['reached_opt'] else 'MISS'})  "
              f"cache: {ev.n_eval_calls} evals, {time.perf_counter() - t0:.0f}s", flush=True)
        write_outputs(system, runs, opt, ev, sweep, time.perf_counter() - t0, len(starts))

    summary = write_outputs(system, runs, opt, ev, sweep, time.perf_counter() - t0,
                            len(starts))
    make_figure(runs, opt)
    print("\n" + summary)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
