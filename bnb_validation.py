"""
bnb_validation.py

Validates branch_and_bound.py against exhaustive ground truth on the
two-panel 6-site system (k=3, sec 7.6) and birds-foot 12-site system
(k=9, sec 7.1).

Configurations per system (to show what each piece contributes):
  A  no starting incumbent, no symmetry breaking
  B  no starting incumbent, symmetry breaking
  C  greedy + swap-search incumbent, symmetry breaking   <- the planned setup
  D  (birds-foot only) start from the 96.8% one-swap trap, symmetry breaking

Pass criteria: the search completes, finds the exhaustive optimum (within
1e-4 relative), and its certified upper bound is >= the optimum.

Symmetries are verified before use with branch_and_bound.verify_symmetry
(exact geometry + matching certified bounds) -- a wrong symmetry would
silently discard the optimum. The exhaustive lambda_min values are NOT used
for this: they carry restart noise (mirror pairs differ by up to ~1%),
which made an earlier version of this script reject a valid symmetry.

A single shared evaluator per system caches leaf evaluations across
configurations, so later configurations run faster; per-configuration
counts (nodes, leaves evaluated) are unaffected by the cache.

Saves: results/bnb_validation_{results.json,summary.txt,counts.png}

Run directly: python -u bnb_validation.py
"""

import sys, os, json, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
from two_panel_six_sites import build_six_site_system, START_THETAS_DEG as TP_THETAS
from swap_search import SetEvaluator, swap_search
from branch_and_bound import BranchAndBound, verify_symmetry
from interactive_optimizer import t as PANEL_T

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, 'bnb_validation_results.json')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'bnb_validation_summary.txt')
OUT_FIG     = os.path.join(RESULTS_DIR, 'bnb_validation_counts.png')

SEED    = 31            # same evaluator seed as the swap validations
OPT_TOL = 1e-4

COLORS = {'A': '#9ec5f4', 'B': '#5598e7', 'C': '#1c5cab', 'D': '#0d366b'}


def load_two_panel_truth():
    with open(os.path.join(RESULTS_DIR, 'two_panel_six_sites_results.json'), encoding='utf-8') as f:
        d = json.load(f)
    truth = {tuple(r['active']): (r['lambda_min'] if r['feasible'] else 0.0)
             for r in d['exhaustive']['3']}
    return truth, d['greedy']['active']


def load_birdsfoot_truth():
    with open(os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_results.json'), encoding='utf-8') as f:
        truth = {tuple(r['subset']): (r['lambda_min'] if r['status'] == 'ok' else 0.0)
                 for r in json.load(f)}
    with open(os.path.join(RESULTS_DIR, 'forward_greedy_topology_results.json'), encoding='utf-8') as f:
        greedy = json.load(f)['active']
    return truth, greedy


def noisy_symmetry_diff(perm, truth):
    """Info only: largest relative lambda_min difference between a set and
    its image in the (noisy) exhaustive data."""
    worst = 0.0
    for s, lam in truth.items():
        img = tuple(sorted(perm[i] for i in s))
        lam2 = truth[img]
        if (lam > 0) != (lam2 > 0):
            return float('inf')
        if lam > 0:
            worst = max(worst, abs(lam - lam2) / lam)
    return worst


def run_system(name, system, thetas, k, truth, greedy, symmetries, trap=None):
    target_rank = system.total_dofs - 6
    ev = SetEvaluator(system, target_rank, thetas, seed=SEED)
    opt = max(truth.values())
    lines = [f"{name}: m={len(system.couplings)}, k={k}, exhaustive optimum {opt:.6f}"]

    syms = []
    for label, p, planes in symmetries:
        ok, info = verify_symmetry(ev, p, planes, k)
        lines.append(f"   symmetry '{label}': {'verified' if ok else 'REJECTED'} "
                     f"(geometry: point err {info['point_err']:.1e}, normal err "
                     f"{info['normal_err']:.1e}; bounds over {info['n_sets_checked']} sets: "
                     f"max rel diff {info['bound_max_rel_diff']:.1e}, "
                     f"{info['feasibility_mismatches']} feasibility mismatches; "
                     f"noisy exhaustive lambda diff {noisy_symmetry_diff(p, truth):.1e}, info only)")
        if ok:
            syms.append(p)

    t0 = time.perf_counter()
    sw = swap_search(ev, greedy, verbose=False)
    t_swap = time.perf_counter() - t0
    lines.append(f"   greedy {sorted(greedy)} -> swap search {sw['active']} "
                 f"lambda_min {sw['lambda_min']:.6f} ({t_swap:.0f}s)")

    configs = [('A', None, []), ('B', None, syms), ('C', sw['active'], syms)]
    if trap is not None:
        configs.append(('D', trap, syms))
    out = {}
    for tag, inc, sy in configs:
        print(f"\n[{name}] config {tag}", flush=True)
        bnb = BranchAndBound(ev, k, symmetries=sy, seed=SEED)
        r = bnb.run(incumbent_active=inc)
        r['passed'] = bool(r['complete'] and r['lambda_min'] >= opt * (1 - OPT_TOL)
                           and r['certified_upper_bound'] >= opt * (1 - 1e-9))
        out[tag] = r
        s = r['stats']
        lines.append(
            f"   {tag}: {'PASS' if r['passed'] else 'FAIL'}  found {r['lambda_min']:.6f} "
            f"{r['active']}  start {r['start_lambda_min']:.6f}; certified UB "
            f"{r['certified_upper_bound']:.6f} (gap {r['certified_gap']:.2e}, "
            f"{100 * r['certified_gap'] / max(r['lambda_min'], 1e-12):.3f}%)")
        lines.append(
            f"      nodes {s['nodes']}; pruned: bound {s['pruned_bound']}, symmetry "
            f"{s['pruned_symmetry']}, rank {s['pruned_rank']}, cut {s['pruned_cut']}, "
            f"cardinality {s['pruned_cardinality']}; sdp errors {s['sdp_errors']}")
        lines.append(
            f"      leaves {s['leaves']}: {s['leaves_pruned_bound']} pruned by bound, "
            f"{s['leaves_infeasible']} infeasible, {s['leaves_evaluated']} evaluated; "
            f"incumbent updates {s['incumbent_updates']}; {r['elapsed_sec']:.0f}s")
    lines.append(f"   shared cache: {ev.n_eval_calls} Riemannian evaluations, "
                 f"{ev.n_bound_calls} leaf bound solves")
    return lines, out, opt


def make_figure(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], 'nodes', 'Branches visited'),
                           (axes[1], 'leaves_evaluated', 'Leaves evaluated with Riemannian')):
        labels, vals, cols = [], [], []
        for sysname, cfgs in results.items():
            for tag, r in cfgs.items():
                labels.append(f"{sysname} {tag}")
                vals.append(r['stats'][key])
                cols.append(COLORS[tag])
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, color=cols, height=0.7, edgecolor='white', linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f' {v}', va='center', fontsize=8, color='#333333')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, color='#333333')
        ax.set_title(title, fontsize=10)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='x', alpha=0.25)
    fig.suptitle('Branch-and-bound validation. A: no incumbent, no symmetry · '
                 'B: + symmetry · C: + greedy/swap incumbent · D: trap incumbent',
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main():
    L = ["Branch-and-bound validation (exhaustive ground truth)", "=" * 78,
         "Prototype: depth-first, most-fractional branching, cut counting, generic-rank "
         "feasibility, manual symmetry breaking, prune tolerance 1e-4.", ""]
    results = {}

    truth, greedy = load_two_panel_truth()
    z_plane = ((0., 0., 1.), -PANEL_T / 2)
    y_plane = ((0., 1., 0.), 0.5)
    tp_syms = [('thickness mirror', [3, 4, 5, 0, 1, 2], [z_plane]),
               ('y mirror', [1, 0, 2, 4, 3, 5], [y_plane]),
               ('both', [4, 3, 5, 1, 0, 2], [z_plane, y_plane])]
    lines, out, _ = run_system('two-panel', build_six_site_system(), TP_THETAS, 3,
                               truth, greedy, tp_syms)
    L += lines + [""]
    results['two-panel'] = out

    truth, greedy = load_birdsfoot_truth()
    perm = list(range(12))
    for j in range(3):
        perm[3 + j], perm[9 + j] = 9 + j, 3 + j
    lines, out, _ = run_system('birds-foot',
                               build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)),
                               BIRDSFOOT_START_THETAS_DEG, 9, truth, greedy,
                               [('O-B <-> O-A mirror', perm, [((1., 0., 0.), 0.5)])],
                               trap=[0, 1, 2, 4, 5, 7, 8, 10, 11])
    L += lines
    results['birds-foot'] = out

    summary = "\n".join(L)
    print("\n" + summary)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({s: {t: {k: v for k, v in r.items()} for t, r in c.items()}
                   for s, c in results.items()}, f, indent=1, default=float)
    make_figure(results)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()
