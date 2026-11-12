"""
birdsfoot24_scale_test.py

Day-2 scale test (KINEMATIC_COUPLING_SUMMARY.md sec 8.7): birds-foot with 6
candidate sites per spoke (m = 24, k = 9, sec 7.7). No exhaustive ground
truth (C(24,9) ~ 1.3M sets). Known facts: the old 3-site champion maps to
[0,1,2,6,8,12,13,14,20] at 0.009263, so the optimum is >= that; the root
bound is 0.014676; forward greedy found only 0.005106.

Three independent modes, meant to run as three processes at once:

  swap-greedy    one-swap search from greedy's set [0,2,5,11,12,14,17,19,20]
  swap-champion  one-swap search from the old champion
  bnb            branch-and-bound, mirror pairs first, verified symmetries,
                 old champion as starting incumbent

Go/no-go evidence for B&B at scale: nodes, Riemannian evaluations,
certified gap, time (sec 8.7 Day 2).

Symmetries (each verified with verify_symmetry before use):
  spoke mirror      O-B <-> O-A (12-site perm 3..5 <-> 9..11), plane x = 0.5
  thickness mirror  site j <-> j+3 on every spoke, plane z = -t/2
  both

Saves: results/birdsfoot24_<mode>_{results.json,summary.txt}

Run: python -u birdsfoot24_scale_test.py <mode> [--max-hours H]
"""

import sys, os, json, time, argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from birdsfoot_six_sites import (build_six_site_birdsfoot, START_THETAS_DEG,
                                 map_12_to_24, MID_Z, OLD_CHAMPION_12)
from swap_search import SetEvaluator, swap_search
from branch_and_bound import BranchAndBound, verify_symmetry

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 31                     # same evaluator seed as the other validations
K = 9
GREEDY_24 = [0, 2, 5, 11, 12, 14, 17, 19, 20]      # sec 7.7
CHAMPION_24 = map_12_to_24(OLD_CHAMPION_12)        # [0,1,2,6,8,12,13,14,20]
ROOT_BOUND = 0.014676                              # sec 7.7


def symmetries():
    spoke = list(range(24))
    for j in range(6):
        spoke[6 + j], spoke[18 + j] = 18 + j, 6 + j
    thick = [6 * (i // 6) + (i % 6 + 3) % 6 for i in range(24)]
    both = [thick[spoke[i]] for i in range(24)]
    sp_plane = ((1., 0., 0.), 0.5)
    th_plane = ((0., 0., 1.), MID_Z)
    return [('spoke mirror', spoke, [sp_plane]),
            ('thickness mirror', thick, [th_plane]),
            ('both', both, [sp_plane, th_plane])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['swap-greedy', 'swap-champion', 'bnb'])
    ap.add_argument('--max-hours', type=float, default=8.0)
    ap.add_argument('--incumbent', type=int, nargs='+', default=None,
                    help="bnb starting design (default: old champion); also "
                         "suffixes the output file name with '_inc'")
    args = ap.parse_args()

    system = build_six_site_birdsfoot()
    target_rank = system.total_dofs - 6
    ev = SetEvaluator(system, target_rank, START_THETAS_DEG, seed=SEED)
    lines = [f"Birds-foot 24 sites, k={K}: {args.mode}", "=" * 78]

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.perf_counter()
    if args.mode.startswith('swap'):
        start = GREEDY_24 if args.mode == 'swap-greedy' else CHAMPION_24
        r = swap_search(ev, start, verbose=True)
        log(f"start {start}: lambda_min {r['start_lambda_min']:.6f}")
        for i, p in enumerate(r['passes']):
            log(f"   pass {i + 1}: {p}")
        log(f"final {r['active']}: lambda_min {r['lambda_min']:.6f} "
            f"({100 * r['lambda_min'] / 0.009263:.1f}% of old champion), "
            f"converged {r['converged']}")
        log(f"sets bounded {r['n_sets_bounded']}, evaluated {r['n_sets_evaluated']}; "
            f"{r['elapsed_sec']:.0f}s")
        out = r
    else:
        syms = []
        for label, p, planes in symmetries():
            ok, info = verify_symmetry(ev, p, planes, K, n_sample_sets=300, seed=SEED)
            log(f"symmetry '{label}': {'verified' if ok else 'REJECTED'} {info}")
            if ok:
                syms.append(p)
        log(f"symmetry verification {time.perf_counter() - t0:.0f}s")
        bnb = BranchAndBound(ev, K, symmetries=syms, seed=SEED,
                             branch_rule='symmetry_first')
        r = bnb.run(incumbent_active=args.incumbent or CHAMPION_24, max_sec=3600 * args.max_hours,
                    log_every=25)
        s = r['stats']
        gap = 100 * r['certified_gap'] / max(r['lambda_min'], 1e-12)
        log(f"complete {r['complete']}; found {r['lambda_min']:.6f} {r['active']} "
            f"(start {r['start_lambda_min']:.6f})")
        log(f"certified UB {r['certified_upper_bound']:.6f} (gap {gap:.2f}%; "
            f"root bound {ROOT_BOUND})")
        log(f"nodes {s['nodes']}; pruned: bound {s['pruned_bound']}, symmetry "
            f"{s['pruned_symmetry']}, rank {s['pruned_rank']}, cut {s['pruned_cut']}, "
            f"cardinality {s['pruned_cardinality']}; sdp errors {s['sdp_errors']}")
        log(f"leaves {s['leaves']}: {s['leaves_pruned_bound']} pruned by bound, "
            f"{s['leaves_infeasible']} infeasible, {s['leaves_evaluated']} evaluated; "
            f"incumbent updates {s['incumbent_updates']}; {r['elapsed_sec']:.0f}s")
        loose = sorted(r['evaluated_leaves'], key=lambda d: -d['bound'])[:10]
        log("loosest evaluated leaves (bound, achieved):")
        for d in loose:
            log(f"   {d['active']}: {d['bound']:.6f}, {d['lambda_min']:.6f}")
        out = r
    log(f"total {time.perf_counter() - t0:.0f}s; Riemannian evaluations "
        f"{ev.n_eval_calls}, leaf bound solves {ev.n_bound_calls}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, f"birdsfoot24_{args.mode.replace('-', '_')}"
                        + ('_inc' if args.incumbent else ''))
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else float(o))
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {base}_{{results.json,summary.txt}}")


if __name__ == "__main__":
    main()
