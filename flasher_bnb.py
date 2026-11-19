"""
flasher_bnb.py

Coupling selection on the whole flasher (full_flasher_system.py: hub + 5
gores, 26 bodies, target rank 150), cyclically symmetric designs with tied
angles. Parts (argv[1]; each saves its own files, so both can run at once):

  allv    All V-grooves: choose 15 of the 24 orbits (150 rows), 1.3M sets.
          Incumbent: one-swap search from a random full-rank set, then B&B
          (screen f = 0.8, bound margin 3e-4), capped at --max-hours.
  mixed   Flat / V-groove / socket per site (72 candidates), exactly 150
          rows (30 per gore). Incumbent: the same all-V swap result (each
          part recomputes it; deterministic), lifted to typed candidates,
          then B&B with the same settings. Likely stops at the cap; the
          incumbent trace records how good it got.

Every SDP uses MOSEK (sdp_theta_optimizer.set_default_solver): on this
system Clarabel fails and its SCS fallback is ~15 min per solve and badly
wrong (flasher_timing.py, 2026-11-18). The run aborts at start if MOSEK
can't solve a test problem (e.g. no license), since without bounds B&B
can't prune anything.

Options: --max-hours H (default 13), --threads N (MOSEK threads, default 7;
set OMP/OPENBLAS/MKL_NUM_THREADS to match), --frac F (screen, default 0.8),
--swap-passes P (default 10; 0 = skip swap, start from the random set).

Saves: results/flasher_bnb_<part>_{summary.txt,results.json} and a UTF-8
progress log results/flasher_bnb_<part>.log.

Run: python -u flasher_bnb.py <allv|mixed> [options]
"""
import sys, os, json, time, argparse
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from full_flasher_system import (build_flasher_system, START_THETAS_DEG, N_SITES,
                                 describe_counts, CREASE_NAMES)
from coupling_types import TypedCandidates
from swap_search import SetEvaluator, swap_search
from branch_and_bound import BranchAndBound, _rank_full
from sdp_theta_optimizer import set_default_solver
from screen_validation import Tee, bnb_line

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 31
MARGIN = 3e-4
K_ALLV = 15


def check_mosek(threads):
    x = cp.Variable((2, 2), symmetric=True)
    p = cp.Problem(cp.Maximize(cp.trace(x)), [x << np.eye(2)])
    try:
        p.solve(solver=cp.MOSEK, mosek_params={'MSK_IPAR_NUM_THREADS': threads})
    except Exception as e:
        sys.exit(f"MOSEK unavailable ({type(e).__name__}: {e}); aborting -- B&B "
                 f"without working bounds can't prune.")
    if p.status != 'optimal':
        sys.exit(f"MOSEK test solve returned {p.status}; aborting.")


def allv_incumbent(args, log):
    """Random full-rank 15-orbit start, then one-swap search (all-V)."""
    system = build_flasher_system()
    tr = system.total_dofs - 6
    ev = SetEvaluator(system, tr, START_THETAS_DEG, seed=SEED)
    rng = np.random.default_rng(SEED)
    while True:
        start = sorted(rng.choice(N_SITES, K_ALLV, replace=False).tolist())
        if _rank_full(system, set(start), tr, rng.uniform(0, np.pi, N_SITES), 1.0):
            break
    t0 = time.perf_counter()
    r0 = ev.evaluate(start)
    log(f"Random full-rank start {start}: lambda_min {r0['lambda_min']:.6e} "
        f"({time.perf_counter() - t0:.0f}s)")
    if args.swap_passes <= 0:
        return system, ev, start, r0['lambda_min'], None
    sw = swap_search(ev, start, max_passes=args.swap_passes, verbose=True)
    dt = time.perf_counter() - t0
    log(f"One-swap search: {sw['active']} lambda_min {sw['lambda_min']:.6e} "
        f"({describe_counts(sw['active'])}); {dt:.0f}s, "
        f"{ev.n_bound_calls} bound solves, {ev.n_eval_calls} evaluations")
    return system, ev, sw['active'], sw['lambda_min'], sw


def run_bnb(bnb, inc, args):
    return bnb.run(incumbent_active=inc, max_nodes=10 ** 9,
                   max_sec=args.max_hours * 3600, log_every=25)


def summarize(L, r, describe):
    L += bnb_line("B&B", r)
    s = r['stats']
    L.append(f"      pruned bound/rank/cut/cardinality {s['pruned_bound']}/{s['pruned_rank']}/"
             f"{s['pruned_cut']}/{s['pruned_cardinality']}; leaves {s['leaves']} "
             f"(pruned by bound {s['leaves_pruned_bound']}, infeasible "
             f"{s['leaves_infeasible']}, evaluated {s['leaves_evaluated']}); SDP errors: "
             f"nodes {s['sdp_errors']}, leaves {s['leaves_sdp_errors']}; node bounds at "
             f"relaxed tolerance {s['sdp_relaxed']}")
    if r['active'] is not None:
        L.append(f"   best design: {describe(r['active'])}")
    L.append("   incumbent history:")
    for u in r['incumbent_trace']:
        L.append(f"      {u['lambda_min']:.6e} {describe(u['active'])}, node {u['node']}, "
                 f"{u['elapsed_sec']:.0f}s")


def save(part, L, results):
    summary = "\n".join(L)
    print("\n" + summary, flush=True)
    base = os.path.join(RESULTS_DIR, f'flasher_bnb_{part}')
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else (o if isinstance(o, str) else float(o)))
    print(f"Saved: {base}_summary.txt / _results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['allv', 'mixed'])
    ap.add_argument('--max-hours', type=float, default=13.0)
    ap.add_argument('--threads', type=int, default=7)
    ap.add_argument('--frac', type=float, default=0.8)
    ap.add_argument('--swap-passes', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, f'flasher_bnb_{args.part}.log'))
    check_mosek(args.threads)
    set_default_solver(cp.MOSEK, mosek_params={'MSK_IPAR_NUM_THREADS': args.threads})

    L = [f"Full flasher B&B, part '{args.part}': hub + 5 gores, 26 bodies, target rank 150, "
         f"24 orbit sites (cyclic, tied angles); MOSEK ({args.threads} threads), bound margin "
         f"{MARGIN:.0e}, screen f = {args.frac}, cap {args.max_hours} h",
         f"Creases per gore (3 sites each, site = 3 * crease + j): {', '.join(CREASE_NAMES)}",
         "=" * 78]
    print("\n".join(L), flush=True)

    def log(s):
        print(s, flush=True)
        L.append(s)

    system, ev, inc, inc_lam, sw = allv_incumbent(args, log)
    results = dict(part=args.part, incumbent_start=inc, incumbent_lambda=inc_lam,
                   swap=None if sw is None else {k: v for k, v in sw.items()
                                                 if k in ('active', 'lambda_min', 'start_active',
                                                          'start_lambda_min', 'passes',
                                                          'converged', 'n_sets_evaluated',
                                                          'n_sets_bounded')})

    if args.part == 'allv':
        bnb = BranchAndBound(ev, k=K_ALLV, seed=SEED, verbose=True, screen_frac=args.frac,
                             bound_margin=MARGIN)
        r = run_bnb(bnb, inc, args)
        summarize(L, r, lambda a: f"{a} ({describe_counts(a)})")
    else:
        tc = TypedCandidates(system)
        tr = system.total_dofs - 6
        evm = SetEvaluator(tc.system, tr, tc.base_thetas_deg, seed=SEED,
                           rows=tc.rows, sites=tc.site_of)
        inc_m = tc.from_site_kinds({s: 'vgroove' for s in inc})
        log(f"Mixed: {tc.m} candidates; incumbent (all-V, lifted) "
            f"{tc.describe(inc_m, 3)}")
        bnb = BranchAndBound(evm, rows=tc.rows, site_of=tc.site_of, seed=SEED, verbose=True,
                             screen_frac=args.frac, bound_margin=MARGIN)
        r = run_bnb(bnb, inc_m, args)
        summarize(L, r, lambda a: tc.describe(a, 3))
        results['incumbent_start_typed'] = inc_m

    results['bnb'] = r
    save(args.part, L, results)


if __name__ == "__main__":
    main()
