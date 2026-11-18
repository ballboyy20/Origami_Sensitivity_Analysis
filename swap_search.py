"""
swap_search.py

Swap-based local search (Fedorov-exchange style) over coupling active
sets -- Phase 0 of the roadmap (KINEMATIC_COUPLING_SUMMARY.md sec 8.4).
Meant to run after forward_greedy() (or from any starting set), and to
supply a strong incumbent for a later branch-and-bound.

A SWAP removes one active coupling and adds one inactive one, so the set
size -- and, with all-V-groove candidates, the row count -- stays exact.
Each pass:

  1. Screen every swap (k * (m - k) of them, minus any that would remove
     a `fixed` coupling) with the certified bound of the swapped COMPLETE
     set (bound_node with nothing undecided). A swap whose bound is <= the
     current achieved lambda_min (times 1 + improve_rel_tol) cannot
     improve the design -- discarded rigorously. Swaps with no full-rank
     reference are discarded as infeasible.
  2. Evaluate survivors with bound_and_evaluate (certified bound +
     Riemannian multistart) in DESCENDING bound order, stopping as soon
     as the next swap's bound can't beat the best achieved value found so
     far this pass. This returns the true best swap in the neighborhood
     (best improvement) while usually evaluating only a few.
  3. Accept the best swap if it improves lambda_min by more than
     improve_rel_tol (relative); otherwise stop -- a 1-swap local optimum.

Why only complete sets: forward greedy's failures (sec 7.6-7.7) came from
the bound being flat across PARTIAL sets. For complete sets the bound is
usually near-exact and the achieved value is available directly, so every
early greedy choice gets revisited with meaningful scores.

Reproducibility / noise: each set's bound reference and Riemannian
restarts are seeded from the set itself (np.random.default_rng([seed,
*sorted_set])), so a given set always evaluates to the same lambda_min no
matter which path reaches it, and results are cached. The improvement
threshold (default 1e-4 relative) sits above the best-of-5 restart noise
measured in sec 7.2 (<= 1.1e-5), so noise can't drive a swap.

No global guarantee: the result is a 1-swap local optimum. Combined with
a root bound it gives a bracket, not a proof.

Two-swap escape (two_swap=True; KINEMATIC_COUPLING_SUMMARY.md sec 8.4 /
sec 8.7 Day 3): when a pass finds no improving 1-swap, try every 2-swap
(remove 2 non-fixed active couplings, add 2 inactive ones) with the same
bound screening and best-bound-first evaluation. If one improves, accept
it and go back to 1-swaps; otherwise stop at a 2-swap local optimum. The
main birds-foot trap ([0,1,2,4,5,7,8,10,11], 96.8%) is exactly 2 swaps
from the champion (sec 7.9). Neighborhood size C(k-f, 2) * C(m-k, 2):
108 on birds-foot (k = 9, m = 12), 3,780 at 24 sites. With
two_swap=False the search is identical to before (same moves, same
order, same pass records apart from the added 'two_swap' key).
"""

import sys, os, time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sdp_topology_bound import bound_node, build_coupling_registry
from bound_and_evaluate import bound_and_evaluate
from riemannian_gradient_sampling import MultiplicityError


class SetEvaluator:
    """Cached, per-set-seeded bound + achieved-value evaluation for one
    CouplingSystem. Shared across swap-search runs on the same system so
    repeated sets (common across many starts) are evaluated once."""

    def __init__(self, system, target_rank, base_thetas_deg, registry=None,
                 n_restarts=5, max_ref_tries=300, seed=0, length_scale=1.0,
                 rows=None, sites=None):
        """rows, sites: mixed coupling types (coupling_types.TypedCandidates
        .rows / .site_of); a set then needs exactly target_rank rows and at
        most one candidate per site to be 'bounded'."""
        self.system = system
        self.rows = rows
        self.sites = sites
        self.m = len(system.couplings)
        self.target_rank = target_rank
        self.base_thetas_deg = np.asarray(base_thetas_deg, dtype=float)
        self.n_restarts = n_restarts
        self.max_ref_tries = max_ref_tries
        self.seed = seed
        self.length_scale = length_scale
        if registry is None:
            registry = build_coupling_registry(
                system, np.radians(self.base_thetas_deg), list(range(self.m)),
                length_scale)
        self.registry = registry
        self._bound_cache = {}
        self._eval_cache = {}
        self._screen_cache = {}    # one-restart results that fell below a floor
        self.n_bound_calls = 0
        self.n_eval_calls = 0      # Riemannian evaluations started (incl. screened)
        self.n_screened = 0        # of those, stopped after one restart

    def _rng(self, key, salt):
        return np.random.default_rng([self.seed, salt, *key])

    def bound(self, active):
        """(status, bound, reference) for the complete set `active`."""
        key = tuple(sorted(int(i) for i in active))
        if (key not in self._bound_cache and self.sites is not None
                and len({int(self.sites[i]) for i in key}) < len(key)):
            self._bound_cache[key] = ('infeasible', None, None)   # two at one site
        if key not in self._bound_cache:
            self.n_bound_calls += 1
            r = bound_node(self.system, forced_active=list(key),
                           forced_inactive=[i for i in range(self.m) if i not in key],
                           undecided=[], k=len(key), length_scale=self.length_scale,
                           target_rank=self.target_rank,
                           base_thetas_deg=self.base_thetas_deg,
                           registry=self.registry, max_ref_tries=self.max_ref_tries,
                           rng=self._rng(key, 1), rows=self.rows, sites=self.sites)
            self._bound_cache[key] = (r['status'], r['bound'], r['reference'])
        return self._bound_cache[key]

    def evaluate(self, active, stop_below=None):
        """dict(lambda_min, theta, t, gap, screened) for `active`, or
        lambda_min=0.0 with feasible=False if the set can't reach full rank.

        stop_below: one-restart screen (sec 7.15). If the first restart
        achieves less than this, return that restart's result with
        screened=True (cached separately, never as a full evaluation).
        Otherwise the full n_restarts evaluation runs, identical to an
        unscreened call (same per-set rng; see riemannian_multistart)."""
        key = tuple(sorted(int(i) for i in active))
        if key in self._eval_cache:
            return self._eval_cache[key]
        if (stop_below is not None and key in self._screen_cache
                and self._screen_cache[key]['lambda_min'] < stop_below):
            return self._screen_cache[key]
        status, t, ref = self.bound(key)
        if status != 'bounded':
            self._eval_cache[key] = dict(feasible=False, lambda_min=0.0,
                                         theta=None, t=None, gap=None, screened=False)
            return self._eval_cache[key]
        self.n_eval_calls += 1
        try:
            r = bound_and_evaluate(self.system, list(key),
                                   length_scale=self.length_scale,
                                   target_rank=self.target_rank,
                                   ref_thetas=ref, n_restarts=self.n_restarts,
                                   rng=self._rng(key, 2), stop_below=stop_below)
        except MultiplicityError:
            # Record and move on rather than abort the search (same
            # convention as bound_and_evaluate_sweep.py).
            self._eval_cache[key] = dict(feasible=False, lambda_min=0.0,
                                         theta=None, t=t, gap=None,
                                         error='multiplicity', screened=False)
            return self._eval_cache[key]
        except RuntimeError as e:
            # certified-bound SDP failed (e.g. an inaccurate SCS fallback):
            # record and move on; an overnight search shouldn't die on one set
            self._eval_cache[key] = dict(feasible=False, lambda_min=0.0,
                                         theta=None, t=t, gap=None,
                                         error=f'sdp: {e}'[:300], screened=False)
            return self._eval_cache[key]
        res = dict(feasible=True, lambda_min=r['lambda_min'], theta=r['theta'],
                   t=r['t'], gap=r['gap'], screened=r['screened'])
        if r['screened']:
            self.n_screened += 1
            self._screen_cache[key] = res
        else:
            self._eval_cache[key] = res
        return res


def _best_move(evaluator, current, moves, threshold, bounded, evaluated):
    """Best-improvement step over `moves` (iterable of (outs, ins) tuples):
    bound-screen every move, then evaluate survivors best-bound-first until
    no remaining bound can beat the best achieved value. Returns
    (best, stats): best is (result, outs, ins, trial) or None."""
    candidates, n_moves, n_infeasible, n_pruned = [], 0, 0, 0
    for outs, ins in moves:
        n_moves += 1
        trial = sorted([c for c in current if c not in outs] + list(ins))
        status, b, _ = evaluator.bound(trial)
        bounded.add(tuple(trial))
        if status != 'bounded':
            n_infeasible += 1
        elif b <= threshold:
            n_pruned += 1
        else:
            candidates.append((b, outs, ins, trial))

    candidates.sort(key=lambda c: -c[0])
    best, n_evaluated = None, 0
    for b, outs, ins, trial in candidates:
        best_lam = best[0]['lambda_min'] if best else threshold
        if b <= best_lam:
            n_pruned += 1      # can't beat the best found this pass
            continue
        n_evaluated += 1
        r = evaluator.evaluate(trial)
        evaluated.add(tuple(trial))
        if r['feasible'] and r['lambda_min'] > best_lam:
            best = (r, outs, ins, trial)
    return best, dict(n_swaps=n_moves, n_infeasible=n_infeasible,
                      n_pruned_by_bound=n_pruned, n_evaluated=n_evaluated)


def swap_search(evaluator, start_active, fixed=(), improve_rel_tol=1e-4,
                max_passes=50, verbose=True, two_swap=False):
    """
    Best-improvement 1-swap local search from `start_active` (see module
    docstring). `fixed` couplings (e.g. mandatory leaf couplings) are never
    removed. A rank-deficient start is allowed (lambda_min 0): any feasible
    swap then counts as an improvement.

    two_swap: when no 1-swap improves, try every 2-swap before stopping
    (module docstring). An accepted 2-swap counts as a pass.

    Returns dict:
        'active', 'lambda_min', 'theta', 't'   -- final set and its evaluation
        'start_active', 'start_lambda_min'
        'passes' : list of per-pass dicts (n_swaps, n_infeasible,
                   n_pruned_by_bound, n_evaluated -- all for the 1-swap
                   neighborhood; 'two_swap': the same four counts for the
                   2-swap neighborhood if it was searched this pass, else
                   None; accepted move or None, with 'kind' 'one_swap'
                   (out/into ints) or 'two_swap' (out/into 2-lists);
                   lambda_min after)
        'converged' : True if stopped at a local optimum (1-swap, or
                   2-swap when two_swap=True)
        'n_two_swap_searches', 'n_two_swap_escapes' : 2-swap neighborhoods
                   searched / 2-swaps accepted
        'n_sets_evaluated', 'n_sets_bounded' : distinct sets this run needed
                   (cost of the run on a cold cache, even if the shared
                   evaluator already had them)
        'elapsed_sec'
    """
    t0 = time.perf_counter()
    m = evaluator.m
    fixed = set(int(i) for i in fixed)
    current = sorted(int(i) for i in start_active)
    evaluated, bounded = {tuple(current)}, {tuple(current)}
    cur = evaluator.evaluate(current)
    start_lam = cur['lambda_min']
    passes = []
    converged = False
    n_two_searches = n_two_escapes = 0

    for _ in range(max_passes):
        threshold = cur['lambda_min'] * (1 + improve_rel_tol) if cur['lambda_min'] > 0 else 0.0
        removable = [c for c in current if c not in fixed]
        inactive = [c for c in range(m) if c not in current]
        one_moves = (((o,), (i,)) for o in removable for i in inactive)
        best, stats = _best_move(evaluator, current, one_moves, threshold,
                                 bounded, evaluated)
        kind, stats2 = 'one_swap', None
        if best is None and two_swap:
            n_two_searches += 1
            two_moves = ((o, i) for o in combinations(removable, 2)
                         for i in combinations(inactive, 2))
            best, stats2 = _best_move(evaluator, current, two_moves, threshold,
                                      bounded, evaluated)
            kind = 'two_swap'

        accepted = None
        if best is not None:
            r, outs, ins, trial = best
            if kind == 'one_swap':
                out_v, in_v = outs[0], ins[0]
            else:
                out_v, in_v = list(outs), list(ins)
                n_two_escapes += 1
            accepted = dict(kind=kind, out=out_v, into=in_v,
                            lambda_before=cur['lambda_min'], lambda_after=r['lambda_min'])
            current, cur = trial, r
        passes.append(dict(**stats, two_swap=stats2, accepted=accepted,
                           lambda_min=cur['lambda_min']))
        if verbose:
            msg = (f"  pass {len(passes)}: {stats['n_swaps']} swaps, "
                   f"{stats['n_infeasible']} infeasible, {stats['n_pruned_by_bound']} "
                   f"pruned by bound, {stats['n_evaluated']} evaluated")
            if stats2 is not None:
                msg += (f"; 2-swaps: {stats2['n_swaps']}, {stats2['n_infeasible']} "
                        f"infeasible, {stats2['n_pruned_by_bound']} pruned, "
                        f"{stats2['n_evaluated']} evaluated")
            msg += " -> "
            msg += (f"{'2-swap' if kind == 'two_swap' else 'swap'} {accepted['out']}->"
                    f"{accepted['into']}, lambda_min {accepted['lambda_before']:.6f} -> "
                    f"{accepted['lambda_after']:.6f}"
                    if accepted else
                    f"no improving {'1- or 2-swap' if two_swap else 'swap'} (local optimum)")
            print(msg, flush=True)
        if accepted is None:
            converged = True
            break

    return dict(active=current, lambda_min=cur['lambda_min'], theta=cur['theta'],
                t=cur['t'], start_active=sorted(int(i) for i in start_active),
                start_lambda_min=start_lam, passes=passes, converged=converged,
                n_two_swap_searches=n_two_searches, n_two_swap_escapes=n_two_escapes,
                n_sets_evaluated=len(evaluated), n_sets_bounded=len(bounded),
                elapsed_sec=time.perf_counter() - t0)
