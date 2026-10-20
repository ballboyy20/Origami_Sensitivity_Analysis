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
"""

import sys, os, time

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
                 n_restarts=5, max_ref_tries=300, seed=0, length_scale=1.0):
        self.system = system
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
        self.n_bound_calls = 0
        self.n_eval_calls = 0

    def _rng(self, key, salt):
        return np.random.default_rng([self.seed, salt, *key])

    def bound(self, active):
        """(status, bound, reference) for the complete set `active`."""
        key = tuple(sorted(int(i) for i in active))
        if key not in self._bound_cache:
            self.n_bound_calls += 1
            r = bound_node(self.system, forced_active=list(key),
                           forced_inactive=[i for i in range(self.m) if i not in key],
                           undecided=[], k=len(key), length_scale=self.length_scale,
                           target_rank=self.target_rank,
                           base_thetas_deg=self.base_thetas_deg,
                           registry=self.registry, max_ref_tries=self.max_ref_tries,
                           rng=self._rng(key, 1))
            self._bound_cache[key] = (r['status'], r['bound'], r['reference'])
        return self._bound_cache[key]

    def evaluate(self, active):
        """dict(lambda_min, theta, t, gap) for `active`, or lambda_min=0.0
        with feasible=False if the set can't reach full rank."""
        key = tuple(sorted(int(i) for i in active))
        if key not in self._eval_cache:
            status, t, ref = self.bound(key)
            if status != 'bounded':
                self._eval_cache[key] = dict(feasible=False, lambda_min=0.0,
                                             theta=None, t=None, gap=None)
            else:
                self.n_eval_calls += 1
                try:
                    r = bound_and_evaluate(self.system, list(key),
                                           length_scale=self.length_scale,
                                           target_rank=self.target_rank,
                                           ref_thetas=ref, n_restarts=self.n_restarts,
                                           rng=self._rng(key, 2))
                    self._eval_cache[key] = dict(feasible=True,
                                                 lambda_min=r['lambda_min'],
                                                 theta=r['theta'], t=r['t'], gap=r['gap'])
                except MultiplicityError:
                    # Record and move on rather than abort the search (same
                    # convention as bound_and_evaluate_sweep.py).
                    self._eval_cache[key] = dict(feasible=False, lambda_min=0.0,
                                                 theta=None, t=t, gap=None,
                                                 error='multiplicity')
        return self._eval_cache[key]


def swap_search(evaluator, start_active, fixed=(), improve_rel_tol=1e-4,
                max_passes=50, verbose=True):
    """
    Best-improvement 1-swap local search from `start_active` (see module
    docstring). `fixed` couplings (e.g. mandatory leaf couplings) are never
    removed. A rank-deficient start is allowed (lambda_min 0): any feasible
    swap then counts as an improvement.

    Returns dict:
        'active', 'lambda_min', 'theta', 't'   -- final set and its evaluation
        'start_active', 'start_lambda_min'
        'passes' : list of per-pass dicts (n_swaps, n_infeasible,
                   n_pruned_by_bound, n_evaluated, accepted swap or None,
                   lambda_min after)
        'converged' : True if stopped at a 1-swap local optimum
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

    for _ in range(max_passes):
        threshold = cur['lambda_min'] * (1 + improve_rel_tol) if cur['lambda_min'] > 0 else 0.0
        candidates, n_swaps, n_infeasible, n_pruned = [], 0, 0, 0
        for out_c in current:
            if out_c in fixed:
                continue
            for in_c in range(m):
                if in_c in current:
                    continue
                n_swaps += 1
                trial = sorted([c for c in current if c != out_c] + [in_c])
                status, b, _ = evaluator.bound(trial)
                bounded.add(tuple(trial))
                if status != 'bounded':
                    n_infeasible += 1
                elif b <= threshold:
                    n_pruned += 1
                else:
                    candidates.append((b, out_c, in_c, trial))

        candidates.sort(key=lambda c: -c[0])
        best, n_evaluated = None, 0
        for b, out_c, in_c, trial in candidates:
            best_lam = best[0]['lambda_min'] if best else threshold
            if b <= best_lam:
                n_pruned += 1      # can't beat the best found this pass
                continue
            n_evaluated += 1
            r = evaluator.evaluate(trial)
            evaluated.add(tuple(trial))
            if r['feasible'] and r['lambda_min'] > best_lam:
                best = (r, out_c, in_c, trial)

        accepted = None
        if best is not None:
            r, out_c, in_c, trial = best
            accepted = dict(out=out_c, into=in_c, lambda_before=cur['lambda_min'],
                            lambda_after=r['lambda_min'])
            current, cur = trial, r
        passes.append(dict(n_swaps=n_swaps, n_infeasible=n_infeasible,
                           n_pruned_by_bound=n_pruned, n_evaluated=n_evaluated,
                           accepted=accepted, lambda_min=cur['lambda_min']))
        if verbose:
            msg = (f"  pass {len(passes)}: {n_swaps} swaps, {n_infeasible} infeasible, "
                   f"{n_pruned} pruned by bound, {n_evaluated} evaluated -> ")
            msg += (f"swap {accepted['out']}->{accepted['into']}, lambda_min "
                    f"{accepted['lambda_before']:.6f} -> {accepted['lambda_after']:.6f}"
                    if accepted else "no improving swap (local optimum)")
            print(msg)
        if accepted is None:
            converged = True
            break

    return dict(active=current, lambda_min=cur['lambda_min'], theta=cur['theta'],
                t=cur['t'], start_active=sorted(int(i) for i in start_active),
                start_lambda_min=start_lam, passes=passes, converged=converged,
                n_sets_evaluated=len(evaluated), n_sets_bounded=len(bounded),
                elapsed_sec=time.perf_counter() - t0)
