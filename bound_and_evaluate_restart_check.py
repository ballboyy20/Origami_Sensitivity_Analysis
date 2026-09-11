"""
bound_and_evaluate_restart_check.py

Follow-up to bound_and_evaluate_sweep.py: of the 220 birds-foot size-9
subsets swept there with n_restarts=5, 29 showed "meaningful" restart
disagreement (max - min achieved > 1e-4 across the 5 restarts) -- not a
soundness problem (the certification bound held throughout), but an open
question about whether 5 restarts is enough to trust the reported
achieved value as close to the true topology optimum.

This reruns exactly those 29 subsets with n_restarts=7, using the SAME
per-subset seed as the original sweep.

NOTE on what this comparison actually is (checked directly, not assumed):
riemannian_multistart()'s n_restarts-1 random STARTING POINTS are indeed
drawn up front, in order, before any optimize_theta_riemannian() call --
so the first 4 random starting points at n_restarts=7 are the same VALUES
as at n_restarts=5. But every restart's own optimization then consumes a
variable, unpredictable number of draws from that SAME shared RNG object
(one sampling perturbation per outer iteration, and iteration count isn't
fixed) -- so requesting 2 more restarts shifts the RNG state consumed
during restart 1's own optimization already (more upfront draws before
restart 1 begins), which cascades: restart 1 itself follows a different
internal sampling path, consumes a different number of draws, and every
subsequent restart's OUTCOME (not its starting theta) diverges from the
n_restarts=5 run despite nominally-identical starting points. Verified
directly on one subset before trusting this script's results.

So this is NOT a strict "same 5 plus 2 new" superset experiment -- it's a
fresh, independently-seeded 7-restart run (related to, but not a superset
of, the original 5). Still answers the question that matters here: does
best-of-7 exceed best-of-5, or has the achieved value already converged?
(A worthwhile follow-up hygiene fix, not done here: give each restart its
own independent child RNG in riemannian_multistart() so restarts stop
being order/count-dependent on each other -- out of scope for this
specific check, which only needed to answer the convergence question.)

Answers: does the best-of-N achieved value keep improving as N grows
(evidence 5 restarts wasn't enough), or has it already converged (evidence
the wide spread reflects some restarts landing in worse local optima,
while the BEST one found was already close to the true optimum even at
N=5)?

Saves: results/bound_and_evaluate_restart_check_results.json/csv,
results/bound_and_evaluate_restart_check_summary.txt

Run directly: python bound_and_evaluate_restart_check.py
Requires results/bound_and_evaluate_sweep_results.json and
results/exhaustive_subset_results.json to already exist.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR       = os.path.join(os.path.dirname(__file__), 'results')
EXHAUSTIVE_JSON   = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
SWEEP_JSON        = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_results.json')
OUT_JSON          = os.path.join(RESULTS_DIR, 'bound_and_evaluate_restart_check_results.json')
OUT_CSV           = os.path.join(RESULTS_DIR, 'bound_and_evaluate_restart_check_results.csv')
OUT_SUMMARY       = os.path.join(RESULTS_DIR, 'bound_and_evaluate_restart_check_summary.txt')

N_RESTARTS_NEW    = 7
SWEEP_SEED        = 11     # must match bound_and_evaluate_sweep.py's SEED
DISAGREEMENT_TOL  = 1e-4   # must match the threshold used to select these 29
MAX_REF_TRIES     = 300
SOUNDNESS_TOL     = 1e-6
IMPROVEMENT_TOL   = 1e-6   # how much better counts as "genuinely improved",
                            # not solver noise


def _evaluate_subset(args):
    subset, seed = args

    from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
    from bound_and_evaluate import bound_and_evaluate
    from sdp_topology_bound import find_full_rank_reference
    from riemannian_gradient_sampling import MultiplicityError

    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    target_rank = system.total_dofs - 6
    base_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG)

    t0 = time.perf_counter()
    try:
        ref_rng = np.random.default_rng(seed + 1000)   # same convention as the sweep
        ref = find_full_rank_reference(
            system, subset, target_rank, length_scale=1.0,
            base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)
        if ref is None:
            return dict(subset=subset, status='no_reference', t=None, lambda_min=None,
                        gap=None, restart_min=None, restart_max=None,
                        elapsed_sec=time.perf_counter() - t0, error=None)

        result = bound_and_evaluate(system, subset, length_scale=1.0,
                                     target_rank=target_rank, ref_thetas=ref,
                                     n_restarts=N_RESTARTS_NEW,
                                     rng=np.random.default_rng(seed))
    except (MultiplicityError, RuntimeError, ValueError) as e:
        return dict(subset=subset, status='error', t=None, lambda_min=None,
                    gap=None, restart_min=None, restart_max=None,
                    elapsed_sec=time.perf_counter() - t0, error=str(e))

    restart_vals = [r['lambda_min'] for r in result['all_riemannian_results']]
    return dict(
        subset=subset, status='ok',
        t=result['t'], lambda_min=result['lambda_min'], gap=result['gap'],
        restart_min=min(restart_vals), restart_max=max(restart_vals),
        elapsed_sec=time.perf_counter() - t0, error=None,
    )


def main():
    with open(EXHAUSTIVE_JSON, 'r', encoding='utf-8') as f:
        de_data = json.load(f)
    de_by_subset = {tuple(sorted(r['subset'])): r for r in de_data if r['size'] == 9}
    all_subsets_sorted = sorted(de_by_subset.keys())   # same ordering as the sweep

    with open(SWEEP_JSON, 'r', encoding='utf-8') as f:
        sweep_results = json.load(f)
    sweep_by_subset = {tuple(sorted(r['subset'])): r for r in sweep_results}

    disagreeing = [r for r in sweep_results if r['status'] == 'ok'
                   and (r['restart_max'] - r['restart_min']) > DISAGREEMENT_TOL]
    print(f"Found {len(disagreeing)} subsets with restart spread > {DISAGREEMENT_TOL:.0e} "
          f"in the original n_restarts=5 sweep")

    tasks = []
    for r in disagreeing:
        subset = tuple(sorted(r['subset']))
        idx = all_subsets_sorted.index(subset)
        seed = SWEEP_SEED + idx
        tasks.append((list(subset), seed))

    n_workers = int(os.environ.get('N_WORKERS', 3))
    print(f"Using {n_workers} worker processes, n_restarts={N_RESTARTS_NEW}")

    results = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_evaluate_subset, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            print(f"  {i}/{len(tasks)} done")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll {len(results)} subsets evaluated in {total_elapsed:.0f}s "
          f"using {n_workers} workers")

    # ── Merge old (n=5) vs new (n=7) ─────────────────────────────────────
    merged = []
    for new_r in results:
        subset = tuple(sorted(new_r['subset']))
        old_r = sweep_by_subset[subset]
        merged.append(dict(
            subset=list(subset),
            status=new_r['status'],
            t=new_r['t'],
            old_n_restarts=5, new_n_restarts=N_RESTARTS_NEW,
            old_achieved=old_r['lambda_min'],
            new_achieved=new_r['lambda_min'],
            old_spread=(old_r['restart_max'] - old_r['restart_min']),
            new_spread=(None if new_r['restart_min'] is None
                        else new_r['restart_max'] - new_r['restart_min']),
            gap=new_r['gap'],
            elapsed_sec=new_r['elapsed_sec'],
            error=new_r['error'],
        ))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=1)

    import csv
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'status', 't', 'old_achieved', 'new_achieved',
                    'improvement', 'old_spread', 'new_spread', 'gap',
                    'elapsed_sec', 'error'])
        for r in merged:
            improvement = (None if r['new_achieved'] is None
                           else r['new_achieved'] - r['old_achieved'])
            w.writerow([
                ' '.join(map(str, r['subset'])), r['status'],
                '' if r['t'] is None else f"{r['t']:.8f}",
                f"{r['old_achieved']:.8f}",
                '' if r['new_achieved'] is None else f"{r['new_achieved']:.8f}",
                '' if improvement is None else f"{improvement:.3e}",
                f"{r['old_spread']:.3e}",
                '' if r['new_spread'] is None else f"{r['new_spread']:.3e}",
                '' if r['gap'] is None else f"{r['gap']:.3e}",
                f"{r['elapsed_sec']:.2f}", r['error'] or '',
            ])
    print(f"Saved: {OUT_JSON}\nSaved: {OUT_CSV}")

    # ── Soundness re-check on the new (n=7) results ─────────────────────
    ok_records = [r for r in merged if r['status'] == 'ok']
    violations = [r for r in ok_records if r['gap'] < -SOUNDNESS_TOL]

    lines = [
        f"bound_and_evaluate.py restart-count check "
        f"(the {len(disagreeing)} subsets with spread > {DISAGREEMENT_TOL:.0e} "
        f"at n_restarts=5)",
        "=" * 78,
        f"Rerun at n_restarts={N_RESTARTS_NEW} using the same per-subset seed as "
        f"the original sweep. NOT a strict 'same 5 plus 2 more' superset (the "
        f"shared-RNG design means restart outcomes diverge once more restarts "
        f"are requested, even though starting points nominally match -- see "
        f"module docstring) -- treat this as an independently-seeded 7-restart "
        f"run, related to but not literally extending the original 5.",
        f"Evaluated: {len(ok_records)}/{len(disagreeing)}",
        f"Total wall-clock: {total_elapsed:.0f}s using {n_workers} workers",
        "",
    ]

    if violations:
        lines.append(f"*** SOUNDNESS CHECK FAILED: {len(violations)} subset(s) ***")
        for r in violations:
            lines.append(f"  subset={r['subset']}  t={r['t']:.6f}  "
                         f"achieved={r['new_achieved']:.6f}  gap={r['gap']:.3e}")
        lines.append("")
    else:
        lines.append(f"Soundness check PASSED: t >= achieved - {SOUNDNESS_TOL:.0e} "
                     f"holds for all {len(ok_records)} rerun subsets.")
        lines.append("")

    if ok_records:
        improvements = np.array([r['new_achieved'] - r['old_achieved'] for r in ok_records])
        n_improved = int(np.sum(improvements > IMPROVEMENT_TOL))
        n_same = len(ok_records) - n_improved
        lines.append(f"Improvement (new_achieved - old_achieved) from adding 2 more "
                    f"restarts (5 -> {N_RESTARTS_NEW}):")
        lines.append(f"  min={improvements.min():.3e}  median={np.median(improvements):.3e}  "
                    f"max={improvements.max():.3e}")
        lines.append(f"  subsets that genuinely improved (> {IMPROVEMENT_TOL:.0e}): "
                    f"{n_improved}/{len(ok_records)}")
        lines.append(f"  subsets that had already converged at n_restarts=5: "
                    f"{n_same}/{len(ok_records)}")
        lines.append("")

        old_spreads = np.array([r['old_spread'] for r in ok_records])
        new_spreads = np.array([r['new_spread'] for r in ok_records])
        lines.append(f"Restart spread: n=5 median={np.median(old_spreads):.3e}  "
                    f"n={N_RESTARTS_NEW} median={np.median(new_spreads):.3e}")
        n_still_wide = int(np.sum(new_spreads > DISAGREEMENT_TOL))
        lines.append(f"  still show spread > {DISAGREEMENT_TOL:.0e} at "
                    f"n_restarts={N_RESTARTS_NEW}: {n_still_wide}/{len(ok_records)}")
        lines.append("")

        if n_improved > 0:
            lines.append("Subsets with a genuine improvement (sorted, largest first):")
            improved_records = sorted(
                [r for r in ok_records if r['new_achieved'] - r['old_achieved'] > IMPROVEMENT_TOL],
                key=lambda r: -(r['new_achieved'] - r['old_achieved']))
            for r in improved_records:
                lines.append(f"  subset={r['subset']}  old={r['old_achieved']:.6f}  "
                            f"new={r['new_achieved']:.6f}  "
                            f"improvement={r['new_achieved']-r['old_achieved']:.3e}  "
                            f"t={r['t']:.6f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
