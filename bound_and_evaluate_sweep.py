"""
bound_and_evaluate_sweep.py

Full-scale (all 220 birds-foot size-9 subsets) run of bound_and_evaluate.py's
split-responsibility design: certified upper bound = SDP stage 1 only;
achieved value = best-of-n_restarts Riemannian gradient sampling. Neither
DE nor the SDP's own stage-2 decode is used to produce an achieved value
anywhere in this sweep.

This is the scaled-up version of test_bound_and_evaluate.py's 25-subset
sample (see results/bound_and_evaluate_sample_summary.txt for that run's
findings: soundness held with zero violations, achieved values were
essentially tied with DE where DE converged, RESOLVED all 8/25 sampled
subsets where DE itself failed to converge, and beat the old (unreliable)
SDP decode in 19/22 comparable subsets). Parallelized via
ProcessPoolExecutor, mirroring exhaustive_subset_baseline.py's own
worker-process pattern (fresh imports/system build per worker -- required
for Windows' spawn-based multiprocessing, not just style).

Saves: results/bound_and_evaluate_sweep_results.json/csv,
results/bound_and_evaluate_sweep_summary.txt

Run directly: python bound_and_evaluate_sweep.py
Requires results/exhaustive_subset_results.json and
results/sdp_size9_results.json to already exist (same ground truth
test_bound_and_evaluate.py uses).
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR     = os.path.join(os.path.dirname(__file__), 'results')
EXHAUSTIVE_JSON = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
SDP_SIZE9_JSON  = os.path.join(RESULTS_DIR, 'sdp_size9_results.json')
OUT_JSON        = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_results.json')
OUT_CSV         = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_results.csv')
OUT_SUMMARY     = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sweep_summary.txt')

N_RESTARTS      = 5
MAX_REF_TRIES   = 300
SOUNDNESS_TOL   = 1e-6
SEED            = 11


# ══════════════════════════════════════════════════════════════════════
# Worker (top-level, importable, picklable args/return -- required for
# Windows' spawn-based multiprocessing; must not depend on any live
# CouplingSystem object built in the parent process)
# ══════════════════════════════════════════════════════════════════════

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
        ref_rng = np.random.default_rng(seed + 1000)
        ref = find_full_rank_reference(
            system, subset, target_rank, length_scale=1.0,
            base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)
        if ref is None:
            return dict(subset=subset, status='no_reference', t=None, lambda_min=None,
                        gap=None, restart_min=None, restart_max=None,
                        elapsed_sec=time.perf_counter() - t0, error=None)

        result = bound_and_evaluate(system, subset, length_scale=1.0,
                                     target_rank=target_rank, ref_thetas=ref,
                                     n_restarts=N_RESTARTS,
                                     rng=np.random.default_rng(seed))
    except (MultiplicityError, RuntimeError, ValueError) as e:
        # One subset's failure (of any of these known-possible kinds --
        # e.g. persistent eigenvalue multiplicity across a wide theta
        # region, or an SDP solve failure) must NOT take down the whole
        # 220-subset ProcessPoolExecutor run -- mirrors
        # exhaustive_subset_baseline.py's own convention of recording a
        # per-subset failure rather than letting one bad subset crash
        # the pool and lose every other worker's progress.
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


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    with open(EXHAUSTIVE_JSON, 'r', encoding='utf-8') as f:
        de_data = json.load(f)
    with open(SDP_SIZE9_JSON, 'r', encoding='utf-8') as f:
        sdp_data = json.load(f)
    de_by_subset = {tuple(sorted(r['subset'])): r for r in de_data if r['size'] == 9}
    sdp_by_subset = {tuple(sorted(r['subset'])): r for r in sdp_data}

    subsets = sorted(de_by_subset.keys())
    print(f"Evaluating {len(subsets)} size-9 subsets, n_restarts={N_RESTARTS}")

    tasks = [(list(s), SEED + i) for i, s in enumerate(subsets)]
    # NOTE: NOT cpu_count()-1 (this repo's usual convention, e.g.
    # exhaustive_subset_baseline.py) -- on this machine (7.6GB total RAM),
    # that gave 7 workers, each fully importing numpy/scipy/cvxpy plus the
    # coupling-system codebase, and exhausted available memory (a real
    # MemoryError inside a worker's cvxpy import killed a full run -- see
    # results/bound_and_evaluate_sweep_summary.txt's notes, if present, or
    # the session history). Overridable via the N_WORKERS env var if a
    # future machine has more headroom.
    n_workers = int(os.environ.get('N_WORKERS', 3))
    print(f"Using {n_workers} worker processes")

    results = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_evaluate_subset, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t_start
                print(f"  {i}/{len(tasks)} done ({elapsed:.0f}s elapsed, "
                      f"~{elapsed / i * (len(tasks) - i):.0f}s remaining)")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll {len(results)} subsets evaluated in {total_elapsed:.0f}s "
          f"using {n_workers} workers")

    results_by_subset = {tuple(sorted(r['subset'])): r for r in results}

    # ── Merge in ground truth for reporting ─────────────────────────────
    merged = []
    for subset in subsets:
        r = results_by_subset[subset]
        de_r = de_by_subset.get(subset)
        sdp_r = sdp_by_subset.get(subset)
        merged.append(dict(
            subset=list(subset),
            status=r['status'],
            t=r['t'], lambda_min=r['lambda_min'], gap=r['gap'],
            restart_min=r['restart_min'], restart_max=r['restart_max'],
            elapsed_sec=r['elapsed_sec'], error=r.get('error'),
            de_lambda_min=(de_r['lambda_min'] if de_r and not de_r['failed'] else None),
            de_failed=(de_r['failed'] if de_r else None),
            old_sdp_decode=(sdp_r['sdp_lambda_min'] if sdp_r else None),
        ))

    # ── Save raw results ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=1)

    import csv
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'status', 't', 'lambda_min', 'gap', 'restart_min',
                    'restart_max', 'de_lambda_min', 'de_failed', 'old_sdp_decode',
                    'elapsed_sec', 'error'])
        for r in merged:
            w.writerow([
                ' '.join(map(str, r['subset'])), r['status'],
                '' if r['t'] is None else f"{r['t']:.8f}",
                '' if r['lambda_min'] is None else f"{r['lambda_min']:.8f}",
                '' if r['gap'] is None else f"{r['gap']:.3e}",
                '' if r['restart_min'] is None else f"{r['restart_min']:.8f}",
                '' if r['restart_max'] is None else f"{r['restart_max']:.8f}",
                '' if r['de_lambda_min'] is None else f"{r['de_lambda_min']:.8f}",
                r['de_failed'],
                '' if r['old_sdp_decode'] is None else f"{r['old_sdp_decode']:.8f}",
                f"{r['elapsed_sec']:.2f}", r['error'] or '',
            ])
    print(f"Saved: {OUT_JSON}\nSaved: {OUT_CSV}")

    # ── Soundness check (loud, not glossed over) ────────────────────────
    ok_records = [r for r in merged if r['status'] == 'ok']
    no_ref = [r for r in merged if r['status'] == 'no_reference']
    errored = [r for r in merged if r['status'] == 'error']
    violations = [r for r in ok_records if r['gap'] < -SOUNDNESS_TOL]

    lines = [
        "bound_and_evaluate.py full sweep (birds-foot, all 220 size-9 subsets)",
        "=" * 78,
        f"n_restarts={N_RESTARTS}",
        f"Evaluated: {len(ok_records)}/220  ({len(no_ref)} skipped -- no full-rank "
        f"reference found in {MAX_REF_TRIES} tries; {len(errored)} errored)",
        f"Total wall-clock: {total_elapsed:.0f}s using {n_workers} workers",
        "",
    ]

    if errored:
        lines.append(f"Subsets that errored ({len(errored)}) -- see JSON for full "
                     f"error text, none of these crashed the sweep itself:")
        for r in errored[:20]:
            lines.append(f"  subset={r['subset']}  error={r['error']}")
        if len(errored) > 20:
            lines.append(f"  ... and {len(errored) - 20} more")
        lines.append("")

    if violations:
        lines.append(f"*** SOUNDNESS CHECK FAILED: {len(violations)} subset(s) have "
                     f"achieved lambda_min > certified bound t (by more than "
                     f"{SOUNDNESS_TOL:.0e}) ***")
        for r in violations:
            lines.append(f"  subset={r['subset']}  t={r['t']:.6f}  "
                         f"achieved={r['lambda_min']:.6f}  gap={r['gap']:.3e}")
        lines.append("")
        summary = "\n".join(lines)
        print("\n" + summary)
        with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write(summary + "\n")
        raise RuntimeError(
            f"{len(violations)} soundness violation(s) -- see printed detail above. "
            f"Do not trust bound_and_evaluate until this is understood.")
    else:
        lines.append(f"Soundness check PASSED: t >= achieved - {SOUNDNESS_TOL:.0e} "
                     f"holds for all {len(ok_records)} evaluated subsets.")
    lines.append("")

    gaps = np.array([r['gap'] for r in ok_records])
    lines.append("Certification gap (t - achieved) distribution:")
    lines.append(f"  min={gaps.min():.3e}  median={np.median(gaps):.3e}  "
                f"mean={gaps.mean():.3e}  max={gaps.max():.3e}")
    lines.append("")

    spreads = np.array([r['restart_max'] - r['restart_min'] for r in ok_records])
    lines.append(f"Restart spread (max - min achieved across {N_RESTARTS} restarts):")
    lines.append(f"  min={spreads.min():.3e}  median={np.median(spreads):.3e}  "
                f"max={spreads.max():.3e}")
    n_disagree = int(np.sum(spreads > 1e-4))
    lines.append(f"  subsets with spread > 1e-4 (restarts meaningfully disagreed): "
                f"{n_disagree}/{len(ok_records)}")
    lines.append("")

    de_conv = [r for r in ok_records if r['de_lambda_min'] is not None]
    de_failed_but_ours_ok = [r for r in ok_records if r['de_failed']]
    if de_conv:
        diff = np.array([r['lambda_min'] - r['de_lambda_min'] for r in de_conv])
        n_better = int(np.sum(diff > 1e-6))
        n_worse = int(np.sum(diff < -1e-6))
        n_tie = len(de_conv) - n_better - n_worse
        lines.append(f"vs. DE ground truth (n={len(de_conv)} DE-converged, "
                    f"{len(de_failed_but_ours_ok)} DE-failed):")
        lines.append(f"  Riemannian better: {n_better}   DE better: {n_worse}   "
                    f"tie (within 1e-6): {n_tie}")
        lines.append(f"  (riemannian - de) distribution: min={diff.min():.3e}  "
                    f"median={np.median(diff):.3e}  max={diff.max():.3e}")
    if de_failed_but_ours_ok:
        resolved = [r for r in de_failed_but_ours_ok if r['lambda_min'] > 1e-6]
        lines.append(f"  Of {len(de_failed_but_ours_ok)} subsets DE itself failed to "
                    f"converge on, Riemannian multi-start produced a non-degenerate "
                    f"result for {len(resolved)}")
        if resolved:
            lam = np.array([r['lambda_min'] for r in resolved])
            lines.append(f"    resolved lambda_min: min={lam.min():.6f}  "
                        f"median={np.median(lam):.6f}  max={lam.max():.6f}")
    lines.append("")

    old_sdp_comp = [r for r in ok_records if r['old_sdp_decode'] is not None]
    if old_sdp_comp:
        diff2 = np.array([r['lambda_min'] - r['old_sdp_decode'] for r in old_sdp_comp])
        lines.append(f"vs. OLD (unreliable) SDP decode (n={len(old_sdp_comp)}):")
        lines.append(f"  (riemannian - old_sdp_decode) distribution: "
                    f"min={diff2.min():.3e}  median={np.median(diff2):.3e}  "
                    f"max={diff2.max():.3e}")
        lines.append(f"  Riemannian better in {int(np.sum(diff2 > 1e-6))}/"
                    f"{len(old_sdp_comp)}")
    lines.append("")

    best = max(ok_records, key=lambda r: r['lambda_min'])
    lines.append("Best size-9 subset found (max achieved lambda_min):")
    lines.append(f"  subset={best['subset']}  t={best['t']:.6f}  "
                f"achieved={best['lambda_min']:.6f}")
    lines.append(f"Previous champion (results/exhaustive_subset_summary.txt): "
                f"lambda_min=0.009259")
    lines.append(f"  -> {'BEATS' if best['lambda_min'] > 0.009259 + 1e-4 else 'MATCHES/does not beat'} "
                f"the previous champion")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
