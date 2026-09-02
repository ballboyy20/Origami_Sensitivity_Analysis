"""
exhaustive_subset_baseline.py
Evaluates every feasible coupling subset of the birds-foot system (all 12
couplings, structural floor 9) to get a true ground-truth baseline for
CouplingOptimizer.prune_couplings()'s greedy result to be judged against
-- not just the 4 naive single-spoke-removal baselines from
test_prune_baselines.py (which turn out to be exactly 4 of the 299
subsets evaluated here).

"Feasible" = size >= 9 (ceil(target_rank/2) = ceil(18/2) = 9, the fewest
couplings that could possibly reach target_rank=18, since each coupling
contributes at most 2 rows). This is a necessary, not a proven-sufficient
condition -- some size-9..12 subsets could still turn out rank-deficient
at their true optimum; this script's per-subset optimize_theta() call is
exactly what determines that empirically (lambda_min == 0 => infeasible).

299 = C(12,9) + C(12,10) + C(12,11) + C(12,12) = 220 + 66 + 12 + 1.

One optimize_theta() call per subset (n_solutions=1 -- a single stage-2
attempt, not multiple candidates), at reduced settings (maxiter=150)
relative to what's used elsewhere in this codebase -- a deliberate
precision/cost tradeoff for a 299-call sweep; see MAXITER below.
Evaluated in parallel across processes (this is about as ideal a case for
that as exists: 299 fully independent calls, no shared state).

Run directly: python exhaustive_subset_baseline.py
"""

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

N_COUPLINGS      = 12
STRUCTURAL_FLOOR = 9
MAXITER          = 150
N_SOLUTIONS      = 1
MAX_RETRIES      = 5
BASE_SEED        = 42
TOL              = 1e-8

RESULTS_DIR      = os.path.join(os.path.dirname(__file__), 'results')
CSV_PATH         = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.csv')
JSON_PATH        = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
SUMMARY_PATH     = os.path.join(RESULTS_DIR, 'exhaustive_subset_summary.txt')
SCATTER_PATH     = os.path.join(RESULTS_DIR, 'exhaustive_subset_scatter.png')
HIST_PATH        = os.path.join(RESULTS_DIR, 'exhaustive_subset_size9_hist.png')


# ══════════════════════════════════════════════════════════════════════
# Worker (top-level, importable, picklable args/return -- required for
# Windows' spawn-based multiprocessing; must not depend on any live
# CouplingSystem/CouplingOptimizer object built in the parent process)
# ══════════════════════════════════════════════════════════════════════

def _evaluate_subset(args):
    subset_indices, maxiter, n_solutions, seed, max_retries = args

    # Imported inside the worker, not module scope, so each worker process
    # does its own fresh build -- avoids relying on picklability of the
    # RigidPanel/KinematicCoupling objects across process boundaries.
    from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
    from coupling_optimizer import CouplingOptimizer

    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    active_set = set(subset_indices)
    for i, c in enumerate(system.couplings):
        c.active = (i in active_set)
    opt = CouplingOptimizer(system)

    t0 = time.perf_counter()
    try:
        results = opt._optimize_theta_with_retries(
            seed, TOL, maxiter, n_solutions, max_retries)
        best = results[0]
        return dict(
            subset=list(subset_indices), size=len(subset_indices),
            lambda_min=best.lambda_min, log_product=best.log_product,
            optimal_thetas=best.optimal_thetas.tolist(),
            converged=bool(best.converged), n_evaluations=best.n_evaluations,
            elapsed_sec=time.perf_counter() - t0, failed=False, error=None,
        )
    except RuntimeError as e:
        return dict(
            subset=list(subset_indices), size=len(subset_indices),
            lambda_min=None, log_product=None, optimal_thetas=None,
            converged=False, n_evaluations=None,
            elapsed_sec=time.perf_counter() - t0, failed=True, error=str(e),
        )


def _spoke_labels():
    """index (0-11, build order) -> spoke name ('O-M'/'O-B'/'O-N'/'O-A'),
    computed once from a reference system in the parent process."""
    from interactive_optimizer import (
        build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name)
    ref = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    return [birdsfoot_spoke_name(c) for c in ref.couplings]


def _spoke_counts(subset_indices, labels):
    counts = {}
    for i in subset_indices:
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    return ','.join(f'{s}:{counts.get(s, 0)}' for s in ('O-M', 'O-B', 'O-N', 'O-A'))


def _naive_baseline_indices(labels):
    """The 4 single-spoke-removal baselines from test_prune_baselines.py,
    as index sets -- each is exactly one of the 299 subsets swept here."""
    out = {}
    for spoke in ('O-M', 'O-B', 'O-N', 'O-A'):
        out[f'no {spoke}'] = tuple(i for i, s in enumerate(labels) if s != spoke)
    return out


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = _spoke_labels()

    subsets = []
    for k in range(STRUCTURAL_FLOOR, N_COUPLINGS + 1):
        subsets.extend(itertools.combinations(range(N_COUPLINGS), k))
    print(f"Evaluating {len(subsets)} feasible subsets "
          f"(sizes {STRUCTURAL_FLOOR}..{N_COUPLINGS}), "
          f"maxiter={MAXITER}, n_solutions={N_SOLUTIONS}")

    tasks = [(s, MAXITER, N_SOLUTIONS, BASE_SEED + i, MAX_RETRIES)
             for i, s in enumerate(subsets)]

    n_workers = max(1, (os.cpu_count() or 2) - 1)
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

    # ── Save raw results (JSON: everything incl. thetas; CSV: summary) ──
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)

    import csv
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['size', 'active_indices', 'spoke_counts', 'lambda_min',
                    'log_product', 'converged', 'n_evaluations',
                    'elapsed_sec', 'failed', 'error'])
        for r in results:
            w.writerow([
                r['size'], ' '.join(map(str, r['subset'])),
                _spoke_counts(r['subset'], labels),
                r['lambda_min'], r['log_product'], r['converged'],
                r['n_evaluations'], f"{r['elapsed_sec']:.2f}",
                r['failed'], r['error'],
            ])
    print(f"Saved: {JSON_PATH}\nSaved: {CSV_PATH}")

    # ── Aggregate analysis ────────────────────────────────────────────
    ok = [r for r in results if not r['failed'] and r['lambda_min'] and r['lambda_min'] > 0]
    failed = [r for r in results if r['failed']]
    rank_deficient = [r for r in results
                      if not r['failed'] and (not r['lambda_min'] or r['lambda_min'] <= 0)]

    lines = [
        "Exhaustive feasible-subset baseline (birds-foot, 299 subsets)",
        "=" * 78,
        f"Total subsets evaluated: {len(results)}",
        f"  reached full rank (lambda_min > 0): {len(ok)}",
        f"  rank-deficient at optimum (lambda_min == 0): {len(rank_deficient)}",
        f"  optimize_theta() failed after {MAX_RETRIES} retries: {len(failed)}",
        f"Total wall-clock: {total_elapsed:.0f}s with {n_workers} workers",
        "",
    ]

    if ok:
        best = max(ok, key=lambda r: (r['lambda_min'], r['log_product']))
        lines.append("Global best subset (max lambda_min, log_product tiebreak):")
        lines.append(f"  size={best['size']}  spokes=[{_spoke_counts(best['subset'], labels)}]")
        lines.append(f"  indices={best['subset']}")
        lines.append(f"  lambda_min={best['lambda_min']:.6f}  log_product={best['log_product']:.6f}")
        lines.append(f"  optimal_thetas (deg)={np.round(np.degrees(best['optimal_thetas']), 2).tolist()}")
        lines.append("")

        lines.append("Best per subset size:")
        for k in range(STRUCTURAL_FLOOR, N_COUPLINGS + 1):
            group = [r for r in ok if r['size'] == k]
            if not group:
                lines.append(f"  size {k}: no feasible subsets found")
                continue
            b = max(group, key=lambda r: (r['lambda_min'], r['log_product']))
            lines.append(f"  size {k}: n={len(group)}  best lambda_min={b['lambda_min']:.6f}  "
                        f"log_product={b['log_product']:.6f}  spokes=[{_spoke_counts(b['subset'], labels)}]")
        lines.append("")

        size9 = [r for r in ok if r['size'] == 9]
        if size9:
            lam9 = np.array([r['lambda_min'] for r in size9])
            lines.append(f"Size-9 distribution (n={len(size9)}): "
                        f"lambda_min min={lam9.min():.6f} median={np.median(lam9):.6f} "
                        f"mean={lam9.mean():.6f} max={lam9.max():.6f}")
            lines.append("")

        baselines = _naive_baseline_indices(labels)
        lines.append("Naive single-spoke-removal baselines (from test_prune_baselines.py) "
                     "within this sweep:")
        size9_sorted = sorted(size9, key=lambda r: (-r['lambda_min'], -r['log_product']))
        for name, idx in baselines.items():
            match = next((r for r in size9 if set(r['subset']) == set(idx)), None)
            if match is None:
                lines.append(f"  {name}: not found among evaluated size-9 subsets (unexpected)")
                continue
            rank = size9_sorted.index(match) + 1
            lines.append(f"  {name}: lambda_min={match['lambda_min']:.6f}  "
                        f"log_product={match['log_product']:.6f}  "
                        f"rank {rank}/{len(size9_sorted)} among size-9 subsets")
        lines.append("")

        print("\nRunning prune_couplings() fresh for a same-session greedy comparison...")
        from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
        from coupling_optimizer import CouplingOptimizer
        greedy_system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
        greedy_opt = CouplingOptimizer(greedy_system)
        greedy_result = greedy_opt.prune_couplings(maxiter=300, n_solutions=2)
        greedy_final = greedy_result.steps[-1]
        greedy_active_idx = tuple(i for i, c in enumerate(greedy_system.couplings)
                                  if getattr(c, 'active', True))
        greedy_lam = greedy_final.theta_result.lambda_min
        greedy_logp = greedy_final.theta_result.log_product
        lines.append(f"Greedy prune_couplings() result: size={len(greedy_active_idx)}  "
                    f"spokes=[{_spoke_counts(greedy_active_idx, labels)}]")
        lines.append(f"  lambda_min={greedy_lam:.6f}  log_product={greedy_logp:.6f}")
        if len(greedy_active_idx) == 9 and size9_sorted:
            # Greedy's own polished result isn't necessarily identical to
            # the sweep's (different settings/seed for that exact subset),
            # so rank it against the sweep's value for that SAME subset,
            # not its own possibly-more-polished number.
            match = next((r for r in size9 if set(r['subset']) == set(greedy_active_idx)), None)
            if match is not None:
                rank = size9_sorted.index(match) + 1
                lines.append(f"  this subset ranks {rank}/{len(size9_sorted)} among size-9 "
                            f"subsets (sweep's lambda_min={match['lambda_min']:.6f} for "
                            f"the same subset, at sweep settings)")
        lines.append("")

    if rank_deficient:
        lines.append(f"Rank-deficient subsets ({len(rank_deficient)}) -- structural floor was "
                     f"necessary but not sufficient for these:")
        for r in rank_deficient[:20]:
            lines.append(f"  size={r['size']}  spokes=[{_spoke_counts(r['subset'], labels)}]")
        if len(rank_deficient) > 20:
            lines.append(f"  ... and {len(rank_deficient) - 20} more")
        lines.append("")

    if failed:
        lines.append(f"optimize_theta() failures ({len(failed)}) -- see JSON for full error text:")
        for r in failed[:20]:
            lines.append(f"  size={r['size']}  spokes=[{_spoke_counts(r['subset'], labels)}]")
        if len(failed) > 20:
            lines.append(f"  ... and {len(failed) - 20} more")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {SUMMARY_PATH}")

    # ── Figures ────────────────────────────────────────────────────────
    if ok:
        rng = np.random.default_rng(0)
        sizes = np.array([r['size'] for r in ok], dtype=float)
        jitter = rng.uniform(-0.15, 0.15, size=len(ok))
        lam = np.array([r['lambda_min'] for r in ok])
        logp = np.array([r['log_product'] for r in ok])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Exhaustive feasible-subset sweep (birds-foot, 299 subsets)',
                    fontweight='bold')
        axes[0].scatter(sizes + jitter, lam, s=14, alpha=0.4, color='#BDC3C7',
                        label='all feasible subsets')
        axes[1].scatter(sizes + jitter, logp, s=14, alpha=0.4, color='#BDC3C7',
                        label='all feasible subsets')

        baselines = _naive_baseline_indices(labels)
        marker_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
        for (name, idx), color in zip(baselines.items(), marker_colors):
            match = next((r for r in ok if set(r['subset']) == set(idx)), None)
            if match is None:
                continue
            axes[0].scatter([match['size']], [match['lambda_min']], s=90, color=color,
                            edgecolor='black', zorder=5, label=name)
            axes[1].scatter([match['size']], [match['log_product']], s=90, color=color,
                            edgecolor='black', zorder=5, label=name)

        axes[0].set_xlabel('Active couplings')
        axes[0].set_ylabel('lambda_min')
        axes[0].set_title('Worst-case stiffness vs. subset size')
        axes[0].set_xticks([9, 10, 11, 12])
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)

        axes[1].set_xlabel('Active couplings')
        axes[1].set_ylabel('log_product')
        axes[1].set_title('Overall stiffness proxy vs. subset size')
        axes[1].set_xticks([9, 10, 11, 12])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(SCATTER_PATH, dpi=150)
        plt.show()
        print(f"Saved: {SCATTER_PATH}")

        size9 = [r for r in ok if r['size'] == 9]
        if size9:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.hist([r['lambda_min'] for r in size9], bins=30, color='#3F8EFC',
                     edgecolor='white')
            ax2.set_xlabel('lambda_min')
            ax2.set_ylabel('count')
            ax2.set_title(f'Size-9 subset lambda_min distribution (n={len(size9)})')
            ax2.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            fig2.savefig(HIST_PATH, dpi=150)
            plt.show()
            print(f"Saved: {HIST_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
