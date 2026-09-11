"""
test_bound_and_evaluate.py

Validation for bound_and_evaluate.py's split-responsibility design:
certified upper bound = SDP stage 1 only; achieved value = multi-start
Riemannian gradient sampling. Neither DE nor the SDP's own stage-2 decode
is used to produce an achieved value anywhere in bound_and_evaluate.py --
this script checks whether that design actually closes the gap the
project's own prior work identified (the SDP decode is unreliable even
though the bound is sound).

Two parts:
  1. Two-panel quick check (known ground truth, fast) -- a sanity anchor.
  2. A SAMPLE of birds-foot size-9 subsets (not the full 220 yet -- see
     this script's own printed timing before deciding whether a full
     sweep is worth the wall-clock cost). For each sampled subset,
     records: the certified bound t, the achieved lambda_min (best of
     n_restarts Riemannian runs), the gap, DE's own ground-truth value,
     the OLD (unreliable) SDP-decoded value, and the spread across
     restarts (a diagnostic for whether n_restarts is large enough).

Soundness check (gap >= -tol for every sampled subset) is checked and
reported loudly, not glossed over, matching this project's established
convention (see sdp_size9_sweep.py, test_sdp_topology_bound.py).

Saves: results/bound_and_evaluate_sample_results.json/csv,
results/bound_and_evaluate_sample_summary.txt

Run directly: python test_bound_and_evaluate.py
"""

import sys, os, json, csv, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)
from bound_and_evaluate import bound_and_evaluate
from sdp_topology_bound import find_full_rank_reference

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
EXHAUSTIVE_JSON = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
SDP_SIZE9_JSON  = os.path.join(RESULTS_DIR, 'sdp_size9_results.json')
OUT_JSON      = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sample_results.json')
OUT_CSV       = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sample_results.csv')
OUT_SUMMARY   = os.path.join(RESULTS_DIR, 'bound_and_evaluate_sample_summary.txt')

N_SAMPLE      = 25
N_RESTARTS    = 5
SOUNDNESS_TOL = 1e-6
SEED          = 11


def test_two_panel_quick_check():
    print("=" * 72)
    print("Two-panel quick check")
    print("=" * 72)

    system = build_system(ARBITRARY_START_THETAS_DEG)
    active = [0, 1, 2]
    ref = np.radians(np.array(ARBITRARY_START_THETAS_DEG))

    result = bound_and_evaluate(system, active, length_scale=1.0, ref_thetas=ref,
                                 n_restarts=N_RESTARTS, rng=np.random.default_rng(0))
    print(f"  certified bound t*  = {result['t']:.6f}  (status={result['stage1_status']})")
    print(f"  achieved lambda_min = {result['lambda_min']:.6f}")
    print(f"  gap                 = {result['gap']:.3e}")
    print(f"  restart spread      = {[round(r['lambda_min'], 8) for r in result['all_riemannian_results']]}")

    ok = result['gap'] >= -SOUNDNESS_TOL and abs(result['lambda_min'] - 0.0133296) < 1e-3
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def _load_ground_truth():
    with open(EXHAUSTIVE_JSON, 'r', encoding='utf-8') as f:
        de = json.load(f)
    with open(SDP_SIZE9_JSON, 'r', encoding='utf-8') as f:
        sdp = json.load(f)
    de_by_subset = {tuple(sorted(r['subset'])): r for r in de if r['size'] == 9}
    sdp_by_subset = {tuple(sorted(r['subset'])): r for r in sdp}
    return de_by_subset, sdp_by_subset


def test_birdsfoot_sample():
    print("=" * 72)
    print(f"Birds-foot sample ({N_SAMPLE} of 220 size-9 subsets, "
          f"n_restarts={N_RESTARTS})")
    print("=" * 72)

    de_by_subset, sdp_by_subset = _load_ground_truth()
    all_subsets = sorted(de_by_subset.keys())
    rng = np.random.default_rng(SEED)
    sample = [all_subsets[i] for i in
              rng.choice(len(all_subsets), size=N_SAMPLE, replace=False)]

    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    target_rank = system.total_dofs - 6
    base_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG)

    records = []
    n_ref_fallback = 0
    t_start = time.perf_counter()
    for i, subset in enumerate(sample, 1):
        subset = list(subset)
        # BIRDSFOOT_START_THETAS_DEG isn't full-rank for every subset (already
        # known -- see sdp_size9_sweep.py/test_sdp_topology_bound.py's own
        # random-reference-search fallback for the same reason). Try the
        # shared default first (cheap, works most of the time), then fall
        # back to a random search over this subset's own reference only if
        # needed.
        ref_rng = np.random.default_rng(SEED + 1000 + i)
        ref = find_full_rank_reference(
            system, subset, target_rank, length_scale=1.0,
            base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=300)
        if ref is None:
            print(f"  [{i}/{N_SAMPLE}] subset={subset}: SKIPPED -- no full-rank "
                  f"reference found in 300 tries")
            continue
        if not np.allclose(ref, np.radians(base_thetas_deg)):
            n_ref_fallback += 1

        t0 = time.perf_counter()
        result = bound_and_evaluate(system, subset, length_scale=1.0,
                                     target_rank=target_rank, ref_thetas=ref,
                                     n_restarts=N_RESTARTS,
                                     rng=np.random.default_rng(SEED + i))
        elapsed = time.perf_counter() - t0

        de_r = de_by_subset.get(tuple(subset))
        sdp_r = sdp_by_subset.get(tuple(subset))
        restart_vals = [r['lambda_min'] for r in result['all_riemannian_results']]

        rec = dict(
            subset=subset,
            t=result['t'],
            lambda_min=result['lambda_min'],
            gap=result['gap'],
            restart_min=min(restart_vals),
            restart_max=max(restart_vals),
            restart_spread=max(restart_vals) - min(restart_vals),
            de_lambda_min=(de_r['lambda_min'] if de_r and not de_r['failed'] else None),
            de_failed=(de_r['failed'] if de_r else None),
            old_sdp_decode=(sdp_r['sdp_lambda_min'] if sdp_r else None),
            elapsed_sec=elapsed,
        )
        records.append(rec)

        de_str = f"{rec['de_lambda_min']:.6f}" if rec['de_lambda_min'] is not None else "FAILED"
        old_sdp_str = f"{rec['old_sdp_decode']:.6f}" if rec['old_sdp_decode'] is not None else "n/a"
        print(f"  [{i}/{N_SAMPLE}] subset={subset}")
        print(f"      t={rec['t']:.6f}  achieved={rec['lambda_min']:.6f}  "
              f"gap={rec['gap']:.3e}  spread={rec['restart_spread']:.2e}  "
              f"time={elapsed:.1f}s")
        print(f"      DE={de_str}  old_SDP_decode={old_sdp_str}")

    total_elapsed = time.perf_counter() - t_start
    n_done = len(records)
    print(f"\n{n_done}/{N_SAMPLE} subsets evaluated in {total_elapsed:.1f}s "
          f"(mean {total_elapsed / max(n_done, 1):.1f}s/subset), "
          f"{n_ref_fallback} needed a fallback (non-default) reference")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=1)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 't', 'lambda_min', 'gap', 'restart_min', 'restart_max',
                    'restart_spread', 'de_lambda_min', 'de_failed', 'old_sdp_decode',
                    'elapsed_sec'])
        for r in records:
            w.writerow([
                ' '.join(map(str, r['subset'])), f"{r['t']:.8f}", f"{r['lambda_min']:.8f}",
                f"{r['gap']:.3e}", f"{r['restart_min']:.8f}", f"{r['restart_max']:.8f}",
                f"{r['restart_spread']:.3e}",
                '' if r['de_lambda_min'] is None else f"{r['de_lambda_min']:.8f}",
                r['de_failed'],
                '' if r['old_sdp_decode'] is None else f"{r['old_sdp_decode']:.8f}",
                f"{r['elapsed_sec']:.2f}",
            ])
    print(f"Saved: {OUT_JSON}\nSaved: {OUT_CSV}")

    # ── Soundness check (loud, not glossed over) ────────────────────────
    violations = [r for r in records if r['gap'] < -SOUNDNESS_TOL]
    lines = [
        "bound_and_evaluate.py sample validation (birds-foot size-9 subsets)",
        "=" * 78,
        f"Sample size: {n_done}/{N_SAMPLE} attempted "
        f"(rest skipped: no full-rank reference found), n_restarts={N_RESTARTS}",
        f"{n_ref_fallback} subset(s) needed a fallback (non-default) reference",
        f"Total wall-clock: {total_elapsed:.1f}s (mean {total_elapsed / max(n_done, 1):.1f}s/subset)",
        "",
    ]
    if violations:
        lines.append(f"*** SOUNDNESS CHECK FAILED: {len(violations)} subset(s) have "
                     f"achieved lambda_min > certified bound t (by more than "
                     f"{SOUNDNESS_TOL:.0e}) ***")
        for r in violations:
            lines.append(f"  subset={r['subset']}  t={r['t']:.6f}  "
                         f"achieved={r['lambda_min']:.6f}  gap={r['gap']:.3e}")
    else:
        lines.append(f"Soundness check PASSED: t >= achieved - {SOUNDNESS_TOL:.0e} "
                     f"holds for all {n_done} sampled subsets.")
    lines.append("")

    gaps = np.array([r['gap'] for r in records])
    lines.append(f"Certification gap (t - achieved) distribution:")
    lines.append(f"  min={gaps.min():.3e}  median={np.median(gaps):.3e}  "
                f"mean={gaps.mean():.3e}  max={gaps.max():.3e}")
    lines.append("")

    spreads = np.array([r['restart_spread'] for r in records])
    lines.append(f"Restart spread (max - min achieved across {N_RESTARTS} restarts) "
                f"distribution:")
    lines.append(f"  min={spreads.min():.3e}  median={np.median(spreads):.3e}  "
                f"max={spreads.max():.3e}")
    n_disagree = int(np.sum(spreads > 1e-4))
    lines.append(f"  subsets with spread > 1e-4 (restarts meaningfully disagreed): "
                f"{n_disagree}/{N_SAMPLE}")
    lines.append("")

    de_comparable = [r for r in records if r['de_lambda_min'] is not None]
    if de_comparable:
        diff = np.array([r['lambda_min'] - r['de_lambda_min'] for r in de_comparable])
        n_better = int(np.sum(diff > 1e-6))
        n_worse = int(np.sum(diff < -1e-6))
        n_tie = len(de_comparable) - n_better - n_worse
        lines.append(f"vs. DE ground truth (n={len(de_comparable)} comparable, "
                    f"{N_SAMPLE - len(de_comparable)} DE-failed):")
        lines.append(f"  Riemannian better: {n_better}   DE better: {n_worse}   "
                    f"tie (within 1e-6): {n_tie}")
        lines.append(f"  (riemannian - de) distribution: min={diff.min():.3e}  "
                    f"median={np.median(diff):.3e}  max={diff.max():.3e}")
        lines.append("")

    old_sdp_comparable = [r for r in records if r['old_sdp_decode'] is not None]
    if old_sdp_comparable:
        diff2 = np.array([r['lambda_min'] - r['old_sdp_decode'] for r in old_sdp_comparable])
        lines.append(f"vs. OLD (unreliable) SDP decode (n={len(old_sdp_comparable)}):")
        lines.append(f"  (riemannian - old_sdp_decode) distribution: "
                    f"min={diff2.min():.3e}  median={np.median(diff2):.3e}  "
                    f"max={diff2.max():.3e}")
        lines.append(f"  Riemannian better in {int(np.sum(diff2 > 1e-6))}/"
                    f"{len(old_sdp_comparable)}")
        lines.append("")

    est_full_sweep_sec = total_elapsed / max(n_done, 1) * 220
    lines.append(f"Extrapolated full-220-subset sweep at this rate: "
                f"~{est_full_sweep_sec:.0f}s (~{est_full_sweep_sec / 60:.1f} min) serially "
                f"-- consider process-pool parallelism (as exhaustive_subset_baseline.py "
                f"uses) before running the full sweep.")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_SUMMARY}")

    return not violations


def main():
    ok1 = test_two_panel_quick_check()
    ok2 = test_birdsfoot_sample()

    print("\n" + "=" * 72)
    print("OVERALL RESULT")
    print("=" * 72)
    print(f"  Two-panel quick check:   {'PASS' if ok1 else 'FAIL'}")
    print(f"  Birds-foot sample:       {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
