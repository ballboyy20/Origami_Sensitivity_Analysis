"""
sdp_size9_sweep.py

Runs the SDP relaxation of stage 1 (sdp_lambda_min, see
sdp_theta_optimizer.py) on every size-9 subset of the birds-foot system's
12 couplings -- the same 220 subsets differential_evolution was run on
in exhaustive_subset_baseline.py (results/exhaustive_subset_results.json),
150 of which DE converged on and ~70 of which DE gave up on ("no stage-2
candidate reached the stage-1 optimum").

Purpose: validate the SDP relaxation against that existing ground truth
(does t >= achieved lambda_min everywhere? how tight is the bound? does
it find plausible answers for the subsets DE failed on? does it beat the
best subset DE ever found?) -- NOT to relax the discrete topology choice
itself (that's a separate, out-of-scope follow-up; every subset here is
still a FIXED, exhaustively-enumerated topology).

Run directly: python sdp_size9_sweep.py
Requires results/exhaustive_subset_results.json (from
exhaustive_subset_baseline.py) to already exist.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name)
from sdp_theta_optimizer import sdp_lambda_min

RESULTS_DIR       = os.path.join(os.path.dirname(__file__), 'results')
INPUT_JSON        = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
OUT_JSON          = os.path.join(RESULTS_DIR, 'sdp_size9_results.json')
OUT_CSV           = os.path.join(RESULTS_DIR, 'sdp_size9_results.csv')
OUT_SUMMARY       = os.path.join(RESULTS_DIR, 'sdp_size9_summary.txt')

GAP_TOL           = 1e-6   # tolerance for the t >= achieved lambda_min sanity check
NONDEGENERATE_TOL = 1e-6   # achieved lambda_min above this counts as "resolved"
N_REF_RETRIES     = 300    # random reference-theta draws tried (cheap rank
                            # check only, see _find_full_rank_reference)
                            # before giving up on a subset

PREVIOUS_CHAMPION_LAMBDA_MIN = 0.009259   # from results/exhaustive_subset_summary.txt
PREVIOUS_CHAMPION_SPOKES     = 'O-M:3,O-B:2,O-N:3,O-A:1'


def _spoke_counts(subset_indices, labels):
    counts = {}
    for i in subset_indices:
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    return ','.join(f'{s}:{counts.get(s, 0)}' for s in ('O-M', 'O-B', 'O-N', 'O-A'))


def _find_full_rank_reference(system, subset, target_rank, length_scale, rng,
                               base_thetas_deg, max_tries):
    """
    A reference theta vector (radians, length = len(system.couplings))
    for which THIS subset's constraint matrix reaches target_rank --
    tried cheaply (just build_constraint_matrix + matrix_rank, not a
    full calibrate()+SDP solve) since some 9-coupling subsets turn out
    rank-deficient at the shared 12-coupling BIRDSFOOT_START_THETAS_DEG
    (that reference was only chosen to avoid 3-parallel-grooves-per-spoke
    degeneracy for the FULL 12-coupling system, not every 9-subset of
    it). The decomposition (*) sdp_lambda_min relies on is exact for any
    reference, so trying many is just about finding ONE that isn't
    accidentally degenerate for this particular subset -- not a search
    over 'better' references.
    """
    active = set(subset)
    for i, c in enumerate(system.couplings):
        c.active = (i in active)

    # First try the shared default, then small jitters around it, then
    # fully random -- cheapest/most-plausible candidates first.
    candidates = [np.radians(np.array(base_thetas_deg))]
    for _ in range(max_tries // 2):
        jitter = rng.uniform(-30, 30, size=len(base_thetas_deg))
        candidates.append(np.radians(np.array(base_thetas_deg) + jitter))
    for _ in range(max_tries - len(candidates)):
        candidates.append(rng.uniform(0, np.pi, size=len(base_thetas_deg)))

    for ref in candidates:
        for c, th in zip(system.couplings, ref):
            c.set_theta(float(th))
        C = system.build_constraint_matrix(length_scale=length_scale)
        if np.linalg.matrix_rank(C) >= target_rank:
            return ref
    return None


def _solve_one(subset, base_thetas_deg, rng, target_rank):
    """
    Finds a full-rank reference for this subset (see
    _find_full_rank_reference), then calls sdp_lambda_min() with it.
    """
    system = build_birdsfoot_system(np.array(base_thetas_deg))
    ref = _find_full_rank_reference(
        system, subset, target_rank, 1.0, rng, base_thetas_deg, N_REF_RETRIES)
    if ref is None:
        raise RuntimeError(
            f"Could not find a full-rank reference for subset={subset} "
            f"after {N_REF_RETRIES} tries.")
    return sdp_lambda_min(system, subset, length_scale=1.0, ref_thetas=ref)


def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        de_results = json.load(f)
    de_size9 = [r for r in de_results if r['size'] == 9]
    print(f"Loaded {len(de_size9)} size-9 subsets from {INPUT_JSON} "
          f"({sum(1 for r in de_size9 if not r['failed'])} DE-converged, "
          f"{sum(1 for r in de_size9 if r['failed'])} DE-failed)")

    ref_system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    labels = [birdsfoot_spoke_name(c) for c in ref_system.couplings]
    target_rank = ref_system.total_dofs - 6

    rng = np.random.default_rng(0)
    results = []
    no_ref_found = []   # subsets where even a cheap random rank search found
                         # nothing >= target_rank -- likely structurally
                         # rank-deficient topologies (no theta works at all),
                         # not merely a hard-to-find optimum. Recorded, not
                         # fatal to the sweep.
    t0 = time.perf_counter()
    for i, de_r in enumerate(de_size9, 1):
        subset = de_r['subset']
        try:
            sdp_r = _solve_one(subset, BIRDSFOOT_START_THETAS_DEG, rng, target_rank)
        except RuntimeError as e:
            no_ref_found.append(subset)
            results.append(dict(
                subset=subset,
                spokes=_spoke_counts(subset, labels),
                sdp_t=0.0,
                sdp_lambda_min=0.0,
                sdp_log_product=float('-inf'),
                sdp_gap=0.0,
                sdp_thetas_deg=None,
                sdp_no_full_rank_reference=True,
                de_failed=de_r['failed'],
                de_lambda_min=de_r['lambda_min'],
                de_log_product=de_r['log_product'],
                de_converged=de_r['converged'],
            ))
            print(f"  [{i}/{len(de_size9)}] subset={subset} spokes=[{_spoke_counts(subset, labels)}]: "
                  f"no full-rank reference found in {N_REF_RETRIES} random tries "
                  f"-- likely structurally rank-deficient, recorded as such: {e}")
            continue
        results.append(dict(
            subset=subset,
            spokes=_spoke_counts(subset, labels),
            sdp_t=sdp_r['t'],
            sdp_lambda_min=sdp_r['lambda_min'],
            sdp_log_product=sdp_r['log_product'],
            sdp_gap=sdp_r['gap'],
            sdp_thetas_deg=np.degrees(sdp_r['thetas']).tolist(),
            sdp_no_full_rank_reference=False,
            de_failed=de_r['failed'],
            de_lambda_min=de_r['lambda_min'],
            de_log_product=de_r['log_product'],
            de_converged=de_r['converged'],
        ))
        if i % 20 == 0 or i == len(de_size9):
            elapsed = time.perf_counter() - t0
            print(f"  {i}/{len(de_size9)} done ({elapsed:.0f}s elapsed)")

    total_elapsed = time.perf_counter() - t0
    print(f"\nAll {len(results)} size-9 subsets solved in {total_elapsed:.1f}s")

    # ── Save raw results ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)

    import csv
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['active_indices', 'spoke_counts', 'sdp_t', 'sdp_lambda_min',
                    'sdp_log_product', 'sdp_gap', 'de_failed', 'de_lambda_min',
                    'de_log_product', 'de_converged'])
        for r in results:
            w.writerow([
                ' '.join(map(str, r['subset'])), r['spokes'],
                f"{r['sdp_t']:.8f}", f"{r['sdp_lambda_min']:.8f}",
                f"{r['sdp_log_product']:.6f}", f"{r['sdp_gap']:.3e}",
                r['de_failed'],
                '' if r['de_lambda_min'] is None else f"{r['de_lambda_min']:.8f}",
                '' if r['de_log_product'] is None else f"{r['de_log_product']:.6f}",
                r['de_converged'],
            ])
    print(f"Saved: {OUT_JSON}\nSaved: {OUT_CSV}")

    # ── Sanity check: t >= achieved lambda_min everywhere ────────────────
    violations = [r for r in results if r['sdp_gap'] < -GAP_TOL]
    lines = [
        "SDP stage-1 relaxation sweep: birds-foot size-9 subsets (220 total)",
        "=" * 78,
        f"Subsets evaluated: {len(results)}",
        f"Total wall-clock: {total_elapsed:.1f}s",
        "",
    ]

    if no_ref_found:
        lines.append(
            f"{len(no_ref_found)} subset(s) had NO full-rank reference found in "
            f"{N_REF_RETRIES} random theta draws -- recorded with sdp_lambda_min=0, "
            f"sdp_t=0 (not fed into the SDP at all). This is evidence (not proof) "
            f"these specific topologies may be structurally rank-deficient for ANY "
            f"theta, not merely hard for DE to optimize:")
        for s in no_ref_found[:10]:
            r = next(r for r in results if r['subset'] == s)
            lines.append(f"  spokes=[{r['spokes']}]  indices={s}  "
                        f"(DE also {'failed' if r['de_failed'] else 'converged'} on this subset)")
        if len(no_ref_found) > 10:
            lines.append(f"  ... and {len(no_ref_found) - 10} more")
        lines.append("")

    if violations:
        lines.append(
            f"*** SANITY CHECK FAILED: {len(violations)} subset(s) have "
            f"t < achieved lambda_min - {GAP_TOL:.0e} -- the SDP bound is "
            f"supposed to be a valid upper bound; this should never happen. ***")
        for r in violations[:10]:
            lines.append(f"  spokes=[{r['spokes']}]  t={r['sdp_t']:.6f}  "
                        f"achieved={r['sdp_lambda_min']:.6f}  gap={r['sdp_gap']:.3e}")
        print("\n".join(lines))
        raise RuntimeError(
            f"{len(violations)} subset(s) violate t >= achieved lambda_min "
            f"(see printed detail above) -- SDP relaxation soundness check FAILED.")
    else:
        lines.append(f"Sanity check PASSED: t >= achieved lambda_min - {GAP_TOL:.0e} "
                     f"holds for all {len(results)} subsets.")
    lines.append("")

    # ── Rounding-gap distribution ─────────────────────────────────────────
    gaps = np.array([r['sdp_gap'] for r in results])
    lines.append(f"Rounding gap (t - achieved lambda_min) distribution:")
    lines.append(f"  min={gaps.min():.6e}  median={np.median(gaps):.6e}  "
                f"mean={gaps.mean():.6e}  max={gaps.max():.6e}")
    lines.append("")

    # ── Cross-check against DE's own results on converged subsets ────────
    converged = [r for r in results if not r['de_failed']]
    if converged:
        diff = np.array([r['sdp_lambda_min'] - r['de_lambda_min'] for r in converged])
        n_sdp_better = int(np.sum(diff > 1e-6))
        n_de_better  = int(np.sum(diff < -1e-6))
        n_tie        = len(converged) - n_sdp_better - n_de_better
        lines.append(f"DE-converged subsets (n={len(converged)}): "
                    f"SDP achieved lambda_min vs. DE's own reported lambda_min")
        lines.append(f"  SDP strictly better: {n_sdp_better}   "
                    f"DE strictly better: {n_de_better}   tie (within 1e-6): {n_tie}")
        lines.append(f"  (sdp - de) distribution: min={diff.min():.3e}  "
                    f"median={np.median(diff):.3e}  max={diff.max():.3e}")
        lines.append("")

    # ── DE-failed subsets: does the SDP resolve them? ─────────────────────
    de_failed = [r for r in results if r['de_failed']]
    if de_failed:
        resolved = [r for r in de_failed if r['sdp_lambda_min'] > NONDEGENERATE_TOL]
        lam_resolved = np.array([r['sdp_lambda_min'] for r in resolved])
        lines.append(f"DE-failed subsets (n={len(de_failed)}, 'no stage-2 candidate "
                    f"reached the stage-1 optimum'):")
        lines.append(f"  SDP found a plausible non-degenerate result "
                    f"(lambda_min > {NONDEGENERATE_TOL:.0e}) for "
                    f"{len(resolved)}/{len(de_failed)}")
        if resolved:
            lines.append(f"  resolved lambda_min: min={lam_resolved.min():.6f}  "
                        f"median={np.median(lam_resolved):.6f}  "
                        f"max={lam_resolved.max():.6f}")
        still_degenerate = [r for r in de_failed if r not in resolved]
        if still_degenerate:
            no_ref = [r for r in still_degenerate if r.get('sdp_no_full_rank_reference')]
            had_ref = [r for r in still_degenerate if not r.get('sdp_no_full_rank_reference')]
            lines.append(f"  still degenerate/near-zero ({len(still_degenerate)}), split by cause:")
            lines.append(f"    no full-rank reference found at all ({len(no_ref)}) -- "
                        f"evidence these topologies may be structurally rank-deficient "
                        f"for ANY theta, not just hard to optimize")
            lines.append(f"    full-rank reference existed, but decode still landed "
                        f"degenerate ({len(had_ref)}) -- the topology CAN reach full "
                        f"rank; the SDP's relaxed optimum for these apparently sits "
                        f"where one or more (u_i, v_i) has near-zero magnitude, so its "
                        f"decoded angle is unreliable (see decode_uv_to_theta's "
                        f"docstring on the (0,0) degeneracy) -- these look like a "
                        f"decode-step limitation, not evidence of infeasibility")
            if had_ref:
                lines.append(f"    e.g. spokes=[{had_ref[0]['spokes']}]  "
                            f"sdp_t={had_ref[0]['sdp_t']:.6f}  "
                            f"decoded lambda_min={had_ref[0]['sdp_lambda_min']:.3e}")
        lines.append("")

    # ── Best-of-run vs. previous champion ─────────────────────────────────
    best = max(results, key=lambda r: (r['sdp_lambda_min'], r['sdp_log_product']))
    beats = best['sdp_lambda_min'] > PREVIOUS_CHAMPION_LAMBDA_MIN
    lines.append("Best size-9 subset found by this SDP sweep (max achieved "
                "lambda_min, log_product tiebreak):")
    lines.append(f"  spokes=[{best['spokes']}]  indices={best['subset']}")
    lines.append(f"  sdp_t={best['sdp_t']:.6f}  achieved lambda_min={best['sdp_lambda_min']:.6f}  "
                f"log_product={best['sdp_log_product']:.6f}")
    lines.append(f"  thetas (deg)={np.round(best['sdp_thetas_deg'], 2).tolist()}")
    lines.append("")
    lines.append(f"Previous champion (exhaustive_subset_summary.txt): "
                f"spokes=[{PREVIOUS_CHAMPION_SPOKES}]  "
                f"lambda_min={PREVIOUS_CHAMPION_LAMBDA_MIN:.6f}")
    lines.append(f"  -> {'BEATS' if beats else 'does NOT beat'} the previous champion "
                f"({best['sdp_lambda_min']:.6f} vs {PREVIOUS_CHAMPION_LAMBDA_MIN:.6f})")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
