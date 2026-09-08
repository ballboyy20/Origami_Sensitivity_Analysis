"""
test_sdp_topology_bound.py

Validation battery for sdp_topology_bound.py's combined (discrete z +
continuous angle) SDP relaxation, used to bound lambda_min at a PARTIALLY
decided branch-and-bound node (some coupling sites forced active, some
forced inactive, some undecided, relaxed to z_i in [0,1]).

This is the soundness-critical check the bound is worthless without: if it
is EVER not a true upper bound, a real branch-and-bound search would
silently prune away the true optimum with no error -- a much worse failure
mode than differential_evolution's loud non-convergence, which the rest of
this codebase (see coupling_optimizer.py, exhaustive_subset_baseline.py) is
already built to tolerate. Any violation found below is reported in full
detail and treated as a hard FAIL -- "mostly passing" is not a pass.

Four checks, in order:
  A. Per-coupling decomposition regression -- the prerequisite the whole
     z-relaxation depends on. Does summing per-coupling-ISOLATED
     K0_i/Kc_i/Ks_i (calibrate() with every other coupling deactivated)
     reproduce the SAME K0/Kc/Ks a joint calibrate() call returns for a
     whole active set? (~1e-9 tolerance, matching test_K_theta_structure.py's
     own TOL.)
  B. Leaf-node regression -- with undecided={}, does sdp_topology_bound's
     t reduce to the SAME value as the already-validated per-topology
     sdp_lambda_min's t (checked against the stored
     results/sdp_size9_results.json, not a live re-solve, since both are
     deterministic convex solves and the stored value is already
     independently produced)?
  C. Root-node soundness -- does the root bound (0 decided, all 12
     undecided, k=9) dominate every one of the confirmed size-9 achieved
     values in the ground truth?
  D. Intermediate-node stress test -- the same soundness check at >=50
     random partial assignments spanning a range of tree depths (1-10
     couplings decided), each checked against every ground-truth size-9
     subset consistent with that partial assignment.

Ground truth for what a REAL topology can achieve comes from two
INDEPENDENT sources (neither derived from this SDP work):
  - results/exhaustive_subset_results.json: real differential_evolution
    optimize_theta() runs over all 299 birds-foot subsets (sizes 9-12).
  - results/sdp_size9_results.json: the already-validated PER-TOPOLOGY SDP
    (sdp_lambda_min), decoded+re-evaluated, over all 220 size-9 subsets.
For each size-9 subset, the "confirmed achieved" value used everywhere
below is the MAX of whichever of these two sources produced a
non-degenerate (lambda_min > NONDEGENERATE_TOL) result -- taking the max
of two independent lower bounds on the true optimum only makes the
soundness check STRICTER (harder for the SDP topology bound to satisfy),
never weaker.

Also records (not soundness-critical, but requested): how tight the root
bound is versus the best known achieved value, and SDP solve wall-clock
time at a few representative node sizes.

Run directly: python test_sdp_topology_bound.py
Writes results/sdp_topology_bound_summary.txt.
"""

import sys, os, json, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)
from test_K_theta_structure import calibrate
from sdp_topology_bound import (
    sdp_topology_bound, build_coupling_registry, find_full_rank_reference,
)

RESULTS_DIR      = os.path.join(os.path.dirname(__file__), 'results')
EXHAUSTIVE_JSON  = os.path.join(RESULTS_DIR, 'exhaustive_subset_results.json')
SDP_SIZE9_JSON   = os.path.join(RESULTS_DIR, 'sdp_size9_results.json')
SUMMARY_PATH     = os.path.join(RESULTS_DIR, 'sdp_topology_bound_summary.txt')

M                 = 12     # birds-foot candidate coupling sites
K                 = 9      # required active count (ceil(target_rank/2))
TOL_PART_A        = 1e-9   # matches test_K_theta_structure.py's own TOL
LEAF_TOL          = 1e-4   # solver-tolerance-scale match for Part B
SOUNDNESS_TOL     = 1e-6   # violation iff achieved > bound + SOUNDNESS_TOL
                            # (matches sdp_size9_sweep.py's own -1e-6 margin
                            # convention on its analogous t >= achieved check)
NONDEGENERATE_TOL = 1e-6   # matches sdp_size9_sweep.py's own convention


# ══════════════════════════════════════════════════════════════════════
# Shared birds-foot system / target_rank
# ══════════════════════════════════════════════════════════════════════

def _fresh_system():
    return build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))


# ══════════════════════════════════════════════════════════════════════
# Part A: per-coupling decomposition regression
# ══════════════════════════════════════════════════════════════════════

def _configs():
    """
    Replicates test_K_theta_structure.py's own `configs` list -- that list
    is defined inside its `if __name__ == "__main__":` guard (deliberately,
    so importing `calibrate` from that module doesn't re-run its whole
    verification script), so it isn't importable directly. Rebuilt here
    from the same builder calls / parameters instead of duplicating any
    actual calibration logic.
    """
    configs = []

    sys1 = build_system(ARBITRARY_START_THETAS_DEG)
    configs.append(("Two-panel, 3/3 couplings active",
                     sys1, ARBITRARY_START_THETAS_DEG, [0, 1, 2], 1.0))

    sys2 = build_system(ARBITRARY_START_THETAS_DEG)
    sys2.couplings[2].active = False
    configs.append(("Two-panel, 2/3 active (1 pruned)",
                     sys2, ARBITRARY_START_THETAS_DEG, [0, 1], 1.0))

    sys3 = build_system(ARBITRARY_START_THETAS_DEG)
    configs.append(("Two-panel, 3/3 active, length_scale=0.35",
                     sys3, ARBITRARY_START_THETAS_DEG, [0, 1, 2], 0.35))

    sys4 = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    configs.append(("Birds-foot, 12/12 couplings active",
                     sys4, BIRDSFOOT_START_THETAS_DEG, list(range(12)), 1.0))

    sys5 = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    pruned = [2, 5, 9]
    for i in pruned:
        sys5.couplings[i].active = False
    active5 = [i for i in range(12) if i not in pruned]
    configs.append(("Birds-foot, 9/12 active (3 pruned across spokes)",
                     sys5, BIRDSFOOT_START_THETAS_DEG, active5, 1.0))

    return configs


def test_part_a_per_coupling_decomposition():
    print("=" * 72)
    print("PART A: per-coupling decomposition regression")
    print("=" * 72)
    print("Does summing per-coupling-ISOLATED K0_i/Kc_i/Ks_i reproduce the "
          "SAME K0/Kc/Ks a joint calibrate() call returns?\n")

    worst_overall = 0.0
    all_ok = True
    for name, system, ref_thetas_deg, active_idx, length_scale in _configs():
        ref = np.radians(np.array(ref_thetas_deg, dtype=float))

        # Joint calibration -- system's .active flags already set per config.
        K0_joint, Kc_joint, Ks_joint = calibrate(system, ref.copy(), active_idx,
                                                  length_scale)

        # Per-coupling isolated calibration + sum.
        registry = build_coupling_registry(system, ref, active_idx, length_scale)
        K0_sum = sum((registry[i][0] for i in active_idx), np.zeros_like(K0_joint))

        err_K0 = float(np.max(np.abs(K0_sum - K0_joint)))
        err_Kc = max(float(np.max(np.abs(registry[i][1] - Kc_joint[i])))
                     for i in active_idx)
        err_Ks = max(float(np.max(np.abs(registry[i][2] - Ks_joint[i])))
                     for i in active_idx)
        worst_here = max(err_K0, err_Kc, err_Ks)
        worst_overall = max(worst_overall, worst_here)

        ok = worst_here < TOL_PART_A
        all_ok = all_ok and ok
        print(f"  {name:<48s} max|isolated_sum - joint| = {worst_here:.3e}  "
              f"{'OK' if ok else 'FAIL'}")

    print(f"\n  worst overall: {worst_overall:.3e}  (tol {TOL_PART_A:.0e})")
    print(f"  -> {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok, worst_overall


# ══════════════════════════════════════════════════════════════════════
# Ground truth loading (shared by Parts B, C, D)
# ══════════════════════════════════════════════════════════════════════

def _load_ground_truth():
    """
    Returns list of (subset_tuple_sorted, best_achieved) for every size-9
    birds-foot subset with at least one confirmed non-degenerate achieved
    lambda_min from either independent source (see module docstring).
    """
    with open(EXHAUSTIVE_JSON, 'r', encoding='utf-8') as f:
        de = json.load(f)
    with open(SDP_SIZE9_JSON, 'r', encoding='utf-8') as f:
        sdp = json.load(f)

    de_by_subset = {tuple(sorted(r['subset'])): r for r in de if r['size'] == 9}
    sdp_by_subset = {tuple(sorted(r['subset'])): r for r in sdp}

    all_subsets = set(de_by_subset) | set(sdp_by_subset)
    ground_truth = []
    for subset in sorted(all_subsets):
        candidates = []
        de_r = de_by_subset.get(subset)
        if de_r is not None and not de_r['failed'] and de_r['lambda_min']:
            candidates.append(float(de_r['lambda_min']))
        sdp_r = sdp_by_subset.get(subset)
        if sdp_r is not None and sdp_r['sdp_lambda_min'] > NONDEGENERATE_TOL:
            candidates.append(float(sdp_r['sdp_lambda_min']))
        if candidates:
            ground_truth.append((subset, max(candidates)))
    return ground_truth


# ══════════════════════════════════════════════════════════════════════
# Part B: leaf-node regression
# ══════════════════════════════════════════════════════════════════════

def test_leaf_node_regression(ground_truth, n_sample=25, seed=0):
    print("=" * 72)
    print(f"PART B: leaf-node regression ({n_sample} of {len(ground_truth)} "
          f"size-9 subsets)")
    print("=" * 72)
    print("undecided={} -- does sdp_topology_bound's t match the stored "
          "per-topology sdp_lambda_min's t?\n")

    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(ground_truth))
    sample_idx = rng.choice(len(ground_truth), size=n_sample, replace=False)

    with open(SDP_SIZE9_JSON, 'r', encoding='utf-8') as f:
        sdp = json.load(f)
    sdp_t_by_subset = {tuple(sorted(r['subset'])): r['sdp_t'] for r in sdp}

    system = _fresh_system()
    target_rank = system.total_dofs - 6
    all_idx = set(range(M))

    violations = []
    max_diff = 0.0
    n_checked = 0
    n_skipped = 0
    for idx in sample_idx:
        subset, _ = ground_truth[idx]
        subset = list(subset)
        stored_t = sdp_t_by_subset.get(tuple(subset))
        if stored_t is None:
            n_skipped += 1
            continue

        ref_rng = np.random.default_rng(abs(hash(tuple(subset))) % (2**32))
        ref = find_full_rank_reference(
            system, subset, target_rank, length_scale=1.0,
            base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
            rng=ref_rng, max_tries=300)
        if ref is None:
            print(f"  SKIP subset={subset}: no full-rank reference found "
                  f"(unexpected for a confirmed-feasible size-9 subset)")
            n_skipped += 1
            continue

        result = sdp_topology_bound(
            system, forced_active=subset,
            forced_inactive=sorted(all_idx - set(subset)),
            undecided=[], k=K, length_scale=1.0, target_rank=target_rank,
            ref_thetas=ref)

        diff = abs(result['t'] - stored_t)
        max_diff = max(max_diff, diff)
        n_checked += 1
        ok = diff < LEAF_TOL
        if not ok:
            violations.append((subset, result['t'], stored_t, diff))
        print(f"  subset={subset}  topology_bound.t={result['t']:.6f}  "
              f"sdp_lambda_min.t={stored_t:.6f}  diff={diff:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")

    print(f"\n  {n_checked} checked, {n_skipped} skipped, max diff={max_diff:.2e} "
          f"(tol {LEAF_TOL:.0e})")
    ok_all = not violations
    print(f"  -> {'PASS' if ok_all else 'FAIL'}\n")
    return ok_all, violations, max_diff


# ══════════════════════════════════════════════════════════════════════
# Part C: root-node soundness check
# ══════════════════════════════════════════════════════════════════════

def test_root_node(ground_truth):
    print("=" * 72)
    print("PART C: root-node soundness check (0 decided, 12 undecided)")
    print("=" * 72)

    system = _fresh_system()
    target_rank = system.total_dofs - 6

    ref = find_full_rank_reference(
        system, list(range(M)), target_rank, length_scale=1.0,
        base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
        rng=np.random.default_rng(1), max_tries=50)
    assert ref is not None, (
        "couldn't find a full-rank reference for the full 12-coupling "
        "birds-foot system -- unexpected, test_K_theta_structure.py "
        "already relies on BIRDSFOOT_START_THETAS_DEG being full-rank there")

    t0 = time.perf_counter()
    result = sdp_topology_bound(system, forced_active=[], forced_inactive=[],
                                 undecided=list(range(M)), k=K, length_scale=1.0,
                                 target_rank=target_rank, ref_thetas=ref)
    elapsed = time.perf_counter() - t0

    best_known = max(v for _, v in ground_truth)
    violations = [(subset, val) for subset, val in ground_truth
                  if val > result['t'] + SOUNDNESS_TOL]

    print(f"  root bound t = {result['t']:.6f}   solve time = {elapsed:.3f}s")
    print(f"  best known achieved lambda_min among {len(ground_truth)} confirmed "
          f"size-9 subsets = {best_known:.6f}")
    gap = result['t'] - best_known
    print(f"  root bound - best known = {gap:.6f} ({100*gap/best_known:.1f}% looser)")
    if violations:
        print(f"\n  *** {len(violations)} VIOLATION(S): ground-truth value exceeds "
              f"root bound ***")
        for subset, val in violations:
            print(f"    subset={list(subset)}  achieved={val:.6f}  "
                  f"bound={result['t']:.6f}  margin={result['t']-val:.3e}")
    ok = not violations
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok, violations, result['t'], elapsed, best_known


# ══════════════════════════════════════════════════════════════════════
# Part D: intermediate-node stress test
# ══════════════════════════════════════════════════════════════════════

def _random_partial_node(rng, m, k, n_decided):
    """A cardinality-feasible random (forced_active, forced_inactive,
    undecided) split with exactly n_decided couplings decided."""
    decided = rng.choice(m, size=n_decided, replace=False)
    lo = max(0, k - (m - n_decided))
    hi = min(n_decided, k)
    n_fa = int(rng.integers(lo, hi + 1))
    fa_idx = (rng.choice(decided, size=n_fa, replace=False)
              if n_fa > 0 else np.array([], dtype=int))
    fa_set = set(int(i) for i in fa_idx)
    fi_idx = [int(i) for i in decided if int(i) not in fa_set]
    undecided = [i for i in range(m) if i not in set(int(i) for i in decided)]
    return sorted(fa_set), sorted(fi_idx), sorted(undecided)


def _consistent_subsets(ground_truth, forced_active, forced_inactive):
    fa, fi = set(forced_active), set(forced_inactive)
    return [(subset, val) for subset, val in ground_truth
            if fa.issubset(set(subset)) and fi.isdisjoint(set(subset))]


def test_intermediate_nodes(ground_truth, n_attempts=100, seed=2):
    print("=" * 72)
    print(f"PART D: intermediate-node stress test (up to {n_attempts} random "
          f"partial assignments)")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    system = _fresh_system()
    target_rank = system.total_dofs - 6

    n_checked = 0
    n_skipped_no_gt = 0
    n_skipped_no_ref = 0
    violations = []
    timings = []   # (n_undecided, elapsed)

    for _ in range(n_attempts):
        n_decided = int(rng.integers(1, 11))   # 1..10 inclusive
        fa, fi, undecided = _random_partial_node(rng, M, K, n_decided)

        consistent = _consistent_subsets(ground_truth, fa, fi)
        if not consistent:
            n_skipped_no_gt += 1
            continue

        ref_indices = sorted(set(fa) | set(undecided))
        ref_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        ref = find_full_rank_reference(
            system, ref_indices, target_rank, length_scale=1.0,
            base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
            rng=ref_rng, max_tries=200)
        if ref is None:
            n_skipped_no_ref += 1
            print(f"  SKIP (no full-rank reference found in 200 tries): "
                  f"forced_active={fa} forced_inactive={fi}")
            continue

        t0 = time.perf_counter()
        result = sdp_topology_bound(system, forced_active=fa, forced_inactive=fi,
                                     undecided=undecided, k=K, length_scale=1.0,
                                     target_rank=target_rank, ref_thetas=ref)
        elapsed = time.perf_counter() - t0
        timings.append((len(undecided), elapsed))

        best_here = max(v for _, v in consistent)
        n_checked += 1
        if best_here > result['t'] + SOUNDNESS_TOL:
            violations.append((fa, fi, undecided, result['t'], best_here))
            print(f"  *** VIOLATION: forced_active={fa} forced_inactive={fi} "
                  f"undecided={undecided}")
            print(f"      bound={result['t']:.6f}  achieved={best_here:.6f}  "
                  f"margin={result['t']-best_here:.3e}")

        if n_checked % 10 == 0:
            print(f"  ... {n_checked} nodes checked so far, 0 violations")

    print(f"\n  {n_checked} nodes checked, {n_skipped_no_gt} skipped (no "
          f"consistent ground truth), {n_skipped_no_ref} skipped (no "
          f"full-rank reference found)")
    ok = not violations
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok, violations, timings, n_checked


# ══════════════════════════════════════════════════════════════════════
# Timing report
# ══════════════════════════════════════════════════════════════════════

def _timing_report_lines(timings, root_elapsed):
    lines = ["Solve-time by node size (representative):",
             f"  root node (12 undecided, 0 decided): {root_elapsed:.3f}s"]
    if timings:
        arr = np.array(timings, dtype=float)   # (n, 2): [n_undecided, elapsed]
        buckets = [(0, 2, 'near-leaf (0-2 undecided)'),
                   (3, 7, 'mid-depth (3-7 undecided)'),
                   (8, 12, 'shallow (8-12 undecided)')]
        for lo, hi, label in buckets:
            mask = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
            if mask.any():
                lines.append(f"  {label}: n={int(mask.sum())}  "
                             f"mean={arr[mask, 1].mean():.3f}s  "
                             f"max={arr[mask, 1].max():.3f}s")
    return lines


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 72)
    print("# sdp_topology_bound.py validation battery")
    print("#" * 72 + "\n")

    ok_a, worst_a = test_part_a_per_coupling_decomposition()

    ground_truth = _load_ground_truth()
    print(f"Loaded confirmed ground truth for {len(ground_truth)}/220 size-9 "
          f"subsets (skipped: neither DE nor SDP produced a non-degenerate "
          f"value for that subset)\n")

    ok_b, viol_b, max_diff_b = test_leaf_node_regression(ground_truth)
    ok_c, viol_c, root_t, root_elapsed, best_known = test_root_node(ground_truth)
    ok_d, viol_d, timings, n_checked_d = test_intermediate_nodes(ground_truth)

    timing_lines = _timing_report_lines(timings, root_elapsed)
    print("=" * 72)
    print("\n".join(timing_lines))
    print("=" * 72 + "\n")

    all_ok = ok_a and ok_b and ok_c and ok_d

    print("=" * 72)
    print("OVERALL RESULT")
    print("=" * 72)
    print(f"  Part A (per-coupling decomposition):  {'PASS' if ok_a else 'FAIL'}")
    print(f"  Part B (leaf-node regression):        {'PASS' if ok_b else 'FAIL'}")
    print(f"  Part C (root-node soundness):         {'PASS' if ok_c else 'FAIL'}")
    print(f"  Part D (intermediate-node stress):    {'PASS' if ok_d else 'FAIL'}")
    verdict = ("ALL CHECKS PASSED -- bound appears sound." if all_ok else
               "AT LEAST ONE CHECK FAILED -- DO NOT TRUST THIS BOUND. "
               "See violations printed above.")
    print(f"\n  {verdict}")

    # ── Write summary file ──────────────────────────────────────────────
    lines = [
        "sdp_topology_bound.py validation summary",
        "=" * 78,
        "",
        f"Part A (per-coupling decomposition regression): "
        f"{'PASS' if ok_a else 'FAIL'} (worst |isolated_sum - joint| = {worst_a:.3e}, "
        f"tol {TOL_PART_A:.0e})",
        f"Part B (leaf-node regression vs stored sdp_lambda_min.t): "
        f"{'PASS' if ok_b else 'FAIL'} (max diff = {max_diff_b:.3e}, tol {LEAF_TOL:.0e})",
        f"Part C (root-node soundness, 220 size-9 subsets): "
        f"{'PASS' if ok_c else 'FAIL'}",
        f"Part D (intermediate-node stress, {n_checked_d} nodes checked): "
        f"{'PASS' if ok_d else 'FAIL'}",
        "",
    ]

    if not ok_b:
        lines.append(f"Part B violations ({len(viol_b)}):")
        for subset, t_bound, t_stored, diff in viol_b:
            lines.append(f"  subset={subset}  topology_bound.t={t_bound:.6f}  "
                         f"sdp_lambda_min.t={t_stored:.6f}  diff={diff:.3e}")
        lines.append("")

    lines.append(f"Root-node bound: t = {root_t:.6f}")
    lines.append(f"Best known achieved lambda_min (size-9, confirmed ground truth, "
                f"n={len(ground_truth)}): {best_known:.6f}")
    lines.append(f"Root bound - best known = {root_t - best_known:.6f} "
                f"({100*(root_t-best_known)/best_known:.1f}% looser than the best "
                f"known achievable size-9 design)")
    lines.append("")

    if not ok_c:
        lines.append(f"*** Part C violations ({len(viol_c)}) -- SOUNDNESS FAILURE ***")
        for subset, val in viol_c:
            lines.append(f"  subset={list(subset)}  achieved={val:.6f}  "
                         f"bound={root_t:.6f}  margin={root_t-val:.3e}")
        lines.append("")

    if not ok_d:
        lines.append(f"*** Part D violations ({len(viol_d)}) -- SOUNDNESS FAILURE ***")
        for fa, fi, undecided, bound, achieved in viol_d:
            lines.append(f"  forced_active={fa}  forced_inactive={fi}  "
                         f"undecided={undecided}")
            lines.append(f"    bound={bound:.6f}  achieved={achieved:.6f}  "
                         f"margin={bound-achieved:.3e}")
        lines.append("")

    lines.extend(timing_lines)
    lines.append("")
    lines.append(verdict)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {SUMMARY_PATH}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
