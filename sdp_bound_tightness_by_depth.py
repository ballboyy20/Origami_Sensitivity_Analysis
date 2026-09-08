"""
sdp_bound_tightness_by_depth.py

Measures how tight sdp_topology_bound's relaxed bound is as a function of
branch-and-bound tree DEPTH (number of coupling sites already decided,
active or inactive) -- the number that actually determines whether B&B is
worth building for a problem this size: if the bound stays about as loose
near the leaves as it is at the root, most branches would survive pruning
until nearly the bottom of the tree and B&B would save little over brute
force here (it may still help once patterns get big enough that brute
force isn't an option at all, since ANY valid pruning beats none -- but
that's a different question from whether THIS bound tightens usefully).

For each depth 0..12 (0 = root, all 12 undecided; 12 = fully decided, i.e.
a leaf -- 9 forced active + 3 forced inactive), samples several random
cardinality-feasible partial nodes and calls bound_node() (see
sdp_topology_bound.py) -- the reference-search-failure-safe wrapper: a
node for which no full-rank reference is found is reported UNRESOLVED and
excluded from the gap statistics, never silently treated as if it were
bounded or pruned. For every successfully BOUNDED node, the gap is the
bound minus the best CONFIRMED achieved lambda_min among all ground-truth
size-9 subsets consistent with that node's forced_active/forced_inactive
split (same two independent sources -- real differential_evolution runs
and the already-validated per-topology SDP -- used throughout this repo's
prior SDP validation work). A node with no consistent ground-truth subset
at all is skipped (nothing confirmed to compare its bound against).

Also re-confirms (loudly, not silently) that no sampled node's bound ever
falls below its comparison value -- this script generates fresh random
nodes beyond what test_sdp_topology_bound.py already checked, so a
regression here would mean either a topology change broke something or
this script's own node generation has a bug worth knowing about
immediately, not a "huh, huh" ignored one-off.

Saves:
  results/sdp_bound_tightness_by_depth.json / .csv  -- every node evaluated
  results/sdp_bound_tightness_by_depth_summary.txt  -- per-depth statistics
  results/sdp_bound_tightness_by_depth.png          -- gap-vs-depth plot

Run directly: python sdp_bound_tightness_by_depth.py
Requires results/exhaustive_subset_results.json and
results/sdp_size9_results.json to already exist (same ground truth
test_sdp_topology_bound.py uses).
"""

import sys, os, json, csv, time

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless -- this script only saves figures, never
                        # shows one interactively (avoids a background run
                        # hanging on a plt.show() with no display attached)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import BIRDSFOOT_START_THETAS_DEG
from sdp_topology_bound import bound_node
from test_sdp_topology_bound import (
    _fresh_system, _load_ground_truth, _random_partial_node,
    _consistent_subsets, M, K,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, 'sdp_bound_tightness_by_depth.json')
OUT_CSV     = os.path.join(RESULTS_DIR, 'sdp_bound_tightness_by_depth.csv')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'sdp_bound_tightness_by_depth_summary.txt')
OUT_PLOT    = os.path.join(RESULTS_DIR, 'sdp_bound_tightness_by_depth.png')

N_PER_DEPTH          = 15   # target distinct nodes sampled per depth level
MAX_ATTEMPTS_FACTOR  = 6    # give up looking for more distinct nodes at a
                             # depth after N_PER_DEPTH * this many draws
MAX_REF_TRIES        = 250
SOUNDNESS_TOL        = 1e-6
SEED                 = 7


def _sample_nodes_at_depth(rng, m, k, depth, n_target):
    """Up to n_target DISTINCT (forced_active, forced_inactive, undecided)
    triples with exactly `depth` couplings decided (deduped by assignment,
    not by whether they turn out to have consistent ground truth -- that's
    filtered by the caller)."""
    seen = set()
    nodes = []
    max_attempts = max(n_target * MAX_ATTEMPTS_FACTOR, 20)
    for _ in range(max_attempts):
        if len(nodes) >= n_target:
            break
        fa, fi, undecided = _random_partial_node(rng, m, k, depth)
        key = (tuple(fa), tuple(fi))
        if key in seen:
            continue
        seen.add(key)
        nodes.append((fa, fi, undecided))
    return nodes


def main():
    system = _fresh_system()
    target_rank = system.total_dofs - 6
    ground_truth = _load_ground_truth()
    print(f"Loaded confirmed ground truth for {len(ground_truth)}/220 size-9 "
          f"subsets\n")

    rng = np.random.default_rng(SEED)
    records = []

    depths = list(range(0, M + 1))   # 0 (root) .. 12 (fully decided / leaf)
    t_start = time.perf_counter()
    for depth in depths:
        nodes = _sample_nodes_at_depth(rng, M, K, depth, N_PER_DEPTH)
        print(f"depth={depth:2d} ({M - depth:2d} undecided): "
              f"{len(nodes)} distinct node(s) sampled")

        for fa, fi, undecided in nodes:
            consistent = _consistent_subsets(ground_truth, fa, fi)
            if not consistent:
                records.append(dict(
                    depth=depth, n_undecided=len(undecided),
                    forced_active=fa, forced_inactive=fi, undecided=undecided,
                    status='no_ground_truth', elapsed_sec=0.0,
                    bound=None, best_achievable=None, gap=None,
                    relative_gap=None, reason=None))
                continue
            best_achievable = max(v for _, v in consistent)

            node_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            t0 = time.perf_counter()
            result = bound_node(
                system, fa, fi, undecided, K, length_scale=1.0,
                target_rank=target_rank,
                base_thetas_deg=np.array(BIRDSFOOT_START_THETAS_DEG),
                rng=node_rng, max_ref_tries=MAX_REF_TRIES)
            elapsed = time.perf_counter() - t0

            rec = dict(depth=depth, n_undecided=len(undecided),
                       forced_active=fa, forced_inactive=fi, undecided=undecided,
                       status=result['status'], elapsed_sec=elapsed,
                       best_achievable=best_achievable, reason=result.get('reason'))
            if result['status'] == 'bounded':
                gap = result['bound'] - best_achievable
                rec['bound'] = result['bound']
                rec['gap'] = gap
                rec['relative_gap'] = gap / best_achievable
            else:
                rec['bound'] = None
                rec['gap'] = None
                rec['relative_gap'] = None
            records.append(rec)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll depths evaluated in {total_elapsed:.1f}s")

    # ── Save raw records ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=1)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['depth', 'n_undecided', 'forced_active', 'forced_inactive',
                    'undecided', 'status', 'bound', 'best_achievable', 'gap',
                    'relative_gap', 'reason', 'elapsed_sec'])
        for r in records:
            w.writerow([
                r['depth'], r['n_undecided'],
                ' '.join(map(str, r['forced_active'])),
                ' '.join(map(str, r['forced_inactive'])),
                ' '.join(map(str, r['undecided'])),
                r['status'],
                '' if r['bound'] is None else f"{r['bound']:.8f}",
                '' if r['best_achievable'] is None else f"{r['best_achievable']:.8f}",
                '' if r['gap'] is None else f"{r['gap']:.8f}",
                '' if r['relative_gap'] is None else f"{r['relative_gap']:.6f}",
                r['reason'] or '',
                f"{r['elapsed_sec']:.3f}",
            ])
    print(f"Saved: {OUT_JSON}\nSaved: {OUT_CSV}")

    # ── Soundness re-check on this fresh batch of random nodes -- flag
    #    loudly and stop, don't silently continue, if ever violated ───────
    bounded = [r for r in records if r['status'] == 'bounded']
    violations = [r for r in bounded if r['gap'] < -SOUNDNESS_TOL]
    if violations:
        print(f"\n*** {len(violations)} SOUNDNESS VIOLATION(S) FOUND -- see below ***")
        for r in violations:
            print(f"  depth={r['depth']} forced_active={r['forced_active']} "
                  f"forced_inactive={r['forced_inactive']} bound={r['bound']:.6f} "
                  f"achieved={r['best_achievable']:.6f} gap={r['gap']:.3e}")
        raise RuntimeError(
            f"{len(violations)} soundness violation(s) in this fresh sample -- "
            f"see printed detail above. Do not trust sdp_topology_bound until "
            f"this is understood.")

    # ── Per-depth summary ────────────────────────────────────────────────
    lines = ["SDP topology-bound tightness vs. branch-and-bound tree depth",
             "(birds-foot, m=12, k=9)", "=" * 78, ""]
    gap_by_depth = {}
    unresolved_total = infeasible_total = no_gt_total = 0
    for depth in depths:
        at_depth = [r for r in records if r['depth'] == depth]
        b   = [r for r in at_depth if r['status'] == 'bounded']
        u   = [r for r in at_depth if r['status'] == 'unresolved']
        inf = [r for r in at_depth if r['status'] == 'infeasible']
        ngt = [r for r in at_depth if r['status'] == 'no_ground_truth']
        unresolved_total += len(u)
        infeasible_total += len(inf)
        no_gt_total += len(ngt)
        gap_by_depth[depth] = [r['gap'] for r in b]

        line = (f"depth={depth:2d} ({M - depth:2d} undecided): "
                f"{len(at_depth)} sampled -> {len(b)} bounded, "
                f"{len(u)} unresolved, {len(inf)} infeasible, "
                f"{len(ngt)} no-ground-truth")
        if b:
            gaps = np.array([r['gap'] for r in b])
            rel = np.array([r['relative_gap'] for r in b])
            line += (f"\n    gap: min={gaps.min():.6f} median={np.median(gaps):.6f} "
                    f"max={gaps.max():.6f}    relative gap median="
                    f"{100 * np.median(rel):.1f}%")
        lines.append(line)

    lines.append("")
    lines.append(f"Totals: {len(bounded)} bounded, {unresolved_total} unresolved "
                f"(reference-search-failure-safe policy -- see bound_node's "
                f"docstring: NOT treated as pruned), {infeasible_total} "
                f"infeasible (cardinality alone), {no_gt_total} skipped "
                f"(no consistent ground truth)")
    lines.append(f"Total wall-clock: {total_elapsed:.1f}s")
    lines.append("")

    # A simple monotonicity read: does the median gap trend down with depth?
    depths_with_data = [d for d in depths if gap_by_depth[d]]
    medians = [float(np.median(gap_by_depth[d])) for d in depths_with_data]
    if len(medians) >= 2:
        root_median, leaf_median = medians[0], medians[-1]
        lines.append(f"Median gap at shallowest sampled depth "
                    f"({depths_with_data[0]}): {root_median:.6f}")
        lines.append(f"Median gap at deepest sampled depth "
                    f"({depths_with_data[-1]}): {leaf_median:.6f}")
        is_monotone = all(medians[i] >= medians[i+1] - 1e-9
                          for i in range(len(medians) - 1))
        lines.append(f"Monotonically non-increasing median gap across all "
                    f"sampled depths: {is_monotone}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_SUMMARY}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('SDP topology-bound tightness vs. branch-and-bound tree depth '
                '(birds-foot, k=9)', fontweight='bold')

    jitter_rng = np.random.default_rng(0)
    xs = np.array([r['depth'] for r in bounded], dtype=float)
    jitter = jitter_rng.uniform(-0.15, 0.15, size=len(bounded))
    gaps = np.array([r['gap'] for r in bounded])
    rels = 100 * np.array([r['relative_gap'] for r in bounded])

    axes[0].scatter(xs + jitter, gaps, s=18, alpha=0.5, color='#3F8EFC',
                    label='sampled nodes')
    axes[0].plot(depths_with_data, medians, color='#E74C3C', marker='o',
                linewidth=2, label='median per depth')
    axes[0].set_xlabel('depth (couplings decided: 0 = root, 12 = leaf)')
    axes[0].set_ylabel('bound - best known achieved lambda_min')
    axes[0].set_title('Absolute gap')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    rel_medians = [100 * float(np.median(
        [r['relative_gap'] for r in bounded if r['depth'] == d]))
        for d in depths_with_data]
    # A handful of nodes have a near-degenerate (barely non-degenerate)
    # ground-truth denominator, blowing relative gap up into the
    # thousands of percent -- legitimate data (see the saved CSV/JSON for
    # exact values), but it makes the plot unreadable if not clipped.
    # Clip the axis, not the data; median lines above already use every
    # point (median is robust to this on its own).
    y_cap = max(150.0, float(np.percentile(rels, 90)) * 1.3)
    n_clipped = int(np.sum(rels > y_cap))
    axes[1].scatter(xs + jitter, rels, s=18, alpha=0.5, color='#3F8EFC',
                    label='sampled nodes')
    axes[1].plot(depths_with_data, rel_medians, color='#E74C3C', marker='o',
                linewidth=2, label='median per depth')
    axes[1].set_xlabel('depth (couplings decided: 0 = root, 12 = leaf)')
    axes[1].set_ylabel('relative gap (%) = (bound - achieved) / achieved')
    title = 'Relative gap'
    if n_clipped:
        title += f' ({n_clipped} outlier(s) > {y_cap:.0f}% clipped, see CSV/JSON)'
    axes[1].set_title(title, fontsize=9)
    axes[1].set_ylim(-5, y_cap)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Saved: {OUT_PLOT}")


if __name__ == "__main__":
    main()
