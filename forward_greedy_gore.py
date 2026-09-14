"""
forward_greedy_gore.py

Forward (bound-guided) greedy coupling-topology construction on the
flasher-gore system (flasher_gore_system.py, m=18 candidates -- 6 interior
creases x 3 contact points each), reusing forward_greedy() from
forward_greedy_topology.py unchanged.

This is KINEMATIC_COUPLING_SUMMARY.md step 5: testing forward greedy at a
larger scale than the birds-foot m=12 case, WITHOUT exhaustive ground
truth to check against (m=18 -> C(18,k) subsets is well past brute-force
range). So there is no "% of known champion" number here the way
forward_greedy_topology.py reports for birds-foot -- the only optimality
evidence available is the certified bound gap (sdp_topology_bound.py) at
the final greedy answer, per the summary's own §4 recommendation.

K = target_rank // 2 = 15: the exact structural minimum (2 constraint
rows per coupling, target_rank=30) -- mirrors birds-foot's K=9 for
target_rank=18, where that exact minimum turned out reachable (a full-
rank reference existed) for most, not all, subsets at that size. If no
full-rank reference is found at K=15 here, that's a real (and itself
informative) result, not a bug -- rerun with K=16 rather than assuming
one.

Saves: results/forward_greedy_gore_results.json,
results/forward_greedy_gore_summary.txt

Run directly: python forward_greedy_gore.py
"""
import sys, os, json, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flasher_gore_system import (
    build_gore_system, GORE_START_THETAS_DEG, gore_spoke_name, GORE_SPOKE_ORDER,
    GORE_CREASES,
)
from forward_greedy_topology import forward_greedy
from sdp_topology_bound import build_coupling_registry, find_full_rank_reference
from bound_and_evaluate import bound_and_evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, 'forward_greedy_gore_results.json')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'forward_greedy_gore_summary.txt')

N_RESTARTS_FINAL = 5
MAX_REF_TRIES    = 300
SEED             = 17


def main():
    system = build_gore_system(np.array(GORE_START_THETAS_DEG))
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    k = int(os.environ.get('GORE_K', target_rank // 2))
    base_thetas_deg = np.array(GORE_START_THETAS_DEG)
    labels = [gore_spoke_name(c) for c in system.couplings]
    spoke_order = list(dict.fromkeys(GORE_SPOKE_ORDER))  # first-seen order, deduped

    rng = np.random.default_rng(SEED)

    # Panels reachable via exactly one crease MUST have all 3 of that
    # crease's couplings active -- it's their only path to pinning their
    # 6 relative DOF, and exactly 6 rows are available there (3 couplings
    # x 2 rows), so there's no slack to skip one. The bound-guided greedy
    # can't see this (best-case-completion optimism treats every
    # candidate as replaceable) -- confirmed empirically: unforced runs at
    # both k=15 and k=16 left the P1-P2 leaf crease (panel 2's only
    # connection) one coupling short and failed to reach full rank. Force
    # it explicitly rather than hoping greedy stumbles onto it.
    from collections import Counter
    panel_degree = Counter()
    for c in GORE_CREASES:
        panel_degree[c['panel_a']] += 1
        panel_degree[c['panel_b']] += 1

    forced_active = []
    for crease_idx, c in enumerate(GORE_CREASES):
        if panel_degree[c['panel_a']] == 1 or panel_degree[c['panel_b']] == 1:
            forced_active.extend([3 * crease_idx, 3 * crease_idx + 1, 3 * crease_idx + 2])

    print("=" * 72)
    print(f"Forward (bound-guided) greedy construction: flasher gore, "
          f"choose {k} of {m}  (target_rank={target_rank})")
    print("=" * 72)
    if forced_active:
        forced_names = sorted({gore_spoke_name(system.couplings[i]) for i in forced_active})
        print(f"Forcing {len(forced_active)} structurally-mandatory couplings active "
              f"(leaf-panel creases: {forced_names}) before greedy starts; "
              f"{k - len(forced_active)} slots left for greedy to choose among "
              f"the remaining {m - len(forced_active)} candidates.\n")

    ref_thetas = np.radians(base_thetas_deg)
    registry = build_coupling_registry(system, ref_thetas, list(range(m)), 1.0)
    print(f"Shared per-coupling registry built once for all {m} candidates.\n")

    t0 = time.perf_counter()
    greedy_result = forward_greedy(
        system, m, k, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, registry=registry, rng=rng,
        forced_active_start=forced_active)
    greedy_elapsed = time.perf_counter() - t0

    active = greedy_result['active']
    spoke_counts = {}
    for i in active:
        spoke_counts[labels[i]] = spoke_counts.get(labels[i], 0) + 1
    spoke_str = ','.join(f'{s}:{spoke_counts.get(s, 0)}' for s in spoke_order)

    print(f"\nGreedy construction finished in {greedy_elapsed:.1f}s "
          f"({greedy_result['n_unresolved_events']} unresolved-reference events "
          f"during construction)")
    print(f"Final active set: {active}  spokes=[{spoke_str}]")

    print("\nEvaluating final set via bound_and_evaluate "
          f"(certified bound + {N_RESTARTS_FINAL}-restart Riemannian achieved value)...")
    ref_rng = np.random.default_rng(SEED + 1000)
    ref = find_full_rank_reference(
        system, active, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)
    if ref is None:
        raise RuntimeError(
            f"Could not find a full-rank reference for the final greedy set "
            f"{active} at k={k} (target_rank/2, the structural minimum) -- "
            f"this is itself a real result (k={k} may not be reachable for "
            f"this topology), not necessarily a bug. Try rerunning with a "
            f"larger k before assuming something is broken.")

    final = bound_and_evaluate(system, active, length_scale=1.0,
                                target_rank=target_rank, ref_thetas=ref,
                                n_restarts=N_RESTARTS_FINAL,
                                rng=np.random.default_rng(SEED + 2000))

    lines = [
        f"Forward (bound-guided) greedy topology construction -- flasher gore, "
        f"choose {k} of {m}",
        "=" * 78,
        "",
        "Method: identical algorithm to forward_greedy_topology.py (birds-foot) "
        "-- score every remaining candidate by the CERTIFIED bound achievable "
        "by adding it, not the raw objective (identically zero below the "
        f"structural minimum k={target_rank}//2={k}). See that file's module "
        "docstring for the full caveat about how literally the classical "
        "(1-1/e) greedy guarantee applies here.",
        "",
        "No exhaustive ground truth exists at this scale (m=18) -- unlike "
        "birds-foot, there is no known champion to compare against. The only "
        "optimality evidence is the certified-bound gap below "
        "(KINEMATIC_COUPLING_SUMMARY.md step 5).",
        "",
        f"Final active set: {active}",
        f"Spoke distribution: [{spoke_str}]",
        f"Greedy construction time: {greedy_elapsed:.1f}s "
        f"({greedy_result['n_unresolved_events']} unresolved-reference events)",
        "",
        f"Certified bound at final set:  t = {final['t']:.6f}",
        f"Achieved lambda_min (Riemannian, {N_RESTARTS_FINAL} restarts): "
        f"{final['lambda_min']:.6f}",
        f"Certification gap: {final['gap']:.3e}",
        f"  (gap as % of achieved value: {100*final['gap']/final['lambda_min']:.2f}%"
        f" -- the smaller this is, the closer this greedy result is provably "
        f"to the true optimum for THIS active set, though not necessarily to "
        f"the global optimum over all C({m},{k}) subsets)",
    ]

    lines.append("")
    lines.append("Step-by-step trace:")
    for tr in greedy_result['trace']:
        lines.append(f"  step {tr['step']+1}: chose coupling {tr['chosen']} "
                    f"(spoke {labels[tr['chosen']]}), bound={tr['chosen_score']:.6f}")

    summary = "\n".join(lines)
    print("\n" + summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(
            m=m, k=k, target_rank=target_rank,
            active=active, spoke_counts=spoke_counts,
            t=final['t'], lambda_min=final['lambda_min'], gap=final['gap'],
            greedy_elapsed_sec=greedy_elapsed,
            n_unresolved_events=greedy_result['n_unresolved_events'],
            trace=greedy_result['trace'],
        ), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
