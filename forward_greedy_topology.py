"""
forward_greedy_topology.py

Forward (additive) greedy construction of a k-of-m coupling topology --
the counterpart to CouplingOptimizer.prune_couplings()'s BACKWARD
(deletion) greedy already in this codebase. These are genuinely different
algorithms, not the same one run in reverse, and it matters which one was
actually tried: prune_couplings() reached only 48.8% of the known birds-
foot size-9 champion (0.004521 vs 0.009259) -- but the classical
(1-1/e) approximation guarantee for greedy cardinality-constrained
submodular maximization (Nemhauser, Wolsey & Fisher 1978) applies to
FORWARD greedy (sequential addition), not backward elimination. That
guarantee has never actually been tested on this problem until this file.

IMPORTANT CAVEAT, stated up front rather than glossed over: this is NOT a
strict textbook application of that theorem. The true objective (achieved
lambda_min) is IDENTICALLY ZERO for every active set smaller than k=9
here (target_rank=18 needs at least 9 couplings -- 2 constraint rows each
-- just to reach full rank at all), so a literal forward greedy driven by
the raw objective has no signal to act on until the very last step. This
implementation instead scores each candidate ADDITION by the CERTIFIED
UPPER BOUND (sdp_topology_bound.py's bound_node -- already validated
sound, zero violations across hundreds of nodes) achievable by the BEST-
CASE completion of "current picks + this candidate" -- an admissible,
optimistic look-ahead heuristic, analogous to bound-guided construction
heuristics used elsewhere in combinatorial optimization when the true
objective isn't evaluable on partial solutions. This is a principled,
citable pattern, but it means the classical theorem's exact hypotheses
(direct evaluation of a monotone submodular objective at every step)
don't strictly apply -- treat this as an empirical test of whether bound-
guided greedy construction works well in practice for THIS problem, not
as a rigorous invocation of the (1-1/e) guarantee.

Once the k=9 set is fully built, the final reported achieved lambda_min
comes from bound_and_evaluate() (multi-start Riemannian gradient
sampling) -- the same validated evaluation method used everywhere else in
this project, not the SDP bound or decode.

This is also the first real exercise of sdp_topology_bound.py's `registry`
parameter (built but never actually tested before) -- a single per-
coupling registry covering all m candidates is computed ONCE and reused
across every bound_node() call in the greedy construction (since
forced_inactive is always empty during pure forward-additive
construction, forced_active UNION undecided is always the full candidate
set, so one shared registry suffices for the whole run). Verified to give
identical bounds to the non-shared path before being trusted here.

Saves: results/forward_greedy_topology_results.json,
results/forward_greedy_topology_summary.txt

Run directly: python forward_greedy_topology.py
"""

import sys, os, json, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name,
)
from sdp_topology_bound import bound_node, build_coupling_registry
from bound_and_evaluate import bound_and_evaluate

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON      = os.path.join(RESULTS_DIR, 'forward_greedy_topology_results.json')
OUT_SUMMARY   = os.path.join(RESULTS_DIR, 'forward_greedy_topology_summary.txt')

K                = 9
N_RESTARTS_FINAL = 5
MAX_REF_TRIES    = 300
SEED             = 17

# Known reference points (from prior work on this repo, see
# KINEMATIC_COUPLING_SUMMARY.md and results/exhaustive_subset_summary.txt)
KNOWN_CHAMPION_LAMBDA_MIN = 0.009259
KNOWN_CHAMPION_SPOKES     = 'O-M:3,O-B:2,O-N:3,O-A:1'
BACKWARD_GREEDY_LAMBDA_MIN = 0.004521
BACKWARD_GREEDY_SPOKES     = 'O-M:3,O-B:3,O-N:1,O-A:2'


def forward_greedy(system, m, k, target_rank, length_scale=1.0,
                    base_thetas_deg=None, registry=None, rng=None,
                    max_ref_tries=MAX_REF_TRIES, verbose=True):
    """
    Bound-guided forward greedy: start from an empty active set, repeatedly
    add whichever remaining candidate yields the highest CERTIFIED bound
    (via bound_node(), best-case completion assumption) once added, until
    k couplings are active. See module docstring for why the bound, not
    the raw (identically-zero-until-k) objective, is used to score
    candidates.

    A candidate whose addition makes bound_node() report 'unresolved'
    (no full-rank reference found within max_ref_tries -- see bound_node's
    own docstring on why this must NOT be silently treated as
    infeasible/rejected) is scored -inf for THIS STEP's comparison only --
    it remains eligible at later steps, where a different set of already-
    committed couplings may make a reference easier to find. Logged, not
    silently dropped.

    Returns dict: 'active' (list[int], the final k-element set, in the
    order chosen), 'trace' (list of per-step dicts: step index, chosen
    candidate, its bound score, and every candidate's score that step),
    'n_unresolved_events' (diagnostic count).
    """
    if rng is None:
        rng = np.random.default_rng()
    if registry is None:
        ref_thetas = (np.radians(base_thetas_deg) if base_thetas_deg is not None
                      else np.array([c.theta for c in system.couplings]))
        registry = build_coupling_registry(system, ref_thetas, list(range(m)), length_scale)

    active = []
    remaining = list(range(m))
    trace = []
    n_unresolved = 0

    for step in range(k):
        scores = {}
        for candidate in remaining:
            trial_active = sorted(active + [candidate])
            trial_undecided = [i for i in remaining if i != candidate]
            result = bound_node(
                system, forced_active=trial_active, forced_inactive=[],
                undecided=trial_undecided, k=k, length_scale=length_scale,
                target_rank=target_rank, base_thetas_deg=base_thetas_deg,
                registry=registry, max_ref_tries=max_ref_tries, rng=rng)
            if result['status'] == 'bounded':
                scores[candidate] = result['bound']
            elif result['status'] == 'unresolved':
                n_unresolved += 1
                scores[candidate] = float('-inf')
            else:  # 'infeasible' -- shouldn't happen by construction, but don't crash
                scores[candidate] = float('-inf')

        best_candidate = max(scores, key=scores.get)
        best_score = scores[best_candidate]
        active.append(best_candidate)
        remaining.remove(best_candidate)

        if verbose:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            print(f"  step {step + 1}/{k}: chose coupling {best_candidate} "
                  f"(bound={best_score:.6f}) -- active now {sorted(active)}")
            print(f"      full ranking this step: "
                  f"{[(c, round(s, 6)) for c, s in ranked]}")

        trace.append(dict(step=step, chosen=best_candidate, chosen_score=best_score,
                          all_scores={str(c): s for c, s in scores.items()}))

    return dict(active=sorted(active), trace=trace, n_unresolved_events=n_unresolved)


def main():
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    base_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG)
    labels = [birdsfoot_spoke_name(c) for c in system.couplings]

    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print(f"Forward (bound-guided) greedy construction: birds-foot, "
          f"choose {K} of {m}")
    print("=" * 72)

    ref_thetas = np.radians(base_thetas_deg)
    registry = build_coupling_registry(system, ref_thetas, list(range(m)), 1.0)
    print(f"Shared per-coupling registry built once for all {m} candidates.\n")

    t0 = time.perf_counter()
    greedy_result = forward_greedy(
        system, m, K, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, registry=registry, rng=rng)
    greedy_elapsed = time.perf_counter() - t0

    active = greedy_result['active']
    spoke_counts = {}
    for i in active:
        spoke_counts[labels[i]] = spoke_counts.get(labels[i], 0) + 1
    spoke_str = ','.join(f'{s}:{spoke_counts.get(s, 0)}' for s in ('O-M', 'O-B', 'O-N', 'O-A'))

    print(f"\nGreedy construction finished in {greedy_elapsed:.1f}s "
          f"({greedy_result['n_unresolved_events']} unresolved-reference events "
          f"during construction)")
    print(f"Final active set: {active}  spokes=[{spoke_str}]")

    # ── Final evaluation via the validated split-responsibility pipeline ──
    print("\nEvaluating final set via bound_and_evaluate "
          f"(certified bound + {N_RESTARTS_FINAL}-restart Riemannian achieved value)...")
    ref_rng = np.random.default_rng(SEED + 1000)
    from sdp_topology_bound import find_full_rank_reference
    ref = find_full_rank_reference(
        system, active, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)
    if ref is None:
        raise RuntimeError(
            f"Could not find a full-rank reference for the final greedy set "
            f"{active} -- unexpected given it was built via bound_node calls "
            f"that themselves require one.")

    final = bound_and_evaluate(system, active, length_scale=1.0,
                                target_rank=target_rank, ref_thetas=ref,
                                n_restarts=N_RESTARTS_FINAL,
                                rng=np.random.default_rng(SEED + 2000))

    pct_of_champion = 100 * final['lambda_min'] / KNOWN_CHAMPION_LAMBDA_MIN
    pct_vs_backward = 100 * final['lambda_min'] / BACKWARD_GREEDY_LAMBDA_MIN

    lines = [
        "Forward (bound-guided) greedy topology construction -- birds-foot, "
        f"choose {K} of {m}",
        "=" * 78,
        "",
        "Method: at each step, score every remaining candidate by the CERTIFIED "
        "bound (bound_node, best-case completion) achievable by adding it -- "
        "NOT the raw objective, which is identically zero for any subset "
        "smaller than k=9 here (target_rank=18 needs >= 9 couplings to reach "
        "full rank at all). See module docstring for the honest caveat this "
        "implies about how literally the classical (1-1/e) greedy guarantee "
        "applies here.",
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
        "",
        "Comparison:",
        f"  Known size-9 champion (exhaustive search): "
        f"{KNOWN_CHAMPION_LAMBDA_MIN:.6f}  spokes=[{KNOWN_CHAMPION_SPOKES}]",
        f"  Backward greedy (prune_couplings, already in this codebase): "
        f"{BACKWARD_GREEDY_LAMBDA_MIN:.6f}  spokes=[{BACKWARD_GREEDY_SPOKES}]",
        f"  Forward (bound-guided) greedy (this run): {final['lambda_min']:.6f}",
        "",
        f"Forward greedy reaches {pct_of_champion:.1f}% of the known champion "
        f"(backward greedy reached {100*BACKWARD_GREEDY_LAMBDA_MIN/KNOWN_CHAMPION_LAMBDA_MIN:.1f}%)",
        f"Forward greedy vs. backward greedy: {pct_vs_backward:.1f}% "
        f"({'better' if final['lambda_min'] > BACKWARD_GREEDY_LAMBDA_MIN else 'worse'})",
        "",
        f"Classical worst-case floor for monotone-submodular forward greedy: "
        f"{100*(1-1/np.e):.1f}% (1-1/e) -- NOTE this floor's hypotheses do not "
        f"strictly apply here (bound-guided proxy objective, not direct "
        f"evaluation -- see caveat above); shown for context, not as a "
        f"guarantee this run must satisfy.",
    ]

    # ── Per-step trace ────────────────────────────────────────────────────
    lines.append("")
    lines.append("Step-by-step trace:")
    for t in greedy_result['trace']:
        lines.append(f"  step {t['step']+1}: chose coupling {t['chosen']} "
                    f"(spoke {labels[t['chosen']]}), bound={t['chosen_score']:.6f}")

    summary = "\n".join(lines)
    print("\n" + summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(
            active=active, spoke_counts=spoke_counts,
            t=final['t'], lambda_min=final['lambda_min'], gap=final['gap'],
            greedy_elapsed_sec=greedy_elapsed,
            n_unresolved_events=greedy_result['n_unresolved_events'],
            trace=greedy_result['trace'],
            pct_of_champion=pct_of_champion,
        ), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
