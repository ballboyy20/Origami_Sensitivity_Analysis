"""
test_arc_refine.py

Soundness checks for the arc cuts in sdp_topology_bound(..., arcs=...) and
for arc_refine.LeafRefiner, on birds-foot leaves, all checked against the
REAL system (true_lambda_min builds C from scratch):

  1. No regression: arcs=None and all-full-circle arcs give the same bound
     as before.
  2. Validity: for random arcs, the bound is >= the true lambda_min at
     every one of many angle vectors sampled INSIDE those arcs.
  3. Monotonicity: halving an arc never raises the bound.
  4. Projection: project_to_arc lands on the arc and is the nearest point.
  5. Refinement soundness: refined leaf bounds are >= what Riemannian
     achieved on that leaf (exhaustive sweep values).

Run directly: python test_arc_refine.py
"""

import sys, os, json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
from swap_search import SetEvaluator
from arc_refine import LeafRefiner, true_lambda_min, project_to_arc, halves, TWO_PI

FAILURES = []
LOOSE_LEAVES = [[0, 1, 2, 4, 5, 7, 8, 10, 11], [0, 1, 2, 3, 5, 6, 8, 10, 11],
                [0, 1, 2, 3, 5, 7, 8, 10, 11]]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    tr = system.total_dofs - 6
    ev = SetEvaluator(system, tr, BIRDSFOOT_START_THETAS_DEG, seed=31)
    lr = LeafRefiner(system, ev.registry, tr)
    rng = np.random.default_rng(5)
    with open(os.path.join(os.path.dirname(__file__), 'results',
                           'bound_and_evaluate_sweep_results.json'), encoding='utf-8') as f:
        sweep = {tuple(r['subset']): r for r in json.load(f)}

    print("1. No regression")
    worst = 0.0
    for leaf in LOOSE_LEAVES:
        st, b_old, ref = ev.bound(leaf)
        b_none, _ = lr.bound(leaf, ref, None)
        b_full, _ = lr.bound(leaf, ref, {i: (rng.uniform(0, TWO_PI), TWO_PI) for i in leaf})
        worst = max(worst, abs(b_none - b_old) / b_old, abs(b_full - b_old) / b_old)
    check("arcs=None and full-circle arcs match the original bound", worst < 1e-6,
          f"(max rel diff {worst:.1e})")

    print("2. Validity under random arcs")
    n_viol, n_pts, min_margin = 0, 0, np.inf
    for leaf in LOOSE_LEAVES:
        _, _, ref = ev.bound(leaf)
        for _ in range(6):
            arcs = {i: (rng.uniform(0, TWO_PI), rng.uniform(0.2, 3.0)) for i in leaf}
            b, _ = lr.bound(leaf, ref, arcs)
            for _ in range(150):
                alpha = np.array([arcs[i][0] + rng.uniform(-0.5, 0.5) * arcs[i][1]
                                  for i in leaf])
                lam = true_lambda_min(system, leaf, (alpha / 2) % np.pi, tr)
                n_pts += 1
                min_margin = min(min_margin, b - lam)
                n_viol += lam > b + 1e-7
    check("bound >= true lambda_min at angles sampled inside the arcs", n_viol == 0,
          f"({n_pts} samples, {n_viol} violations, min margin {min_margin:.2e})")

    print("3. Monotonicity")
    n_bad, n = 0, 0
    for leaf in LOOSE_LEAVES:
        _, _, ref = ev.bound(leaf)
        arcs = {i: (rng.uniform(0, TWO_PI), rng.uniform(1.0, 4.0)) for i in leaf}
        b_parent, _ = lr.bound(leaf, ref, arcs)
        for i in leaf[:4]:
            for h in halves(arcs[i]):
                child = dict(arcs); child[i] = h
                b_child, _ = lr.bound(leaf, ref, child)
                n += 1
                n_bad += b_child > b_parent + 1e-7
    check("halving an arc never raises the bound", n_bad == 0, f"({n} children)")

    print("4. Projection")
    bad = 0
    for _ in range(500):
        arc = (rng.uniform(0, TWO_PI), rng.uniform(0.1, 6.0))
        x = rng.uniform(-1, 1, 2)
        p, alpha = project_to_arc(x, arc)
        grid = arc[0] + np.linspace(-arc[1] / 2, arc[1] / 2, 2001)
        pts = np.column_stack([np.cos(grid), np.sin(grid)])
        bad += np.linalg.norm(x - p) > np.min(np.linalg.norm(pts - x, axis=1)) + 1e-3
    check("projection is the nearest arc point", bad == 0, "(500 random cases)")

    print("5. Refinement soundness (loose leaves vs. sweep's achieved values)")
    incumbent = 0.009263
    for leaf in LOOSE_LEAVES:
        _, b0, ref = ev.bound(leaf)
        r = lr.refine(leaf, ref, incumbent, max_nodes=60)
        achieved = sweep[tuple(leaf)]['lambda_min']
        check(f"leaf {leaf}: refined bound >= achieved",
              r['bound'] >= achieved - 1e-7,
              f"(bound {b0:.6f} -> {r['bound']:.6f}, achieved {achieved:.6f}, "
              f"proved={r['proved']}, {r['nodes']} nodes, {r['solves']} solves, "
              f"{r['halves_removed']} halves removed by tightening, {r['elapsed_sec']:.0f}s)")

    print("=" * 72)
    print("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
