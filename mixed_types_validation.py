"""
mixed_types_validation.py

Validates mixed coupling types (flat / V-groove / tetrahedral socket per
site, exact row budget) in the bound and the branch-and-bound -- items 3-4
of the Phase 1 validation plan (KINEMATIC_COUPLING_SUMMARY.md sec 8.4).
Items 1-2 (per-type rows, Kelvin/Maxwell rank, affine structure,
regression) are in test_coupling_types.py.

Parts (select with argv[1]: 'two-panel' (default), 'birds-foot', 'all'):

  two-panel, 3 sites   Exhaustive: all 10 exact-6-row designs evaluated;
                       B&B (no incumbent, no symmetry) must find the best.
  two-panel, 6 sites   Exhaustive: all 336 designs. B&B without and with
                       (lifted, verified) symmetry must find the best.
                       Soundness: for random partial nodes, the mixed-type
                       bound must be >= the best exhaustive value among the
                       node's completions.
  birds-foot, 12 sites No exhaustive ground truth (1.7M designs). B&B with
                       the O-B <-> O-A mirror, incumbent = the all-V
                       champion. Then a random-sample check: no sampled
                       valid design may beat B&B's answer.

All weights are equal (one unit-weight row per contact normal); the
socket's walls are at 45 deg like the groove's.

Saves: results/mixed_types_validation_{two_panel,birdsfoot}_{results.json,summary.txt}

Run: python -u mixed_types_validation.py [two-panel|birds-foot|all]
"""

import sys, os, json, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (build_system, ARBITRARY_START_THETAS_DEG,
                                   build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG)
from interactive_optimizer import t as PANEL_T
from two_panel_six_sites import build_six_site_system, START_THETAS_DEG as TP6_THETAS
from coupling_types import TypedCandidates
from swap_search import SetEvaluator
from branch_and_bound import BranchAndBound, verify_symmetry
from sdp_topology_bound import bound_node

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED     = 31
OPT_TOL  = 1e-4
N_SOUND  = 120        # random partial nodes for the soundness check
N_SAMPLE = 300        # random designs for the birds-foot sample check
BF_CHAMPION_SITES = [0, 1, 2, 3, 5, 6, 7, 8, 11]


def make_evaluator(tc):
    target_rank = tc.system.total_dofs - 6
    return SetEvaluator(tc.system, target_rank, tc.base_thetas_deg, seed=SEED,
                        rows=tc.rows, sites=tc.site_of)


def exhaustive(tc, ev):
    """{design tuple: lambda_min (0 if infeasible)} over every design."""
    out = {}
    for d in tc.enumerate_designs(ev.target_rank):
        r = ev.evaluate(d)
        out[tuple(d)] = r['lambda_min'] if r['feasible'] else 0.0
    return out


def run_bnb(tc, ev, opt, symmetries=(), incumbent=None, tag=''):
    bnb = BranchAndBound(ev, symmetries=list(symmetries), seed=SEED, verbose=False,
                         rows=tc.rows, site_of=tc.site_of)
    r = bnb.run(incumbent_active=incumbent)
    s = r['stats']
    passed = None
    if opt is not None:
        passed = bool(r['complete'] and r['lambda_min'] >= opt * (1 - OPT_TOL)
                      and r['certified_upper_bound'] >= opt * (1 - 1e-9))
    lines = [
        f"   {tag}: {'' if passed is None else ('PASS  ' if passed else 'FAIL  ')}"
        f"found {r['lambda_min']:.6f} [{tc.describe(r['active'] or [], 3)}]; "
        f"certified UB {r['certified_upper_bound']:.6f} "
        f"(gap {100 * r['certified_gap'] / max(r['lambda_min'], 1e-12):.3f}%), "
        f"complete {r['complete']}",
        f"      nodes {s['nodes']}; pruned: bound {s['pruned_bound']}, symmetry "
        f"{s['pruned_symmetry']}, rank {s['pruned_rank']}, cut {s['pruned_cut']}, "
        f"cardinality {s['pruned_cardinality']}; sdp errors {s['sdp_errors']}",
        f"      leaves {s['leaves']}: {s['leaves_pruned_bound']} pruned by bound, "
        f"{s['leaves_infeasible']} infeasible, {s['leaves_evaluated']} evaluated; "
        f"{r['elapsed_sec']:.0f}s"]
    r['passed'] = passed
    return r, lines


def kind_summary(tc, truth, ev, top=8):
    ranked = sorted(truth.items(), key=lambda kv: -kv[1])
    feas = sum(1 for v in truth.values() if v > 0)
    mult = [d for d in truth if ev.evaluate(list(d)).get('error') == 'multiplicity']
    lines = [f"   {len(truth)} designs, {feas} feasible (full rank), {len(mult)} lost to "
             f"MultiplicityError{' -- ground truth incomplete' if mult else ''}. Best {top}:"]
    for d, lam in ranked[:top]:
        lines.append(f"      {lam:.6f}  [{tc.describe(d, 3)}]")
    return lines


def soundness(tc, ev, truth, n_nodes, rng):
    """Random partial nodes: mixed bound >= best completion's value."""
    m, worst, n_checked, n_infeas, violations = tc.m, np.inf, 0, 0, 0
    designs = [set(d) for d in truth]
    for _ in range(n_nodes):
        d = list(designs[rng.integers(len(designs))])     # a design to stay consistent with
        fa = [i for i in d if rng.random() < 0.4]
        others = [i for i in range(m) if i not in d]
        fi = [i for i in others if rng.random() < 0.4]
        fi += [j for i in fa for j in range(m) if j != i and tc.site_of[j] == tc.site_of[i]
               and j not in fi]
        fi = sorted(set(fi) - set(fa))
        und = [i for i in range(m) if i not in fa and i not in fi]
        best = max((v for dd, v in truth.items()
                    if set(fa) <= set(dd) and not (set(fi) & set(dd))), default=0.0)
        r = bound_node(tc.system, fa, fi, und, None, target_rank=ev.target_rank,
                       base_thetas_deg=tc.base_thetas_deg, registry=ev.registry,
                       rng=np.random.default_rng([SEED, n_checked]),
                       rows=tc.rows, sites=tc.site_of)
        if r['status'] != 'bounded':
            n_infeas += 1
            if best > 0:
                violations += 1      # pruned something with a feasible completion
            continue
        n_checked += 1
        worst = min(worst, r['bound'] - best)
        if r['bound'] < best * (1 - 1e-6):
            violations += 1
    return dict(n_bounded=n_checked, n_not_bounded=n_infeas,
                min_margin=float(worst), violations=violations)


def two_panel():
    L = ["Mixed coupling types: two-panel validation (exhaustive ground truth)", "=" * 78,
         "Types per site: flat (1 row), V-groove (2), tetrahedral socket (3); exactly 6 rows;"
         " equal unit weights; socket walls at 45 deg.", ""]
    results = {}

    # ── 3 sites ─────────────────────────────────────────────────────────
    tc = TypedCandidates(build_system(ARBITRARY_START_THETAS_DEG))
    ev = make_evaluator(tc)
    t0 = time.perf_counter()
    truth = exhaustive(tc, ev)
    opt = max(truth.values())
    L += [f"Two-panel, 3 sites (p1, p2, p3): exhaustive optimum {opt:.6f} "
          f"({time.perf_counter() - t0:.0f}s)"] + kind_summary(tc, truth, ev, top=10)
    r, lines = run_bnb(tc, ev, opt, tag='B&B (no incumbent, no symmetry)')
    L += lines + [""]
    results['three_sites'] = dict(exhaustive={tc.describe(d): v for d, v in truth.items()},
                                  bnb=r)

    # ── 6 sites ─────────────────────────────────────────────────────────
    tc = TypedCandidates(build_six_site_system(), base_thetas_deg=TP6_THETAS)
    ev = make_evaluator(tc)
    t0 = time.perf_counter()
    truth = exhaustive(tc, ev)
    opt = max(truth.values())
    L += [f"Two-panel, 6 sites (p1, p2, p3 and their thickness mirrors): exhaustive "
          f"optimum {opt:.6f} ({time.perf_counter() - t0:.0f}s, "
          f"{ev.n_eval_calls} Riemannian evaluations)"] + kind_summary(tc, truth, ev)
    v_only = max(v for d, v in truth.items() if all(tc.kind_of[i] == 'vgroove' for i in d))
    L.append(f"   best all-V design {v_only:.6f} (sec 7.6: 0.013330); mixed types "
             f"{'improve' if opt > v_only * (1 + OPT_TOL) else 'do not improve'} "
             f"on it ({100 * opt / v_only:.1f}%)")

    z_plane = ((0., 0., 1.), -PANEL_T / 2)
    y_plane = ((0., 1., 0.), 0.5)
    syms = []
    all_designs = [list(d) for d in truth]
    for label, p, planes in [('thickness mirror', [3, 4, 5, 0, 1, 2], [z_plane]),
                             ('y mirror', [1, 0, 2, 4, 3, 5], [y_plane]),
                             ('both', [4, 3, 5, 1, 0, 2], [z_plane, y_plane])]:
        lp = tc.lift_perm(p)
        ok, info = verify_symmetry(ev, lp, planes, None, designs=all_designs)
        L.append(f"   symmetry '{label}': {'verified' if ok else 'REJECTED'} (point err "
                 f"{info['point_err']:.1e}, bounds over {info['n_sets_checked']} designs: "
                 f"max rel diff {info['bound_max_rel_diff']:.1e}, "
                 f"{info['feasibility_mismatches']} feasibility mismatches)")
        if ok:
            syms.append(lp)
    rA, lines = run_bnb(tc, ev, opt, tag='A (no incumbent, no symmetry)')
    L += lines
    rB, lines = run_bnb(tc, ev, opt, symmetries=syms, tag='B (no incumbent, symmetry)')
    L += lines

    snd = soundness(tc, ev, truth, N_SOUND, np.random.default_rng(SEED))
    L.append(f"   soundness: {snd['n_bounded']} random partial nodes bounded "
             f"({snd['n_not_bounded']} infeasible by row count/rank); min (bound - best "
             f"completion) {snd['min_margin']:.2e}; violations {snd['violations']}")
    results['six_sites'] = dict(exhaustive={tc.describe(d): v for d, v in truth.items()},
                                bnb_A=rA, bnb_B=rB, soundness=snd,
                                best_all_v=v_only)
    all_pass = (results['three_sites']['bnb']['passed'] and rA['passed'] and rB['passed']
                and snd['violations'] == 0)
    L += ["", f"OVERALL: {'PASS' if all_pass else 'FAIL'}"]
    save('two_panel', L, results)


def birdsfoot():
    L = ["Mixed coupling types: birds-foot (12 sites, 18 rows)", "=" * 78,
         "No exhaustive ground truth (1.7M designs). Equal unit weights; socket walls at 45 deg.",
         ""]
    tc = TypedCandidates(build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)))
    ev = make_evaluator(tc)
    champ = tc.from_site_kinds({s: 'vgroove' for s in BF_CHAMPION_SITES})
    c = ev.evaluate(champ)
    L.append(f"All-V champion [{tc.describe(champ, 3)}]: lambda_min {c['lambda_min']:.6f} "
             f"(sec 7.1: 0.009263)")

    perm = list(range(12))
    for j in range(3):
        perm[3 + j], perm[9 + j] = 9 + j, 3 + j
    lp = tc.lift_perm(perm)
    rng = np.random.default_rng(SEED)
    sample = random_designs(tc, ev.target_rank, 60, rng)
    ok, info = verify_symmetry(ev, lp, [((1., 0., 0.), 0.5)], None, designs=sample)
    L.append(f"Symmetry O-B <-> O-A: {'verified' if ok else 'REJECTED'} (point err "
             f"{info['point_err']:.1e}, bounds over {info['n_sets_checked']} random designs: "
             f"max rel diff {info['bound_max_rel_diff']:.1e}, "
             f"{info['feasibility_mismatches']} feasibility mismatches)")

    r, lines = run_bnb(tc, ev, None, symmetries=[lp] if ok else [], incumbent=champ,
                       tag='B&B (all-V champion incumbent, symmetry)')
    L += lines
    if r['incumbent_trace']:
        L.append("   incumbent updates:")
        for u in r['incumbent_trace']:
            L.append(f"      {u['lambda_min']:.6f} [{tc.describe(u['active'], 3)}] "
                     f"node {u['node']}, {u['elapsed_sec']:.0f}s")
    loose = sorted(r['evaluated_leaves'], key=lambda e: -e['bound'])[:5]
    L.append("   loosest evaluated leaves (bound / achieved):")
    for e in loose:
        L.append(f"      {e['bound']:.6f} / {e['lambda_min']:.6f}  [{tc.describe(e['active'], 3)}]")

    # random-sample check
    best = r['lambda_min']
    designs = random_designs(tc, ev.target_rank, N_SAMPLE, np.random.default_rng(SEED + 1))
    n_feas, n_above, beats = 0, 0, []
    for d in designs:
        st, b, _ = ev.bound(d)
        if st != 'bounded':
            continue
        n_feas += 1
        if b > best * (1 + OPT_TOL):
            n_above += 1
            e = ev.evaluate(d)
            if e['feasible'] and e['lambda_min'] > best * (1 + OPT_TOL):
                beats.append((e['lambda_min'], tc.describe(d, 3)))
    L.append(f"Random-sample check: {len(designs)} valid designs, {n_feas} full rank, "
             f"{n_above} with bound above B&B's best (evaluated), "
             f"{len(beats)} beat it {'-- FAIL' if beats else '-- PASS'}")
    for lam, desc in beats:
        L.append(f"      {lam:.6f} [{desc}]")
    save('birdsfoot', L, dict(bnb=r, champion=c, sample_beats=beats))


def random_designs(tc, target_rank, n, rng):
    """n distinct random valid designs: per site uniform over {none, kinds},
    rejection-sampled to exactly target_rank rows."""
    out, seen = [], set()
    choices = [None] + list(tc.kinds)
    while len(out) < n:
        a = [choices[j] for j in rng.integers(0, len(choices), size=tc.n_sites)]
        d = tuple(tc.from_site_kinds(a))
        if tc.n_rows(d) == target_rank and d not in seen:
            seen.add(d)
            out.append(list(d))
    return out


def save(name, lines, results):
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, f'mixed_types_validation_{name}')
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else float(o))
    print(f"Saved: {base}_summary.txt / _results.json")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else 'two-panel'
    if part in ('two-panel', 'all'):
        two_panel()
    if part in ('birds-foot', 'all'):
        birdsfoot()
