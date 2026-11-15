"""
screen_validation.py

Validates the one-restart leaf screen in branch-and-bound
(BranchAndBound(screen_frac=...), KINEMATIC_COUPLING_SUMMARY.md sec 7.15).

The screen never changes the certificate (B&B's certified upper bound uses
leaf bounds only), so what needs checking is search quality: does
screening ever throw away a leaf that would have raised the incumbent?
And how much time does it save? If no incumbent-raising leaf is screened,
the incumbent history is unchanged, so the tree, every pruning count and
the certified bound must come out IDENTICAL to the unscreened run; only
time and restart counts drop. That is the main pass criterion below.

Parts (argv[1]; each saves its own files, so they can run concurrently):

  replay          ~30-40 min. Replays the saved birds-foot mixed run
                  (results/mixed_types_validation_birdsfoot_results.json):
                  first-restart value for every evaluated leaf whose
                  best-of-5 was >= 0.6 x the running incumbent, so the
                  screen's effect at f = 0.7 / 0.8 / 0.9 can be read off
                  without re-running B&B. PASS at f if none of the 9
                  incumbent-raising leaves would have been screened.
                  Also re-runs the full evaluation on those 9 to confirm
                  this machine reproduces the saved values.
  small           ~5-15 min. Unscreened vs. screened (f = 0.8) B&B on
                  two-panel mixed (3 and 6 sites) and birds-foot 12-site
                  all-V, separate evaluators for fair timing. PASS if the
                  optimum, certified bound and node counts match.
  birdsfoot-mixed Expected ~4-5 h (unscreened: 9.6 h). The sec 7.15 run with
                  f = 0.8, compared field-by-field to the saved result.
  birdsfoot24     The sec 7.14 24-site all-V B&B with f = 0.8.
                  --incumbent best (default): from 0.009630, compare to
                  the 6.2 h complete run (birdsfoot24_bnb_inc).
                  --incumbent champion: from the old champion; unscreened,
                  this did not finish in 8 h (birdsfoot24_bnb).

Options: --frac F (default 0.8), --recheck N (default 20),
         --max-hours H (B&B cap; default 12).

Each part writes results/screen_validation_<part>_{summary.txt,results.json}
and a UTF-8 progress log results/screen_validation_<part>.log.

Run: python -u screen_validation.py <part> [options]
"""

import sys, os, json, time, argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (build_system, ARBITRARY_START_THETAS_DEG,
                                   build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG)
from interactive_optimizer import t as PANEL_T
from two_panel_six_sites import build_six_site_system, START_THETAS_DEG as TP6_THETAS
from coupling_types import TypedCandidates
from swap_search import SetEvaluator
from branch_and_bound import BranchAndBound, verify_symmetry
import mixed_types_validation as mtv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 31                    # same evaluator seed as every saved run
OPT_TOL = 1e-4
SAVED_BF_MIXED = os.path.join(RESULTS_DIR, 'mixed_types_validation_birdsfoot_results.json')
BEST_24 = [0, 1, 2, 7, 9, 13, 14, 19, 21]          # sec 7.14, 0.009630
STAT_KEYS = ['nodes', 'pruned_cardinality', 'pruned_cut', 'pruned_symmetry',
             'pruned_rank', 'pruned_bound', 'leaves', 'leaves_pruned_bound',
             'leaves_infeasible', 'leaves_evaluated']


class Tee:
    """Mirror stdout to a UTF-8 log file (PowerShell redirection writes UTF-16)."""
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()

    def __getattr__(self, name):          # encoding, isatty, ... from the console
        return getattr(self.out, name)


def save(part, lines, results):
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    base = os.path.join(RESULTS_DIR, f'screen_validation_{part.replace("-", "_")}')
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else (o if isinstance(o, str) else float(o)))
    print(f"Saved: {base}_summary.txt / _results.json")


def compare(ref, new, rel=1e-9):
    """Lines comparing two B&B results (ref unscreened, new screened), and
    whether the search was identical (same incumbent, bound, tree)."""
    same_lam = abs(new['lambda_min'] - ref['lambda_min']) <= rel * max(ref['lambda_min'], 1e-12)
    ub_r, ub_n = ref['certified_upper_bound'], new['certified_upper_bound']
    same_ub = ub_r == ub_n or abs(ub_n - ub_r) <= rel * max(abs(ub_r), 1e-12)
    diff = [k for k in STAT_KEYS if ref['stats'].get(k) != new['stats'].get(k)]
    s = new['stats']
    lines = [
        f"      unscreened: {ref['lambda_min']:.6f}, UB {ub_r:.6f}, "
        f"{ref['stats']['leaves_evaluated']} leaves evaluated, {ref['elapsed_sec']:.0f}s",
        f"      screened:   {new['lambda_min']:.6f}, UB {ub_n:.6f}, "
        f"{s['leaves_evaluated']} leaves evaluated ({s.get('leaves_screened', 0)} screened, "
        f"{s.get('screen_rechecked', 0)} rechecked, {s.get('screen_recheck_updates', 0)} "
        f"recheck updates), {new['elapsed_sec']:.0f}s "
        f"({100 * new['elapsed_sec'] / max(ref['elapsed_sec'], 1e-9):.0f}% of unscreened)",
        f"      same best {same_lam}, same certified UB {same_ub}, tree counts "
        f"{'identical' if not diff else 'DIFFER in ' + ', '.join(diff)}"]
    return lines, same_lam and same_ub and not diff


def bnb_line(tag, r):
    s = r['stats']
    gap = 100 * r['certified_gap'] / max(r['lambda_min'], 1e-12)
    return [f"   {tag}: found {r['lambda_min']:.6f} {r['active']}; certified UB "
            f"{r['certified_upper_bound']:.6f} (gap {gap:.2f}%), complete {r['complete']}",
            f"      nodes {s['nodes']}; leaves evaluated {s['leaves_evaluated']} "
            f"({s.get('leaves_screened', 0)} screened, {s.get('screen_rechecked', 0)} "
            f"rechecked, {s.get('screen_recheck_updates', 0)} recheck updates); "
            f"{r['elapsed_sec']:.0f}s"]


# ── replay ──────────────────────────────────────────────────────────────

def replay(args):
    saved = json.load(open(SAVED_BF_MIXED, encoding='utf-8'))['bnb']
    tc = TypedCandidates(build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)))
    ev = mtv.make_evaluator(tc)
    L = ["Screen replay on the saved birds-foot mixed run (sec 7.15)", "=" * 78]

    # Running incumbent at the time each leaf was evaluated
    inc = saved['start_lambda_min']
    rows = []
    for e in saved['evaluated_leaves']:
        rows.append(dict(active=e['active'], bound=e['bound'], best5=e['lambda_min'],
                         inc=inc, raised=e['lambda_min'] > inc))
        inc = max(inc, e['lambda_min'])
    todo = [r for r in rows if r['best5'] >= 0.6 * r['inc']]
    L.append(f"{len(rows)} evaluated leaves; {len(todo)} with best-of-5 >= 0.6 x running "
             f"incumbent (all others are screened at any f <= 0.6, since restart 1 <= best-of-5); "
             f"{sum(r['raised'] for r in rows)} raised the incumbent")

    # 1. Reproducibility: full evaluation of the incumbent-raising leaves.
    # Separate evaluator: a full result in ev's cache would be returned in
    # place of the first-restart value below.
    ev_full = mtv.make_evaluator(tc)
    t0 = time.perf_counter()
    rep = []
    for r in rows:
        if r['raised']:
            v = ev_full.evaluate(r['active'])['lambda_min']
            rep.append(abs(v - r['best5']) / r['best5'])
            print(f"   repro {r['active']}: saved {r['best5']:.6f}, now {v:.6f}", flush=True)
    repro_ok = max(rep) < 1e-6
    L.append(f"Reproducibility (full evaluation of the {len(rep)} incumbent-raising leaves): "
             f"max rel diff {max(rep):.1e} -- {'OK' if repro_ok else 'MISMATCH (results below are suspect)'}"
             f" ({time.perf_counter() - t0:.0f}s)")

    # 2. First-restart values
    t0 = time.perf_counter()
    for i, r in enumerate(todo):
        e = ev.evaluate(r['active'], stop_below=np.inf)    # always stops after restart 1
        r['first'] = e['lambda_min'] if e['feasible'] else 0.0
        if (i + 1) % 25 == 0:
            print(f"   [{time.perf_counter() - t0:6.0f}s] {i + 1}/{len(todo)} first restarts",
                  flush=True)
    L.append(f"First restarts: {len(todo)} leaves, {time.perf_counter() - t0:.0f}s "
             f"({(time.perf_counter() - t0) / max(len(todo), 1):.1f}s each)")
    ratio = np.array([r['first'] / r['best5'] for r in todo])
    L.append(f"   restart 1 / best-of-5: median {np.median(ratio):.3f}, 10th pct "
             f"{np.percentile(ratio, 10):.3f}, min {ratio.min():.3f}; restart 1 was the best "
             f"in {int(np.sum(ratio >= 1 - 1e-9))}/{len(todo)}")

    verdicts = {}
    for f in (0.7, 0.8, 0.9):
        cut = [r for r in rows if r['best5'] < 0.6 * r['inc'] or r['first'] < f * r['inc']]
        bad = [r for r in cut if r['raised']]
        # restarts saved: a screened leaf runs 1 of 5
        L.append(f"   f = {f}: {len(cut)}/{len(rows)} leaves screened "
                 f"({100 * len(cut) / len(rows):.0f}%; restarts {len(rows) * 5} -> "
                 f"{len(rows) * 5 - 4 * len(cut)}); incumbent-raising leaves screened: "
                 f"{len(bad)} -- {'PASS' if not bad else 'FAIL'}")
        for r in bad:
            L.append(f"      lost {r['active']}: restart 1 {r['first']:.6f}, best-of-5 "
                     f"{r['best5']:.6f}, incumbent then {r['inc']:.6f}")
        verdicts[str(f)] = not bad
    save('replay', L, dict(leaves=todo, reproducibility_ok=repro_ok, verdicts=verdicts))


# ── small ───────────────────────────────────────────────────────────────

def pair(make_ev, run_kwargs, bnb_kwargs, frac, recheck):
    ev_a, ev_b = make_ev(), make_ev()
    ra = BranchAndBound(ev_a, verbose=False, **bnb_kwargs).run(**run_kwargs)
    rb = BranchAndBound(ev_b, verbose=False, screen_frac=frac, screen_recheck=recheck,
                        **bnb_kwargs).run(**run_kwargs)
    return ra, rb


def small(args):
    L = [f"Screen regression on small systems (f = {args.frac}, recheck {args.recheck})",
         "=" * 78]
    results, all_ok = {}, True

    # two-panel mixed, 3 sites (sec 7.13: optimum 0.017142)
    tc = TypedCandidates(build_system(ARBITRARY_START_THETAS_DEG))
    ra, rb = pair(lambda: mtv.make_evaluator(tc), {},
                  dict(seed=SEED, rows=tc.rows, site_of=tc.site_of), args.frac, args.recheck)
    lines, ok = compare(ra, rb)
    ok = ok and rb['lambda_min'] >= 0.017142 * (1 - OPT_TOL)
    L += [f"two-panel mixed, 3 sites (optimum 0.017142): {'PASS' if ok else 'FAIL'}"] + lines
    results['two_panel_3'] = dict(unscreened=ra, screened=rb, passed=ok)
    all_ok &= ok

    # two-panel mixed, 6 sites, verified symmetries, mirror pairs first
    tc = TypedCandidates(build_six_site_system(), base_thetas_deg=TP6_THETAS)
    ev = mtv.make_evaluator(tc)
    designs = [list(d) for d in tc.enumerate_designs(ev.target_rank)]
    z_plane, y_plane = ((0., 0., 1.), -PANEL_T / 2), ((0., 1., 0.), 0.5)
    syms = []
    for p, planes in [([3, 4, 5, 0, 1, 2], [z_plane]), ([1, 0, 2, 4, 3, 5], [y_plane]),
                      ([4, 3, 5, 1, 0, 2], [z_plane, y_plane])]:
        lp = tc.lift_perm(p)
        if verify_symmetry(ev, lp, planes, None, designs=designs)[0]:
            syms.append(lp)
    ra, rb = pair(lambda: mtv.make_evaluator(tc), {},
                  dict(seed=SEED, rows=tc.rows, site_of=tc.site_of, symmetries=syms,
                       branch_rule='symmetry_first'), args.frac, args.recheck)
    lines, ok = compare(ra, rb)
    ok = ok and rb['lambda_min'] >= 0.017142 * (1 - OPT_TOL)
    L += [f"two-panel mixed, 6 sites, {len(syms)} symmetries (optimum 0.017142): "
          f"{'PASS' if ok else 'FAIL'}"] + lines
    results['two_panel_6'] = dict(unscreened=ra, screened=rb, passed=ok)
    all_ok &= ok

    # birds-foot all-V, 12 sites, k = 9, no incumbent (sec 7.10: optimum 0.009263)
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    tr = system.total_dofs - 6
    mk = lambda: SetEvaluator(system, tr, BIRDSFOOT_START_THETAS_DEG, seed=SEED)
    perm = list(range(12))
    for j in range(3):
        perm[3 + j], perm[9 + j] = 9 + j, 3 + j
    ok_sym, _ = verify_symmetry(mk(), perm, [((1., 0., 0.), 0.5)], 9)
    ra, rb = pair(mk, {}, dict(k=9, seed=SEED, symmetries=[perm] if ok_sym else [],
                               branch_rule='symmetry_first'), args.frac, args.recheck)
    lines, ok = compare(ra, rb)
    ok = ok and rb['lambda_min'] >= 0.009263 * (1 - OPT_TOL)
    L += [f"birds-foot all-V, 12 sites (optimum 0.009263): {'PASS' if ok else 'FAIL'}"] + lines
    results['birdsfoot_allv'] = dict(unscreened=ra, screened=rb, passed=ok)
    all_ok &= ok

    L += ["", f"OVERALL: {'PASS' if all_ok else 'FAIL'}"]
    save('small', L, results)


# ── birds-foot mixed rerun ──────────────────────────────────────────────

def birdsfoot_mixed(args):
    saved = json.load(open(SAVED_BF_MIXED, encoding='utf-8'))['bnb']
    tc = TypedCandidates(build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)))
    ev = mtv.make_evaluator(tc)
    champ = tc.from_site_kinds({s: 'vgroove' for s in mtv.BF_CHAMPION_SITES})
    L = [f"Birds-foot mixed (12 sites, 18 rows) with screen f = {args.frac}, "
         f"recheck {args.recheck}", "=" * 78]

    # Same symmetry check as mixed_types_validation.birdsfoot (same sample)
    perm = list(range(12))
    for j in range(3):
        perm[3 + j], perm[9 + j] = 9 + j, 3 + j
    lp = tc.lift_perm(perm)
    sample = mtv.random_designs(tc, ev.target_rank, 60, np.random.default_rng(SEED))
    ok, _ = verify_symmetry(ev, lp, [((1., 0., 0.), 0.5)], None, designs=sample, rel_tol=1e-2)
    L.append(f"Symmetry O-B <-> O-A: {'verified' if ok else 'REJECTED'}")
    print("\n".join(L), flush=True)

    bnb = BranchAndBound(ev, symmetries=[lp] if ok else [], seed=SEED, rows=tc.rows,
                         site_of=tc.site_of, branch_rule='symmetry_first',
                         screen_frac=args.frac, screen_recheck=args.recheck)
    r = bnb.run(incumbent_active=champ, max_sec=3600 * args.max_hours, log_every=25)
    L += bnb_line('B&B (screened)', r)
    lines, same = compare(saved, r)
    L += ["   vs. saved unscreened run (sec 7.15):"] + lines
    L.append(f"   {'PASS: identical search, only cheaper' if same else 'DIFFERENT: see above'}")
    for u in r['incumbent_trace']:
        L.append(f"      incumbent {u['lambda_min']:.6f} [{tc.describe(u['active'], 3)}] "
                 f"node {u['node']}, {u['elapsed_sec']:.0f}s")
    L.append(f"   evaluator: {ev.n_eval_calls} Riemannian evaluations started, "
             f"{ev.n_screened} stopped after restart 1")
    save('birdsfoot-mixed', L, dict(bnb=r, identical_to_saved=same))


# ── birds-foot 24 sites ─────────────────────────────────────────────────

def birdsfoot24(args):
    from birdsfoot_six_sites import build_six_site_birdsfoot, START_THETAS_DEG
    import birdsfoot24_scale_test as b24
    inc = BEST_24 if args.incumbent == 'best' else b24.CHAMPION_24
    ref_file = 'birdsfoot24_bnb_inc' if args.incumbent == 'best' else 'birdsfoot24_bnb'
    system = build_six_site_birdsfoot()
    ev = SetEvaluator(system, system.total_dofs - 6, START_THETAS_DEG, seed=SEED)
    L = [f"Birds-foot 24 sites, k = 9, from {args.incumbent} {inc}, screen f = {args.frac}, "
         f"recheck {args.recheck}", "=" * 78]
    syms = []
    for label, p, planes in b24.symmetries():
        ok, _ = verify_symmetry(ev, p, planes, b24.K, n_sample_sets=300, seed=SEED)
        L.append(f"symmetry '{label}': {'verified' if ok else 'REJECTED'}")
        if ok:
            syms.append(p)
    print("\n".join(L), flush=True)
    bnb = BranchAndBound(ev, b24.K, symmetries=syms, seed=SEED, branch_rule='symmetry_first',
                         screen_frac=args.frac, screen_recheck=args.recheck)
    r = bnb.run(incumbent_active=inc, max_sec=3600 * args.max_hours, log_every=25)
    L += bnb_line('B&B (screened)', r)
    ref_path = os.path.join(RESULTS_DIR, ref_file + '_results.json')
    same = None
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding='utf-8'))
        if ref.get('certified_upper_bound') is None:
            ref['certified_upper_bound'] = float('inf')
        lines, same = compare(ref, r)
        L += [f"   vs. saved unscreened run ({ref_file}, complete {ref.get('complete')}):"] + lines
    for u in r['incumbent_trace']:
        L.append(f"      incumbent {u['lambda_min']:.6f} {u['active']} node {u['node']}, "
                 f"{u['elapsed_sec']:.0f}s")
    L.append(f"   evaluator: {ev.n_eval_calls} Riemannian evaluations started, "
             f"{ev.n_screened} stopped after restart 1")
    save(f'birdsfoot24-{args.incumbent}', L, dict(bnb=r, identical_to_saved=same))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['replay', 'small', 'birdsfoot-mixed', 'birdsfoot24'])
    ap.add_argument('--frac', type=float, default=0.8)
    ap.add_argument('--recheck', type=int, default=20)
    ap.add_argument('--max-hours', type=float, default=12.0)
    ap.add_argument('--incumbent', choices=['best', 'champion'], default='best')
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.part + (f'-{args.incumbent}' if args.part == 'birdsfoot24' else '')
    sys.stdout = Tee(os.path.join(RESULTS_DIR, f'screen_validation_{tag.replace("-", "_")}.log'))
    {'replay': replay, 'small': small, 'birdsfoot-mixed': birdsfoot_mixed,
     'birdsfoot24': birdsfoot24}[args.part](args)
