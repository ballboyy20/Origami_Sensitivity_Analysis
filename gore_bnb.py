"""
gore_bnb.py

Branch-and-bound on the flasher gore systems (KINEMATIC_COUPLING_SUMMARY.md
sec 8.7, Days 3 and 6). Parts (argv[1]; each saves its own files):

  single-gore      Day 3. Single gore (sec 7.4): 6 panels, 18 candidates,
                   k = 15 V-grooves (30 rows). Panels 2 and 5 hang off one
                   crease each, so those 6 couplings are forced; choose 9
                   of the other 12 = 220 sets. Exhaustive sweep (bound +
                   5-restart Riemannian on every set) as ground truth, then
                   B&B with the forced couplings as fixed_active,
                   unscreened and screened (f = 0.8). PASS if both find the
                   exhaustive optimum, certify a bound >= it, and no set's
                   bound is below its achieved value. ~1-2 h.
  full-gore-allv   Day 6, all V-grooves. Full gore (sec 7.4): 8 panels,
                   27 candidates, target rank 42 = exactly 21 V-grooves.
                   Panel 5's 3 couplings are forced, so C(24,18) = 134,596
                   sets. Generic rank test on every set (rank is generic in
                   theta, sec 2.7; 2 random-theta draws, full rank in
                   either counts). Answers: does ANY exactly constrained
                   all-V design exist? If so, B&B (k = 21, forced
                   couplings, screen f = 0.8) for the best one, capped at
                   --max-hours. Sweep: minutes.
  full-gore-mixed  Day 6, mixed types: flat / V-groove / socket at each of
                   the 27 sites (81 candidates), exactly 42 rows. B&B with
                   screen f = 0.8, no incumbent, no symmetry, capped at
                   --max-hours. Probably will not finish; the first
                   full-rank design it evaluates already answers whether a
                   valid 42-row design exists, and the incumbent trace
                   records how good it got. Leaf-panel couplings are NOT
                   forced here: the B&B crease rule (a bridge crease gets
                   exactly 6 rows) covers it for any type mix.

Options: --frac F (screen fraction, default 0.8), --max-hours H (default 20).

Saves: results/gore_bnb_<part>_{summary.txt,results.json} and a UTF-8
progress log results/gore_bnb_<part>.log.

Run: python -u gore_bnb.py <part> [options]
"""

import sys, os, json, time, argparse
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flasher_gore_system import (build_gore_system, GORE_START_THETAS_DEG,
                                 GORE_PANELS, GORE_CREASES)
from full_flasher_gore_system import (build_full_gore_system, FULL_GORE_START_THETAS_DEG,
                                      FULL_GORE_MANDATORY_COUPLINGS)
from gore_system_common import mandatory_leaf_couplings
from coupling_types import TypedCandidates
from swap_search import SetEvaluator
from branch_and_bound import BranchAndBound
from screen_validation import Tee, compare, bnb_line

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 31
OPT_TOL = 1e-4


def save(part, lines, results):
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    base = os.path.join(RESULTS_DIR, f'gore_bnb_{part.replace("-", "_")}')
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else float(o))
    print(f"Saved: {base}_summary.txt / _results.json")


def trace_lines(r, describe=str):
    out = []
    for u in r['incumbent_trace']:
        out.append(f"      incumbent {u['lambda_min']:.6f} {describe(u['active'])} "
                   f"node {u['node']}, {u['elapsed_sec']:.0f}s")
    return out


# ── single gore (Day 3) ─────────────────────────────────────────────────

def single_gore(args):
    system = build_gore_system(np.array(GORE_START_THETAS_DEG))
    m, tr = len(system.couplings), system.total_dofs - 6
    k = tr // 2
    forced = mandatory_leaf_couplings(GORE_PANELS, GORE_CREASES)
    free = [i for i in range(m) if i not in forced]
    mk = lambda: SetEvaluator(system, tr, GORE_START_THETAS_DEG, seed=SEED)
    L = [f"Single gore: m = {m}, k = {k} (target rank {tr}); forced {forced}; "
         f"choose {k - len(forced)} of {len(free)}", "=" * 78]
    print("\n".join(L), flush=True)

    # exhaustive ground truth
    ev = mk()
    t0 = time.perf_counter()
    truth, gaps = {}, []
    sets = [sorted(forced + list(c)) for c in combinations(free, k - len(forced))]
    for n, s in enumerate(sets):
        e = ev.evaluate(s)
        truth[tuple(s)] = e['lambda_min'] if e['feasible'] else 0.0
        if e['feasible']:
            gaps.append(e['t'] - e['lambda_min'])
        if (n + 1) % 20 == 0:
            print(f"   [{time.perf_counter() - t0:6.0f}s] exhaustive {n + 1}/{len(sets)}",
                  flush=True)
    t_ex = time.perf_counter() - t0
    opt = max(truth.values())
    best = sorted(truth.items(), key=lambda kv: -kv[1])[:8]
    n_feas = sum(v > 0 for v in truth.values())
    sound = min(gaps) >= -1e-8 * max(opt, 1e-12)
    L.append(f"Exhaustive: {len(sets)} sets, {n_feas} feasible, optimum {opt:.6f} "
             f"({t_ex:.0f}s, {ev.n_eval_calls} Riemannian evaluations); min (bound - achieved) "
             f"{min(gaps):.2e} -- soundness {'OK' if sound else 'VIOLATED'}")
    L.append(f"   greedy (sec 7.4): 0.012351; best 8:")
    L += [f"      {v:.6f}  {list(s)}" for s, v in best]

    ra = BranchAndBound(mk(), k=k, fixed_active=forced, seed=SEED, verbose=False).run()
    rb = BranchAndBound(mk(), k=k, fixed_active=forced, seed=SEED, verbose=False,
                        screen_frac=args.frac).run()
    ok = True
    for tag, r in (('B&B unscreened', ra), (f'B&B screened f = {args.frac}', rb)):
        p = (r['complete'] and r['lambda_min'] >= opt * (1 - OPT_TOL)
             and r['certified_upper_bound'] >= opt * (1 - 1e-9))
        ok &= p
        L += bnb_line(f"{tag}: {'PASS' if p else 'FAIL'}", r)
    lines, same = compare(ra, rb)
    L += ["   screened vs. unscreened:"] + lines
    ok &= sound
    L += ["", f"OVERALL: {'PASS' if ok else 'FAIL'}"]
    save('single-gore', L, dict(exhaustive={str(list(s)): v for s, v in truth.items()},
                                optimum=opt, unscreened=ra, screened=rb, passed=ok))


# ── full gore, all V-grooves (Day 6) ────────────────────────────────────

def full_gore_allv(args):
    system = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    m, tr = len(system.couplings), system.total_dofs - 6
    k = tr // 2
    forced = list(FULL_GORE_MANDATORY_COUPLINGS)
    free = [i for i in range(m) if i not in forced]
    L = [f"Full gore, all V-grooves: m = {m}, k = {k} (target rank {tr}, exact constraint); "
         f"forced {forced}; choose {k - len(forced)} of {len(free)}", "=" * 78]
    print("\n".join(L), flush=True)

    # C at random theta, all couplings active; V-groove rows are contiguous,
    # 2 per coupling, in coupling order (CouplingSystem.build_constraint_matrix)
    rng = np.random.default_rng(SEED)
    Cs = []
    saved = [(c.active, c.theta) for c in system.couplings]
    for _ in range(2):
        for c in system.couplings:
            c.active = True
            c.set_theta(float(rng.uniform(0, np.pi)))
        C = system.build_constraint_matrix()
        assert C.shape == (2 * m, system.total_dofs), C.shape
        Cs.append(C)
    for c, (a, th) in zip(system.couplings, saved):
        c.active = a
        c.set_theta(th)

    t0 = time.perf_counter()
    sets = [sorted(forced + list(c)) for c in combinations(free, k - len(forced))]
    feasible = np.zeros(len(sets), dtype=bool)
    smin = np.zeros(len(sets))
    B = 4096
    for start in range(0, len(sets), B):
        chunk = np.array(sets[start:start + B])
        rows = np.stack([chunk * 2, chunk * 2 + 1], axis=2).reshape(len(chunk), -1)
        for C in Cs:
            S = np.linalg.svd(C[rows], compute_uv=False)      # (batch, 42)
            tol = S[:, :1] * max(rows.shape[1], C.shape[1]) * np.finfo(float).eps
            full = np.sum(S > tol, axis=1) >= tr
            feasible[start:start + len(chunk)] |= full
            smin[start:start + len(chunk)] = np.maximum(
                smin[start:start + len(chunk)], S[:, tr - 1] / S[:, 0])
        print(f"   [{time.perf_counter() - t0:6.0f}s] {min(start + B, len(sets))}/{len(sets)} "
              f"sets, {int(feasible.sum())} full rank so far", flush=True)
    n_feas = int(feasible.sum())
    L.append(f"Generic rank sweep: {len(sets)} sets, {n_feas} reach rank {tr} "
             f"({time.perf_counter() - t0:.0f}s)")
    if n_feas:
        cond = np.sort(smin[feasible])
        L.append(f"   relative sigma_{tr} of feasible sets: min {cond[0]:.1e}, median "
                 f"{np.median(cond):.1e} (a value near 1e-13 would be numerically suspect)")
        # per-crease coupling counts of feasible sets (3 per crease)
        counts = {}
        for i in np.flatnonzero(feasible):
            key = tuple(np.bincount(np.array(sets[i]) // 3, minlength=m // 3))
            counts[key] = counts.get(key, 0) + 1
        L.append(f"   {len(counts)} distinct per-crease count patterns; most common:")
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
            L.append(f"      {list(key)}: {n} sets")
    else:
        L.append("   NO exactly constrained all-V design exists: every valid full-gore "
                 "design needs another coupling type (or over-constraint).")
    results = dict(n_sets=len(sets), n_feasible=n_feas,
                   feasible_sets=[sets[i] for i in np.flatnonzero(feasible)][:5000])

    if n_feas:
        print("\n".join(L), flush=True)
        ev = SetEvaluator(system, tr, FULL_GORE_START_THETAS_DEG, seed=SEED)
        bnb = BranchAndBound(ev, k=k, fixed_active=forced, seed=SEED,
                             screen_frac=args.frac)
        r = bnb.run(max_sec=3600 * args.max_hours, log_every=25)
        L += bnb_line(f"B&B (k = {k}, forced, screen f = {args.frac}, "
                      f"cap {args.max_hours:.0f} h)", r) + trace_lines(r)
        L.append(f"   (sec 7.4's over-constrained greedy design, 23 V = 46 rows: 0.005185)")
        results['bnb'] = r
    save('full-gore-allv', L, results)


# ── full gore, mixed types (Day 6) ──────────────────────────────────────

def full_gore_mixed(args):
    base = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    tc = TypedCandidates(base, base_thetas_deg=FULL_GORE_START_THETAS_DEG)
    tr = tc.system.total_dofs - 6
    ev = SetEvaluator(tc.system, tr, tc.base_thetas_deg, seed=SEED,
                      rows=tc.rows, sites=tc.site_of)
    L = [f"Full gore, mixed types: {tc.n_sites} sites x 3 types = {tc.m} candidates, "
         f"exactly {tr} rows; screen f = {args.frac}; cap {args.max_hours:.0f} h", "=" * 78]
    print("\n".join(L), flush=True)
    bnb = BranchAndBound(ev, seed=SEED, rows=tc.rows, site_of=tc.site_of,
                         screen_frac=args.frac)
    r = bnb.run(max_sec=3600 * args.max_hours, log_every=25)
    L += bnb_line("B&B (no incumbent, no symmetry)", r)
    L += trace_lines(r, lambda a: f"[{tc.describe(a, 3)}]")
    if r['incumbent_trace']:
        u = r['incumbent_trace'][0]
        L.append(f"   first full-rank design found at {u['elapsed_sec']:.0f}s: a valid "
                 f"exactly constrained mixed design EXISTS")
    else:
        L.append("   no full-rank design found within the cap (not a proof that none exists"
                 + (")" if not r['complete'] else "; search complete, so none exists)"))
    L.append(f"   (sec 7.4's over-constrained all-V greedy design, 46 rows: 0.005185)")
    save('full-gore-mixed', L, dict(bnb=r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['single-gore', 'full-gore-allv', 'full-gore-mixed'])
    ap.add_argument('--frac', type=float, default=0.8)
    ap.add_argument('--max-hours', type=float, default=20.0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, f'gore_bnb_{args.part.replace("-", "_")}.log'))
    {'single-gore': single_gore, 'full-gore-allv': full_gore_allv,
     'full-gore-mixed': full_gore_mixed}[args.part](args)
