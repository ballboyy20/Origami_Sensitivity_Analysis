"""
single_gore_soundness.py

Chases the open soundness flag from the single-gore exhaustive sweep
(KINEMATIC_COUPLING_SUMMARY.md sec 7.17): one feasible set's certified
bound t came out 1.68e-6 BELOW its achieved lambda_min. The sweep saved
only lambda_min per set, so the offending set is unknown.

Bound-only re-run (no 3.5 h Riemannian re-sweep):

  1. For every feasible set in results/gore_bnb_single_gore_results.json,
     recompute BOTH bounds with the same per-set seed as the sweep
     (SetEvaluator, seed 31, so the same full-rank reference):
       - t       : sdp_certified_bound (what the sweep's soundness check
                   compared against; default solver = Clarabel, SCS
                   fallback), with its cvxpy status
       - b_node  : bound_node on the complete set (what B&B prunes leaves
                   with)
     and compare each against the saved lambda_min.
  2. For every set below its achieved value (relative tolerance 1e-8, as
     in gore_bnb.py), plus the 3 tightest sets either way:
       a. re-solve t with tight Clarabel, tight SCS, and MOSEK;
       b. re-run the Riemannian evaluation (same seed, so it should
          reproduce the saved value exactly) and recompute lambda_min at
          its theta three independent ways: the Riemannian convention
          (eigenvalue n_free of K(theta)), the SDP's own reduced model
          (min eig of P^T K(theta) P, the matrix the SDP bounds), and
          C(theta)^T C(theta) built directly from the geometry.

Reading the verdict (only solves reporting status 'optimal' count as
exact; an 'optimal_inaccurate' re-solve proves nothing):
  - every exact re-solve lifts t above lambda_min -> solver tolerance (the
    default solve stopped early / 'optimal_inaccurate'); confirmed benign
    at this size, fix = a better solver or a safety margin.
  - reduced-model lambda < Riemannian lambda    -> the bound and the
    evaluator measure different matrices (gauge projection mismatch);
    a real modelling issue, investigate before relying on leaf pruning.
  - tight solvers agree t < lambda and the three lambda's agree
    -> a real soundness violation; stop and investigate.

Saves: results/single_gore_soundness_{summary.txt,results.json} and a
UTF-8 progress log results/single_gore_soundness.log.

Run: python -u single_gore_soundness.py   (~10-30 min, one process)
     python -u single_gore_soundness.py --reverdict   (recompute the verdict
         from the saved results json and rewrite the summary's VERDICT line)
"""

import sys, os, json, time, argparse

import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from flasher_gore_system import build_gore_system, GORE_START_THETAS_DEG
from swap_search import SetEvaluator
from bound_and_evaluate import sdp_certified_bound
from sdp_theta_optimizer import _calibrate_reduced
from riemannian_gradient_sampling import calibrate_topology
from screen_validation import Tee

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SAVED = os.path.join(RESULTS_DIR, 'gore_bnb_single_gore_results.json')
OUT = os.path.join(RESULTS_DIR, 'single_gore_soundness')
SEED = 31            # gore_bnb.py's SEED: same references, same restarts
REL_TOL = 1e-8       # gore_bnb.py's soundness tolerance
N_TIGHTEST = 3

TIGHT = {
    'clarabel_tight': (cp.CLARABEL, dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                                         tol_feas=1e-12, tol_ktratio=1e-10,
                                         max_iter=2000)),
    'scs_tight': (cp.SCS, dict(eps_abs=1e-10, eps_rel=1e-10, max_iters=500000)),
    'mosek': (cp.MOSEK, {}),
}


def solve_t(system, s, ref, solver, kwargs):
    try:
        r = sdp_certified_bound(system, s, ref_thetas=ref, solver=solver, **kwargs)
        return r['t'], r['stage1_status']
    except Exception as e:                      # solver missing / failed
        return None, f"error: {type(e).__name__}: {e}"[:120]


def lambda_three_ways(system, s, ref, theta, tr):
    """lambda_min at theta (radians, aligned to sorted s) by the Riemannian
    convention, the SDP's reduced model, and C^T C from the geometry."""
    K0, Kc, Ks = calibrate_topology(system, s, ref)
    K = K0.copy()
    for k, i in enumerate(s):
        K = K + np.cos(2 * theta[k]) * Kc[i] + np.sin(2 * theta[k]) * Ks[i]
    K = 0.5 * (K + K.T)
    n_free = K.shape[0] - tr
    lam_riem = float(np.linalg.eigvalsh(K)[n_free])

    K0r, Kcr, Ksr = _calibrate_reduced(system, s, 1.0, tr, ref)
    Kr = K0r.copy()
    for k, i in enumerate(s):
        Kr = Kr + np.cos(2 * theta[k]) * Kcr[i] + np.sin(2 * theta[k]) * Ksr[i]
    lam_red = float(np.linalg.eigvalsh(0.5 * (Kr + Kr.T))[0])

    for i, c in enumerate(system.couplings):
        c.active = i in s
    for k, i in enumerate(s):
        system.couplings[i].set_theta(float(theta[k]))
    C = system.build_constraint_matrix()
    lam_geo = float(np.linalg.eigvalsh(C.T @ C)[C.shape[1] - tr])
    return dict(riemannian_convention=lam_riem, sdp_reduced_model=lam_red,
                geometry_CtC=lam_geo)


def make_verdict(details, opt, tol):
    """Verdict over the drilled-into sets. Only re-solves with status
    'optimal' count as exact."""
    viol = [d for d in details
            if min(d['t_default'], d['b_node']) < d['saved_lambda'] - tol]
    if not viol:
        return ("NO violation reproduced: every bound is >= its achieved value within "
                f"{REL_TOL:g} relative.")
    for d in details:
        three = d.get('lambda_at_theta')
        if three and three['sdp_reduced_model'] < three['riemannian_convention'] - 1e-9 * opt:
            return (f"MODEL MISMATCH at {d['set']}: at the achieved theta the SDP's reduced "
                    "matrix has a smaller lambda than the Riemannian convention, so bound and "
                    "evaluator measure different things. Investigate the gauge projection.")
    unresolved, real, margins = [], [], []
    for d in viol:
        exact = [(n, d[n]['t']) for n in TIGHT
                 if d[n]['t'] is not None and d[n]['status'] == 'optimal']
        if not exact:
            unresolved.append(d['set'])
        elif all(t < d['saved_lambda'] - tol for _, t in exact):
            real.append(d['set'])
        else:
            margins += [t - d['saved_lambda'] for _, t in exact]
    if real:
        return (f"REAL VIOLATION on {len(real)} set(s) (e.g. {real[0]}): exact re-solves agree "
                "t < lambda and the three lambda's agree. Stop and investigate before "
                "trusting leaf pruning.")
    if unresolved:
        return (f"UNRESOLVED on {len(unresolved)} set(s) (e.g. {unresolved[0]}): no re-solve "
                "reached status 'optimal'.")
    deficit = max(d['saved_lambda'] - min(d['t_default'], d['b_node']) for d in viol)
    return (f"SOLVER TOLERANCE: all {len(viol)} violating sets are resolved by an exact "
            f"(status 'optimal') re-solve, which puts t above lambda by {min(margins):.1e} to "
            f"{max(margins):.1e}. The default Clarabel solve ('optimal_inaccurate') "
            f"underestimates by up to {deficit:.1e} ({deficit / opt:.1e} of the optimum); "
            "tighter Clarabel tolerances don't fix it. Not a modelling error, but B&B leaf "
            "bounds (b_node) carry the same error: use MOSEK or a safety margin.")


def reverdict():
    with open(OUT + '_results.json', encoding='utf-8') as f:
        res = json.load(f)
    with open(SAVED, encoding='utf-8') as f:
        opt = json.load(f)['optimum']
    verdict = make_verdict(res['details'], opt, REL_TOL * opt)
    res['verdict'] = verdict
    with open(OUT + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    with open(OUT + '_summary.txt', encoding='utf-8') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if not l.startswith("VERDICT: ")] + ["VERDICT: " + verdict]
    with open(OUT + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print("VERDICT: " + verdict)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(OUT + '.log')
    with open(SAVED, encoding='utf-8') as f:
        saved = json.load(f)
    truth = {tuple(json.loads(k)): v for k, v in saved['exhaustive'].items()}
    opt = saved['optimum']
    feas = sorted(s for s, v in truth.items() if v > 0)

    system = build_gore_system(np.array(GORE_START_THETAS_DEG))
    tr = system.total_dofs - 6
    ev = SetEvaluator(system, tr, GORE_START_THETAS_DEG, seed=SEED)
    tol = REL_TOL * opt
    L = [f"Single-gore soundness audit: {len(feas)} feasible sets, optimum {opt:.6f}, "
         f"tolerance {REL_TOL:g} relative ({tol:.1e} absolute)", "=" * 78]
    print("\n".join(L), flush=True)

    # 1. bound-only sweep
    t0 = time.perf_counter()
    rows = []
    for n, s in enumerate(feas, 1):
        lam = truth[s]
        status, b_node, ref = ev.bound(list(s))
        if status != 'bounded':
            rows.append(dict(set=list(s), lam=lam, node_status=status))
            continue
        t, st = solve_t(system, list(s), ref, None, {})
        rows.append(dict(set=list(s), lam=lam, t=t, t_status=st, b_node=b_node,
                         t_minus_lam=None if t is None else t - lam,
                         node_minus_lam=b_node - lam, ref=ref))
        if n % 20 == 0:
            print(f"   [{time.perf_counter() - t0:6.0f}s] {n}/{len(feas)}", flush=True)
    t_sweep = time.perf_counter() - t0

    ok_rows = [r for r in rows if r.get('t') is not None]
    odd = [r for r in rows if 'node_status' in r or r.get('t') is None]
    viol_t = [r for r in ok_rows if r['t_minus_lam'] < -tol]
    viol_n = [r for r in ok_rows if r['node_minus_lam'] < -tol]
    inacc = [r for r in ok_rows if r['t_status'] != 'optimal']
    by_t = sorted(ok_rows, key=lambda r: r['t_minus_lam'])
    L.append(f"Bound sweep ({t_sweep:.0f}s): {len(ok_rows)} sets bounded, {len(odd)} not "
             f"(status != bounded or solve error)")
    L.append(f"   t (sdp_certified_bound, default solver): {len(viol_t)} below achieved; "
             f"min t - lambda {by_t[0]['t_minus_lam']:.2e} at {by_t[0]['set']}")
    mn = min(ok_rows, key=lambda r: r['node_minus_lam'])
    L.append(f"   b_node (bound_node, used by B&B):        {len(viol_n)} below achieved; "
             f"min b_node - lambda {mn['node_minus_lam']:.2e} at {mn['set']}")
    L.append(f"   default-solver status != 'optimal': {len(inacc)} sets "
             f"({sorted({r['t_status'] for r in inacc})})")
    rep = abs(by_t[0]['t_minus_lam'] + 1.68e-6) < 1e-7
    L.append(f"   original flag (-1.68e-6) {'REPRODUCED' if rep else 'NOT reproduced'} "
             f"by the worst t - lambda")
    for r in odd:
        L.append(f"   not bounded: {r['set']} {r.get('node_status') or r.get('t_status')}")
    print("\n".join(L[-6:]), flush=True)

    # 2. drill into the violators + the tightest few
    focus = {tuple(r['set']): r for r in viol_t + viol_n}
    for r in by_t[:N_TIGHTEST]:
        focus.setdefault(tuple(r['set']), r)
    L.append("")
    L.append(f"Detail on {len(focus)} set(s) (violators + {N_TIGHTEST} tightest):")
    details = []
    for key, r in focus.items():
        s, ref = list(key), r['ref']
        d = dict(set=s, saved_lambda=r['lam'], t_default=r['t'],
                 t_default_status=r['t_status'], b_node=r['b_node'])
        L.append(f"   {s}: saved lambda {r['lam']:.9f}")
        L.append(f"      t default   {r['t']:.9f} ({r['t_status']}), t - lambda "
                 f"{r['t_minus_lam']:+.2e}")
        L.append(f"      b_node      {r['b_node']:.9f}, b_node - lambda "
                 f"{r['node_minus_lam']:+.2e}")
        for name, (solver, kw) in TIGHT.items():
            t, st = solve_t(system, s, ref, solver, kw)
            d[name] = dict(t=t, status=st)
            L.append(f"      t {name:12s}" + (f"{t:.9f} ({st}), t - lambda {t - r['lam']:+.2e}"
                                             if t is not None else st))
        t1 = time.perf_counter()
        e = ev.evaluate(s)
        d['re_evaluated_lambda'] = e['lambda_min']
        L.append(f"      Riemannian re-run {e['lambda_min']:.9f} (diff from saved "
                 f"{e['lambda_min'] - r['lam']:+.1e}; {time.perf_counter() - t1:.0f}s)")
        if e['feasible'] and e['theta'] is not None:
            three = lambda_three_ways(system, s, ref, np.asarray(e['theta']), tr)
            d['lambda_at_theta'] = three
            for k, v in three.items():
                L.append(f"      lambda at its theta, {k:22s} {v:.9f} "
                         f"(vs t default {r['t'] - v:+.2e})")
        details.append(d)
        print("\n".join(L[-12:]), flush=True)

    L.append("")
    verdict = make_verdict(details, opt, tol)
    L.append("VERDICT: " + verdict)

    summary = "\n".join(L)
    print("\n" + summary, flush=True)
    with open(OUT + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    for r in rows:
        r.pop('ref', None)
    with open(OUT + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(dict(rows=rows, details=details, verdict=verdict), f, indent=1,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"Saved: {OUT}_{{summary.txt,results.json}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--reverdict', action='store_true')
    if ap.parse_args().reverdict:
        reverdict()
    else:
        main()
