"""
polish_birdsfoot_champion.py

Stage-2 log_product polish (log_product_polish.py) on the birds-foot
size-9 champion topology [0,1,2,3,5,6,7,8,11] (spokes O-M:3,O-B:2,O-N:3,
O-A:1 -- found by both exhaustive search and forward greedy).

Reproduces forward_greedy_topology.py's final evaluation exactly (same
reference search seed, same bound_and_evaluate seed, 5 restarts), then:

  1. Polishes EVERY restart tied with the best lambda_min (stage-2
     threshold, rel_tol=1e-7 -- same as coupling_optimizer.py), and
     reports whether tied restarts differ in log_product.
  2. Reports the locked-eigenvalue spectrum at theta* (multiplicity
     structure of lambda_min -- is the optimum a plateau, or an isolated
     point where several eigenvalues coalesce?).
  3. Tolerance sweep: re-polishes the best restart allowing lambda_min to
     drop by rel_tol in {1e-7, 1e-5, 1e-4, 1e-3, 1e-2}, to show how much
     log_product a small, deliberate lambda_min sacrifice would buy. This
     is a diagnostic trade-off curve, not a change to the lexicographic
     default.
  4. Loosened tie tolerance: polishes EVERY restart within rel_tol in
     {1e-4, 1e-3} of the best lambda_min (polish_tied_restarts with that
     rel_tol), to see whether tied restarts converge to one design or to
     distinct designs with different log_product.

Saves: results/polish_birdsfoot_champion_results.json,
results/polish_birdsfoot_champion_summary.txt

Run directly: python polish_birdsfoot_champion.py
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG, birdsfoot_spoke_name,
)
from sdp_topology_bound import find_full_rank_reference
from bound_and_evaluate import bound_and_evaluate, polish_tied_restarts
from log_product_polish import ReducedKModel, polish_log_product, wrapped_delta

CHAMPION      = [0, 1, 2, 3, 5, 6, 7, 8, 11]
SEED          = 17          # forward_greedy_topology.py's SEED
# Overridable via N_RESTARTS env var (default 5 reproduces forward_greedy_
# topology.py's evaluation exactly; any other value writes to _r<N> files).
N_RESTARTS    = int(os.environ.get('N_RESTARTS', 5))

_SUFFIX     = '' if N_RESTARTS == 5 else f'_r{N_RESTARTS}'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, f'polish_birdsfoot_champion{_SUFFIX}_results.json')
OUT_SUMMARY = os.path.join(RESULTS_DIR, f'polish_birdsfoot_champion{_SUFFIX}_summary.txt')
MAX_REF_TRIES = 300
SWEEP_REL_TOL = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2]
LOOSE_TIE_REL_TOL = [1e-4, 1e-3]
LOOSE_MAX_ITERS   = 1000
N_EIGS_SHOWN  = 6


def fmt_deg(theta):
    return '[' + ', '.join(f'{d:6.2f}' for d in np.degrees(theta)) + ']'


def main():
    system = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    target_rank = system.total_dofs - 6
    base = np.array(BIRDSFOOT_START_THETAS_DEG)
    labels = [birdsfoot_spoke_name(system.couplings[i]) for i in CHAMPION]

    ref = find_full_rank_reference(system, CHAMPION, target_rank, length_scale=1.0,
                                   base_thetas_deg=base,
                                   rng=np.random.default_rng(SEED + 1000),
                                   max_tries=MAX_REF_TRIES)
    t0 = time.perf_counter()
    res = bound_and_evaluate(system, CHAMPION, length_scale=1.0,
                             target_rank=target_rank, ref_thetas=ref,
                             n_restarts=N_RESTARTS,
                             rng=np.random.default_rng(SEED + 2000), polish=True)
    t_eval = time.perf_counter() - t0
    pol = res['polish']
    restarts = res['all_riemannian_results']
    best_stage1 = max(restarts, key=lambda r: r['lambda_min'])

    model = ReducedKModel(system, CHAMPION, best_stage1['theta'], target_rank)

    # ── Tolerance sweep on the best stage-1 restart ──
    t0 = time.perf_counter()
    sweep = []
    for rt in SWEEP_REL_TOL:
        p = polish_log_product(system, CHAMPION, best_stage1['theta'],
                               lambda_star=res['lambda_min'], target_rank=target_rank,
                               rel_tol=rt, model=model)
        sweep.append((rt, p))
    t_sweep = time.perf_counter() - t0

    # ── Loosened tie tolerance: polish every restart within the band ──
    t0 = time.perf_counter()
    loose = []
    for rt in LOOSE_TIE_REL_TOL:
        lp = polish_tied_restarts(system, CHAMPION, restarts, length_scale=1.0,
                                  target_rank=target_rank,
                                  polish_kwargs=dict(rel_tol=rt,
                                                     max_iters=LOOSE_MAX_ITERS))
        loose.append((rt, lp))
    t_loose = time.perf_counter() - t0

    # ── Report ──
    L = []
    L.append("Stage-2 log_product polish -- birds-foot size-9 champion "
             f"{CHAMPION} (spokes O-M:3,O-B:2,O-N:3,O-A:1)")
    L.append("=" * 78)
    L.append(f"Certified bound t = {res['t']:.6f}; stage-1 achieved lambda_min = "
             f"{res['lambda_min']:.8f} (gap {res['gap']:.3e}) -- reproduces "
             f"forward_greedy_topology.py's evaluation.")
    L.append(f"bound_and_evaluate + polish: {t_eval:.1f}s; tolerance sweep: {t_sweep:.1f}s")
    L.append("")
    L.append("1. Restarts (stage 1) and stage-2 polish of each tied restart")
    L.append(f"   threshold = {pol['threshold']:.8f} (lambda* - 1e-7*lambda*)")
    polished = dict(pol['per_start'])
    for idx, r in enumerate(restarts):
        lp0 = model.log_product(r['theta'])
        line = (f"   restart {idx}: lambda_min={r['lambda_min']:.8f}  "
                f"log_product={lp0:.6f}")
        if idx in polished:
            p = polished[idx]
            line += (f"  -> polished log_product={p['log_product']:.6f} "
                     f"(+{p['log_product'] - p['log_product_start']:.2e}), "
                     f"max move {p['max_abs_move_deg']:.3f} deg, {p['stop_reason']}")
        else:
            line += "  (not tied -- not polished)"
        L.append(line)
    tied = [p for _, p in pol['per_start']]
    if len(tied) > 1:
        th_ref = tied[0]['theta']
        spread = max(np.max(np.abs(np.degrees(wrapped_delta(p['theta'], th_ref))))
                     for p in tied)
        L.append(f"   {len(tied)} tied restarts; max theta separation among them "
                 f"(after polish): {spread:.3f} deg; log_product range "
                 f"{min(p['log_product'] for p in tied):.6f} .. "
                 f"{max(p['log_product'] for p in tied):.6f}")
    else:
        L.append("   only 1 restart tied with the best -- no tie-break comparison possible")
    L.append("")
    b = pol['best']
    L.append("   Final (polished) design:")
    L.append(f"     lambda_min  {b['lambda_min_start']:.8f} -> {b['lambda_min']:.8f}")
    L.append(f"     log_product {b['log_product_start']:.6f} -> {b['log_product']:.6f}  "
             f"(product of locked eigs x{np.exp(b['log_product'] - b['log_product_start']):.6f})")
    L.append(f"     theta (deg) start  {fmt_deg(b['theta_start'])}")
    L.append(f"     theta (deg) final  {fmt_deg(b['theta'])}")
    L.append(f"     spokes             {labels}")
    L.append("")

    L.append(f"2. Smallest {N_EIGS_SHOWN} locked eigenvalues at theta* (multiplicity structure)")
    mu = b['locked_eigs']
    L.append("   " + "  ".join(f"{v:.8f}" for v in mu[:N_EIGS_SHOWN]))
    L.append("   relative gaps to lambda_min: " + "  ".join(
        f"{(v - mu[0]) / mu[0]:.2e}" for v in mu[1:N_EIGS_SHOWN]))
    n_cluster = int(np.sum((mu - mu[0]) / mu[0] < 1e-4))
    L.append(f"   eigenvalues within 1e-4 (relative) of lambda_min: {n_cluster}")
    L.append("")

    L.append("3. Tolerance sweep (best stage-1 restart; how much log_product a "
             "deliberate lambda_min sacrifice buys)")
    L.append(f"   {'rel_tol':>8}  {'lambda_min':>12}  {'lam loss %':>10}  "
             f"{'log_product':>12}  {'d log_prod':>10}  {'prod ratio':>10}  "
             f"{'lambda_2':>11}  {'max move':>8}  stop")
    for rt, p in sweep:
        L.append(f"   {rt:8.0e}  {p['lambda_min']:12.8f}  "
                 f"{100 * (1 - p['lambda_min'] / res['lambda_min']):10.4f}  "
                 f"{p['log_product']:12.6f}  "
                 f"{p['log_product'] - p['log_product_start']:10.4f}  "
                 f"{np.exp(p['log_product'] - p['log_product_start']):10.4f}  "
                 f"{p['locked_eigs'][1]:11.8f}  {p['max_abs_move_deg']:7.2f}d  "
                 f"{p['stop_reason']}")

    L.append("")
    L.append(f"4. Loosened tie tolerance: polish EVERY restart within the band "
             f"(max_iters={LOOSE_MAX_ITERS}; {t_loose:.1f}s)")
    for rt, lp in loose:
        L.append(f"   rel_tol={rt:.0e}  threshold={lp['threshold']:.8f}  "
                 f"({len(lp['per_start'])}/{len(restarts)} restarts tied)")
        for idx, p in lp['per_start']:
            L.append(f"     restart {idx}: lambda_min {p['lambda_min_start']:.8f} -> "
                     f"{p['lambda_min']:.8f}  log_product {p['log_product_start']:.6f} -> "
                     f"{p['log_product']:.6f}  lambda_2 {p['locked_eigs'][1]:.8f}  "
                     f"move {p['max_abs_move_deg']:5.2f}d  {p['stop_reason']} "
                     f"({p['n_iterations']} it)")
        ps = [p for _, p in lp['per_start']]
        ids = [i for i, _ in lp['per_start']]
        if len(ps) > 1:
            L.append("     pairwise max theta separation AFTER polish (deg):")
            for a, pa in zip(ids, ps):
                L.append(f"       {a}: " + " ".join(
                    f"{np.max(np.abs(np.degrees(wrapped_delta(pa['theta'], pb['theta'])))):6.2f}"
                    for pb in ps))
        wb = lp['best']
        i_best = next(i for i, p in lp['per_start'] if p is wb)
        L.append(f"     winner: restart {i_best}, log_product {wb['log_product']:.6f} "
                 f"(x{np.exp(wb['log_product'] - b['log_product']):.4f} vs strict-tie "
                 f"design), lambda_min {wb['lambda_min']:.8f} "
                 f"({100 * (1 - wb['lambda_min'] / res['lambda_min']):.4f}% below lambda*)")
        L.append(f"     winner theta (deg) {fmt_deg(wb['theta'])}")

    summary = "\n".join(L)
    print(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tolist = lambda a: np.asarray(a).tolist()
    pack = lambda p: dict(
        theta_deg=tolist(np.degrees(p['theta'])),
        theta_start_deg=tolist(np.degrees(p['theta_start'])),
        lambda_min=p['lambda_min'], lambda_min_start=p['lambda_min_start'],
        log_product=p['log_product'], log_product_start=p['log_product_start'],
        threshold=p['threshold'], max_abs_move_deg=p['max_abs_move_deg'],
        locked_eigs=tolist(p['locked_eigs']), stop_reason=p['stop_reason'],
        n_iterations=p['n_iterations'])
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(
            active=CHAMPION, spokes=labels, t=res['t'],
            stage1_lambda_min=res['lambda_min'], gap=res['gap'],
            restarts=[dict(lambda_min=r['lambda_min'],
                           theta_deg=tolist(np.degrees(r['theta'])),
                           log_product=model.log_product(r['theta']))
                      for r in restarts],
            polished_tied=[dict(restart=i, **pack(p)) for i, p in pol['per_start']],
            final=pack(b),
            tolerance_sweep=[dict(rel_tol=rt, **pack(p)) for rt, p in sweep],
            loose_tie=[dict(rel_tol=rt, threshold=lp['threshold'],
                            polished=[dict(restart=i, **pack(p))
                                      for i, p in lp['per_start']])
                       for rt, lp in loose],
        ), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
