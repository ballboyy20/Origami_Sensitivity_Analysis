"""
flasher_timing.py

Go/no-go timing check for B&B on the full flasher (full_flasher_system.py;
26 bodies, target rank 150, 24 cyclic orbits). Answers, before committing
the lab machine overnight:

  1. How long does one SDP bound take at this size (root node, leaf), with
     default Clarabel and with MOSEK (if installed)?
  2. How accurate is default Clarabel here (solver status, Clarabel vs.
     MOSEK value)? sec 7.18 found Clarabel low by up to 1.5e-4 relative on
     the 30x30 gore SDPs; B&B's bound_margin must cover the error at 150.
  3. How long does one 5-restart Riemannian evaluation take, and does the
     margined bound sit above the achieved value?
  4. Mixed types (72 candidates): root bound time.

Each SDP time is split into cvxpy compile (single-threaded Python) and
solver time: threads only help the solver part (MOSEK uses all cores by
default; Clarabel is mostly single-threaded).

Leaves: 3 random exactly constrained all-V sets (15 of 24 orbits) that
reach full rank.

Saves: results/flasher_timing_{summary.txt,results.json}.

Run: python -u flasher_timing.py     (lab machine; set *_NUM_THREADS as for
                                      the other lab runs)
"""
import sys, os, json, time
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))
from full_flasher_system import build_flasher_system, START_THETAS_DEG, N_SITES, describe_counts
from coupling_types import TypedCandidates
from swap_search import SetEvaluator
from branch_and_bound import BranchAndBound, _rank_full
from sdp_topology_bound import sdp_topology_bound, bound_node
from sdp_theta_optimizer import LAST_SOLVE_TIMING

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 31
MARGIN = 3e-4
K = 15


def split():
    """' (cvxpy compile Xs + solver Ys)' for the last SDP solve."""
    T = LAST_SOLVE_TIMING
    if 'compile_sec' not in T or T['compile_sec'] is None:
        return ''
    return f" (cvxpy compile {T['compile_sec']:.1f}s + solver {T['solver_sec'] or 0:.1f}s)"


def timed(f, *a, **kw):
    LAST_SOLVE_TIMING.clear()
    t0 = time.perf_counter()
    try:
        r = f(*a, **kw)
    except Exception as e:                       # report, don't abort the check
        return dict(error=f"{type(e).__name__}: {e}"), time.perf_counter() - t0
    return r, time.perf_counter() - t0


def main():
    L, out = [], {}

    def log(s=''):
        print(s, flush=True)
        L.append(s)

    solvers = [('Clarabel (default)', None)]
    if 'MOSEK' in cp.installed_solvers():
        solvers.append(('MOSEK', cp.MOSEK))
    log("Full flasher timing check: 26 bodies, 156 DOFs, target rank 150, 24 orbits, "
        f"k = {K} (all-V)")
    log(f"Solvers: {', '.join(n for n, _ in solvers)}; threads: "
        f"OMP={os.environ.get('OMP_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}")
    log("=" * 78)

    system = build_flasher_system()
    tr = system.total_dofs - 6
    t0 = time.perf_counter()
    ev = SetEvaluator(system, tr, START_THETAS_DEG, seed=SEED)
    log(f"Registry (24 orbits): {time.perf_counter() - t0:.1f}s")
    bnb = BranchAndBound(ev, k=K, seed=SEED, verbose=False, bound_margin=MARGIN)

    # ── 1. root bound, all-V ──
    und = list(range(N_SITES))
    ref = bnb._reference((), und, (0,))
    out['root_allv'] = []
    for name, solver in solvers:
        r, dt = timed(sdp_topology_bound, system, [], [], und, K, target_rank=tr,
                      ref_thetas=ref, registry=ev.registry, solver=solver)
        rec = dict(solver=name, sec=dt, t=r.get('t'), status=r.get('solver_status'),
                   error=r.get('error'), timing=dict(LAST_SOLVE_TIMING))
        out['root_allv'].append(rec)
        log(f"Root bound (all-V, 24 undecided), {name}: {dt:.1f}s{split()}, t = {rec['t']}, "
            f"status {rec['status']}" + (f", ERROR {rec['error']}" if rec['error'] else ''))

    # ── 2. leaf bounds on random full-rank sets ──
    rng = np.random.default_rng(SEED)
    leaves = []
    while len(leaves) < 3:
        S = sorted(rng.choice(N_SITES, K, replace=False).tolist())
        if _rank_full(system, set(S), tr, rng.uniform(0, np.pi, N_SITES), 1.0):
            leaves.append(S)
    out['leaves'] = []
    for S in leaves:
        log(f"\nLeaf {S}  ({describe_counts(S)})")
        rec = dict(active=S, bounds=[])
        for name, solver in solvers:
            r, dt = timed(bound_node, system, forced_active=S,
                          forced_inactive=[i for i in range(N_SITES) if i not in S],
                          undecided=[], k=K, target_rank=tr, base_thetas_deg=START_THETAS_DEG,
                          registry=ev.registry, rng=np.random.default_rng([SEED, 1, *S]),
                          solver=solver)
            b = dict(solver=name, sec=dt, status=r.get('status'), bound=r.get('bound'),
                     solver_status=r.get('solver_status'), error=r.get('error'),
                     timing=dict(LAST_SOLVE_TIMING))
            rec['bounds'].append(b)
            log(f"   bound, {name}: {dt:.1f}s{split()}, {b['status']}, bound {b['bound']}, "
                f"solver status {b['solver_status']}" + (f", ERROR {b['error']}" if b['error'] else ''))
        vals = [b['bound'] for b in rec['bounds'] if b['bound'] is not None]
        if len(vals) == 2:
            log(f"   Clarabel - MOSEK: {vals[0] - vals[1]:.2e} "
                f"({(vals[0] - vals[1]) / vals[1]:.2e} relative; margin is {MARGIN:.0e})")
        out['leaves'].append(rec)

    # ── 3. one 5-restart evaluation ──
    S = leaves[0]
    r, dt = timed(ev.evaluate, S)
    out['evaluation'] = dict(active=S, sec=dt, lambda_min=r.get('lambda_min'),
                             feasible=r.get('feasible'), t=r.get('t'), error=r.get('error'))
    log(f"\nEvaluation of {S} (5 restarts, includes its Clarabel bound, cached): {dt:.1f}s, "
        f"lambda_min {r.get('lambda_min')}, bound t {r.get('t')}")
    if r.get('feasible') and r.get('t') is not None:
        ok = r['t'] * (1 + MARGIN) >= r['lambda_min']
        log(f"   bound x (1 + margin) >= achieved: {'yes' if ok else 'NO -- margin too small'} "
            f"(bound - achieved = {r['t'] - r['lambda_min']:.2e})")

    # ── 4. mixed types root bound ──
    tc = TypedCandidates(system)
    t0 = time.perf_counter()
    evm = SetEvaluator(tc.system, tr, tc.base_thetas_deg, seed=SEED, rows=tc.rows, sites=tc.site_of)
    log(f"\nMixed: {tc.m} candidates, registry {time.perf_counter() - t0:.1f}s")
    bm = BranchAndBound(evm, rows=tc.rows, site_of=tc.site_of, seed=SEED, verbose=False,
                        bound_margin=MARGIN)
    undm = list(range(tc.m))
    refm = bm._reference((), undm, (0,))
    r, dt = timed(sdp_topology_bound, tc.system, [], [], undm, None, target_rank=tr,
                  ref_thetas=refm, registry=evm.registry, rows=tc.rows, sites=tc.site_of)
    out['root_mixed'] = dict(sec=dt, t=r.get('t'), status=r.get('solver_status'), error=r.get('error'),
                             timing=dict(LAST_SOLVE_TIMING))
    log(f"Root bound (mixed, 72 undecided), Clarabel: {dt:.1f}s{split()}, t = {r.get('t')}, "
        f"status {r.get('solver_status')}" + (f", ERROR {r.get('error')}" if r.get('error') else ''))

    # ── verdict inputs ──
    root = out['root_allv'][0]['sec']
    leaf = np.mean([l['bounds'][0]['sec'] for l in out['leaves']])
    log("\nRough throughput at these per-solve times (one process):")
    log(f"   ~{3600 / max(root, 1e-9):.0f} node bounds/h, ~{3600 / max(leaf, 1e-9):.0f} leaf bounds/h, "
        f"~{3600 / max(out['evaluation']['sec'], 1e-9):.0f} leaf evaluations/h")
    log("   For scale: full-gore all-V B&B (sec 7.17) took 7,625 nodes and 780 leaves.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, 'flasher_timing')
    with open(base + '_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")
    with open(base + '_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else float(o))
    print(f"Saved: {base}_summary.txt / _results.json")


if __name__ == "__main__":
    main()
