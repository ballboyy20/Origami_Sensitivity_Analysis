"""
two_panel_six_sites.py

Contact-LOCATION choice on the two-panel system: 6 candidate coupling
sites instead of 3, choose which k to use (plus each one's theta).

Candidate sites (all on the shared x=1 mating face): the existing three,
plus each one mirrored through the panels' mid-thickness plane
z = -t/2 (panels span z in [-t, 0]):

    0  p1  = (1, 0.2,  0 )  top face         3  p1' = (1, 0.2, -t)  bottom face
    1  p2  = (1, 0.8,  0 )  top face         4  p2' = (1, 0.8, -t)  bottom face
    2  p3  = (1, 0.5, -t )  bottom face      5  p3' = (1, 0.5,  0)  top face

(p3 sits on the BOTTOM face, z=-t, not at mid-thickness -- so its mirror
p3' is a genuinely new top-face point, not a duplicate.) The original
3-site design [0,1,2] is one of the candidate subsets, so the comparison
"does extra location choice help?" is direct.

Two collinear triples exist: {0,5,1} (all on the top edge, y=.2/.5/.8)
and {3,2,4} (all on the bottom edge). Three contacts on one line leave
rotation about that line unconstrained, so these should come out
structurally rank-deficient at every theta -- a built-in sanity check.

What this runs:
  1. Exhaustive ground truth: bound_and_evaluate (certified SDP bound +
     5-restart Riemannian) for EVERY subset of size k = 3..6. m=6 is
     small enough (42 subsets) that this is cheap.
  2. Forward bound-guided greedy (forward_greedy_topology.forward_greedy)
     at k=3, compared against the exhaustive optimum.
  3. Stage-2 log_product polish (log_product_polish.py) of the best k=3
     design -- two-panel has a known lambda_min plateau, so the polish is
     what picks the specific theta to build.
  4. Before/after 3-D figure: all 6 candidates at their starting thetas
     vs. the chosen k=3 design at its polished thetas, weakest locked
     mode overlaid on each.

Saves: results/two_panel_six_sites_{results.json,summary.txt,3d_before_after.png}

Run directly: python two_panel_six_sites.py
"""

import sys, os, json, time
from itertools import combinations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import KinematicCoupling, CouplingSystem
from interactive_optimizer import (
    panel_A, panel_B, face_normal, t, CONTACT_POINTS, ARBITRARY_START_THETAS_DEG,
)
from sdp_topology_bound import find_full_rank_reference, build_coupling_registry
from bound_and_evaluate import bound_and_evaluate
from forward_greedy_topology import forward_greedy
from visualization_rigid import draw_3d_config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON    = os.path.join(RESULTS_DIR, 'two_panel_six_sites_results.json')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'two_panel_six_sites_summary.txt')
OUT_FIG     = os.path.join(RESULTS_DIR, 'two_panel_six_sites_3d_before_after.png')

SEED          = 23
N_RESTARTS    = 5
MAX_REF_TRIES = 300
K_GREEDY      = 3

MID_Z = -t / 2.0


def mirror_mid_thickness(p):
    q = np.array(p, dtype=float).copy()
    q[2] = 2.0 * MID_Z - q[2]
    return q


SITES = list(CONTACT_POINTS) + [mirror_mid_thickness(p) for p in CONTACT_POINTS]
SITE_LABELS = ['p1 (y.2 top)', 'p2 (y.8 top)', 'p3 (y.5 bot)',
               "p1' (y.2 bot)", "p2' (y.8 bot)", "p3' (y.5 top)"]
# Originals keep the established two-panel start; mirrored sites get a
# different non-degenerate spread so no two grooves start parallel.
START_THETAS_DEG = tuple(ARBITRARY_START_THETAS_DEG) + (51.0, 118.0, 129.0)
ORIGINAL_SET = [0, 1, 2]


def build_six_site_system(theta_degs=START_THETAS_DEG):
    system = CouplingSystem([panel_A, panel_B])
    for p, deg in zip(SITES, theta_degs):
        system.add_coupling(KinematicCoupling(
            panel_A, panel_B, p, face_normal, theta=np.radians(deg)))
    return system


def evaluate_subset(active, polish=False, seed=SEED):
    """Fresh system each call so no state leaks between subsets."""
    system = build_six_site_system()
    target_rank = system.total_dofs - 6
    ref = find_full_rank_reference(system, active, target_rank,
                                   base_thetas_deg=np.array(START_THETAS_DEG),
                                   rng=np.random.default_rng(seed),
                                   max_tries=MAX_REF_TRIES)
    if ref is None:
        return system, None
    res = bound_and_evaluate(system, active, target_rank=target_rank,
                             ref_thetas=ref, n_restarts=N_RESTARTS,
                             rng=np.random.default_rng(seed + 1), polish=polish)
    return system, res


def apply_design(system, active, theta):
    by_idx = dict(zip(sorted(active), theta))
    for i, c in enumerate(system.couplings):
        c.active = i in by_idx
        if c.active:
            c.set_theta(float(by_idx[i]))


def main():
    m = len(SITES)
    lines = []
    out = dict(sites=[p.tolist() for p in SITES], site_labels=SITE_LABELS,
               start_thetas_deg=list(START_THETAS_DEG), exhaustive={})

    # ── 1. Exhaustive ground truth ────────────────────────────────────
    t0 = time.perf_counter()
    table = {}
    for k in range(3, m + 1):
        rows = []
        for active in combinations(range(m), k):
            active = list(active)
            _, res = evaluate_subset(active)
            if res is None:
                rows.append(dict(active=active, feasible=False))
            else:
                rows.append(dict(active=active, feasible=True, t=res['t'],
                                 lambda_min=res['lambda_min'], gap=res['gap'],
                                 theta_deg=np.degrees(res['theta']).tolist()))
        table[k] = rows
        out['exhaustive'][str(k)] = rows
    t_exh = time.perf_counter() - t0

    # ── 2. Forward greedy at k=3 ───────────────────────────────────────
    t0 = time.perf_counter()
    gsys = build_six_site_system()
    target_rank = gsys.total_dofs - 6
    registry = build_coupling_registry(gsys, np.radians(START_THETAS_DEG),
                                       list(range(m)), 1.0)
    greedy = forward_greedy(gsys, m, K_GREEDY, target_rank,
                            base_thetas_deg=np.array(START_THETAS_DEG),
                            registry=registry, rng=np.random.default_rng(SEED + 5),
                            verbose=False)
    t_greedy = time.perf_counter() - t0

    # ── 3. Polish the best k=3 design ──────────────────────────────────
    feas3 = [r for r in table[3] if r['feasible']]
    best3 = max(feas3, key=lambda r: r['lambda_min'])
    sys_after, res_after = evaluate_subset(best3['active'], polish=True)
    pol = res_after['polish']['best']
    apply_design(sys_after, best3['active'], pol['theta'])

    # Original 3-site design, polished too, for a like-for-like comparison
    sys_orig, res_orig = evaluate_subset(ORIGINAL_SET, polish=True)
    pol_orig = res_orig['polish']['best']

    # ── Report ─────────────────────────────────────────────────────────
    lab = lambda active: '{' + ', '.join(SITE_LABELS[i].split(' ')[0] for i in active) + '}'
    lines.append("Two-panel, 6 candidate contact sites (3 original + 3 mirrored "
                 "through mid-thickness z=-t/2)")
    lines.append("=" * 78)
    for i, (p, l) in enumerate(zip(SITES, SITE_LABELS)):
        lines.append(f"  site {i}: {l:15s} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:+.2f})  "
                     f"start theta {START_THETAS_DEG[i]:.0f} deg")
    lines.append("")
    lines.append(f"1. Exhaustive ground truth ({sum(len(v) for v in table.values())} "
                 f"subsets, k=3..{m}; {t_exh:.1f}s)")
    max_viol = 0.0
    for k in range(3, m + 1):
        feas = [r for r in table[k] if r['feasible']]
        infeas = [r for r in table[k] if not r['feasible']]
        best = max(feas, key=lambda r: r['lambda_min'])
        max_viol = max(max_viol, max(-r['gap'] for r in feas))
        lines.append(f"   k={k}: {len(feas)}/{len(table[k])} full-rank; best "
                     f"lambda_min={best['lambda_min']:.6f} at {best['active']} {lab(best['active'])}"
                     + (f"; rank-deficient: {[r['active'] for r in infeas]}" if infeas else ""))
    lines.append(f"   soundness: max (achieved - certified bound) = {max_viol:.2e} "
                 f"({'PASS' if max_viol < 1e-6 else 'FAIL'})")
    lines.append("")
    lines.append("   All k=3 subsets, ranked:")
    for r in sorted(table[3], key=lambda r: -r.get('lambda_min', -1)):
        if r['feasible']:
            tag = '  <- original design' if r['active'] == ORIGINAL_SET else ''
            lines.append(f"     {str(r['active']):12s} {lab(r['active']):22s} "
                         f"lambda_min={r['lambda_min']:.6f}  bound={r['t']:.6f}{tag}")
        else:
            lines.append(f"     {str(r['active']):12s} {lab(r['active']):22s} "
                         f"rank-deficient (no full-rank theta found)")
    lines.append("")
    orig_row = next(r for r in table[3] if r['active'] == ORIGINAL_SET)
    n_tied = sum(1 for r in feas3 if r['lambda_min'] >= best3['lambda_min'] * (1 - 1e-4))
    lines.append(f"   Best k=3: {best3['active']} {lab(best3['active'])} "
                 f"lambda_min={best3['lambda_min']:.6f}  vs. original {ORIGINAL_SET}: "
                 f"{orig_row['lambda_min']:.6f}  (x{best3['lambda_min'] / orig_row['lambda_min']:.3f})")
    lines.append(f"   k=3 subsets within 1e-4 (relative) of the best: {n_tied}")
    lines.append("")
    lines.append(f"2. Forward greedy, k={K_GREEDY} ({t_greedy:.1f}s, "
                 f"{greedy['n_unresolved_events']} unresolved events)")
    g_row = next(r for r in table[3] if r['active'] == greedy['active'])
    lines.append(f"   chose {greedy['active']} {lab(greedy['active'])}: lambda_min="
                 f"{g_row.get('lambda_min', float('nan')):.6f} = "
                 f"{100 * g_row.get('lambda_min', 0) / best3['lambda_min']:.1f}% of exhaustive best")
    for tr in greedy['trace']:
        ranked = sorted(tr['all_scores'].items(), key=lambda kv: -kv[1])
        lines.append(f"   step {tr['step'] + 1}: chose site {tr['chosen']} "
                     f"(bound {tr['chosen_score']:.6f}); scores "
                     + ', '.join(f"{c}:{s:.5f}" for c, s in ranked))
    lines.append("")
    lines.append("3. Stage-2 log_product polish (rel_tol 1e-7)")
    for name, active, pz in (('best k=3', best3['active'], pol),
                             ('original', ORIGINAL_SET, pol_orig)):
        lines.append(f"   {name:9s} {str(active):12s} lambda_min {pz['lambda_min']:.6f}  "
                     f"log_product {pz['log_product_start']:.4f} -> {pz['log_product']:.4f}  "
                     f"theta(deg) [{', '.join(f'{d:.1f}' for d in np.degrees(pz['theta']))}]  "
                     f"locked eigs [{', '.join(f'{v:.4f}' for v in pz['locked_eigs'])}]")

    summary = "\n".join(lines)
    print(summary)

    out.update(
        greedy=dict(active=greedy['active'], trace=greedy['trace'],
                    elapsed_sec=t_greedy),
        best_k3=dict(active=best3['active'], lambda_min=pol['lambda_min'],
                     log_product=pol['log_product'],
                     theta_deg=np.degrees(pol['theta']).tolist(),
                     locked_eigs=np.asarray(pol['locked_eigs']).tolist(),
                     t=res_after['t']),
        original=dict(active=ORIGINAL_SET, lambda_min=pol_orig['lambda_min'],
                      log_product=pol_orig['log_product'],
                      theta_deg=np.degrees(pol_orig['theta']).tolist(),
                      locked_eigs=np.asarray(pol_orig['locked_eigs']).tolist()),
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")

    # ── 4. Before/after 3-D figure ─────────────────────────────────────
    sys_before = build_six_site_system()
    lam_before = sys_before.get_rigidity_eigenvalue()
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('Two panels, 6 candidate contact sites: all candidates vs. chosen design',
                 fontweight='bold', fontsize=12)
    ax_b = fig.add_subplot(1, 2, 1, projection='3d')
    ax_a = fig.add_subplot(1, 2, 2, projection='3d')
    draw_3d_config(ax_b, sys_before,
                   title=f'Before: all {m} candidates active at start thetas\n'
                         f'(lambda_min={lam_before:.6f})',
                   show_lambda_min_mode=True)
    draw_3d_config(ax_a, sys_after,
                   title=f'After: best k=3 sites {best3["active"]}, polished thetas\n'
                         f'(lambda_min={pol["lambda_min"]:.6f}, cert. bound={res_after["t"]:.6f}; '
                         f'original 3-site: {orig_row["lambda_min"]:.6f})',
                   show_lambda_min_mode=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    if '--show' not in sys.argv:
        matplotlib.use('Agg')
    main()
    if '--show' in sys.argv:
        plt.show()
