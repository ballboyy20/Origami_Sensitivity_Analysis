"""
birdsfoot_six_sites.py

Contact-LOCATION choice on birds-foot: 6 candidate sites per spoke (m=24)
instead of 3 (m=12), forward bound-guided greedy choosing k=9 -- does
giving greedy more options let it match or beat the 3-site result
(exact exhaustive champion, lambda_min 0.009263)?

Sites per spoke (spoke-major order, 6 per spoke): the existing three
(20%/80% along the spoke on the top face, 50% on the bottom face), then
each mirrored through the mid-thickness plane z = -t/2 -- the same
construction as two_panel_six_sites.py (see KINEMATIC_COUPLING_SUMMARY.md
sec 7). Index 6*s + j: j=0..2 original, j=3..5 mirror of j-3.

The old 3-site champion [0,1,2,3,5,6,7,8,11] maps to
[0,1,2,6,8,12,13,14,20] here, so it is one of the candidate subsets: the
true 9-of-24 optimum is >= the old champion. It is re-evaluated in the
24-site system as a consistency check.

No exhaustive ground truth (C(24,9) ~ 1.3M subsets). Evidence instead:
  - the root certified bound (bound_node with everything undecided): an
    upper bound on lambda_min for ANY 9-of-24 design;
  - the final set's own certified bound vs. achieved value;
  - per-step tie counts (candidates within 1e-6 of the best bound score)
    -- the failure mode found on two-panel (sec 7.3).

Figures (same types as forward_greedy_full_gore.py):
  results/birdsfoot_six_sites_3d_before_after.png  -- all 24 candidates /
      old 3-site champion / new greedy result, weakest mode overlaid
  results/birdsfoot_six_sites_hinges.png           -- one hinge layout per spoke
  results/birdsfoot_six_sites_convergence.png      -- bound chosen per step

Saves: results/birdsfoot_six_sites_{results.json,summary.txt}

Run directly: python birdsfoot_six_sites.py
"""

import sys, os, json, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import KinematicCoupling, CouplingSystem
from interactive_optimizer import (
    BIRDSFOOT_PANELS, BIRDSFOOT_SPOKES, BIRDSFOOT_START_THETAS_DEG,
    build_birdsfoot_system, birdsfoot_spoke_name, _spoke_face_normal,
    _spoke_contact_points, _bfO, t,
)
from sdp_topology_bound import (
    bound_node, build_coupling_registry, find_full_rank_reference,
)
from bound_and_evaluate import bound_and_evaluate
from forward_greedy_topology import forward_greedy
from visualization_rigid import draw_3d_config, draw_hinge_layout

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON     = os.path.join(RESULTS_DIR, 'birdsfoot_six_sites_results.json')
OUT_SUMMARY  = os.path.join(RESULTS_DIR, 'birdsfoot_six_sites_summary.txt')
FIG_3D       = os.path.join(RESULTS_DIR, 'birdsfoot_six_sites_3d_before_after.png')
FIG_HINGES   = os.path.join(RESULTS_DIR, 'birdsfoot_six_sites_hinges.png')
FIG_CONVERGE = os.path.join(RESULTS_DIR, 'birdsfoot_six_sites_convergence.png')

K                = 9
N_RESTARTS_FINAL = 5
MAX_REF_TRIES    = 300
MAX_K_BUMPS      = 3
SEED             = 17          # same as forward_greedy_topology.py
TIE_TOL          = 1e-6

OLD_CHAMPION_12    = [0, 1, 2, 3, 5, 6, 7, 8, 11]
OLD_CHAMPION_LAM   = 0.009263   # forward_greedy_topology.py / exhaustive
SITES_PER_SPOKE    = 6
SITE_TAGS = ['20% top', '80% top', '50% bot', "20% bot'", "80% bot'", "50% top'"]

MID_Z = -t / 2.0


def _mirror(p):
    q = np.array(p, dtype=float).copy()
    q[2] = 2.0 * MID_Z - q[2]
    return q


def six_site_start_thetas_deg():
    """Originals keep BIRDSFOOT_START_THETAS_DEG; mirrored sites get a
    fixed +47 deg offset (mod 180) so no mirror starts parallel to its
    original."""
    out = []
    for s in range(len(BIRDSFOOT_SPOKES)):
        orig = BIRDSFOOT_START_THETAS_DEG[3 * s: 3 * s + 3]
        out.extend(orig)
        out.extend(((d + 47.0) % 180.0) or 4.0 for d in orig)
    return np.array(out)


START_THETAS_DEG = six_site_start_thetas_deg()


def map_12_to_24(idx12):
    return sorted(SITES_PER_SPOKE * (i // 3) + (i % 3) for i in idx12)


def build_six_site_birdsfoot(theta_degs=START_THETAS_DEG):
    system = CouplingSystem(list(BIRDSFOOT_PANELS))
    idx = 0
    for panel_A, panel_B, far_point in BIRDSFOOT_SPOKES:
        normal = _spoke_face_normal(far_point, panel_A, panel_B)
        base = _spoke_contact_points(far_point)
        for p in base + [_mirror(q) for q in base]:
            system.add_coupling(KinematicCoupling(
                panel_A, panel_B, p, normal, theta=np.radians(theta_degs[idx])))
            idx += 1
    return system


def apply_design(system, active, theta):
    by_idx = dict(zip(sorted(active), theta))
    for i, c in enumerate(system.couplings):
        c.active = i in by_idx
        if c.active:
            c.set_theta(float(by_idx[i]))


def evaluate(system, active, target_rank, seed):
    ref = find_full_rank_reference(system, active, target_rank,
                                   base_thetas_deg=START_THETAS_DEG,
                                   rng=np.random.default_rng(seed + 1000),
                                   max_tries=MAX_REF_TRIES)
    if ref is None:
        return None
    return bound_and_evaluate(system, active, target_rank=target_rank,
                              ref_thetas=ref, n_restarts=N_RESTARTS_FINAL,
                              rng=np.random.default_rng(seed + 2000), polish=True)


def site_desc(system, i):
    return f"{birdsfoot_spoke_name(system.couplings[i])}:{SITE_TAGS[i % SITES_PER_SPOKE]}"


def main():
    system = build_six_site_birdsfoot()
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    k = K
    spoke_order = ['O-M', 'O-B', 'O-N', 'O-A']
    registry = build_coupling_registry(system, np.radians(START_THETAS_DEG),
                                       list(range(m)), 1.0)

    # ── Root bound: upper limit for ANY 9-of-24 design ───────────────
    root = bound_node(system, forced_active=[], forced_inactive=[],
                      undecided=list(range(m)), k=K, target_rank=target_rank,
                      base_thetas_deg=START_THETAS_DEG, registry=registry,
                      rng=np.random.default_rng(SEED))

    # ── Forward greedy, with k-bumps if the k=9 set is rank-deficient ──
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    g = forward_greedy(system, m, k, target_rank, base_thetas_deg=START_THETAS_DEG,
                       registry=registry, rng=rng, verbose=True)
    trace = list(g['trace'])
    ref = find_full_rank_reference(system, g['active'], target_rank,
                                   base_thetas_deg=START_THETAS_DEG,
                                   rng=np.random.default_rng(SEED + 1000),
                                   max_tries=MAX_REF_TRIES)
    n_bumps = 0
    while ref is None and n_bumps < MAX_K_BUMPS:
        n_bumps += 1
        k += 1
        print(f"\nNo full-rank reference at k={k - 1}; extending greedy to k={k}.")
        g = forward_greedy(system, m, k, target_rank, base_thetas_deg=START_THETAS_DEG,
                           registry=registry, rng=rng, forced_active_start=g['active'])
        trace.extend(g['trace'])
        ref = find_full_rank_reference(system, g['active'], target_rank,
                                       base_thetas_deg=START_THETAS_DEG,
                                       rng=np.random.default_rng(SEED + 1000),
                                       max_tries=MAX_REF_TRIES)
    t_greedy = time.perf_counter() - t0
    if ref is None:
        raise RuntimeError(f"No full-rank reference after {MAX_K_BUMPS} k-bumps")
    active = g['active']

    # ── Evaluate new greedy set and the old champion, same pipeline ──
    sys_new = build_six_site_birdsfoot()
    new = evaluate(sys_new, active, target_rank, SEED)
    apply_design(sys_new, active, new['polish']['best']['theta'])

    old24 = map_12_to_24(OLD_CHAMPION_12)
    sys_old = build_six_site_birdsfoot()
    old = evaluate(sys_old, old24, target_rank, SEED)
    apply_design(sys_old, old24, old['polish']['best']['theta'])

    # ── Report ─────────────────────────────────────────────────────────
    def spoke_counts(act, sysx):
        c = {s: 0 for s in spoke_order}
        for i in act:
            c[birdsfoot_spoke_name(sysx.couplings[i])] += 1
        return ','.join(f'{s}:{c[s]}' for s in spoke_order)

    n_mirror = sum(1 for i in active if i % SITES_PER_SPOKE >= 3)
    L = []
    L.append(f"Birds-foot, 6 candidate sites per spoke (m={m}): forward greedy, k={k}")
    L.append("=" * 78)
    L.append("Sites per spoke: originals (20% top, 80% top, 50% bottom) + each mirrored "
             "through mid-thickness z=-t/2.")
    L.append("")
    L.append(f"Root certified bound (ANY {K}-of-{m} design): "
             + (f"{root['bound']:.6f}" if root['status'] == 'bounded' else root['status']))
    L.append(f"   (3-site problem's best: {OLD_CHAMPION_LAM:.6f}; the 24-site optimum lies "
             f"between that and the root bound)")
    L.append("")
    L.append(f"Forward greedy: {t_greedy:.1f}s, final k={k} ({n_bumps} k-bump(s)), "
             f"{g['n_unresolved_events']} unresolved events in last round")
    L.append(f"   active: {active}")
    L.append(f"   spokes: [{spoke_counts(active, system)}]; {n_mirror} of {len(active)} "
             f"are mirrored sites")
    L.append(f"   sites:  {[site_desc(system, i) for i in active]}")
    L.append("")
    L.append("Comparison (same evaluation pipeline: certified bound + 5-restart "
             "Riemannian + strict stage-2 polish)")
    L.append(f"   {'design':32s} {'lambda_min':>10s} {'bound t':>10s} {'gap %':>7s} "
             f"{'log_prod':>9s}")
    for name, r in ((f'old 3-site champion {old24}', old), (f'new 6-site greedy', new)):
        L.append(f"   {name:32s} {r['lambda_min']:10.6f} {r['t']:10.6f} "
                 f"{100 * r['gap'] / r['lambda_min']:7.3f} "
                 f"{r['polish']['best']['log_product']:9.4f}")
    ratio = new['lambda_min'] / old['lambda_min']
    L.append(f"   new / old = {ratio:.4f}  ->  "
             + ("BETTER" if ratio > 1 + 1e-4 else "SAME (within 1e-4)" if ratio > 1 - 1e-4
                else "WORSE"))
    if root['status'] == 'bounded':
        L.append(f"   new greedy is {100 * new['lambda_min'] / root['bound']:.1f}% of the "
                 f"root bound (a guaranteed-valid but possibly loose ceiling)")
    L.append("")
    L.append("Greedy trace (ties = candidates within 1e-6 of the chosen bound score):")
    for step, tr in enumerate(trace):
        n_ties = sum(1 for s in tr['all_scores'].values()
                     if s >= tr['chosen_score'] - TIE_TOL)
        L.append(f"   step {step + 1}: site {tr['chosen']:2d} ({site_desc(system, tr['chosen'])}), "
                 f"bound {tr['chosen_score']:.6f}, {n_ties}/{len(tr['all_scores'])} tied")
    summary = "\n".join(L)
    print("\n" + summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pk = lambda r, act: dict(active=act, lambda_min=r['lambda_min'], t=r['t'], gap=r['gap'],
                             log_product=r['polish']['best']['log_product'],
                             theta_deg=np.degrees(r['polish']['best']['theta']).tolist())
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(m=m, k=k, n_k_bumps=n_bumps, start_thetas_deg=START_THETAS_DEG.tolist(),
                       root_bound=root.get('bound'), greedy_elapsed_sec=t_greedy,
                       new_greedy=pk(new, active), old_champion=pk(old, old24),
                       trace=trace), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")

    # ── Figure 1: 3-D, all candidates / old champion / new greedy ─────
    sys_all = build_six_site_birdsfoot()
    lam_all = sys_all.get_rigidity_eigenvalue()
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle(f'Birds-foot, 6 sites per spoke (m={m}): all candidates vs. '
                 f'old 3-site champion vs. new forward-greedy result',
                 fontweight='bold', fontsize=12)
    for col, (sx, ttl) in enumerate((
            (sys_all, f'All {m} candidates active (start thetas)\n'
                      f'lambda_min={lam_all:.5f}'),
            (sys_old, f'Old 3-site champion (9 of the original 12)\n'
                      f'lambda_min={old["lambda_min"]:.6f}'),
            (sys_new, f'New: forward greedy over 24 sites, {len(active)} active\n'
                      f'lambda_min={new["lambda_min"]:.6f}, bound={new["t"]:.6f}'))):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        draw_3d_config(ax, sx, title=ttl, show_lambda_min_mode=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_3D, dpi=150)
    plt.close(fig)

    # ── Figure 2: hinge layouts, old vs new, one row per design ──────
    fig = plt.figure(figsize=(20, 8.5))
    fig.suptitle('Groove arrangement per spoke: old 3-site champion (top) vs. '
                 'new 6-site greedy (bottom)', fontweight='bold', fontsize=12)
    for row, (sx, tag) in enumerate(((sys_old, 'old'), (sys_new, 'new'))):
        for s, (_, _, far_point) in enumerate(BIRDSFOOT_SPOKES):
            ax = fig.add_subplot(2, 4, 4 * row + s + 1)
            cs = sx.couplings[SITES_PER_SPOKE * s: SITES_PER_SPOKE * (s + 1)]
            act = [c for c in cs if c.active]
            ina = [c for c in cs if not c.active]
            draw_hinge_layout(ax, act, _bfO, far_point, cs[0].face_normal, t,
                              title=f'{tag}: spoke {birdsfoot_spoke_name(cs[0])} '
                                    f'({len(act)}/6 active)',
                              inactive_couplings=ina)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_HINGES, dpi=150)
    plt.close(fig)

    # ── Figure 3: convergence ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(1, len(trace) + 1)
    ax.plot(xs, [tr['chosen_score'] for tr in trace], marker='o', color='steelblue',
            label='Certified bound chosen at each greedy step')
    ax.axhline(new['lambda_min'], color='#E74C3C', linestyle='--', linewidth=2,
               label=f'New greedy achieved lambda_min ({new["lambda_min"]:.6f})')
    ax.axhline(old['lambda_min'], color='black', linestyle=':', linewidth=1.5,
               label=f'Old 3-site champion ({old["lambda_min"]:.6f})')
    if root['status'] == 'bounded':
        ax.axhline(root['bound'], color='#2ECC71', linestyle=':', linewidth=1.5,
                   label=f'Root bound, any 9-of-24 ({root["bound"]:.6f})')
    ax.set_xlabel('Number of active couplings')
    ax.set_ylabel('Certified bound (best-case completion)')
    ax.set_title(f'Forward-greedy convergence: birds-foot, 6 sites per spoke (m={m})')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_CONVERGE, dpi=150)
    plt.close(fig)

    for p in (OUT_JSON, OUT_SUMMARY, FIG_3D, FIG_HINGES, FIG_CONVERGE):
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
