"""
forward_greedy_full_gore.py

Forward (bound-guided) greedy coupling-topology construction on the
CORRECTED flasher-gore system (full_flasher_gore_system.py, m=27
candidates -- 9 interior creases x 3 contact points, including the
inter-gore creases the original single-gore file omitted). Reuses
forward_greedy() unchanged, with the leaf-panel forced-active fix from
forward_greedy_gore.py generalized via gore_system_common's
mandatory_leaf_couplings().

Per KINEMATIC_COUPLING_SUMMARY.md sec 5.4 / roadmap item 7: this is
EXPLICITLY still a test of forward greedy's behavior at larger scale, not
a claim about the true assembled flasher pattern's lambda_min -- one
neighbor gore's panels are duplicated in as real, independently-moving
rigid panels, but that's one arbitrary boundary condition on a properly
periodic problem, not a substitute for full-assembly evaluation or
cyclic-symmetry/Bloch-wave reduction.

Also generates the same 3 visualizations produced for birds-foot:
  1. 3-D config, before (all m active) vs after (greedy result, achieved
     thetas), each with the weakest constrained eigenmode overlaid --
     mirrors test_prune_couplings.py's prune_before_after_3d.png.
  2. A grid of 2-D hinge-layout plots, one per crease, showing each
     crease's 3 candidate contact points (active: theta + n1/n2 vectors;
     inactive: removed marker) -- mirrors
     two_panel_optimizer_baseline.py's draw_hinge_layout usage.
  3. A convergence plot of the greedy construction itself: certified
     bound per step vs. active-coupling count, with the final Riemannian-
     achieved lambda_min and certified bound marked for comparison.

Saves: results/forward_greedy_full_gore_results.json,
results/forward_greedy_full_gore_summary.txt,
results/forward_greedy_full_gore_3d_before_after.png,
results/forward_greedy_full_gore_hinges.png,
results/forward_greedy_full_gore_convergence.png

Run directly: python forward_greedy_full_gore.py
"""
import sys, os, json, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from full_flasher_gore_system import (
    build_full_gore_system, FULL_GORE_START_THETAS_DEG, full_gore_spoke_name,
    FULL_GORE_SPOKE_ORDER, FULL_GORE_CREASES, FULL_GORE_PANELS,
    FULL_GORE_MANDATORY_COUPLINGS, FULL_GORE_THICKNESS,
)
from forward_greedy_topology import forward_greedy
from sdp_topology_bound import build_coupling_registry, find_full_rank_reference
from bound_and_evaluate import bound_and_evaluate
from visualization_rigid import draw_3d_config, draw_hinge_layout

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
OUT_JSON      = os.path.join(RESULTS_DIR, 'forward_greedy_full_gore_results.json')
OUT_SUMMARY   = os.path.join(RESULTS_DIR, 'forward_greedy_full_gore_summary.txt')
FIG_3D        = os.path.join(RESULTS_DIR, 'forward_greedy_full_gore_3d_before_after.png')
FIG_HINGES    = os.path.join(RESULTS_DIR, 'forward_greedy_full_gore_hinges.png')
FIG_CONVERGE  = os.path.join(RESULTS_DIR, 'forward_greedy_full_gore_convergence.png')

N_RESTARTS_FINAL = 5
MAX_REF_TRIES     = 300
SEED              = 17
MAX_K_BUMPS       = 4   # safety cap on the "no full-rank ref -> try k+1" retry loop


def main():
    system = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    m = len(system.couplings)
    target_rank = system.total_dofs - 6
    k = target_rank // 2
    base_thetas_deg = np.array(FULL_GORE_START_THETAS_DEG)
    labels = [full_gore_spoke_name(c) for c in system.couplings]
    spoke_order = list(dict.fromkeys(FULL_GORE_SPOKE_ORDER))

    forced_active = list(FULL_GORE_MANDATORY_COUPLINGS)

    print("=" * 72)
    print(f"Forward (bound-guided) greedy construction: FULL flasher gore "
          f"(inter-gore couplings included), choose {k} of {m}  "
          f"(target_rank={target_rank})")
    print("=" * 72)
    if forced_active:
        forced_names = sorted({labels[i] for i in forced_active})
        print(f"Forcing {len(forced_active)} structurally-mandatory couplings active "
              f"(leaf-panel creases: {forced_names}).\n")

    ref_thetas = np.radians(base_thetas_deg)
    registry = build_coupling_registry(system, ref_thetas, list(range(m)), 1.0)
    print(f"Shared per-coupling registry built once for all {m} candidates.\n")

    rng = np.random.default_rng(SEED)
    all_trace = []
    t0 = time.perf_counter()

    greedy_result = forward_greedy(
        system, m, k, target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, registry=registry, rng=rng,
        forced_active_start=forced_active)
    all_trace.extend(greedy_result['trace'])

    ref_rng = np.random.default_rng(SEED + 1000)
    ref = find_full_rank_reference(
        system, greedy_result['active'], target_rank, length_scale=1.0,
        base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)

    n_bumps = 0
    while ref is None and n_bumps < MAX_K_BUMPS:
        # Same situation documented in KINEMATIC_COUPLING_SUMMARY.md sec 5.2
        # for the single-gore case: the exact structural-minimum k can land
        # on an active set that provably can't reach full rank. Rather than
        # restart from scratch, extend the SAME active set by exactly one
        # more greedy step at k+1 (forced_active_start = everything already
        # chosen) -- no wasted computation, and the trace stays one
        # continuous record.
        n_bumps += 1
        k += 1
        print(f"\nNo full-rank reference at k={k - 1} -- extending greedy by "
              f"one more step (k={k}) rather than restarting.")
        greedy_result = forward_greedy(
            system, m, k, target_rank, length_scale=1.0,
            base_thetas_deg=base_thetas_deg, registry=registry, rng=rng,
            forced_active_start=greedy_result['active'])
        all_trace.extend(greedy_result['trace'])
        ref = find_full_rank_reference(
            system, greedy_result['active'], target_rank, length_scale=1.0,
            base_thetas_deg=base_thetas_deg, rng=ref_rng, max_tries=MAX_REF_TRIES)

    greedy_elapsed = time.perf_counter() - t0

    if ref is None:
        raise RuntimeError(
            f"Still no full-rank reference after {MAX_K_BUMPS} k-bumps "
            f"(final k={k}) -- this is a real result worth reporting as-is, "
            f"not pushing further automatically.")

    active = greedy_result['active']
    spoke_counts = {}
    for i in active:
        spoke_counts[labels[i]] = spoke_counts.get(labels[i], 0) + 1
    spoke_str = ','.join(f'{s}:{spoke_counts.get(s, 0)}' for s in spoke_order)

    print(f"\nGreedy construction finished in {greedy_elapsed:.1f}s "
          f"(final k={k}, {n_bumps} k-bump(s), "
          f"{sum(g['n_unresolved_events'] for g in [greedy_result])} "
          f"unresolved-reference events in final round)")
    print(f"Final active set: {active}  spokes=[{spoke_str}]")

    print("\nEvaluating final set via bound_and_evaluate "
          f"(certified bound + {N_RESTARTS_FINAL}-restart Riemannian achieved value)...")
    final = bound_and_evaluate(system, active, length_scale=1.0,
                                target_rank=target_rank, ref_thetas=ref,
                                n_restarts=N_RESTARTS_FINAL,
                                rng=np.random.default_rng(SEED + 2000))

    lines = [
        f"Forward (bound-guided) greedy topology construction -- FULL flasher "
        f"gore (inter-gore couplings included), choose {k} of {m}",
        "=" * 78,
        "",
        "TEST OF THE ALGORITHM AT LARGER SCALE, NOT A FINAL PHYSICAL RESULT: "
        "this gore includes real duplicate panels for ONE neighboring gore "
        "so the inter-gore creases the original single-gore file omitted can "
        "be represented, but that is one arbitrary boundary condition on "
        "what is properly a rotationally-periodic problem -- see "
        "KINEMATIC_COUPLING_SUMMARY.md sec 5.4 / roadmap item 7 (full-"
        "assembly evaluation or cyclic-symmetry/Bloch-wave reduction are "
        "the two rigorous next steps for a trustworthy physical number).",
        "",
        f"Structurally-mandatory couplings forced active up front: "
        f"{forced_active} ({sorted({labels[i] for i in forced_active})})",
        f"Final active set ({len(active)} of {m}): {active}",
        f"Spoke distribution: [{spoke_str}]",
        f"Greedy construction time: {greedy_elapsed:.1f}s "
        f"(k ended at {k} after {n_bumps} bump(s) past the structural "
        f"minimum {target_rank // 2})",
        "",
        f"Certified bound at final set:  t = {final['t']:.6f}",
        f"Achieved lambda_min (Riemannian, {N_RESTARTS_FINAL} restarts): "
        f"{final['lambda_min']:.6f}",
        f"Certification gap: {final['gap']:.3e} "
        f"({100 * final['gap'] / final['lambda_min']:.2f}% of achieved value)",
    ]

    lines.append("")
    lines.append("Step-by-step trace (includes any k-bump extension steps):")
    for tr in all_trace:
        lines.append(f"  chose coupling {tr['chosen']} "
                    f"(spoke {labels[tr['chosen']]}), bound={tr['chosen_score']:.6f}")

    summary = "\n".join(lines)
    print("\n" + summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(
            m=m, k=k, target_rank=target_rank, n_k_bumps=n_bumps,
            forced_active=forced_active,
            active=active, spoke_counts=spoke_counts,
            t=final['t'], lambda_min=final['lambda_min'], gap=final['gap'],
            greedy_elapsed_sec=greedy_elapsed,
            trace=all_trace,
        ), f, indent=1)
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}")

    # ════════════════════════════════════════════════════════════════
    # Build the "after" system: final topology at its achieved thetas
    # ════════════════════════════════════════════════════════════════
    system_after = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    active_set = set(active)
    theta_by_active_idx = dict(zip(active, final['theta']))
    for i, c in enumerate(system_after.couplings):
        if i in active_set:
            c.active = True
            c.set_theta(theta_by_active_idx[i])
        else:
            c.active = False

    system_before = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))  # all m active

    lam_before = system_before.get_rigidity_eigenvalue(length_scale=1.0)

    # ── 1. 3-D before/after ─────────────────────────────────────────
    fig1 = plt.figure(figsize=(14, 7))
    fig1.suptitle('Full flasher gore: all candidates vs. forward-greedy result',
                  fontweight='bold', fontsize=12)
    ax_before = fig1.add_subplot(1, 2, 1, projection='3d')
    ax_after  = fig1.add_subplot(1, 2, 2, projection='3d')
    draw_3d_config(ax_before, system_before,
                   title=f'Before: all {m} candidates active\n(lambda_min={lam_before:.4f})',
                   show_lambda_min_mode=True)
    draw_3d_config(ax_after, system_after,
                   title=f'After: forward-greedy result, {len(active)} of {m} active\n'
                         f'(lambda_min={final["lambda_min"]:.4f}, cert. bound={final["t"]:.4f})',
                   show_lambda_min_mode=True)
    plt.tight_layout()
    fig1.savefig(FIG_3D, dpi=150)
    print(f"Saved: {FIG_3D}")
    plt.close(fig1)

    # ── 2. 2-D hinge layouts, one per crease ────────────────────────
    n_creases = len(FULL_GORE_CREASES)
    ncols = 3
    nrows = -(-n_creases // ncols)
    fig2 = plt.figure(figsize=(5 * ncols, 4 * nrows))
    fig2.suptitle('Full flasher gore: groove arrangement per crease (after greedy)',
                  fontweight='bold', fontsize=12)
    for ci, crease in enumerate(FULL_GORE_CREASES):
        ax = fig2.add_subplot(nrows, ncols, ci + 1)
        crease_couplings = system_after.couplings[3 * ci: 3 * ci + 3]
        active_c   = [c for c in crease_couplings if c.active]
        inactive_c = [c for c in crease_couplings if not c.active]
        face_normal = crease_couplings[0].face_normal
        spoke_label = full_gore_spoke_name(crease_couplings[0])
        draw_hinge_layout(ax, active_c, crease['p_start'], crease['p_end'],
                          face_normal, FULL_GORE_THICKNESS,
                          title=f'Crease {spoke_label} '
                                f'({len(active_c)}/3 active)',
                          inactive_couplings=inactive_c)
    plt.tight_layout()
    fig2.savefig(FIG_HINGES, dpi=150)
    print(f"Saved: {FIG_HINGES}")
    plt.close(fig2)

    # ── 3. Convergence plot ─────────────────────────────────────────
    n_forced = len(forced_active)
    xs = list(range(n_forced, n_forced + len(all_trace) + 1))
    bound_scores = [None] + [tr['chosen_score'] for tr in all_trace]
    # x[0] (n_forced active, all mandatory, no greedy choice yet) has no
    # bound score of its own -- start the plotted line from the first
    # actual greedy step instead of inventing a value for it.
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    ax3.plot(xs[1:], bound_scores[1:], marker='o', color='steelblue',
             linewidth=1.6, markersize=4,
             label='Certified bound chosen at each greedy step')
    ax3.axvline(n_forced, color='gray', linestyle=':', linewidth=1,
                label=f'{n_forced} mandatory couplings forced active first')
    ax3.axhline(final['lambda_min'], color='#E74C3C', linewidth=2,
                linestyle='--',
                label=f'Final achieved lambda_min ({final["lambda_min"]:.5f}, '
                      f'Riemannian)')
    ax3.axhline(final['t'], color='#2ECC71', linewidth=1.5,
                linestyle=':',
                label=f'Final certified bound t ({final["t"]:.5f})')
    ax3.set_xlabel('Number of active couplings', fontsize=10)
    ax3.set_ylabel('Certified bound (best-case completion)', fontsize=10)
    ax3.set_title(
        f'Forward-greedy convergence: full flasher gore (m={m}, final k={k})',
        fontsize=11)
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(FIG_CONVERGE, dpi=150)
    print(f"Saved: {FIG_CONVERGE}")
    plt.close(fig3)


if __name__ == "__main__":
    main()
