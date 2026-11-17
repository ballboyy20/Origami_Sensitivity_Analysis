"""
birdsfoot_mixed_figures.py

Side-by-side figures: the best all-V-groove birds-foot design (sec 7.1,
[VVV V-V VVV --V], 0.009263) vs. the best mixed-type design found by B&B
(sec 7.15, [FFT V-V V-T V-V], 0.010358). Both use the 12 birds-foot sites,
an exact 18-row budget and unit row weights.

The results JSON stores the active sets but not their angles, so both are
re-evaluated with the same SetEvaluator settings as mixed_types_validation.py
(seed 31, per-set seeded rng) and lambda_min is checked against the saved
values.

  1. 3-D view of each design at its achieved angles, contacts marked by
     type (flat / V-groove / tetrahedral socket) with their contact
     normals, and the weakest constrained eigenmode overlaid.
  2. Hinge slices, one per spoke (rows: all-V, mixed), showing each site's
     type and its contact normals projected onto the mating face. A flat's
     single normal is the face normal, which points out of the page.

Saves: results/birdsfoot_mixed_vs_allv_{3d,hinges}.png and
results/birdsfoot_mixed_vs_allv_designs.json.

Run: python birdsfoot_mixed_figures.py
"""
import sys, os, json, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import (build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
                                   BIRDSFOOT_SPOKES, birdsfoot_spoke_name)
from interactive_optimizer import t as PANEL_T, _bfO as HUB_O
from coupling_types import TypedCandidates
from mixed_types_validation import make_evaluator, BF_CHAMPION_SITES
from visualization_rigid import (draw_panel_box, compute_lambda_min_eigenvector,
                                 draw_mode_arrows, _set_3d_axes,
                                 PANEL_COLORS, EIGVEC_COLOR, REMOVED_COLOR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
IN_JSON = os.path.join(RESULTS_DIR, 'mixed_types_validation_birdsfoot_results.json')
OUT_BASE = os.path.join(RESULTS_DIR, 'birdsfoot_mixed_vs_allv')
ALLV_GAP = 0.176              # sec 7.10: all-V certified gap at 12 sites

KIND_STYLE = {   # marker, contact color, normal color, label
    'flat':    dict(marker='s', color='#2C7FB8', ncolor='#2C7FB8', label='Flat (1 row)'),
    'vgroove': dict(marker='o', color='#F28E2B', ncolor='#D9480F', label='V-groove (2 rows)'),
    'tet':     dict(marker='^', color='#C2185B', ncolor='#C2185B', label='Tetrahedral socket (3 rows)'),
}


def build_design(active, theta):
    """A fresh TypedCandidates system with only `active` on, at `theta`."""
    tc = TypedCandidates(build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)))
    th = dict(zip(sorted(active), theta))
    for i, c in enumerate(tc.system.couplings):
        c.active = i in th
        if c.active:
            c.set_theta(th[i])
    return tc


def draw_3d_typed(ax, tc, title, normal_scale=0.12):
    system = tc.system
    for panel in system.panels:
        draw_panel_box(ax, panel, color=PANEL_COLORS[panel.id % len(PANEL_COLORS)])
    for s in range(tc.n_sites):        # unused sites
        if not any(system.couplings[i].active for i in np.flatnonzero(tc.site_of == s)):
            ax.scatter(*tc.base_system.couplings[s].point, color=REMOVED_COLOR,
                       marker='x', s=35, linewidth=1.3)
    for c in system.couplings:
        if not c.active:
            continue
        st = KIND_STYLE[c.kind]
        ax.scatter(*c.point, color=st['color'], marker=st['marker'], s=60,
                   edgecolor='black', linewidth=0.5, depthshade=False, zorder=5)
        for n in c.normals:
            ax.quiver(*c.point, *n, color=st['ncolor'], linewidth=1.3,
                      arrow_length_ratio=0.35, length=normal_scale, normalize=True)
    lam, vec = compute_lambda_min_eigenvector(system)
    if vec is not None:
        draw_mode_arrows(ax, system, vec, scale=0.3)
    _set_3d_axes(ax, system)
    ax.set_title(title, fontsize=10)
    return lam


def draw_hinge_typed(ax, tc, spoke, title, arrow_frac=0.18):
    """2-D slice of one spoke's mating face (x along the edge, y = depth);
    every contact normal projected onto that plane."""
    sites = range(3 * spoke, 3 * spoke + 3)
    base = tc.base_system.couplings[3 * spoke]
    fn = base.face_normal
    Z = np.array([0., 0., 1.])
    e_along = np.cross(Z, fn)
    e_along /= np.linalg.norm(e_along)
    far = BIRDSFOOT_SPOKES[spoke][2]
    origin = HUB_O
    if np.dot(far - origin, e_along) < 0:
        e_along = -e_along
    L = np.linalg.norm(far - origin)
    a = arrow_frac * L

    for s in sites:
        p = tc.base_system.couplings[s].point
        x, z = np.dot(p - origin, e_along), p[2]
        act = [tc.system.couplings[i] for i in np.flatnonzero(tc.site_of == s)
               if tc.system.couplings[i].active]
        if not act:
            ax.scatter([x], [z], color=REMOVED_COLOR, marker='x', s=45, linewidth=1.5)
            continue
        c = act[0]
        st = KIND_STYLE[c.kind]
        if c.kind == 'flat':
            # normal = face normal, out of the page: draw as a dotted circle
            ax.scatter([x], [z], s=170, facecolor='white', edgecolor=st['ncolor'],
                       linewidth=1.6, zorder=4)
            ax.scatter([x], [z], s=12, color=st['ncolor'], zorder=6)
            ax.annotate('F', (x, z), xytext=(7, 5), textcoords='offset points',
                        fontsize=8, color=st['ncolor'], fontweight='bold')
            continue
        for n in c.normals:
            dx, dz = np.dot(n, e_along), np.dot(n, Z)
            ax.annotate('', xy=(x + dx * a, z + dz * a), xytext=(x, z),
                        arrowprops=dict(arrowstyle='->', lw=1.6, color=st['ncolor']))
        ax.scatter([x], [z], color=st['color'], marker=st['marker'], s=60,
                   edgecolor='black', linewidth=0.5, zorder=5)
        ax.annotate(f"{'V' if c.kind == 'vgroove' else 'T'} {np.degrees(c.theta) % 180:.0f}°",
                    (x, z), xytext=(7, 5), textcoords='offset points', fontsize=8,
                    color=st['ncolor'], fontweight='bold')

    ax.plot([0, L], [0, 0], color='black', lw=1, alpha=0.35)
    ax.plot([0, L], [-PANEL_T, -PANEL_T], color='black', lw=1, alpha=0.35)
    m = 1.15 * a
    ax.set_xlim(-m, L + m)
    ax.set_ylim(-PANEL_T - m, m)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def legend_handles():
    h = [Line2D([0], [0], marker=st['marker'], color=st['color'], markeredgecolor='black',
                linestyle='', markersize=8, label=st['label']) for st in KIND_STYLE.values()]
    h += [Line2D([0], [0], marker='x', color=REMOVED_COLOR, linestyle='', label='Unused site'),
          Line2D([0], [0], color=EIGVEC_COLOR, lw=2, label='Weakest constrained mode')]
    return h


def main():
    with open(IN_JSON, encoding='utf-8') as f:
        saved = json.load(f)
    bnb = saved['bnb']

    tc0 = TypedCandidates(build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG)))
    ev = make_evaluator(tc0)
    designs = {
        'allv':  dict(active=tc0.from_site_kinds({s: 'vgroove' for s in BF_CHAMPION_SITES}),
                      saved=saved['champion']['lambda_min']),
        'mixed': dict(active=sorted(bnb['active']), saved=bnb['lambda_min']),
    }
    for k, d in designs.items():
        t0 = time.perf_counter()
        e = ev.evaluate(d['active'])
        d.update(lambda_min=float(e['lambda_min']), theta=np.asarray(e['theta'], float),
                 leaf_bound=float(e['t']), code=tc0.describe(d['active'], 3))
        print(f"{k:5s} [{d['code']}]: lambda_min {d['lambda_min']:.6f} (saved {d['saved']:.6f}, "
              f"rel diff {abs(d['lambda_min'] - d['saved']) / d['saved']:.1e}), "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    with open(OUT_BASE + '_designs.json', 'w', encoding='utf-8') as f:
        json.dump({k: dict(code=d['code'], active=d['active'], theta_rad=d['theta'].tolist(),
                           theta_deg=np.degrees(d['theta']).tolist(),
                           lambda_min=d['lambda_min'], leaf_bound=d['leaf_bound'])
                   for k, d in designs.items()}, f, indent=1)

    A, M = designs['allv'], designs['mixed']
    tcA, tcM = build_design(A['active'], A['theta']), build_design(M['active'], M['theta'])
    ratio = M['lambda_min'] / A['lambda_min']
    ub = bnb['certified_upper_bound']

    # 1. 3-D side by side
    fig = plt.figure(figsize=(15, 7.4))
    fig.suptitle(f"Birds-foot, 12 sites, 18 rows: best all-V-groove vs. best mixed-type design "
                 f"(mixed = {100 * ratio:.1f}% of all-V)", fontweight='bold', fontsize=12)
    lamA = draw_3d_typed(fig.add_subplot(1, 2, 1, projection='3d'), tcA,
                         f"V-grooves only  [{A['code']}]\n9 V-grooves; "
                         f"λ_min = {A['lambda_min']:.6f} (certified within {100 * ALLV_GAP:.1f}%)")
    lamM = draw_3d_typed(fig.add_subplot(1, 2, 2, projection='3d'), tcM,
                         f"All types allowed  [{M['code']}]\n2 flats + 5 V-grooves + 2 sockets; "
                         f"λ_min = {M['lambda_min']:.6f} (certified UB {ub:.6f}, gap "
                         f"{100 * (ub / M['lambda_min'] - 1):.1f}%)")
    for lam, d in ((lamA, A), (lamM, M)):
        assert abs(lam - d['lambda_min']) / d['lambda_min'] < 1e-3, (lam, d['lambda_min'])
    fig.legend(handles=legend_handles(), loc='lower center', ncol=5, fontsize=9, frameon=False)
    fig.text(0.5, 0.065, "Unit row weights (each flat, groove-wall and socket-wall contact "
             "equally stiff); socket walls at 45°.  Letters per spoke O-M, O-B, O-N, O-A: "
             "F flat, V V-groove, T socket, - unused.", ha='center', fontsize=8, color='0.35')
    plt.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUT_BASE + '_3d.png', dpi=150)
    plt.close(fig)

    # 2. hinge slices: rows = designs, columns = spokes
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.2))
    fig.suptitle("Birds-foot coupling arrangement per spoke (normals projected onto the "
                 "mating face; x = distance from hub O, y = depth)", fontweight='bold', fontsize=12)
    for row, (tc, d, name) in enumerate(((tcA, A, 'V-grooves only'),
                                         (tcM, M, 'All types'))):
        for sp in range(4):
            spoke = birdsfoot_spoke_name(tc.base_system.couplings[3 * sp])
            draw_hinge_typed(axes[row, sp], tc, sp,
                             f"{name} — spoke {spoke}  [{d['code'].split()[sp]}]")
        axes[row, 0].set_ylabel(f"{name}\nλ_min = {d['lambda_min']:.6f}\n\ndepth", fontsize=9)
    h = legend_handles()[:4]
    h.insert(1, Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='#2C7FB8',
                       linestyle='', markersize=10, label='Flat normal (out of page)'))
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT_BASE + '_hinges.png', dpi=150)
    plt.close(fig)
    print("Saved: " + ", ".join(OUT_BASE + s for s in ('_3d.png', '_hinges.png', '_designs.json')))


if __name__ == "__main__":
    main()
