import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem

# ── Geometry (mirrors test_RigidBodyModel.py) ─────────────────────────
t = 0.1

panel_A = RigidPanel(0, vertices=np.array([
    [0., 0., 0.], [1., 0., 0.],
    [1., 1., 0.], [0., 1., 0.]]), thickness=t)

panel_B = RigidPanel(1, vertices=np.array([
    [1., 0., 0.], [2., 0., 0.],
    [2., 1., 0.], [1., 1., 0.]]), thickness=t)

face_normal = np.array([1., 0., 0.])

p1 = np.array([1., 0.2,  0.0])
p2 = np.array([1., 0.8,  0.0])
p3 = np.array([1., 0.5, -t  ])

PANEL_COLORS    = ['steelblue', 'coral']
COUPLING_COLORS = ['crimson', 'darkorange', 'purple']
EV_TRANS_COLORS = ['gold', 'limegreen', 'deepskyblue', 'magenta', 'orange', 'cyan']
EV_ROT_COLORS   = ['goldenrod', 'seagreen', 'royalblue', 'darkmagenta', 'sienna', 'teal']


def draw_panel(ax, panel, color, alpha=0.30):
    top = panel.vertices
    bot = top.copy()
    bot[:, 2] -= panel.thickness
    n = len(top)

    faces = [top.tolist(), bot.tolist()]
    for i in range(n):
        j = (i + 1) % n
        faces.append([top[i].tolist(), top[j].tolist(),
                      bot[j].tolist(),  bot[i].tolist()])

    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                            edgecolor='k', linewidth=1.0)
    ax.add_collection3d(poly)
    c = panel.centroid
    ax.text(c[0], c[1], c[2] + panel.thickness * 0.6 + 0.06,
            f'Panel {panel.id}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    # Centroid marker + coordinate label
    ax.scatter(*c, color='k', s=60, marker='*', zorder=6)
    ax.text(c[0] + 0.05, c[1] + 0.05, c[2] - 0.08,
            f'({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})',
            fontsize=6, color='k', va='top')


def draw_coupling(ax, coupling, idx, color='red'):
    """Draw contact point, face normal, and both V-groove normals n1/n2."""
    p = coupling.point
    ax.scatter(*p, color=color, s=90, zorder=5, marker='o')

    # Face normal — thin, semi-transparent
    fn = coupling.face_normal * 0.18
    ax.quiver(*p, *fn, color=color, arrow_length_ratio=0.3,
              linewidth=1.2, alpha=0.45,
              label=f'C{idx} face_normal' if idx == 0 else '_nolegend_')

    # n1 — solid, full weight
    n1 = coupling.n1 * 0.20
    ax.quiver(*p, *n1, color=color, arrow_length_ratio=0.35, linewidth=2.2,
              label=f'C{idx} n1' if idx == 0 else '_nolegend_')

    # n2 — same color, slightly shorter so both are visible
    n2 = coupling.n2 * 0.16
    ax.quiver(*p, *n2, color=color, arrow_length_ratio=0.35, linewidth=2.2,
              alpha=0.55,
              label=f'C{idx} n2' if idx == 0 else '_nolegend_')


def draw_p_vectors(ax, system):
    """Dashed arrows from each panel centroid to each coupling contact point."""
    id_to_color = {p.id: PANEL_COLORS[i] for i, p in enumerate(system.panels)}

    for coupling in system.couplings:
        for panel in (coupling.panel_A, coupling.panel_B):
            c = panel.centroid
            r = coupling.point - c
            ax.quiver(*c, *r, color=id_to_color.get(panel.id, 'gray'),
                      arrow_length_ratio=0.10, linewidth=1.2,
                      linestyle='dashed', alpha=0.75)


def draw_eigenvectors(ax, system, tol=1e-9, scale=0.28):
    """Arrows for non-zero eigenvectors of K = C^T C at each panel centroid."""
    C = system.build_constraint_matrix()
    if C.shape[0] == 0:
        return
    K = C.T @ C
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    nonzero_mask = eigenvalues > tol
    vals = eigenvalues[nonzero_mask]
    vecs = eigenvectors[:, nonzero_mask]

    legend_added = set()

    for i, (lam, vec) in enumerate(zip(vals, vecs.T)):
        t_color = EV_TRANS_COLORS[i % len(EV_TRANS_COLORS)]
        r_color = EV_ROT_COLORS[i % len(EV_ROT_COLORS)]

        for panel in system.panels:
            s = panel.dof_start
            trans = vec[s:s + 3]
            rot   = vec[s + 3:s + 6]
            c = panel.centroid

            t_norm = np.linalg.norm(trans)
            if t_norm > 1e-10:
                tv  = trans / t_norm * scale
                lbl = f'EV{i+1} trans (λ={lam:.3f})'
                ax.quiver(*c, *tv, color=t_color, arrow_length_ratio=0.35,
                          linewidth=2,
                          label=lbl if lbl not in legend_added else '_nolegend_')
                legend_added.add(lbl)

            r_norm = np.linalg.norm(rot)
            if r_norm > 1e-10:
                rv     = rot / r_norm * scale
                origin = c + np.array([0.06, 0.06, 0.0])
                lbl    = f'EV{i+1} rot (λ={lam:.3f})'
                ax.quiver(*origin, *rv, color=r_color, arrow_length_ratio=0.35,
                          linewidth=2,
                          label=lbl if lbl not in legend_added else '_nolegend_')
                legend_added.add(lbl)


def make_axes(ax, title):
    ax.set_xlabel('X', labelpad=4)
    ax.set_ylabel('Y', labelpad=4)
    ax.set_zlabel('Z', labelpad=4)
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_zlim(-0.35, 0.35)
    ax.set_title(title, pad=8, fontsize=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)


# ── Build scenes (mirror key test cases) ──────────────────────────────

# Scene 1 — Test 3a: 1 groove, θ=0 → 2 constraints
sys1 = CouplingSystem([panel_A, panel_B])
sys1.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.))

# Scene 2 — Test 3b: 3 PARALLEL grooves at p1/p2/p3, θ=0 → 5 constraints
# (grooves open toward face_normal but stay parallel, so 1 shared slide DOF
# remains free — same behavior as a real Kelvin/Maxwell V-groove)
sys2 = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    sys2.add_coupling(KinematicCoupling(panel_A, panel_B, p, face_normal, theta=0.))

# Scene 3 — Test 3c: 3 grooves at same point p1, θ = 0, π/4, π/2 → ≤3 constraints
sys3 = CouplingSystem([panel_A, panel_B])
for k in range(3):
    sys3.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal,
                                        theta=k * np.pi / 4))

# Scene 4 — Test 4a: grooves at p1/p2/p3 NOT parallel (groove 1 at θ=0,
# grooves 2 & 3 rotated by π/4 relative to it) → fully locks to 6 constraints
sys4 = CouplingSystem([panel_A, panel_B])
sys4.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.))
for p in [p2, p3]:
    sys4.add_coupling(KinematicCoupling(panel_A, panel_B, p, face_normal,
                                        theta=np.pi / 4))

scenes = [
    (sys1, 'Test 3a: 1 groove, θ=0\n(2 constraints)'),
    (sys2, 'Test 3b: 3 parallel grooves at p1/p2/p3, θ=0\n(5 constraints — 1 shared slide DOF free)'),
    (sys3, 'Test 3c: 3 grooves at p1, θ=0/π/4/π/2\n(≤3 constraints — same contact point)'),
    (sys4, 'Test 4a: groove 1 at θ=0, grooves 2/3 at θ=π/4\n(6 constraints — non-parallel grooves fully lock)'),
]

# ── Plot ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(22, 6),
                         subplot_kw={'projection': '3d'})

for ax, (system, title) in zip(axes, scenes):
    for panel, color in zip(system.panels, PANEL_COLORS):
        draw_panel(ax, panel, color)
    for i, coupling in enumerate(system.couplings):
        draw_coupling(ax, coupling, i, color=COUPLING_COLORS[i % len(COUPLING_COLORS)])
    draw_p_vectors(ax, system)
    draw_eigenvectors(ax, system)
    make_axes(ax, title)

fig.suptitle('Rigid Body Panels — V-Groove Kinematic Couplings & Non-zero Eigenvectors',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
