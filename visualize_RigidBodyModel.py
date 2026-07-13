import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem

# ── Geometry (mirrors test_RigidBodyModel.py) ─────────────────────────
t = 0.1

panel_A = RigidPanel(0, vertices=np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
]), thickness=t)

panel_B = RigidPanel(1, vertices=np.array([
    [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    [2.0, 1.0, 0.0], [1.0, 1.0, 0.0],
]), thickness=t)

PANEL_COLORS    = ['steelblue', 'coral']
COUPLING_COLORS = ['crimson', 'darkorange', 'purple', 'darkgreen']
EV_TRANS_COLORS = ['gold', 'limegreen', 'deepskyblue', 'magenta']
EV_ROT_COLORS   = ['goldenrod', 'seagreen', 'royalblue', 'darkmagenta']


def draw_panel(ax, panel, color, alpha=0.30):
    """Draw a panel as a 3D box using top vertices + thickness."""
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


def draw_p_vectors(ax, system):
    """Draw arrows from each panel centroid to each coupling contact point."""
    panel_colors = {p.id: PANEL_COLORS[i] for i, p in enumerate(system.panels)}

    for coupling in system.couplings:
        for panel in (coupling.panel_A, coupling.panel_B):
            c = panel.centroid
            r = coupling.point - c          # the p-vector used in get_interpolation_matrix
            color = panel_colors.get(panel.id, 'gray')
            ax.quiver(*c, *r, color=color, arrow_length_ratio=0.12,
                      linewidth=1.2, linestyle='dashed', alpha=0.8)


def draw_coupling(ax, coupling, idx, color='red'):
    p = coupling.point
    n = coupling.normal * 0.22
    ax.scatter(*p, color=color, s=90, zorder=5, marker='o')
    ax.quiver(*p, *n, color=color, arrow_length_ratio=0.35, linewidth=2.5,
              label=f'Coupling {idx} normal')


def draw_eigenvectors(ax, system, tol=1e-9, scale=0.28):
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
                tv = trans / t_norm * scale
                lbl = f'EV{i+1} trans (λ={lam:.3f})'
                ax.quiver(*c, *tv, color=t_color, arrow_length_ratio=0.35,
                          linewidth=2,
                          label=lbl if lbl not in legend_added else '_nolegend_')
                legend_added.add(lbl)

            r_norm = np.linalg.norm(rot)
            if r_norm > 1e-10:
                rv = rot / r_norm * scale
                origin = c + np.array([0.06, 0.06, 0.0])
                lbl = f'EV{i+1} rot (λ={lam:.3f})'
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


# ── Build test systems (mirror test_RigidBodyModel.py) ────────────────

# Test 2: One Z-contact
sys2 = CouplingSystem([panel_A, panel_B])
sys2.add_coupling(KinematicCoupling(panel_A, panel_B,
    point=np.array([1.0, 0.25, 0.0]), normal=np.array([0.0, 0.0, 1.0])))

# Test 3: Three Z-contacts in a triangle
sys3 = CouplingSystem([panel_A, panel_B])
for pt in [[1.0, 0.2, 0.0], [1.0, 0.8, 0.0], [0.5, 0.5, 0.0]]:
    sys3.add_coupling(KinematicCoupling(panel_A, panel_B,
        point=np.array(pt), normal=np.array([0.0, 0.0, 1.0])))

# Test 4: Three COLLINEAR Z-contacts (rank 2)
sys4 = CouplingSystem([panel_A, panel_B])
for pt in [[1.0, 0.2, 0.0], [1.0, 0.5, 0.0], [1.0, 0.8, 0.0]]:
    sys4.add_coupling(KinematicCoupling(panel_A, panel_B,
        point=np.array(pt), normal=np.array([0.0, 0.0, 1.0])))

# Bonus: realistic 3D mating-face contacts from test module-level geometry
sys_3d = CouplingSystem([panel_A, panel_B])
sys_3d.add_coupling(KinematicCoupling(panel_A, panel_B,
    point=np.array([1.0, 0.2,  0.0]), normal=np.array([0., 0., 1.])))
sys_3d.add_coupling(KinematicCoupling(panel_A, panel_B,
    point=np.array([1.0, 0.8,  0.0]), normal=np.array([0., 0., 1.])))
sys_3d.add_coupling(KinematicCoupling(panel_A, panel_B,
    point=np.array([1.0, 0.5, -t  ]), normal=np.array([0., 0., 1.])))

scenes = [
    (sys2,  'Test 2: One Z-contact\n(1 constraint)'),
    (sys3,  'Test 3: Triangle Z-contacts\n(3 independent constraints)'),
    (sys4,  'Test 4: Collinear Z-contacts\n(rank 2 — 2 constraints)'),
    (sys_3d,'Mating face: 2 top + 1 bottom Z-contacts\n'
            '(rank 2 — Z position irrelevant for Z-normal)'),
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

fig.suptitle('Rigid Body Panels — Kinematic Couplings & Non-zero Eigenvectors',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
