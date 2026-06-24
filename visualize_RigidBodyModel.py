import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem

# ── Geometry ──────────────────────────────────────────────────────────
panel_A = RigidPanel(0, vertices=np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
]))
panel_B = RigidPanel(1, vertices=np.array([
    [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    [2.0, 1.0, 0.0], [1.0, 1.0, 0.0],
]))

PANEL_COLORS   = ['steelblue', 'coral']
COUPLING_COLORS = ['crimson', 'darkorange', 'purple']
EV_TRANS_COLORS = ['gold', 'limegreen', 'deepskyblue', 'magenta']
EV_ROT_COLORS   = ['goldenrod', 'seagreen', 'royalblue', 'darkmagenta']


def draw_panel(ax, panel, color, alpha=0.35):
    verts = [panel.vertices.tolist()]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                            edgecolor='k', linewidth=1.5)
    ax.add_collection3d(poly)
    c = panel.centroid
    ax.text(c[0], c[1], c[2] + 0.1, f'Panel {panel.id}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')


def draw_coupling(ax, coupling, idx, color='red'):
    p = coupling.point
    n = coupling.normal * 0.22
    ax.scatter(*p, color=color, s=90, zorder=5, marker='o')
    ax.quiver(*p, *n, color=color, arrow_length_ratio=0.35, linewidth=2.5,
              label=f'Coupling {idx} normal')


def draw_eigenvectors(ax, system, tol=1e-9, scale=0.28):
    """For each non-zero eigenvector of K = C^T C, draw translation and
    rotation components at each panel centroid."""
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
                t = trans / t_norm * scale
                lbl = f'EV{i+1} trans (λ={lam:.3f})'
                ax.quiver(*c, *t, color=t_color, arrow_length_ratio=0.35,
                          linewidth=2,
                          label=lbl if lbl not in legend_added else '_nolegend_')
                legend_added.add(lbl)

            r_norm = np.linalg.norm(rot)
            if r_norm > 1e-10:
                r = rot / r_norm * scale
                # Offset start slightly so rotation arrow doesn't overlap translation
                origin = c + np.array([0.06, 0.06, 0.0])
                lbl = f'EV{i+1} rot (λ={lam:.3f})'
                ax.quiver(*origin, *r, color=r_color, arrow_length_ratio=0.35,
                          linewidth=2,
                          label=lbl if lbl not in legend_added else '_nolegend_')
                legend_added.add(lbl)


def make_axes(ax, title):
    ax.set_xlabel('X', labelpad=4)
    ax.set_ylabel('Y', labelpad=4)
    ax.set_zlabel('Z', labelpad=4)
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_zlim(-0.45, 0.45)
    ax.set_title(title, pad=8, fontsize=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)


# ── Build test systems ────────────────────────────────────────────────

# Test 2: One Z-contact
sys2 = CouplingSystem([panel_A, panel_B])
sys2.add_coupling(KinematicCoupling(panel_A, panel_B,
    point=np.array([1.0, 0.25, 0.0]), normal=np.array([0.0, 0.0, 1.0])))

# Test 3: Three Z-contacts (triangle — full rank 3)
sys3 = CouplingSystem([panel_A, panel_B])
for pt in [[1.0, 0.2, 0.0], [1.0, 0.8, 0.0], [0.5, 0.5, 0.0]]:
    sys3.add_coupling(KinematicCoupling(panel_A, panel_B,
        point=np.array(pt), normal=np.array([0.0, 0.0, 1.0])))

# Test 4: Three COLLINEAR Z-contacts (rank 2)
sys4 = CouplingSystem([panel_A, panel_B])
for pt in [[1.0, 0.2, 0.0], [1.0, 0.5, 0.0], [1.0, 0.8, 0.0]]:
    sys4.add_coupling(KinematicCoupling(panel_A, panel_B,
        point=np.array(pt), normal=np.array([0.0, 0.0, 1.0])))

scenes = [
    (sys2, 'Test 2: One Z-contact\n(1 constraint)'),
    (sys3, 'Test 3: Triangle Z-contacts\n(3 independent constraints)'),
    (sys4, 'Test 4: Collinear Z-contacts\n(rank 2 — only 2 constraints)'),
]

# ── Plot ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={'projection': '3d'})

for ax, (system, title) in zip(axes, scenes):
    for panel, color in zip(system.panels, PANEL_COLORS):
        draw_panel(ax, panel, color)
    for i, coupling in enumerate(system.couplings):
        draw_coupling(ax, coupling, i, color=COUPLING_COLORS[i % len(COUPLING_COLORS)])
    draw_eigenvectors(ax, system)
    make_axes(ax, title)

fig.suptitle('Rigid Body Panels — Kinematic Couplings & Non-zero Eigenvectors',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
