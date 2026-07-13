"""
visualization_rigid.py
Visualization helpers for RigidBodyModel — panels, couplings,
constraint matrices, and eigenvalue spectra.

All public functions accept a matplotlib Axes (or Axes3D) so callers
can embed them freely into any figure layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch



# ── Color palette ──────────────────────────────────────────────────────
PANEL_COLORS  = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple',
                 'goldenrod',  'tomato']
CONTACT_COLOR = 'orange'
N1_COLOR      = 'green'
N2_COLOR      = 'royalblue'
BISECT_COLOR  = 'gray'
CONSTRAINED   = '#E74C3C'
FREE_MODE     = '#BDC3C7'


# ══════════════════════════════════════════════════════════════════════
# Panel drawing
# ══════════════════════════════════════════════════════════════════════

def draw_panel_box(ax, panel, color='steelblue', alpha=0.18):
    """
    Draw one rigid panel as a transparent 3-D box.

    Parameters
    ----------
    ax     : Axes3D
    panel  : RigidPanel
    color  : face and edge colour
    alpha  : face transparency
    """
    verts = panel.vertices
    t     = panel.thickness
    n     = len(verts)

    top    = list(verts)
    bottom = [v + np.array([0., 0., -t]) for v in verts]

    faces = [top, bottom]
    for i in range(n):
        faces.append([top[i], top[(i+1) % n],
                      bottom[(i+1) % n], bottom[i]])

    poly = Poly3DCollection(faces, alpha=alpha,
                            facecolor=color, edgecolor=color,
                            linewidth=0.5)
    ax.add_collection3d(poly)

    cx, cy, cz = panel.centroid
    ax.text(cx, cy, cz + t * 0.75,
            f'Panel {panel.id}',
            fontsize=8, ha='center',
            color=color, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════
# 3-D configuration view
# ══════════════════════════════════════════════════════════════════════

def draw_3d_config(ax, system, title='',
                   show_normals=True, normal_scale=0.12):
    """
    Draw all panels and coupling contact points in 3-D.

    Parameters
    ----------
    ax            : Axes3D
    system        : CouplingSystem
    title         : subplot title string
    show_normals  : draw n1 / n2 arrows at each contact
    normal_scale  : length of normal arrows in model units
    """
    for panel in system.panels:
        color = PANEL_COLORS[panel.id % len(PANEL_COLORS)]
        draw_panel_box(ax, panel, color=color)

    for coupling in system.couplings:
        p = coupling.point
        ax.scatter(*p, color=CONTACT_COLOR, s=50,
                   zorder=5, depthshade=True)

        if show_normals:
            ax.quiver(*p, *(coupling.n1 * normal_scale),
                      color=N1_COLOR, linewidth=1.5,
                      arrow_length_ratio=0.35)
            ax.quiver(*p, *(coupling.n2 * normal_scale),
                      color=N2_COLOR, linewidth=1.5,
                      arrow_length_ratio=0.35)

    _set_3d_axes(ax, system)

    ax.legend(handles=[
        Line2D([0], [0], color=CONTACT_COLOR, marker='o',
                   linestyle='', label='Contact point'),
        Line2D([0], [0], color=N1_COLOR,
                   linewidth=2, label='n1'),
        Line2D([0], [0], color=N2_COLOR,
                   linewidth=2, label='n2'),
    ], fontsize=7, loc='upper left')

    ax.set_title(title, fontsize=9)


def _set_3d_axes(ax, system):
    """Fit axis limits to the panel geometry and label axes."""
    all_verts = np.vstack([p.vertices for p in system.panels])
    xmin, ymin = all_verts[:, :2].min(axis=0) - 0.25
    xmax, ymax = all_verts[:, :2].max(axis=0) + 0.25
    t_max      = max(p.thickness for p in system.panels)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-t_max * 1.5, t_max * 1.5)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=22, azim=-50)


# ══════════════════════════════════════════════════════════════════════
# Eigenvalue spectrum
# ══════════════════════════════════════════════════════════════════════

def draw_eigenvalue_bar(ax, system, title='', tol=1e-9):
    """
    Bar chart of eigenvalues of K = CᵀC.
    Constrained modes (λ > tol) in red, free modes in gray.

    Parameters
    ----------
    ax     : Axes
    system : CouplingSystem
    title  : subplot title
    tol    : threshold separating zero from nonzero eigenvalues
    """
    C = system.build_constraint_matrix()
    K = (C.T @ C if C.shape[0] > 0
         else np.zeros((system.total_dofs, system.total_dofs)))

    eigs   = np.sort(np.linalg.eigvalsh(K))
    colors = [CONSTRAINED if e > tol else FREE_MODE for e in eigs]

    ax.bar(range(len(eigs)), eigs,
           color=colors, edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Mode index', fontsize=8)
    ax.set_ylabel('Eigenvalue',  fontsize=8)
    ax.set_title(title,          fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)

    n_constrained = int(np.sum(eigs > tol))
    n_free        = int(np.sum(eigs <= tol))
    ax.legend(handles=[
        Patch(color=CONSTRAINED, label=f'Constrained ({n_constrained})'),
        Patch(color=FREE_MODE,   label=f'Free ({n_free})'),
    ], fontsize=7)


# ══════════════════════════════════════════════════════════════════════
# Constraint matrix heatmap
# ══════════════════════════════════════════════════════════════════════

def draw_constraint_heatmap(ax, C, n_panels=2, title=''):
    """
    Colour-map of the constraint matrix C.
    Columns are labelled by panel DOF (ux, uy, uz, ωx, ωy, ωz).

    Parameters
    ----------
    ax       : Axes
    C        : (M, 6*N) ndarray
    n_panels : number of panels (for column labels)
    title    : subplot title
    """
    dof_labels = [f'P{i}_{d}'
                  for i in range(n_panels)
                  for d in ['ux', 'uy', 'uz', 'ωx', 'ωy', 'ωz']]

    vmax = np.abs(C).max() or 1.0
    im   = ax.imshow(C, cmap='RdBu', aspect='auto',
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(C.shape[1]))
    ax.set_xticklabels(dof_labels, rotation=45,
                       ha='right', fontsize=7)
    ax.set_yticks(range(C.shape[0]))
    ax.set_yticklabels([f'c{i}' for i in range(C.shape[0])],
                       fontsize=7)
    ax.set_xlabel('DOF',             fontsize=8)
    ax.set_ylabel('Constraint row',  fontsize=8)
    ax.set_title(title,              fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


# ══════════════════════════════════════════════════════════════════════
# Groove normals in the mating-face plane
# ══════════════════════════════════════════════════════════════════════

def draw_groove_normals_2d(ax, coupling, title=''):
    """
    Show n1, n2, and the bisector projected onto the mating face plane
    (Y and Z components only — the face-normal X component is zero
    for both groove normals by construction).

    Parameters
    ----------
    ax       : Axes
    coupling : KinematicCoupling
    title    : subplot title
    """
    n1, n2   = coupling.n1, coupling.n2
    bisector = n1 + n2
    if np.linalg.norm(bisector) > 1e-10:
        bisector = bisector / np.linalg.norm(bisector)

    arrow_kw = dict(xytext=(0, 0), textcoords='data',
                    xycoords='data')

    for vec, color, label in [
        (n1,       N1_COLOR,    'n1'),
        (n2,       N2_COLOR,    'n2'),
        (bisector * 0.65, BISECT_COLOR, 'bisector'),
    ]:
        ax.annotate('', xy=(vec[1], vec[2]),
                    arrowprops=dict(arrowstyle='->', lw=2, color=color),
                    **arrow_kw)
        ax.text(vec[1] * 1.2, vec[2] * 1.2, label,
                color=color, fontsize=9,
                fontweight='bold', ha='center', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.35)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.35)
    ax.set_xlabel('Y', fontsize=9)
    ax.set_ylabel('Z', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


# ══════════════════════════════════════════════════════════════════════
# Convenience: complete figures used by the test suite
# ══════════════════════════════════════════════════════════════════════

def figure_section1_groove_normals(panel_A, panel_B, p, face_normal,
                                   KinematicCoupling):
    """
    4-panel figure showing groove normals at θ = 0, 45°, 90°, 135°.
    Returns the Figure so the caller can plt.show() or save it.
    """
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(
        'Section 1: V-groove normals in mating face plane (YZ)',
        fontweight='bold')

    for ax, th in zip(axes, thetas):
        g = KinematicCoupling(panel_A, panel_B, p, face_normal, theta=th)
        draw_groove_normals_2d(
            ax, g,
            title=f'θ = {th:.2f} rad ({np.degrees(th):.0f}°)')

    plt.tight_layout()
    return fig


def figure_section2_heatmaps(C_1groove, C_3grooves, n_panels=2):
    """
    2-row heatmap figure: one row per constraint matrix.
    Returns the Figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.suptitle('Section 2: Constraint matrix C', fontweight='bold')

    for ax, (C_mat, title) in zip(axes, [
        (C_1groove,  'Test 2a: 1 groove  →  C shape (2×12)'),
        (C_3grooves, 'Test 2b: 3 grooves →  C shape (6×12)'),
    ]):
        draw_constraint_heatmap(ax, C_mat, n_panels=n_panels, title=title)

    plt.tight_layout()
    return fig


def figure_section3_eigenvalues(systems_and_titles):
    """
    Two-row figure: 3-D config (top) + eigenvalue bar (bottom)
    for each (system, title) pair supplied.

    Parameters
    ----------
    systems_and_titles : list of (CouplingSystem, str)
    """
    n = len(systems_and_titles)
    fig = plt.figure(figsize=(5 * n, 8))
    fig.suptitle('Section 3: Eigenvalue counts', fontweight='bold')

    for col, (sys, title) in enumerate(systems_and_titles):
        ax3d  = fig.add_subplot(2, n, col + 1, projection='3d')
        ax_bar = fig.add_subplot(2, n, col + 1 + n)
        draw_3d_config(ax3d, sys, title=title)
        draw_eigenvalue_bar(ax_bar, sys)

    plt.tight_layout()
    return fig


def figure_section4_robustness(panel_A, panel_B, p1, p2, p3,
                                face_normal, lambda_mins_sampled,
                                KinematicCoupling, CouplingSystem,
                                count_eigenvalues):
    """
    Two-panel figure:
      left  — λ_min vs θ (dense sweep + sampled test points)
      right — λ_min for 8 face-normal orientations

    Parameters
    ----------
    lambda_mins_sampled : list collected during Test 4a assertions
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Section 4: Physical robustness', fontweight='bold')

    # ── Left: λ_min vs θ ────────────────────────────────────────────
    thetas_fine = np.linspace(0, np.pi / 2, 60)
    lmins_fine  = []
    for th in thetas_fine:
        sys = CouplingSystem([panel_A, panel_B])
        for p in [p1, p2, p3]:
            sys.add_coupling(
                KinematicCoupling(panel_A, panel_B, p,
                                  face_normal, theta=th))
        nz, _ = count_eigenvalues(sys)
        lmins_fine.append(min(nz) if nz else 0.)

    sample_thetas = np.linspace(0, np.pi / 2, len(lambda_mins_sampled))
    axes[0].plot(np.degrees(thetas_fine), lmins_fine,
                 color='steelblue', linewidth=2, label='λ_min (dense)')
    axes[0].scatter(np.degrees(sample_thetas), lambda_mins_sampled,
                    color='red', zorder=5, s=60,
                    label='Test 4a sample points')
    axes[0].set_xlabel('θ (degrees)', fontsize=9)
    axes[0].set_ylabel('λ_min',       fontsize=9)
    axes[0].set_title('Test 4a: λ_min vs groove angle θ', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # ── Right: λ_min for different face orientations ─────────────────
    angles_deg  = [0, 30, 45, 60, 90, 120, 135, 150]
    lmin_angles = []
    for angle_deg in angles_deg:
        a  = np.radians(angle_deg)
        fn = np.array([np.cos(a), np.sin(a), 0.])
        sys = CouplingSystem([panel_A, panel_B])
        for p in [p1, p2, p3]:
            sys.add_coupling(
                KinematicCoupling(panel_A, panel_B, p, fn, theta=0.))
        nz, _ = count_eigenvalues(sys)
        lmin_angles.append(min(nz) if nz else 0.)

    axes[1].bar(range(len(angles_deg)), lmin_angles,
                color='steelblue', edgecolor='white')
    axes[1].set_xticks(range(len(angles_deg)))
    axes[1].set_xticklabels([f'{a}°' for a in angles_deg])
    axes[1].set_xlabel('Face normal angle from +X', fontsize=9)
    axes[1].set_ylabel('λ_min',                     fontsize=9)
    axes[1].set_title('Tests 4b/4c: λ_min for different face orientations',
                      fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig