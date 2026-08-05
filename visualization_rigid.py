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
EIGVEC_COLOR  = '#8E44AD'


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

def compute_lambda_min_eigenvector(system):
    """
    Eigenvector of K = C^T C for the weakest genuinely-locked direction —
    same rank-aware convention as CouplingOptimizer._compute_lambda_min:
    skip the n_free = total_dofs - rank(C) zero/gauge eigenvalues and
    return the one at index n_free, rather than eigs.min() (which would
    just return a gauge mode, not a constrained direction).

    Returns
    -------
    lam : float — the eigenvalue (0. if there are no couplings at all)
    vec : (total_dofs,) ndarray or None — its eigenvector
    """
    C = system.build_constraint_matrix()
    if C.shape[0] == 0:
        return 0., None

    rank = np.linalg.matrix_rank(C)
    K = C.T @ C
    eigs, vecs = np.linalg.eigh(K)
    n_free = C.shape[1] - rank
    return float(eigs[n_free]), vecs[:, n_free]


def draw_mode_arrows(ax, system, mode_vec, scale=0.3, color=EIGVEC_COLOR):
    """
    Draw the rigid-body "pull" each panel's corners feel under a given
    generalized eigenvector of K = C^T C.

    For panel i with twist (t_i, omega_i) = mode_vec's 6-dof slice, each
    top-face corner p moves as t_i + omega_i x (p - centroid_i) — the same
    rigid-body kinematics as RigidPanel.get_interpolation_matrix. Arrow
    lengths are scaled together (not individually normalized) so corners
    far from the centroid, which feel more leverage from the rotational
    part, still show up longer.

    Parameters
    ----------
    ax       : Axes3D
    system   : CouplingSystem
    mode_vec : (total_dofs,) ndarray — e.g. from compute_lambda_min_eigenvector
    scale    : on-screen length of the largest arrow, in model units
    color    : arrow color
    """
    origins, vectors = [], []
    for panel in system.panels:
        i = panel.dof_start
        t_i     = mode_vec[i:i+3]
        omega_i = mode_vec[i+3:i+6]
        for p in panel.vertices:
            r = p - panel.centroid
            origins.append(p)
            vectors.append(t_i + np.cross(omega_i, r))

    origins = np.array(origins)
    vectors = np.array(vectors)
    max_norm = np.linalg.norm(vectors, axis=1).max()
    if max_norm < 1e-12:
        return
    vectors = vectors * (scale / max_norm)

    ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
              vectors[:, 0], vectors[:, 1], vectors[:, 2],
              color=color, linewidth=1.8, arrow_length_ratio=0.3)


def draw_3d_config(ax, system, title='',
                   show_normals=True, normal_scale=0.12,
                   show_lambda_min_mode=False, eigvec_scale=0.3):
    """
    Draw all panels and coupling contact points in 3-D.

    Parameters
    ----------
    ax                   : Axes3D
    system               : CouplingSystem
    title                : subplot title string
    show_normals         : draw n1 / n2 arrows at each contact
    normal_scale         : length of normal arrows in model units
    show_lambda_min_mode : overlay corner arrows for the weakest
                           constrained eigenmode of K = C^T C (see
                           compute_lambda_min_eigenvector / draw_mode_arrows)
    eigvec_scale         : on-screen length of the largest mode arrow
    """
    for panel in system.panels:
        color = PANEL_COLORS[panel.id % len(PANEL_COLORS)]
        draw_panel_box(ax, panel, color=color)

    for coupling in system.couplings:
        p = coupling.point
        ax.scatter(*p, color=CONTACT_COLOR, s=50,
                   zorder=5, depthshade=True)

        if show_normals:
            # normalize=True + length=normal_scale draws both arrows at the
            # same on-screen length regardless of any floating-point drift
            # in |n1|, |n2| — so they always render as unit vectors.
            ax.quiver(*p, *coupling.n1, color=N1_COLOR, linewidth=1.5,
                      arrow_length_ratio=0.35,
                      length=normal_scale, normalize=True)
            ax.quiver(*p, *coupling.n2, color=N2_COLOR, linewidth=1.5,
                      arrow_length_ratio=0.35,
                      length=normal_scale, normalize=True)

    lam_min = None
    if show_lambda_min_mode:
        lam_min, vec = compute_lambda_min_eigenvector(system)
        if vec is not None:
            draw_mode_arrows(ax, system, vec, scale=eigvec_scale)

    _set_3d_axes(ax, system)

    legend_handles = [
        Line2D([0], [0], color=CONTACT_COLOR, marker='o',
                   linestyle='', label='Contact point'),
        Line2D([0], [0], color=N1_COLOR,
                   linewidth=2, label='n1'),
        Line2D([0], [0], color=N2_COLOR,
                   linewidth=2, label='n2'),
    ]
    if lam_min is not None:
        legend_handles.append(
            Line2D([0], [0], color=EIGVEC_COLOR, linewidth=2,
                   label=f'λ_min mode (λ={lam_min:.4f})'))
    ax.legend(handles=legend_handles, fontsize=7, loc='upper left')

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
    # Match the box aspect to the real data ranges — otherwise matplotlib
    # stretches each axis to fill the same cube regardless of scale (Z here
    # spans far less than X/Y), which distorts angles in the projection and
    # makes orthogonal vectors (e.g. n1/n2) look skewed even though they
    # aren't.
    ax.set_box_aspect((xmax - xmin, ymax - ymin, 3 * t_max))
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


def print_constraint_matrix(system, n_panels=None, title=''):
    """
    Console printout of C (constraint rows x DOFs) and K = CᵀC, the
    rigidity matrix the optimizer's lambda_min is computed from.

    Parameters
    ----------
    system   : CouplingSystem
    n_panels : number of panels (for column labels); defaults to
               len(system.panels)
    title    : printed above the tables if given
    """
    n_panels = n_panels if n_panels is not None else len(system.panels)
    dof_labels = [f'P{i}_{d}'
                  for i in range(n_panels)
                  for d in ['ux', 'uy', 'uz', 'ωx', 'ωy', 'ωz']]

    C = system.build_constraint_matrix()

    if title:
        print(title)

    print(f"C  ({C.shape[0]} constraint rows x {C.shape[1]} DOFs)")
    header = " " * 6 + " ".join(f"{lbl:>8}" for lbl in dof_labels)
    print(header)
    for i, row in enumerate(C):
        print(f"c{i:<4} " + " ".join(f"{v:8.3f}" for v in row))

    K = (C.T @ C if C.shape[0] > 0
         else np.zeros((system.total_dofs, system.total_dofs)))
    rank = np.linalg.matrix_rank(C) if C.shape[0] > 0 else 0

    print(f"\nK = CᵀC  ({K.shape[0]} x {K.shape[1]}, rank(C) = {rank})")
    print(header)
    for i, row in enumerate(K):
        print(f"{dof_labels[i]:<6}" + " ".join(f"{v:8.3f}" for v in row))


# ══════════════════════════════════════════════════════════════════════
# Groove normals in the mating-face plane
# ══════════════════════════════════════════════════════════════════════

def draw_groove_normals_2d(ax, coupling, title=''):
    """
    Show n1, n2, and the bisector projected onto the V-groove's actual
    cross-section plane — spanned by face_normal (vertical axis) and w,
    the in-plane transverse direction (horizontal axis) — since the
    groove's slide axis u is, by construction, perpendicular to both
    n1 and n2 and carries no information about the V shape itself.

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

    fn = coupling.face_normal
    w  = coupling.w

    def project(vec):
        return (np.dot(vec, w), np.dot(vec, fn))

    arrow_kw = dict(xytext=(0, 0), textcoords='data',
                    xycoords='data')

    for vec, color, label in [
        (n1,       N1_COLOR,    'n1'),
        (n2,       N2_COLOR,    'n2'),
        (bisector * 0.65, BISECT_COLOR, 'bisector'),
    ]:
        px, py = project(vec)
        ax.annotate('', xy=(px, py),
                    arrowprops=dict(arrowstyle='->', lw=2, color=color),
                    **arrow_kw)
        ax.text(px * 1.2, py * 1.2, label,
                color=color, fontsize=9,
                fontweight='bold', ha='center', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.35)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.35)
    ax.set_xlabel('w (transverse, in-plane)', fontsize=9)
    ax.set_ylabel('face_normal', fontsize=9)
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

def figure_optimization_result(system_before, system_after, result):
    """
    Four-panel summary figure for a completed optimisation:
      top-left     — 3D config before optimisation
      top-right    — 3D config after optimisation
      bottom-left  — eigenvalue spectrum before vs after
      bottom-right — lambda_min convergence history
 
    Parameters
    ----------
    system_before : CouplingSystem  (thetas at theta=0 start)
    system_after  : CouplingSystem  (thetas at optimal values)
    result        : OptimizationResult
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Groove angle optimisation result', fontweight='bold',
                 fontsize=13)
 
    ax_3d_before = fig.add_subplot(2, 2, 1, projection='3d')
    ax_3d_after  = fig.add_subplot(2, 2, 2, projection='3d')
    ax_spec      = fig.add_subplot(2, 2, 3)
    ax_hist      = fig.add_subplot(2, 2, 4)
 
    # ── 3D configs ───────────────────────────────────────────────────
    draw_3d_config(ax_3d_before, system_before,
                   title=f'Before  (λ_min = {result.lambda_min_initial:.4f})',
                   show_lambda_min_mode=True)
    draw_3d_config(ax_3d_after,  system_after,
                   title=f'After   (λ_min = {result.lambda_min:.4f})',
                   show_lambda_min_mode=True)
 
    # ── Eigenvalue spectra overlaid ───────────────────────────────────
    def _eigs(system):
        C = system.build_constraint_matrix()
        K = C.T @ C if C.shape[0] > 0 else np.zeros(
            (system.total_dofs, system.total_dofs))
        return np.sort(np.linalg.eigvalsh(K))
 
    eigs_before = _eigs(system_before)
    eigs_after  = _eigs(system_after)
    x = np.arange(len(eigs_before))
    w = 0.35
    ax_spec.bar(x - w/2, eigs_before, w, label='Before',
                color='#BDC3C7', edgecolor='white')
    ax_spec.bar(x + w/2, eigs_after,  w, label='After',
                color=CONSTRAINED, edgecolor='white', alpha=0.85)
    ax_spec.set_xlabel('Mode index', fontsize=9)
    ax_spec.set_ylabel('Eigenvalue',  fontsize=9)
    ax_spec.set_title('Eigenvalue spectrum: before vs after', fontsize=9)
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, axis='y', alpha=0.25, linewidth=0.5)
 
    # ── Convergence history ───────────────────────────────────────────
    lam_hist = result.history['lambda_min']
    ax_hist.plot(lam_hist, color='steelblue', linewidth=1.2, alpha=0.6,
                 label='Each evaluation')
    # Running maximum — what the optimiser has found so far
    running_max = np.maximum.accumulate(lam_hist)
    ax_hist.plot(running_max, color='#E74C3C', linewidth=2,
                 label='Best so far')
    ax_hist.axhline(result.lambda_min, color='k', linewidth=1,
                    linestyle='--', alpha=0.5, label=f'Optimal ({result.lambda_min:.4f})')
    ax_hist.set_xlabel('Evaluation', fontsize=9)
    ax_hist.set_ylabel('λ_min',       fontsize=9)
    ax_hist.set_title(
        f'Convergence  ({result.n_evaluations} evaluations, '
        f'converged={result.converged})',
        fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.25, linewidth=0.5)
 
    plt.tight_layout()
    return fig


def figure_optimization_heatmaps(system_before, system_after, result,
                                  n_panels=2):
    """
    2-row heatmap figure comparing the constraint matrix C before and
    after groove-angle optimisation. Same layout as
    figure_section2_heatmaps, but titled with each system's lambda_min
    instead of a fixed coupling count.

    Parameters
    ----------
    system_before : CouplingSystem  (thetas at their starting values)
    system_after  : CouplingSystem  (thetas at optimal values)
    result        : OptimizationResult
    n_panels      : number of panels (for column labels)
    """
    C_before = system_before.build_constraint_matrix()
    C_after  = system_after.build_constraint_matrix()

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.suptitle('Constraint matrix C: before vs after optimisation',
                 fontweight='bold')

    for ax, (C_mat, title) in zip(axes, [
        (C_before, f'Before  (λ_min = {result.lambda_min_initial:.4f})  '
                   f'→  C shape {C_before.shape}'),
        (C_after,  f'After   (λ_min = {result.lambda_min:.4f})  '
                   f'→  C shape {C_after.shape}'),
    ]):
        draw_constraint_heatmap(ax, C_mat, n_panels=n_panels, title=title)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Eigenmode report — which relative DOFs are free / constrained
# ══════════════════════════════════════════════════════════════════════

DOF_LABELS_6 = ['Δx', 'Δy', 'Δz', 'Δωx', 'Δωy', 'Δωz']


def compute_eigenmode_table(system, tol=1e-9):
    """
    Decompose each eigenvalue of K = C^T C into the relative motion
    (panel A's loading minus panel B's) it represents along
    [x, y, z, ωx, ωy, ωz]. Only meaningful for exactly 2 panels.

    Parameters
    ----------
    system : CouplingSystem  (must have exactly 2 panels)
    tol    : float — eigenvalues <= tol are reported as free (0)

    Returns
    -------
    eigs      : (12,) ndarray, sorted ascending
    free_mask : (12,) bool ndarray, True where eigs <= tol
    deltas    : (12, 6) ndarray — row i is mode i's relative-DOF loading
    """
    if len(system.panels) != 2:
        raise ValueError(
            "compute_eigenmode_table requires exactly 2 panels "
            f"(got {len(system.panels)}) — the relative-DOF framing "
            "(panel A minus panel B) is only defined for a single pair.")

    C = system.build_constraint_matrix()
    K = (C.T @ C if C.shape[0] > 0
         else np.zeros((system.total_dofs, system.total_dofs)))

    eigs, vecs = np.linalg.eigh(K)
    free_mask  = eigs <= tol

    panel_A, panel_B = system.panels
    slice_A = slice(panel_A.dof_start, panel_A.dof_start + 6)
    slice_B = slice(panel_B.dof_start, panel_B.dof_start + 6)
    deltas  = (vecs[slice_A, :] - vecs[slice_B, :]).T   # (12, 6)

    return eigs, free_mask, deltas


def print_eigenmode_table(system, tol=1e-9, title=''):
    """
    Console table: one row per eigenmode (ascending eigenvalue), with
    its FREE/CONSTRAINED status and relative-DOF loading.
    """
    eigs, free_mask, deltas = compute_eigenmode_table(system, tol=tol)

    if title:
        print(title)

    header = (f"{'Mode':>4} {'lambda':>12} {'Status':>12}  " +
              " ".join(f"{lbl:>8}" for lbl in DOF_LABELS_6))
    print(header)
    print('-' * len(header))

    for i, (lam, free, row) in enumerate(zip(eigs, free_mask, deltas)):
        status = 'FREE' if free else 'CONSTRAINED'
        vals   = " ".join(f"{v:8.3f}" for v in row)
        print(f"{i:>4} {lam:12.6f} {status:>12}  {vals}")


def draw_eigenmode_heatmap(ax, system, tol=1e-9, title=''):
    """
    Heatmap version of compute_eigenmode_table: rows = modes (sorted
    ascending, y-labels colored by FREE/CONSTRAINED status), columns =
    the 6 relative DOFs, color = loading (sign + magnitude).
    """
    eigs, free_mask, deltas = compute_eigenmode_table(system, tol=tol)

    vmax = np.abs(deltas).max() or 1.0
    im   = ax.imshow(deltas, cmap='RdBu', aspect='auto',
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(DOF_LABELS_6)))
    ax.set_xticklabels(DOF_LABELS_6, fontsize=8)
    ax.set_yticks(range(len(eigs)))
    ax.set_yticklabels([f'{i}: λ={lam:.4f}' for i, lam in enumerate(eigs)],
                       fontsize=7)
    for tick, free in zip(ax.get_yticklabels(), free_mask):
        tick.set_color(FREE_MODE if free else CONSTRAINED)

    ax.set_xlabel('Relative DOF (panel A − panel B)', fontsize=8)
    ax.set_ylabel('Mode',                              fontsize=8)
    ax.set_title(title, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def figure_eigenmode_report(system, title='Eigenmode report', tol=1e-9):
    """Single-system eigenmode heatmap. Returns the Figure."""
    fig, ax = plt.subplots(figsize=(7, 8))
    draw_eigenmode_heatmap(ax, system, tol=tol, title=title)
    plt.tight_layout()
    return fig


def figure_optimization_eigenmodes(system_before, system_after, result,
                                    tol=1e-9):
    """
    Side-by-side eigenmode heatmaps: before vs after optimisation.
    Mirrors figure_optimization_heatmaps but for the relative-DOF
    breakdown instead of the raw constraint matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    fig.suptitle('Eigenmode report: before vs after optimisation',
                 fontweight='bold')

    draw_eigenmode_heatmap(
        axes[0], system_before, tol=tol,
        title=f'Before  (λ_min = {result.lambda_min_initial:.4f})')
    draw_eigenmode_heatmap(
        axes[1], system_after, tol=tol,
        title=f'After   (λ_min = {result.lambda_min:.4f})')

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

    # ── Left: λ_min vs δ (relative rotation of grooves 2 & 3 vs groove 1) ──
    deltas_fine = np.linspace(0, np.pi / 2, 60)
    lmins_fine  = []
    for delta in deltas_fine:
        sys = CouplingSystem([panel_A, panel_B])
        sys.add_coupling(KinematicCoupling(panel_A, panel_B, p1,
                                           face_normal, theta=0.))
        sys.add_coupling(KinematicCoupling(panel_A, panel_B, p2,
                                           face_normal, theta=delta))
        sys.add_coupling(KinematicCoupling(panel_A, panel_B, p3,
                                           face_normal, theta=delta))
        nz, _ = count_eigenvalues(sys)
        lmins_fine.append(min(nz) if nz else 0.)

    sample_deltas = np.linspace(0, np.pi / 2, len(lambda_mins_sampled))
    axes[0].plot(np.degrees(deltas_fine), lmins_fine,
                 color='steelblue', linewidth=2, label='λ_min (dense)')
    axes[0].scatter(np.degrees(sample_deltas), lambda_mins_sampled,
                    color='red', zorder=5, s=60,
                    label='Test 4a sample points')
    axes[0].set_xlabel('δ — relative groove rotation (degrees)', fontsize=9)
    axes[0].set_ylabel('λ_min',       fontsize=9)
    axes[0].set_title('Test 4a: λ_min vs relative groove rotation δ\n'
                       '(groove 1 fixed at θ=0)', fontsize=9)
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