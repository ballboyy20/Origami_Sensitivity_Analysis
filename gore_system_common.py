"""
gore_system_common.py

Shared helpers for building a CouplingSystem from DXF-extracted gore
geometry (dxf_to_panels.load_gore_geometry's panels/creases), factored out
of flasher_gore_system.py so full_flasher_gore_system.py doesn't duplicate
it. Mirrors the recipe interactive_optimizer.py uses for birds-foot:
3 KinematicCouplings per interior crease (20%/80% along the top face, 1 at
mid-thickness), non-degenerate spread-out starting angles.
"""
import numpy as np

from RigidBodyModel import KinematicCoupling, CouplingSystem


def spokes_from_creases(panels, creases):
    """(panel_A, panel_B, p_start, p_end) per crease, panel_A/B order
    arbitrary (see flasher_gore_system's original note: doesn't affect
    rank/eigenvalues)."""
    return [(panels[c['panel_a']], panels[c['panel_b']], c['p_start'], c['p_end'])
            for c in creases]


def spoke_names(creases):
    return {frozenset({c['panel_a'], c['panel_b']}): f"P{c['panel_a']}-P{c['panel_b']}"
            for c in creases}


def spoke_name_of(coupling, names):
    return names[frozenset({coupling.panel_A.id, coupling.panel_B.id})]


def default_start_thetas_deg(n_spokes):
    """3 non-degenerate starting angles (degrees) per spoke, deterministic
    (not tuned), spread to avoid 3 parallel grooves at any spoke and to
    avoid the 0/90/180 seams -- mirrors BIRDSFOOT_START_THETAS_DEG's
    hand-picked spread, generalized to an arbitrary spoke count."""
    return tuple(
        float(((17 + 41 * s + 63 * k) % 173) + 4)
        for s in range(n_spokes)
        for k in range(3)
    )


def _spoke_face_normal(p_start, p_end, panel_A, panel_B):
    edge_vec = p_end - p_start
    normal = np.cross(edge_vec, np.array([0., 0., 1.]))
    normal /= np.linalg.norm(normal)
    if np.dot(normal, panel_A.centroid - panel_B.centroid) < 0:
        normal = -normal
    return normal


def _spoke_contact_points(p_start, p_end, thickness):
    return [
        p_start + 0.2 * (p_end - p_start),
        p_start + 0.8 * (p_end - p_start),
        p_start + 0.5 * (p_end - p_start) + np.array([0., 0., -thickness]),
    ]


def build_gore_system(panels, spokes, theta_degs, thickness, length_scale=1.0):
    """Fresh CouplingSystem + 3*len(spokes) KinematicCouplings at the
    given angles."""
    system = CouplingSystem(list(panels))
    idx = 0
    for panel_A, panel_B, p_start, p_end in spokes:
        normal = _spoke_face_normal(p_start, p_end, panel_A, panel_B)
        for p in _spoke_contact_points(p_start, p_end, thickness):
            system.add_coupling(KinematicCoupling(
                panel_A, panel_B, p, normal, theta=np.radians(theta_degs[idx])))
            idx += 1
    return system


def mandatory_leaf_couplings(panels, creases):
    """Indices (into the 3-per-crease-ordered coupling list) of couplings
    on a crease that is some panel's ONLY connection to the rest of the
    system -- structurally mandatory in any full-rank topology (see
    KINEMATIC_COUPLING_SUMMARY.md sec 5.2). Order of `creases` must match
    the order spokes_from_creases/build_gore_system used to assign
    coupling indices (3 per crease, in crease order)."""
    from collections import Counter
    degree = Counter()
    for c in creases:
        degree[c['panel_a']] += 1
        degree[c['panel_b']] += 1

    forced = []
    for i, c in enumerate(creases):
        if degree[c['panel_a']] == 1 or degree[c['panel_b']] == 1:
            forced.extend([3 * i, 3 * i + 1, 3 * i + 2])
    return forced
