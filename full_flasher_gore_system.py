"""
full_flasher_gore_system.py

Panel/coupling system for the flasher gore WITH inter-gore couplings
represented (Full_Flasher_gore_reference.dxf) -- the corrected version of
flasher_gore_system.py, which the user identified had silently omitted
the couplings between adjacent gores (see KINEMATIC_COUPLING_SUMMARY.md
sec 5.4). This is still a TEST of forward greedy at larger scale, not a
final representation of the physical assembly -- one neighbor gore's
boundary panels are duplicated in so their shared creases can be modeled,
but this is one arbitrary boundary condition on what is properly a
periodic problem, not the true assembled λ_min (see roadmap item 7 in the
summary doc: full-assembly evaluation or cyclic-symmetry/Bloch-wave
reduction are the two rigorous paths).

The DXF's inter-gore lines (added via a mirrored block INSERT) don't land
on bit-identical coordinates with the original gore's vertices -- off by
up to ~0.45 units against a geometry scale where the next-nearest
genuinely distinct vertex pair is ~80 units away -- so this load uses
dxf_to_panels's snap_tol to merge them (see that module's docstring).

8 panels, 9 interior creases -> m=27 candidate KinematicCouplings.
Panel 5 remains a leaf (single crease to panel 0) -- panel 2, previously
also a leaf in the single-gore version, now has a second crease (to the
new panel 7) since that's exactly the connection that was missing before.
"""
import os
import numpy as np

from dxf_to_panels import load_gore_geometry
from gore_system_common import (
    spokes_from_creases, spoke_names, spoke_name_of, default_start_thetas_deg,
    build_gore_system as _build_gore_system, mandatory_leaf_couplings,
)

DXF_PATH = os.path.join(os.path.dirname(__file__), 'Full_Flasher_gore_reference.dxf')
SNAP_TOL = 1.0  # see module docstring; safe margin (~80x) below the smallest genuine edge

FULL_GORE_PANELS, FULL_GORE_CREASES, FULL_GORE_THICKNESS = load_gore_geometry(
    DXF_PATH, snap_tol=SNAP_TOL)
FULL_GORE_SPOKES = spokes_from_creases(FULL_GORE_PANELS, FULL_GORE_CREASES)
FULL_GORE_SPOKE_NAMES = spoke_names(FULL_GORE_CREASES)
FULL_GORE_SPOKE_ORDER = [FULL_GORE_SPOKE_NAMES[frozenset({c['panel_a'], c['panel_b']})]
                          for c in FULL_GORE_CREASES]
FULL_GORE_START_THETAS_DEG = default_start_thetas_deg(len(FULL_GORE_SPOKES))
FULL_GORE_MANDATORY_COUPLINGS = mandatory_leaf_couplings(FULL_GORE_PANELS, FULL_GORE_CREASES)


def full_gore_spoke_name(coupling):
    return spoke_name_of(coupling, FULL_GORE_SPOKE_NAMES)


def build_full_gore_system(theta_degs, length_scale=1.0):
    return _build_gore_system(FULL_GORE_PANELS, FULL_GORE_SPOKES, theta_degs,
                               FULL_GORE_THICKNESS, length_scale=length_scale)


if __name__ == "__main__":
    system = build_full_gore_system(np.array(FULL_GORE_START_THETAS_DEG))
    print(f"{len(FULL_GORE_PANELS)} panels, {len(system.couplings)} couplings "
          f"(m={len(system.couplings)}), thickness={FULL_GORE_THICKNESS:.4f}")
    print(f"target_rank = total_dofs - 6 = {system.total_dofs - 6}")
    labels = [full_gore_spoke_name(c) for c in system.couplings]
    print("spoke per coupling:", labels)
    print("mandatory (leaf-panel) coupling indices:", FULL_GORE_MANDATORY_COUPLINGS)
    lam = system.get_rigidity_eigenvalue(length_scale=1.0)
    print(f"lambda_min at full m={len(system.couplings)} active "
          f"(sanity check, not yet optimized): {lam:.6f}")
