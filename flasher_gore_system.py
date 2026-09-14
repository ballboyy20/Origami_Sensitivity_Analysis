"""
flasher_gore_system.py

Panel/coupling system for the single-gore flasher crease pattern
(Flasher_gore_reference_cleaned.dxf), built the same way birds-foot's
system is in interactive_optimizer.py -- see gore_system_common.py for the
shared build recipe (3 KinematicCouplings per interior crease, spread-out
non-degenerate starting angles).

Unlike birds-foot, panels/creases here are extracted from the DXF
automatically (dxf_to_panels.py) rather than hand-coded, since this
pattern has no single shared hub vertex every crease runs through (it's a
two-hub tree: panel 0 and panel 1 each border 3 other panels).

6 panels, 6 interior creases -> 18 candidate KinematicCouplings (m=18),
vs. birds-foot's m=12 -- the "larger scale" forward-greedy test called for
in KINEMATIC_COUPLING_SUMMARY.md step 5. This is one gore (wedge) of the
full flasher pattern with NO inter-gore couplings represented -- see
full_flasher_gore_system.py for the corrected version, and
KINEMATIC_COUPLING_SUMMARY.md sec 5.4 for why that omission matters.
"""
import os
import numpy as np

from dxf_to_panels import load_gore_geometry
from gore_system_common import (
    spokes_from_creases, spoke_names, spoke_name_of, default_start_thetas_deg,
    build_gore_system as _build_gore_system,
)

DXF_PATH = os.path.join(os.path.dirname(__file__), 'Flasher_gore_reference_cleaned.dxf')

GORE_PANELS, GORE_CREASES, GORE_THICKNESS = load_gore_geometry(DXF_PATH)
GORE_SPOKES = spokes_from_creases(GORE_PANELS, GORE_CREASES)
GORE_SPOKE_NAMES = spoke_names(GORE_CREASES)
GORE_SPOKE_ORDER = [GORE_SPOKE_NAMES[frozenset({c['panel_a'], c['panel_b']})]
                     for c in GORE_CREASES]
GORE_START_THETAS_DEG = default_start_thetas_deg(len(GORE_SPOKES))


def gore_spoke_name(coupling):
    """Spoke name ('P0-P1', etc.) a coupling belongs to, keyed by its two
    panel ids -- mirrors interactive_optimizer.birdsfoot_spoke_name."""
    return spoke_name_of(coupling, GORE_SPOKE_NAMES)


def build_gore_system(theta_degs, length_scale=1.0):
    return _build_gore_system(GORE_PANELS, GORE_SPOKES, theta_degs, GORE_THICKNESS,
                               length_scale=length_scale)


if __name__ == "__main__":
    system = build_gore_system(np.array(GORE_START_THETAS_DEG))
    print(f"{len(GORE_PANELS)} panels, {len(system.couplings)} couplings "
          f"(m={len(system.couplings)}), thickness={GORE_THICKNESS:.4f}")
    print(f"target_rank = total_dofs - 6 = {system.total_dofs - 6}")
    labels = [gore_spoke_name(c) for c in system.couplings]
    print("spoke per coupling:", labels)
    lam = system.get_rigidity_eigenvalue(length_scale=1.0)
    print(f"lambda_min at full m=18 active (sanity check, not yet optimized): {lam:.6f}")
