#!/usr/bin/env python3
"""Interactive rigidity/optimization explorer for panel systems joined by
sphere-in-groove contacts.

Mirrors Usevitch/origami_rigidity_tool_optimization.py's slider-driven
GUI (theta sliders, a length_scale slider, an Optimize button running a
global search), but built directly on RigidBodyModel/CouplingOptimizer so
it stays generalized (arbitrary panel/coupling geometry, rank-aware
lambda_min) rather than the professor's fixed 2-panel closed-form matrix.

Two selectable configurations (radio button):
  "Two panels"  — same fixed panel_A/panel_B/p1,p2,p3 as test_optimizer.py.
                  theta1/theta2/theta3 and length_scale are exposed as
                  sliders.
  "Birds foot"  — 4 panels (2 triangles + 2 trapezoids tiling a unit
                  square, meeting at the square's centroid), 12
                  KinematicCouplings (3 per shared spoke edge, mirroring
                  the two-panel recipe). 12 individual sliders would be
                  impractical, so those thetas are tracked internally and
                  only reachable via Reset / Optimize angles.

Run: python interactive_optimizer.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem
from coupling_optimizer import CouplingOptimizer
from visualization_rigid import (
    draw_3d_config,
    draw_eigenvalue_bar,
    draw_constraint_heatmap,
)


# ── Shared geometry (same as test_optimizer.py) ─────────────────────────
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

CONTACT_POINTS = (p1, p2, p3)

# Non-degenerate starting angles (degrees) — (0,0,0) would start the GUI
# at a rank-deficient, all-free configuration (parallel grooves).
ARBITRARY_START_THETAS_DEG = (17.0, 84.0, 151.0)


def build_system(theta_degs, length_scale=1.0):
    """Fresh CouplingSystem + 3 KinematicCouplings at the given angles."""
    system = CouplingSystem([panel_A, panel_B])
    for p, deg in zip(CONTACT_POINTS, theta_degs):
        system.add_coupling(KinematicCoupling(
            panel_A, panel_B, p, face_normal, theta=np.radians(deg)))
    return system


# ── Birds-foot geometry: 2 triangles + 2 trapezoids tiling a unit square ─
# Central vertex O at the square's centroid. Two collinear creases run to
# the midpoints of a pair of opposite sides (M, N); the other two creases
# run to the two corners of the side whose midpoint is M (A, B), mirrored
# across the O-M-N line. This produces 2 triangles (O-A-M, O-M-B) and 2
# trapezoids (O-B-C-N, O-N-D-A) meeting at O.
_bfA = np.array([0., 0., 0.])
_bfB = np.array([1., 0., 0.])
_bfC = np.array([1., 1., 0.])
_bfD = np.array([0., 1., 0.])
_bfO = np.array([0.5, 0.5, 0.])
_bfM = np.array([0.5, 0., 0.])   # midpoint of AB
_bfN = np.array([0.5, 1., 0.])   # midpoint of DC

birdsfoot_panel_0 = RigidPanel(0, vertices=np.array([_bfO, _bfA, _bfM]), thickness=t)
birdsfoot_panel_1 = RigidPanel(1, vertices=np.array([_bfO, _bfM, _bfB]), thickness=t)
birdsfoot_panel_2 = RigidPanel(2, vertices=np.array([_bfO, _bfB, _bfC, _bfN]), thickness=t)
birdsfoot_panel_3 = RigidPanel(3, vertices=np.array([_bfO, _bfN, _bfD, _bfA]), thickness=t)
BIRDSFOOT_PANELS = [birdsfoot_panel_0, birdsfoot_panel_1,
                    birdsfoot_panel_2, birdsfoot_panel_3]

# Each spoke is (panel_A, panel_B, far_point) — the shared edge runs from
# O to far_point. face_normal is derived below to point from panel_B
# toward panel_A, mirroring KinematicCoupling's convention.
BIRDSFOOT_SPOKES = [
    (birdsfoot_panel_1, birdsfoot_panel_0, _bfM),   # edge O-M: panels 0,1
    (birdsfoot_panel_2, birdsfoot_panel_1, _bfB),   # edge O-B: panels 1,2
    (birdsfoot_panel_3, birdsfoot_panel_2, _bfN),   # edge O-N: panels 2,3
    (birdsfoot_panel_0, birdsfoot_panel_3, _bfA),   # edge O-A: panels 3,0
]

# Non-degenerate starting angles (degrees), 3 per spoke in BIRDSFOOT_SPOKES
# order — chosen spread out like ARBITRARY_START_THETAS_DEG above so no
# spoke starts with 3 parallel (rank-deficient) grooves.
BIRDSFOOT_START_THETAS_DEG = (
    17.0,  84.0, 151.0,    # spoke 0 (O-M)
    23.0,  97.0, 142.0,    # spoke 1 (O-B)
    11.0,  76.0, 163.0,    # spoke 2 (O-N)
    35.0, 108.0, 155.0,    # spoke 3 (O-A)
)


def _spoke_face_normal(far_point, panel_A, panel_B):
    """In-plane normal perpendicular to the O->far_point spoke, pointing
    from panel_B toward panel_A (see KinematicCoupling's face_normal
    convention)."""
    edge_vec = far_point - _bfO
    normal = np.cross(edge_vec, np.array([0., 0., 1.]))
    normal /= np.linalg.norm(normal)
    if np.dot(normal, panel_A.centroid - panel_B.centroid) < 0:
        normal = -normal
    return normal


def _spoke_contact_points(far_point):
    """3 contact points along the O->far_point spoke, mirroring the
    two-panel recipe: 20%/80% along the top face, plus the midpoint
    through the panel thickness."""
    return [
        _bfO + 0.2 * (far_point - _bfO),
        _bfO + 0.8 * (far_point - _bfO),
        _bfO + 0.5 * (far_point - _bfO) + np.array([0., 0., -t]),
    ]


def build_birdsfoot_system(theta_degs, length_scale=1.0):
    """Fresh CouplingSystem + 12 KinematicCouplings (3 per spoke) at the
    given angles."""
    system = CouplingSystem(list(BIRDSFOOT_PANELS))
    idx = 0
    for panel_A, panel_B, far_point in BIRDSFOOT_SPOKES:
        normal = _spoke_face_normal(far_point, panel_A, panel_B)
        for p in _spoke_contact_points(far_point):
            system.add_coupling(KinematicCoupling(
                panel_A, panel_B, p, normal, theta=np.radians(theta_degs[idx])))
            idx += 1
    return system


class RigidityExplorer:
    """Matplotlib GUI for geometry, eigenvalue spectrum, and constraint matrix."""

    def __init__(self):
        self.initial_thetas_deg   = np.array(ARBITRARY_START_THETAS_DEG)
        self.initial_length_scale = 1.0
        self.mode                 = "two_panel"   # or "birdsfoot"
        self.birdsfoot_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG, dtype=float)
        self._matrix_cbar_ax = None   # tracked so repeated colorbars don't accumulate

        self.figure = plt.figure(figsize=(15.5, 9.5), constrained_layout=False)
        self.figure.canvas.manager.set_window_title(
            "Coupling optimizer explorer")
        grid = self.figure.add_gridspec(
            2, 2, left=0.055, right=0.98, top=0.94, bottom=0.27,
            height_ratios=(1.7, 1.0), width_ratios=(1.25, 1.0),
            hspace=0.32, wspace=0.22,
        )
        self.ax_3d    = self.figure.add_subplot(grid[0, 0], projection="3d")
        self.ax_eig   = self.figure.add_subplot(grid[0, 1])
        self.ax_matrix = self.figure.add_subplot(grid[1, :])

        slider_color = "#4c78a8"
        self.angle_sliders = [
            Slider(
                self.figure.add_axes([0.10, 0.195 - 0.045 * i, 0.34, 0.023]),
                rf"$\theta_{i + 1}$", 0.0, 180.0,
                valinit=float(self.initial_thetas_deg[i]), valstep=0.5,
                color=slider_color,
            )
            for i in range(3)
        ]
        self.length_scale_slider = Slider(
            self.figure.add_axes([0.10, 0.045, 0.34, 0.023]),
            "L", 0.1, 2.0, valinit=self.initial_length_scale, valstep=0.05,
            color=slider_color,
        )

        self.mode_radio = RadioButtons(
            self.figure.add_axes([0.57, 0.185, 0.30, 0.095]),
            ("Two panels", "Birds foot"),
        )
        self.reset_button = Button(
            self.figure.add_axes([0.57, 0.13, 0.12, 0.045]), "Reset"
        )
        self.optimize_button = Button(
            self.figure.add_axes([0.71, 0.13, 0.17, 0.045]), "Optimize angles"
        )
        self.status_text = self.figure.text(
            0.57, 0.075, "", ha="left", va="center", fontsize=9
        )

        for slider in self.angle_sliders:
            slider.on_changed(self.update)
        self.length_scale_slider.on_changed(self.update)
        self.mode_radio.on_clicked(self.set_mode)
        self.reset_button.on_clicked(self.reset)
        self.optimize_button.on_clicked(self.run_optimizer)
        self._sync_slider_visibility()
        self.update()

    def _build_current_system(self):
        length_scale = self.length_scale_slider.val
        if self.mode == "two_panel":
            theta_degs = np.array([s.val for s in self.angle_sliders])
            system = build_system(theta_degs, length_scale)
        else:
            system = build_birdsfoot_system(self.birdsfoot_thetas_deg, length_scale)
        return system, length_scale

    def _sync_slider_visibility(self):
        # 12 individual sliders for birds-foot would be impractical, so
        # the theta sliders only apply in two-panel mode; birds-foot
        # thetas are only reachable via Reset / Optimize angles.
        visible = (self.mode == "two_panel")
        for slider in self.angle_sliders:
            slider.ax.set_visible(visible)

    def set_mode(self, label):
        self.mode = "two_panel" if label == "Two panels" else "birdsfoot"
        self._sync_slider_visibility()
        self.status_text.set_text("")
        self.update()

    def reset(self, _event=None):
        if self.mode == "two_panel":
            for slider in self.angle_sliders:
                slider.reset()
        else:
            self.birdsfoot_thetas_deg = np.array(BIRDSFOOT_START_THETAS_DEG, dtype=float)
        self.length_scale_slider.reset()
        self.status_text.set_text("")
        self.update()

    def run_optimizer(self, _event=None):
        self.optimize_button.label.set_text("Optimizing...")
        self.status_text.set_text("Global search in progress")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        system, length_scale = self._build_current_system()
        opt = CouplingOptimizer(system, length_scale=length_scale)
        try:
            results = opt.optimize_theta(
                method="differential_evolution",
                n_solutions=2,
                maxiter=300,   # lower than the library default (1000) for GUI responsiveness
            )
        except RuntimeError as e:
            self.status_text.set_text(f"Optimizer: {e}")
            self.optimize_button.label.set_text("Optimize angles")
            self.figure.canvas.draw_idle()
            return

        best = results[0]   # best log_product among the tied-for-best-lambda_min set
        if self.mode == "two_panel":
            for slider, theta_rad in zip(self.angle_sliders, best.optimal_thetas):
                slider.set_val(float(np.degrees(theta_rad)))   # triggers update() via on_changed
        else:
            self.birdsfoot_thetas_deg = np.degrees(best.optimal_thetas)

        self.status_text.set_text(
            rf"$\lambda_{{min}}={best.lambda_min:.6f}$   "
            rf"log-vol$={best.log_product:.6f}$   "
            rf"({len(results)} candidate{'s' if len(results) != 1 else ''} found)"
        )
        self.optimize_button.label.set_text("Optimize angles")
        self.update()
        self.figure.canvas.draw_idle()

    def update(self, _value=None):
        system, length_scale = self._build_current_system()
        C = system.build_constraint_matrix(length_scale=length_scale)
        n_panels = len(system.panels)

        self.ax_3d.clear()
        draw_3d_config(self.ax_3d, system, title="Coupling configuration")

        self.ax_eig.clear()
        draw_eigenvalue_bar(
            self.ax_eig, system,
            title=f"Eigenvalue spectrum  (L={length_scale:.2f})",
            length_scale=length_scale,
        )

        self.ax_matrix.clear()
        if self._matrix_cbar_ax is not None:
            self._matrix_cbar_ax.remove()
            self._matrix_cbar_ax = None
        axes_before = set(self.figure.axes)
        draw_constraint_heatmap(
            self.ax_matrix, C, n_panels=n_panels,
            title=f"Constraint matrix C  (L={length_scale:.2f})",
        )
        new_axes = set(self.figure.axes) - axes_before
        if new_axes:
            self._matrix_cbar_ax = new_axes.pop()

        self.figure.canvas.draw_idle()


def main():
    explorer = RigidityExplorer()
    plt.show()


if __name__ == "__main__":
    main()
