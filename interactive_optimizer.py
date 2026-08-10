#!/usr/bin/env python3
"""Interactive rigidity/optimization explorer for two panels joined by
sphere-in-groove contacts.

Mirrors Usevitch/origami_rigidity_tool_optimization.py's slider-driven
GUI (theta sliders, a length_scale slider, an Optimize button running a
global search), but built directly on RigidBodyModel/CouplingOptimizer so
it stays generalized (arbitrary panel/coupling geometry, rank-aware
lambda_min) rather than the professor's fixed 2-panel closed-form matrix.
Geometry is intentionally fixed (same panel_A/panel_B/p1,p2,p3 as
test_optimizer.py) — only theta1/theta2/theta3 and length_scale are
exposed as sliders.

Run: python interactive_optimizer.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

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


class RigidityExplorer:
    """Matplotlib GUI for geometry, eigenvalue spectrum, and constraint matrix."""

    def __init__(self):
        self.initial_thetas_deg  = np.array(ARBITRARY_START_THETAS_DEG)
        self.initial_length_scale = 1.0
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
        self.reset_button.on_clicked(self.reset)
        self.optimize_button.on_clicked(self.run_optimizer)
        self.update()

    @property
    def parameters(self):
        theta_degs   = np.array([s.val for s in self.angle_sliders])
        length_scale = self.length_scale_slider.val
        return theta_degs, length_scale

    def reset(self, _event=None):
        for slider in self.angle_sliders:
            slider.reset()
        self.length_scale_slider.reset()
        self.status_text.set_text("")

    def run_optimizer(self, _event=None):
        theta_degs, length_scale = self.parameters
        self.optimize_button.label.set_text("Optimizing...")
        self.status_text.set_text("Global search in progress")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        system = build_system(theta_degs, length_scale)
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
        for slider, theta_rad in zip(self.angle_sliders, best.optimal_thetas):
            slider.set_val(float(np.degrees(theta_rad)))   # triggers update() via on_changed

        self.status_text.set_text(
            rf"$\lambda_{{min}}={best.lambda_min:.6f}$   "
            rf"log-vol$={best.log_product:.6f}$   "
            rf"({len(results)} candidate{'s' if len(results) != 1 else ''} found)"
        )
        self.optimize_button.label.set_text("Optimize angles")
        self.figure.canvas.draw_idle()

    def update(self, _value=None):
        theta_degs, length_scale = self.parameters
        system = build_system(theta_degs, length_scale)
        C = system.build_constraint_matrix(length_scale=length_scale)

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
            self.ax_matrix, C, n_panels=2,
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
