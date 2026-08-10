#!/usr/bin/env python3
"""Interactive rigidity explorer for two panels joined by sphere-in-groove contacts.

The left panel occupies x in [-0.5, 0], the right panel x in [0, 0.5],
both span y in [0, 1] and z in [0, h].  At each coincident contact point,
the right panel carries a sphere and the left panel carries an orthogonal
two-plane groove.

Dependencies: numpy, scipy, and matplotlib.

Examples
--------
    python origami_rigidity_tool.py
    python origami_rigidity_tool.py --theta1 0 --theta2 15 --theta3 -20
    python origami_rigidity_tool.py --optimize --solutions 4
    python origami_rigidity_tool.py --print-matrix --no-show
    python origami_rigidity_tool.py --save rigidity.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import differential_evolution


CONTACT_Y = np.array([0.1, 0.5, 0.9], dtype=float)
CONTACT_HEIGHT_FRACTIONS = np.array([0.25, 0.75, 0.25], dtype=float)
TWIST_LABELS = (
    r"$v_{Lx}$", r"$v_{Ly}$", r"$v_{Lz}$",
    r"$\omega_{Lx}$", r"$\omega_{Ly}$", r"$\omega_{Lz}$",
    r"$v_{Rx}$", r"$v_{Ry}$", r"$v_{Rz}$",
    r"$\omega_{Rx}$", r"$\omega_{Ry}$", r"$\omega_{Rz}$",
)
ROW_LABELS = ("1 +", "1 -", "2 +", "2 -", "3 +", "3 -")


def contact_points(height: float) -> np.ndarray:
    """Return the three global contact positions as rows."""
    return np.column_stack(
        (np.zeros(3), CONTACT_Y, CONTACT_HEIGHT_FRACTIONS * height)
    )


def groove_frame(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return groove axis and the two orthogonal groove-plane normals.

    theta is in radians.  At theta=0 the groove axis is +z.  Positive theta
    follows the right-hand convention for a rotation about +x.
    """
    s, c = np.sin(theta), np.cos(theta)
    groove_axis = np.array([0.0, -s, c])
    transverse = np.array([0.0, c, s])
    outward = np.array([1.0, 0.0, 0.0])
    normal_plus = (outward + transverse) / np.sqrt(2.0)
    normal_minus = (outward - transverse) / np.sqrt(2.0)
    return groove_axis, normal_plus, normal_minus


def rigidity_matrix(height: float, theta_degrees: np.ndarray) -> np.ndarray:
    """Construct the 6 x 12 bilateral rigidity matrix."""
    theta_radians = np.deg2rad(np.asarray(theta_degrees, dtype=float))
    rows: list[np.ndarray] = []

    for y, k, theta in zip(CONTACT_Y, CONTACT_HEIGHT_FRACTIONS, theta_radians):
        a = y - 0.5
        d = (k - 0.5) * height
        s, c = np.sin(theta), np.cos(theta)

        for sigma in (1.0, -1.0):
            row = np.array(
                [
                    -1.0,
                    -sigma * c,
                    -sigma * s,
                    -sigma * (a * s - d * c),
                    -d + sigma * s / 4.0,
                    a - sigma * c / 4.0,
                    1.0,
                    sigma * c,
                    sigma * s,
                    sigma * (a * s - d * c),
                    d + sigma * s / 4.0,
                    -a - sigma * c / 4.0,
                ],
                dtype=float,
            )
            rows.append(row / np.sqrt(2.0))

    return np.vstack(rows)


def scaled_rigidity_matrix(
    height: float, theta_degrees: np.ndarray, length_scale: float
) -> np.ndarray:
    """Return R for the dimensionless twist [v, L*omega] on each panel."""
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    matrix = rigidity_matrix(height, theta_degrees).copy()
    matrix[:, 3:6] /= length_scale
    matrix[:, 9:12] /= length_scale
    return matrix


def rigidity_metrics(
    height: float, theta_degrees: np.ndarray, length_scale: float
) -> tuple[np.ndarray, float]:
    """Return descending singular values and the log product of singular values."""
    values = np.linalg.svd(
        scaled_rigidity_matrix(height, theta_degrees, length_scale),
        compute_uv=False,
    )
    log_product = float(np.log(np.maximum(values, 1.0e-15)).sum())
    return values, log_product


def optimize_angles(
    height: float,
    length_scale: float = 1.0,
    angle_bound: float = 90.0,
    solutions: int = 3,
    seed: int = 1,
    maxiter: int = 500,
) -> tuple[float, list[dict[str, object]]]:
    """Lexicographically optimize minimum singular value, then log product.

    A global differential-evolution search first finds the best attainable
    minimum singular value.  Further independent searches maximize the log
    product while remaining within a small numerical tolerance of that primary
    optimum.  Distinct solutions are returned in descending tie-break score.
    """
    bounds = [(-angle_bound, angle_bound)] * 3

    def primary_objective(angles: np.ndarray) -> float:
        values, _ = rigidity_metrics(height, angles, length_scale)
        return -float(values[-1])

    primary = differential_evolution(
        primary_objective,
        bounds,
        seed=seed,
        popsize=16,
        maxiter=maxiter,
        tol=1.0e-10,
        polish=True,
        updating="immediate",
    )
    best_minimum = -float(primary.fun)
    feasibility_tolerance = max(1.0e-9, 1.0e-7 * best_minimum)
    threshold = best_minimum - feasibility_tolerance

    def secondary_objective(angles: np.ndarray) -> float:
        values, log_product = rigidity_metrics(height, angles, length_scale)
        if values[-1] < threshold:
            # Lexicographic barrier: an improved volume cannot compensate for
            # losing the primary worst-direction rigidity target.
            return 1.0e6 + 1.0e8 * float(threshold - values[-1])
        return -log_product

    candidates: list[dict[str, object]] = []
    number_of_runs = max(1, int(solutions))
    for run in range(number_of_runs):
        result = differential_evolution(
            secondary_objective,
            bounds,
            seed=seed + 101 * run,
            popsize=18,
            maxiter=maxiter,
            tol=1.0e-9,
            polish=True,
            updating="immediate",
        )
        angles = np.asarray(result.x, dtype=float)
        values, log_product = rigidity_metrics(height, angles, length_scale)
        if values[-1] < threshold:
            continue
        if any(np.linalg.norm(angles - np.asarray(item["angles"])) < 0.05
               for item in candidates):
            continue
        candidates.append(
            {
                "angles": angles,
                "singular_values": values,
                "minimum_singular_value": float(values[-1]),
                "log_product": log_product,
            }
        )

    candidates.sort(key=lambda item: float(item["log_product"]), reverse=True)
    return best_minimum, candidates


def panel_faces(x0: float, x1: float, height: float) -> list[list[tuple[float, ...]]]:
    """Vertices of the six faces of a rectangular panel."""
    v = np.array(
        [
            [x0, 0.0, 0.0], [x1, 0.0, 0.0],
            [x1, 1.0, 0.0], [x0, 1.0, 0.0],
            [x0, 0.0, height], [x1, 0.0, height],
            [x1, 1.0, height], [x0, 1.0, height],
        ]
    )
    indices = (
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    )
    return [[tuple(v[j]) for j in face] for face in indices]


class RigidityExplorer:
    """Matplotlib GUI for geometry, normals, matrix, and singular values."""

    def __init__(
        self, height: float, angles: np.ndarray, length_scale: float = 1.0
    ) -> None:
        self.initial_height = float(height)
        self.initial_angles = np.asarray(angles, dtype=float)
        self.initial_length_scale = float(length_scale)

        self.figure = plt.figure(figsize=(15.5, 9.5), constrained_layout=False)
        self.figure.canvas.manager.set_window_title("Origami rigidity explorer")
        grid = self.figure.add_gridspec(
            2, 2, left=0.055, right=0.98, top=0.94, bottom=0.27,
            height_ratios=(1.7, 1.0), width_ratios=(1.25, 1.0),
            hspace=0.32, wspace=0.22,
        )
        self.ax_3d = self.figure.add_subplot(grid[0, 0], projection="3d")
        self.ax_svd = self.figure.add_subplot(grid[0, 1])
        self.ax_matrix = self.figure.add_subplot(grid[1, :])

        slider_color = "#4c78a8"
        self.height_slider = Slider(
            self.figure.add_axes([0.10, 0.18, 0.34, 0.025]),
            "h", 0.05, 0.5, valinit=self.initial_height, valstep=0.01,
            color=slider_color,
        )
        self.length_slider = Slider(
            self.figure.add_axes([0.10, 0.135, 0.34, 0.025]),
            "L", 0.1, 2.0, valinit=self.initial_length_scale, valstep=0.05,
            color=slider_color,
        )
        self.angle_sliders = [
            Slider(
                self.figure.add_axes([0.57, 0.195 - 0.045 * i, 0.33, 0.023]),
                rf"$\theta_{i + 1}$", -90.0, 90.0,
                valinit=float(self.initial_angles[i]), valstep=0.1,
                color=slider_color,
            )
            for i in range(3)
        ]
        self.reset_button = Button(
            self.figure.add_axes([0.10, 0.075, 0.09, 0.038]), "Reset"
        )
        self.optimize_button = Button(
            self.figure.add_axes([0.205, 0.075, 0.13, 0.038]), "Optimize angles"
        )
        self.status_text = self.figure.text(
            0.35, 0.094, "", ha="left", va="center", fontsize=9
        )

        self.height_slider.on_changed(self.update)
        self.length_slider.on_changed(self.update)
        for slider in self.angle_sliders:
            slider.on_changed(self.update)
        self.reset_button.on_clicked(self.reset)
        self.optimize_button.on_clicked(self.run_optimizer)
        self.update()

    @property
    def parameters(self) -> tuple[float, float, np.ndarray]:
        return (
            self.height_slider.val,
            self.length_slider.val,
            np.array([s.val for s in self.angle_sliders]),
        )

    def reset(self, _event=None) -> None:
        self.height_slider.reset()
        self.length_slider.reset()
        for slider in self.angle_sliders:
            slider.reset()
        self.status_text.set_text("")

    def run_optimizer(self, _event=None) -> None:
        height, length_scale, _ = self.parameters
        self.optimize_button.label.set_text("Optimizing...")
        self.status_text.set_text("Global search in progress")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        best_minimum, candidates = optimize_angles(
            height,
            length_scale=length_scale,
            solutions=2,
            seed=1,
            maxiter=400,
        )
        if candidates:
            best = candidates[0]
            for slider, value in zip(self.angle_sliders, best["angles"]):
                slider.set_val(float(value))
            self.status_text.set_text(
                rf"$\sigma_{{min}}={float(best['minimum_singular_value']):.6f}$; "
                rf"$\sum\log\sigma={float(best['log_product']):.6f}$"
            )
        else:
            self.status_text.set_text(
                f"No secondary candidate found; primary optimum {best_minimum:.6f}"
            )
        self.optimize_button.label.set_text("Optimize angles")
        self.figure.canvas.draw_idle()

    def update(self, _value=None) -> None:
        height, length_scale, angles = self.parameters
        matrix = scaled_rigidity_matrix(height, angles, length_scale)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-8))

        self.draw_geometry(height, angles)
        self.draw_singular_values(singular_values, rank)
        self.draw_matrix(matrix, length_scale)
        self.figure.canvas.draw_idle()

    def draw_geometry(self, height: float, angles: np.ndarray) -> None:
        ax = self.ax_3d
        ax.clear()

        colors = (cm.Blues(0.48), cm.Oranges(0.48))
        for faces, color, label in (
            (panel_faces(-0.5, 0.0, height), colors[0], "left: grooves"),
            (panel_faces(0.0, 0.5, height), colors[1], "right: spheres"),
        ):
            collection = Poly3DCollection(
                faces, facecolor=color, edgecolor=(0.25, 0.25, 0.25, 0.45),
                linewidth=0.6, alpha=0.22,
            )
            collection.set_label(label)
            ax.add_collection3d(collection)

        points = contact_points(height)
        ax.plot(
            points[[0, 1, 2, 0], 0], points[[0, 1, 2, 0], 1],
            points[[0, 1, 2, 0], 2], color="0.2", linewidth=1.3,
            label="contact triangle",
        )

        axis_half_length = 0.10
        normal_length = 0.13
        plane_axis_half = 0.075
        plane_cross_half = 0.055

        for i, (point, angle_deg) in enumerate(zip(points, angles), start=1):
            groove_axis, normal_plus, normal_minus = groove_frame(np.deg2rad(angle_deg))
            ax.scatter(*point, s=48, color="black", depthshade=False)
            ax.text(*(point + np.array([0.015, 0.015, 0.015])), str(i), fontsize=9)

            ends = np.vstack(
                (point - axis_half_length * groove_axis,
                 point + axis_half_length * groove_axis)
            )
            ax.plot(*ends.T, color="#7a5195", linewidth=4.0,
                    solid_capstyle="round", label="groove axis" if i == 1 else None)

            for sigma, normal, color in (
                (1.0, normal_plus, "#d62728"),
                (-1.0, normal_minus, "#2ca02c"),
            ):
                ax.quiver(
                    *point, *(normal_length * normal), color=color,
                    linewidth=1.8, arrow_length_ratio=0.20,
                    label=(r"$n_+$" if sigma > 0 else r"$n_-$") if i == 1 else None,
                )

                # A second in-plane direction; together with the groove axis it
                # spans the corresponding groove plane.
                outward = np.array([1.0, 0.0, 0.0])
                transverse = np.array([0.0, np.cos(np.deg2rad(angle_deg)),
                                        np.sin(np.deg2rad(angle_deg))])
                plane_cross = (outward - sigma * transverse) / np.sqrt(2.0)
                corners = [
                    point - plane_axis_half * groove_axis - plane_cross_half * plane_cross,
                    point + plane_axis_half * groove_axis - plane_cross_half * plane_cross,
                    point + plane_axis_half * groove_axis + plane_cross_half * plane_cross,
                    point - plane_axis_half * groove_axis + plane_cross_half * plane_cross,
                ]
                patch = Poly3DCollection(
                    [corners], facecolor=color, edgecolor=color,
                    linewidth=0.7, alpha=0.12,
                )
                ax.add_collection3d(patch)

        ax.set_title("Reference geometry, groove planes, and normals")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-0.58, 0.58)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(0.0, max(0.52, height * 1.2))
        ax.set_box_aspect((1.16, 1.10, max(0.52, height * 1.2)))
        ax.view_init(elev=23, azim=-58)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    def draw_singular_values(self, values: np.ndarray, rank: int) -> None:
        ax = self.ax_svd
        ax.clear()
        indices = np.arange(1, 7)
        colors = ["#4c78a8" if value > 1.0e-8 else "#bab0ac" for value in values]
        bars = ax.bar(indices, values, color=colors, width=0.68)
        for bar, value in zip(bars, values):
            label = f"{value:.3f}" if value >= 1.0e-4 else f"{value:.1e}"
            ax.annotate(
                label, (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=8,
            )
        ax.set_xticks(indices, [rf"$\sigma_{i}$" for i in indices])
        ax.set_ylabel("singular value")
        ax.set_title(
            f"Rigidity spectrum: rank {rank}/6, relative mechanisms {6 - rank}"
        )
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0.0, max(1.0e-3, values[0] * 1.16))

    def draw_matrix(self, matrix: np.ndarray, length_scale: float) -> None:
        ax = self.ax_matrix
        ax.clear()
        limit = max(1.0e-9, float(np.max(np.abs(matrix))))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        del image
        ax.set_xticks(np.arange(12), TWIST_LABELS)
        ax.set_yticks(np.arange(6), ROW_LABELS)
        ax.set_title(
            rf"Scaled rigidity matrix $\widetilde R$ with $L={length_scale:.2f}$  "
            r"(rows: contact and $\pm$ plane)"
        )
        ax.tick_params(axis="x", labelsize=9)
        for row in range(6):
            for column in range(12):
                value = matrix[row, column]
                text_color = "white" if abs(value) > 0.58 * limit else "black"
                label = "0" if abs(value) < 5.0e-5 else f"{value:.3f}"
                ax.text(column, row, label, ha="center", va="center",
                        fontsize=7.4, color=text_color)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=0.25, help="panel z extent")
    parser.add_argument("--theta1", type=float, default=0.0, help="groove 1 angle in degrees")
    parser.add_argument("--theta2", type=float, default=0.0, help="groove 2 angle in degrees")
    parser.add_argument("--theta3", type=float, default=0.0, help="groove 3 angle in degrees")
    parser.add_argument(
        "--length-scale", type=float, default=1.0,
        help="characteristic length L used to scale angular velocities",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="globally optimize the groove angles before plotting",
    )
    parser.add_argument(
        "--solutions", type=int, default=3,
        help="number of independent secondary searches during optimization",
    )
    parser.add_argument("--seed", type=int, default=1, help="optimizer random seed")
    parser.add_argument(
        "--maxiter", type=int, default=500,
        help="maximum differential-evolution generations per search",
    )
    parser.add_argument("--save", type=Path, help="save the initial figure to this image path")
    parser.add_argument("--print-matrix", action="store_true", help="print R and its singular values")
    parser.add_argument("--no-show", action="store_true", help="do not open the GUI window")
    args = parser.parse_args()
    if not 0.05 <= args.height <= 0.5:
        parser.error("--height must be between 0.05 and 0.5")
    if args.length_scale <= 0.0:
        parser.error("--length-scale must be positive")
    if args.solutions < 1:
        parser.error("--solutions must be at least 1")
    if args.maxiter < 1:
        parser.error("--maxiter must be at least 1")
    return args


def main() -> None:
    args = parse_arguments()
    angles = np.array([args.theta1, args.theta2, args.theta3], dtype=float)

    if args.optimize:
        primary_optimum, candidates = optimize_angles(
            args.height,
            length_scale=args.length_scale,
            solutions=args.solutions,
            seed=args.seed,
            maxiter=args.maxiter,
        )
        print(f"primary optimum sigma_min = {primary_optimum:.9f}")
        if not candidates:
            raise RuntimeError("optimizer did not return a feasible secondary solution")
        print("distinct lexicographic optima found:")
        for index, candidate in enumerate(candidates, start=1):
            candidate_angles = np.asarray(candidate["angles"])
            candidate_values = np.asarray(candidate["singular_values"])
            print(
                f"  {index}: theta_deg={np.array2string(candidate_angles, precision=6)}, "
                f"sigma_min={float(candidate['minimum_singular_value']):.9f}, "
                f"sum_log_sigma={float(candidate['log_product']):.9f}"
            )
            print(f"     singular_values={np.array2string(candidate_values, precision=6)}")
        angles = np.asarray(candidates[0]["angles"], dtype=float)

    matrix = scaled_rigidity_matrix(args.height, angles, args.length_scale)

    if args.print_matrix:
        np.set_printoptions(precision=6, suppress=True, linewidth=180)
        print("scaled R =")
        print(matrix)
        print("singular values =")
        print(np.linalg.svd(matrix, compute_uv=False))
        print(f"rank = {np.linalg.matrix_rank(matrix, tol=1.0e-8)}")

    explorer = RigidityExplorer(args.height, angles, args.length_scale)
    if args.save:
        explorer.figure.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"saved {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(explorer.figure)


if __name__ == "__main__":
    main()
