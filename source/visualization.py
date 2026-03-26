import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
from matplotlib import animation

"""
visualization.py
Plotting utilities for comparing Bloom pattern sensitivity results.
"""



from typing import Union, Dict

def calculate_stats(data: Union[np.ndarray, list], use_sample: bool = True) -> Dict[str, float]:
    """Calculates mean, median, standard deviation, and CV."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot calculate statistics on an empty array.")
        
    ddof = 1 if use_sample else 0
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_dev = np.std(arr, ddof=ddof)
    
    cv = np.nan if mean_val == 0 else std_dev / mean_val
        
    return {
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "cv": cv
    }

import matplotlib.pyplot as plt

def plot_fold_pattern(fold_data, title="Crease Pattern"):
    """
    Plots a .fold dictionary using Matplotlib.
    Mountain folds (M) are blue.
    Valley folds (V) are red.
    Boundaries (B) are black.
    Unassigned/other (U/F) are gray.
    """
    
    # Extract data from the dictionary
    vertices = fold_data.get("vertices_coords", [])
    edges = fold_data.get("edges_vertices", [])
    assignments = fold_data.get("edges_assignment", [])
    
    # Setup color and linewidth mapping — standard origami drafting convention:
    # M = Mountain: bold red dashed  (----)
    # V = Valley:   light blue dash-dot  (-.-.)
    # B = Boundary: thick black solid
    style_map = {
        "M": {"color": "#D62728", "linewidth": 2.5, "linestyle": "--"},
        "V": {"color": "#5BC8F5", "linewidth": 1.5, "linestyle": "-."},
        "B": {"color": "black",   "linewidth": 3.0, "linestyle": "-"},
        "U": {"color": "gray",    "linewidth": 1.0, "linestyle": ":"},
        "F": {"color": "gray",    "linewidth": 1.0, "linestyle": ":"}
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Iterate through edges and plot them
    for i, edge in enumerate(edges):
        v1_idx, v2_idx = edge
        x_coords = [vertices[v1_idx][0], vertices[v2_idx][0]]
        y_coords = [vertices[v1_idx][1], vertices[v2_idx][1]]
        
        # Default to 'U' (Unassigned) if assignment is missing or unknown
        assign = assignments[i] if i < len(assignments) else "U"
        style = style_map.get(assign, style_map["U"])
        
        ax.plot(x_coords, y_coords, 
                color=style["color"], 
                linewidth=style["linewidth"], 
                linestyle=style["linestyle"])

    # Format the plot for accurate geometric viewing
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Create a custom legend matching the style_map above
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#D62728', lw=2.5, linestyle='--', label='Mountain (M)'),
        Line2D([0], [0], color='#5BC8F5', lw=1.5, linestyle='-.', label='Valley (V)'),
        Line2D([0], [0], color='black',   lw=3.0, linestyle='-',  label='Boundary (B)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


class SensitivityVisualizationMixin:
    """
    Mixin class containing all plotting, animation, and print/report methods
    for SensitivityModel. Kept separate to avoid crowding SensitivityAnalysis.py.
    """

    def check_integration_rigidity(self, num_steps=50, step_size=0.02):
        """
        Integrates the folding path and tracks the change in hinge lengths
        (stretching error) for every individual hinge at each iteration,
        then plots the accumulated error to verify rigid kinematics.
        """
        print(f"\n--- Verifying Rigid Kinematics ({num_steps} steps) ---")
        target_fold_vector = self.build_target_fold_vector()

        # 1. Store initial coordinates and exact initial hinge lengths
        original_coords = [n.coordinates.copy() for n in self.nodes]

        initial_hinge_lengths = []
        for h in self.hinges:
            vec = h.node_k.coordinates - h.node_j.coordinates
            initial_hinge_lengths.append(np.linalg.norm(vec))

        # Initialize error tracking dictionary
        hinge_errors = {i: [] for i in range(len(self.hinges))}
        steps_taken = []

        # 2. Integration Loop
        for step in range(num_steps):
            v_dom = self.get_instantaneous_mechanism(target_fold_vector)

            if v_dom is None:
                print(f"Something went wrong... maybe kinematic lock-up reached at step {step}.")
                break

            steps_taken.append(step + 1)
            v_reshaped = v_dom.reshape(-1, 3)

            # Step the physical nodes forward
            for i, node in enumerate(self.nodes):
                node.coordinates = node.coordinates + (v_reshaped[i] * step_size)

            # --- Track Hinge Stretching for EVERY Hinge ---
            for i, h in enumerate(self.hinges):
                vec = h.node_k.coordinates - h.node_j.coordinates
                current_length = np.linalg.norm(vec)
                error = abs(current_length - initial_hinge_lengths[i])/initial_hinge_lengths[i]
                hinge_errors[i].append(error)

        # 3. Reset model back to pristine flat state
        for i, node in enumerate(self.nodes):
            node.coordinates = original_coords[i]

        print("Rigidity check complete. Generating error plot...")

        # 4. Plot the tracked errors
        plt.figure(figsize=(10, 6))
        for i in range(len(self.hinges)):
            assignment = self.hinges[i].fold_assignment
            plt.plot(steps_taken, hinge_errors[i], label=f'Hinge {i} ({assignment})', marker='.', linewidth=1.5)

        plt.title(f"Euler Integration Drift: Hinge Line Stretching (Step Size: {step_size})")
        plt.xlabel("Integration Step")
        plt.ylabel("Absolute Length Error (units)")

        # Using a scientific notation formatter for the Y-axis since the errors are usually tiny
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def animate_nonlinear_folding(self, num_steps=1000, step_size=0.01, interval=50):
        """
        Integrates the folding path by re-evaluating the SVD at every frame.
        Nodes follow true nonlinear arcs. No panel stretching occurs.
        """
        print(f"\nIntegrating folding path ({num_steps} steps)...")

        target_fold_vector = self.build_target_fold_vector()

        # Store original coordinates so we don't permanently ruin the model
        original_coords = [n.coordinates.copy() for n in self.nodes]

        trajectory = []
        trajectory.append(np.array(original_coords))

        # --- Integration Loop ---
        for step in range(num_steps):
            v_dom = self.get_instantaneous_mechanism(target_fold_vector)

            if v_dom is None:
                print(f"Kinematic lock-up reached at step {step}. Stopping integration.")
                break

            v_reshaped = v_dom.reshape(-1, 3)

            # Update the physical nodes
            for i, node in enumerate(self.nodes):
                node.coordinates = node.coordinates + (v_reshaped[i] * step_size)

            # Save the new state
            trajectory.append(np.array([n.coordinates.copy() for n in self.nodes]))

        # Reset model to original state
        for i, node in enumerate(self.nodes):
            node.coordinates = original_coords[i]

        print("Integration complete. Rendering animation...")

        # --- Setup Animation ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Nonlinear Rigid Folding (Iterative SVD)")
        ax.axis('off')

        # Use the final folded state to set the camera bounding box
        max_coords = trajectory[-1]
        all_coords = np.vstack((original_coords, max_coords))
        max_range = np.ptp(all_coords, axis=0).max() / 2.0
        mid = np.mean(original_coords, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Initialize lines
        bar_lines = [ax.plot([], [], [], color='black', alpha=0.3, linewidth=1)[0] for _ in self.bars]
        hinge_lines = [ax.plot([], [], [], color='blue' if h.fold_assignment == 'M' else 'red', linewidth=3)[0] for h in self.hinges]

        def update(frame):
            # Ping-pong loop calculation
            max_frame = len(trajectory) - 1
            cycle_length = max_frame * 2
            current_frame = frame % cycle_length
            if current_frame > max_frame:
                current_frame = cycle_length - current_frame # reverse direction

            current_coords = trajectory[current_frame]

            for i, bar in enumerate(self.bars):
                p1, p2 = current_coords[bar.nodes[0].id], current_coords[bar.nodes[1].id]
                bar_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                bar_lines[i].set_3d_properties([p1[2], p2[2]])

            for i, h in enumerate(self.hinges):
                p1, p2 = current_coords[h.node_j.id], current_coords[h.node_k.id]
                hinge_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                hinge_lines[i].set_3d_properties([p1[2], p2[2]])

            return bar_lines + hinge_lines

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory)*2, interval=interval, blit=False)
        plt.show()

    def plot_sensitivity_over_deployment(self, num_steps=100, step_size=0.01):
        """
        Integrates the folding path by re-evaluating the SVD at every frame using
        the core analyze_sensitivity() method. Tracks and plots the ABSOLUTE,
        non-normalized hinge sensitivities as they deploy.
        """
        print(f"\n--- Tracking Absolute Hinge Sensitivities over Deployment ({num_steps} steps) ---")

        # 1. Store original coordinates so we don't permanently deform the model
        original_coords = [n.coordinates.copy() for n in self.nodes]

        # Initialize data tracking dictionary
        hinge_sensitivities = {i: [] for i in range(len(self.hinges))}
        deployment_steps = []

        # 2. Integration Loop
        for step in range(num_steps):
            deployment_steps.append(step * step_size)

            # Call analyze_sensitivity silently
            current_sens = self.analyze_sensitivity(show_plot=None, silent=True)

            # Track the ABSOLUTE value for each hinge (no normalization)
            for i in range(len(self.hinges)):
                hinge_sensitivities[i].append(abs(current_sens[i]))

            if not hasattr(self, 'v_dominant') or self.v_dominant is None:
                print(f"Kinematic lock-up reached at step {step}. Stopping integration.")
                break

            # Step the physical nodes forward along the nonlinear arc
            v_reshaped = self.v_dominant.reshape(-1, 3)
            for i, node in enumerate(self.nodes):
                node.coordinates = node.coordinates + (v_reshaped[i] * step_size)

        # 3. Reset model back to the pristine flat state
        for i, node in enumerate(self.nodes):
            node.coordinates = original_coords[i]

        print("Integration complete. Generating absolute sensitivity drift plot...")

        # 4. Plot the tracked sensitivities
        plt.figure(figsize=(12, 7))
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.1, 0.9, len(self.hinges)))

        for i, h in enumerate(self.hinges):
            assignment = h.fold_assignment
            # Mountain (+) = Solid, Valley (-) = Dashed, Unassigned = Dotted
            # Retained for visual clarity even though all values are absolute
            l_style = '-' if assignment == 'M' else ('--' if assignment == 'V' else ':')

            plt.plot(deployment_steps, hinge_sensitivities[i],
                     label=f'H{i} ({assignment})',
                     color=colors[i],
                     linestyle=l_style,
                     linewidth=2,
                     alpha=0.8)

        plt.title("Absolute Hinge Sensitivities During Deployment", fontsize=14, fontweight='bold')
        plt.xlabel("Deployment Pseudo-Time (Steps × Step Size)", fontsize=12)
        plt.ylabel("Absolute Sensitivity (rad/unit)", fontsize=12)

        # Draw baseline for reference
        plt.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)

        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        # Push legend outside
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2 if len(self.hinges) > 15 else 1, fontsize=9)
        plt.tight_layout()
        plt.show()

        return hinge_sensitivities

    def report_singular_values(self, S_sv, best_r):
        """Prints the mechanism subspace singular values, highlighting the chosen mode."""
        print(f"\nMechanism subspace singular values (fold efficiency per unit displacement):")
        for r, sv in enumerate(S_sv):
            marker = f"  ← selected (best M/V alignment, rank {r})" if r == best_r else ""
            print(f"  σ_{r} = {sv:.6f}{marker}")

    def report_alignment(self, best_sensitivity, target_fold_vector):
        """Compares the computed sensitivity vector to the target fold vector derived from M/V assignments, and reports the quality of alignment."""
        # Report alignment quality
        norm_s = np.linalg.norm(best_sensitivity)
        norm_t = np.linalg.norm(target_fold_vector)
        if norm_s > 1e-12 and norm_t > 1e-12:
            cos_sim = np.dot(best_sensitivity, target_fold_vector) / (norm_s * norm_t)
            quality = 'excellent' if cos_sim > 0.99 else \
                      'good'      if cos_sim > 0.90 else \
                      'moderate'  if cos_sim > 0.50 else 'poor'
            print(f"M/V alignment score (cosine similarity with target): "
                  f"{cos_sim:.6f}  ({quality})")

    def print_system_matrices(self, dihedral_jacobian, constraint_matrix, singular_values, Vh,
                              sensitivity_vector, mechanism_indices=None,
                              Q=None, A=None, U_sv=None, S_sv=None, Vt_sv=None,
                              v_dominant=None, t=None, chosen_mode_idx=None):
        """
        Comprehensive diagnostic report of every matrix and intermediate result
        produced by analyze_sensitivity, printed in pipeline order.

        Sections
        --------
        [1] Constraint Matrix  C
        [2] Dihedral Jacobian  J
        [3] Singular value spectrum of C  (full, classified)
        [4] Complete null space of C      (RBM + Mechanism rows)
        [5] Mechanism null space basis  Q
        [6] Fold angle matrix  A = J @ Q^T
        [7] SVD of A  (Σ, U, Vt)
        [8] Dominant nodal displacement  v*
        [9] Final sensitivity vector + M/V validation
        """
        W = 110   # report width

        # ── Label setup ───────────────────────────────────────────────────────
        col_labels   = []
        for n in self.nodes:
            col_labels.extend([f"N{n.id}_x", f"N{n.id}_y", f"N{n.id}_z"])
        bar_labels   = [f"Bar {i}"   for i in range(len(self.bars))]
        hinge_labels = [f"Hinge {i}" for i in range(len(self.hinges))]

        n_nodes  = len(self.nodes)
        n_bars   = len(self.bars)
        n_hinges = len(self.hinges)
        n_dofs   = 3 * n_nodes
        n_sv     = len(singular_values)
        n_dof    = Vh.shape[0]
        mech_set = set(mechanism_indices) if mechanism_indices else set()
        k_mech   = len(mech_set)
        null_indices = [i for i in range(n_dof)
                        if (singular_values[i] if i < n_sv else 0.0) < 1e-9]

        # ── Formatting helpers ────────────────────────────────────────────────
        def header(title):
            print("\n" + "═" * W)
            print(f"  {title}")
            print("═" * W)

        def subheader(title):
            bar = "─" * max(0, W - len(title) - 5)
            print(f"\n  ┌─ {title} {bar}")

        def print_matrix(matrix, r_labels, c_labels):
            """Print a 2-D matrix with row and column labels."""
            if len(matrix) == 0:
                print("    (empty)")
                return
            lw = max(len(str(l)) for l in r_labels) + 1
            cw = 9
            # header row
            print("    " + f"{'':>{lw}} ║ " + " │ ".join(f"{c:>{cw}}" for c in c_labels))
            print("    " + "─" * lw + "═╬═" + "═╪═".join("═" * cw for _ in c_labels))
            for lbl, row in zip(r_labels, matrix):
                vals = " │ ".join(
                    f"{'  0.0   ':>{cw}}" if abs(v) < 1e-9 else f"{v:>{cw}.4f}"
                    for v in row
                )
                print(f"    {lbl:>{lw}} ║ {vals}")

        # ══════════════════════════════════════════════════════════════════════
        header("ORIGAMI SENSITIVITY ANALYSIS — FULL DIAGNOSTIC REPORT")
        print(f"  Problem size:  {n_nodes} nodes  │  {n_dofs} DOFs  │  "
              f"{n_bars} bars  │  {n_hinges} hinges")
        print(f"  Null space:    {len(null_indices)} modes (σ < 1e-9)  │  "
              f"{k_mech} mechanism mode(s) selected")

        # ── [1] Constraint Matrix C ───────────────────────────────────────────
        header(f"[1]  CONSTRAINT MATRIX  C     shape: {n_bars} × {n_dofs}")
        print("  One row per bar.  C[i] · v = 0  means bar i doesn't stretch under displacement v.")
        print()
        print_matrix(constraint_matrix, bar_labels, col_labels)

        # ── [2] Dihedral Jacobian J ───────────────────────────────────────────
        header(f"[2]  DIHEDRAL JACOBIAN  J     shape: {n_hinges} × {n_dofs}")
        print("  One row per hinge.  J[i] · v = change in dihedral angle at hinge i.")
        print()
        print_matrix(dihedral_jacobian, hinge_labels, col_labels)

        # ── [3] Singular Value Spectrum of C ──────────────────────────────────
        header("[3]  SINGULAR VALUE SPECTRUM  of  C     (sorted high → low)")
        print("  Rows with σ ≈ 0 span the null space — the only legal nodal movements.")
        print()
        print(f"  {'Idx':>5} │ {'σ':>16} │ {'‖J·v‖₁ fold mag':>18} │  Classification")
        print(f"  {'─'*5}─┼─{'─'*16}─┼─{'─'*18}─┼─{'─'*42}")
        for i in range(n_dof):
            sv     = singular_values[i] if i < n_sv else 0.0
            v      = Vh[i, :]
            fmag   = np.sum(np.abs(dihedral_jacobian @ v))
            if sv >= 1e-9:
                cls    = "Constrained (resisted by bars)"
                fstr   = "—"
                sv_str = f"{sv:>16.6e}"
            else:
                sv_str = f"{sv:>16.2e}" if sv > 0 else f"{'0  (exact)':>16}"
                fstr   = f"{fmag:.6f}"
                if fmag < 1e-5:
                    cls = "NULL — Rigid Body / Spurious z-mode"
                elif i in mech_set:
                    cls = "★ NULL — MECHANISM (selected)"
                else:
                    cls = "NULL — Mechanism (not selected)"
            print(f"  {i:>5} │ {sv_str} │ {fstr:>18} │  {cls}")

        # ── [4] Complete Null Space of C ──────────────────────────────────────
        header(f"[4]  COMPLETE NULL SPACE  of  C     ({len(null_indices)} vectors, σ < 1e-9)")
        print("  Every row satisfies  C · v = 0.  Rows marked MECH are the selected mechanisms.")
        print()
        if null_indices:
            null_labels = []
            for idx in null_indices:
                fmag = np.sum(np.abs(dihedral_jacobian @ Vh[idx, :]))
                tag  = "MECH" if idx in mech_set else "RBM "
                null_labels.append(f"[{tag}] Mode {idx:>2}")
            print_matrix(Vh[null_indices, :], null_labels, col_labels)
        else:
            print("    No null space modes found.")

        # ── [5] Mechanism Null Space Basis Q ──────────────────────────────────
        if Q is not None and mechanism_indices is not None:
            header(f"[5]  MECHANISM NULL SPACE BASIS  Q     shape: {k_mech} × {n_dofs}")
            print("  Q = rows of Vh for mechanism modes only.")
            print("  Each row is a unit nodal-displacement vector that folds at least one hinge.")
            print()
            q_row_labels = [f"Mode {idx:>2}" for idx in mechanism_indices]
            print_matrix(Q, q_row_labels, col_labels)

        # ── [6] Fold Angle Matrix A = J @ Q^T ─────────────────────────────────
        if A is not None and mechanism_indices is not None:
            header(f"[6]  FOLD ANGLE MATRIX  A = J · Qᵀ     shape: {n_hinges} × {k_mech}")
            print("  A[:,r] = hinge fold angles when mechanism mode r is activated at unit amplitude.")
            print("  Large column entries → that mode drives significant folding at those hinges.")
            print()
            a_col_labels = [f"Mode {idx:>2}" for idx in mechanism_indices]
            print_matrix(A, hinge_labels, a_col_labels)

        # ── [7] SVD of A ──────────────────────────────────────────────────────
        if U_sv is not None and S_sv is not None and Vt_sv is not None:
            k = len(S_sv)
            header(f"[7]  SVD  of  A  →  A = U · Σ · Vᵀ     ({k} singular mode(s))")
            print("  σ_r  = fold efficiency of mode r: max fold output per unit null-space displacement.")
            print("  U[:,r] = fold pattern in hinge space  (what you see on the hinges).")
            print("  Vt[r,:] = mixing weights in mechanism space  (how to combine Q rows).")

            # Σ — singular values
            selected_r = chosen_mode_idx if chosen_mode_idx is not None else 0
            subheader("Σ  —  Singular Values  (fold efficiency, dominant first)")
            print(f"    {'Rank':>6} │ {'σ':>14} │  Note")
            print(f"    {'─'*6}─┼─{'─'*14}─┼─{'─'*35}")
            for r, sv in enumerate(S_sv):
                note = f"  ← SELECTED (best M/V alignment)" if r == selected_r else ""
                print(f"    {r:>6} │ {sv:>14.6f} │{note}")

            # U — fold patterns in hinge space
            subheader("U  —  Left Singular Vectors  (fold patterns in hinge space)")
            print("    Column r = fold pattern for the r-th principal mode.")
            print(f"    best_sensitivity = U[:,{selected_r}] · σ_{selected_r}   (selected mode)\n")
            u_col_labels = [f"σ_{r}={S_sv[r]:.3f}" for r in range(k)]
            print_matrix(U_sv, hinge_labels, u_col_labels)

            # Vt — mixing weights in mechanism space
            subheader("Vᵀ  —  Right Singular Vectors  (mixing weights in mechanism space)")
            print("    Row r = weights applied to Q rows to produce the r-th fold pattern.")
            print("    v_dominant = Qᵀ · Vt[0,:]  →  the actual nodal displacement vector.\n")
            vt_row_labels = [f"SVD mode {r}" for r in range(k)]
            q_short_labels = ([f"Q_m{idx}" for idx in mechanism_indices]
                              if mechanism_indices else [f"q{r}" for r in range(Vt_sv.shape[1])])
            print_matrix(Vt_sv, vt_row_labels, q_short_labels)

        # ── [8] Dominant Nodal Displacement v* ────────────────────────────────
        if v_dominant is not None:
            header("[8]  DOMINANT NODAL DISPLACEMENT  v*  =  Qᵀ · Vt[0,:]")
            print("  The physical nodal displacement that produces best_sensitivity.")
            print("  Verify:  C · v* = 0  (all bars satisfied).  J · v* = best_sensitivity.")
            print()
            v3 = np.array(v_dominant).reshape(-1, 3)
            print(f"  {'Node':>6} │ {'dx':>13} │ {'dy':>13} │ {'dz':>13} │ {'‖d‖':>10}")
            print(f"  {'─'*6}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*10}")
            for i, (dx, dy, dz) in enumerate(v3):
                mag  = float(np.sqrt(dx**2 + dy**2 + dz**2))
                fmtv = lambda x: f"{'  0.0':>13}" if abs(x) < 1e-9 else f"{x:>13.6f}"
                print(f"  {i:>6} │ {fmtv(dx)} │ {fmtv(dy)} │ {fmtv(dz)} │ {mag:>10.6f}")

            # Quick verification: J @ v* should equal best_sensitivity
            if sensitivity_vector is not None:
                jv = dihedral_jacobian @ v_dominant
                max_err = np.max(np.abs(jv - np.array(sensitivity_vector)))
                print(f"\n  Verification  ‖J·v* − best_sensitivity‖∞ = {max_err:.2e}"
                      f"  {'✓ consistent' if max_err < 1e-9 else '⚠ check sign flip'}")

        # ── [9] Final Sensitivity + M/V Validation ────────────────────────────
        _r = chosen_mode_idx if chosen_mode_idx is not None else 0
        header(f"[9]  FINAL SENSITIVITY VECTOR  =  U[:,{_r}] · σ_{_r}  (sign-corrected, best M/V alignment)")
        print("  These are the hinge fold sensitivities for the selected physical mechanism.")
        print()
        print(f"  {'Hinge':>7} │ {'Assign':>7} │ {'Target t':>10} │ {'Sensitivity':>14} │ {'M/V Check':>12}")
        print(f"  {'─'*7}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*12}")
        all_match = True
        for i, h in enumerate(self.hinges):
            asgn  = h.fold_assignment
            t_val = t[i] if t is not None else float('nan')
            s_val = sensitivity_vector[i] if sensitivity_vector is not None else float('nan')
            if asgn == 'M':
                ok = s_val >= 0
            elif asgn == 'V':
                ok = s_val <= 0
            else:
                ok = None
            check = ("✓" if ok else "✗ MISMATCH") if ok is not None else "— (unassigned)"
            if ok is False:
                all_match = False
            t_str = f"{t_val:>+10.1f}" if t is not None else f"{'N/A':>10}"
            print(f"  {i:>7} │ {asgn:>7} │ {t_str} │ {s_val:>+14.6f} │ {check:>12}")

        if sensitivity_vector is not None and t is not None:
            ns = np.linalg.norm(sensitivity_vector)
            nt = np.linalg.norm(t)
            if ns > 1e-12 and nt > 1e-12:
                cos_sim = np.dot(sensitivity_vector, t) / (ns * nt)
                quality = ('excellent' if cos_sim > 0.99 else
                           'good'      if cos_sim > 0.90 else
                           'moderate'  if cos_sim > 0.50 else 'poor')
                print(f"\n  M/V cosine alignment:  {cos_sim:.6f}  ({quality})")
        verdict = "  ✓ All folds match M/V assignments." if all_match else \
                  "  ⚠ WARNING: One or more folds are inconsistent with M/V assignments."
        print(verdict)

        print("\n" + "═" * W + "\n")

    def mountain_valley_check(self, sensitivity_vector):
        # 5. Validate: every hinge's sensitivity sign should match its .fold assignment.
        #    Mountain (M) → positive,  Valley (V) → negative.
        print("\n--- FOLD ASSIGNMENT VALIDATION ---")
        all_match = True
        for i, h in enumerate(self.hinges):
            s_val = sensitivity_vector[i]
            if h.fold_assignment == 'M':
                match = s_val >= 0
            elif h.fold_assignment == 'V':
                match = s_val <= 0
            else:
                continue  # skip unassigned hinges
            status = "✓" if match else "✗ MISMATCH"
            if not match:
                all_match = False
            print(f"  H{i} ({h.fold_assignment}): s = {s_val:+.6f}  {status}")
        if all_match:
            print("  All folds are consistent with .fold assignments.")
        else:
            print("  WARNING: Some folds are inconsistent with .fold assignments!")
        print("-" * 40)

    def plot_pattern_vector(self, sensitivity_vector=None, nodal_vectors=None, vector_scale=1.0, vector_color='green', show_node_labels=False, show_hinge_labels=False, show_magnitudes=False, title="Pattern", show_colorbar=True, normalize=True, save_path=None):
        """
        Plot the origami pattern with:
          - Pattern boundary edges drawn in light grey (internal cross-bars hidden).
          - Top-down view locked for publication-ready figures.
          - Hinges colored strictly by absolute fold rate magnitude (0 to 1).
          - Hinge styles based on direction: Mountain (+) = Solid, Valley (-) = Dashed.
          - Optional nodal displacement vectors drawn as quiver arrows.
        """
        import matplotlib.lines as mlines
        import matplotlib.colors as mcolors # Added this so your custom colormap works!

        plt.close('all')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Sensitivity data & Normalization ---
        raw_sens = np.zeros(len(self.hinges))
        if sensitivity_vector is not None:
            raw_sens = np.array(sensitivity_vector).flatten()

        max_abs_val = np.max(np.abs(raw_sens))
        if max_abs_val < 1e-12:
            max_abs_val = 1.0

        if normalize:
            # Scale absolute magnitudes from 0 to 1
            abs_sens = np.abs(raw_sens) / max_abs_val
            limit = 1.0
        else:
            abs_sens = np.abs(raw_sens)
            limit = max_abs_val

        # --- Color map ---
        # Viridis is the academic standard for sequential data (0 to 1)
        base_cmap = plt.cm.viridis

        # Sample the colormap from 0.3 to 1.0 (skipping the lightest 30%)
        # Adjust 0.3 up or down to make the baseline yellow darker or lighter
        color_subset = base_cmap(np.linspace(0.3, .99, 256))

        # Create a brand new, darker colormap
        cmap = mcolors.ListedColormap(color_subset)
        cnorm = plt.Normalize(0, limit)

        # --- Nodes ---
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', s=20, alpha=0.4)

        if show_node_labels:
            for n in self.nodes:
                ax.text(n.coordinates[0], n.coordinates[1], n.coordinates[2],
                        f"{n.id}", fontsize=8, color='grey')

        # --- Pattern Boundary Outlines (No Internal Cross Bars) ---
        edge_panel_counts = {}
        for panel in self.panels:
            num_nodes = len(panel.nodes)
            for i in range(num_nodes):
                node_a = panel.nodes[i]
                node_b = panel.nodes[(i + 1) % num_nodes]
                edge_id = tuple(sorted((node_a.id, node_b.id)))
                edge_panel_counts[edge_id] = edge_panel_counts.get(edge_id, 0) + 1

        plotted_edges = set()
        for panel in self.panels:
            num_nodes = len(panel.nodes)
            for i in range(num_nodes):
                node_a = panel.nodes[i]
                node_b = panel.nodes[(i + 1) % num_nodes]
                edge_id = tuple(sorted((node_a.id, node_b.id)))

                # Plot only if it's a true boundary edge (belongs to 1 panel)
                if edge_panel_counts[edge_id] == 1 and edge_id not in plotted_edges:
                    plotted_edges.add(edge_id)
                    p1, p2 = node_a.coordinates, node_b.coordinates
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color='black', alpha=0.5, linewidth=1.5)

        # --- Nodal displacement vectors (quiver) ---
        if nodal_vectors is not None:
            nv = np.array(nodal_vectors)
            if nv.ndim == 1 and len(nv) == 3 * len(self.nodes):
                nv = nv.reshape(-1, 3)
            if nv.ndim == 2 and len(nv) == len(self.nodes) and nv.shape[1] == 3:
                ax.quiver(xs, ys, zs,
                          nv[:, 0], nv[:, 1], nv[:, 2],
                          color=vector_color, length=vector_scale,
                          normalize=False, arrow_length_ratio=0.15)

        # --- Hinges: Color = Magnitude, LineStyle = Mountain/Valley ---
        for h_id, h in enumerate(self.hinges):
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates

            raw_val = raw_sens[h_id]
            mag_val = abs_sens[h_id]

            # Mountain (+) = Solid, Valley (-) = Dashed
            l_style = '-' if raw_val >= -1e-9 else (0, (2.0, 0.35))

            color = cmap(cnorm(mag_val))

            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]],
                    color=color, linestyle=l_style, linewidth=5.5, alpha=0.95)

            # --- Optional Text Labels ---
            if show_hinge_labels or show_magnitudes:
                mid = (p_j + p_k) / 2
                label_text = ""

                # Add Hinge ID if requested
                if show_hinge_labels:
                    label_text += f"H{h_id}"

                # Add Magnitude if requested (rounded to 2 decimal places)
                if show_magnitudes:
                    if label_text:
                        label_text += ": "
                    label_text += f"{mag_val:.2f}"

                # Print the text with a heavy white outline so it stands out against the lines
                ax.text(mid[0], mid[1], mid[2], label_text,
                        color='black', fontsize=10, fontweight='bold',
                        zorder=10, # Forces text to draw on top of the lines
                        path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

        # --- Axis limits & Top-Down Publication View ---
        all_coords = np.array([xs, ys, zs])
        max_range = np.ptp(all_coords, axis=1).max() / 2.0
        mid_x = np.mean(all_coords[0])
        mid_y = np.mean(all_coords[1])
        mid_z = np.mean(all_coords[2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=90, azim=-90)
        ax.set_axis_off()
        ax.set_title(title, pad=0, y=0.95, fontsize=16, fontweight='bold')

        # --- Colorbar (Magnitude) ---
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6,aspect=15, pad=0.0005)
            cbar.ax.tick_params(labelsize=20)
            # You commented these out in your script, but you can turn them back on anytime!
            # cbar_label = 'Absolute Normalized Fold Rate' if normalize else 'Absolute Hinge Sensitivity (rad/unit)'
            # cbar.set_label(cbar_label, rotation=270, labelpad=20)
            cbar.outline.set_visible(False)

        fig.tight_layout(pad=0)

        # --- Automated Save Logic ---
        if save_path:
            # bbox_inches='tight' crops out all the extra white space
            # transparent=True removes the white background so it blends perfectly into the document
            fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
            print(f"Saved high-res figure to: {save_path}")

        plt.show()
