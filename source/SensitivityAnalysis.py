import numpy as np
import sympy as sp
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
from scipy.linalg import eigh
from matplotlib import animation

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import *
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *

"""
This is the meat of this script
Febuaray 2026
Jake Sutton
"""

class SensitivityModel:
    def __init__(self, fold_file_path):
        """ Upon initializing this class makes the origami pattern, then adds the bars between
        nodes in a panel to make it rigid, and also slaps on some hinges. Telling it where the hignes
        are is helpful for calculatring the dihedral angle jacobian. """
        self.coordinates, self.panel_indices, self.crease_info = self.extract_pattern_data_from_fold_file(fold_file_path)

        self.nodes, self.panels = self.generate_geometry(self.coordinates, self.panel_indices)

        self.bars = self.generate_bars()
        self.hinges = self.generate_hinges()
        
    def analyze_sensitivity(self, show_plot=None, plot_title=None,show_colorbar=True, save_path=None, silent=False):
        """
        Identifies the physical folding mechanism via SVD. Auto-calibrates 
        hinges to align with target M/V assignments from the .fold file.
        """
        # 1. Build Matrices
        dihedral_jacobian = self.build_dihedral_jacobian()
        constraint_matrix = self.build_constraint_matrix()

        # 2. Solve SVD on Constraint Matrix
        _, singular_values, Vh = np.linalg.svd(constraint_matrix)

        # 3. Isolate mechanism subspace
        mechanism_indices = self.isolate_mechanism_subspace(singular_values, Vh, dihedral_jacobian)
        if not mechanism_indices:
            return np.zeros(len(self.hinges))

        # 4. Build mechanism subspace matrix Q and Fold matrix A
        Q = Vh[mechanism_indices, :]
        A = dihedral_jacobian @ Q.T

        # 5. Build target vector
        target_fold_vector = self.build_target_fold_vector()

        # 6. Extract dominant mode (Initial Pass)
        best_sensitivity, v_dominant, U_sv, S_sv, Vt_sv, best_r = self.extract_dominant_mode(A, Q, target_fold_vector)

        # 7. Auto-Calibrate (Swap backwards hinges and rerun if necessary)
        recal_results = self.auto_calibrate_hinges(best_sensitivity, target_fold_vector, Q)
        if recal_results is not None:
            best_sensitivity, v_dominant, U_sv, S_sv, Vt_sv, best_r, dihedral_jacobian, A = recal_results

        # 8. Non-dimensionalize sensitivity by characteristic length to get units of radians per model-length-unit
        characteristic_length = self.get_characteristic_length()
        best_sensitivity = best_sensitivity * characteristic_length
        
        if not silent:
            print(f"\nNon-dimensionalized sensitivity using characteristic length: {characteristic_length:.4f} units")

        # 9. Report & Validate
        if not silent:
            self.report_singular_values(S_sv, best_r)
            self.report_alignment(best_sensitivity, target_fold_vector)
            self.mountain_valley_check(best_sensitivity)
        
        
            self.print_system_matrices(
                dihedral_jacobian, constraint_matrix, singular_values, Vh, best_sensitivity,
                mechanism_indices=mechanism_indices, Q=Q, A=A, U_sv=U_sv, S_sv=S_sv, Vt_sv=Vt_sv,
                v_dominant=v_dominant, t=target_fold_vector, chosen_mode_idx=best_r
            )


            if show_plot is 'yes':
                self.plot_pattern_vector(best_sensitivity,
                                        show_magnitudes=False,
                                        title=plot_title,
                                        normalize=True,
                                        show_colorbar=show_colorbar,
                                        save_path=save_path)

        self.best_sensitivity = best_sensitivity
        self.v_dominant = v_dominant

        # The Kinematic Efficiency
        max_sensitivity = np.max(np.abs(best_sensitivity))
        if not silent:
            print(f"\nKinematic Efficiency: max |sensitivity| = {max_sensitivity:.6f} radians per nothing (non-dimensionalized)")

        # The Normalized Vector (0 to 1)
        s_normalized_to_max_sensitivity = np.abs(best_sensitivity) / max_sensitivity

        # Coefficient of Variation (CV = std / mean)
        cv = np.std(s_normalized_to_max_sensitivity) / np.mean(s_normalized_to_max_sensitivity)
        cv_percentage = cv * 100
        if not silent:
            print(f"Coefficient of Variation (CV) of normalized sensitivity: {cv:.4f} ({cv_percentage:.2f}%) - lower means more uniform sensitivity across hinges")

        # The Dead Hinge metric
        min_fold = np.min(s_normalized_to_max_sensitivity)

        if not silent:
            print(f"Dead Hinge Metric (min of normalized sensitivity): {min_fold:.6f} (higher is better, 0 means at least one completely dead hinge)")

        return best_sensitivity
    
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
    
    def get_instantaneous_mechanism(self, target_fold_vector):
        """
        A silent, streamlined version of analyze_sensitivity used purely for 
        iterative path integration. Returns the normalized displacement vector.
        """
        J = self.build_dihedral_jacobian()
        C = self.build_constraint_matrix()

        _, sv, Vh = np.linalg.svd(C)
        
        # Isolate mechanisms (thresholds might need tuning once out of flat state)
        mechanism_indices = []
        for i in range(Vh.shape[0]):
            s_val = sv[i] if i < len(sv) else 0.0
            if s_val < 1e-6: # Relaxed slightly for numerical drift during integration
                v = Vh[i, :]
                fold_changes = J @ v
                if np.sum(np.abs(fold_changes)) >= 1e-5:
                    mechanism_indices.append(i)

        if not mechanism_indices:
            return None # Pattern has locked up (kinematic singularity)

        Q = Vh[mechanism_indices, :]
        A = J @ Q.T

        U_sv, S_sv, Vt_sv = np.linalg.svd(A, full_matrices=False)

        # Find best match to target fold vector
        best_r = 0
        best_cos = -1.0
        if np.linalg.norm(target_fold_vector) > 1e-12:
            for r in range(len(S_sv)):
                if S_sv[r] < 1e-3 * S_sv[0]: continue
                cos = np.dot(U_sv[:, r], target_fold_vector) / (np.linalg.norm(U_sv[:, r]) * np.linalg.norm(target_fold_vector))
                if abs(cos) > best_cos:
                    best_cos = abs(cos)
                    best_r = r

        v_dominant = Q.T @ Vt_sv[best_r, :]
        best_sens = U_sv[:, best_r] * S_sv[best_r]

        # Keep the global sign consistent with the target
        if np.dot(best_sens, target_fold_vector) < 0:
            v_dominant = -v_dominant

        return v_dominant
    
    def step_and_reanalyze(self, step_scale=0.03, show_plot=False):
        """
        Pushes the flat pattern slightly into the 3D deployed state using the 
        linear tangent vector (v_dominant), and re-runs the sensitivity analysis.
        This breaks the flat-state singularity.
        """
        print(f"\n{'='*60}")
        print(f" STEPPING OUT OF FLAT STATE (Step Scale: {step_scale})")
        print(f"{'='*60}")

        # 1. Ensure we have a dominant mode to follow from the flat state
        if not hasattr(self, 'v_dominant') or self.v_dominant is None:
            print("Running initial flat-state analysis to find deployment path...")
            self.analyze_sensitivity(show_plot=False)
            
        # 2. Reshape the 1D displacement vector into (N, 3) for the nodes
        v_reshaped = self.v_dominant.reshape(-1, 3)

        # 3. Apply the displacement to every node
        for i, node in enumerate(self.nodes):
            node.coordinates = node.coordinates + (v_reshaped[i] * step_scale)

        print(f"Nodes perturbed by {step_scale} * v_dominant. Re-running analysis on 3D geometry...\n")
        
        # 4. Re-run the analysis on the now-3D geometry
        new_sensitivity = self.analyze_sensitivity(show_plot=show_plot)
        
        return new_sensitivity

    def get_characteristic_length(self):
        """
        Calculates the bounding radius of the array from its geometric center
        using the raw .fold file coordinates.
        """
        coords = np.array(self.coordinates)
        center = np.mean(coords, axis=0) # Find the geometric center (X,Y,Z)
        
        # Calculate the distance from the center to every single vertex
        distances = np.linalg.norm(coords - center, axis=1)
        
        # The characteristic length is the distance to the furthest vertex
        return np.max(distances)
    
    def isolate_mechanism_subspace(self, singular_values, Vh, dihedral_jacobian):
        """Filters the null space to remove pure rigid body motions and zero-energy noise."""
        n_dof = Vh.shape[0]
        mechanism_indices = []   

        for i in range(n_dof):
            s_val = singular_values[i] if i < len(singular_values) else 0.0 
            if s_val < 1e-9:
                v = Vh[i, :]
                fold_changes = dihedral_jacobian @ v
                total_folding = np.sum(np.abs(fold_changes))

                if total_folding >= 1e-5:
                    mechanism_indices.append(i)

        if not mechanism_indices:
            print("WARNING: No mechanism detected in the Null Space.")
            
        return mechanism_indices
    
    def extract_dominant_mode(self, A, Q, target_fold_vector):
        """Runs SVD on the fold matrix and finds the mode that best matches the target vector."""
        U_sv, S_sv, Vt_sv = np.linalg.svd(A, full_matrices=False)

        if np.linalg.norm(target_fold_vector) > 1e-12:
            best_r = 0
            best_cos = -1.0
            rel_threshold = 1e-3 * S_sv[0]   
            for r in range(len(S_sv)):
                if S_sv[r] < rel_threshold:  
                    continue
                cos = np.dot(U_sv[:, r], target_fold_vector) / (np.linalg.norm(U_sv[:, r]) * np.linalg.norm(target_fold_vector))
                if abs(cos) > best_cos:      
                    best_cos = abs(cos)
                    best_r = r
        else:
            best_r = 0                       

        best_sensitivity = U_sv[:, best_r] * S_sv[best_r]
        v_dominant = Q.T @ Vt_sv[best_r, :]

        # Fix the global sign
        if np.linalg.norm(target_fold_vector) > 1e-12 and np.dot(best_sensitivity, target_fold_vector) < 0:
            best_sensitivity = -best_sensitivity
            v_dominant = -v_dominant
            
        return best_sensitivity, v_dominant, U_sv, S_sv, Vt_sv, best_r
    
    def auto_calibrate_hinges(self, best_sensitivity, target_fold_vector, Q):
        """Checks for flipped hinge signs and re-runs the Jacobian math if any are swapped."""
        print("\nChecking for scrambled hinge orientations based on M/V assignments...")
        mismatches_found = False
        
        for i, h in enumerate(self.hinges):
            s_val = best_sensitivity[i]
            
            # If math says negative, but assignment is Mountain (+)
            if h.fold_assignment == 'M' and s_val < -1e-5:
                h.wing_nodes_1, h.wing_nodes_2 = h.wing_nodes_2, h.wing_nodes_1
                h.node_i, h.node_l = h.node_l, h.node_i
                mismatches_found = True
                
            # If math says positive, but assignment is Valley (-)
            elif h.fold_assignment == 'V' and s_val > 1e-5:
                h.wing_nodes_1, h.wing_nodes_2 = h.wing_nodes_2, h.wing_nodes_1
                h.node_i, h.node_l = h.node_l, h.node_i
                mismatches_found = True

        if mismatches_found:
            print("Mismatches found! Swapping internal panel definitions and rerunning Dihedral Jacobian...")
            dihedral_jacobian = self.build_dihedral_jacobian()
            A = dihedral_jacobian @ Q.T
            
            # Re-extract with the newly corrected matrices
            best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r = self.extract_dominant_mode(A, Q, target_fold_vector)
            print("Rerun complete. Hinges are now permanently aligned to the .fold file.")
            
            return best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r, dihedral_jacobian, A
            
        print("No scrambled orientations found. Initial pass is perfectly aligned.")
        return None
    
    def report_singular_values(self, S_sv, best_r):
        """Prints the mechanism subspace singular values, highlighting the chosen mode."""
        print(f"\nMechanism subspace singular values (fold efficiency per unit displacement):")
        for r, sv in enumerate(S_sv):
            marker = f"  ← selected (best M/V alignment, rank {r})" if r == best_r else ""
            print(f"  σ_{r} = {sv:.6f}{marker}")

    def build_target_fold_vector(self):
        """Creates the +1 (Mountain) and -1 (Valley) target vector from hinge assignments."""
        target_fold_vector = np.zeros(len(self.hinges))
        # print("\nTarget fold vector (t):")
        
        for i, h in enumerate(self.hinges):
            if h.fold_assignment == 'M':
                target_fold_vector[i] = +1.0
            elif h.fold_assignment == 'V':
                target_fold_vector[i] = -1.0
                
            # print(f"  Hinge {i:>4} ({h.fold_assignment}): t = {target_fold_vector[i]:+.1f}")
            
        return target_fold_vector
    
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
            
    def build_dihedral_jacobian(self):
        number_of_hinges = len(self.hinges)
        number_of_nodes = len(self.nodes)
        total_DOFs = 3 * number_of_nodes

        dihedral_jacobian = np.zeros((number_of_hinges, total_DOFs))

        for i, hinge in enumerate(self.hinges):
            dihedral_jacobian[i,:] = hinge.get_jacobian_row(total_DOFs)

        return dihedral_jacobian
    
    def build_constraint_matrix(self):
        """
        Builds the constraint matrix (from the bars)
        """
        number_of_bars = len(self.bars)
        number_of_nodes = len(self.nodes)
        total_DOFs = 3 * number_of_nodes

        constraint_matrix = np.zeros((number_of_bars, total_DOFs))

        for i, bar in enumerate(self.bars):
            constraint_matrix[i, :] = bar.get_compatibility_matrix_row(total_DOFs)

        return constraint_matrix

    def extract_pattern_data_from_fold_file(self, fold_file_path):
        """
        Parses a .fold file and yanks the data from it. 

        Returns a type list of [x,y,z] coords
        panel_indices type list
        crease_lines (set)
        """
        with open(fold_file_path, 'r') as file:
            data = json.load(file)

        # extract coordinates
        coordinates = data['vertices_coords']
        #if z-coord is missing, add zero for z coord
        if len(coordinates[0]) == 2:
            coordinates = [[c[0], c[1], 0.0] for c in coordinates]

        # extract panel indices
        panel_indices = data['faces_vertices']

        # extract crease lines
        # M = mountian V = valley B = boundry U = unassigned
        crease_info = {}
        if 'edges_vertices' in data and 'edges_assignment' in data:
            for edge, assignment in zip(data['edges_vertices'], data['edges_assignment']):
                # Filter for Mountains and Valleys only
                if assignment in ['M', 'V']: 
                    # Sort indices so (1,2) is the same as (2,1)
                    u, v = sorted(edge)
                    crease_info[(u, v)] = assignment

                    """
                    crease_info is a dictionary that looks like this:
                    {
                    (13, 14): "M",  # From Index 1
                    (3, 15): "M",   # From Index 2
                    ...
                    (11, 21): "V"   # From Index 50
                    }
                    """

        return coordinates, panel_indices, crease_info

    def generate_geometry(self,coordinates, panel_indices):
        """ Generates the node objects and the panel obejects.
        These are simple object. See helper_classes to look at them."""
        
        nodes = self.generate_nodes(coordinates)
        panels = self.generate_panels(nodes,panel_indices)

        return nodes, panels
    
    def generate_panels(self, nodes, panel_indices):
        # This loop assigns nodes to different panels
        panels = []
        for i, idxs in enumerate(panel_indices):
            """ We look up the Node objects using the indices provided.
            If panel 1 uses node index 2, and panel 2 uses node index 2,
            they both get the EXACT SAME Node object from memory. 
            This makes sure that if Node 2 moves, it moves for both panels"""

            p_nodes = [nodes[k] for k in idxs]
            panels.append(Panel(i, p_nodes))

            
        return panels

    def generate_nodes(self, coordinates):
        """ Coordinates: List of [X,Y,Z] for every unique vertex (node)
        panel_indices: List of lists, e.g., [[0,1,2], [0,2,3,4]]
        This handles n-sides polygon panels"""

        # This loop creates all the nodes with the provided coordinates
        nodes = []
        count = 0
        for coordinate_list in coordinates:
            x = coordinate_list[0]
            y = coordinate_list[1]
            z = coordinate_list[2]

            new_node = Node(count, x ,y ,z)
            nodes.append(new_node)
            count += 1

        return nodes

    def generate_bars(self):
        """ 
        Creates a rigid "truss" for every panel, regardless of how many sides the panel has.
        It makes a bar between a node and every other node. 4 nodes = 6 bars, 3 nodes = regular trianlge
        
        WARNING: This creates non-deterministic panels. If this program were to be scaled/edited to analyze 
        panel bending/shearing, this function would need to be edited to generate deterministic panels. 

        """

        unique_edges = set()
        bars = []

        for panel in self.panels:
            # Connect every node to every other node in this specific panel
            for node_a, node_b in itertools.combinations(panel.nodes,2):

                # Sort IDs to ensure Edge(1,2) = Edge (2,1)
                edge_id = tuple(sorted((node_a.id, node_b.id)))

                if edge_id not in unique_edges:
                    unique_edges.add(edge_id)
                    #Add the rigid bar
                    bars.append(BarElement(node_a, node_b))

        # for panel in self.panels:
        #     nodes = panel.nodes
        #     n = len(nodes)

        #     # 1. All perimeter edges
        #     for i in range(n):
        #         a, b = nodes[i], nodes[(i + 1) % n]
        #         edge_id = tuple(sorted((a.id, b.id)))
        #         if edge_id not in unique_edges:
        #             unique_edges.add(edge_id)
        #             bars.append(BarElement(a, b))

        #     # 2. Fan diagonals from node 0 to nodes 2, 3, ..., n-2
        #     #    (node 0→1 and node 0→n-1 are already perimeter edges)
        #     for i in range(2, n - 1):
        #         a, b = nodes[0], nodes[i]
        #         edge_id = tuple(sorted((a.id, b.id)))
        #         if edge_id not in unique_edges:
        #             unique_edges.add(edge_id)
        #             bars.append(BarElement(a, b))

        return bars

    def generate_hinges(self):
        """ Creates a hinge where the .fold file says there should be a hinge. If there is no .fold file,
          it creates a hinge between every edge that has 2 panels on it."""

        hinges = []

        # map edges to panels
        edge_to_panels = {}

        # this loops checks to see if an edge has panels touching it already
        for panel in self.panels:
            number_nodes = len(panel.nodes)
            for i in range(number_nodes):
                node1 = panel.nodes[i]
                node2 = panel.nodes[(i+1) % number_nodes] #wrap around

                edge_key = tuple(sorted((node1.id, node2.id)))
                if edge_key not in edge_to_panels:
                    edge_to_panels[edge_key] = []
                edge_to_panels[edge_key].append(panel)

        for edge_key, panel_list in edge_to_panels.items():
            count = len(panel_list)

            if count > 2: # if there is more than 2 panels on an edge, something is wrong
                panel_ids = [panel.id for panel in panel_list]
                raise ValueError(f"TOPOLOGY ERROR: Edge between Nodes {edge_key} is shared by {count} panels "
                    f"(Panels: {panel_ids}).\n"
                    "Real origami edges can only connect 2 panels. "
                    "Check your input indices for overlapping panels."
                )
            # if only 1 panel, its a free edge and no hinge is needed there
            if count < 2:
                continue

            # Default assignment if no dictionary is provided
            assignment = 'U' 

            # If we have the dictionary from the .fold file...
            if hasattr(self, 'crease_info') and self.crease_info is not None:
                # Check if this edge exists in our "Valid Creases" list
                if edge_key in self.crease_info:
                    assignment = self.crease_info[edge_key] # assignment should be grabbing an "M" or a "V", and then assiging it to the hinge.
                else:
                    # If it's NOT in the dictionary, it's likely a Boundary ('B')
                    # that we filtered out in the parser. Skip it!
                    continue

            # This logic below is if there are just 2 panels, we create a hinge
            panel1 = panel_list[0]
            panel2 = panel_list[1]

            # Identify Axis Nodes (j, k)
            # Find the actual Node objects in panel_1 matching the IDs in edge_key
            node_j = next(n for n in panel1.nodes if n.id == edge_key[0])
            node_k = next(n for n in panel1.nodes if n.id == edge_key[1])

            # Collect ALL non-hinge nodes for each panel (centroid-based Jacobian)
            wing_nodes_1 = [n for n in panel1.nodes if n.id not in edge_key]
            wing_nodes_2 = [n for n in panel2.nodes if n.id not in edge_key]

            hinges.append(HingeElement(wing_nodes_1, node_j, node_k, wing_nodes_2, assignment))
        
        return hinges
    
    
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