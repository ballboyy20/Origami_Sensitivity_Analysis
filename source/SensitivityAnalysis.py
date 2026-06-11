import numpy as np
import sympy as sp
from scipy.linalg import eigh

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import *
    from source.visualization import SensitivityVisualizationMixin
    from source.OrigamiModel import OrigamiModel
    from source.ConstraintModel import ConstraintModel
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *
    from visualization import SensitivityVisualizationMixin
    from OrigamiModel import OrigamiModel
    from ConstraintModel import ConstraintModel

"""
This is the meat of this script
Febuaray 2026
Jake Sutton
"""

class SensitivityModel(SensitivityVisualizationMixin):
    def __init__(self, fold_file_path):
        """ Upon initializing this class makes the origami pattern, then adds the bars between
        nodes in a panel to make it rigid, and also slaps on some hinges. Telling it where the hignes
        are is helpful for calculatring the dihedral angle jacobian. """

        self.geometry = OrigamiModel(fold_file_path)
        self.zero_tolerance = 1e-9
        self.constraints = ConstraintModel(self.geometry)
        self.hinges = self.generate_hinges()

    @property
    def nodes(self):
        return self.geometry.nodes

    @property
    def panels(self):
        return self.geometry.panels

    @property
    def coordinates(self):
        return self.geometry.coordinates

    @property
    def panel_indices(self):
        return self.geometry.panel_indices

    @property
    def crease_info(self):
        return self.geometry.crease_info

    def get_characteristic_length(self):
        return self.geometry.get_characteristic_length()

    @property
    def bars(self):
        return self.constraints.bars

    def build_constraint_matrix(self):
        return self.constraints.build_constraint_matrix()

    def analyze_sensitivity(self, show_plot=None, plot_title=None,show_colorbar=True, save_path=None, silent=None):
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
            self.v_dominant = None
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
        # best_sensitivity = best_sensitivity * characteristic_length
        
        # 9. Report & Validate
        if not silent:
            self.report_singular_values(S_sv, best_r)
            self.report_alignment(best_sensitivity, target_fold_vector)        
            self.print_system_matrices(
                dihedral_jacobian, constraint_matrix, singular_values, Vh, best_sensitivity,
                mechanism_indices=mechanism_indices, Q=Q, A=A, U_sv=U_sv, S_sv=S_sv, Vt_sv=Vt_sv,
                v_dominant=v_dominant, t=target_fold_vector, chosen_mode_idx=best_r
            )

        self.best_sensitivity = best_sensitivity
        self.v_dominant = v_dominant

        # The Kinematic Efficiency
        max_sensitivity = np.max(np.abs(best_sensitivity))
            
        # The Normalized Vector (0 to 1)
        s_normalized_to_max_sensitivity = np.abs(best_sensitivity) / max_sensitivity

        # Coefficient of Variation (CV = std / mean)
        cv = np.std(s_normalized_to_max_sensitivity) / np.mean(s_normalized_to_max_sensitivity)
        cv_percentage = cv * 100

        # The Dead Hinge metric
        min_fold = np.min(s_normalized_to_max_sensitivity)
        
        if not silent:
            print(f"\nKinematic Efficiency: max |sensitivity| = {max_sensitivity:.6f} radians per nothing (non-dimensionalized)")
            print(f"Coefficient of Variation (CV) of normalized sensitivity: {cv:.4f} ({cv_percentage:.2f}%) - lower means more uniform sensitivity across hinges")
            print(f"Dead Hinge Metric (min of normalized sensitivity): {min_fold:.6f} (higher is better, 0 means at least one completely dead hinge)")

        if show_plot == 'yes':
            self.plot_pattern_vector(best_sensitivity,
                                    show_magnitudes=True,
                                    title=plot_title,
                                    normalize=True,
                                    show_colorbar=show_colorbar,
                                    save_path=save_path)

        return best_sensitivity
    
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

    def step_and_reanalyze(self, step_scale=0.03, show_plot=False, silent=None):
        """
        Steps the pattern slightly using the 
        linear tangent vector (v_dominant), and re-runs the sensitivity analysis.
        This breaks the flat-state singularity.
        """
        if not silent:
            print(f"\n{'='*60}")
            print(f" STEPPING OUT OF FLAT STATE (Step Scale: {step_scale})")
            print(f"{'='*60}")

        # 1. Ensure we have a dominant mode to follow from the flat state
        if not hasattr(self, 'v_dominant') or self.v_dominant is None:
            print("Running initial flat-state analysis to find deployment path...")
            self.analyze_sensitivity(show_plot=False, silent=silent)
            
        # 2. Reshape the 1D displacement vector into (N, 3) for the nodes
        v_reshaped = self.v_dominant.reshape(-1, 3)

        # 3. Apply the displacement to every node
        for i, node in enumerate(self.nodes):
            node.coordinates = node.coordinates + (v_reshaped[i] * step_scale)

        if not silent:
            print(f"Nodes perturbed by {step_scale} * v_dominant. Re-running analysis on 3D geometry...\n")
        
        # 4. Re-run the analysis on the now-3D geometry
        new_sensitivity = self.analyze_sensitivity(show_plot=show_plot, silent=silent)
        
        return new_sensitivity

    def isolate_mechanism_subspace(self, singular_values, Vh, dihedral_jacobian):
        """Filters the null space to remove pure rigid body motions and zero-energy noise."""
        n_dof = Vh.shape[0]
        mechanism_indices = []   

        for i in range(n_dof):
            s_val = singular_values[i] if i < len(singular_values) else 0.0 
            if s_val < self.zero_tolerance:
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

############
        cos_vals = sorted([
            abs(np.dot(U_sv[:, r], target_fold_vector) /
                (np.linalg.norm(U_sv[:, r]) * np.linalg.norm(target_fold_vector)))
            for r in range(len(S_sv)) if S_sv[r] >= 1e-3 * S_sv[0]
        ], reverse=True)

        self.cosine_quality = cos_vals[0] if len(cos_vals) > 0 else 0.0
        self.cosine_quality_runner_up = cos_vals[1] if len(cos_vals) > 1 else 0.0
     ###############   


        best_sensitivity = U_sv[:, best_r] * S_sv[best_r]
        v_dominant = Q.T @ Vt_sv[best_r, :]

        # Fix the global sign
        if np.linalg.norm(target_fold_vector) > 1e-12 and np.dot(best_sensitivity, target_fold_vector) < 0:
            best_sensitivity = -best_sensitivity
            v_dominant = -v_dominant
            
        return best_sensitivity, v_dominant, U_sv, S_sv, Vt_sv, best_r
    
    def auto_calibrate_hinges(self, best_sensitivity, target_fold_vector, Q):
        """Checks for flipped hinge signs and re-runs the Jacobian math if any are swapped."""
        # print("\nChecking for scrambled hinge orientations based on M/V assignments...")
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
            # print("Mismatches found! Swapping internal panel definitions and rerunning Dihedral Jacobian...")
            dihedral_jacobian = self.build_dihedral_jacobian()
            A = dihedral_jacobian @ Q.T
            
            # Re-extract with the newly corrected matrices
            best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r = self.extract_dominant_mode(A, Q, target_fold_vector)
            # print("Rerun complete. Hinges are now permanently aligned to the .fold file.")
            
            return best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r, dihedral_jacobian, A
            
        # print("No scrambled orientations found. Initial pass is perfectly aligned.")
        return None
    
    def build_dihedral_jacobian(self):
        number_of_hinges = len(self.hinges)
        number_of_nodes = len(self.nodes)
        total_DOFs = 3 * number_of_nodes

        dihedral_jacobian = np.zeros((number_of_hinges, total_DOFs))

        for i, hinge in enumerate(self.hinges):
            dihedral_jacobian[i,:] = hinge.get_jacobian_row(total_DOFs)

        return dihedral_jacobian
    
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
    
    
