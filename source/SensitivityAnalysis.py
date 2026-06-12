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
    from source.KinematicsModel import KinematicsModel
    from source.FlatStateAnalysis import FlatStateAnalysis
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *
    from visualization import SensitivityVisualizationMixin
    from OrigamiModel import OrigamiModel
    from ConstraintModel import ConstraintModel
    from KinematicsModel import KinematicsModel
    from FlatStateAnalysis import FlatStateAnalysis

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
        self.constraints = ConstraintModel(self.geometry)
        self.kinematics = KinematicsModel(self.geometry)
        self.analysis = FlatStateAnalysis(self.geometry, self.constraints, self.kinematics)

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

    @property
    def hinges(self):
        return self.kinematics.hinges

    def build_dihedral_jacobian(self):
        return self.kinematics.build_dihedral_jacobian()

    def build_target_fold_vector(self):
        return self.analysis.build_target_fold_vector()

    def isolate_mechanism_subspace(self, singular_values, Vh, dihedral_jacobian):
        return self.analysis.isolate_mechanism_subspace(singular_values, Vh, dihedral_jacobian)

    def extract_dominant_mode(self, A, Q, target_fold_vector):
        return self.analysis.extract_dominant_mode(A, Q, target_fold_vector)

    def auto_calibrate_hinges(self, best_sensitivity, target_fold_vector, Q):
        return self.analysis.auto_calibrate_hinges(best_sensitivity, target_fold_vector, Q)

    @property
    def cosine_quality(self):
        return getattr(self.analysis, 'cosine_quality', 0.0)

    @property
    def cosine_quality_runner_up(self):
        return getattr(self.analysis, 'cosine_quality_runner_up', 0.0)

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
