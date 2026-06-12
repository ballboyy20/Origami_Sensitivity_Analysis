import numpy as np

"""
FlatStateAnalysis: the Analysis layer.
Consumes the constraint matrix (ConstraintModel) and dihedral Jacobian
(KinematicsModel) to extract folding mechanism modes and sensitivities.
"""

class FlatStateAnalysis:
    def __init__(self, geometry, constraints, kinematics):
        self.geometry = geometry
        self.constraints = constraints
        self.kinematics = kinematics
        self.zero_tolerance = 1e-9

    def __repr__(self):
        return f"FlatStateAnalysis(zero_tolerance={self.zero_tolerance:.0e})"

    def build_target_fold_vector(self):
        """Creates the +1 (Mountain) and -1 (Valley) target vector from hinge assignments."""
        target_fold_vector = np.zeros(len(self.kinematics.hinges))
        # print("\nTarget fold vector (t):")

        for i, h in enumerate(self.kinematics.hinges):
            if h.fold_assignment == 'M':
                target_fold_vector[i] = +1.0
            elif h.fold_assignment == 'V':
                target_fold_vector[i] = -1.0

            # print(f"  Hinge {i:>4} ({h.fold_assignment}): t = {target_fold_vector[i]:+.1f}")

        return target_fold_vector

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

        for i, h in enumerate(self.kinematics.hinges):
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
            dihedral_jacobian = self.kinematics.build_dihedral_jacobian()
            A = dihedral_jacobian @ Q.T

            # Re-extract with the newly corrected matrices
            best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r = self.extract_dominant_mode(A, Q, target_fold_vector)
            # print("Rerun complete. Hinges are now permanently aligned to the .fold file.")

            return best_sens, v_dom, U_sv, S_sv, Vt_sv, best_r, dihedral_jacobian, A

        # print("No scrambled orientations found. Initial pass is perfectly aligned.")
        return None

    # --- New, not-yet-designed entry points (stubs, per big-picture plan) ---

    def get_misalignment_modes(self):
        """
        STUB - not yet implemented.

        Intended purpose: replaces the current SVD/mechanism-isolation logic in
        SensitivityModel.analyze_sensitivity (isolate_mechanism_subspace +
        extract_dominant_mode). Will run SVD on the constraint matrix C, isolate
        the mechanism null-space, and return the dominant folding mode(s) ranked
        by alignment with the target M/V fold vector. Real definition to be
        designed as its own focused step.
        """
        raise NotImplementedError("get_misalignment_modes is not yet implemented")

    def get_perturbation_sensitivity(self):
        """
        STUB - not yet implemented.

        Intended purpose: the "row-norm" sensitivity approach - for each hinge,
        the norm of its row in the dihedral Jacobian (or a related projected
        matrix), giving a per-hinge sensitivity measure without requiring a
        single dominant mode. Real definition to be designed as its own focused
        step.
        """
        raise NotImplementedError("get_perturbation_sensitivity is not yet implemented")

    def get_piston_sensitivity(self, panel_id, weights):
        """
        STUB - not yet implemented.

        Intended purpose: for optics use-cases - how much does a point p on
        panel `panel_id` (given as a barycentric combination `weights` of that
        panel's corners) translate/"piston" out of plane per unit of the
        dominant folding mode. Depends on
        KinematicsModel.get_panel_interpolation_matrix (Phi(p)), which is itself
        still a stub. Real definition to be designed as its own focused step.
        """
        raise NotImplementedError("get_piston_sensitivity is not yet implemented")

    def rank_coupling_candidates(self):
        """
        STUB - not yet implemented.

        Intended purpose: for optimization - rank candidate hinge pairs (or
        other coupling locations) by how much adding a rigid coupling between
        them would change the misalignment/sensitivity modes. Depends on
        ConstraintModel.add_coupling_constraint / CouplingConstraint, which do
        not exist yet ("Eventually" item in the big-picture plan). Real
        definition to be designed as its own focused step.
        """
        raise NotImplementedError("rank_coupling_candidates is not yet implemented")
