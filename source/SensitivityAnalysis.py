import numpy as np
import sympy as sp
import itertools
import json
from scipy.linalg import eigh

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import *
    from source.visualization import SensitivityVisualizationMixin
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *
    from visualization import SensitivityVisualizationMixin

"""
This is the meat of this script
Febuaray 2026
Jake Sutton
"""

class SensitivityModel(SensitivityVisualizationMixin):
    def __init__(self, fold_file_path, verbose=True):
        """ Upon initializing this class makes the origami pattern, then adds the bars between
        nodes in a panel to make it rigid, and also slaps on some hinges. Telling it where the hignes
        are is helpful for calculatring the dihedral angle jacobian. """
        self.verbose = verbose
        self.coordinates, self.panel_indices, self.crease_info = self.extract_pattern_data_from_fold_file(fold_file_path)

        self.nodes, self.panels = self.generate_geometry(self.coordinates, self.panel_indices)

        self.bars = self.generate_bars()
        self.hinges = self.generate_hinges()

    def analyze_sensitivity(self, show_plot=None, plot_title=None,show_colorbar=True, save_path=None, silent=None):
        """
        Identifies the physical folding mechanism via SVD. Auto-calibrates
        hinges to align with target M/V assignments from the .fold file.
        """
        if silent is None:
            silent = not self.verbose

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
        self._print(f"\n{'='*60}")
        self._print(f" STEPPING OUT OF FLAT STATE (Step Scale: {step_scale})")
        self._print(f"{'='*60}")

        # 1. Ensure we have a dominant mode to follow from the flat state
        if not hasattr(self, 'v_dominant') or self.v_dominant is None:
            self._print("Running initial flat-state analysis to find deployment path...")
            self.analyze_sensitivity(show_plot=False)
            
        # 2. Reshape the 1D displacement vector into (N, 3) for the nodes
        v_reshaped = self.v_dominant.reshape(-1, 3)

        # 3. Apply the displacement to every node
        for i, node in enumerate(self.nodes):
            node.coordinates = node.coordinates + (v_reshaped[i] * step_scale)

        self._print(f"Nodes perturbed by {step_scale} * v_dominant. Re-running analysis on 3D geometry...\n")
        
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
            self._print("WARNING: No mechanism detected in the Null Space.")
            
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
    
    
