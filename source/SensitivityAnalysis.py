import numpy as np
import sympy as sp
import itertools
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.linalg import eigh

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import *
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *

"""
This is the meat of this script
January 2026
Jake Sutton
"""

class SensitivityModel:
    def __init__(self, coordinates, panel_indices):
        self.nodes, self.panels = self.generate_geometry(coordinates, panel_indices)

        self.bars = self.generate_bars()
        self.hinges = self.generate_hinges()
        self.total_K = None

    def generate_geometry(self,coordinates, panel_indices):
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

        # This loop assigns nodes to different panels
        panels = []
        for i, idxs in enumerate(panel_indices):
            """ We look up the Ndoe objects using the indices provided.
            If panel 1 uses node index 2, and panel 2 uses node index 2,
            they both get the EXACT SAME Node object from memory. 
            This makes sure that if Node 2 moves, it moves for both panels"""

            p_nodes = [nodes[k] for k in idxs]
            panels.append(Panel(i, p_nodes))

        return nodes, panels

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

        return bars


    def generate_hinges(self):
        """ Detects shared edges and creates a hinge element. """
        hinges = []

        # map edges to panels
        edge_to_panels = {}

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

            if count > 2:
                panel_ids = [panel.id for panel in panel_list]
                raise ValueError(f"TOPOLOGY ERROR: Edge between Nodes {edge_key} is shared by {count} panels "
                    f"(Panels: {panel_ids}).\n"
                    "Real origami edges can only connect 2 panels. "
                    "Check your input indices for overlapping panels."
                )
            # if only 1 panel, its a free edge and no hinge is needed there
            if count < 2:
                continue

            # This logic below is if there are just 2 panels, we create a hinge
            panel1 = panel_list[0]
            panel2 = panel_list[1]

            # Identify Axis Nodes (j, k)
            # Find the actual Node objects in panel_1 matching the IDs in edge_key
            node_j = next(n for n in panel1.nodes if n.id == edge_key[0])
            node_k = next(n for n in panel1.nodes if n.id == edge_key[1])

            # Identify "Wing" Nodes (i, l) - any node NOT on the axis
            node_i = next(n for n in panel1.nodes if n.id not in edge_key)
            node_l = next(n for n in panel2.nodes if n.id not in edge_key)

            hinges.append(HingeElement(node_i, node_j, node_k, node_l))
        
        return hinges


    def assemble_stiffness_matrix(self):
        """
        Constructs the Global Stiffness Matrix (K_total) by combining:
        1. Bar Stiffness (Stretching energy) -> K_bars
        2. Hinge Stiffness (Folding energy) -> K_hinges
        """
        num_dof = len(self.nodes) * 3
        num_bars = len(self.bars)
        num_hinges = len(self.hinges)

        # 1. Intialize Compatibility Matrix and Bar Stiffness matrix (which is diagonal)
        # Compatibility Matrix dimensions: [Number of Bars x Total DOFs]
        self.compatibility_matrix = np.zeros((num_bars, num_dof)) 
        bar_stiffness_matrix = np.zeros(num_bars)

        for i, bar in enumerate(self.bars):
            self.compatibility_matrix[i, :] = bar.get_compatibility_matrix_row(num_dof)
            bar_stiffness_matrix[i] = bar.stiffness

        # 2. Build Jacobian Matrix and Hinge Stiffness (which is diagonal)
        # Jacobian Matrix dimensions: [Number of Hinges x Total DOFs]
        self.jacobian_matrix = np.zeros((num_hinges, num_dof))
        hinge_stiffness_matrix = np.zeros(num_hinges)

        for i, hinge in enumerate(self.hinges):
            self.jacobian_matrix[i, :] = hinge.get_jacobian_row(num_dof)
            hinge_stiffness_matrix[i] = hinge.stiffness

        # Matrix Multiplication and addition to get K_total 
        # We use np.diag() to turn the 1D stiffness arrays into diagonal matrices
        K_bars = self.compatibility_matrix.T @ np.diag(bar_stiffness_matrix) @ self.compatibility_matrix
        K_hinges = self.jacobian_matrix.T @ np.diag(hinge_stiffness_matrix) @ self.jacobian_matrix

        total_K = K_bars + K_hinges
        return total_K

    def print_stiffness_matrix(self):
        sp.init_printing(use_unicode=True)
        if getattr(self, 'total_K', None) is None:
            print("Total K not found. Assembling now...")
            self.total_K = self.assemble_stiffness_matrix()
        K_sym = sp.Matrix(np.round(self.total_K, 4)) 
        sp.pretty_print(K_sym)
        

    def solve_for_eigenvalues(self):
        """
        Solves the generalized eigenvalue problem: K * v = lambda * v
        Returns sorted eigenvalues and eigenvectors.
        """
        if getattr(self, 'total_K', None) is None:
            self.total_K = self.assemble_stiffness_matrix()


        # Solve for eigenvalues (eigh is optimized for symmetric/Hermitian matrices)
        eigenvalues, eigenvectors = eigh(self.total_K)

        # Sort results (smallest eigenvalues first)
        # The index array 'idx' tells us how to re-order the vectors to match the values
        idx = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]

        return sorted_eigenvalues, sorted_eigenvectors
    


    def analyze_sensitivity(self, num_modes_to_check=3, return_mode_index=None):
        """
        Performs analysis on mechanism modes.
        
        Arguments:
        - num_modes_to_check: Number of modes to print details for.
        - return_mode_index: (int or list) 
             If int: returns sensitivity of that specific mode (e.g., 6).
             If list: returns the SUM of sensitivities of those modes (e.g., [6, 7]).
             If None: returns the primary mechanism mode (Mode 7/Index 6).
        """
        eigenvalues, eigenvectors = self.solve_for_eigenvalues()
        
        print("\n" + "="*40)
        print("      EIGENVALUE ANALYSIS RESULTS")
        print("="*40)
        
        # Rigid Body Check
        if np.any(eigenvalues[:6] > 1e-3):
            print(f"WARNING: Non-zero rigid body modes: {np.round(eigenvalues[:6], 5)}")
        else:
            print("Pass: Rigid body modes are effectively zero.")

        start_index = 6
        print(f"\n--- Mechanism Modes (Checking first {num_modes_to_check}) ---")
        
        # --- Print Info Loop ---
        for i in range(start_index, start_index + num_modes_to_check):
            if i >= len(eigenvalues): break
            e_val = eigenvalues[i]
            
            # Degeneracy Check
            if i > start_index:
                prev_val = eigenvalues[i-1]
                ratio = abs(e_val - prev_val) / (prev_val + 1e-12)
                if ratio < 0.01:
                    print(f"   [!] ALERT: Mode {i+1} is DEGENERATE with Mode {i}")

            # Calculate Sensitivity for display
            mode_v = eigenvectors[:, i]
            # Ensure you use self.J_matrix or self.jacobian_matrix (whichever is in your init)
            sens = self.jacobian_matrix @ mode_v 
            sens_norm = sens / (np.max(np.abs(sens)) + 1e-12)

            print(f"\n>> MODE {i+1} (Index {i}) | Energy: {e_val:.5e}")
            # ... (Print logic omitted for brevity, same as before) ...

        # --- Return Logic for Plotting ---
        # 1. Default case (Mode 7 / Index 6)
        target_indices = [6] 
        
        # 2. User specified specific mode or combination
        if return_mode_index is not None:
            if isinstance(return_mode_index, int):
                target_indices = [return_mode_index]
            elif isinstance(return_mode_index, list):
                target_indices = return_mode_index

        # Calculate Combined Sensitivity
        combined_sensitivity = np.zeros(self.jacobian_matrix.shape[0])
        
        print(f"\n--- Returning Sensitivity for Mode(s): {target_indices} ---")
        for idx in target_indices:
            if idx < len(eigenvalues):
                # We add the ABSOLUTE sensitivity vectors to visualize total motion area
                # (Or you can add signed vectors if you want to see cancellation)
                v = eigenvectors[:, idx]
                s = self.jacobian_matrix @ v
                combined_sensitivity += np.abs(s) 
            else:
                print(f"Error: Mode index {idx} out of bounds.")

        # Final Normalization
        max_val = np.max(np.abs(combined_sensitivity))
        if max_val > 1e-9:
            combined_sensitivity /= max_val
            
        return combined_sensitivity
        
    def plot_pattern(self, sensitivity_vector=None, show_node_labels=True, show_hinge_labels=True, title="Pattern"):
        """
        Visualizes the mechanism with Blue-to-Red sensitivity mapping.
        """
        plt.close('all') 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Process Colors ---
        # Default to blue (0.0) if no data provided
        norm_sens = np.zeros(len(self.hinges))
        
        if sensitivity_vector is not None:
            sens_abs = np.abs(sensitivity_vector)
            max_val = np.max(sens_abs)
            
            if max_val > 1e-12:
                # Normalize 0.0 to 1.0
                norm_sens = sens_abs / max_val
        
        # Create Color Map (Coolwarm: Blue=Low, Red=High)
        cmap = plt.cm.coolwarm

        # --- Plot Nodes & Bars ---
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', s=20, alpha=0.4)

        if show_node_labels:
            for n in self.nodes:
                ax.text(n.coordinates[0], n.coordinates[1], n.coordinates[2], 
                        f"{n.id}", fontsize=8, color='grey')

        for bar in self.bars:
            p1, p2 = bar.nodes[0].coordinates, bar.nodes[1].coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='black', alpha=0.3, linewidth=1)

        # --- Plot Hinges with Color Scale ---
        for h_id, h in enumerate(self.hinges):
            p_j, p_k = h.node_j.coordinates, h.node_k.coordinates
            
            intensity = norm_sens[h_id] # 0.0 to 1.0
            color = cmap(intensity)     # Get RGBA color
            
            # Thicker lines for active hinges
            width = 1.0 + (4.0 * intensity) 
            alpha = 0.4 + (0.6 * intensity)

            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]], 
                    color=color, linestyle='-', linewidth=width, alpha=alpha)

            if show_hinge_labels and intensity > 0.15: # Only label active hinges
                mid = (p_j + p_k) / 2
                ax.text(mid[0], mid[1], mid[2], f"H{h_id}", 
                        color=color, fontsize=10, fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])

        # --- Formatting ---
        # Equal aspect ratio hack for 3D
        all_coords = np.array([xs, ys, zs])
        max_range = np.ptp(all_coords, axis=1).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(all_coords[0]), np.mean(all_coords[1]), np.mean(all_coords[2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_title(title)
        
        # Add a Colorbar for reference
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Relative Sensitivity (Abs)', rotation=270, labelpad=15)

        plt.show()   