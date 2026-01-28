import numpy as np
import sympy as sp
import itertools
import matplotlib.pyplot as plt
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

        # Iterate through unique pairs of panels
        # Logic: Compare Panel 0 with 1, 2, 3... then Panel 1 with 2, 3...
        for i in range(len(self.panels)):
            for j in range(i + 1, len(self.panels)):
                panel1 = self.panels[i]
                panel2 = self.panels[j]

                ids1 = {n.id for n in panel1.nodes}
                ids2 = {n.id for n in panel2.nodes}

                occupied_axis = set()

                # Intersection finds shared nodes
                shared = sorted(ids1.intersection(ids2))

                # If they share exactly 2 nodes, it's a hinge axis
                if len(shared) == 2:

                    # The next few lines check to see if theres already an axis here
                    axis_key = tuple(sorted(shared))
                    if axis_key in occupied_axis:
                        continue
                    occupied_axis.add(axis_key)


                    # 1. Identify Hinge Axis Nodes (j, k)
                    # Note: We grab the node objects from panel1's list
                    node_j = next(n for n in panel1.nodes if n.id == shared[0])
                    node_k = next(n for n in panel1.nodes if n.id == shared[1])

                    
                    # 2. Identify "Wing" Nodes (i, l)
                    # Use 'next' to find the first node in the panel that ISN'T in the shared set
                    # This works for triangles AND quads/n-gons (any point off the axis defines the plane)
                    node_i = next(n for n in panel1.nodes if n.id not in shared)
                    node_l = next(n for n in panel2.nodes if n.id not in shared)
                    
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
    


    def analyze_sensitivity(self):
        """
        Performs the analysis to find the mechanism sensitivity.
        1. Solves Eigenvalues
        2. Isolates Mode 7 (The first non-rigid mechanism)
        3. Calculates Sensitivity (Change in fold angles) = J * eigenvector
        """
        eigenvalues, eigenvectors = self.solve_for_eigenvalues()
        
        print("\n--- Eigenvalue Analysis Results ---")
        # Check Rigid Body Modes (Modes 0-5 should be practically zero)
        print("First 6 Eigenvalues (should be ~0 for rigid body motion):")
        print(np.round(eigenvalues[:6], 5))
        
        print(f"Mode 7 Eigenvalue (Mechanism energy): {eigenvalues[6]:.5e}")

        # --- Isolate Mode 7 ---
        # In a free-floating 3D structure, the first 6 modes are Rigid Body Motions (3 trans + 3 rot).
        # Therefore, the 7th mode (index 6) is the first actual deformation mechanism.
        mechanism_mode_index = 6
        
        if len(eigenvalues) <= mechanism_mode_index:
            print("Error: System does not have enough DOFs for a mechanism mode.")
            return None

        # This is 'v', the displacement vector of the nodes
        mechanism_eigenvector = eigenvectors[:, mechanism_mode_index]

        # --- Calculate Sensitivity ---
        # Sensitivity = How much do the hinges rotate for this mechanism?
        # Formula: d_theta = J * v
        sensitivity = self.jacobian_matrix @ mechanism_eigenvector
        
        # Normalize sensitivity so the max fold change is 1.0 (makes it easier to read)
        sensitivity = sensitivity / np.max(np.abs(sensitivity))

        print("\n--- Sensitivity Analysis (Fold Angle Changes) ---")
        for i, val in enumerate(sensitivity):
            print(f"Hinge {i}: Sensitivity = {val:.4f}")

        return sensitivity
        

    def plot_pattern(self, sensitivity_vector=None):
        """
        Visualizes the mechanism.
        If 'sensitivity_vector' is provided, it color-codes hinges:
        Blue = Stationary, Red = Moving
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- 1. Plot Nodes ---
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', marker='o', s=20)

        # Node Labels
        for n in self.nodes:
            ax.text(n.coordinates[0], n.coordinates[1], n.coordinates[2], 
                    f"{n.id}", fontsize=8, color='black', zorder=10)

        # --- 2. Plot Bars ---
        for bar in self.bars:
            p1 = bar.nodes[0].coordinates
            p2 = bar.nodes[1].coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    'k-', alpha=0.2, linewidth=1)

        # --- 3. Plot Hinges (With Sensitivity Coloring) ---
        
        # Prepare color map if sensitivity data exists
        if sensitivity_vector is not None:
            # Normalize to 0.0 - 1.0
            max_val = np.max(np.abs(sensitivity_vector))
            if max_val > 0:
                norm_sens = np.abs(sensitivity_vector) / max_val
            else:
                norm_sens = np.zeros(len(self.hinges))
        
        for h_id, h in enumerate(self.hinges):
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates

            # Determine Color and Width
            if sensitivity_vector is not None:
                intensity = norm_sens[h_id]
                color = plt.cm.coolwarm(intensity) # Blue -> Red
                width = 1.5 + (3 * intensity) 
                label_color = color
                should_label = intensity > 0.1 # Only label active hinges
            else:
                color = 'blue'
                width = 2
                label_color = 'blue'
                should_label = True

            # Plot Hinge Line
            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]], 
                    color=color, linestyle='--', linewidth=width)

            # Label hinge
            if should_label:
                mx = 0.5 * (p_j[0] + p_k[0])
                my = 0.5 * (p_j[1] + p_k[1])
                mz = 0.5 * (p_j[2] + p_k[2])
                
                ax.text(mx, my, mz, f"H{h_id}", 
                        fontsize=10, color=label_color, fontweight='bold')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        """
        Visualizes the mechanism.
        If 'sensitivity_vector' is provided, it color-codes hinges:
        Blue = Stationary, Red = Moving
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- 1. Plot Nodes ---
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', marker='o', s=20)

        # Node Labels
        for n in self.nodes:
            ax.text(n.coordinates[0], n.coordinates[1], n.coordinates[2], 
                    f"{n.id}", fontsize=8, color='black', zorder=10)

        # --- 2. Plot Bars ---
        for bar in self.bars:
            p1 = bar.nodes[0].coordinates
            p2 = bar.nodes[1].coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    'k-', alpha=0.2, linewidth=1)

        # --- 3. Plot Hinges (With Sensitivity Coloring) ---
        
        # Prepare color map if sensitivity data exists
        if sensitivity_vector is not None:
            # Normalize to 0.0 - 1.0
            max_val = np.max(np.abs(sensitivity_vector))
            if max_val > 0:
                norm_sens = np.abs(sensitivity_vector) / max_val
            else:
                norm_sens = np.zeros(len(self.hinges))
        
        for h_id, h in enumerate(self.hinges):
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates

            # Determine Color and Width
            if sensitivity_vector is not None:
                # Get intensity (0 to 1)
                intensity = norm_sens[h_id]
                # Use Matplotlib Colormap (Coolwarm: Blue->Red)
                color = plt.cm.coolwarm(intensity) 
                # Make active hinges thicker
                width = 1 + (4 * intensity) 
                label_color = color
            else:
                color = 'blue'
                width = 2
                label_color = 'blue'

            # Plot Hinge Line
            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]], 
                    color=color, linestyle='--', linewidth=width)

            # Plot Hinge Label (H0, H1...) at midpoint
            mx = 0.5 * (p_j[0] + p_k[0])
            my = 0.5 * (p_j[1] + p_k[1])
            mz = 0.5 * (p_j[2] + p_k[2])
            
            # Only label if it's moving (cleaner plot) or if no sensitivity data
            if sensitivity_vector is None or norm_sens[h_id] > 0.1:
                ax.text(mx, my, mz, f"H{h_id}", 
                        fontsize=10, color=label_color, fontweight='bold')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()