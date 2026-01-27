import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from source.helper_classes import *

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

        # 3. Matrix Multiplication to get K_total 
        # We use np.diag() to turn the 1D stiffness arrays into diagonal matrices

        # K_bars = compatibility_matrix.T * bar_stiffness_matrix * compatibility_matrix
        K_bars = self.compatibility_matrix.T @ np.diag(bar_stiffness_matrix) @ self.compatibility_matrix

        # K_hinges = jacobian_matrix.T * hinge_stiffness_matrix * jacobian_matrix
        K_hinges = self.jacobian_matrix.T @ np.diag(hinge_stiffness_matrix) @ self.jacobian_matrix

        # K_total = K_bars + K_hinges
        return K_bars + K_hinges

    def solve_for_eigenvalues(self):
        """
        Solves the generalized eigenvalue problem: K * v = lambda * v
        Returns sorted eigenvalues and eigenvectors.
        """
        # 1. Get the stiffness matrix
        K = self.assemble_stiffness_matrix()

        # 2. Solve for eigenvalues (eigh is optimized for symmetric/Hermitian matrices)
        eigenvalues, eigenvectors = eigh(K)

        # 3. Sort results (smallest eigenvalues first)
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
        

    def plot_pattern(self):
        """Visualizes nodes, bars, hinges, node indices, and hinge indices."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot Nodes
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', marker='o')

        # --- NODE LABELS ---
        for n in self.nodes:
            x, y, z = n.coordinates

            if hasattr(n, "vertex_id"):
                label = f"{n.id} (v{n.vertex_id})"
            else:
                label = f"{n.id}"

            ax.text(
                x, y, z,
                label,
                fontsize=8,
                color='darkred'
            )

        # 2. Plot Bars (Blue Lines)
        for bar in self.bars:
            p1 = bar.nodes[0].coordinates
            p2 = bar.nodes[1].coordinates
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                'b-',
                alpha=0.3
            )

        # 3. Plot Hinges (Red Dashed Lines + Labels)
        for h_id, h in enumerate(self.hinges):
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates

            # Draw hinge line
            ax.plot(
                [p_j[0], p_k[0]],
                [p_j[1], p_k[1]],
                [p_j[2], p_k[2]],
                'r--',
                linewidth=2
            )

            # Hinge midpoint
            mx = 0.5 * (p_j[0] + p_k[0])
            my = 0.5 * (p_j[1] + p_k[1])
            mz = 0.5 * (p_j[2] + p_k[2])

            # Label hinge
            ax.text(
                mx, my, mz,
                f"H{h_id}",
                fontsize=9,
                color='blue',
                fontweight='bold'
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
                