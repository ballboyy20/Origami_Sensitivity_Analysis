import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
import sys
import os
from source.helper_classes import *

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

                # Intersection finds shared nodes
                shared = sorted(ids1.intersection(ids2))

                # If they share exactly 2 nodes, it's a hinge axis
                if len(shared) == 2:
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


    def analyze_sensitivity(self):
        number_DOFs = len(self.nodes) * 3 # three DOF for each node, (not each panel. FEA only sees nodes and elements. Our rigid panels are made when we add realyl high stiffness to the bar elements)

        # STEP 1: Assemble Matrices
            # Step 1.1: build combatibility matrix
            # Step 1.2: Build jacobian matrix
            # Step 1.3: Build physics matrix

        # STEP 2: Build Stiff Matrix
            # K_bars = compatiblity_matrix.transpose() * bar_stiffness_matrix * compatibility_matrix
            # K_hinges = jacobian_matrix.transpose() * higne_stiffness_matrix * jacobian_matrix
            # K_total = K_bars + K_hignes

        # STEP 3: Solve Eigenvalues

        # STEP 4: Isolate mode 7:
            # make sure that modes 1-6 (0-5) are actually zero or near zero

        # STEP 5: Calculate Sensitivity: 
            # Multiply jacobian_matrix by eigen vector 7

            # and boom thats your sensitivity

    def plot_pattern(self):
        """ This plots the whole dang thing so I can see what the beep is happening"""

        """ Visualizes the generated nodes, bars, and hinge axes. """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot Nodes
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', marker='o')

        # 2. Plot Bars (Blue Lines)
        for bar in self.bars:
            p1 = bar.nodes[0].coordinates
            p2 = bar.nodes[1].coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.3)

        # 3. Plot Hinges (Red Dashed Lines) to verify detection
        for h in self.hinges:
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates
            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]], 'r--', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        

        