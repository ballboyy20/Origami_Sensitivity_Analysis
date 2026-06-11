import numpy as np
import itertools

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import BarElement
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import BarElement

"""
ConstraintModel: the Constraint layer.
Builds the rigid bar network for each panel and the constraint
(compatibility) matrix C that the Analysis layer runs SVD on.
"""

class ConstraintModel:
    def __init__(self, geometry):
        self.geometry = geometry
        self.bars = self.generate_bars()

    def __repr__(self):
        return f"ConstraintModel({len(self.bars)} bars)"

    def build_constraint_matrix(self):
        """
        Builds the constraint matrix (from the bars)
        """
        number_of_bars = len(self.bars)
        number_of_nodes = len(self.geometry.nodes)
        total_DOFs = 3 * number_of_nodes

        constraint_matrix = np.zeros((number_of_bars, total_DOFs))

        for i, bar in enumerate(self.bars):
            constraint_matrix[i, :] = bar.get_compatibility_matrix_row(total_DOFs)

        return constraint_matrix

    def generate_bars(self):
        """
        Creates a rigid "truss" for every panel, regardless of how many sides the panel has.
        It makes a bar between a node and every other node. 4 nodes = 6 bars, 3 nodes = regular trianlge

        WARNING: This creates non-deterministic panels. If this program were to be scaled/edited to analyze
        panel bending/shearing, this function would need to be edited to generate deterministic panels.

        """

        unique_edges = set()
        bars = []

        for panel in self.geometry.panels:
            # Connect every node to every other node in this specific panel
            for node_a, node_b in itertools.combinations(panel.nodes,2):

                # Sort IDs to ensure Edge(1,2) = Edge (2,1)
                edge_id = tuple(sorted((node_a.id, node_b.id)))

                if edge_id not in unique_edges:
                    unique_edges.add(edge_id)
                    #Add the rigid bar
                    bars.append(BarElement(node_a, node_b))

        # for panel in self.geometry.panels:
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
