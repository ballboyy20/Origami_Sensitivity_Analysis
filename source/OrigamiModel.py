import numpy as np
import json

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import Node, Panel
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import Node, Panel

"""
OrigamiModel: the Geometry layer.
Parses a .fold file and builds the Node/Panel objects that the
Constraint, Kinematics, and Analysis layers operate on.
"""

class OrigamiModel:
    def __init__(self, fold_file_path):
        self.coordinates, self.panel_indices, self.crease_info = self.extract_pattern_data_from_fold_file(fold_file_path)
        self.nodes, self.panels = self.generate_geometry(self.coordinates, self.panel_indices)

    def __repr__(self):
        return (f"OrigamiModel("
                f"{len(self.nodes)} nodes, "
                f"{len(self.panels)} panels)")

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
