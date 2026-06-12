import numpy as np

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import HingeElement
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import HingeElement

"""
KinematicsModel: the Kinematics layer.
Builds the hinge network for each crease and the dihedral-angle
Jacobian that maps nodal velocities to fold-angle rates.
"""

class KinematicsModel:
    def __init__(self, geometry):
        self.geometry = geometry
        self.hinges = self.generate_hinges()

    def __repr__(self):
        return f"KinematicsModel({len(self.hinges)} hinges)"

    def build_dihedral_jacobian(self):
        number_of_hinges = len(self.hinges)
        number_of_nodes = len(self.geometry.nodes)
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
        for panel in self.geometry.panels:
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
            if hasattr(self.geometry, 'crease_info') and self.geometry.crease_info is not None:
                # Check if this edge exists in our "Valid Creases" list
                if edge_key in self.geometry.crease_info:
                    assignment = self.geometry.crease_info[edge_key] # assignment should be grabbing an "M" or a "V", and then assiging it to the hinge.
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

    def get_panel_interpolation_matrix(self, panel_id, weights):
        """
        STUB - not yet implemented.

        Intended purpose (Phi(p)): return a (3 x total_DOFs) matrix mapping the
        full nodal-displacement vector to the (3,) displacement of a point p
        inside panel `panel_id`, expressed as a barycentric combination
        (`weights`) of that panel's corner-node displacements (valid since each
        panel is a rigid body). Will be used by the future Analysis layer's
        get_piston_sensitivity().
        """
        raise NotImplementedError("get_panel_interpolation_matrix is not yet implemented")
