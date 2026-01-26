import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

class Node:
    def __init__(self, id, x,y,z):
        # NOTE: These are the vertex coordinates of the pattern in the deployed state
        self.id = id
        self.coordinates = np.array([x,y,z], dtype=float)

class BarElement:
    def __init__(self,node_a, node_b, stiffness = 1e12):
        self.nodes = [node_a, node_b]
        self.stiffness = stiffness # Has really high stiffness to simulate the bars being rigid. We are assuming the panels themselves are ideally rigid. 
        
    def get_compatibility_matrix_row(self,total_DOFs):
        #TODO...Calculate direction cosines ...
        # extract the node coordinates
        point1 = self.nodes[0].coordinates
        point2 = self.nodes[1].coordinates

        # Calculate length between the nodes (length of the bar)
        vector_between_nodes = point2 - point1
        length_of_bar = np.linalg.norm(vector_between_nodes)

        # The direction cosines are the normalized components
        direction_cosine_x, direction_cosine_y, direction_cosine_z = vector_between_nodes/ length_of_bar

        # Intialize the row vector that will get put into the compatibility matrix
        row_vector = np.zeros(total_DOFs)

        # Node 1 indices
        # The syntax "* 3" is how we skip through the array to find the exact "starting slot" for a specific node.
        # The "index_1 : index_1 + 3" put the direction cosines for this bar in the right spot of the matrix
        index_1 = self.nodes[0].id * 3
        row_vector[index_1 : index_1 + 3] = [-direction_cosine_x, - direction_cosine_y, -direction_cosine_z]

        # Node 2 indices
        index_2 = self.nodes[1].id * 3
        row_vector[index_2 : index_2 + 3] = [direction_cosine_x, direction_cosine_y, direction_cosine_z]

        
        # Return the row for the compatibility matrix
        return row_vector

class HingeElement:
    def __init__(self,node_i, node_j, node_k, node_l, stiffness = 1.0):
        """By convention: j and k are the SHARED nodes (the hinge line)
             and l are the unique nodes on either side"""
        
        # Enforce hinge direction j -> k, in case they are input incorrectly into intializer
        if node_j.id > node_k.id:
            node_j, node_k = node_k, node_j
            node_i, node_l = node_l, node_i

        self.node_i = node_i
        self.node_j = node_j
        self.node_k = node_k
        self.node_l = node_l
        self.stiffness = stiffness # Has much lower stiffness than the bar element because this is the part we are interested in, we want the hignes to move

    def calculate_vectors(self):
        """ calculates/intializes several vectors used in this class to reduce duplicate code """

        # Vector for the hinge line (intersection between the panels)
        self.hinge_line_vector = self.node_k.coordinates - self.node_j.coordinates
        self.length_of_hinge_line = np.linalg.norm(self.hinge_line_vector)

        # Vectors from the hinge to the outer nodes 
        self.r_ji = self.node_i.coordinates - self.node_j.coordinates # (j -> i)
        self.r_jl = self.node_l.coordinates - self.node_j.coordinates # (j -> l)

        # Get vectors normal to each panel
        self.panel_1_normal_vector = np.cross(self.r_ji, self.hinge_line_vector)
        self.panel_2_normal_vector = np.cross(self.hinge_line_vector, self.r_jl)

    def calculate_dihedral_angle(self): 
        """The dihedral angle is the angle between two planes, 
            we find this by finding the angle between the vectors normal to each plane"""
        
        self.calculate_vectors()

        # use arctan2 to calculate angle between the two panels
        y_component = np.dot(np.cross(self.panel_1_normal_vector,self.panel_2_normal_vector), self.hinge_line_vector) / self.length_of_hinge_line
        x_component = np.dot(self.panel_1_normal_vector, self.panel_2_normal_vector)

        return np.arctan2(y_component,x_component)

    def get_jacobian_row(self, total_DOFs):
        """Mathematically, the Jacobian (J) answers:
        "If I wiggle Node i in some direction, exactly how much does the fold angle θ change?"""
        
        self.calculate_vectors()

        # Calculate squared magnitudes (needed for the denominator)
        panel_1_normal_vector_squared = np.dot(self.panel_1_normal_vector, self.panel_1_normal_vector)
        panel_2_normal_vector_squared = np.dot(self.panel_2_normal_vector, self.panel_2_normal_vector)

        # Safety check for zero-area triangles (collinear nodes)
        if panel_1_normal_vector_squared < 1e-12 or panel_2_normal_vector_squared < 1e-12:
            return np.zeros(total_DOFs)
        
        # Calculate gradients (The "sensitivity" vectors)
        # [cite_start]Formulas derived from Schenk & Guest (2011)

        # Gradient for nodes i & l
        gradient_i = (self.length_of_hinge_line / panel_1_normal_vector_squared) * self.panel_1_normal_vector
        gradient_l = (self.length_of_hinge_line / panel_2_normal_vector_squared) * self.panel_2_normal_vector

        # Projection factors: How far along the hinge are node i and l?
        alpha_i = np.dot(self.r_ji, self.hinge_line_vector) / (self.length_of_hinge_line * self.length_of_hinge_line)
        alpha_l = np.dot(self.r_jl, self.hinge_line_vector) / (self.length_of_hinge_line * self.length_of_hinge_line)

        # gradient for j & k
        gradient_j = (alpha_i - 1) * gradient_i + (alpha_l - 1) * gradient_l
        gradient_k = (-alpha_i) * gradient_i - alpha_l * gradient_l

        # intialize empty row vector to populate with gradients
        row = np.zeros(total_DOFs)

        # Helper function  to stamp 3 values at a time into the correct slots
        def stamp(node_id, vector):
            idx = node_id * 3
            row[idx : idx+3] = vector

        stamp(self.node_i.id, gradient_i)
        stamp(self.node_j.id, gradient_j)
        stamp(self.node_k.id, gradient_k)
        stamp(self.node_l.id, gradient_l)

        return row

    
class Panel:
    def __init__(self, id, nodes):
        ## a panel will hold its own nodes. I will check each panel against all other panels to see if it shares nodes with other panels
        ## If it does share nodes we will make a hinge there. If the panel already has 4 hinges, we won't check it against other panels. waste of compute
        self.id = id
        self.nodes = nodes

    def get_nodes(self):
        return self.nodes



class FlasherSensitivityModel:
    def __init__(self, nodes, bars, hinges):
        self.nodes = nodes
        self.bars = bars
        self.hinges = hinges

    def generate_bars(self):
        pass

    def generate_hinge(self):
        pass

    def analyze_sensativity(self):
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
        pass

        