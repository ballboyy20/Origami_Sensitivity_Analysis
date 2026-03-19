'''program written by Kelvin Wang'''

import numpy as np
import matplotlib.pyplot as plt

class Bloom_Yoshimura:
    def __init__(self, M, H, S):
        """ input variables """
        self.M = M ; self.H = H ; self.S = S
        """ variables for computation: """
        self.point_set = set()
        self.point_map = dict()
        self.crease_set = set()
        self.edge_set = set()
        self.orthogonal_crease_set = set()
        self.diagonal_crease_set = set()
        self.transformed_point_set = set()
        self.facet_set = set()
        """ numerical values for computation: """
        self.alpha_angle = 2 * np.pi / self.M # alpha = 2pi/M
        self.height_length = 1/(2*np.tan(self.alpha_angle/2)) # h = 1 / 2tan(alpha/2)
        self.big_alpha_sequence = tuple(self.alpha_angle * np.array(range(0,self.M)))
        self.k_sequence = tuple(range(0,self.M)) # k = {0,1,...,M-1}
        self.j_sequence = tuple(range(0,self.H+1)) # j = {0,1,...,H}
        self.c_sequence = tuple(range(0,self.H)) # c = {0,1,...,H-1}
        """ variables for plotting: """
        self.plot_origin = bool()
        self.plot_points = bool()
        self.plot_facets = bool()
        self.plot_lines = bool()
        self.line_style = 1
        self.line_width = 1 # default line_width
        self.crease_is_invert = False # Boolean Value

    def graph(self):
        """ initialization: """
        self.define_point_set()
        self.point_map_initialize()
        self.define_crease_set()
        self.define_facet_set()
        """ computation of transformations: """
        self.slant_linear_transformation()
        self.translation_transformation()
        self.scale_linear_transformation()
        self.define_point_radial_duplicates()
        self.define_crease_radial_duplicates()
        self.define_facet_radial_duplicates()
        self.sequential_rotation_linear_transformation()
        """ plotting: """
        if self.plot_origin: self.plot_origin_point()
        if self.plot_points: self.plot_point_set()
        self.classify_crease()
        if self.plot_lines:
            if self.line_style: self.plot_colored_crease_set()
            else: self.plot_monochromatic_crease_set()
        if self.plot_facets: self.plot_facet_set()
        self.show_plot()


    '''DEFINING THE MODEL'''
    '''DEFINING POINTS'''
    def define_point_set(self):
        self.define_full_length_point_set()
        self.define_half_length_point_set()

    def define_full_length_point_set(self): # p(i,j,k) = ({0,1,...,H+1-j},j, not_assigned), j = {0,1,...,H}
        H = self.H
        j_sequence = self.j_sequence
        full_length_point_set = set()
        for j in j_sequence:
            i_sequence = tuple(range(0,H+2-j)) # i = {0,1,...,H+1-j}
            for i in i_sequence:
                full_length_point = tuple([i,j, 0])
                full_length_point_set.add(full_length_point)
        self.point_set = self.point_set.union(full_length_point_set)

    def define_half_length_point_set(self): # p(i,j,k) = (H+1/2-c, 1/2+c, not_assigned), c = {0,1,...,H-1}
        H = self.H ; c_sequence = self.c_sequence ; half_length_point_set = set()
        for c in c_sequence:
            half_length_point = tuple([H+1/2-c, 1/2+c, 0])
            half_length_point_set.add(half_length_point)
        self.point_set = self.point_set.union(half_length_point_set)
    
    def point_map_initialize(self): # Maps the ID of a point p(i,j,k) to its numerical value (i',j'), to prepare for linear transformations.
        point_set = self.point_set
        for point_id in point_set: self.point_map[point_id] = point_id[:2]  # p(i,j,k) = (i',j')

    def point_map_read(self, point_id): return self.point_map[point_id]

    def point_map_update(self, point_id, new_value): self.point_map[point_id] = new_value

    def define_point_radial_duplicates(self): # expands both point_map and point_set to include radial "mirrors" of a wedge of points.
        point_set = self.point_set ; k_sequence = self.k_sequence ; duplicate_point_set = set()
        for point_id in point_set:
            point_coordinates = self.point_map_read(point_id)
            for k in k_sequence:
                kth_radial_duplicate = tuple([point_id[0], point_id[1], k])
                duplicate_point_set.add(kth_radial_duplicate)
                self.point_map_update(kth_radial_duplicate, point_coordinates)
        self.point_set = duplicate_point_set

    '''DEFINING LINES'''
    def define_crease_set(self): # this function defines a set of creases of only one wedge of the pattern.
        point_set = self.point_set
        crease_set = set()
        for point_id in point_set:
            i,j,k = point_id[0],point_id[1],0
            potential_creases = [(i+1,j+1,k), (i+1,j,k), (i,j+1,k), (i+.5,j+.5,k), (i-.5,j+.5,k), (i+.5,j-.5,k)] #diagonal_points, horizontal_points, vertical_points, half_length_points A, B, and C
            for potential_crease in potential_creases:
                if potential_crease in point_set:
                    crease_set.add(frozenset({point_id,potential_crease})) # An crease (line segment) is defined as A––B = B––A, or {A,B} = {B,A}, where A and B are points.
        self.crease_set = crease_set

    def classify_crease(self): # three categories: diagonal crease, orthogonal crease, and edge. Can be used before OR after radial duplication.
        crease_set = self.crease_set
        H = self.H
        edge_set = set()
        orthogonal_crease_set = set()
        diagonal_crease_set = set()
        for crease in crease_set:
            (i,j,k),(i2,j2,k2) = crease # (i,j,k) and (i",j",k") are points in point_set
            if i2+j2==i+j: # implies either i"=i-.5,j"=j+.5 or i"=i+.5,j"=j-.5, which means the crease is a negative diagonal EDGE.
                edge_set.add(crease)
            elif j==H and j2==H: # means both points are vertices of an EDGE on top of the pattern.
                edge_set.add(crease)
            elif j==j2 and abs(i-i2)==1: # j=j" means the points are HORIZONTAL and |i-i"|=1 means they are one unit apart.
                orthogonal_crease_set.add(crease)
            elif i==i2 and abs(j-j2)==1: # i=i" means the points are VERTICAL and |j-j"|=1 means they are one unit apart.
                orthogonal_crease_set.add(crease)
            elif abs(i+j-i2-j2)==2: # |i+j-i"-j"|=2 means if two points are DIAGONAL and two taxi block units apart.
                diagonal_crease_set.add(crease)
            elif abs(i+j-i2-j2)==1 and abs(i-i2)==.5 and abs(j-j2)==.5: # if |i+j-i"-j"|=1, |i-i"|=.5 and |j-j"|=.5 , the two points are DIAGONAL and only half taxi-block-unit apart.
                diagonal_crease_set.add(crease)
            else:
                print("ERROR: misclassified crease")
        self.edge_set = edge_set
        self.orthogonal_crease_set = orthogonal_crease_set
        self.diagonal_crease_set = diagonal_crease_set

    def invert_crease_mountain_valley_assignment(self): # inverts crease mountain/valley assignment
            crease_is_invert = self.crease_is_invert
            self.crease_is_invert = not crease_is_invert

    def define_crease_radial_duplicates(self): # expands crease set to include radial "mirrors" of a wedge of creases.
        crease_set = self.crease_set
        duplicate_crease_set = set()
        k_sequence = self.k_sequence
        for crease in crease_set:
            (i1,j1,k1),(i2,j2,k2) = crease
            for k in k_sequence:
                duplicate_crease = frozenset({(i1,j1,k),(i2,j2,k)})
                duplicate_crease_set.add(duplicate_crease)
        self.crease_set = duplicate_crease_set

    '''DEFINING facetS'''
    def define_facet_set(self):
        facet_set = self.facet_set
        search_crease_set = self.crease_set
        for crease in search_crease_set: # before radial duplication
            p1,p2 = crease
            # Search the crease set for two other lines that satisfy {p1,p2},{p2,p3},{p1,p3}.
            for crease_2 in search_crease_set:
                p3,p4 = crease_2
                if p1==p3:
                    crease_3 = {p2,p4}
                    if crease_3 in search_crease_set:
                        facet = frozenset({p1,p2,p4})
                        facet_set.add(facet)
        self.facet_set = facet_set

    def define_facet_radial_duplicates(self):
        facet_set = self.facet_set
        duplicate_facet_set = set()
        k_sequence = self.k_sequence
        for facet in facet_set:
            (i1,j1,k1),(i2,j2,k2),(i3,j3,k3) = facet
            for k in k_sequence:
                duplicate_facet = frozenset({(i1,j1,k),(i2,j2,k),(i3,j3,k)})
                duplicate_facet_set.add(duplicate_facet)
        self.facet_set = duplicate_facet_set


    '''GRAPHING THE MODEL'''
    '''TRANSFORMATIONS'''
    def slant_linear_transformation(self): # this function inputs a set of vectors and the angle to slant them around the x axis (i-hat remains unchanged, j-hat axis slants, both i-hat and j-hat are not scaled).
        point_set = self.point_set
        angle = self.alpha_angle
        slant_matrix = tuple([[1, np.cos(angle)],
                              [0, np.sin(angle)]])
        transformed_point_set = set()
        for point_id in point_set:
            point = self.point_map_read(point_id)
            transformed_point = np.dot(slant_matrix, point) # matrix-vector multiplication
            transformed_point = tuple(transformed_point)
            transformed_point_set.add(transformed_point)
            self.point_map_update(point_id, transformed_point)
        self.transformed_point_set = transformed_point_set

    def translation_transformation(self): # this function inputs change in (x and y) direction and translates all coordinates by that direction.
        point_set = self.point_set ; delta_x = -0.5 ; delta_y = self.height_length
        translation_point = tuple([delta_x,delta_y]) ; transformed_point_set = set()
        for point_id in point_set:
            point = self.point_map_read(point_id)
            transformed_point = np.add(point, translation_point) # vector addition
            transformed_point = tuple(transformed_point)
            transformed_point_set.add(transformed_point)
            self.point_map_update(point_id, transformed_point)
        self.transformed_point_set = transformed_point_set

    def scale_linear_transformation(self): # this function inputs a scale factor and scales all coordinates by that factor.
        point_set = self.point_set ; scale_factor = self.S ; transformed_point_set = set()
        for point_id in point_set:
            point = self.point_map_read(point_id)
            point = np.array(point)
            transformed_point = scale_factor * point # scalar vector multiplication
            transformed_point = tuple(transformed_point)
            transformed_point_set.add(transformed_point)
            self.point_map_update(point_id, transformed_point)
        self.transformed_point_set = transformed_point_set

    def sequential_rotation_linear_transformation(self): # this function inputs a set of vectors and the angle to rotate them counterclockwise around the origin.
        point_set = self.point_set
        alpha_angle = self.alpha_angle
        k_sequence = self.k_sequence
        transformed_point_set = set()
        for k in k_sequence:
            angle = alpha_angle * k
            rotation_matrix = tuple([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
            for point_id in point_set:
                if point_id[2] == k:
                    point = self.point_map_read(point_id)
                    transformed_point = np.dot(rotation_matrix, point) # matrix-vector multiplication --> vector rotation
                    transformed_point = tuple(transformed_point)
                    transformed_point_set.add(transformed_point)
                    self.point_map_update(point_id, transformed_point)
        self.transformed_point_set = transformed_point_set

    '''PLOTTING'''
    def plot_origin_point(self): plt.plot(0, 0, "*", color = "green")

    def plot_point_set(self):
        graph = plt.subplot() ; point_map = self.point_map ; s = self.S
        x, y = zip(*point_map.values())
        for point in self.point_map.keys():
            i, j, k = point ; i2, j2 = self.point_map_read(point) ; shift = int()
            if i == 0: shift = 0.1
            elif j == 0: shift = -0.1
            else: shift = +0.1
            plt.text(i2 - 0.2*s, j2 + s*(-0.04 + shift), '({},{},{})'.format(i, j, k), fontsize = 6, fontweight = 1000)
        plt.plot(x, y, "o", color = "gray")
        graph.set_aspect("equal")
    
    def set_line_width(self, new_width): self.line_width = new_width

    def plot_crease_set(self):
        graph = plt.subplot()
        point_map = self.point_map
        x,y = zip(*point_map.values())
        crease_set = self.crease_set
        for crease in crease_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "blue")
        graph.set_aspect("equal")

    def plot_colored_crease_set(self): # if crease_is_invert is True, then switch between mountain/valley assignment.
        crease_is_invert = self.crease_is_invert ; line_width = self.line_width
        graph = plt.subplot() ; point_map = self.point_map
        x,y = zip(*point_map.values())
        edge_set = self.edge_set ; orthogonal_crease_set = self.orthogonal_crease_set ; diagonal_crease_set = self.diagonal_crease_set
        colors = ("black", "red", "blue") # Blue creases are mountain folds, Red creases are valley folds, Black lines are edges.
        for crease in edge_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "black", linewidth = line_width)
        for crease in orthogonal_crease_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "red" if not crease_is_invert else "blue", linewidth = line_width)
        for crease in diagonal_crease_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "blue" if not crease_is_invert else "red", linewidth = line_width)
        graph.set_aspect("equal")

    def plot_monochromatic_crease_set(self): # if crease_is_invert is True, then switch between mountain/valley assignment.
        crease_is_invert = self.crease_is_invert ; line_width = self.line_width
        graph = plt.subplot() ; point_map = self.point_map
        x,y = zip(*point_map.values())
        edge_set = self.edge_set ; orthogonal_crease_set = self.orthogonal_crease_set ; diagonal_crease_set = self.diagonal_crease_set
        colors = ("black", "red", "blue") # Blue creases are mountain folds, Red creases are valley folds, Black lines are edges.
        for crease in edge_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "black", linewidth = line_width*1.3)
        for crease in orthogonal_crease_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "black", linestyle = "dashed" if not crease_is_invert else "solid", linewidth = line_width)
        for crease in diagonal_crease_set:
            A,B = crease ; x,y = zip(*(self.point_map_read(A),self.point_map_read(B)))
            plt.plot(x, y, color = "black", linestyle = "solid" if not crease_is_invert else "dashed", linewidth = line_width)
        graph.set_aspect("equal")

    def plot_facet_set(self):
        graph = plt.subplot() ; facet_set = self.facet_set ; k_sequence = self.k_sequence
        '''plot central polygon'''
        polygon_point_coordinates = list()
        for k in k_sequence:
            polygon_point_coordinates.append(self.point_map_read((0,0,k)))
        x,y = zip(*polygon_point_coordinates)
        plt.fill(x, y, facecolor='yellow', edgecolor='white', linewidth=self.line_width)
        '''plot wedge facets'''
        for facet in facet_set:
            A,B,C = facet
            x,y = zip(*(self.point_map_read(A),self.point_map_read(B),self.point_map_read(C)))
            plt.fill(x, y, facecolor='lime', edgecolor='white', linewidth=self.line_width)
        graph.set_aspect("equal")

    def show_plot(self):
        #to export svg file, you can activate the line below and insert the directory you wish to save to:
        #plt.savefig("your/directory/filename.svg", dpi=300)
        plt.show()

    def export_to_fold(self, filename="bloom_yoshimura.fold"):
        import json
        import math
        
        # 1. Merge vertices using physical 2D coordinates to close the mesh
        vertex_index_map = {}
        vertices_coords = []
        
        # Tolerance check to zip overlapping wedges together
        def is_close(c1, c2, tol=1e-5):
            return math.isclose(c1[0], c2[0], abs_tol=tol) and math.isclose(c1[1], c2[1], abs_tol=tol)

        for point_id, coords in self.point_map.items():
            found_idx = -1
            for i, v_coords in enumerate(vertices_coords):
                if is_close(coords, v_coords):
                    found_idx = i
                    break
            
            # If coordinates are new, add them to our unique vertices list
            if found_idx == -1:
                found_idx = len(vertices_coords)
                vertices_coords.append([float(coords[0]), float(coords[1]), 0.0])
            
            vertex_index_map[point_id] = found_idx

        # 2. Map faces using merged indices
        faces_vertices = []
        for facet in self.facet_set:
            face_indices = []
            for pt in facet:
                idx = vertex_index_map[pt]
                if idx not in face_indices:
                    face_indices.append(idx)
            # Only add valid triangular faces (ignores collapsed/degenerate faces)
            if len(face_indices) == 3:
                faces_vertices.append(face_indices)
                
        # Dynamically reconstruct the central polygon face properly
        central_polygon = []
        for k in self.k_sequence:
            pt_id = (0, 0, k)
            if pt_id in vertex_index_map:
                idx = vertex_index_map[pt_id]
                # Avoid sequential duplicates if any exist
                if not central_polygon or central_polygon[-1] != idx:
                    central_polygon.append(idx)
                    
        # Check for wrap-around duplicate at the start/end
        if central_polygon and central_polygon[0] == central_polygon[-1]:
            central_polygon.pop()
            
        if len(central_polygon) > 2:
            faces_vertices.append(central_polygon)

        # 3. Process Edges 
        # We use a dictionary with sorted vertex tuples as keys to prevent duplicate lines
        edges_dict = {}
        
        orth_assign = "M" if self.crease_is_invert else "V"
        diag_assign = "V" if self.crease_is_invert else "M"
        
        def add_edge(edge_set, assignment):
            for crease in edge_set:
                pts = list(crease)
                v1, v2 = vertex_index_map[pts[0]], vertex_index_map[pts[1]]
                if v1 != v2:
                    # Sort indices to ensure the dictionary key is consistent regardless of direction
                    key = tuple(sorted((v1, v2)))
                    # Assign the edge. Boundaries override internal hinges if there is a conflict.
                    if key not in edges_dict or assignment == "B":
                        edges_dict[key] = assignment

        # Populate the edges in order so boundaries are prioritized correctly
        add_edge(self.diagonal_crease_set, diag_assign)
        add_edge(self.orthogonal_crease_set, orth_assign)
        add_edge(self.edge_set, "B")

        # 4. Construct JSON file structure
        edges_vertices = []
        edges_assignment = []
        for (v1, v2), assign in edges_dict.items():
            edges_vertices.append([v1, v2])
            edges_assignment.append(assign)

        fold_dict = {
            "file_spec": 1.1,
            "file_creator": "Bloom_Yoshimura Python Generator",
            "frame_title": f"Bloom Yoshimura M={self.M} H={self.H}",
            "frame_attributes": ["3D"],
            "frame_unit": "unit",
            "vertices_coords": vertices_coords,
            "faces_vertices": faces_vertices,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment
        }

        with open(filename, 'w') as f:
            json.dump(fold_dict, f, indent=2)
            
        print(f"Successfully exported {len(vertices_coords)} unique vertices and {len(edges_vertices)} edges to {filename}")



import json

def generate_miura_fold(filename = "miura.fold", cols=5, rows=5, dx=10.0, dy=10.0, tilt=4.0):
    """
    Generates a Miura-ori crease pattern in the .fold format.
    
    Parameters:
    - filename: The name of the file to save the fold pattern to
    - cols: Number of parallelogram columns
    - rows: Number of parallelogram rows
    - dx: Width of each cell
    - dy: Height of each cell
    - tilt: The horizontal shift applied to alternating rows to create the zigzag
    """
    
    fold_data = {
        "file_spec": 1.2,
        "file_creator": "Python Miura-ori Generator",
        "file_classes": ["singleModel"],
        "frame_title": "Miura-ori Tessellation",
        "frame_classes": ["creasePattern"],
        "frame_attributes": ["2D"],
        "vertices_coords": [],
        "edges_vertices": [],
        "edges_assignment": [],
        "faces_vertices": []
    }

    # 1. Generate Vertices
    # The grid alternates horizontally by 'tilt' on every odd row
    for j in range(rows + 1):
        for i in range(cols + 1):
            x = (i * dx) + (tilt if j % 2 == 1 else 0.0)
            y = j * dy
            fold_data["vertices_coords"].append([x, y])

    # Helper to get vertex index based on row and column
    def get_v_idx(i, j):
        return j * (cols + 1) + i

    # 2. Generate Horizontal Edges
    for j in range(rows):
        for i in range(cols + 1):
            v1 = get_v_idx(i, j)
            v2 = get_v_idx(i + 1, j)
            fold_data["edges_vertices"].append([v1, v2])

            # Edge Assignments: Boundaries are 'B', interior horizontal segments alternate M/V
            # by column index (i) so they flip at every vertex across the row
            if j == 0 or j == rows:
                fold_data["edges_assignment"].append("B")
            elif (i + j) % 2 == 1:
                fold_data["edges_assignment"].append("M")
            else:
                fold_data["edges_assignment"].append("V")

    # 3. Generate Vertical (Zigzag) Edges
    for j in range(rows +1):
        for i in range(cols):
            v1 = get_v_idx(i, j)
            v2 = get_v_idx(i, j + 1)
            fold_data["edges_vertices"].append([v1, v2])

            # Edge Assignments: Boundaries are 'B', interior zigzag lines alternate M/V
            # by column index (i) so each continuous zigzag line is one uniform type
            if i == 0 or i == cols:
                fold_data["edges_assignment"].append("B")
            elif i % 2 == 1:
                fold_data["edges_assignment"].append("V")
            else:
                fold_data["edges_assignment"].append("M")

    # 4. Generate Faces (Optional for standard CP, but helpful for 3D folded state viewers)
    for j in range(rows):
        for i in range(cols):
            v1 = get_v_idx(i, j)           # Bottom-left (in grid terms)
            v2 = get_v_idx(i + 1, j)       # Bottom-right
            v3 = get_v_idx(i + 1, j + 1)   # Top-right
            v4 = get_v_idx(i, j + 1)       # Top-left
            # CCW order for standard face definitions
            fold_data["faces_vertices"].append([v1, v2, v3, v4])

    return fold_data
