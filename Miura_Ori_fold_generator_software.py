# Generating a Miura-ori crease pattern in the .fold format. Don't edit the function, scroll down to the bottom of the file to see how to use it.
import json

# Function to generate a Miura-ori crease pattern (don't edit)
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
    for j in range(rows + 1):
        for i in range(cols):
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
    for j in range(rows):
        for i in range(cols + 1):
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

# EDIT BELOW
# Change line below to change the output filename
filename = "my_miura_ori.fold"
# Change the parameters below to change the crease pattern dimensions
cols = 5      # Number of parallelogram columns
rows = 5      # Number of parallelogram rows
dx = 20.0     # Width of each cell
dy = 20.0     # Height of each cell
tilt = 10.0   # The horizontal shift applied to alternating rows to create the zigzag

fold_data = generate_miura_fold(cols=cols, rows=rows, dx=dx, dy=dy, tilt=tilt)
with open(filename, "w") as f:   
    json.dump(fold_data, f, indent=2)