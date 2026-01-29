import numpy as np
from math import cos, sin, radians, pi
from source.SensitivityAnalysis import SensitivityModel



if __name__ == "__main__":

    # =========================
    # NODE COORDINATES (ALL FLAT)
    # =========================
    coords = [
        [0.0, 0.0, 0.0],    # 0  center

        [1.0, 0.0, 0.0],    # 1
        [0.5, 0.87, 0.0],   # 2
        [-0.5, 0.87, 0.0],  # 3
        [-1.0, 0.0, 0.0],   # 4
        [-0.5, -0.87, 0.0], # 5
        [0.5, -0.87, 0.0],  # 6

        # # Outer ring
        [2.0, 0.0, 0.0],    # 7
        [1.0, 1.73, 0.0],   # 8
        [-1.0, 1.73, 0.0],  # 9
        [-2.0, 0.0, 0.0],   # 10
        [-1.0, -1.73, 0.0], # 11
        [1.0, -1.73, 0.0],  # 12
    ]

    # =========================
    # PANEL DEFINITIONS
    # =========================
    indices = [

        # ---- Inner star (6 triangles) ----
        [ 1, 2,0],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],

        # # ---- Outer ring (6 quads) ----
        [1, 7, 8, 2],
        [2, 8, 9, 3],
        [3, 9, 10, 4],
        [4, 10, 11, 5],
        [5, 11, 12, 6],
        [6, 12, 7, 1],
    ]

    model = SensitivityModel(coords, indices)

    print("\n==== FLAT COMPLEX ORIGAMI PATCH ====")
    print(f"Nodes:   {len(model.nodes)}")
    print(f"Panels:  {len(model.panels)}")
    print(f"Bars:    {len(model.bars)}")
    print(f"Hinges:  {len(model.hinges)} ")

    sensitivity_results = model.analyze_sensitivity()

   
    combined_sens = model.analyze_sensitivity(num_modes_to_check=5, return_mode_index=[8])

    # 2. Plot it
    model.plot_pattern(
            sensitivity_vector=combined_sens,
            show_node_labels=False,   # Turn OFF node IDs for a cleaner look
            show_hinge_labels=True,    # Keep Hinge IDs ON
            title=" Modes 9 (Breathing Mode)"
        )