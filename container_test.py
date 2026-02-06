from origami_interpreter.OrigamiContainer import OrigamiContainer as oc

def main():
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

    oc1 = oc(coords=coords, panels=indices, name="myPatternName")

    print("OrigamiContainer successfully created from native python representation.")
    print("Object: ", oc1)
    print("Python Representation: ", oc1.get_pyrepr())
    print("Name: ", oc1._origami_name)

    oc2 = oc(origami_filepath="C:\\Users\\thetk\\Documents\\BYU\\CMR Labs\\Origami_Sensitivity_Analysis\\Origami_Sensitivity_Analysis\\pattern_files\\sample-color-svgrepo-com.svg")

    print("\nOrigamiContainer successfully created from native python representation.")
    print("Object: ", oc2)
    print("Python Representation: ", oc2.get_pyrepr())
    print("Name: ", oc2._origami_name)

if __name__ == "__main__":
    main()