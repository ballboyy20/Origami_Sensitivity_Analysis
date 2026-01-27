from source.SensitivityAnalysis import SensitivityModel

if __name__ == "__main__":

    # =========================
    # NODE COORDINATES
    # =========================
    coords = [
    [0.0, 0.0, 0.0],  # 0 bottom-left
    [1.0, 0.0, 0.0],  # 1 bottom-mid
    [2.0, 0.0, 0.0],  # 2 bottom-right

    [0.0, 1.0, 0.0],  # 3 mid-left
    [1.0, 1.0, 0.0],  # 4 center
    [2.0, 1.0, 0.0],  # 5 mid-right

    [0.0, 2.0, 0.0],  # 6 top-left
    [1.0, 2.0, 0.0],  # 7 top-mid
    [2.0, 2.0, 0.0],  # 8 top-right
]
    # =========================
    # PANEL DEFINITIONS
    # =========================
    panel_indices = [
    # Bottom row
    [0, 1, 4,3],

    [1, 2, 5],
    [1, 5, 4],

    # Top row
    [3, 4, 7],
    [3, 7, 6],

    [4, 5, 8],
    [4, 8, 7],
]

    model = SensitivityModel(coords, panel_indices)

    print("\n==== COMPLEX ORIGAMI PATCH ====")
    print(f"Nodes:   {len(model.nodes)}")
    print(f"Panels:  {len(model.panels)}")
    print(f"Bars:    {len(model.bars)}")
    print(f"Hinges:  {len(model.hinges)}")
    
    model.analyze_sensitivity()
    model.plot_pattern()
    