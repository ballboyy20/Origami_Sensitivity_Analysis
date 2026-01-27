import numpy as np
from math import cos, sin, radians, pi
from source.SensitivityAnalysis import SensitivityModel

def generate_flasher_geometry(inner_radius=1.0, outer_radius=3.0, twist_angle=0.0):
    """
    Generates coordinates and panels for a hexagonal flasher pattern.
    
    Args:
        inner_radius (float): Radius of the central gray hexagon.
        outer_radius (float): Approximate radius of the outer boundary.
        twist_angle (float): Rotation of the outer ring relative to inner (0 for flat state).
    """
    coords = []
    
    # --- 1. Generate Inner Hexagon Nodes (0-5) ---
    # Aligned with 30, 90, 150... degrees to have flat top/bottom edges
    for i in range(6):
        theta = radians(30 + i * 60)
        coords.append([
            inner_radius * cos(theta), 
            inner_radius * sin(theta), 
            0.0
        ])

    # --- 2. Generate Outer Star Nodes (6-17) ---
    # The outer boundary in the image is jagged (12 vertices).
    # We alternate between "Tips" (center of panels) and "Valleys" (fold lines).
    
    # Tip Radius (The furthest points)
    r_tip = outer_radius
    # Valley Radius (The inner 'kinks' of the outer boundary)
    r_valley = outer_radius * 0.866 # Slightly pulled in
    
    for i in range(6):
        # -- Node A: The "Valley" (Aligned with Inner Vertex) --
        # This connects to the corners of the inner hex (Gusset lines)
        theta_v = radians(30 + i * 60) + radians(twist_angle)
        coords.append([
            r_valley * cos(theta_v),
            r_valley * sin(theta_v),
            0.0
        ])
        
        # -- Node B: The "Tip" (Aligned with Inner Edge centers) --
        # This forms the outer edge of the trapezoidal panels
        theta_t = radians(60 + i * 60) + radians(twist_angle)
        coords.append([
            r_tip * cos(theta_t),
            r_tip * sin(theta_t),
            0.0
        ])

    # --- 3. Generate Panels ---
    indices = []
    
    # -- A. Central Hub (Optional, usually rigid or hole) --
    indices.append([0, 1, 2, 3, 4, 5]) 

    # -- B. Trapezoidal Spokes (The main panels) --
    # These connect an inner edge to an outer tip.
    # Pattern: [Inner_i, Inner_i+1, Outer_Tip, Outer_Valley?]
    # Let's verify connectivity based on the image:
    # The trapezoid connects Inner(i), Inner(i+1) to Outer(2*i+1), Outer(2*i+2) roughly.
    
    for i in range(6):
        # Inner indices (wrapping around)
        in_a = i
        in_b = (i + 1) % 6
        
        # Outer indices
        # The outer ring has 12 nodes. 
        # Node 6 is a Valley aligned with Inner Node 0.
        # Node 7 is a Tip aligned with Edge 0-1.
        # Node 8 is a Valley aligned with Inner Node 1.
        
        # Outer nodes corresponding to this sector
        out_tip = 6 + (2 * i) + 1  # The tip (7, 9, 11...)
        out_val_left = 6 + (2 * i) # The valley to the left (6, 8, 10...)
        out_val_right = 6 + ((2 * i + 2) % 12) # The valley to the right
        
        # Panel 1: The Trapezoid (Spoke)
        # Connects Edge(in_a, in_b) to the Outer Tip
        # Note: To match the image, this is usually 2 triangles or 1 quad.
        # Let's make it a Quad: [Inner A, Inner B, Outer Valley B, Outer Valley A]? 
        # No, that skips the tip.
        
        # Looking at Image (b): It's composed of Quads.
        # The Quad is: [Inner A, Inner B, Outer Valley Right, Outer Valley Left?] No.
        
        # Let's try: [Inner A, Inner B, Outer Tip] -> Triangle?
        # Let's construct it as:
        # 1. Gusset Triangle (at the vertex)
        # 2. Main Panel (Trapezoid)
        
        # Correct Topology for Flasher:
        # Gusset: [Inner A, Outer Valley Left, Outer Tip Left??]
        
        # Let's keep it simple and robust for your solver:
        # 1 Quad per side (The Spoke):
        indices.append([in_a, in_b, out_val_right, out_tip, out_val_left]) # 5-node panel?
        # Actually, let's split it into standard Flasher triangles to be safe.
        
        # Triangle 1 (Left part of spoke): [Inner A, Outer Valley Left, Outer Tip]
        indices.append([in_a, out_val_left, out_tip])
        
        # Triangle 2 (Right part of spoke): [Inner B, Outer Tip, Outer Valley Right]
        indices.append([in_b, out_tip, out_val_right])
        
        # Triangle 3 (The Gusset between spokes - optional if covered above)
        # The "Gusset" is usually the fold between spokes.
        # In this topology, Inner B connects to Outer Valley Right in both sectors, so it's closed.

    return coords, indices

if __name__ == "__main__":
    # Generate the pattern
    coords, indices = generate_flasher_geometry(inner_radius=1.0, outer_radius=2.5)
    
    model = SensitivityModel(coords, indices)

    print("\n==== PARAMETRIC FLASHER GENERATED ====")
    print(f"Nodes:   {len(model.nodes)} (Expected 18)")
    print(f"Panels:  {len(model.panels)}")
    print(f"Bars:    {len(model.bars)}")
    print(f"Hinges:  {len(model.hinges)}")

    model.plot_pattern()