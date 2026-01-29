import unittest
import numpy as np
# Assuming your main file is named SensitivityAnalysis.py
from SensitivityAnalysis import *
import sys
import os
from helper_classes import *

class TestBarPhysics(unittest.TestCase):
    
    def setUp(self):
        """This runs before every test to set up common objects."""
        self.n1 = Node(0, 0, 0, 0)
        self.n2 = Node(1, 10, 0, 0) # Bar 10 units long on X-axis
        self.bar = BarElement(self.n1, self.n2)

    def test_direction_cosines(self):
        """Verify the bar identifies its orientation correctly."""
        row = self.bar.get_compatibility_matrix_row(total_DOFs=6)
        # Check Node 1 (Tail) entries
        np.testing.assert_array_almost_equal(row[0:3], [-1.0, 0.0, 0.0])
        # Check Node 2 (Head) entries
        np.testing.assert_array_almost_equal(row[3:6], [1.0, 0.0, 0.0])

    def test_rigid_body_invariance(self):
        """CRITICAL: Moving the whole flasher shouldn't create 'fake' stretch."""
        row = self.bar.get_compatibility_matrix_row(6)
        # Move both nodes 5 units in Y (a translation, not a stretch)
        u_vector = np.array([0, 5, 0, 0, 5, 0])
        stretch = np.dot(row, u_vector)
        self.assertAlmostEqual(stretch, 0.0, places=12)

""" Help function for TestHingePhysics class"""
def wrap_angle_difference(delta):
    return (delta + np.pi) % (2*np.pi) - np.pi

class TestHingePhysics(unittest.TestCase):
    
    def setUp(self):
        # Create 4 nodes forming two triangles in the XY plane
        # IDs are 0, 1, 2, 3 so we can map them to the 12-DOF vector easily
        self.ni = Node(0, 0, 1, 0)   # Wing 1 (top)
        self.nj = Node(1, 0, 0, 0)   # Hinge Start
        self.nk = Node(2, 5, 0, 0)   # Hinge End
        self.nl = Node(3, 8, -1, 0)  # Wing 2 (bottom)
        
        # Initial flat hinge
        self.hinge = HingeElement(self.ni, self.nj, self.nk, self.nl)

    def test_flat_angle(self):
        """A flat assembly should return a 0 or pi dihedral angle."""
        angle = self.hinge.calculate_dihedral_angle()
        self.assertAlmostEqual(angle, 0.0, places=7)

    def test_right_angle_fold(self):
        """Rotate one wing to 90 degrees and verify the angle."""
        # Save original to reset later if needed
        original = self.ni.coordinates.copy()
        
        # Move node 'ni' to be on the Z-axis (90 deg fold)
        self.ni.coordinates = np.array([0, 0, 1]) 
        angle = self.hinge.calculate_dihedral_angle()
        
        # Expected: pi/2 (90 degrees)
        self.assertAlmostEqual(abs(angle), np.pi/2, places=7)
        
        # Reset for other tests
        self.ni.coordinates = original

    def test_hinge_invariance(self):
        """Translation of the hinge shouldn't change the dihedral angle."""
        initial_angle = self.hinge.calculate_dihedral_angle()
        
        # Shift everything by 10 units in Z
        # We perform a deep copy of coords to avoid messing up other tests
        original_coords = [n.coordinates.copy() for n in [self.ni, self.nj, self.nk, self.nl]]
        
        for node in [self.ni, self.nj, self.nk, self.nl]:
            node.coordinates += np.array([0, 0, 10])
            
        new_angle = self.hinge.calculate_dihedral_angle()
        self.assertAlmostEqual(initial_angle, new_angle, places=7)
        
        # Reset
        for i, node in enumerate([self.ni, self.nj, self.nk, self.nl]):
            node.coordinates = original_coords[i]

    def test_hinge_flip_sign(self):
        angle1 = self.hinge.calculate_dihedral_angle()

        # Swap i and l (mirror the hinge)
        self.hinge.node_i, self.hinge.node_l = self.hinge.node_l, self.hinge.node_i
        angle2 = self.hinge.calculate_dihedral_angle()

        self.assertAlmostEqual(angle1, -angle2, places=7)

    def test_jacobian_analytical_vs_numerical(self):
        """
        GOLDEN TEST: Verifies the analytical Jacobian gradients (1/h * n)
        against a brute-force Finite Difference calculation.
        """
        # 1. Setup
        total_dofs = 12  # 4 nodes * 3 DOFs
         
        numerical_J = np.zeros(total_dofs)
        
        # List of nodes to iterate over
        nodes = [self.ni, self.nj, self.nk, self.nl]
        
        # 2. Numerical Differentiation Loop
        for node in nodes:
            original_coords = node.coordinates.copy()

            epsilon = 1e-7 * max(1.0, np.linalg.norm(node.coordinates)) # makes epsilon large or small depending on hinge size
            
            for axis in range(3): # Loop x, y, z
                # --- Perturb Positive (+epsilon) ---
                node.coordinates[axis] = original_coords[axis] + epsilon
                angle_plus = self.hinge.calculate_dihedral_angle()
                
                # --- Perturb Negative (-epsilon) ---
                node.coordinates[axis] = original_coords[axis] - epsilon
                angle_minus = self.hinge.calculate_dihedral_angle()
                
                # --- Central Difference Formula ---
                # slope = (f(x+h) - f(x-h)) / 2h
                delta = wrap_angle_difference(angle_plus - angle_minus)
                gradient = delta / (2 * epsilon)
                
                # --- Store in Numerical Vector ---
                # Calculate index: NodeID * 3 + Axis(0,1,2)
                idx = node.id * 3 + axis
                numerical_J[idx] = gradient
                
                # Reset coordinate for next iteration
                node.coordinates[axis] = original_coords[axis]

        # 3. Get Analytical Jacobian (The fast math from the paper)
        analytical_J = self.hinge.get_jacobian_row(total_dofs)

        # 4. Compare
        # We use a slightly looser tolerance (atol=1e-5) because finite difference 
        # is an approximation, whereas the analytical method is exact.
        print("\nAnalytical Jacobian:")
        print(analytical_J)

        print("\nNumerical Jacobian:")
        print(numerical_J)

        print("\nDifference (A - N):")
        print(analytical_J - numerical_J)

        print("\nRelative Difference:")
        print((analytical_J - numerical_J) / (np.abs(numerical_J) + 1e-12))
        try:
            np.testing.assert_allclose(
                analytical_J,
                numerical_J,
                rtol=1e-4,
                atol=1e-5,
                err_msg="MISMATCH: Analytical vs Numerical Jacobian"
            )
        except AssertionError:
            print("\nAnalytical Jacobian:")
            print(analytical_J)

            print("\nNumerical Jacobian:")
            print(numerical_J)

            print("\nDifference (A - N):")
            print(analytical_J - numerical_J)

            print("\nRelative Difference:")
            print((analytical_J - numerical_J) / (np.abs(numerical_J) + 1e-12))

            raise

    def test_random_geometries(self):
        """
        FUZZ TEST: Generates random valid hinge geometries and verifies
        the Jacobian against the numerical method for every single one.
        """
        np.random.seed(42) # For reproducibility
        total_dofs = 12
        
        # Run 50 random test cases
        for i in range(50):
            # 1. Generate Random Coordinates
            # We add logic to ensure the hinge isn't degenerate (0 length or 0 area)
            coords = np.random.rand(4, 3) * 10  # 4 nodes, random values 0-10
            
            # Assign to nodes
            self.ni.coordinates = coords[0]
            self.nj.coordinates = coords[1]
            self.nk.coordinates = coords[2]
            self.nl.coordinates = coords[3]
            
            # Check if valid (not collinear) - Re-using your class's internal check would be better
            # but for a test, we can just skip if it returns a zero vector
            analytical_J = self.hinge.get_jacobian_row(total_dofs)
            if np.all(analytical_J == 0):
                continue # Skip degenerate random cases
            
            # 2. Run Numerical Gradient
            numerical_J = np.zeros(total_dofs)
            nodes = [self.ni, self.nj, self.nk, self.nl]
            
            for node in nodes:
                original_coords = node.coordinates.copy()
                epsilon = 1e-7 * max(1.0, np.linalg.norm(node.coordinates)) # makes epsilon large or small depending on hinge size
                for axis in range(3):
                    # Perturb +
                    node.coordinates[axis] = original_coords[axis] + epsilon
                    angle_plus = self.hinge.calculate_dihedral_angle()
                    
                    # Perturb -
                    node.coordinates[axis] = original_coords[axis] - epsilon
                    angle_minus = self.hinge.calculate_dihedral_angle()
                    
                    # Gradient
                    delta = wrap_angle_difference(angle_plus - angle_minus)
                    gradient = delta / (2 * epsilon)
                    
                    idx = node.id * 3 + axis
                    numerical_J[idx] = gradient
                    
                    # Reset
                    node.coordinates[axis] = original_coords[axis]
            
            # 3. Assert Match
            # We use a slightly larger tolerance for random geometries as
            # certain acute angles can amplify numerical noise.
            try:
                np.testing.assert_allclose(
                    analytical_J, 
                    numerical_J, 
                    rtol=1e-3, 
                    atol=1e-4
                )
            except AssertionError as e:
                print(f"\nFAILED on Random Iteration {i}")
                print(f"Coords:\n{coords}")
                raise e
            
class TestModelAssembly(unittest.TestCase):
    
    def test_shared_node_memory(self):
        """CRITICAL: Verify that if two panels share a node index, 
        they point to the EXACT same Node object in memory."""
        
        # Geometry: Two triangles sharing Node 1 and Node 2
        # P0: 0-1-2
        # P1: 1-2-3
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        indices = [[0,1,2], [1,2,3]]
        
        model = SensitivityModel(coords, indices)
        
        # Get Node 1 from Panel 0
        n1_p0 = next(n for n in model.panels[0].nodes if n.id == 1)
        # Get Node 1 from Panel 1
        n1_p1 = next(n for n in model.panels[1].nodes if n.id == 1)
        
        # Assert they are the same object
        self.assertIs(n1_p0, n1_p1, "Shared nodes must be the same object instance")

    def test_bar_generation_triangle(self):
        """Verify a simple triangle creates exactly 3 bars."""
        coords = [[0,0,0], [1,0,0], [0,1,0]]
        indices = [[0,1,2]]
        model = SensitivityModel(coords, indices)
        
        # A triangle has 3 edges -> 3 bars
        self.assertEqual(len(model.bars), 3)

    def test_bar_generation_quad_truss(self):
        """Verify your 'Fully Connected' logic: A 4-node panel should have 6 bars."""
        # Square: 4 perimeter bars + 2 diagonal bars = 6 bars
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        indices = [[0,1,2,3]] # One big panel
        model = SensitivityModel(coords, indices)
        
        self.assertEqual(len(model.bars), 6, "A 4-node panel should be triangulated into 6 bars")

    def test_duplicate_bar_removal(self):
        """Verify that shared edges do not create double bars."""
        # Two triangles sharing one edge.
        # Triangle 1 (3 bars) + Triangle 2 (3 bars) = 6 raw bars
        # Minus 1 duplicate shared bar = 5 unique bars expected.
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        indices = [[0,1,2], [1,2,3]] # Share edge 1-2
        model = SensitivityModel(coords, indices)
        
        self.assertEqual(len(model.bars), 5)

    def test_hinge_detection(self):
        """Verify the model correctly identifies a hinge and its axis."""
        # Two triangles sharing edge 1-2
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        indices = [[0,1,2], [1,2,3]]
        model = SensitivityModel(coords, indices)
        
        self.assertEqual(len(model.hinges), 1)
        
        h = model.hinges[0]
        # Check that the hinge axis is indeed nodes 1 and 2
        axis_ids = sorted([h.node_j.id, h.node_k.id])
        self.assertEqual(axis_ids, [1, 2], "Hinge axis should be on the shared nodes")

    def test_hinge_wing_nodes(self):
        """Verify the hinge correctly identifies the non-shared 'wing' nodes."""
        # P0: 0-1-2 (Wing is 0)
        # P1: 1-2-3 (Wing is 3)
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        indices = [[0,1,2], [1,2,3]]
        model = SensitivityModel(coords, indices)
        
        h = model.hinges[0]
        wings = sorted([h.node_i.id, h.node_l.id])
        self.assertEqual(wings, [0, 3], "Hinge wings should be the unique nodes")


class TestTopologyLogic(unittest.TestCase):

    def setUp(self):
        """
        Define a simple 4-node diamond shape for testing.
        Nodes 0 and 1 form the shared 'spine'.
        """
        self.coords = [
            [0.0, 0.0, 0.0],  # Node 0 (Spine Bottom)
            [0.0, 1.0, 0.0],  # Node 1 (Spine Top)
            [1.0, 0.5, 0.0],  # Node 2 (Right Wing)
            [-1.0, 0.5, 0.0], # Node 3 (Left Wing)
            [0.0, 0.5, 1.0]   # Node 4 (Vertical Fin - for failure test)
        ]

    def test_valid_topology(self):
        """
        Test that a normal 2-panel hinge does NOT raise an error.
        """
        print("\nRunning Valid Topology Test...")
        valid_indices = [
            [0, 1, 2], # Right Triangle
            [0, 1, 3]  # Left Triangle
        ]
        
        try:
            model = SensitivityModel(self.coords, valid_indices)
            print(f"Success! Created {len(model.hinges)} hinge(s) for 2 panels.")
        except ValueError as e:
            self.fail(f"Valid topology raised ValueError unexpectedly: {e}")

    def test_invalid_sandwich_topology(self):
        """
        Test that 3 panels sharing the same edge (0-1) RAISES a ValueError.
        This simulates the Hexagon+Triangle overlapping issue.
        """
        print("\nRunning Invalid (3-Panel) Topology Test...")
        
        sandwich_indices = [
            [0, 1, 2], # Panel A (Right)
            [0, 1, 3], # Panel B (Left)
            [0, 1, 4]  # Panel C (Sticking up) - All share edge 0-1
        ]

        # We assert that this MUST raise a ValueError
        with self.assertRaises(ValueError) as context:
            SensitivityModel(self.coords, sandwich_indices)
        
        print(f"Caught Expected Error: {context.exception}")

if __name__ == '__main__':
    unittest.main()