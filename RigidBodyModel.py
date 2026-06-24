import numpy as np

class RigidPanel:
    """One rigid panel. 6 DOFs. No nodal discretization."""
    def __init__(self, panel_id, vertices):
        self.id = panel_id
        self.vertices = np.array(vertices)
        self.centroid = self.vertices.mean(axis=0)
        self.dof_start = panel_id * 6  # index into q vector
    
    def get_interpolation_matrix(self, p, total_dofs):
        """3 x total_dofs matrix Phi(p)."""
        r = p - self.centroid
        skew_r = np.array([
            [ 0,    -r[2],  r[1]],
            [ r[2],  0,    -r[0]],
            [-r[1],  r[0],  0   ]
        ])
        Phi = np.zeros((3, total_dofs))
        i = self.dof_start
        Phi[:, i:i+3] = np.eye(3)       # translation part
        Phi[:, i+3:i+6] = -skew_r       # rotation part
        return Phi


class KinematicCoupling:
    """One sphere-groove contact between two panels."""
    def __init__(self, panel_A, panel_B, point, normal):
        self.panel_A = panel_A
        self.panel_B = panel_B
        self.point = np.array(point)
        self.normal = np.array(normal) / np.linalg.norm(normal)
    
    def get_constraint_row(self, total_dofs):
        """One row of C."""
        Phi_A = self.panel_A.get_interpolation_matrix(
                    self.point, total_dofs)
        Phi_B = self.panel_B.get_interpolation_matrix(
                    self.point, total_dofs)
        return self.normal @ (Phi_A - Phi_B)


class CouplingSystem:
    """Full system of panels and couplings."""
    def __init__(self, panels):
        self.panels = panels
        self.couplings = []
        self.total_dofs = 6 * len(panels)
    
    def add_coupling(self, coupling):
        self.couplings.append(coupling)
    
    def build_constraint_matrix(self):
        if not self.couplings:
            return np.zeros((0, self.total_dofs))
        rows = [c.get_constraint_row(self.total_dofs) 
                for c in self.couplings]
        return np.array(rows)
    
    def get_rigidity_eigenvalue(self):
        C = self.build_constraint_matrix()
        K = C.T @ C
        eigenvalues = np.linalg.eigvalsh(K)
        nonzero = eigenvalues[eigenvalues > 1e-10]
        return np.min(nonzero) if len(nonzero) > 0 else 0.0