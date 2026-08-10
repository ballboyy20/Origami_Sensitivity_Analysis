import numpy as np

class RigidPanel:
    def __init__(self, panel_id, vertices, thickness=0.0):
        """
        vertices: top-face corners (z = top of panel)
        thickness: panel depth in Z
        centroid: center of the 3D box, at z = top_z - t/2
        """
        self.id = panel_id
        self.vertices = np.array(vertices)
        self.thickness = thickness
        
        face_centroid = self.vertices.mean(axis=0)
        self.centroid = face_centroid.copy()
        self.centroid[2] = face_centroid[2] - thickness / 2.0
        
        self.dof_start = panel_id * 6
    
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
    """
    One sphere-in-V-groove contact between two panels.
    
    The groove is cut into the mating face. Its orientation is defined
    by theta: the angle of the groove bisector measured from the +Y axis,
    rotating about the face normal (X-axis for a face at x=1).
    
    The V-angle is fixed at 90 degrees (each face is 45 deg from bisector).
    This produces two orthogonal normals n1, n2 and two constraint rows.
    """
    
    HALF_ANGLE = np.pi / 4  # 45 degrees — fixed 90 deg V-groove
    
    def __init__(self, panel_A, panel_B, point, face_normal, theta=0.0):
        """
        panel_A:     RigidPanel — panel carrying the sphere
        panel_B:     RigidPanel — panel carrying the groove
        point:       (3,) contact point on the mating face
        face_normal: (3,) unit vector normal to the mating face
                     (points from panel_B toward panel_A)
        theta:       groove rotation angle (radians) about face_normal,
                     measured from +Z at theta=0
        """
        self.panel_A    = panel_A
        self.panel_B    = panel_B
        self.point      = np.array(point,       dtype=float)
        self.face_normal = np.array(face_normal, dtype=float)
        self.face_normal /= np.linalg.norm(self.face_normal)
        self.theta      = theta

        self.active = True   # set False to exclude from constraint matrix (not yet implemnted)
        
        self._compute_groove_normals()
    
    def _compute_groove_normals(self):
        """
        Build n1 and n2 from theta.

        The groove is a V-shaped channel cut into the mating face, as in a
        Kelvin/Maxwell kinematic coupling: the two walls open toward
        face_normal (so contact captures relative motion along the face
        normal) and splay ± HALF_ANGLE from it in the transverse (w)
        direction. The groove's own length axis u — the one direction each
        wall's normal stays perpendicular to, and therefore the only
        direction sliding remains free — lies in the mating-face plane and
        is rotated by theta, measured from +Z at theta=0.

        Since mating faces are always orthogonal to the XY plane,
        face_normal always lies in XY, so Z is a stable in-plane reference.
        """

        in_plane_Z = np.array([0., 0., 1.])
        in_plane_Y = np.cross(in_plane_Z, self.face_normal)
        in_plane_Y /= np.linalg.norm(in_plane_Y)

        u = (np.cos(self.theta) * in_plane_Z +
             np.sin(self.theta) * in_plane_Y)
        u /= np.linalg.norm(u)

        # w is transverse to u, in the mating-face plane
        w = np.cross(self.face_normal, u)
        w /= np.linalg.norm(w)

        c = np.cos(self.HALF_ANGLE)
        s = np.sin(self.HALF_ANGLE)

        self.u  = u
        self.w  = w
        self.n1 = c * self.face_normal + s * w
        self.n2 = c * self.face_normal - s * w
        self.n1 /= np.linalg.norm(self.n1)
        self.n2 /= np.linalg.norm(self.n2)
    
    def get_constraint_rows(self, total_dofs):
        """
        Returns two constraint rows — one per groove face.
        Each row is shape (total_dofs,).
        """
        Phi_A = self.panel_A.get_interpolation_matrix(self.point, total_dofs)
        Phi_B = self.panel_B.get_interpolation_matrix(self.point, total_dofs)
        delta_Phi = Phi_A - Phi_B
        
        row1 = self.n1 @ delta_Phi
        row2 = self.n2 @ delta_Phi
        
        return row1, row2
    
    def set_theta(self, theta):
        """Update groove orientation and recompute normals."""
        self.theta = theta
        self._compute_groove_normals()


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
        
        rows = []
        for c in self.couplings:
            if getattr(c, 'active', True):          # ← respect active flag
                r1, r2 = c.get_constraint_rows(self.total_dofs)
                rows.append(r1)
                rows.append(r2)
        return np.zeros((0, self.total_dofs)) if not rows else np.array(rows) 
    
    def get_rigidity_eigenvalue(self):
        C = self.build_constraint_matrix()
        K = C.T @ C
        eigenvalues = np.linalg.eigvalsh(K)
        nonzero = eigenvalues[eigenvalues > 1e-10]
        return np.min(nonzero) if len(nonzero) > 0 else 0.0