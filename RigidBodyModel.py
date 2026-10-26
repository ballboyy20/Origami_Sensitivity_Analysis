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
    One sphere contact between two panels, of one of three types:

      'flat'    — sphere on a flat:      1 normal (the face normal), 1 row.
      'vgroove' — sphere in a V-groove:  2 normals n1, n2, 2 rows.
      'tet'     — sphere in a tetrahedral (trihedral) socket: 3 normals,
                  evenly spaced 120 deg apart about the face normal, 3 rows.

    The groove is cut into the mating face. Its orientation is defined
    by theta: the angle of the groove axis u measured from the in-plane
    reference direction ref_dir, rotating about the face normal.

    The V-angle is fixed at 90 degrees (each wall is 45 deg from the face
    normal), and the tetrahedral socket's walls use the same 45 deg tilt,
    so every contact normal of every type makes the same angle with the
    face normal except the flat's. Each type carries an equal (unit)
    weight per row.

    Flats and tetrahedral sockets have no angle dependence in K: a flat
    has one fixed normal, and the socket's sum of n n^T over its three
    evenly spaced normals is the same for every theta. set_theta still
    rotates the socket's normals (so drawings show something definite),
    but K is unchanged.
    """

    HALF_ANGLE = np.pi / 4  # 45 degrees — fixed 90 deg V-groove
    TET_TILT   = np.pi / 4  # 45 degrees — socket walls, same as the groove's
    KINDS      = ('flat', 'vgroove', 'tet')
    N_ROWS     = {'flat': 1, 'vgroove': 2, 'tet': 3}

    def __init__(self, panel_A, panel_B, point, face_normal, theta=0.0,
                 kind='vgroove', ref_dir=None):
        """
        panel_A:     RigidPanel — panel carrying the sphere
        panel_B:     RigidPanel — panel carrying the groove
        point:       (3,) contact point on the mating face
        face_normal: (3,) unit vector normal to the mating face
                     (points from panel_B toward panel_A)
        theta:       groove rotation angle (radians) about face_normal,
                     measured from ref_dir at theta=0
        kind:        'flat', 'vgroove' or 'tet'
        ref_dir:     (3,) in-plane reference direction for theta = 0.
                     Only its component in the mating-face plane is used.
                     Defaults to +Z, which suits faces orthogonal to the
                     XY plane (every crease face so far); a face parallel
                     to the panel must supply its own.
        """
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.panel_A    = panel_A
        self.panel_B    = panel_B
        self.point      = np.array(point,       dtype=float)
        self.face_normal = np.array(face_normal, dtype=float)
        self.face_normal /= np.linalg.norm(self.face_normal)
        self.theta      = theta
        self.kind       = kind
        self.n_rows     = self.N_ROWS[kind]

        ref = np.array([0., 0., 1.]) if ref_dir is None else np.array(ref_dir, dtype=float)
        ref = ref - np.dot(ref, self.face_normal) * self.face_normal
        if np.linalg.norm(ref) < 1e-9:
            raise ValueError("ref_dir is parallel to face_normal; supply an "
                             "in-plane reference direction for this face")
        self.ref_dir = ref / np.linalg.norm(ref)

        self.active = True   # set False to exclude from constraint matrix

        self._compute_groove_normals()

    def _compute_groove_normals(self):
        """
        Build the contact normals from theta.

        The groove is a V-shaped channel cut into the mating face, as in a
        Kelvin/Maxwell kinematic coupling: the two walls open toward
        face_normal (so contact captures relative motion along the face
        normal) and splay ± HALF_ANGLE from it in the transverse (w)
        direction. The groove's own length axis u — the one direction each
        wall's normal stays perpendicular to, and therefore the only
        direction sliding remains free — lies in the mating-face plane and
        is rotated by theta, measured from ref_dir at theta=0.

        The in-plane frame is (e1, e2) = (ref_dir, ref_dir x face_normal).
        With the default ref_dir = +Z and face_normal in XY this is exactly
        the original (Z, Z x face_normal) frame, so V-groove normals are
        unchanged from before ref_dir existed.
        """

        e1 = self.ref_dir
        e2 = np.cross(e1, self.face_normal)
        e2 /= np.linalg.norm(e2)

        u = (np.cos(self.theta) * e1 +
             np.sin(self.theta) * e2)
        u /= np.linalg.norm(u)

        # w is transverse to u, in the mating-face plane
        w = np.cross(self.face_normal, u)
        w /= np.linalg.norm(w)

        self.u  = u
        self.w  = w

        if self.kind == 'flat':
            self.normals = [self.face_normal.copy()]
        elif self.kind == 'vgroove':
            c = np.cos(self.HALF_ANGLE)
            s = np.sin(self.HALF_ANGLE)
            self.n1 = c * self.face_normal + s * w
            self.n2 = c * self.face_normal - s * w
            self.n1 /= np.linalg.norm(self.n1)
            self.n2 /= np.linalg.norm(self.n2)
            self.normals = [self.n1, self.n2]
        else:  # 'tet'
            c = np.cos(self.TET_TILT)
            s = np.sin(self.TET_TILT)
            self.normals = []
            for phi in (0., 2 * np.pi / 3, 4 * np.pi / 3):
                n = c * self.face_normal + s * (np.cos(phi) * u + np.sin(phi) * w)
                self.normals.append(n / np.linalg.norm(n))

    def get_constraint_rows(self, total_dofs):
        """
        Returns one constraint row per contact normal (n_rows of them).
        Each row is shape (total_dofs,).
        """
        Phi_A = self.panel_A.get_interpolation_matrix(self.point, total_dofs)
        Phi_B = self.panel_B.get_interpolation_matrix(self.point, total_dofs)
        delta_Phi = Phi_A - Phi_B

        return tuple(n @ delta_Phi for n in self.normals)
    
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
    
    def build_constraint_matrix(self, length_scale=1.0):
        """
        length_scale: reference length used to nondimensionalize rotational
        DOFs against translational ones. Translation columns of C are unit
        normal components (dimensionless); rotation columns are r x n
        (physical length). Dividing the rotation columns by length_scale
        makes the relative weighting between "a rotation" and "a
        translation" in K = C^T C an explicit, user-chosen quantity instead
        of an accident of whatever units the panel geometry happens to use.
        This is an elementary column operation, so it cannot change rank(C).
        """
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")

        if not self.couplings:
            return np.zeros((0, self.total_dofs))

        rows = []
        for c in self.couplings:
            if getattr(c, 'active', True):          # ← respect active flag
                rows.extend(c.get_constraint_rows(self.total_dofs))
        C = np.zeros((0, self.total_dofs)) if not rows else np.array(rows)

        if length_scale != 1.0 and C.shape[0] > 0:
            for panel in self.panels:
                i = panel.dof_start
                C[:, i+3:i+6] /= length_scale

        return C

    def get_rigidity_eigenvalue(self, length_scale=1.0):
        C = self.build_constraint_matrix(length_scale=length_scale)
        K = C.T @ C
        eigenvalues = np.linalg.eigvalsh(K)
        nonzero = eigenvalues[eigenvalues > 1e-10]
        return np.min(nonzero) if len(nonzero) > 0 else 0.0