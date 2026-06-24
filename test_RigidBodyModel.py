import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem

def count_nonzero_eigenvalues(system, tol=1e-9):
    """Returns eigenvalues of K = C^T C, split into zero and nonzero."""
    C = system.build_constraint_matrix()
    if C.shape[0] == 0:
        K = np.zeros((system.total_dofs, system.total_dofs))
    else:
        K = C.T @ C
    eigs = np.linalg.eigvalsh(K)
    nonzero = eigs[eigs > tol]
    zero    = eigs[eigs <= tol]
    return sorted(nonzero), len(zero)

# ── Geometry: two unit squares side by side in XY plane ──────────────
#
#   Panel A          Panel B
#   (0,0)-(1,1)      (1,0)-(2,1)
#
panel_A = RigidPanel(0, vertices=np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]))

panel_B = RigidPanel(1, vertices=np.array([
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
]))

# ── TEST 1: No couplings ──────────────────────────────────────────────
print("=" * 55)
print("TEST 1: No couplings")
system = CouplingSystem([panel_A, panel_B])
nonzero_eigs, n_zero = count_nonzero_eigenvalues(system)

print(f"  Total DOFs:           {system.total_dofs}  (expect 12)")
print(f"  Zero eigenvalues:     {n_zero}             (expect 12)")
print(f"  Nonzero eigenvalues:  {len(nonzero_eigs)}             (expect 0)")
print(f"  Rigidity eigenvalue:  {system.get_rigidity_eigenvalue():.6f}  (expect 0)")
assert system.total_dofs == 12
assert n_zero == 12
assert len(nonzero_eigs) == 0

# ── TEST 2: One Z-contact at panel centroid region ────────────────────
print("\nTEST 2: One Z-direction contact")
system = CouplingSystem([panel_A, panel_B])
c1 = KinematicCoupling(
    panel_A  = panel_A,
    panel_B  = panel_B,
    point    = np.array([1.0, 0.25, 0.0]),  # on shared edge region
    normal   = np.array([0.0, 0.0, 1.0]),   # Z direction (piston)
)
system.add_coupling(c1)
nonzero_eigs, n_zero = count_nonzero_eigenvalues(system)

print(f"  Zero eigenvalues:     {n_zero}             (expect 11)")
print(f"  Nonzero eigenvalues:  {len(nonzero_eigs)}             (expect 1)")
assert len(nonzero_eigs) == 1
assert n_zero == 11

# ── TEST 3: Three Z-contacts (triangle) ──────────────────────────────
print("\nTEST 3: Three Z-contacts in a triangle")
system = CouplingSystem([panel_A, panel_B])
contacts = [
    KinematicCoupling(panel_A, panel_B,
        point=np.array([1.0, 0.2, 0.0]),
        normal=np.array([0.0, 0.0, 1.0])),
    KinematicCoupling(panel_A, panel_B,
        point=np.array([1.0, 0.8, 0.0]),
        normal=np.array([0.0, 0.0, 1.0])),
    KinematicCoupling(panel_A, panel_B,
        point=np.array([0.5, 0.5, 0.0]),  # offset in X for non-collinearity
        normal=np.array([0.0, 0.0, 1.0])),
]
for c in contacts:
    system.add_coupling(c)
nonzero_eigs, n_zero = count_nonzero_eigenvalues(system)

print(f"  Zero eigenvalues:     {n_zero}             (expect 9)")
print(f"  Nonzero eigenvalues:  {len(nonzero_eigs)}             (expect 3)")
print(f"  Nonzero eig values:   {[f'{e:.4f}' for e in nonzero_eigs]}")
assert len(nonzero_eigs) == 3
assert n_zero == 9

# ── TEST 4: Collinear contacts should NOT give 3 independent constraints
print("\nTEST 4: Three COLLINEAR Z-contacts (degenerate — expect rank 2)")
system = CouplingSystem([panel_A, panel_B])
collinear = [
    KinematicCoupling(panel_A, panel_B,
        point=np.array([1.0, 0.2, 0.0]),
        normal=np.array([0.0, 0.0, 1.0])),
    KinematicCoupling(panel_A, panel_B,
        point=np.array([1.0, 0.5, 0.0]),
        normal=np.array([0.0, 0.0, 1.0])),
    KinematicCoupling(panel_A, panel_B,
        point=np.array([1.0, 0.8, 0.0]),
        normal=np.array([0.0, 0.0, 1.0])),  # all at x=1.0 — collinear!
]
for c in collinear:
    system.add_coupling(c)
nonzero_eigs, n_zero = count_nonzero_eigenvalues(system)

print(f"  Zero eigenvalues:     {n_zero}             (expect 10, not 9)")
print(f"  Nonzero eigenvalues:  {len(nonzero_eigs)}             (expect 2, not 3)")
assert len(nonzero_eigs) == 2, \
    "Collinear contacts should only provide 2 independent constraints"

print("\n✓ All tests passed.")