import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem

def count_eigenvalues(system, tol=1e-9):
    C = system.build_constraint_matrix()
    if C.shape[0] == 0:
        return [], 12
    K = C.T @ C
    eigs = np.linalg.eigvalsh(K)
    return sorted(eigs[eigs > tol]), int(np.sum(eigs <= tol))

# ── Shared geometry ────────────────────────────────────────────────────
t = 0.1

panel_A = RigidPanel(0, vertices=np.array([
    [0., 0., 0.], [1., 0., 0.],
    [1., 1., 0.], [0., 1., 0.]]), thickness=t)

panel_B = RigidPanel(1, vertices=np.array([
    [1., 0., 0.], [2., 0., 0.],
    [2., 1., 0.], [1., 1., 0.]]), thickness=t)

face_normal = np.array([1., 0., 0.])

# Three contact locations distributed in Y and Z on the mating face
p1 = np.array([1., 0.2,  0.0])
p2 = np.array([1., 0.8,  0.0])
p3 = np.array([1., 0.5, -t  ])

c45 = np.cos(np.pi / 4)  # 1/√2

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: GROOVE NORMAL VECTOR GEOMETRY
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Groove normal vector geometry")
print("=" * 60)

# ── Test 1a: Exact normal values at θ=0 ───────────────────────────────
# At θ=0, face_normal=[1,0,0]:
#   in_plane_Z = [0,0,1],  in_plane_Y = [0,1,0]
#   bisector   = [0,0,1]
#   perp = cross([1,0,0], [0,0,1]) = [0,-1,0]
#   n1 = c45*[0,0,1] + c45*[0,-1,0] = [0, -0.707, 0.707]
#   n2 = c45*[0,0,1] - c45*[0,-1,0] = [0,  0.707, 0.707]
print("\nTest 1a: Exact normal values at θ=0, face_normal=[1,0,0]")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.0)
assert np.allclose(g.n1, [0, -c45,  c45], atol=1e-10), f"n1 wrong: {g.n1}"
assert np.allclose(g.n2, [0,  c45,  c45], atol=1e-10), f"n2 wrong: {g.n2}"
print(f"  n1 = {np.round(g.n1, 4)}  (expect [0, -0.7071,  0.7071]) ✓")
print(f"  n2 = {np.round(g.n2, 4)}  (expect [0,  0.7071,  0.7071]) ✓")

# ── Test 1b: Exact normal values at θ=π/2 ─────────────────────────────
# At θ=π/2:
#   bisector = [0,1,0]
#   perp = cross([1,0,0], [0,1,0]) = [0,0,1]
#   n1 = c45*[0,1,0] + c45*[0,0,1] = [0,  0.707,  0.707]
#   n2 = c45*[0,1,0] - c45*[0,0,1] = [0,  0.707, -0.707]
print("\nTest 1b: Exact normal values at θ=π/2  (would crash old code)")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=np.pi/2)
assert np.allclose(g.n1, [0,  c45,  c45], atol=1e-10), f"n1 wrong: {g.n1}"
assert np.allclose(g.n2, [0,  c45, -c45], atol=1e-10), f"n2 wrong: {g.n2}"
print(f"  n1 = {np.round(g.n1, 4)}  (expect [0,  0.7071,  0.7071]) ✓")
print(f"  n2 = {np.round(g.n2, 4)}  (expect [0,  0.7071, -0.7071]) ✓")

# ── Test 1c: Properties hold for all theta ────────────────────────────
print("\nTest 1c: n1 ⊥ n2, unit length, both ⊥ face_normal — for 50 random θ")
for _ in range(50):
    theta = np.random.uniform(0, 2 * np.pi)
    g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=theta)
    assert abs(np.linalg.norm(g.n1) - 1.0) < 1e-10, f"|n1| ≠ 1 at θ={theta:.3f}"
    assert abs(np.linalg.norm(g.n2) - 1.0) < 1e-10, f"|n2| ≠ 1 at θ={theta:.3f}"
    assert abs(np.dot(g.n1, g.n2))          < 1e-10, f"n1·n2 ≠ 0 at θ={theta:.3f}"
    assert abs(np.dot(g.n1, face_normal))   < 1e-10, f"n1·fn ≠ 0 at θ={theta:.3f}"
    assert abs(np.dot(g.n2, face_normal))   < 1e-10, f"n2·fn ≠ 0 at θ={theta:.3f}"
print("  All 50 passed ✓")

# ── Test 1d: Properties hold for tilted face normals ──────────────────
print("\nTest 1d: Same properties for 8 different face normal orientations")
for angle_deg in range(0, 180, 22):
    a = np.radians(angle_deg)
    fn = np.array([np.cos(a), np.sin(a), 0.])
    for theta in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
        g = KinematicCoupling(panel_A, panel_B, p1, fn, theta=theta)
        assert abs(np.linalg.norm(g.n1) - 1.0) < 1e-10
        assert abs(np.linalg.norm(g.n2) - 1.0) < 1e-10
        assert abs(np.dot(g.n1, g.n2))          < 1e-10
        assert abs(np.dot(g.n1, fn))            < 1e-10
        assert abs(np.dot(g.n2, fn))            < 1e-10
print("  All passed for 8 face normals × 5 theta values ✓")

# ── Test 1e: set_theta updates normals correctly ───────────────────────
print("\nTest 1e: set_theta updates normals and is reversible")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.0)
n1_original = g.n1.copy()
g.set_theta(np.pi / 2)
assert not np.allclose(g.n1, n1_original), "n1 unchanged after set_theta"
g.set_theta(0.0)
assert np.allclose(g.n1, n1_original, atol=1e-10), "n1 didn't return to original"
print("  ✓")

# ── Test 1f: Two rows per groove are always independent ────────────────
print("\nTest 1f: n1 and n2 constraint rows are independent for all theta")
for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
    g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=theta)
    r1, r2 = g.get_constraint_rows(12)
    rank = np.linalg.matrix_rank(np.vstack([r1, r2]), tol=1e-10)
    assert rank == 2, f"Rows not independent at θ={theta:.3f} (rank={rank})"
print("  Independent at all 20 theta values ✓")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: CONSTRAINT MATRIX STRUCTURE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Constraint matrix structure")
print("=" * 60)

print("\nTest 2a: 1 groove → C shape (2, 12)")
system = CouplingSystem([panel_A, panel_B])
system.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.))
C = system.build_constraint_matrix()
assert C.shape == (2, 12), f"Expected (2,12), got {C.shape}"
print(f"  {C.shape} ✓")

print("\nTest 2b: 3 grooves → C shape (6, 12)")
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p, face_normal, theta=0.))
C = system.build_constraint_matrix()
assert C.shape == (6, 12), f"Expected (6,12), got {C.shape}"
print(f"  {C.shape} ✓")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: EIGENVALUE COUNTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Eigenvalue counts")
print("=" * 60)

print("\nTest 3a: 1 groove → 2 nonzero eigenvalues, 10 zero")
system = CouplingSystem([panel_A, panel_B])
system.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) == 2 and n_zero == 10
print(f"  Nonzero: {len(nonzero)}  Zero: {n_zero} ✓")

print("\nTest 3b: 3 grooves (non-collinear) → 6 nonzero, 6 zero (only global RBM)")
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p, face_normal, theta=0.))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) == 6 and n_zero == 6
print(f"  Nonzero: {len(nonzero)}  Zero: {n_zero} ✓")
print(f"  λ values: {[f'{e:.4f}' for e in nonzero]}")

print("\nTest 3c: 3 grooves at same point → max rank 3 from Φ, so ≤ 3 nonzero")
system = CouplingSystem([panel_A, panel_B])
for _ in range(3):
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal,
                                          theta=_ * np.pi / 4))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) <= 3, f"Expected ≤ 3, got {len(nonzero)}"
print(f"  Nonzero: {len(nonzero)}  (expect ≤ 3) ✓")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: PHYSICAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Physical robustness")
print("=" * 60)

print("\nTest 4a: θ rotation changes λ_min — optimization has something to tune")
lambda_mins = []
for theta in np.linspace(0, np.pi / 2, 6):
    system = CouplingSystem([panel_A, panel_B])
    for p in [p1, p2, p3]:
        system.add_coupling(KinematicCoupling(panel_A, panel_B, p,
                                              face_normal, theta=theta))
    nonzero, _ = count_eigenvalues(system)
    lam = min(nonzero) if nonzero else 0.
    lambda_mins.append(lam)
    print(f"  θ={theta:.3f} rad → λ_min={lam:.5f}")
assert not all(abs(l - lambda_mins[0]) < 1e-10 for l in lambda_mins), \
    "λ_min identical across all θ — groove orientation has no effect"
print("  λ_min varies with θ ✓")

print("\nTest 4b: Tilted mating face (45°) → same 6/6 split")
fn_tilted = np.array([np.cos(np.radians(45)), np.sin(np.radians(45)), 0.])
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p,
                                          fn_tilted, theta=0.))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) == 6 and n_zero == 6
print(f"  Nonzero: {len(nonzero)}  Zero: {n_zero} ✓")

print("\nTest 4c: Multiple face normal orientations all give valid grooves")
for angle_deg in [0, 30, 45, 60, 90, 120, 135, 150]:
    a = np.radians(angle_deg)
    fn = np.array([np.cos(a), np.sin(a), 0.])
    system = CouplingSystem([panel_A, panel_B])
    for p in [p1, p2, p3]:
        system.add_coupling(KinematicCoupling(panel_A, panel_B, p,
                                              fn, theta=0.))
    nonzero, n_zero = count_eigenvalues(system)
    assert len(nonzero) == 6 and n_zero == 6, \
        f"Failed at face angle {angle_deg}°: {len(nonzero)} nonzero"
print("  All 8 face orientations give 6 nonzero eigenvalues ✓")

print("\n✓ All tests passed.")