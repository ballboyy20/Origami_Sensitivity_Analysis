import numpy as np
import sys, os
import matplotlib.pyplot as plt                          # ← ADD
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem
from visualization_rigid import (                        # ← ADD
    figure_section1_groove_normals,
    figure_section2_heatmaps,
    figure_section3_eigenvalues,
    figure_section4_robustness,
)

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

p1 = np.array([1., 0.2,  0.0])
p2 = np.array([1., 0.8,  0.0])
p3 = np.array([1., 0.5, -t  ])

c45 = np.cos(np.pi / 4)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: GROOVE NORMAL VECTOR GEOMETRY
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Groove normal vector geometry")
print("=" * 60)

print("\nTest 1a: Exact normal values at θ=0, face_normal=[1,0,0]")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.0)
assert np.allclose(g.n1, [c45, -c45,  0], atol=1e-10), f"n1 wrong: {g.n1}"
assert np.allclose(g.n2, [c45,  c45,  0], atol=1e-10), f"n2 wrong: {g.n2}"
print(f"  n1 = {np.round(g.n1, 4)}  (expect [0.7071, -0.7071,  0]) ✓")
print(f"  n2 = {np.round(g.n2, 4)}  (expect [0.7071,  0.7071,  0]) ✓")

print("\nTest 1b: Exact normal values at θ=π/2  (would crash old code)")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=np.pi/2)
assert np.allclose(g.n1, [c45,  0,  c45], atol=1e-10), f"n1 wrong: {g.n1}"
assert np.allclose(g.n2, [c45,  0, -c45], atol=1e-10), f"n2 wrong: {g.n2}"
print(f"  n1 = {np.round(g.n1, 4)}  (expect [0.7071,  0,  0.7071]) ✓")
print(f"  n2 = {np.round(g.n2, 4)}  (expect [0.7071,  0, -0.7071]) ✓")

print("\nTest 1c: n1 ⊥ n2, unit length, both ⊥ slide axis u, both opening")
print("          toward face_normal by cos(45°) — for 50 random θ")
for _ in range(50):
    theta = np.random.uniform(0, 2 * np.pi)
    g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=theta)
    assert abs(np.linalg.norm(g.n1) - 1.0) < 1e-10, f"|n1| ≠ 1 at θ={theta:.3f}"
    assert abs(np.linalg.norm(g.n2) - 1.0) < 1e-10, f"|n2| ≠ 1 at θ={theta:.3f}"
    assert abs(np.dot(g.n1, g.n2))          < 1e-10, f"n1·n2 ≠ 0 at θ={theta:.3f}"
    assert abs(np.dot(g.n1, g.u))           < 1e-10, f"n1·u ≠ 0 at θ={theta:.3f}"
    assert abs(np.dot(g.n2, g.u))           < 1e-10, f"n2·u ≠ 0 at θ={theta:.3f}"
    assert abs(np.dot(g.n1, face_normal) - c45) < 1e-10, f"n1·fn ≠ cos45 at θ={theta:.3f}"
    assert abs(np.dot(g.n2, face_normal) - c45) < 1e-10, f"n2·fn ≠ cos45 at θ={theta:.3f}"
print("  All 50 passed ✓")

print("\nTest 1d: Same properties for 8 different face normal orientations")
for angle_deg in range(0, 180, 22):
    a = np.radians(angle_deg)
    fn = np.array([np.cos(a), np.sin(a), 0.])
    for theta in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
        g = KinematicCoupling(panel_A, panel_B, p1, fn, theta=theta)
        assert abs(np.linalg.norm(g.n1) - 1.0) < 1e-10
        assert abs(np.linalg.norm(g.n2) - 1.0) < 1e-10
        assert abs(np.dot(g.n1, g.n2))          < 1e-10
        assert abs(np.dot(g.n1, g.u))           < 1e-10
        assert abs(np.dot(g.n2, g.u))           < 1e-10
        assert abs(np.dot(g.n1, fn) - c45)      < 1e-10
        assert abs(np.dot(g.n2, fn) - c45)      < 1e-10
print("  All passed for 8 face normals × 5 theta values ✓")

print("\nTest 1e: set_theta updates normals and is reversible")
g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.0)
n1_original = g.n1.copy()
g.set_theta(np.pi / 2)
assert not np.allclose(g.n1, n1_original), "n1 unchanged after set_theta"
g.set_theta(0.0)
assert np.allclose(g.n1, n1_original, atol=1e-10), "n1 didn't return to original"
print("  ✓")

print("\nTest 1f: n1 and n2 constraint rows are independent for all theta")
for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
    g = KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=theta)
    r1, r2 = g.get_constraint_rows(12)
    rank = np.linalg.matrix_rank(np.vstack([r1, r2]), tol=1e-10)
    assert rank == 2, f"Rows not independent at θ={theta:.3f} (rank={rank})"
print("  Independent at all 20 theta values ✓")

# ── Section 1 figure ──────────────────────────────────────────────────
figure_section1_groove_normals(
    panel_A, panel_B, p1, face_normal, KinematicCoupling)
plt.show()

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
C_2a = C.copy()                                          # ← capture for figure

print("\nTest 2b: 3 grooves → C shape (6, 12)")
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p, face_normal, theta=0.))
C = system.build_constraint_matrix()
assert C.shape == (6, 12), f"Expected (6,12), got {C.shape}"
print(f"  {C.shape} ✓")
C_2b = C.copy()                                          # ← capture for figure

# ── Section 2 figure ──────────────────────────────────────────────────
figure_section2_heatmaps(C_2a, C_2b, n_panels=2)
plt.show()

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
system_3a = system                                       # ← capture for figure

print("\nTest 3b: 3 parallel grooves (same θ) → 5 nonzero, 7 zero")
print("  (Grooves now open toward face_normal, so X-translation IS")
print("   constrained by the grooves themselves. The one relative DOF")
print("   left free is sliding along the grooves' own shared slide")
print("   axis u — same behavior as a real Kelvin/Maxwell V-groove.)")
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(
        panel_A, panel_B, p, face_normal, theta=0.))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) == 5 and n_zero == 7, \
    f"Got {len(nonzero)} nonzero, {n_zero} zero"
print(f"  Nonzero: {len(nonzero)}  Zero: {n_zero} ✓")
print(f"  λ values: {[f'{e:.4f}' for e in nonzero]}")
system_3b = system                                       # ← capture for figure

# X-translation is NOT free anymore — verify it's actually constrained
C = system.build_constraint_matrix()
rel_x = np.zeros(12)
rel_x[0]  =  1.   # ux_A
rel_x[6]  = -1.   # ux_B
residual = C @ rel_x
print(f"  C @ (relative X-translation) = {np.round(residual, 4)}")
assert not np.allclose(residual, 0), \
    "X-translation should now be constrained by the grooves"
print("  → nonzero: X is now constrained by the grooves, not left to magnets ✓")

# Each groove's own slide axis u remains individually unconstrained by it
for c in system.couplings:
    assert abs(np.dot(c.n1, c.u)) < 1e-10
    assert abs(np.dot(c.n2, c.u)) < 1e-10
print("  Each groove's n1, n2 ⊥ its own slide axis u ✓")

print("\nTest 3c: 3 grooves at same point → max rank 3 from Φ, so ≤ 3 nonzero")
system = CouplingSystem([panel_A, panel_B])
for _ in range(3):
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal,
                                          theta=_ * np.pi / 4))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) <= 3, f"Expected ≤ 3, got {len(nonzero)}"
print(f"  Nonzero: {len(nonzero)}  (expect ≤ 3) ✓")
system_3c = system                                       # ← capture for figure

# ── Section 3 figure ──────────────────────────────────────────────────
figure_section3_eigenvalues([
    (system_3a, 'Test 3a: 1 groove\n(expect 2 nonzero)'),
    (system_3b, 'Test 3b: 3 grooves\n(expect 5 nonzero)'),
    (system_3c, 'Test 3c: same point\n(expect ≤3 nonzero)'),
])
plt.show()

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: PHYSICAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Physical robustness")
print("=" * 60)

print("\nTest 4a: Relative groove orientation is the real tuning knob")
print("  (groove 1 fixed at θ=0; grooves 2 & 3 rotated together by δ")
print("   relative to groove 1 — rotating ALL grooves in lock-step")
print("   changes nothing, since they stay parallel and always leave")
print("   the same shared slide axis free)")
lambda_mins = []
ranks = []
for delta in np.linspace(0, np.pi / 2, 6):
    system = CouplingSystem([panel_A, panel_B])
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p1, face_normal, theta=0.))
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p2, face_normal, theta=delta))
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p3, face_normal, theta=delta))
    nonzero, _ = count_eigenvalues(system)
    lam = min(nonzero) if nonzero else 0.
    lambda_mins.append(lam)
    ranks.append(len(nonzero))
    print(f"  δ={delta:.3f} rad ({np.degrees(delta):.0f}°) → rank={len(nonzero)}  λ_min={lam:.5f}")
assert ranks[0] == 5, \
    "Parallel grooves (δ=0) should leave exactly 1 relative DOF free (rank 5)"
assert all(r == 6 for r in ranks[1:]), \
    "Any relative rotation away from parallel should fully lock the coupling (rank 6)"
print("  Parallel grooves (δ=0) leave a shared sliding DOF free (rank 5);")
print("  any relative rotation between grooves fully locks the coupling")
print("  (rank 6) — this is the real optimization lever, not a shared θ ✓")

print("\nTest 4b: Tilted mating face (45°) → same 5/7 split as Test 3b")
fn_tilted = np.array([np.cos(np.radians(45)), np.sin(np.radians(45)), 0.])
system = CouplingSystem([panel_A, panel_B])
for p in [p1, p2, p3]:
    system.add_coupling(KinematicCoupling(panel_A, panel_B, p,
                                          fn_tilted, theta=0.))
nonzero, n_zero = count_eigenvalues(system)
assert len(nonzero) == 5 and n_zero == 7
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
    assert len(nonzero) == 5 and n_zero == 7, \
        f"Failed at face angle {angle_deg}°: {len(nonzero)} nonzero"
print("  All 8 face orientations give 5 nonzero eigenvalues ✓")

# ── Section 4 figure ──────────────────────────────────────────────────
figure_section4_robustness(
    panel_A, panel_B, p1, p2, p3,
    face_normal, lambda_mins,
    KinematicCoupling, CouplingSystem, count_eigenvalues)
plt.show()

print("\n✓ All tests passed.")