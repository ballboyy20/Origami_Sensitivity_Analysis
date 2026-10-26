"""
test_coupling_types.py

Validation for mixed coupling types (RigidBodyModel kinds + coupling_types.py),
items 1-2 of the Phase 1 validation plan (KINEMATIC_COUPLING_SUMMARY.md sec 8.4):

  1. Per-type normals and rows against hand-derived values:
     - flat: one row, normal = face normal;
     - tet: three unit normals at TET_TILT from the face normal, 120 deg apart;
     - sum of n n^T: V = c^2*2ff^T + s^2*2ww^T, flat = ff^T,
       tet = 3cos^2(a) ff^T + (3/2) sin^2(a) (I - ff^T), and the last two are
       theta-independent;
     - each row = n . (Phi_A - Phi_B), translation part [n, -n] and rotation
       part [r_A x n, -(r_B x n)].
  2. Rank of classic couplings between two panels (generic angles):
     Kelvin (T+V+F, all 6 placements) and Maxwell (V+V+V) reach rank 6;
     two sockets (T+T) leave rotation about their joining line free (rank 5).
  3. Affine structure (sec 2.1) holds for mixed designs, with Kc = Ks = 0 for
     flats and sockets (black-box calibrate, as in test_K_theta_structure.py).
  4. Regression: in the expanded birds-foot system, the V candidates of any
     design give exactly the original system's K.
  5. ref_dir: default reproduces the +Z frame; a face parallel to the panel
     works with an explicit ref_dir and raises without one.

Run directly: python test_coupling_types.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import KinematicCoupling, CouplingSystem
from coupling_types import TypedCandidates
from test_K_theta_structure import calibrate, get_K
from interactive_optimizer import (
    build_system, ARBITRARY_START_THETAS_DEG, panel_A, panel_B, face_normal,
    build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG,
)

TOL = 1e-12
rng = np.random.default_rng(0)
failures = []


def check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ''))
    if not ok:
        failures.append(name)


def sum_nnT(c):
    return sum(np.outer(n, n) for n in c.normals)


# ── 1. Per-type normals and rows ────────────────────────────────────────
print("1. Per-type normals and rows")
f = np.array([1., 0., 0.])
p = np.array([1., 0.3, -0.04])
for th in rng.uniform(0, np.pi, size=5):
    fl = KinematicCoupling(panel_A, panel_B, p, f, theta=th, kind='flat')
    vg = KinematicCoupling(panel_A, panel_B, p, f, theta=th, kind='vgroove')
    te = KinematicCoupling(panel_A, panel_B, p, f, theta=th, kind='tet')
    assert (fl.n_rows, vg.n_rows, te.n_rows) == (1, 2, 3)
    assert len(fl.normals) == 1 and np.allclose(fl.normals[0], f, atol=TOL)

    a = KinematicCoupling.TET_TILT
    tn = np.array(te.normals)
    assert np.allclose(np.linalg.norm(tn, axis=1), 1, atol=TOL)
    assert np.allclose(tn @ f, np.cos(a), atol=TOL)
    inplane = tn - np.outer(tn @ f, f)
    for i in range(3):
        j = (i + 1) % 3
        cosang = inplane[i] @ inplane[j] / (np.linalg.norm(inplane[i]) * np.linalg.norm(inplane[j]))
        assert abs(cosang - np.cos(2 * np.pi / 3)) < 1e-12

    P_f = np.outer(f, f)
    c, s = np.cos(vg.HALF_ANGLE), np.sin(vg.HALF_ANGLE)
    assert np.allclose(sum_nnT(vg), 2 * c**2 * P_f + 2 * s**2 * np.outer(vg.w, vg.w), atol=TOL)
    assert np.allclose(sum_nnT(fl), P_f, atol=TOL)
    assert np.allclose(sum_nnT(te), 3 * np.cos(a)**2 * P_f
                       + 1.5 * np.sin(a)**2 * (np.eye(3) - P_f), atol=TOL)

    for cpl in (fl, vg, te):
        rows = cpl.get_constraint_rows(12)
        assert len(rows) == cpl.n_rows
        rA, rB = p - panel_A.centroid, p - panel_B.centroid
        for n, row in zip(cpl.normals, rows):
            expect = np.concatenate([n, np.cross(rA, n), -n, -np.cross(rB, n)])
            assert np.allclose(row, expect, atol=TOL)
check("normals, sum n n^T identities, hand-derived rows (5 random theta)", True)


# ── 2. Rank of classic two-panel couplings ──────────────────────────────
print("2. Two-panel couplings (sites p1, p2, p3), generic angles")
base = build_system(ARBITRARY_START_THETAS_DEG)
tc = TypedCandidates(base)


def rank_of(tc, kinds_per_site, n_draws=3):
    active = tc.from_site_kinds(kinds_per_site)
    ranks = []
    for _ in range(n_draws):
        for i, cpl in enumerate(tc.system.couplings):
            cpl.active = i in active
            cpl.set_theta(rng.uniform(0, np.pi))
        ranks.append(np.linalg.matrix_rank(tc.system.build_constraint_matrix()))
    return max(ranks), min(ranks)


from itertools import permutations
for perm in permutations(('tet', 'vgroove', 'flat')):
    hi, lo = rank_of(tc, list(perm))
    check(f"Kelvin {''.join(k[0].upper() for k in perm)}: rank 6", hi == lo == 6, f"rank {lo}-{hi}")
hi, lo = rank_of(tc, ['vgroove'] * 3)
check("Maxwell VVV: rank 6", hi == lo == 6, f"rank {lo}-{hi}")
for kinds in (['tet', 'tet', None], ['tet', None, 'tet'], [None, 'tet', 'tet']):
    hi, lo = rank_of(tc, kinds)
    check(f"{kinds}: rank 5 (rotation about the joining line)", hi == lo == 5, f"rank {lo}-{hi}")
n_designs = sum(1 for _ in tc.enumerate_designs(6))
check("10 exact-6-row designs on 3 sites", n_designs == 10, f"{n_designs}")


# ── 3. Affine structure with mixed types ────────────────────────────────
print("3. Affine K(theta) structure with mixed types")
bf = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
tcb = TypedCandidates(bf)
target_rank = tcb.system.total_dofs - 6
worst_free, worst_fit = 0.0, 0.0
for trial in range(5):
    # random design: one random kind (or none) per site
    kinds = [tcb.kinds[j] if j < 3 else None for j in rng.integers(0, 4, size=tcb.n_sites)]
    active = tcb.from_site_kinds(kinds)
    for i, cpl in enumerate(tcb.system.couplings):
        cpl.active = i in active
    ref = rng.uniform(0, np.pi, size=tcb.m)
    K0, Kc, Ks = calibrate(tcb.system, ref.copy(), active, 1.0)
    for i in active:
        if tcb.angle_free(i):
            worst_free = max(worst_free, np.abs(Kc[i]).max(), np.abs(Ks[i]).max())
    for _ in range(20):
        th = ref.copy()
        th[active] = rng.uniform(0, np.pi, size=len(active))
        pred = K0 + sum(np.cos(2 * th[i]) * Kc[i] + np.sin(2 * th[i]) * Ks[i] for i in active)
        worst_fit = max(worst_fit, np.abs(get_K(tcb.system, th, 1.0) - pred).max())
check("Kc = Ks = 0 for flats and sockets", worst_free < 1e-12, f"max {worst_free:.1e}")
check("joint random sweep matches affine model", worst_fit < 1e-9, f"max {worst_fit:.1e}")


# ── 4. Regression: V candidates reproduce the original system ───────────
print("4. Expanded birds-foot, V candidates only = original system")
worst = 0.0
for trial in range(10):
    th_deg = rng.uniform(0, 180, size=bf.couplings.__len__())
    orig = build_birdsfoot_system(th_deg)
    sites = sorted(rng.choice(12, size=9, replace=False))
    for i, cpl in enumerate(orig.couplings):
        cpl.active = i in sites
    tce = TypedCandidates(orig)
    active = tce.from_site_kinds({s: 'vgroove' for s in sites})
    for i, cpl in enumerate(tce.system.couplings):
        cpl.active = i in active
    worst = max(worst, np.abs(orig.build_constraint_matrix() - tce.system.build_constraint_matrix()).max())
check("C identical (10 random 9-site designs, random angles)", worst == 0.0, f"max {worst:.1e}")
check("symmetry lift maps kinds to themselves",
      all(tcb.kind_of[j] == tcb.kind_of[i] for i, j in enumerate(tcb.lift_perm(np.arange(12)[::-1]))))


# ── 5. ref_dir ──────────────────────────────────────────────────────────
print("5. ref_dir")
a = KinematicCoupling(panel_A, panel_B, p, f, theta=0.7)
b = KinematicCoupling(panel_A, panel_B, p, f, theta=0.7, ref_dir=[0, 0.3, 2.0])
check("default ref_dir is +Z", np.allclose(a.ref_dir, [0, 0, 1]))
check("ref_dir uses only its in-plane part", np.allclose(b.ref_dir, [0, 0.3, 2.0] / np.linalg.norm([0, 0.3, 2.0])))
top = np.array([0., 0., 1.])
try:
    KinematicCoupling(panel_A, panel_B, p, top, kind='vgroove')
    check("face parallel to the panel without ref_dir raises", False)
except ValueError:
    check("face parallel to the panel without ref_dir raises", True)
g = KinematicCoupling(panel_A, panel_B, p, top, theta=0.4, kind='vgroove', ref_dir=[1, 0, 0])
ok = (abs(g.u @ top) < TOL and abs(g.n1 @ g.n2) < TOL
      and abs(g.n1 @ top - np.cos(g.HALF_ANGLE)) < TOL and abs(g.n1 @ g.u) < TOL)
check("groove on a face parallel to the panel (explicit ref_dir)", ok)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All checks passed.")
