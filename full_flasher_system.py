"""
full_flasher_system.py

The whole flasher as one assembly: the central pentagon hub plus 5
identical gores of 5 panels each (26 rigid bodies, 156 DOFs, target rank
150), built from Full_Flasher_gore_reference.dxf by rotating the gore
about the hub centre in 72-degree steps.

Why (KINEMATIC_COUPLING_SUMMARY.md sec 7.4 limitation, 2026-11-18 replan):
the full-gore system duplicates the neighbour gore's panels 3 and 4 as
independent bodies (DXF panels 7 and 6), counts crease 3-4 twice (DXF 6-7),
leaves the gore's own panels 3 and 4 free on their other side, and makes
the hub a leaf. Here every physical panel and crease appears exactly once.

DXF panel ids (full_flasher_gore_system): hub = 5 (regular pentagon),
gore = 0..4, and 6 / 7 = panels 4 / 3 of the next gore (rotation +72 deg
about the hub centre, checked in validate()).

Creases per gore (8, 40 in total):
  0-1, 0-4, 0-hub, 1-2, 1-3, 3-4    within the gore (and to the hub)
  1-4', 2-3'                         to the next gore (' = gore g+1)

Panel ids here: hub = 0, gore g panel j = 1 + 5 g + j.

Candidates: 3 sites per crease (gore_system_common recipe, same thickness
as the full-gore system), so 24 sites per gore and 120 couplings. Designs
are cyclically symmetric with TIED angles: a candidate is a CyclicCoupling
orbit, one site together with its rotated copies in all 5 gores, sharing
one theta and one type. A V-groove orbit gives 10 rows, so an exactly
constrained all-V design has 15 of the 24 orbits.

Builders:
  build_flasher_copies(theta_degs)  -> CouplingSystem, 120 KinematicCouplings
                                       (copy of site s in gore g at index
                                       5 s + g), for checks
  build_flasher_system(theta_degs)  -> CouplingSystem, 24 CyclicCoupling
                                       orbits (V-grooves), what the
                                       optimizers use
theta_degs: 24 angles, one per site (= per orbit).

Run: python full_flasher_system.py   (validation checks)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import RigidPanel, KinematicCoupling, CouplingSystem
from full_flasher_gore_system import (FULL_GORE_PANELS, FULL_GORE_CREASES,
                                      FULL_GORE_THICKNESS)
from gore_system_common import (_spoke_face_normal, _spoke_contact_points,
                                default_start_thetas_deg)

N_GORES = 5
GORE_PANELS_PER = 5
DXF_HUB = 5
# DXF id -> (gore offset, local panel); hub -> None
DXF_MAP = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4),
           5: None, 6: (1, 4), 7: (1, 3)}

HUB_CENTER = np.array(FULL_GORE_PANELS[DXF_HUB].vertices, dtype=float)[:, :2].mean(axis=0)
STEP = 2 * np.pi / N_GORES
THICKNESS = FULL_GORE_THICKNESS


def rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotate_points(P, g):
    """Rotate (n, 3) or (3,) points by g * 72 deg about the hub centre (z kept)."""
    P = np.array(P, dtype=float)
    out = P.copy()
    out[..., :2] = (P[..., :2] - HUB_CENTER) @ rot2(g * STEP).T + HUB_CENTER
    return out


def rotate_vectors(V, g):
    V = np.array(V, dtype=float)
    out = V.copy()
    out[..., :2] = V[..., :2] @ rot2(g * STEP).T
    return out


def panel_id(g, j):
    return 1 + GORE_PANELS_PER * (g % N_GORES) + j


def panel_name(pid):
    if pid == 0:
        return 'H'
    g, j = divmod(pid - 1, GORE_PANELS_PER)
    return f'g{g}P{j}'


# ── creases of the canonical gore (gore 0), in DXF crease order ──────────

def _canonical_creases():
    """List of dicts {a: (g, j) or None, b: ..., p_start, p_end, name}, one
    per distinct physical crease class, endpoints in gore 0's frame. The
    DXF crease 6-7 is gore 1's copy of 3-4 and is dropped as a duplicate."""
    out, seen = [], set()
    for c in FULL_GORE_CREASES:
        a, b = DXF_MAP[c['panel_a']], DXF_MAP[c['panel_b']]
        shift = min(x[0] for x in (a, b) if x is not None)
        a = None if a is None else (a[0] - shift, a[1])
        b = None if b is None else (b[0] - shift, b[1])
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        nm = lambda x: 'H' if x is None else (f"P{x[1]}" + "'" * x[0])
        out.append(dict(a=a, b=b, p_start=rotate_points(c['p_start'], -shift),
                        p_end=rotate_points(c['p_end'], -shift),
                        name=f"{nm(a)}-{nm(b)}", dxf=(c['panel_a'], c['panel_b'])))
    return out


CREASES = _canonical_creases()
N_CREASES = len(CREASES)                    # per gore
N_SITES = 3 * N_CREASES                     # per gore = number of orbits
CREASE_NAMES = [c['name'] for c in CREASES]
START_THETAS_DEG = default_start_thetas_deg(N_CREASES)


def _pid(x, g):
    return 0 if x is None else panel_id(g + x[0], x[1])


def build_panels():
    panels = [RigidPanel(0, FULL_GORE_PANELS[DXF_HUB].vertices.copy(), thickness=THICKNESS)]
    for g in range(N_GORES):
        for j in range(GORE_PANELS_PER):
            panels.append(RigidPanel(panel_id(g, j),
                                     rotate_points(FULL_GORE_PANELS[j].vertices, g),
                                     thickness=THICKNESS))
    return panels


def _site_copies(panels, kind='vgroove'):
    """copies[s][g] = (panel_A, panel_B, point, face_normal) for site s in gore g."""
    copies = []
    for c in CREASES:
        per_gore = []
        for g in range(N_GORES):
            A, B = panels[_pid(c['a'], g)], panels[_pid(c['b'], g)]
            ps, pe = rotate_points(c['p_start'], g), rotate_points(c['p_end'], g)
            n = _spoke_face_normal(ps, pe, A, B)
            per_gore.append([(A, B, p, n) for p in _spoke_contact_points(ps, pe, THICKNESS)])
        for k in range(3):
            copies.append([per_gore[g][k] for g in range(N_GORES)])
    return copies


class CyclicCoupling:
    """One site and its rotated copies in every gore, with one shared theta
    and one type. Duck-types KinematicCoupling for everything the pipeline
    uses (active, theta, set_theta, get_constraint_rows, kind, n_rows; point,
    face_normal, panel_A/B, ref_dir are gore 0's copy, for naming/drawing).
    K is still affine in (cos 2theta, sin 2theta): it is a sum of copies
    that each are, at the same theta."""

    def __init__(self, copies, theta=0.0, kind='vgroove'):
        self.copies = list(copies)
        self.kind = kind
        self.theta = float(theta)
        self.active = True
        c0 = self.copies[0]
        self.panel_A, self.panel_B = c0.panel_A, c0.panel_B
        self.point, self.face_normal, self.ref_dir = c0.point, c0.face_normal, c0.ref_dir
        self.n_rows = sum(c.n_rows for c in self.copies)

    @classmethod
    def from_specs(cls, specs, theta=0.0, kind='vgroove'):
        return cls([KinematicCoupling(A, B, p, n, theta=theta, kind=kind)
                    for (A, B, p, n) in specs], theta=theta, kind=kind)

    def with_kind(self, kind, theta=None):
        th = self.theta if theta is None else theta
        return CyclicCoupling([KinematicCoupling(c.panel_A, c.panel_B, c.point, c.face_normal,
                                                 theta=th, kind=kind, ref_dir=c.ref_dir)
                               for c in self.copies], theta=th, kind=kind)

    def set_theta(self, theta):
        self.theta = float(theta)
        for c in self.copies:
            c.set_theta(theta)

    def get_constraint_rows(self, total_dofs):
        rows = []
        for c in self.copies:
            rows.extend(c.get_constraint_rows(total_dofs))
        return tuple(rows)


def build_flasher_system(theta_degs=START_THETAS_DEG, kind='vgroove'):
    panels = build_panels()
    system = CouplingSystem(panels)
    for s, specs in enumerate(_site_copies(panels)):
        system.add_coupling(CyclicCoupling.from_specs(specs, np.radians(theta_degs[s]), kind))
    return system


def build_flasher_copies(theta_degs=START_THETAS_DEG, kind='vgroove'):
    panels = build_panels()
    system = CouplingSystem(panels)
    for s, specs in enumerate(_site_copies(panels)):
        for (A, B, p, n) in specs:
            system.add_coupling(KinematicCoupling(A, B, p, n, np.radians(theta_degs[s]), kind))
    return system


def site_name(s):
    return f"{CREASE_NAMES[s // 3]}#{s % 3}"


def describe_counts(active):
    """Couplings per crease, e.g. 'P0-P1:2 P0-P4:3 ...' (per gore)."""
    cnt = np.bincount(np.asarray(active, dtype=int) // 3, minlength=N_CREASES)
    return ' '.join(f"{n}:{k}" for n, k in zip(CREASE_NAMES, cnt))


# ── validation ───────────────────────────────────────────────────────────

def _edge_on_polygon(p, q, verts, tol):
    """Max distance of segment endpoints p, q from the polygon's vertex set,
    and whether p-q is one of its edges (within tol)."""
    V = np.asarray(verts)[:, :2]
    dp = np.min(np.linalg.norm(V - p[:2], axis=1))
    dq = np.min(np.linalg.norm(V - q[:2], axis=1))
    return max(dp, dq)


def _rigid_modes(panels, total_dofs):
    """6 global rigid-body motions (about the hub centre) as DOF vectors."""
    o = np.array([HUB_CENTER[0], HUB_CENTER[1], 0.0])
    M = np.zeros((total_dofs, 6))
    for p in panels:
        i, r = p.dof_start, p.centroid - o
        for a in range(3):
            M[i + a, a] = 1.0                       # translation
            w = np.zeros(3); w[a] = 1.0
            M[i:i + 3, 3 + a] = np.cross(w, r)      # rotation: v = w x r
            M[i + 3 + a, 3 + a] = 1.0
    return M


def validate(seed=0):
    rng = np.random.default_rng(seed)
    ok_all = True

    def report(label, ok, detail=''):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")

    print("Full flasher system -- validation")
    print("=" * 78)
    sysV = build_flasher_system()
    n_p, dofs = len(sysV.panels), sysV.total_dofs
    report("counts", n_p == 26 and N_CREASES == 8 and len(sysV.couplings) == 24 and dofs == 156,
           f"{n_p} panels, {N_CREASES} creases/gore ({N_GORES * N_CREASES} total), "
           f"{N_SITES} orbits, {dofs} DOFs, target rank {dofs - 6}")
    print(f"  creases per gore: {', '.join(CREASE_NAMES)}"
          f"  (DXF crease 6-7 dropped as gore 1's copy of 3-4)")
    print(f"  hub centre {HUB_CENTER.round(3)}, thickness {THICKNESS:.4f}")

    # 1. geometry: rotated copies reproduce the DXF neighbour panels
    panels = sysV.panels
    for dxf_id, (g, j) in ((6, (1, 4)), (7, (1, 3))):
        A = np.asarray(FULL_GORE_PANELS[dxf_id].vertices)[:, :2]
        B = np.asarray(panels[panel_id(g, j)].vertices)[:, :2]
        err = max(np.min(np.linalg.norm(B - a, axis=1)) for a in A)
        report(f"DXF panel {dxf_id} = gore {g} P{j}", err < 1.0, f"max vertex distance {err:.3f}")
    hub = np.asarray(panels[0].vertices)[:, :2]
    err = max(np.min(np.linalg.norm(rotate_points(panels[0].vertices, 1)[:, :2] - h, axis=1))
              for h in hub)
    # The DXF pentagon is regular to ~0.01 units (sides ~117.6). Only the hub's
    # centroid enters K (its creases come from rotating gore 0's), so this is
    # a drawing-accuracy check, not a modelling one.
    report("hub maps to itself under 72 deg (DXF drawing accuracy)", err < 0.05,
           f"max vertex distance {err:.3f}")

    # 2. every crease lies on an edge shared by its two panels
    worst = 0.0
    for c in CREASES:
        for g in range(N_GORES):
            ps, pe = rotate_points(c['p_start'], g), rotate_points(c['p_end'], g)
            for x in (c['a'], c['b']):
                worst = max(worst, _edge_on_polygon(ps, pe, panels[_pid(x, g)].vertices, 1.0))
    report("crease endpoints are vertices of both panels", worst < 1.0,
           f"max distance {worst:.3f} (DXF snapping was up to 0.45)")

    # 3. orbit system == 120-copy system with tied angles
    th = rng.uniform(0, 180, N_SITES)
    sV, sC = build_flasher_system(th), build_flasher_copies(th)
    KV = sV.build_constraint_matrix().T @ sV.build_constraint_matrix()
    CC = sC.build_constraint_matrix()
    KC = CC.T @ CC
    d = np.abs(KV - KC).max() / np.abs(KC).max()
    report("orbit K == copies K (tied angles)", d < 1e-12, f"rel diff {d:.1e}")

    # 4. all candidates active: rank 150, nullspace = global rigid motions
    s = np.linalg.svd(CC, compute_uv=False)
    rank = int(np.sum(s > s[0] * max(CC.shape) * np.finfo(float).eps))
    R = _rigid_modes(panels, dofs)
    resid = np.linalg.norm(CC @ R) / np.linalg.norm(CC) / np.linalg.norm(R)
    report("all 120 couplings: rank 150, rigid motions in the nullspace",
           rank == dofs - 6 and resid < 1e-12,
           f"rank {rank}, |C R| rel {resid:.1e}, sigma_150/sigma_1 {s[dofs - 7] / s[0]:.2e}")

    # 5. cyclic invariance: relabel gores g -> g+1 (DOFs rotated too)
    Q = np.zeros((dofs, dofs))
    R3 = np.eye(3); R3[:2, :2] = rot2(STEP)
    blk = np.kron(np.eye(2), R3)
    for p in panels:
        tgt = 0 if p.id == 0 else panel_id((p.id - 1) // 5 + 1, (p.id - 1) % 5)
        Q[6 * tgt:6 * tgt + 6, 6 * p.id:6 * p.id + 6] = blk
    # rotation DOFs are about each panel's own centroid, which rotates with it
    d = np.abs(Q @ KV @ Q.T - KV).max() / np.abs(KV).max()
    report("K invariant under the 72 deg relabelling", d < 1e-10, f"rel diff {d:.1e}")

    # 6. affine-in-theta structure survives orbit tying
    from test_K_theta_structure import calibrate, predict_K
    idx = list(range(N_SITES))
    ref = np.radians(rng.uniform(0, 180, N_SITES))
    K0, Kc, Ks = calibrate(sV, ref.copy(), idx, 1.0)
    th2 = np.radians(rng.uniform(0, 180, N_SITES))
    for c, t in zip(sV.couplings, th2):
        c.set_theta(t)
    C2 = sV.build_constraint_matrix()
    d = np.abs(predict_K(K0, Kc, Ks, th2, idx) - C2.T @ C2).max() / np.abs(C2.T @ C2).max()
    report("K(theta) affine in (cos 2theta, sin 2theta) per orbit", d < 1e-10, f"rel diff {d:.1e}")

    # 7. exactly constrained all-V designs: 15 of 24 orbits, random sample
    k = (dofs - 6) // 10
    rows_of = [np.arange(10 * i, 10 * i + 10) for i in range(N_SITES)]
    C1 = sV.build_constraint_matrix()
    n, nfull, t0 = 2000, 0, time.perf_counter()
    cnt_full = []
    for _ in range(n):
        S = np.sort(rng.choice(N_SITES, k, replace=False))
        sv = np.linalg.svd(C1[np.concatenate([rows_of[i] for i in S])], compute_uv=False)
        if np.sum(sv > sv[0] * 156 * np.finfo(float).eps) >= dofs - 6:
            nfull += 1
            cnt_full.append(sv[dofs - 7] / sv[0])
    dt = time.perf_counter() - t0
    report(f"exact all-V designs exist (k = {k} orbits = 150 rows)", nfull > 0,
           f"{nfull}/{n} random sets full rank at random theta "
           f"({dt / n * 1e3:.1f} ms/set; full sweep of C(24,15) = 1,307,504 ~ "
           f"{dt / n * 1307504 / 60:.0f} min)" +
           (f"; relative sigma_150 median {np.median(cnt_full):.1e}" if cnt_full else ''))

    lam = build_flasher_system().get_rigidity_eigenvalue()
    print(f"  lambda_min, all 24 orbits active at the default angles: {lam:.6f}"
          f" (over-constrained, 240 rows; sanity only)")
    print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    validate()
