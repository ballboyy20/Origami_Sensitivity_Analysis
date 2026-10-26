"""
coupling_types.py

Mixed coupling types -- Phase 1 of the roadmap (KINEMATIC_COUPLING_SUMMARY.md
sec 8.4, goal B): every candidate SITE may host a flat (1 row), a V-groove
(2 rows) or a tetrahedral socket (3 rows), as in a Kelvin coupling
(1 + 2 + 3 = 6).

Representation: each site is expanded into one candidate coupling per kind,
all at the site's point and face. Every module that works on candidate
indices (per-coupling registry, calibrate, rank tests, Riemannian) keeps
working unchanged, because flats and sockets simply have Kc = Ks = 0 (no
angle dependence, sec 8.3). A design is a set of active candidates with two
extra rules:

  - at most one candidate per site;
  - total rows == target_rank (exact constraint, sec 8.1).

Candidate index = site * len(kinds) + position of the kind in `kinds`.
"""

import sys, os
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from RigidBodyModel import KinematicCoupling, CouplingSystem

KINDS = KinematicCoupling.KINDS          # ('flat', 'vgroove', 'tet')
KIND_LETTER = {'flat': 'F', 'vgroove': 'V', 'tet': 'T'}


class TypedCandidates:
    """
    One candidate per (site, kind) for every coupling (site) of
    `base_system`.

    Attributes
    ----------
    system : CouplingSystem
        The expanded system (shares the base system's panels).
    kinds : tuple of str
    n_sites, m : number of sites, number of candidates (n_sites * len(kinds))
    site_of, rows : (m,) int arrays -- each candidate's site and row count
    kind_of : list of str, length m
    base_thetas_deg : (m,) array -- each candidate starts at its site's angle
    """

    def __init__(self, base_system, kinds=KINDS, base_thetas_deg=None):
        self.kinds = tuple(kinds)
        for k in self.kinds:
            if k not in KINDS:
                raise ValueError(f"unknown kind {k!r}")
        self.base_system = base_system
        self.n_sites = len(base_system.couplings)
        K = len(self.kinds)
        self.m = self.n_sites * K

        if base_thetas_deg is None:
            thetas = np.array([c.theta for c in base_system.couplings], dtype=float)
            base_thetas_deg = np.degrees(thetas)
        else:
            base_thetas_deg = np.asarray(base_thetas_deg, dtype=float)
            thetas = np.radians(base_thetas_deg)
        self.base_thetas_deg = np.repeat(base_thetas_deg, K)

        self.system = CouplingSystem(base_system.panels)
        for s, c in enumerate(base_system.couplings):
            for kind in self.kinds:
                self.system.add_coupling(KinematicCoupling(
                    c.panel_A, c.panel_B, c.point, c.face_normal,
                    theta=thetas[s], kind=kind, ref_dir=c.ref_dir))

        self.site_of = np.repeat(np.arange(self.n_sites), K)
        self.kind_of = [k for _ in range(self.n_sites) for k in self.kinds]
        self.rows = np.array([KinematicCoupling.N_ROWS[k] for k in self.kind_of])

    # ── indexing ────────────────────────────────────────────────────────
    def index(self, site, kind):
        return int(site) * len(self.kinds) + self.kinds.index(kind)

    def from_site_kinds(self, assignment):
        """Active candidate list from {site: kind} (or a per-site sequence
        with None for an empty site)."""
        if not isinstance(assignment, dict):
            assignment = {s: k for s, k in enumerate(assignment) if k is not None}
        return sorted(self.index(s, k) for s, k in assignment.items())

    def site_kinds(self, active):
        """Per-site kind (or None) for an active candidate list."""
        out = [None] * self.n_sites
        for i in active:
            s = int(self.site_of[i])
            if out[s] is not None:
                raise ValueError(f"site {s} has two active candidates")
            out[s] = self.kind_of[i]
        return out

    def describe(self, active, group=None):
        """Compact string, one letter per site ('-' = empty), e.g. 'TVF VV-'.
        `group`: sites per space-separated block (e.g. 3 per crease)."""
        letters = [KIND_LETTER[k] if k else '-' for k in self.site_kinds(active)]
        if not group:
            return ''.join(letters)
        return ' '.join(''.join(letters[i:i + group])
                        for i in range(0, len(letters), group))

    def lift_perm(self, site_perm):
        """Candidate permutation from a site permutation (kinds map to
        themselves), for symmetry breaking."""
        K = len(self.kinds)
        site_perm = np.asarray(site_perm, dtype=int)
        return np.array([site_perm[i // K] * K + i % K for i in range(self.m)])

    # ── design rules ────────────────────────────────────────────────────
    def n_rows(self, active):
        return int(sum(self.rows[i] for i in active))

    def is_valid(self, active, target_rank):
        """At most one candidate per site and exactly target_rank rows."""
        sites = [int(self.site_of[i]) for i in active]
        return len(set(sites)) == len(sites) and self.n_rows(active) == target_rank

    def enumerate_designs(self, target_rank):
        """Every design with exactly target_rank rows (one kind or nothing
        per site). Exhaustive -- only for small systems."""
        choices = [None] + list(self.kinds)
        for assign in product(choices, repeat=self.n_sites):
            rows = sum(KinematicCoupling.N_ROWS[k] for k in assign if k)
            if rows == target_rank:
                yield self.from_site_kinds(list(assign))

    def angle_free(self, i):
        """True if candidate i's K doesn't depend on its angle."""
        return self.kind_of[i] != 'vgroove'
