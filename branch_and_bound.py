"""
branch_and_bound.py

Prototype branch-and-bound (B&B) for choosing which k of m candidate
couplings to activate -- Phase 2 of the roadmap (KINEMATIC_COUPLING_SUMMARY.md
sec 8.4), in its first, V-groove-only form.

A NODE fixes some couplings on (forced_active), some off (forced_inactive),
and leaves the rest undecided. Search is depth-first from the root. At
each node, in order of cost:

  1. Cardinality: enough undecided couplings left to reach k, and not too
     many forced on.
  2. Cut counting (exact constraint, sec 8.3): for every split of the
     panels into two groups, the couplings that could still be active
     must supply >= 6 rows across the split, and the forced-active rows
     inside each group must not exceed 6*(group size - 1) (more would be
     redundant).
  3. Symmetry: user-supplied panel-geometry symmetries (as permutations of
     coupling indices). Only the lexicographically largest design of each
     symmetric family is kept; a node is discarded as soon as its decided
     couplings make it certain that a symmetric copy wins.
  4. Generic rank: can forced_active + undecided reach full rank? Rank is
     generic in theta (if any theta works, almost every theta does), so a
     few random-theta draws decide this; no "unresolved" status.
  5. Certified bound (sdp_topology_bound): discard if bound <=
     incumbent * (1 + prune_rel_tol).

Surviving nodes branch on the MOST-FRACTIONAL undecided coupling (relaxed z
closest to 0.5), exploring the child that rounds z first (z >= 0.5 -> on).

A LEAF is a complete k-set: certified bound first (via the shared
SetEvaluator), then the full Riemannian evaluation only if the bound beats
the incumbent. The incumbent starts from a supplied design (normally
greedy + swap search) and is updated whenever a leaf beats it.

Certificate: every discarded node's best completion is <= its bound <=
incumbent * (1 + tol), and every evaluated leaf's true optimum is <= its
own bound. So

    certified upper bound = max(incumbent * (1 + tol), max bound over
                                evaluated leaves)

and the certified gap is that minus the incumbent. The incumbent is only
as good as the leaf evaluation (Riemannian is a local method); the gap
quantifies exactly that uncertainty.
"""

import sys, os, time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sdp_topology_bound import sdp_topology_bound

ROWS_PER_COUPLING = 2      # V-grooves only, for this prototype


def _rank_full(system, active, target_rank, theta, length_scale):
    saved = [(c.active, c.theta) for c in system.couplings]
    try:
        for i, c in enumerate(system.couplings):
            c.active = i in active
            c.set_theta(float(theta[i]))
        C = system.build_constraint_matrix(length_scale=length_scale)
        return C.shape[0] > 0 and np.linalg.matrix_rank(C) >= target_rank
    finally:
        for c, (a, th) in zip(system.couplings, saved):
            c.active = a
            c.set_theta(th)


def reflect(points, planes):
    """Reflect points (n, 3) through each plane in turn; a plane is
    (unit normal, offset c) for the plane {p : p . normal = c}."""
    p = np.array(points, dtype=float)
    for nrm, c in planes:
        nrm = np.asarray(nrm, dtype=float)
        p = p - 2.0 * ((p @ nrm) - c)[:, None] * nrm[None, :]
    return p


def verify_symmetry(evaluator, perm, planes, k, n_sample_sets=300, rel_tol=1e-4,
                    seed=0):
    """
    Check that `perm` (perm[i] = image of coupling i) is a genuine symmetry
    before using it to discard branches -- a wrong one would silently
    discard the optimum. Deliberately does NOT compare Riemannian lambda_min
    values: those carry restart noise (sec 7.1: mirror pairs differing by up
    to ~1%), which once caused a valid symmetry to be rejected.

      1. Geometry (exact): reflecting each candidate's contact point through
         `planes` lands on its partner's point, and its face normal is
         parallel (either sign) to the partner's.
      2. Certified bounds (deterministic convex optimum): for every k-set
         (or n_sample_sets random ones, if there are more), the set and its
         image have the same bound to rel_tol and the same feasibility.

    Returns (ok, details dict).
    """
    system = evaluator.system
    perm = np.asarray(perm, dtype=int)
    m = len(system.couplings)
    pts = np.array([c.point for c in system.couplings])
    nrm = np.array([c.face_normal for c in system.couplings])
    pts_r = reflect(pts, planes)
    nrm_r = reflect(nrm, [(n, 0.0) for n, _ in planes])   # linear part only
    point_err = float(np.max(np.linalg.norm(pts_r - pts[perm], axis=1)))
    normal_err = float(np.max(1.0 - np.abs(np.sum(nrm_r * nrm[perm], axis=1))))
    geom_ok = point_err < 1e-9 and normal_err < 1e-9

    all_sets = list(combinations(range(m), k))
    rng = np.random.default_rng(seed)
    if len(all_sets) > n_sample_sets:
        idx = rng.choice(len(all_sets), size=n_sample_sets, replace=False)
        sets = [all_sets[i] for i in idx]
    else:
        sets = all_sets
    worst, feas_mismatch = 0.0, 0
    for s in sets:
        img = tuple(sorted(int(perm[i]) for i in s))
        st1, b1, _ = evaluator.bound(s)
        st2, b2, _ = evaluator.bound(img)
        if (st1 == 'bounded') != (st2 == 'bounded'):
            feas_mismatch += 1
        elif st1 == 'bounded':
            worst = max(worst, abs(b1 - b2) / max(abs(b1), 1e-12))
    bound_ok = feas_mismatch == 0 and worst <= rel_tol
    return geom_ok and bound_ok, dict(point_err=point_err, normal_err=normal_err,
                                      n_sets_checked=len(sets),
                                      bound_max_rel_diff=worst,
                                      feasibility_mismatches=feas_mismatch)


class BranchAndBound:
    """
    Parameters
    ----------
    evaluator : swap_search.SetEvaluator
        Shared, cached leaf evaluator (also holds system, registry,
        target_rank, base angles). Sharing it with a prior swap search
        reuses that search's evaluations.
    k : int
        Number of couplings to activate.
    symmetries : list of permutations (each a list/array p with p[i] = image
        of coupling i), or None. Each must be a genuine symmetry of the
        system (same lambda_min for a design and its image) -- verify
        before passing one in.
    fixed_active : couplings that must be active (e.g. leaf-panel ones).
    prune_rel_tol : relative pruning tolerance (bound <= incumbent*(1+tol)).
    n_rank_draws : random-theta draws for the generic-rank test.
    """

    def __init__(self, evaluator, k, symmetries=None, fixed_active=(),
                 prune_rel_tol=1e-4, n_rank_draws=3, seed=0, verbose=True):
        self.ev = evaluator
        self.system = evaluator.system
        self.m = evaluator.m
        self.k = k
        self.target_rank = evaluator.target_rank
        self.length_scale = evaluator.length_scale
        self.symmetries = [np.asarray(p, dtype=int) for p in (symmetries or [])]
        self.fixed_active = sorted(int(i) for i in fixed_active)
        self.prune_rel_tol = prune_rel_tol
        self.n_rank_draws = n_rank_draws
        self.seed = seed
        self.verbose = verbose

        # Panel pair per coupling, for cut counting
        self.n_panels = len(self.system.panels)
        self.ends = [(c.panel_A.id, c.panel_B.id) for c in self.system.couplings]
        panels = list(range(self.n_panels))
        self.cuts = []                     # (S as frozenset), each split once
        for r in range(1, self.n_panels):
            for S in combinations(panels[1:], r - 1):
                S = frozenset((0,) + S)
                if len(S) < self.n_panels:
                    self.cuts.append(S)

    # ── node tests ──────────────────────────────────────────────────────
    def _cut_ok(self, fa, und):
        # Both rules assume exact constraint (rows == target_rank); skip
        # them otherwise rather than prune something valid.
        if ROWS_PER_COUPLING * self.k != self.target_rank:
            return True
        possible = set(fa) | set(und)
        for S in self.cuts:
            cross = sum(1 for i in possible if (self.ends[i][0] in S) != (self.ends[i][1] in S))
            if ROWS_PER_COUPLING * cross < 6:
                return False
            for side, size in ((S, len(S)), (None, self.n_panels - len(S))):
                if side is None:
                    inside = sum(1 for i in fa if self.ends[i][0] not in S
                                 and self.ends[i][1] not in S)
                else:
                    inside = sum(1 for i in fa if self.ends[i][0] in S and self.ends[i][1] in S)
                if ROWS_PER_COUPLING * inside > 6 * (size - 1):
                    return False
        return True

    def _symmetry_ok(self, x):
        """x: array of 1 (on), 0 (off), -1 (undecided). Keep the design only
        if it is >= each symmetric image in lexicographic order; discard
        only when that is already certain to fail."""
        for p in self.symmetries:
            y = x[p]
            for xi, yi in zip(x, y):
                if xi < 0 or yi < 0:
                    break              # undetermined from here on
                if xi != yi:
                    if xi < yi:
                        return False   # certain: the image is larger
                    break              # certain: x is larger
        return True

    def _reference(self, fa, und, key):
        rng = np.random.default_rng([self.seed, 7, *key])
        union = set(fa) | set(und)
        for _ in range(self.n_rank_draws):
            theta = rng.uniform(0, np.pi, size=self.m)
            if _rank_full(self.system, union, self.target_rank, theta, self.length_scale):
                return theta
        return None

    # ── search ──────────────────────────────────────────────────────────
    def run(self, incumbent_active=None, max_nodes=200000):
        t0 = time.perf_counter()
        st = dict(nodes=0, pruned_cardinality=0, pruned_cut=0, pruned_symmetry=0,
                  pruned_rank=0, pruned_bound=0, sdp_errors=0, leaves=0,
                  leaves_pruned_bound=0, leaves_infeasible=0, leaves_evaluated=0,
                  incumbent_updates=0)
        inc_set, inc_lam = None, 0.0
        if incumbent_active is not None:
            r = self.ev.evaluate(incumbent_active)
            if r['feasible']:
                inc_set, inc_lam = sorted(incumbent_active), r['lambda_min']
        start_lam = inc_lam
        evaluated_leaves = []       # (set, bound, achieved)
        trace = []                  # incumbent history

        def thresh():
            return inc_lam * (1 + self.prune_rel_tol)

        stack = [(tuple(self.fixed_active), ())]   # (forced_active, forced_inactive)
        while stack and st['nodes'] < max_nodes:
            fa, fi = stack.pop()
            st['nodes'] += 1
            und = [i for i in range(self.m) if i not in fa and i not in fi]

            # cardinality, and normalize forced completions to leaves
            if len(fa) > self.k or len(fa) + len(und) < self.k:
                st['pruned_cardinality'] += 1
                continue
            if len(fa) == self.k or len(fa) + len(und) == self.k:
                leaf = sorted(fa) if len(fa) == self.k else sorted(set(fa) | set(und))
                x = np.zeros(self.m, dtype=int); x[leaf] = 1
                if not self._symmetry_ok(x):
                    st['pruned_symmetry'] += 1
                    continue
                st['leaves'] += 1
                status, b, _ = self.ev.bound(leaf)
                if status != 'bounded':
                    st['leaves_infeasible'] += 1
                    continue
                if b <= thresh():
                    st['leaves_pruned_bound'] += 1
                    continue
                st['leaves_evaluated'] += 1
                r = self.ev.evaluate(leaf)
                evaluated_leaves.append((leaf, b, r['lambda_min'] if r['feasible'] else 0.0))
                if r['feasible'] and r['lambda_min'] > inc_lam:
                    inc_set, inc_lam = leaf, r['lambda_min']
                    st['incumbent_updates'] += 1
                    trace.append(dict(node=st['nodes'], lambda_min=inc_lam, active=leaf,
                                      elapsed_sec=time.perf_counter() - t0))
                    if self.verbose:
                        print(f"    new incumbent {inc_lam:.6f} {leaf} (node {st['nodes']})")
                continue

            if not self._cut_ok(fa, und):
                st['pruned_cut'] += 1
                continue
            x = -np.ones(self.m, dtype=int); x[list(fa)] = 1; x[list(fi)] = 0
            if not self._symmetry_ok(x):
                st['pruned_symmetry'] += 1
                continue
            key = tuple(int(v) for v in x + 1)
            ref = self._reference(fa, und, key)
            if ref is None:
                st['pruned_rank'] += 1
                continue
            try:
                res = sdp_topology_bound(self.system, list(fa), list(fi), und, self.k,
                                         length_scale=self.length_scale,
                                         target_rank=self.target_rank, ref_thetas=ref,
                                         registry=self.ev.registry)
                b, z = res['t'], res['z']
            except (RuntimeError, ValueError):
                st['sdp_errors'] += 1          # can't bound: branch, never prune
                b, z = float('inf'), np.full(len(und), 0.5)
            if b <= thresh():
                st['pruned_bound'] += 1
                continue

            j = int(np.argmin(np.abs(z - 0.5)))
            v = und[j]
            on = (tuple(sorted(fa + (v,))), fi)
            off = (fa, tuple(sorted(fi + (v,))))
            # depth-first: push the preferred child last so it pops first
            stack.extend([off, on] if z[j] >= 0.5 else [on, off])

        complete = not stack
        ub = max([thresh()] + [b for _, b, _ in evaluated_leaves])
        if not complete:
            ub = float('inf')
        return dict(active=inc_set, lambda_min=inc_lam, start_lambda_min=start_lam,
                    certified_upper_bound=ub, certified_gap=ub - inc_lam,
                    complete=complete, stats=st, incumbent_trace=trace,
                    evaluated_leaves=[dict(active=s, bound=b, lambda_min=l)
                                      for s, b, l in evaluated_leaves],
                    elapsed_sec=time.perf_counter() - t0)
