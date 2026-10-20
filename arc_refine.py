"""
arc_refine.py

Tightens the certified bound of ONE complete coupling set (a B&B leaf) by
restricting each coupling's angle to an arc and branching on arcs -- the
argument-division approach of Lu, Deng, Zhang & Fang (J. Glob. Optim. 70,
2018, 171-187), adapted from complex QP to our lambda_min SDP.

Why it's needed (KINEMATIC_COUPLING_SUMMARY.md sec 7.10): the leaf bound
relaxes each coupling's point x_i = (cos 2theta_i, sin 2theta_i) from the
unit circle to the filled disk, and for some leaves the relaxed optimum
sits well inside the disk, so the bound is loose (up to 55% above the
achieved value on birds-foot). That looseness, not the tree search, set
the prototype's 17.6% certified gap.

Geometry (their sec. 2, Lemma 2): if x_i is restricted to the arc of
width w centred at angle phi (in 2*theta space), its convex hull is the
disk plus ONE linear cut, <x_i, (cos phi, sin phi)> >= cos(w/2); a
relaxed point can then sit at most 1 - cos(w/2) inside the circle.
sdp_topology_bound(..., arcs=...) adds these cuts.

Two pieces, both in refine():

  B. Bound tightening (Coffrin, Hijazi & Van Hentenryck's idea for power
     flow): at the root, for each coupling, bound each half of its arc;
     any half whose bound is <= the threshold can't hold the optimum and
     is discarded. Repeated until nothing changes (or `obbt_rounds`).
  A. Best-first branch-and-bound on arcs (their Algorithm 1): repeatedly
     take the open node with the LARGEST bound; branch on the coupling
     whose relaxed point is furthest from its projection onto its arc;
     bisect that arc. Children with bound <= threshold are discarded.
     Each node's relaxed solution is also ROUNDED onto the arcs to give a
     real design (their rounding step); a rounding that beats the
     incumbent is polished with Riemannian gradient sampling and becomes
     the new incumbent.

The result is a valid upper bound on this leaf's true optimum: the
largest bound among open nodes (or among discarded nodes, if none stay
open). If it's <= threshold, the leaf is proved unable to beat the
incumbent.
"""

import sys, os, time, heapq

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sdp_topology_bound import sdp_topology_bound
from riemannian_gradient_sampling import optimize_theta_riemannian, MultiplicityError

TWO_PI = 2.0 * np.pi


def true_lambda_min(system, active, theta, target_rank, length_scale=1.0):
    """Locked lambda_min of the real system for `active` at angles `theta`
    (aligned to sorted(active)); 0.0 if rank-deficient. Restores state."""
    active = sorted(active)
    saved = [(c.active, c.theta) for c in system.couplings]
    try:
        for i, c in enumerate(system.couplings):
            c.active = i in active
        for kk, i in enumerate(active):
            system.couplings[i].set_theta(float(theta[kk]))
        C = system.build_constraint_matrix(length_scale=length_scale)
        rank = np.linalg.matrix_rank(C)
        if rank < target_rank:
            return 0.0
        eigs = np.sort(np.linalg.eigvalsh(C.T @ C))
        return float(eigs[C.shape[1] - rank])
    finally:
        for c, (a, th) in zip(system.couplings, saved):
            c.active = a
            c.set_theta(th)


def project_to_arc(x, arc):
    """Nearest point of the arc (phi, w) to x; returns (point, alpha)."""
    phi, w = arc
    alpha = np.arctan2(x[1], x[0])
    d = (alpha - phi + np.pi) % TWO_PI - np.pi          # signed offset from centre
    if abs(d) > w / 2:
        alpha = phi + np.sign(d) * w / 2
    return np.array([np.cos(alpha), np.sin(alpha)]), alpha


def halves(arc):
    phi, w = arc
    return (phi - w / 4, w / 2), (phi + w / 4, w / 2)


class LeafRefiner:
    """Arc-restricted bounds and refinement for complete sets of one system.

    `registry` and the per-leaf full-rank `ref` come from the shared
    SetEvaluator (evaluator.registry, evaluator.bound(leaf)[2])."""

    def __init__(self, system, registry, target_rank, length_scale=1.0, seed=0):
        self.system = system
        self.m = len(system.couplings)
        self.registry = registry
        self.target_rank = target_rank
        self.length_scale = length_scale
        self.seed = seed
        self.n_solves = 0

    def bound(self, active, ref, arcs):
        """(t, X) for complete set `active` under `arcs`; X[kk] = relaxed
        point of coupling sorted(active)[kk]. t = -inf if infeasible."""
        active = sorted(active)
        self.n_solves += 1
        try:
            r = sdp_topology_bound(self.system, active,
                                   [i for i in range(self.m) if i not in active], [],
                                   len(active), length_scale=self.length_scale,
                                   target_rank=self.target_rank, ref_thetas=ref,
                                   registry=self.registry, arcs=arcs)
        except RuntimeError:
            # Cuts can make the relaxation infeasible (e.g. empty
            # intersection) -- then no design in this region exists.
            return -np.inf, None
        return r['t'], np.column_stack([r['u'], r['v']])

    def _round(self, active, X, arcs):
        th = np.empty(len(active))
        dist = np.empty(len(active))
        for kk, i in enumerate(active):
            p, alpha = project_to_arc(X[kk], arcs[i])
            th[kk] = (alpha / 2.0) % np.pi
            dist[kk] = np.linalg.norm(X[kk] - p)
        return th, dist

    def tighten(self, active, ref, arcs, thresh, max_rounds=3):
        """Piece B. Returns (arcs, n_halves_removed), or (None, n) if every
        region is cut out (the node can't beat the threshold)."""
        arcs = dict(arcs)
        removed = 0
        for _ in range(max_rounds):
            changed = False
            for i in sorted(active):
                lo, hi = halves(arcs[i])
                keep = []
                for h in (lo, hi):
                    trial = dict(arcs); trial[i] = h
                    t, _ = self.bound(active, ref, trial)
                    if t > thresh:
                        keep.append(h)
                if not keep:
                    return None, removed + 2
                if len(keep) == 1:
                    arcs[i] = keep[0]
                    removed += 1
                    changed = True
            if not changed:
                break
        return arcs, removed

    def refine(self, active, ref, incumbent, prune_rel_tol=1e-4, max_nodes=200,
               min_width=1e-3, obbt_rounds=3, polish=True, verbose=False,
               history_every=0):
        """
        Pieces B (at the root) then A. `incumbent` is the best achieved
        lambda_min so far (any design); it can rise during the run if a
        rounding + polish beats it.

        Returns dict:
            'bound'      : valid upper bound on this leaf's optimum
            'proved'     : True if bound <= final threshold
            'incumbent', 'incumbent_theta' : best achieved (may be improved)
            'best_rounded' : best rounding value seen
            'nodes', 'solves', 'halves_removed', 'elapsed_sec'
            'history'    : [(nodes, largest open bound, n open, seconds)] every
                           `history_every` nodes (if > 0)
        """
        t0 = time.perf_counter()
        solves0 = self.n_solves
        active = sorted(active)
        inc, inc_theta = incumbent, None
        thresh = lambda: inc * (1 + prune_rel_tol)

        arcs = {i: (0.0, TWO_PI) for i in active}
        removed = 0
        if obbt_rounds > 0:
            arcs, removed = self.tighten(active, ref, arcs, thresh(), obbt_rounds)
            if arcs is None:
                return dict(bound=thresh(), proved=True, incumbent=inc,
                            incumbent_theta=None, best_rounded=0.0, nodes=0,
                            solves=self.n_solves - solves0, halves_removed=removed,
                            elapsed_sec=time.perf_counter() - t0)

        t, X = self.bound(active, ref, arcs)
        heap, counter = [], 0
        max_discarded = -np.inf
        best_rounded = 0.0
        if t > thresh():
            heapq.heappush(heap, (-t, counter, arcs, X)); counter += 1
        else:
            max_discarded = t
        nodes = 0
        history = []
        while heap and nodes < max_nodes:
            if history_every and nodes % history_every == 0:
                history.append((nodes, -heap[0][0], len(heap), time.perf_counter() - t0))
            negt, _, arcs, X = heapq.heappop(heap)
            t = -negt
            if t <= thresh():                       # incumbent rose since push
                max_discarded = max(max_discarded, t)
                continue
            nodes += 1
            th, dist = self._round(active, X, arcs)
            lam = true_lambda_min(self.system, active, th, self.target_rank,
                                  self.length_scale)
            best_rounded = max(best_rounded, lam)
            if polish and lam > inc * (1 + prune_rel_tol):
                try:
                    ref_full = np.array(ref, dtype=float).copy()
                    ref_full[active] = th
                    r = optimize_theta_riemannian(
                        self.system, active, length_scale=self.length_scale,
                        theta_init=th, target_rank=self.target_rank,
                        ref_thetas=ref_full,
                        rng=np.random.default_rng([self.seed, *active]))
                    if r['lambda_min'] > lam:
                        lam, th = r['lambda_min'], r['theta']
                except (MultiplicityError, ValueError):
                    pass
            if lam > inc:
                inc, inc_theta = lam, np.asarray(th, dtype=float)
                if verbose:
                    print(f"      leaf {active}: new incumbent {inc:.6f}")

            widths = np.array([arcs[i][1] for i in active])
            cand = np.where(widths > min_width, dist, -1.0)
            if cand.max() < 0:                       # can't branch further
                heapq.heappush(heap, (negt, counter, arcs, X)); counter += 1
                break
            i_star = active[int(np.argmax(cand))]
            for h in halves(arcs[i_star]):
                child = dict(arcs); child[i_star] = h
                tc, Xc = self.bound(active, ref, child)
                if tc > thresh():
                    heapq.heappush(heap, (-tc, counter, child, Xc)); counter += 1
                else:
                    max_discarded = max(max_discarded, tc)

        open_max = max((-e[0] for e in heap), default=-np.inf)
        bound = open_max if heap else max(max_discarded, -np.inf)
        if not np.isfinite(bound):
            bound = thresh()
        return dict(bound=float(bound), proved=bool(bound <= thresh()),
                    incumbent=inc, incumbent_theta=inc_theta,
                    best_rounded=best_rounded, nodes=nodes,
                    solves=self.n_solves - solves0, halves_removed=removed,
                    elapsed_sec=time.perf_counter() - t0, history=history)
