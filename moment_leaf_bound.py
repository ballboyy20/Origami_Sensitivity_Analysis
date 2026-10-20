"""
moment_leaf_bound.py

Moment (Henrion-Lasserre) relaxation of the leaf problem, for tightening
loose certified bounds (KINEMATIC_COUPLING_SUMMARY.md sec 2.6, 7.11).

For a complete coupling set, with x = (cos 2theta_1, sin 2theta_1, ...,
cos 2theta_n, sin 2theta_n) in R^{2n} and the gauge-reduced stiffness

    K(x) = A0 + sum_j x_j A_j          (target_rank x target_rank)

the question "can this set reach lambda_min >= tau?" is: is there x on
the n circles with G_tau(x) = K(x) - tau*I >= 0 (PSD)? That's a
polynomial matrix inequality (PMI). Its order-d moment relaxation
(Henrion & Lasserre, IEEE TAC 2006) replaces x by a vector y of
"moments" y_alpha (alpha = monomials of degree <= 2d) with y_0 = 1 and:

  - moment matrix  M_d(y) = [ y_{a+b} ]  over monomials a, b of degree <= d,  PSD
  - localizing     L_{d-1}(G y) = [ sum_g G_g y_{a+b+g} ]  over a, b of
                   degree <= d-1 (G has degree 1, so g in {1, x_j}),     PSD
  - circles        y_{m + 2c_i} + y_{m + 2s_i} - y_m = 0 for monomials m
                   of degree <= 2d - 2.

Every real x on the circles gives a feasible y (its point moments), so
infeasibility of the relaxation PROVES no real design reaches tau. Order
1 is exactly the disk bound already in use (M_1 PSD + circle equality
projects onto the disks). Order 2 adds the product moments x_i x_j,
which is what the collective looseness needs.

For robustness, `margin(tau)` maximizes eps subject to L_{d-1}(G_tau y)
>= eps*I: eps* < 0 means infeasible (proved), eps* >= 0 means the
relaxation can't rule tau out. `bound()` bisects on tau for the order-d
upper bound itself.
"""

import sys, os, time
from itertools import combinations_with_replacement

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

sys.path.insert(0, os.path.dirname(__file__))


def monomials(nvars, max_deg):
    """All monomials of degree <= max_deg as sorted index tuples."""
    out = [()]
    for d in range(1, max_deg + 1):
        out.extend(combinations_with_replacement(range(nvars), d))
    return out


class MomentLeafRelaxation:
    """
    A0 : (r, r) constant term; A : list of 2n (r, r) matrices, ordered
    (cos_1, sin_1, cos_2, sin_2, ...). order : relaxation order d (1 or 2).
    """

    def __init__(self, A0, A, order=2):
        self.A0 = np.asarray(A0, dtype=float)
        self.A = [np.asarray(a, dtype=float) for a in A]
        self.r = self.A0.shape[0]
        self.nv = len(self.A)
        self.n = self.nv // 2
        self.d = order

        self.mons = monomials(self.nv, 2 * order)
        self.idx = {m: k for k, m in enumerate(self.mons)}
        self.N = len(self.mons)
        self.basis_M = monomials(self.nv, order)          # moment matrix rows
        self.basis_L = monomials(self.nv, order - 1)      # localizing rows
        self._build()

    def _key(self, *parts):
        return self.idx[tuple(sorted(sum(parts, ())))]

    def _build(self):
        nv, r = self.nv, self.r
        # Moment matrix selector: vec_F(M) = S_M @ y
        B = self.basis_M
        nb = len(B)
        rows, cols = [], []
        for j, b in enumerate(B):
            for i, a in enumerate(B):
                rows.append(i + j * nb)
                cols.append(self._key(a, b))
        self.S_M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nb * nb, self.N))
        self.nb_M = nb

        # Localizing: block (a, b) = sum_g G_g * y_{a+b+g}; G_0 = A0 - tau*I
        # (tau part kept separate), G_j = A_j. vec_F(L) = (T_K - tau*T_I) @ y
        BL = self.basis_L
        nl = len(BL)
        size = nl * r
        gens = [((), self.A0)] + [((j,), self.A[j]) for j in range(nv)]
        rK, cK, vK = [], [], []
        rI, cI, vI = [], [], []
        ii, jj = np.meshgrid(np.arange(r), np.arange(r), indexing='ij')
        ii, jj = ii.ravel(), jj.ravel()
        for bj, b in enumerate(BL):
            for ai, a in enumerate(BL):
                base_r = ai * r + ii
                base_c = bj * r + jj
                flat = base_r + base_c * size
                for g, Gg in gens:
                    vals = Gg[ii, jj]
                    nz = vals != 0
                    rK.append(flat[nz]); cK.append(np.full(nz.sum(), self._key(a, b, g)))
                    vK.append(vals[nz])
                diag = ii == jj
                rI.append(flat[diag]); cI.append(np.full(diag.sum(), self._key(a, b)))
                vI.append(np.ones(diag.sum()))
        self.T_K = sp.csr_matrix((np.concatenate(vK), (np.concatenate(rK), np.concatenate(cK))),
                                 shape=(size * size, self.N))
        self.T_I = sp.csr_matrix((np.concatenate(vI), (np.concatenate(rI), np.concatenate(cI))),
                                 shape=(size * size, self.N))
        self.size_L = size

        # Circle equalities, for monomials m of degree <= 2d - 2
        eq_rows, eq_cols, eq_vals = [], [], []
        k = 0
        for i in range(self.n):
            c, s = 2 * i, 2 * i + 1
            for m in monomials(nv, 2 * self.d - 2):
                eq_rows += [k, k, k]
                eq_cols += [self._key(m, (c, c)), self._key(m, (s, s)), self._key(m)]
                eq_vals += [1.0, 1.0, -1.0]
                k += 1
        self.E = sp.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(k, self.N))

    def margin(self, tau, solver=None, verbose=False, **solver_kwargs):
        """max eps s.t. L(G_tau y) >= eps*I, M(y) >= 0, circles, y_0 = 1.
        Returns (eps, status, seconds)."""
        t0 = time.perf_counter()
        y = cp.Variable(self.N)
        eps = cp.Variable()
        M = cp.reshape(self.S_M @ y, (self.nb_M, self.nb_M), order='F')
        L = cp.reshape((self.T_K - tau * self.T_I) @ y, (self.size_L, self.size_L), order='F')
        cons = [y[self.idx[()]] == 1, self.E @ y == 0,
                0.5 * (M + M.T) >> 0,
                0.5 * (L + L.T) - eps * np.eye(self.size_L) >> 0,
                eps <= 1.0]
        prob = cp.Problem(cp.Maximize(eps), cons)
        if solver is not None:
            prob.solve(solver=solver, verbose=verbose, **solver_kwargs)
        else:
            try:
                prob.solve(solver=cp.CLARABEL, verbose=verbose)
            except cp.error.SolverError:
                prob.solve(solver=cp.SCS, verbose=verbose)
        return (float(eps.value) if eps.value is not None else float('nan'),
                prob.status, time.perf_counter() - t0)

    def bound(self, lo, hi, rel_tol=1e-4, solver=None, log=None):
        """Bisection for the largest tau with margin >= 0, in [lo, hi].
        Requires margin(lo) >= 0 (e.g. lo = an achieved value)."""
        while (hi - lo) > rel_tol * hi:
            mid = 0.5 * (lo + hi)
            e, st, sec = self.margin(mid, solver)
            if log:
                log(f"    tau {mid:.6f}: eps {e:+.3e} ({st}, {sec:.0f}s)")
            if e >= 0:
                lo = mid
            else:
                hi = mid
        return hi


class LinearMultiplierRelaxation:
    """
    A cheap lifted relaxation using only degree-2 moments. On the circle,
    any linear form l(x) = 1 + <u, x_i> (u a unit vector) is >= 0, and so
    are arc cuts <x_i, m> - cos(w/2). Multiplying the stiffness constraint
    G_tau(x) = K(x) - tau*I >= 0 by such an l gives l(x) G_tau(x) >= 0,
    which is valid for every real design and, in moment form, involves the
    products x_i x_j:

        l(x) G(x) -> (A0 - tau I) l0 + sum_j A_j (l0 y_j + sum_c l_c Y_{c j})
                     + sum_c l_c y_c (A0 - tau I)

    where l(x) = l0 + sum_c l_c x_c, y = first moments, Y = second moments.
    Together with the 19x19 moment matrix [1 y'; y Y] >= 0 and the circle
    equalities Y_cc + Y_ss = 1, this is a valid relaxation that is tighter
    than the disk bound (multiplier l = 1 recovers it). It sits between
    order 1 and order 2 of the moment hierarchy.

    multipliers: list of (l0, {var_index: coeff}) linear forms.
    """

    def __init__(self, A0, A, multipliers):
        self.A0 = np.asarray(A0, dtype=float)
        self.A = [np.asarray(a, dtype=float) for a in A]
        self.r = self.A0.shape[0]
        self.nv = len(self.A)
        self.mult = [(1.0, {})] + list(multipliers)

    @staticmethod
    def direction_multipliers(n_couplings, n_dirs):
        out = []
        for i in range(n_couplings):
            for k in range(n_dirs):
                a = 2 * np.pi * k / n_dirs
                out.append((1.0, {2 * i: np.cos(a), 2 * i + 1: np.sin(a)}))
        return out

    def margin(self, tau, solver=None, **solver_kwargs):
        nv, r = self.nv, self.r
        y = cp.Variable(nv)
        Y = cp.Variable((nv, nv), symmetric=True)
        eps = cp.Variable()
        M = cp.bmat([[np.ones((1, 1)), cp.reshape(y, (1, nv), order='F')],
                     [cp.reshape(y, (nv, 1), order='F'), Y]])
        G0 = self.A0 - tau * np.eye(r)
        cons = [M >> 0, eps <= 1.0]
        cons += [Y[2 * i, 2 * i] + Y[2 * i + 1, 2 * i + 1] == 1 for i in range(nv // 2)]
        for l0, lc in self.mult:
            expr = l0 * G0
            for j in range(nv):
                coef = l0 * y[j]
                for c, lv in lc.items():
                    coef = coef + lv * Y[c, j]
                expr = expr + coef * self.A[j]
            for c, lv in lc.items():
                expr = expr + (lv * y[c]) * G0
            cons.append(0.5 * (expr + expr.T) - eps * np.eye(r) >> 0)
        prob = cp.Problem(cp.Maximize(eps), cons)
        t0 = time.perf_counter()
        if solver is not None:
            prob.solve(solver=solver, **solver_kwargs)
        else:
            try:
                prob.solve(solver=cp.CLARABEL)
            except cp.error.SolverError:
                prob.solve(solver=cp.SCS)
        return (float(eps.value) if eps.value is not None else float('nan'),
                prob.status, time.perf_counter() - t0)

    def bound(self, lo, hi, rel_tol=1e-4, log=None):
        while (hi - lo) > rel_tol * hi:
            mid = 0.5 * (lo + hi)
            e, st, sec = self.margin(mid)
            if log:
                log(f"    tau {mid:.6f}: eps {e:+.3e} ({st}, {sec:.1f}s)")
            lo, hi = (mid, hi) if e >= 0 else (lo, mid)
        return hi
