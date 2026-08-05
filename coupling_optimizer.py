"""
CouplingOptimizer.py
Optimizes V-groove orientations (theta) to maximize the smallest
rank-constrained eigenvalue of K = C^T C — the rigidity metric for a
kinematic coupling system. See CouplingOptimizer's docstring for why
"rank-constrained" (not just "nonzero") matters here.

Optimization hierarchy (in order of implementation):
    1. Groove angle theta        ← current: one float per coupling
    2. Coupling on/off selection ← next: binary per coupling (active flag)
    3. Contact location          ← last priority
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize


# ══════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════

class OptimizationResult:
    """
    Container for the output of CouplingOptimizer.optimize_theta().

    Attributes
    ----------
    optimal_thetas     : (n,) ndarray — best groove angles found (radians)
    lambda_min         : float        — lambda_min at optimal_thetas
    lambda_min_initial : float        — lambda_min before optimization
    improvement        : float        — lambda_min - lambda_min_initial
    history            : dict         — {'thetas': [...], 'lambda_min': [...]}
                                        one entry per objective evaluation
    n_evaluations      : int          — total objective function calls
    converged          : bool         — scipy convergence flag
    """

    def __init__(self, optimal_thetas, lambda_min, lambda_min_initial,
                 history, converged, n_evaluations):
        self.optimal_thetas     = np.asarray(optimal_thetas)
        self.lambda_min         = float(lambda_min)
        self.lambda_min_initial = float(lambda_min_initial)
        self.improvement        = self.lambda_min - self.lambda_min_initial
        self.history            = history
        self.converged          = converged
        self.n_evaluations      = n_evaluations

    def __repr__(self):
        return (
            f"OptimizationResult(\n"
            f"  λ_min:       {self.lambda_min_initial:.6f}  →  "
            f"{self.lambda_min:.6f}\n"
            f"  improvement: {self.improvement:+.6f}\n"
            f"  converged:   {self.converged}\n"
            f"  evaluations: {self.n_evaluations}\n"
            f"  θ_optimal:   {np.round(np.degrees(self.optimal_thetas), 2)} deg\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════════

class CouplingOptimizer:
    """
    Maximizes lambda_min of K = C^T C over groove angles theta.

    Fixed during optimization
    -------------------------
    - Contact locations (p per coupling)
    - Panel geometry
    - Which couplings are active (coupling.active flag)

    Variable
    --------
    - theta per active coupling (one float, groove orientation in
      the mating face plane)

    Objective
    ---------
    maximize  lambda_min( C(theta)^T C(theta) )

    where lambda_min is the smallest eigenvalue among the DOFs this
    coupling set actually locks — i.e. the (total_dofs - rank(C))-th
    eigenvalue in sorted order, NOT "smallest eigenvalue above a fixed
    magnitude threshold". A magnitude threshold silently drops whatever
    is smallest instead of penalizing it, so a configuration that loses
    rank (parallel grooves, or a disabled coupling) can look artificially
    stiffer than one that doesn't. Any configuration with rank(C) below
    target_rank returns 0. outright, so under-constrained configurations
    can never outscore fully-constrained ones — critical for the future
    on/off extension below, where comparing configurations of different
    rank is the whole point.

    Future extension
    ----------------
    Greedy on/off selection uses the same _compute_lambda_min()
    infrastructure — just toggle coupling.active and re-optimise theta.
    Note two panels need at least 3 active couplings to reach full rank
    at all (2 couplings contribute at most 4 constraint rows, capping
    rank(C) <= 4 < target_rank=6), so the greedy search must not prune
    below that floor.
    """

    def __init__(self, system, target_rank=None):
        """
        Parameters
        ----------
        system : CouplingSystem
            The panel + coupling system to optimise.
            Active couplings are those with coupling.active == True
            (default True for all couplings unless explicitly set).
        target_rank : int or None
            The constraint-matrix rank a fully-locked assembly should
            reach. Defaults to ``system.total_dofs - 6``, i.e. every
            relative DOF constrained except the 6 unavoidable rigid-body
            gauge modes of the whole assembly — correct for any number
            of panels as long as they form one connected rigid group.
            Override this if couplings can be turned off in a way that
            splits the assembly into multiple disconnected rigid groups
            (each such group needs its own 6 gauge DOFs subtracted).
        """
        self.system      = system
        self.target_rank = (target_rank if target_rank is not None
                             else system.total_dofs - 6)
        self._history     = {'thetas': [], 'lambda_min': []}

    # ── Public API ─────────────────────────────────────────────────────

    def optimize_theta(self,
                       method='differential_evolution',
                       bounds=None,
                       seed=42,
                       tol=1e-8):
        """
        Optimise groove angles to maximise lambda_min.

        Parameters
        ----------
        method : str
            'differential_evolution'  global optimiser, recommended.
                                      Explores the full landscape before
                                      polishing with Nelder-Mead.
            'nelder_mead'             local optimiser. Faster but
                                      sensitive to starting point.
                                      Good for refining a known solution.
        bounds : list of (lo, hi) or None
            Per-coupling angle bounds in radians.
            Default: (0, pi) — one full period of V-groove symmetry.
            theta and theta+pi produce identical n1, n2 (normals swap),
            so [0, pi] covers all distinct configurations.
        seed : int
            Random seed for differential_evolution reproducibility.
        tol : float
            Convergence tolerance passed to the scipy solver.

        Returns
        -------
        OptimizationResult
            Does NOT automatically apply the result to the system.
            Call apply_result(result) to update coupling thetas.
        """
        active = self._active_couplings()
        n      = len(active)

        if n == 0:
            raise ValueError("No active couplings to optimise.")

        if bounds is None:
            bounds = [(0., np.pi)] * n

        # Baseline before any optimisation
        thetas_initial     = np.array([c.theta for c in active])
        lambda_min_initial = self._compute_lambda_min(thetas_initial)

        # Reset history
        self._history = {'thetas': [], 'lambda_min': []}

        # ── Run optimiser ────────────────────────────────────────────
        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective,
                bounds  = bounds,
                tol     = tol,
                maxiter = 1000,
                polish  = True,   # L-BFGS-B polish of the best DE member
                workers = 1,      # required: _objective mutates self.system
                                  # and self._history, so evaluations must
                                  # stay in-process (see _compute_lambda_min)
            )
            converged = result.success

        elif method == 'nelder_mead':
            x0 = thetas_initial.copy()
            result = minimize(
                self._objective,
                x0,
                method  = 'Nelder-Mead',
                bounds  = bounds,
                options = {'xatol': tol, 'fatol': tol, 'maxiter': 10000},
            )
            converged = result.success

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose 'differential_evolution' or 'nelder_mead'.")

        # scipy minimises, so result.fun = -lambda_min
        return OptimizationResult(
            optimal_thetas     = result.x,
            lambda_min         = -result.fun,
            lambda_min_initial = lambda_min_initial,
            history            = dict(self._history),
            converged          = converged,
            n_evaluations      = len(self._history['lambda_min']),
        )

    def apply_result(self, result):
        """
        Apply optimal thetas from an OptimizationResult to the system.

        Important: after optimize_theta() the couplings are left with
        whatever thetas were evaluated last, which is NOT necessarily
        the optimum. Always call apply_result() before using the system.
        """
        active = self._active_couplings()
        if len(active) != len(result.optimal_thetas):
            raise ValueError(
                "Result has different number of thetas than active couplings. "
                "Did the active set change between optimize and apply?")
        for coupling, theta in zip(active, result.optimal_thetas):
            coupling.set_theta(float(theta))

    def lambda_min(self):
        """
        Lambda_min for the system's current coupling configuration.
        Useful for checking before/after apply_result().
        """
        active = self._active_couplings()
        thetas = np.array([c.theta for c in active])
        return self._compute_lambda_min(thetas)

    # ── Private helpers ────────────────────────────────────────────────

    def _active_couplings(self):
        """Return couplings with active=True (default True if unset)."""
        return [c for c in self.system.couplings
                if getattr(c, 'active', True)]

    def _objective(self, thetas):
        """
        Objective function for scipy: returns -lambda_min.
        Records each evaluation in self._history.
        """
        lam = self._compute_lambda_min(thetas)
        self._history['thetas'].append(thetas.copy())
        self._history['lambda_min'].append(lam)
        return -lam

    def _compute_lambda_min(self, thetas):
        """
        Set groove angles, rebuild C, return the smallest eigenvalue
        among the DOFs this coupling set actually locks.

        Why not just take eigs.min() or filter by a magnitude threshold
        (`eigs > EIG_TOL`)? Both break the moment two configurations have
        different rank — e.g. rotating two grooves into parallel, or (for
        the planned on/off search) disabling a coupling. Either change can
        remove a weak-but-constrained direction and open a brand-new
        *free* one in the same stroke. A magnitude filter can't tell the
        two apart: it silently drops the weak eigenvalue from view and
        never penalizes the zero mode left behind, so an under-constrained
        system can score HIGHER than a fully constrained one. (Confirmed
        empirically: disabling one coupling in a 3-groove test system
        raised the naive lambda_min from 0.0133 to 0.098, even though the
        system went from fully locked to a 2-DOF mechanism.)

        So instead this is rank-aware:
          1. rank(C) tells us exactly how many DOFs are locked. Anything
             short of target_rank is under-constrained — return 0. (same
             convention as the "no couplings" case), so the optimizer can
             never prefer an under-constrained configuration over a fully
             constrained one.
          2. Once rank == target_rank, the first `n_free = total_dofs -
             rank` eigenvalues are the assembly's unavoidable rigid-body
             gauge modes (always ~0, never physically meaningful) — skip
             those and report eigs[n_free], the weakest genuinely-locked
             direction. With the default target_rank (total_dofs - 6),
             n_free == 6, so this is eigs[6] — the 7th eigenvalue.
        """
        active = self._active_couplings()
        for coupling, theta in zip(active, thetas):
            coupling.set_theta(float(theta))

        C = self.system.build_constraint_matrix()
        if C.shape[0] == 0:
            return 0.

        rank = np.linalg.matrix_rank(C)
        if rank < self.target_rank:
            return 0.

        K = C.T @ C
        eigs   = np.sort(np.linalg.eigvalsh(K))
        n_free = C.shape[1] - rank
        return float(eigs[n_free])