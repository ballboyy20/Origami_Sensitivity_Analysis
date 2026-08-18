"""
CouplingOptimizer.py
Optimizes V-groove orientations (theta) via a two-stage lexicographic
search:

    Stage 1 maximizes lambda_min, the smallest rank-constrained eigenvalue
    of K = C^T C — see CouplingOptimizer's docstring for why
    "rank-constrained" (not just "nonzero") matters here.

    Stage 2 searches among configurations tied with stage 1's optimum
    (within numerical tolerance) to maximize log_product — the log-product
    of every genuinely locked eigenvalue, a proxy for overall/volumetric
    stiffness across ALL constrained directions, not just the worst one.
    Multiple independent stage-2 searches (n_solutions) surface distinct
    near-optimal designs instead of one arbitrary tie-broken answer.

This mirrors the professor's origami_rigidity_tool_optimization.py
two-stage approach: stage 1 never trades away worst-case stiffness for a
better-conditioned overall spectrum, because stage 2 only operates among
configurations that already match stage 1's optimum.

length_scale nondimensionalizes rotational DOFs against translational
ones before either metric is computed (see
CouplingSystem.build_constraint_matrix) — without it, the relative
weighting between "a rotation" and "a translation" in K is an accident of
whatever units the panel geometry happens to use, not a deliberate
choice, and the optimum theta isn't scale-invariant.

Optimization hierarchy (in order of implementation):
    1. Groove angle theta        ← done: one float per coupling
    2. Coupling on/off selection ← done: greedy leave-one-out pruning,
                                    see CouplingOptimizer.prune_couplings
    3. Contact location          ← next priority
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize


# ══════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════

class OptimizationResult:
    """
    Container for one candidate returned by CouplingOptimizer.optimize_theta().

    Attributes
    ----------
    optimal_thetas      : (n,) ndarray — best groove angles found (radians)
    lambda_min           : float — lambda_min at optimal_thetas
    lambda_min_initial   : float — lambda_min before optimization
    improvement          : float — lambda_min - lambda_min_initial
    primary_lambda_min   : float — the stage-1 optimum every candidate
                                    from the same optimize_theta() call is
                                    held to (identical across all
                                    candidates returned by one call)
    log_product          : float — sum(log(locked eigenvalues)) at
                                    optimal_thetas; -inf if rank-deficient
    log_product_initial  : float — log_product before optimization
    history              : dict  — {'thetas': [...], 'lambda_min': [...],
                                     'log_product': [...]}, one entry per
                                     objective evaluation belonging to
                                     THIS candidate (stage-1 trunk plus
                                     this candidate's own stage-2 run —
                                     not a union across candidates)
    n_evaluations        : int   — len(history['lambda_min'])
    converged            : bool  — stage-1 AND stage-2 scipy convergence
    """

    def __init__(self, optimal_thetas, lambda_min, lambda_min_initial,
                 primary_lambda_min, log_product, log_product_initial,
                 history, converged, n_evaluations):
        self.optimal_thetas      = np.asarray(optimal_thetas)
        self.lambda_min          = float(lambda_min)
        self.lambda_min_initial  = float(lambda_min_initial)
        self.improvement         = self.lambda_min - self.lambda_min_initial
        self.primary_lambda_min  = float(primary_lambda_min)
        self.log_product         = float(log_product)
        self.log_product_initial = float(log_product_initial)
        self.history              = history
        self.converged            = converged
        self.n_evaluations        = n_evaluations

    def __repr__(self):
        return (
            f"OptimizationResult(\n"
            f"  λ_min:       {self.lambda_min_initial:.6f}  →  "
            f"{self.lambda_min:.6f}  (primary optimum: "
            f"{self.primary_lambda_min:.6f})\n"
            f"  improvement: {self.improvement:+.6f}\n"
            f"  log-vol:     {self.log_product_initial:.6f}  →  "
            f"{self.log_product:.6f}\n"
            f"  converged:   {self.converged}\n"
            f"  evaluations: {self.n_evaluations}\n"
            f"  θ_optimal:   {np.round(np.degrees(self.optimal_thetas), 2)} deg\n"
            f")"
        )


class PruneStep:
    """
    One round of CouplingOptimizer.prune_couplings(): a full stage-1/2
    theta optimization on the active set at the time, followed by
    (usually) one coupling removal.

    Attributes
    ----------
    n_active_before            : int — active coupling count entering
                                  this round, before any removal
    theta_result                : OptimizationResult — stage 1+2 outcome
                                   for this round's active set (already
                                   applied to the system)
    removed_coupling             : KinematicCoupling or None — the
                                    coupling deactivated at the end of
                                    this round; None on the final round,
                                    where nothing was removed
    removed_was_second_choice    : bool — True if the second-least-
                                    critical coupling was removed instead
                                    of the least-critical one (see
                                    second_choice_prob)
    lambda_min_after_removal     : float or None — lambda_min of the
                                    reduced active set, at this round's
                                    thetas, immediately after removal
                                    (before the next round's
                                    reoptimization); None when nothing
                                    was removed
    log_product_after_removal    : float or None — log_product
                                    counterpart to lambda_min_after_removal
    """

    def __init__(self, n_active_before, theta_result, removed_coupling,
                 removed_was_second_choice, lambda_min_after_removal,
                 log_product_after_removal):
        self.n_active_before           = n_active_before
        self.theta_result              = theta_result
        self.removed_coupling          = removed_coupling
        self.removed_was_second_choice = removed_was_second_choice
        self.lambda_min_after_removal  = lambda_min_after_removal
        self.log_product_after_removal = log_product_after_removal


class PruneResult:
    """
    Full trajectory returned by CouplingOptimizer.prune_couplings().

    Attributes
    ----------
    steps         : list[PruneStep], one per round, in order
    stop_reason   : 'min_couplings_reached' — the active count hit
                     min_couplings before any coupling became
                     unremovable
                    'rank_floor_reached' — no active coupling could be
                     removed without dropping rank below target_rank,
                     even though more than min_couplings were still
                     active (can happen before the count-based floor —
                     see prune_couplings' docstring)
    min_couplings : int — the floor actually used (after defaulting)
    """

    def __init__(self, steps, stop_reason, min_couplings):
        self.steps         = steps
        self.stop_reason   = stop_reason
        self.min_couplings = min_couplings

    @property
    def lambda_min_trajectory(self):
        """[(n_active_couplings, lambda_min), ...], one entry per round,
        for plotting stiffness vs. coupling count."""
        return [(step.n_active_before, step.theta_result.lambda_min)
                for step in self.steps]

    def __repr__(self):
        n_removed = sum(1 for s in self.steps if s.removed_coupling is not None)
        final_n   = self.steps[-1].n_active_before   # last round removes nothing
        return (
            f"PruneResult(\n"
            f"  rounds:        {len(self.steps)}\n"
            f"  removed:       {n_removed}\n"
            f"  final active:  {final_n}\n"
            f"  stop_reason:   {self.stop_reason}\n"
            f"  min_couplings: {self.min_couplings}\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════════

class CouplingOptimizer:
    """
    Two-stage lexicographic optimizer over groove angles theta.

    Fixed during optimization
    -------------------------
    - Contact locations (p per coupling)
    - Panel geometry
    - Which couplings are active (coupling.active flag)
    - length_scale (set at construction — see __init__)

    Variable
    --------
    - theta per active coupling (one float, groove orientation in
      the mating face plane)

    Objective (stage 1 — primary)
    ------------------------------
    maximize  lambda_min( C(theta)^T C(theta) )

    where lambda_min is the smallest eigenvalue among the DOFs this
    coupling set actually locks — i.e. the (total_dofs - rank(C))-th
    eigenvalue in sorted order, NOT "smallest eigenvalue above a fixed
    magnitude threshold". A magnitude threshold silently drops whatever
    is smallest instead of penalizing it, so a configuration that loses
    rank (parallel grooves, or a disabled coupling) can look artificially
    stiffer than one that doesn't. Any configuration with rank(C) below
    target_rank returns lambda_min = 0., so under-constrained
    configurations can never outscore fully-constrained ones — critical
    for the on/off extension, where comparing configurations of
    different rank is the whole point.

    Objective (stage 2 — secondary, tie-break)
    -------------------------------------------
    maximize  log_product(theta) = sum(log(locked eigenvalues))
    subject to lambda_min(theta) >= primary_lambda_min - tol

    Stage 1 alone is blind to everything except the single worst
    direction: many distinct theta configurations can share the exact
    same optimal lambda_min while differing hugely in how stiff the
    OTHER constrained directions are (including having a second,
    almost-as-weak direction — a much more fragile design). Stage 2
    breaks that tie by maximizing the product of every locked
    eigenvalue (a volumetric/overall-stiffness proxy), searched
    n_solutions times to surface distinct near-optimal designs rather
    than whichever one the search happened to land on first.

    Stage 3 (greedy on/off selection)
    ----------------------------------
    prune_couplings() reuses this same _locked_eigenvalues() /
    lambda_min / log_product infrastructure: each round runs stages 1+2
    to convergence, then a leave-one-out pass toggles each active
    coupling.active off in turn to measure its individual contribution,
    and deactivates the least-critical one (occasionally the second-
    least, for variety) before the next round re-optimises theta on the
    smaller active set. E.g. two panels need at least 3 active couplings
    to reach full rank at all (2 couplings contribute at most 4
    constraint rows, capping rank(C) <= 4 < target_rank=6) — the
    structural floor prune_couplings() computes and never prunes below.
    """

    def __init__(self, system, target_rank=None, length_scale=1.0):
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
        length_scale : float
            Reference length used to nondimensionalize rotational DOFs
            against translational ones (see
            CouplingSystem.build_constraint_matrix) before either
            lambda_min or log_product is computed. Must be positive.
            Defaults to 1.0 (no rescaling).
        """
        if length_scale <= 0:
            raise ValueError(
                f"length_scale must be positive, got {length_scale}")

        self.system       = system
        self.target_rank  = (target_rank if target_rank is not None
                              else system.total_dofs - 6)
        self.length_scale = length_scale
        self._threshold    = float('-inf')  # set for real before stage 2 runs
        self._history       = {'thetas': [], 'lambda_min': [], 'log_product': []}

    # ── Public API ─────────────────────────────────────────────────────

    def optimize_theta(self,
                       method='differential_evolution',
                       bounds=None,
                       seed=42,
                       tol=1e-8,
                       maxiter=1000,
                       n_solutions=1):
        """
        Optimise groove angles via a two-stage lexicographic search.

        Parameters
        ----------
        method : str
            'differential_evolution'  global optimiser, recommended.
                                      Runs the full two-stage search
                                      described in the class docstring.
            'nelder_mead'             local optimiser. Faster but
                                      sensitive to starting point. No
                                      secondary/tie-break stage — that
                                      search is inherently local, not
                                      suited to lexicographic
                                      tie-breaking across a whole
                                      feasible region. Returns a
                                      single-element list for API
                                      uniformity with the DE branch.
        bounds : list of (lo, hi) or None
            Per-coupling angle bounds in radians.
            Default: (0, pi) — one full period of V-groove symmetry.
            theta and theta+pi produce identical n1, n2 (normals swap),
            so [0, pi] covers all distinct configurations.
        seed : int
            Random seed for stage 1, and the base seed for stage 2 (each
            of the n_solutions stage-2 runs uses seed + 101*run).
        tol : float
            Convergence tolerance passed to the scipy solver(s).
        maxiter : int
            Max generations (differential_evolution) or iterations
            (nelder_mead) per scipy solver call — applies to both
            stages when method='differential_evolution'.
        n_solutions : int
            Number of independent stage-2 searches to attempt
            (differential_evolution only; ignored by nelder_mead).
            Distinct candidates (>0.05 rad apart in theta-space) are
            kept; duplicates and infeasible runs are discarded.

        Returns
        -------
        list[OptimizationResult]
            Sorted by log_product, descending. Never empty for a
            successful call — raises RuntimeError if no stage-2 run
            reaches lambda_min within tolerance of the stage-1 optimum.
            Does NOT automatically apply any result to the system.
            Call apply_result(results[0]) (or any chosen candidate) to
            update coupling thetas.
        """
        active = self._active_couplings()
        n      = len(active)

        if n == 0:
            raise ValueError("No active couplings to optimise.")

        if bounds is None:
            bounds = [(0., np.pi)] * n

        # Baseline before any optimisation
        thetas_initial       = np.array([c.theta for c in active])
        lambda_min_initial   = self._compute_lambda_min(thetas_initial)
        log_product_initial  = self._compute_log_product(thetas_initial)
        if log_product_initial is None:
            log_product_initial = float('-inf')

        # Reset history
        self._history = {'thetas': [], 'lambda_min': [], 'log_product': []}

        if method == 'differential_evolution':
            return self._optimize_two_stage(
                bounds, seed, tol, maxiter, n_solutions,
                lambda_min_initial, log_product_initial)
        elif method == 'nelder_mead':
            return self._optimize_nelder_mead(
                thetas_initial, bounds, tol, maxiter,
                lambda_min_initial, log_product_initial)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose 'differential_evolution' or 'nelder_mead'.")

    def apply_result(self, result):
        """
        Apply optimal thetas from an OptimizationResult to the system.

        Important: after optimize_theta() the couplings are left with
        whatever thetas were evaluated last, which is NOT necessarily
        the optimum. Always call apply_result() with your chosen
        candidate (e.g. results[0]) before using the system.
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

    def log_product(self):
        """
        Log-product (sum of log of locked eigenvalues) for the system's
        current coupling configuration. -inf if rank-deficient.
        """
        active = self._active_couplings()
        thetas = np.array([c.theta for c in active])
        lp = self._compute_log_product(thetas)
        return lp if lp is not None else float('-inf')

    def prune_couplings(self, min_couplings=None, second_choice_prob=0.15,
                         rng_seed=123, theta_seed=42, tol=1e-8,
                         maxiter=1000, n_solutions=1):
        """
        Stage 3: greedily deactivate couplings, re-optimizing theta after
        each removal, until a minimum active-coupling floor is reached.

        Each round runs the full stage-1/2 search (optimize_theta) on the
        current active set, applies the best candidate, then measures
        every active coupling's contribution to stiffness by leave-one-
        out at those same thetas (no re-optimization per candidate — see
        _removal_candidates) and deactivates whichever coupling's removal
        would hurt lambda_min/log_product least, i.e. contributes least.
        With probability second_choice_prob, the second-least-critical
        coupling is deactivated instead, for variety across runs.

        A coupling is only a removal candidate if the system without it
        still reaches target_rank. Once no active coupling qualifies —
        which can happen strictly before the active count reaches
        min_couplings, e.g. if pruning has stripped every coupling off
        one shared edge — the search stops even though min_couplings
        hasn't been reached. See PruneResult.stop_reason.

        Parameters
        ----------
        min_couplings : int or None
            Stop once the active count is <= this. Defaults to
            ceil(target_rank / 2) — the theoretical minimum coupling
            count able to reach target_rank at all (2 constraint rows
            per coupling). Raises ValueError if given a value below that
            floor.
        second_choice_prob : float
            Probability in [0, 1] of removing the second-least-critical
            coupling instead of the least-critical one. 0 always removes
            the strict worst; 1 always removes the second-worst (only
            meaningful with >= 2 removal candidates).
        rng_seed : int
            Seeds the removal-choice randomness (numpy Generator),
            independent of theta_seed, so the same prune trajectory is
            reproducible.
        theta_seed, tol, maxiter, n_solutions
            Forwarded to optimize_theta() every round (differential
            evolution only — this method always uses that method, since
            stage 3 depends on stage 2's tie-breaking).

        Returns
        -------
        PruneResult
        """
        structural_floor = int(np.ceil(self.target_rank / 2))
        if min_couplings is None:
            min_couplings = structural_floor
        elif min_couplings < structural_floor:
            raise ValueError(
                f"min_couplings={min_couplings} is below the structural "
                f"floor ceil(target_rank/2)={structural_floor} — fewer "
                f"couplings than that can never reach target_rank="
                f"{self.target_rank} (2 rows per coupling).")

        rng = np.random.default_rng(rng_seed)
        steps = []

        while True:
            results = self.optimize_theta(
                method='differential_evolution', seed=theta_seed,
                tol=tol, maxiter=maxiter, n_solutions=n_solutions)
            best = results[0]
            self.apply_result(best)

            n_active = len(self._active_couplings())

            if n_active <= min_couplings:
                steps.append(PruneStep(n_active, best, None, False, None, None))
                return PruneResult(steps, 'min_couplings_reached', min_couplings)

            candidates = self._removal_candidates()
            if not candidates:
                steps.append(PruneStep(n_active, best, None, False, None, None))
                return PruneResult(steps, 'rank_floor_reached', min_couplings)

            (chosen, lam_after, logp_after), was_second = self._select_removal(
                candidates, rng, second_choice_prob)
            chosen.active = False

            steps.append(PruneStep(
                n_active, best, chosen, was_second, lam_after, logp_after))

    # ── Private helpers ────────────────────────────────────────────────

    def _active_couplings(self):
        """Return couplings with active=True (default True if unset)."""
        return [c for c in self.system.couplings
                if getattr(c, 'active', True)]

    def _removal_candidates(self):
        """
        Leave-one-out stiffness impact of removing each currently active
        coupling, at each coupling's own current theta (no
        re-optimization — that happens at the start of the next
        prune_couplings() round via optimize_theta).

        Returns
        -------
        list of (coupling, lambda_min_without, log_product_without)
        — one entry per active coupling whose removal keeps rank(C) >=
        target_rank. Couplings whose removal would make the system
        rank-deficient are omitted entirely (not real candidates this
        round).
        """
        candidates = []
        for coupling in self._active_couplings():
            coupling.active = False
            remaining = self._active_couplings()
            remaining_thetas = np.array([c.theta for c in remaining])
            _, locked = self._locked_eigenvalues(remaining_thetas)
            coupling.active = True

            if locked is None:
                continue   # removing this coupling loses rank — not removable

            lam  = float(locked[0])
            logp = float(np.sum(np.log(np.maximum(locked, self._log_floor(locked)))))
            candidates.append((coupling, lam, logp))

        return candidates

    @staticmethod
    def _select_removal(candidates, rng, second_choice_prob):
        """
        Pick which removal candidate to actually deactivate.

        Ranks candidates descending by (lambda_min_without,
        log_product_without) — the same lexicographic priority stage 1/2
        already use — so the top-ranked candidate is the coupling whose
        removal hurts stiffness least, i.e. contributes least. Usually
        that one is chosen; with probability second_choice_prob the
        second-ranked candidate is chosen instead (only possible with
        >= 2 candidates), so pruning doesn't always walk the single
        greedy path.

        Parameters
        ----------
        candidates          : list of (coupling, lambda_min_without,
                               log_product_without), as returned by
                               _removal_candidates() — must be non-empty
        rng                 : numpy.random.Generator
        second_choice_prob  : float in [0, 1]

        Returns
        -------
        (chosen_candidate, was_second_choice)
        """
        ranked = sorted(candidates, key=lambda item: (item[1], item[2]),
                        reverse=True)
        if len(ranked) >= 2 and rng.random() < second_choice_prob:
            return ranked[1], True
        return ranked[0], False

    def _locked_eigenvalues(self, thetas):
        """
        Set groove angles, rebuild C (at self.length_scale), and return
        (rank, locked_eigs) — the one source of truth both lambda_min and
        log_product are derived from.

        locked_eigs is None whenever the configuration is under-
        constrained (rank < target_rank), including the "no couplings"
        case. Otherwise it's the `rank`-length slice of genuinely-locked
        eigenvalues — eigs[n_free:], where n_free = total_dofs - rank
        skips exactly the unavoidable rigid-body gauge modes (always
        ~0, never physically meaningful), NOT a magnitude-threshold
        filter (see the class docstring for why that distinction
        matters: a magnitude filter can't tell a weak-but-constrained
        direction apart from a newly-opened free one).
        """
        active = self._active_couplings()
        for coupling, theta in zip(active, thetas):
            coupling.set_theta(float(theta))

        C = self.system.build_constraint_matrix(length_scale=self.length_scale)
        if C.shape[0] == 0:
            return 0, None

        try:
            rank = np.linalg.matrix_rank(C)
        except np.linalg.LinAlgError:
            # SVD occasionally fails to converge for a handful of the many
            # thousands of matrices a DE search evaluates (a LAPACK
            # numerical edge case, not a modeling error). Treat this
            # theta as infeasible rather than crashing the whole search.
            return 0, None
        if rank < self.target_rank:
            return rank, None

        K = C.T @ C
        eigs   = np.sort(np.linalg.eigvalsh(K))
        n_free = C.shape[1] - rank
        return rank, eigs[n_free:]

    @staticmethod
    def _log_floor(locked):
        """
        Floor for the log-product's argument, scaled to the locked
        eigenvalues' own magnitude rather than a bare constant.

        eigvalsh noise can leave a "locked" eigenvalue at a tiny but
        nonzero value even when rank == target_rank (matrix_rank and
        eigvalsh use different algorithms/tolerances). A fixed floor
        isn't safe across the full length_scale range this optimizer
        supports: length_scale as low as 0.1 inflates the rotational
        block of K by up to 100x, moving any constant noise floor out
        of a safe relative range. This floor is used ONLY inside the
        log computation — never applied to lambda_min itself, which
        should report its true (possibly tiny) value.
        """
        return max(1e-12, 1e-10 * float(np.max(locked)))

    def _compute_lambda_min(self, thetas):
        _, locked = self._locked_eigenvalues(thetas)
        return float(locked[0]) if locked is not None else 0.0

    def _compute_log_product(self, thetas):
        _, locked = self._locked_eigenvalues(thetas)
        if locked is None:
            return None
        floor = self._log_floor(locked)
        return float(np.sum(np.log(np.maximum(locked, floor))))

    def _record_and_evaluate(self, thetas):
        """
        Single shared eigendecomposition per DE evaluation: derives both
        lambda_min and log_product from one _locked_eigenvalues() call
        (rather than each objective independently recomputing it), and
        records all three in self._history.
        """
        _, locked = self._locked_eigenvalues(thetas)
        if locked is None:
            lam, logp = 0.0, float('-inf')
        else:
            lam  = float(locked[0])
            logp = float(np.sum(np.log(np.maximum(locked, self._log_floor(locked)))))

        self._history['thetas'].append(np.asarray(thetas, dtype=float).copy())
        self._history['lambda_min'].append(lam)
        self._history['log_product'].append(logp)
        return lam, logp

    def _objective(self, thetas):
        """Stage-1 objective for scipy: returns -lambda_min."""
        lam, _ = self._record_and_evaluate(thetas)
        return -lam

    def _secondary_objective(self, thetas):
        """
        Stage-2 objective for scipy: penalty-barrier on lambda_min
        falling below self._threshold, else -log_product. This is a
        lexicographic guarantee, not a weighted trade-off: no amount of
        log_product improvement can compensate for missing the stage-1
        floor, because the penalty (1e6 baseline + steep linear term)
        is always larger than any feasible -log_product value.
        """
        lam, logp = self._record_and_evaluate(thetas)
        if lam < self._threshold:
            return 1e6 + 1e8 * (self._threshold - lam)
        return -logp

    def _optimize_two_stage(self, bounds, seed, tol, maxiter, n_solutions,
                             lambda_min_initial, log_product_initial):
        # ── Stage 1: maximise lambda_min ────────────────────────────────
        primary = differential_evolution(
            self._objective,
            bounds  = bounds,
            tol     = tol,
            maxiter = maxiter,
            seed    = seed,
            polish  = True,   # L-BFGS-B polish of the best DE member
            workers = 1,      # required: _objective mutates self.system
                              # and self._history, so evaluations must
                              # stay in-process
        )
        primary_end     = len(self._history['lambda_min'])
        best_lambda_min = -primary.fun
        self._threshold = best_lambda_min - max(1e-9, 1e-7 * best_lambda_min)

        # ── Stage 2: among near-ties, maximise log_product ──────────────
        candidates = []
        for run in range(max(1, n_solutions)):
            run_start = len(self._history['lambda_min'])
            secondary = differential_evolution(
                self._secondary_objective,
                bounds  = bounds,
                tol     = tol,
                maxiter = maxiter,
                seed    = seed + 101 * run,
                polish  = True,
                workers = 1,   # _secondary_objective mutates shared state
                               # exactly like _objective does above
            )
            run_end = len(self._history['lambda_min'])

            _, locked = self._locked_eigenvalues(secondary.x)
            if locked is None:
                continue
            lam = float(locked[0])
            if lam < self._threshold:
                continue
            if any(np.linalg.norm(secondary.x - c.optimal_thetas) < 0.05
                   for c in candidates):
                continue   # duplicate of an already-accepted candidate

            log_prod = float(np.sum(np.log(np.maximum(
                locked, self._log_floor(locked)))))
            cand_history = {
                key: (self._history[key][:primary_end]
                      + self._history[key][run_start:run_end])
                for key in self._history
            }
            candidates.append(OptimizationResult(
                optimal_thetas      = secondary.x,
                lambda_min          = lam,
                lambda_min_initial  = lambda_min_initial,
                primary_lambda_min  = best_lambda_min,
                log_product         = log_prod,
                log_product_initial = log_product_initial,
                history             = cand_history,
                converged           = bool(primary.success and secondary.success),
                n_evaluations       = len(cand_history['lambda_min']),
            ))

        if not candidates:
            raise RuntimeError(
                f"No stage-2 candidate reached the stage-1 optimum "
                f"(best_lambda_min={best_lambda_min:.6g}, "
                f"threshold={self._threshold:.6g}, "
                f"n_solutions={n_solutions}). "
                f"Try increasing maxiter or n_solutions.")

        candidates.sort(key=lambda r: r.log_product, reverse=True)
        return candidates

    def _optimize_nelder_mead(self, thetas_initial, bounds, tol, maxiter,
                               lambda_min_initial, log_product_initial):
        result = minimize(
            self._objective,
            thetas_initial.copy(),
            method  = 'Nelder-Mead',
            bounds  = bounds,
            options = {'xatol': tol, 'fatol': tol, 'maxiter': maxiter},
        )

        _, locked = self._locked_eigenvalues(result.x)
        if locked is None:
            lam, log_prod = 0.0, float('-inf')
        else:
            lam      = float(locked[0])
            log_prod = float(np.sum(np.log(np.maximum(
                locked, self._log_floor(locked)))))

        return [OptimizationResult(
            optimal_thetas      = result.x,
            lambda_min          = lam,
            lambda_min_initial  = lambda_min_initial,
            primary_lambda_min  = lam,
            log_product         = log_prod,
            log_product_initial = log_product_initial,
            history             = dict(self._history),
            converged           = result.success,
            n_evaluations       = len(self._history['lambda_min']),
        )]
