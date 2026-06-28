"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for IRS optimization.

CMA-ES is a derivative-free optimization algorithm that adapts the
covariance matrix of a multivariate normal search distribution.

Key design decisions in the improved implementation:
    - Screening initialization: Evaluates many random candidates plus
      analytical heuristics, then uses the best as the CMA-ES mean.
      No dependency on AO or any other optimizer.
    - Unwrapped angular space: CMA-ES internals (mean, covariance,
      evolution paths) operate in unconstrained Euclidean space.
      Angles are wrapped to [-π,π) only for fitness evaluation.
      This prevents covariance corruption from angle discontinuities.
    - Multiple starts are used to reduce dependence on one initialization.

Implementation follows:
    N. Hansen, "The CMA Evolution Strategy: A Tutorial," 2016.
"""

import numpy as np
from phase_shift_model import quantize_angles, wrap_angle
from objective import compute_channel_gain
from config import CMAES_MAX_ITER, CMAES_SIGMA0, CMAES_TOL


def _screening_init(Phi, h_d, N, use_practical, discrete_set, rng):
    """
    Find a high-quality starting point by screening many random candidates.

    Generates a large pool of random phase vectors plus analytical
    heuristics (phase-alignment, anti-phase), evaluates all of them,
    and returns the best one as the CMA-ES starting mean.

    This is fundamentally different from AO:
    - No iterative coordinate descent
    - No inter-element coupling optimization
    - Just parallel evaluation of random candidates (embarrassingly parallel)

    The cost is ~10*N fitness evaluations (e.g., 400 for N=40), which is
    negligible compared to the main CMA-ES loop (300 generations × ~30
    offspring = ~9000 evaluations).

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    N : int
        Number of IRS elements.
    use_practical : bool
        Whether to use practical phase shift model.
    discrete_set : ndarray or None
        Discrete phase values, if applicable.
    rng : np.random.Generator

    Returns
    -------
    theta_best : ndarray, shape (N,)
        Best phase shift vector found from screening.
    """
    n_candidates = max(200, 10 * N)

    # Generate random candidates uniformly in [-π, π]
    candidates = rng.uniform(-np.pi, np.pi, size=(n_candidates, N))

    # Inject analytical heuristics as additional candidates
    # 1) Phase-alignment: align reflected path with direct channel
    projection = Phi @ h_d  # shape (N,)
    phase_align = np.angle(projection)
    candidates[0] = phase_align

    # 2) Anti-phase: opposite phase, sometimes better under practical
    #    model due to β(θ) amplitude-phase coupling
    candidates[1] = wrap_angle(phase_align + np.pi)

    # 3) Phase-alignment with perturbations at various scales
    for i, sigma in enumerate([0.3, 0.5, 0.8, 1.0]):
        if 2 + i < n_candidates:
            candidates[2 + i] = wrap_angle(phase_align + rng.normal(0, sigma, size=N))

    # Evaluate all candidates
    eval_candidates = wrap_angle(candidates)
    if discrete_set is not None:
        eval_candidates = quantize_angles(eval_candidates, discrete_set)
    fitness = compute_channel_gain(eval_candidates, Phi, h_d, use_practical)

    # Return the best candidate
    best_idx = np.argmax(fitness)
    return candidates[best_idx].copy()


def _run_cmaes_single(Phi, h_d, N, use_practical, discrete_set,
                      m_init, sigma_init, max_gen, tol, rng,
                      population_multiplier=2):
    """
    Run one CMA-ES session.

    Parameters
    ----------
    m_init : ndarray, shape (N,)
        Initial distribution mean.
    sigma_init : float
        Initial step size.
    max_gen : int
        Maximum generations for this session.
    tol : float
        Convergence tolerance on step size.

    Returns
    -------
    best_theta : ndarray, shape (N,)
    best_obj : float
    """
    # ================================================================
    # Strategy Parameters (from Hansen's tutorial, with 2x population)
    # ================================================================
    lam = population_multiplier * (4 + int(3 * np.log(N)))  # offspring per generation
    mu = lam // 2                             # number of parents

    # Recombination weights (log-linear)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)       # normalize
    mu_eff = 1.0 / np.sum(weights ** 2)       # variance-effective selection mass

    # Step-size control parameters
    c_sigma = (mu_eff + 2) / (N + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + c_sigma

    # Covariance matrix adaptation parameters
    c_c = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
    c_1 = 2.0 / ((N + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2) ** 2 + mu_eff))

    # Expected length of a N(0,I) random vector
    chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    # ================================================================
    # State Initialization
    # ================================================================
    m = m_init.copy().astype(float)
    sigma = sigma_init

    # Covariance: use eigendecomposition C = B D² B^T
    B = np.eye(N)                  # eigenvectors (rotation matrix)
    D = np.ones(N)                 # sqrt of eigenvalues
    C = np.eye(N)                  # covariance matrix

    # Evolution paths
    p_sigma = np.zeros(N)          # conjugate evolution path for σ
    p_c = np.zeros(N)              # evolution path for C

    # Track best solution
    best_theta = wrap_angle(m.copy())
    best_obj = -np.inf

    # Counter for eigendecomposition
    eigen_update_interval = max(1, lam // (10 * N))

    # ================================================================
    # Main Generational Loop (unwrapped angular space)
    # ================================================================
    # Key design: CMA-ES operates in unconstrained Euclidean space.
    # Angles are wrapped ONLY for fitness evaluation. This prevents
    # covariance corruption from angle discontinuities at ±π.

    for gen in range(max_gen):
        # ---- Sample offspring (unwrapped) ----
        z = rng.standard_normal((lam, N))
        y = z @ np.diag(D) @ B.T             # y_k = B D z_k
        x = m[np.newaxis, :] + sigma * y     # x_k = m + σ y_k

        # ---- Evaluate fitness (wrap only for evaluation) ----
        eval_x = wrap_angle(x)
        if discrete_set is not None:
            eval_x = quantize_angles(eval_x, discrete_set)
        fitness = compute_channel_gain(eval_x, Phi, h_d, use_practical)

        # ---- Sort by fitness (descending — we maximize) ----
        ranking = np.argsort(-fitness)
        x_sorted = x[ranking]
        # ---- Update best ----
        if fitness[ranking[0]] > best_obj:
            best_obj = fitness[ranking[0]]
            best_theta = wrap_angle(x_sorted[0].copy())
            if discrete_set is not None:
                best_theta = quantize_angles(
                    best_theta[np.newaxis, :], discrete_set
                )[0]

        # ---- Update mean (unwrapped Euclidean) ----
        m_old = m.copy()
        m = weights @ x_sorted[:mu]

        # Mean step (simple Euclidean difference, no angle wrapping)
        dm = m - m_old

        # ---- Update evolution path for σ (cumulation) ----
        C_invsqrt = B @ np.diag(1.0 / D) @ B.T
        p_sigma = (1 - c_sigma) * p_sigma + \
                  np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_invsqrt @ dm / sigma

        # ---- Update step size σ ----
        sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))

        # Prevent sigma from exploding or collapsing
        sigma = np.clip(sigma, 1e-10, 10 * np.pi)

        # ---- Heaviside function for stalling ----
        h_sigma = 1.0 if (np.linalg.norm(p_sigma) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                         (1.4 + 2.0 / (N + 1)) * chi_N else 0.0

        # ---- Update evolution path for C ----
        p_c = (1 - c_c) * p_c + \
              h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * dm / sigma

        # ---- Update covariance matrix C ----
        # Rank-one update
        rank_one = c_1 * (np.outer(p_c, p_c) +
                          (1 - h_sigma) * c_c * (2 - c_c) * C)

        # Rank-μ update (Euclidean differences, no angle wrapping)
        y_diff = np.zeros((mu, N))
        for i in range(mu):
            y_diff[i] = (x_sorted[i] - m_old) / sigma
        rank_mu = c_mu * sum(weights[i] * np.outer(y_diff[i], y_diff[i])
                             for i in range(mu))

        C = (1 - c_1 - c_mu) * C + rank_one + rank_mu

        # ---- Eigendecomposition of C (for numerical stability) ----
        if gen % eigen_update_interval == 0 or gen == max_gen - 1:
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)  # ensure positivity
            D = np.sqrt(eigenvalues)
            C = B @ np.diag(eigenvalues) @ B.T  # reconstruct

        # ---- Convergence check ----
        if sigma * np.max(D) < tol:
            return best_theta, best_obj

    return best_theta, best_obj


def cmaes_default_optimize(Phi, h_d, N, use_practical=True,
                           discrete_set=None,
                           max_iter=CMAES_MAX_ITER,
                           sigma0=CMAES_SIGMA0,
                           tol=CMAES_TOL,
                           rng=None):
    """
    Standard single-start CMA-ES baseline.

    This version starts from one random mean, uses the standard population
    size from Hansen's rule of thumb, and skips the screening/multi-start
    choices used by ``cmaes_optimize``.
    """
    if rng is None:
        rng = np.random.default_rng()

    m_init = rng.uniform(-np.pi, np.pi, size=N)
    if discrete_set is not None:
        m_init = quantize_angles(m_init[np.newaxis, :], discrete_set)[0]

    return _run_cmaes_single(
        Phi, h_d, N, use_practical, discrete_set,
        m_init, sigma0, max_iter, tol, rng,
        population_multiplier=1
    )


def cmaes_optimize(Phi, h_d, N, use_practical=True,
                   discrete_set=None,
                   max_iter=CMAES_MAX_ITER,
                   sigma0=CMAES_SIGMA0,
                   tol=CMAES_TOL,
                   rng=None):
    """
    Multi-start CMA-ES for IRS phase shift optimization.

    Runs multiple independent CMA-ES sessions from independently
    screened starting points and returns the best result across all runs.

    Notes:
    Multiple starts are an implementation choice for the improved variant.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    N : int
        Number of IRS elements (problem dimension).
    use_practical : bool
        Whether to use practical phase shift model.
    discrete_set : ndarray, optional
        If provided, quantize before evaluation.
    max_iter : int
        Maximum total number of generations (split across starts).
    sigma0 : float
        Initial step size.
    tol : float
        Convergence tolerance on step size.
    rng : np.random.Generator, optional

    Returns
    -------
    theta_best : ndarray, shape (N,)
        Best phase shift vector found.
    obj_best : float
        Best channel gain achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ================================================================
    # Multi-Start Strategy
    # ================================================================
    # Each start uses an INDEPENDENT screening (different random pool),
    # so each explores a different region of the phase space.
    n_starts = 3
    gens_per_start = max(50, max_iter // n_starts)

    best_theta_global = None
    best_obj_global = -np.inf

    for _ in range(n_starts):
        # Spawn an independent child RNG for each start so that
        # adding/removing starts doesn't alter other starts' behavior.
        start_rng = np.random.default_rng(rng.integers(0, 2**63))

        # ---- Independent screening for this start ----
        m_init = _screening_init(Phi, h_d, N, use_practical, discrete_set, start_rng)

        sigma = min(sigma0, np.pi / 3)

        # ---- Run CMA-ES from this starting point ----
        best_theta, best_obj = _run_cmaes_single(
            Phi, h_d, N, use_practical, discrete_set,
            m_init, sigma, gens_per_start, tol, start_rng
        )

        # ---- Track global best across all starts ----
        if best_obj > best_obj_global:
            best_obj_global = best_obj
            best_theta_global = best_theta.copy()

    return best_theta_global, best_obj_global
