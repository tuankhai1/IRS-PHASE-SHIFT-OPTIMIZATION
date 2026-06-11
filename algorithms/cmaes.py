"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for IRS optimization.

CMA-ES is a state-of-the-art derivative-free optimization algorithm that
adapts the full covariance matrix of a multivariate normal search distribution.

Key advantages over PSO for this problem:
    1. Learns correlations between IRS elements' phase shifts
    2. Self-adapting step size (no manual tuning of inertia/coefficients)
    3. Rotation-invariant — natural for angular optimization
    4. Strong convergence properties

Key design decisions in this implementation:
    - Screening initialization: Evaluates many random candidates plus
      analytical heuristics, then uses the best as the CMA-ES mean.
      No dependency on AO or any other optimizer.
    - Unwrapped angular space: CMA-ES internals (mean, covariance,
      evolution paths) operate in unconstrained Euclidean space.
      Angles are wrapped to [-π,π) only for fitness evaluation.
      This prevents covariance corruption from angle discontinuities.
    - Doubled population for better exploration in high dimensions.
    - Restart mechanism: If CMA-ES converges early, it restarts from
      the best-so-far with increased step size to escape local optima.

Implementation follows:
    N. Hansen, "The CMA Evolution Strategy: A Tutorial," 2016.
"""

import numpy as np
from phase_shift_model import wrap_angle, beta, reflection_vector
from objective import compute_channel_gain
from config import CMAES_MAX_ITER, CMAES_SIGMA0, CMAES_TOL, BETA_MIN, K_PARAM, PHI_PARAM

# GPU acceleration
try:
    from gpu_backend import GPUAccelerator
    _GPU_OK = True
except ImportError:
    _GPU_OK = False


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

    # Evaluate all candidates (GPU-accelerated if available)
    eval_candidates = wrap_angle(candidates)
    if discrete_set is not None:
        eval_candidates = _quantize_batch(eval_candidates, discrete_set)
    _gpu_s = GPUAccelerator(Phi, h_d) if _GPU_OK else None
    if _gpu_s is not None:
        fitness = _gpu_s.batch_channel_gain(eval_candidates, use_practical)
    else:
        fitness = compute_channel_gain(eval_candidates, Phi, h_d, use_practical)

    # Return the best candidate
    best_idx = np.argmax(fitness)
    return candidates[best_idx].copy()


def _gradient_polish(theta, Phi, h_d, use_practical, n_steps=20, lr_init=0.05):
    """
    Refine a phase shift solution using analytical gradient ascent.

    Unlike AO's coordinate descent (which optimizes one element at a time),
    gradient ascent updates ALL N elements simultaneously in the direction
    of steepest increase. This can escape coordinate-wise local optima
    where no single-element improvement exists but a multi-element step
    would improve the objective.

    The gradient is computed analytically from the channel model:
        ∂f/∂θ_n = 2 Re(∂v_n*/∂θ_n · q_n)
    where q = Φ (v^H Φ + h_d^H)^H is the back-projected effective channel.

    Parameters
    ----------
    theta : ndarray, shape (N,)
        Starting phase shift vector.
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    use_practical : bool
        Whether to use practical phase shift model.
    n_steps : int
        Maximum number of gradient ascent steps.
    lr_init : float
        Initial learning rate.

    Returns
    -------
    best_theta : ndarray, shape (N,)
        Refined phase shift vector.
    best_obj : float
        Refined channel gain.
    """
    theta = wrap_angle(theta.copy())
    best_obj = compute_channel_gain(theta, Phi, h_d, use_practical)
    best_theta = theta.copy()
    lr = lr_init

    for _ in range(n_steps):
        # ---- Compute effective channel ----
        v = reflection_vector(theta, use_practical)    # (N,)
        h_eff = v.conj() @ Phi + h_d.conj()            # (M,)

        # ---- Back-project to element space ----
        q = Phi @ h_eff.conj()                          # (N,)

        # ---- Analytical gradient ----
        if use_practical:
            # ∂v_n*/∂θ_n = (β'(θ_n) - jβ(θ_n)) · exp(-jθ_n)
            b = beta(theta)                              # (N,)
            s = (np.sin(theta - PHI_PARAM) + 1) / 2
            b_prime = ((1 - BETA_MIN) * K_PARAM *
                       np.maximum(s, 1e-20) ** (K_PARAM - 1) *
                       np.cos(theta - PHI_PARAM) / 2)
            dv_conj = (b_prime - 1j * b) * np.exp(-1j * theta)
        else:
            # Ideal model: ∂v_n*/∂θ_n = -j exp(-jθ_n)
            dv_conj = -1j * np.exp(-1j * theta)

        grad = 2.0 * np.real(dv_conj * q)               # (N,)

        # ---- Gradient ascent with adaptive step size ----
        theta_new = wrap_angle(theta + lr * grad)
        obj_new = compute_channel_gain(theta_new, Phi, h_d, use_practical)

        if obj_new > best_obj:
            best_obj = obj_new
            best_theta = theta_new.copy()
            theta = theta_new
            lr = min(lr * 1.1, 0.5)    # cautiously increase
        else:
            lr *= 0.5                   # backtrack
            theta = best_theta.copy()   # revert to best
            if lr < 1e-8:
                break                   # converged

    return best_theta, best_obj


def _run_cmaes_single(Phi, h_d, N, use_practical, discrete_set,
                       m_init, sigma_init, max_gen, tol, rng):
    """
    Run one CMA-ES session (used by the restart mechanism).

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
    gens_used : int
    """
    # ================================================================
    # Strategy Parameters (from Hansen's tutorial, with 2x population)
    # ================================================================
    # Doubled population for better exploration in angular space
    lam = 2 * (4 + int(3 * np.log(N)))        # offspring per generation
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

    # GPU-accelerated batch evaluator
    _gpu_c = GPUAccelerator(Phi, h_d) if _GPU_OK else None

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
            eval_x = _quantize_batch(eval_x, discrete_set)
        if _gpu_c is not None:
            fitness = _gpu_c.batch_channel_gain(eval_x, use_practical)
        else:
            fitness = compute_channel_gain(eval_x, Phi, h_d, use_practical)

        # ---- Sort by fitness (descending — we maximize) ----
        ranking = np.argsort(-fitness)
        x_sorted = x[ranking]
        z_sorted = z[ranking]

        # ---- Update best ----
        if fitness[ranking[0]] > best_obj:
            best_obj = fitness[ranking[0]]
            best_theta = wrap_angle(x_sorted[0].copy())
            if discrete_set is not None:
                best_theta = _quantize_batch(best_theta[np.newaxis, :], discrete_set)[0]

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
            return best_theta, best_obj, gen + 1

    return best_theta, best_obj, max_gen


def cmaes_optimize(Phi, h_d, N, use_practical=True,
                   discrete_set=None,
                   max_iter=CMAES_MAX_ITER,
                   sigma0=CMAES_SIGMA0,
                   tol=CMAES_TOL,
                   rng=None):
    """
    Multi-start CMA-ES for IRS phase shift optimization.

    Runs multiple independent CMA-ES sessions from independently
    screened starting points, each followed by gradient polishing.
    Returns the best result across all runs.

    This gives CMA-ES a structural advantage over AO:
    - AO: 1 random start → 1 local optimum
    - CMA-ES: 3 independent screened starts → best of 3 optima

    Over many channel realizations, CMA-ES consistently finds the
    best basin more often than AO's single-start approach.

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

    for start in range(n_starts):
        # ---- Independent screening for this start ----
        m_init = _screening_init(Phi, h_d, N, use_practical, discrete_set, rng)

        # Light gradient refinement on screened starting point
        if discrete_set is None:
            m_init, _ = _gradient_polish(m_init, Phi, h_d, use_practical, n_steps=5)

        sigma = min(sigma0, np.pi / 3)

        # ---- Run CMA-ES from this starting point ----
        best_theta, best_obj, _ = _run_cmaes_single(
            Phi, h_d, N, use_practical, discrete_set,
            m_init, sigma, gens_per_start, tol, rng
        )

        # ---- Heavy gradient polish on CMA-ES result ----
        if discrete_set is None:
            polished, polished_obj = _gradient_polish(
                best_theta, Phi, h_d, use_practical,
                n_steps=30, lr_init=0.08)
            if polished_obj > best_obj:
                best_theta = polished
                best_obj = polished_obj

        # ---- Track global best across all starts ----
        if best_obj > best_obj_global:
            best_obj_global = best_obj
            best_theta_global = best_theta.copy()

    return best_theta_global, best_obj_global


def _quantize_batch(positions, discrete_set):
    """
    Quantize positions to nearest discrete phase values.

    Parameters
    ----------
    positions : ndarray, shape (..., N)
    discrete_set : ndarray, shape (K,)

    Returns
    -------
    ndarray
        Quantized positions.
    """
    original_shape = positions.shape
    flat = positions.reshape(-1)
    diffs = np.abs(wrap_angle(flat[:, np.newaxis] - discrete_set[np.newaxis, :]))
    nearest_idx = np.argmin(diffs, axis=1)
    return discrete_set[nearest_idx].reshape(original_shape)
