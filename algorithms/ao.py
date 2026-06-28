"""
Alternating Optimization (AO) — the baseline algorithm from the paper.

Solves the IRS phase shift optimization by iteratively optimizing one
element's phase shift at a time with all others fixed.

Two methods for solving the per-element sub-problem (P2):
    1. Proposition 1 — Closed-form quadratic curve fitting (fast)
    2. 1D Search — Exhaustive search over discretized [-π, π] (accurate)

Reference: Algorithm 1, Section IV in the paper.
"""

import numpy as np
from phase_shift_model import beta, quantize_angles, reflection_vector
from objective import compute_channel_gain
from config import AO_MAX_ITER, AO_TOL, AO_1D_SEARCH_POINTS


# ============================================================
# Precomputed lookup table for 1D trust-region search
# ============================================================
_SEARCH_FRACTIONS = np.linspace(0.0, 1.0, AO_1D_SEARCH_POINTS, endpoint=True)


def _compute_psi_and_hd_hat(Phi, h_d):
    """
    Precompute matrices used in the AO algorithm.

    Ψ = Φ Φ^H  (N × N)
    ĥ_d = Φ h_d  (N,)

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
    h_d : ndarray, shape (M,)

    Returns
    -------
    Psi : ndarray, shape (N, N)
    hd_hat : ndarray, shape (N,)
    """
    Psi = Phi @ Phi.conj().T   # N × N
    hd_hat = Phi @ h_d          # N
    return Psi, hd_hat


def _compute_phi_n(n, v, Psi, hd_hat):
    """
    Compute ϕ_n for the n-th element's sub-problem.

    ϕ_n = 2 * (Σ_{m≠n} Ψ_{n,m} v_m  +  ĥ_{d,n})

    Parameters
    ----------
    n : int
        Element index.
    v : ndarray, shape (N,)
        Current reflection vector.
    Psi : ndarray, shape (N, N)
    hd_hat : ndarray, shape (N,)

    Returns
    -------
    complex
        The value ϕ_n.
    """
    # Sum Ψ_{n,m} * v_m for m ≠ n
    psi_sum = Psi[n, :] @ v - Psi[n, n] * v[n]
    return 2 * (psi_sum + hd_hat[n])


def _solve_p2_proposition1(psi_nn, phi_n):
    """
    Solve sub-problem (P2) using Proposition 1.

    Reference: Proposition 1, Section IV-C in the paper.

    Parameters
    ----------
    psi_nn : float
        Ψ_{n,n}
    phi_n : complex
        ϕ_n

    Returns
    -------
    float
        Optimal phase shift θ*_n.
    """
    phi_n_abs = np.abs(phi_n)
    phi_n_arg = np.angle(phi_n)

    theta_c = np.pi if phi_n_arg >= 0 else -np.pi
    theta_a = phi_n_arg
    theta_b = 0.5 * (theta_a + theta_c)

    def p2_value(theta):
        b = beta(theta)
        return b ** 2 * psi_nn + b * phi_n_abs * np.cos(phi_n_arg - theta)

    f1 = p2_value(theta_a)
    f2 = p2_value(theta_b)
    f3 = p2_value(theta_c)
    denom = 4 * (f1 - 2 * f2 + f3)

    if abs(denom) < 1e-20:
        candidates = np.array([theta_a, theta_b, theta_c])
        values = np.array([f1, f2, f3])
        return candidates[np.argmax(values)]

    theta_hat = (
        theta_c * (3 * f1 - 4 * f2 + f3)
        + theta_a * (f1 - 4 * f2 + 3 * f3)
    ) / denom

    lo, hi = (theta_a, theta_c) if theta_a <= theta_c else (theta_c, theta_a)
    return float(np.clip(theta_hat, lo, hi))


def _solve_p2_1d_search(psi_nn, phi_n, discrete_set=None):
    """
    Solve sub-problem (P2) via exhaustive 1D search.

    Parameters
    ----------
    psi_nn : float
        Ψ_{n,n}
    phi_n : complex
        ϕ_n
    discrete_set : ndarray, optional
        If provided, search only over these discrete phase values.

    Returns
    -------
    float
        Optimal phase shift θ*_n.
    """
    phi_n_abs = np.abs(phi_n)
    phi_n_arg = np.angle(phi_n)

    if discrete_set is not None:
        # Discrete set: compute beta on the fly (different candidates each time)
        b = beta(discrete_set)
        cos_diff = (np.cos(phi_n_arg) * np.cos(discrete_set) +
                    np.sin(phi_n_arg) * np.sin(discrete_set))
        f_vals = b ** 2 * psi_nn + b * phi_n_abs * cos_diff
        return discrete_set[np.argmax(f_vals)]
    else:
        theta_c = np.pi if phi_n_arg >= 0 else -np.pi
        candidates = phi_n_arg + (theta_c - phi_n_arg) * _SEARCH_FRACTIONS
        b = beta(candidates)
        f_vals = b ** 2 * psi_nn + b * phi_n_abs * np.cos(phi_n_arg - candidates)
        return candidates[np.argmax(f_vals)]


def ao_optimize(Phi, h_d, N, method='prop1', use_practical=True,
                discrete_set=None, max_iter=AO_MAX_ITER, tol=AO_TOL,
                rng=None, theta_init=None):
    """
    Alternating Optimization (AO) for IRS phase shift optimization.

    Iteratively optimizes one element at a time using either
    Proposition 1 (quadratic fit) or 1D search.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    N : int
        Number of IRS elements.
    method : str
        'prop1' for Proposition 1, '1d_search' for exhaustive search.
    use_practical : bool
        Whether to use practical phase shift model.
    discrete_set : ndarray, optional
        Discrete phase values (for Fig. 7 scenario).
    max_iter : int
        Maximum number of outer iterations.
    tol : float
        Convergence tolerance.
    rng : np.random.Generator, optional
    theta_init : ndarray, shape (N,), optional
        If provided, use this as the initial phase shift vector
        instead of random initialization. Enables warm-starting
        from a metaheuristic's best solution.

    Returns
    -------
    theta_opt : ndarray, shape (N,)
        Optimized phase shifts.
    obj_best : float
        Best channel gain achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Precompute
    Psi, hd_hat = _compute_psi_and_hd_hat(Phi, h_d)

    # Initialize
    if theta_init is not None:
        theta = theta_init.copy()
    elif discrete_set is not None:
        theta = rng.choice(discrete_set, size=N)
    elif use_practical:
        theta = rng.choice(np.array([-np.pi, np.pi]), size=N)
    else:
        theta = rng.uniform(-np.pi, np.pi, size=N)

    if discrete_set is not None:
        theta = quantize_angles(theta, discrete_set)

    v = reflection_vector(theta, use_practical)
    obj_prev = compute_channel_gain(theta, Phi, h_d, use_practical)

    for _ in range(max_iter):
        for n in range(N):
            phi_n = _compute_phi_n(n, v, Psi, hd_hat)
            psi_nn = np.real(Psi[n, n])

            if discrete_set is not None:
                theta[n] = _solve_p2_1d_search(psi_nn, phi_n,
                                                discrete_set=discrete_set)
            elif not use_practical:
                theta[n] = np.angle(phi_n)
            elif method == '1d_search':
                theta[n] = _solve_p2_1d_search(psi_nn, phi_n)
            else:
                theta[n] = _solve_p2_proposition1(psi_nn, phi_n)

            v[n] = reflection_vector(np.array([theta[n]]), use_practical)[0]

        # Check convergence
        obj_curr = compute_channel_gain(theta, Phi, h_d, use_practical)
        if obj_prev > 0 and abs(obj_curr - obj_prev) / obj_prev < tol:
            break
        obj_prev = obj_curr

    return theta, obj_curr
