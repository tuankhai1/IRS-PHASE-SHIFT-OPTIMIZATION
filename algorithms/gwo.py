"""
Grey Wolf Optimizer (GWO) for IRS optimization.

Simulates the social hierarchy and hunting behavior of grey wolves.
The pack is organized into four levels:
    - Alpha (α): best solution found so far
    - Beta (β): second best solution
    - Delta (δ): third best solution
    - Omega (ω): remaining wolves, guided by α, β, δ

Position update mechanism:
    D_α = |C₁·X_α - X|,  X₁ = X_α - A₁·D_α
    D_β = |C₂·X_β - X|,  X₂ = X_β - A₂·D_β
    D_δ = |C₃·X_δ - X|,  X₃ = X_δ - A₃·D_δ
    X(t+1) = (X₁ + X₂ + X₃) / 3

Control parameter a decreases linearly from 2 to 0:
    - |A| > 1 → exploration (search for prey)
    - |A| < 1 → exploitation (attack prey)

Reference:
    S. Mirjalili, S.M. Mirjalili, A. Lewis, "Grey Wolf Optimizer,"
    Advances in Engineering Software, vol. 69, pp. 46-61, 2014.
"""

import numpy as np

from config import (
    GWO_POP_SIZE, GWO_MAX_ITER,
    COMP_GWO_POP_SIZE, COMP_GWO_MAX_ITER,
    OMEGA as CIRCUIT_OMEGA, Z0,
)
from objective import compute_channel_gain, compute_rate_from_components
from phase_shift_model import wrap_angle
from circuit_model import get_component_bounds


def _update_hierarchy(fitness, positions, alpha_pos, alpha_score,
                      beta_pos, beta_score, delta_pos, delta_score):
    """
    Update α, β, δ wolves based on current population fitness.

    Uses cascading promotion: when a new α is found, old α becomes β,
    old β becomes δ. This preserves the best three solutions found.

    Parameters
    ----------
    fitness : ndarray, shape (pop_size,)
    positions : ndarray, shape (pop_size, dim)
    alpha_pos, beta_pos, delta_pos : ndarray, shape (dim,)
    alpha_score, beta_score, delta_score : float

    Returns
    -------
    Updated (alpha_pos, alpha_score, beta_pos, beta_score,
             delta_pos, delta_score).
    """
    for i in range(len(fitness)):
        if fitness[i] > alpha_score:
            # Cascade: α→β→δ
            delta_pos, delta_score = beta_pos.copy(), beta_score
            beta_pos, beta_score = alpha_pos.copy(), alpha_score
            alpha_pos, alpha_score = positions[i].copy(), float(fitness[i])
        elif fitness[i] > beta_score:
            # Cascade: β→δ
            delta_pos, delta_score = beta_pos.copy(), beta_score
            beta_pos, beta_score = positions[i].copy(), float(fitness[i])
        elif fitness[i] > delta_score:
            delta_pos, delta_score = positions[i].copy(), float(fitness[i])

    return alpha_pos, alpha_score, beta_pos, beta_score, delta_pos, delta_score


def _gwo_position_update(positions, alpha_pos, beta_pos, delta_pos,
                          a, rng):
    """
    Vectorized GWO position update for all wolves simultaneously.

    Parameters
    ----------
    positions : ndarray, shape (pop_size, dim)
    alpha_pos, beta_pos, delta_pos : ndarray, shape (dim,)
    a : float
        Linearly decreasing parameter (2 → 0).
    rng : np.random.Generator

    Returns
    -------
    new_positions : ndarray, shape (pop_size, dim)
    """
    pop_size, dim = positions.shape

    # Coefficients for α
    A1 = 2 * a * rng.random((pop_size, dim)) - a
    C1 = 2 * rng.random((pop_size, dim))
    # Coefficients for β
    A2 = 2 * a * rng.random((pop_size, dim)) - a
    C2 = 2 * rng.random((pop_size, dim))
    # Coefficients for δ
    A3 = 2 * a * rng.random((pop_size, dim)) - a
    C3 = 2 * rng.random((pop_size, dim))

    # Distance to α, β, δ
    D_alpha = np.abs(C1 * alpha_pos[np.newaxis, :] - positions)
    D_beta = np.abs(C2 * beta_pos[np.newaxis, :] - positions)
    D_delta = np.abs(C3 * delta_pos[np.newaxis, :] - positions)

    # New positions guided by α, β, δ
    X1 = alpha_pos[np.newaxis, :] - A1 * D_alpha
    X2 = beta_pos[np.newaxis, :] - A2 * D_beta
    X3 = delta_pos[np.newaxis, :] - A3 * D_delta

    return (X1 + X2 + X3) / 3.0


# ============================================================
# Phase-Level GWO (optimizes θ ∈ [-π, π]^N)
# ============================================================

def gwo_optimize(Phi, h_d, N, use_practical=True,
                 pop_size=GWO_POP_SIZE, max_iter=GWO_MAX_ITER,
                 rng=None):
    """
    Grey Wolf Optimizer for IRS phase-shift optimization.

    Optimizes θ ∈ [-π, π]^N to maximize channel gain
    F(v) = ||v^H Φ + h_d^H||².

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    N : int
        Number of IRS reflecting elements.
    use_practical : bool
        Whether to use practical phase shift model.
    pop_size : int
        Number of wolves in the pack.
    max_iter : int
        Maximum number of iterations.
    rng : np.random.Generator, optional

    Returns
    -------
    alpha_pos : ndarray, shape (N,)
        Best phase shift vector found (α wolf position).
    alpha_score : float
        Best channel gain achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Initialize wolf pack uniformly in [-π, π]
    positions = rng.uniform(-np.pi, np.pi, size=(pop_size, N))

    # Evaluate fitness (channel gain)
    fitness = compute_channel_gain(positions, Phi, h_d, use_practical)

    # Initialize α, β, δ hierarchy
    sorted_idx = np.argsort(-fitness)
    alpha_pos = positions[sorted_idx[0]].copy()
    alpha_score = float(fitness[sorted_idx[0]])
    beta_pos = positions[sorted_idx[1]].copy()
    beta_score = float(fitness[sorted_idx[1]])
    delta_pos = positions[sorted_idx[2]].copy()
    delta_score = float(fitness[sorted_idx[2]])

    for t in range(max_iter):
        # Linearly decrease a from 2 to 0
        a = 2.0 - 2.0 * t / max_iter

        # Update all wolf positions (vectorized)
        positions = _gwo_position_update(
            positions, alpha_pos, beta_pos, delta_pos, a, rng
        )

        # Wrap angles to [-π, π]
        positions = wrap_angle(positions)

        # Evaluate fitness
        fitness = compute_channel_gain(positions, Phi, h_d, use_practical)

        # Update hierarchy
        (alpha_pos, alpha_score, beta_pos, beta_score,
         delta_pos, delta_score) = _update_hierarchy(
            fitness, positions,
            alpha_pos, alpha_score,
            beta_pos, beta_score,
            delta_pos, delta_score
        )

    return alpha_pos, alpha_score


# ============================================================
# Component-Level GWO (optimizes L1, L2, C, R ∈ ℝ^{4N})
# ============================================================

def gwo_component_optimize(Phi, h_d, N,
                            pop_size=COMP_GWO_POP_SIZE,
                            max_iter=COMP_GWO_MAX_ITER,
                            omega=CIRCUIT_OMEGA, Z0_val=Z0,
                            rng=None, return_history=False):
    """
    Grey Wolf Optimizer for component-level IRS optimization.

    Optimizes x = [L1_1, L2_1, C_1, R_1, ..., L1_N, L2_N, C_N, R_N]
    to maximize R_SE = log₂(1 + P_T·||v^H Φ + h_d^H||²/σ²).

    Uses box constraints (clamping) on physical component bounds.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    N : int
        Number of IRS reflecting elements.
    pop_size : int
        Number of wolves.
    max_iter : int
        Maximum iterations.
    omega : float
        Angular frequency (rad/s).
    Z0_val : float
        Free-space impedance (Ω).
    rng : np.random.Generator, optional
    return_history : bool
        If True, also return convergence history.

    Returns
    -------
    alpha_pos : ndarray, shape (4*N,)
        Best component parameter vector.
    alpha_score : float
        Best R_SE achieved.
    history : ndarray, shape (max_iter+1,), optional
        Best-so-far R_SE per iteration (if return_history=True).
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = 4 * N
    lower, upper = get_component_bounds(N)

    # Step 1: Initialize wolves uniformly within component bounds
    positions = rng.uniform(lower, upper, size=(pop_size, dim))

    # Step 6: Evaluate fitness = R_SE
    fitness = compute_rate_from_components(
        positions, N, Phi, h_d, omega, Z0_val
    )

    # Initialize hierarchy
    sorted_idx = np.argsort(-fitness)
    alpha_pos = positions[sorted_idx[0]].copy()
    alpha_score = float(fitness[sorted_idx[0]])
    beta_pos = positions[sorted_idx[1]].copy()
    beta_score = float(fitness[sorted_idx[1]])
    delta_pos = positions[sorted_idx[2]].copy()
    delta_score = float(fitness[sorted_idx[2]])

    history = [alpha_score] if return_history else None

    for t in range(max_iter):
        # Linearly decrease a from 2 to 0
        a = 2.0 - 2.0 * t / max_iter

        # Step 7: Update positions (vectorized)
        positions = _gwo_position_update(
            positions, alpha_pos, beta_pos, delta_pos, a, rng
        )

        # Step 2: Enforce physical constraints (clamp to bounds)
        positions = np.clip(positions, lower, upper)

        # Steps 3-6: Compute Z_n → v_n → v → R_SE
        fitness = compute_rate_from_components(
            positions, N, Phi, h_d, omega, Z0_val
        )

        # Update hierarchy
        (alpha_pos, alpha_score, beta_pos, beta_score,
         delta_pos, delta_score) = _update_hierarchy(
            fitness, positions,
            alpha_pos, alpha_score,
            beta_pos, beta_score,
            delta_pos, delta_score
        )

        if return_history:
            history.append(alpha_score)

    if return_history:
        return alpha_pos, alpha_score, np.array(history)
    return alpha_pos, alpha_score
