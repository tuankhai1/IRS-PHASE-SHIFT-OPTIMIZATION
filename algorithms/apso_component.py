"""
Adaptive PSO (APSO) for component-level IRS optimization.

Identical to standard PSO except for the inertia weight schedule:
    w(t) = w_max - (w_max - w_min) × t / max_iter

This linearly decreases inertia from 0.9 (exploration) to 0.4
(exploitation) over the course of optimization.

Reference:
    Y. Shi, R. Eberhart, "A modified particle swarm optimizer,"
    Proc. IEEE International Conference on Evolutionary Computation,
    pp. 69-73, 1998.
"""

import numpy as np

from config import (
    COMP_APSO_POP_SIZE, COMP_APSO_MAX_ITER,
    COMP_APSO_W_MAX, COMP_APSO_W_MIN,
    APSO_C1, APSO_C2,
    OMEGA as CIRCUIT_OMEGA, Z0,
)
from circuit_model import get_component_bounds
from objective import compute_rate_from_components


def apso_component_optimize(
    Phi, h_d, N,
    pop_size=COMP_APSO_POP_SIZE,
    max_iter=COMP_APSO_MAX_ITER,
    w_max=COMP_APSO_W_MAX,
    w_min=COMP_APSO_W_MIN,
    c1=APSO_C1, c2=APSO_C2,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    rng=None, return_history=False,
):
    """
    Adaptive PSO with linearly decreasing inertia weight.

    w(t) = w_max - (w_max - w_min) × t / max_iter
         = 0.9  → 0.4 over iterations

    Early iterations: high inertia → large velocity → exploration.
    Late iterations:  low inertia → small velocity → exploitation.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    N : int
        Number of IRS reflecting elements.
    pop_size : int
        Number of particles.
    max_iter : int
        Maximum iterations.
    w_max : float
        Initial inertia weight (default 0.9).
    w_min : float
        Final inertia weight (default 0.4).
    c1, c2 : float
        Cognitive and social coefficients.
    omega : float
        Angular frequency for circuit model (rad/s).
    Z0_val : float
        Free-space impedance (Ω).
    rng : np.random.Generator, optional
    return_history : bool
        If True, also return convergence history.

    Returns
    -------
    gbest_pos : ndarray, shape (4*N,)
        Best component parameter vector found.
    gbest_val : float
        Best R_SE achieved.
    history : ndarray, shape (max_iter+1,), optional
        Best-so-far R_SE per iteration (if return_history=True).
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = 4 * N
    lower, upper = get_component_bounds(N)

    # Velocity clamp: 20% of each parameter's range
    v_max = 0.2 * (upper - lower)

    # Step 1: Initialize particles uniformly within component bounds
    positions = rng.uniform(lower, upper, size=(pop_size, dim))
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, size=(pop_size, dim))

    # Step 6: Evaluate fitness = R_SE
    fitness = compute_rate_from_components(
        positions, N, Phi, h_d, omega, Z0_val
    )

    # Personal and global bests
    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmax(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = float(fitness[gbest_idx])

    history = [gbest_val] if return_history else None

    for t in range(max_iter):
        # Adaptive inertia: linearly decrease from w_max to w_min
        w = w_max - (w_max - w_min) * t / max_iter

        # Step 7: PSO velocity and position update with adaptive inertia
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        cognitive = c1 * r1 * (pbest_pos - positions)
        social = c2 * r2 * (gbest_pos[np.newaxis, :] - positions)
        velocities = w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities

        # Step 2: Enforce physical constraints (absorbing walls)
        positions = np.clip(positions, lower, upper)

        # Steps 3-6: Compute Z_n → v_n → v → R_SE
        fitness = compute_rate_from_components(
            positions, N, Phi, h_d, omega, Z0_val
        )

        # Update personal bests
        improved = fitness > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        # Update global best
        best_particle = int(np.argmax(pbest_val))
        if pbest_val[best_particle] > gbest_val:
            gbest_pos = pbest_pos[best_particle].copy()
            gbest_val = float(pbest_val[best_particle])

        if return_history:
            history.append(gbest_val)

    if return_history:
        return gbest_pos, gbest_val, np.array(history)
    return gbest_pos, gbest_val
