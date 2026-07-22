"""
Hybrid Phase-Component Optimizer for IRS optimization.

Two-stage approach that combines the strengths of:
    1. Analytical phase-level optimization (AO) — fast, near-optimal phases
    2. Component-level PSO — direct hardware parameter optimization

Stage 1: AO (Proposition 1) finds optimal phase shifts θ* under the
         practical model. This is fast (~10ms) and near-optimal.

Stage 2: For each IRS element n, find hardware parameters (L1, L2, C, R)
         that produce a reflection coefficient close to the target
         v*_n = β(θ*_n)·exp(jθ*_n). This is the "inverse mapping".

Stage 3: Component-level PSO is initialized with a warm-started population:
         - 50% particles near the inverse-mapped hardware parameters
         - 50% random particles (to maintain exploration diversity)

The warm start guides the component search toward the region of parameter
space that already produces good phase shifts, while random particles
ensure the optimizer can still discover better solutions.

Reference:
    Abeywickrama et al., "Intelligent Reflecting Surface: Practical Phase
    Shift Model and Beamforming Optimization."
"""

import numpy as np

from config import (
    COMP_PSO_POP_SIZE, COMP_PSO_MAX_ITER,
    COMP_GWO_POP_SIZE, COMP_GWO_MAX_ITER,
    PSO_INERTIA, PSO_C1, PSO_C2,
    OMEGA as CIRCUIT_OMEGA, Z0,
    L1_BOUNDS, L2_BOUNDS, C_BOUNDS, R_BOUNDS,
)
from algorithms.ao import ao_optimize
from algorithms.pso import pso_optimize
from algorithms.gwo import _gwo_position_update, _update_hierarchy
from circuit_model import (
    compute_impedance, compute_reflection_coefficient, get_component_bounds,
)
from phase_shift_model import reflection_vector
from objective import compute_rate_from_components


def _find_components_for_phases(theta_opt, N, rng, omega=CIRCUIT_OMEGA,
                                Z0_val=Z0, n_samples=5000):
    """
    Inverse mapping: find hardware parameters that approximate target phases.

    For each IRS element n, randomly samples (L1, L2, C, R) combinations
    and selects the one whose reflection coefficient v_n is closest to the
    target v*_n = β(θ*_n)·exp(jθ*_n).

    Parameters
    ----------
    theta_opt : ndarray, shape (N,)
        Optimal phase shifts from AO.
    N : int
        Number of IRS reflecting elements.
    rng : np.random.Generator
        Random number generator.
    omega : float
        Angular frequency (rad/s).
    Z0_val : float
        Free-space impedance (Ω).
    n_samples : int
        Number of random samples per element for inverse search.

    Returns
    -------
    x_warm : ndarray, shape (4*N,)
        Hardware parameter vector approximating the target phases.
    """
    # Compute target reflection coefficients from practical model
    v_target = reflection_vector(theta_opt, use_practical=True)

    x_warm = np.zeros(4 * N)

    # Generate random component samples (shared across elements for speed)
    L1_samples = rng.uniform(*L1_BOUNDS, size=n_samples)
    L2_samples = rng.uniform(*L2_BOUNDS, size=n_samples)
    C_samples = rng.uniform(*C_BOUNDS, size=n_samples)
    R_samples = rng.uniform(*R_BOUNDS, size=n_samples)

    # Compute impedance and reflection for all samples
    Z_samples = compute_impedance(L1_samples, L2_samples, C_samples,
                                  R_samples, omega)
    v_samples = compute_reflection_coefficient(Z_samples, Z0_val)

    for n in range(N):
        # Find sample with closest reflection coefficient to target
        errors = np.abs(v_samples - v_target[n])
        best_idx = int(np.argmin(errors))

        x_warm[4*n] = L1_samples[best_idx]
        x_warm[4*n + 1] = L2_samples[best_idx]
        x_warm[4*n + 2] = C_samples[best_idx]
        x_warm[4*n + 3] = R_samples[best_idx]

    return x_warm


def _warm_started_component_pso(
    Phi, h_d, N, x_warm,
    pop_size=COMP_PSO_POP_SIZE,
    max_iter=COMP_PSO_MAX_ITER,
    inertia=PSO_INERTIA,
    c1=PSO_C1, c2=PSO_C2,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    warm_ratio=0.5,
    rng=None, return_history=False,
):
    """Run component-level PSO with part of the swarm near x_warm."""
    if rng is None:
        rng = np.random.default_rng()

    dim = 4 * N
    lower, upper = get_component_bounds(N)

    n_warm = max(1, int(pop_size * warm_ratio))
    n_random = pop_size - n_warm

    noise_scale = 0.1 * (upper - lower)
    warm_positions = x_warm[np.newaxis, :] + rng.normal(
        0, noise_scale, size=(n_warm, dim)
    )
    warm_positions = np.clip(warm_positions, lower, upper)
    random_positions = rng.uniform(lower, upper, size=(n_random, dim))
    positions = np.vstack([warm_positions, random_positions])

    v_max = 0.2 * (upper - lower)
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, size=(pop_size, dim))

    fitness = compute_rate_from_components(
        positions, N, Phi, h_d, omega, Z0_val
    )
    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmax(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = float(fitness[gbest_idx])
    history = [gbest_val] if return_history else None

    for _ in range(max_iter):
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        cognitive = c1 * r1 * (pbest_pos - positions)
        social = c2 * r2 * (gbest_pos[np.newaxis, :] - positions)
        velocities = inertia * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = np.clip(positions + velocities, lower, upper)
        fitness = compute_rate_from_components(
            positions, N, Phi, h_d, omega, Z0_val
        )

        improved = fitness > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        best_particle = int(np.argmax(pbest_val))
        if pbest_val[best_particle] > gbest_val:
            gbest_pos = pbest_pos[best_particle].copy()
            gbest_val = float(pbest_val[best_particle])

        if return_history:
            history.append(gbest_val)

    if return_history:
        return gbest_pos, gbest_val, np.array(history)
    return gbest_pos, gbest_val


def _warm_started_component_gwo(
    Phi, h_d, N, x_warm,
    pop_size=COMP_GWO_POP_SIZE,
    max_iter=COMP_GWO_MAX_ITER,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    warm_ratio=0.5,
    rng=None, return_history=False,
):
    """Run component-level GWO with part of the pack near x_warm."""
    if rng is None:
        rng = np.random.default_rng()

    dim = 4 * N
    lower, upper = get_component_bounds(N)

    n_warm = max(3, int(pop_size * warm_ratio))
    n_warm = min(n_warm, pop_size)
    n_random = pop_size - n_warm

    noise_scale = 0.1 * (upper - lower)
    warm_positions = x_warm[np.newaxis, :] + rng.normal(
        0, noise_scale, size=(n_warm, dim)
    )
    warm_positions = np.clip(warm_positions, lower, upper)
    random_positions = rng.uniform(lower, upper, size=(n_random, dim))
    positions = np.vstack([warm_positions, random_positions])

    fitness = compute_rate_from_components(
        positions, N, Phi, h_d, omega, Z0_val
    )
    sorted_idx = np.argsort(-fitness)
    alpha_pos = positions[sorted_idx[0]].copy()
    alpha_score = float(fitness[sorted_idx[0]])
    beta_pos = positions[sorted_idx[1]].copy()
    beta_score = float(fitness[sorted_idx[1]])
    delta_pos = positions[sorted_idx[2]].copy()
    delta_score = float(fitness[sorted_idx[2]])
    history = [alpha_score] if return_history else None

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter
        positions = _gwo_position_update(
            positions, alpha_pos, beta_pos, delta_pos, a, rng
        )
        positions = np.clip(positions, lower, upper)
        fitness = compute_rate_from_components(
            positions, N, Phi, h_d, omega, Z0_val
        )
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


def hybrid_phase_component_optimize(
    Phi, h_d, N,
    pop_size=COMP_PSO_POP_SIZE,
    max_iter=COMP_PSO_MAX_ITER,
    inertia=PSO_INERTIA,
    c1=PSO_C1, c2=PSO_C2,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    warm_ratio=0.5,
    rng=None, return_history=False,
):
    """
    Hybrid two-stage optimizer: AO phase-level → warm-started component PSO.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct AP-to-User channel.
    N : int
        Number of IRS reflecting elements.
    pop_size : int
        Number of particles for component-level PSO.
    max_iter : int
        Maximum iterations for component-level PSO.
    inertia : float
        Inertia weight (Clerc's constriction: 0.729).
    c1, c2 : float
        Cognitive and social coefficients.
    omega : float
        Angular frequency for circuit model (rad/s).
    Z0_val : float
        Free-space impedance (Ω).
    warm_ratio : float
        Fraction of particles initialized from phase-level solution (0-1).
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

    # ================================================================
    # Stage 1: AO phase-level optimization → optimal phase shifts
    # ================================================================
    theta_opt, _ = ao_optimize(
        Phi, h_d, N, method='prop1', use_practical=True,
        rng=np.random.default_rng(rng.integers(0, 2**31))
    )

    # ================================================================
    # Stage 2: Inverse mapping → approximate hardware parameters
    # ================================================================
    x_warm = _find_components_for_phases(
        theta_opt, N,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        omega=omega, Z0_val=Z0_val
    )

    # ================================================================
    # Stage 3: Warm-started component-level PSO
    # ================================================================
    n_warm = max(1, int(pop_size * warm_ratio))
    n_random = pop_size - n_warm

    # Warm particles: x_warm + Gaussian noise (10% of parameter range)
    noise_scale = 0.1 * (upper - lower)
    warm_positions = x_warm[np.newaxis, :] + rng.normal(
        0, noise_scale, size=(n_warm, dim)
    )
    warm_positions = np.clip(warm_positions, lower, upper)

    # Random particles for diversity
    random_positions = rng.uniform(lower, upper, size=(n_random, dim))

    positions = np.vstack([warm_positions, random_positions])

    # Velocity initialization
    v_max = 0.2 * (upper - lower)
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, size=(pop_size, dim))

    # Evaluate initial fitness
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
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        cognitive = c1 * r1 * (pbest_pos - positions)
        social = c2 * r2 * (gbest_pos[np.newaxis, :] - positions)
        velocities = inertia * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

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


def hybrid_pso_pso_component_optimize(
    Phi, h_d, N,
    pop_size=COMP_PSO_POP_SIZE,
    max_iter=COMP_PSO_MAX_ITER,
    inertia=PSO_INERTIA,
    c1=PSO_C1, c2=PSO_C2,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    warm_ratio=0.5,
    rng=None, return_history=False,
):
    """
    Hybrid optimizer: phase-level PSO -> warm-started component-level PSO.

    The first PSO searches the phase vector theta. The resulting practical
    reflection vector is inverse-mapped to circuit components, then used to
    initialize half of the component-level PSO swarm.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_opt, _ = pso_optimize(
        Phi, h_d, N, use_practical=True,
        rng=np.random.default_rng(rng.integers(0, 2**31))
    )
    x_warm = _find_components_for_phases(
        theta_opt, N,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        omega=omega, Z0_val=Z0_val
    )

    return _warm_started_component_pso(
        Phi, h_d, N, x_warm,
        pop_size=pop_size,
        max_iter=max_iter,
        inertia=inertia,
        c1=c1, c2=c2,
        omega=omega, Z0_val=Z0_val,
        warm_ratio=warm_ratio,
        rng=rng,
        return_history=return_history,
    )


def hybrid_pso_gwo_component_optimize(
    Phi, h_d, N,
    pop_size=COMP_GWO_POP_SIZE,
    max_iter=COMP_GWO_MAX_ITER,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    warm_ratio=0.5,
    rng=None, return_history=False,
):
    """
    Hybrid optimizer: phase-level PSO -> warm-started component-level GWO.

    The first stage is identical to PSO-PSO. The second stage uses GWO over
    the circuit-component vector, with part of the pack initialized near the
    inverse-mapped PSO phase solution.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_opt, _ = pso_optimize(
        Phi, h_d, N, use_practical=True,
        rng=np.random.default_rng(rng.integers(0, 2**31))
    )
    x_warm = _find_components_for_phases(
        theta_opt, N,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        omega=omega, Z0_val=Z0_val
    )

    return _warm_started_component_gwo(
        Phi, h_d, N, x_warm,
        pop_size=pop_size,
        max_iter=max_iter,
        omega=omega, Z0_val=Z0_val,
        warm_ratio=warm_ratio,
        rng=rng,
        return_history=return_history,
    )
