"""
Particle Swarm Optimization (PSO) for IRS phase shift optimization.

Each particle represents a candidate phase shift vector θ ∈ [-π, π]^N.
The swarm collectively searches for the phase shifts that maximize
the channel gain ||v^H Φ + h_d^H||².

Key design decisions in this implementation:
    - Independent initialization: Multi-strategy seeding with
      phase-alignment heuristic, anti-phase particles, and random
      particles — no dependency on AO or any other optimizer.
    - Constriction factor (Clerc-Kennedy, 2002): Replaces linear
      inertia decay with a theoretically grounded velocity damping
      that guarantees convergence. Better than inertia weight for
      multi-modal problems.
    - Ring topology: Each particle only sees its 2 nearest neighbors'
      best, preventing premature convergence to a single basin.
      Critical for the multi-modal IRS phase shift landscape.
    - Stagnation recovery: If global best stalls, reinitialize the
      worst particles randomly to inject fresh diversity.

Reference:
    J. Kennedy and R. Eberhart, "Particle Swarm Optimization," 1995.
    M. Clerc and J. Kennedy, "The particle swarm — explosion,
    stability, and convergence," IEEE TEC, 2002.
"""

import numpy as np
from phase_shift_model import quantize_angles, wrap_angle
from objective import compute_channel_gain
from algorithms.polishing import gradient_polish
from config import (
    PSO_POP_SIZE, PSO_MAX_ITER,
    PSO_C1, PSO_C2, PSO_V_MAX,
)


def _phase_alignment_init(Phi, h_d):
    """
    Compute a starting phase vector by aligning the reflected path
    with the direct channel.

    Heuristic: θ_n = angle((Φ h_d)_n) aligns each element's reflected
    contribution with the direct path.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
    h_d : ndarray, shape (M,)
    Returns
    -------
    theta_ref : ndarray, shape (N,)
    """
    projection = Phi @ h_d  # shape (N,)
    return np.angle(projection)


def pso_optimize(Phi, h_d, N, use_practical=True,
                 discrete_set=None,
                 pop_size=PSO_POP_SIZE,
                 max_iter=PSO_MAX_ITER,
                 c1=PSO_C1, c2=PSO_C2, v_max=PSO_V_MAX,
                 rng=None):
    """
    Particle Swarm Optimization for IRS phase shift optimization.

    Uses independent multi-strategy initialization:
      - 20% phase-alignment particles (align reflected with direct path)
      - 20% anti-phase particles (opposite phase, sometimes better under
        the practical model due to β(θ) amplitude coupling)
      - 60% fully random for global exploration

    Features Clerc-Kennedy constriction factor and ring topology
    for robust convergence on multi-modal landscapes.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix.
    h_d : ndarray, shape (M,)
        Direct channel.
    N : int
        Number of IRS elements (dimension of search space).
    use_practical : bool
        Whether to use practical phase shift model.
    discrete_set : ndarray, optional
        If provided, quantize positions to nearest discrete value before evaluation.
    pop_size : int
        Number of particles in the swarm.
    max_iter : int
        Maximum number of iterations.
    c1, c2 : float
        Cognitive and social acceleration coefficients.
        Must satisfy c1 + c2 > 4 for the constriction factor.
    v_max : float
        Maximum velocity magnitude per dimension.
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
    # Constriction Factor (Clerc-Kennedy, 2002)
    # ================================================================
    # The constriction factor guarantees convergence when φ = c1 + c2 > 4.
    # Standard recommendation: c1 = c2 = 2.05 (φ = 4.1, χ ≈ 0.7298).
    phi_total = c1 + c2
    if phi_total <= 4.0:
        raise ValueError(
            f"Constriction factor requires c1 + c2 > 4, got {phi_total:.2f}. "
            f"Set PSO_C1 and PSO_C2 >= 2.05 in config.py."
        )
    chi = 2.0 / abs(2.0 - phi_total - np.sqrt(phi_total ** 2 - 4.0 * phi_total))

    # ================================================================
    # Independent Multi-Strategy Initialization (no AO dependency)
    # ================================================================
    theta_ref = _phase_alignment_init(Phi, h_d)

    # 20% phase-alignment particles (moderate perturbation σ=0.5)
    n_align = max(1, int(pop_size * 0.2))
    # 20% anti-phase particles (opposite phase, sometimes better
    # under practical model due to β(θ) amplitude coupling)
    n_anti = max(1, int(pop_size * 0.2))
    # 60% fully random for exploration
    n_random = pop_size - n_align - n_anti

    align_positions = theta_ref[np.newaxis, :] + rng.normal(0, 0.5, size=(n_align, N))
    anti_positions = wrap_angle(theta_ref + np.pi)[np.newaxis, :] + rng.normal(0, 0.5, size=(n_anti, N))
    random_positions = rng.uniform(-np.pi, np.pi, size=(n_random, N))

    positions = wrap_angle(np.vstack([align_positions, anti_positions, random_positions]))

    # Velocities: small random initial velocities
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, size=(pop_size, N))

    # Evaluate initial fitness
    eval_pos = quantize_angles(positions, discrete_set) if discrete_set is not None else positions
    fitness = compute_channel_gain(eval_pos, Phi, h_d, use_practical)

    # Personal best
    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    # Global best
    gbest_idx = np.argmax(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = fitness[gbest_idx]

    # ================================================================
    # Ring Topology: each particle's neighborhood = itself + 2 neighbors
    # ================================================================
    # Pre-compute neighbor indices (ring: left neighbor, self, right neighbor)
    neighbors = np.zeros((pop_size, 3), dtype=int)
    for i in range(pop_size):
        neighbors[i] = [(i - 1) % pop_size, i, (i + 1) % pop_size]

    # Stagnation tracking
    stagnation_counter = 0
    stagnation_threshold = 20  # iterations without improvement

    # ================================================================
    # Main Loop
    # ================================================================
    for t in range(max_iter):
        # ---- Determine social attractor (ring topology) ----
        # Each particle is attracted to the best in its neighborhood
        lbest_pos = np.zeros_like(positions)
        for i in range(pop_size):
            nbr_idx = neighbors[i]
            best_nbr = nbr_idx[np.argmax(pbest_val[nbr_idx])]
            lbest_pos[i] = pbest_pos[best_nbr]

        # ---- Velocity update ----
        r1 = rng.random((pop_size, N))
        r2 = rng.random((pop_size, N))

        cognitive = c1 * r1 * wrap_angle(pbest_pos - positions)
        social = c2 * r2 * wrap_angle(lbest_pos - positions)

        # Constriction factor velocity update (always active)
        velocities = chi * (velocities + cognitive + social)

        # Velocity clamping
        velocities = np.clip(velocities, -v_max, v_max)

        # ---- Position update ----
        positions = wrap_angle(positions + velocities)

        # ---- Evaluate fitness ----
        eval_pos = quantize_angles(positions, discrete_set) if discrete_set is not None else positions
        fitness = compute_channel_gain(eval_pos, Phi, h_d, use_practical)

        # ---- Update personal bests ----
        improved = fitness > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        # ---- Update global best ----
        best_particle = np.argmax(pbest_val)
        prev_gbest = gbest_val
        if pbest_val[best_particle] > gbest_val:
            gbest_pos = pbest_pos[best_particle].copy()
            gbest_val = pbest_val[best_particle]

        # ---- Stagnation recovery ----
        if gbest_val <= prev_gbest:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= stagnation_threshold:
            # Reinitialize bottom 30% of particles with fresh random positions
            n_reinit = max(1, int(pop_size * 0.3))
            worst_idx = np.argsort(pbest_val)[:n_reinit]

            positions[worst_idx] = rng.uniform(-np.pi, np.pi, size=(n_reinit, N))
            velocities[worst_idx] = rng.uniform(-v_max * 0.1, v_max * 0.1,
                                                 size=(n_reinit, N))
            # Re-evaluate reinitialized particles
            eval_reinit = quantize_angles(positions[worst_idx], discrete_set) \
                if discrete_set is not None else positions[worst_idx]
            fitness_reinit = compute_channel_gain(
                eval_reinit, Phi, h_d, use_practical
            )

            # Update personal bests for reinitialized particles
            for j, idx in enumerate(worst_idx):
                pbest_pos[idx] = positions[idx]
                pbest_val[idx] = fitness_reinit[j]

            stagnation_counter = 0

    # ---- Final Gradient Polish ----
    # Refine PSO's best solution with gradient ascent (25 steps).
    # Gradient updates ALL elements simultaneously, which can push
    # past the coordinate-wise optima that PSO dynamics may settle near.
    if discrete_set is None:
        polished_pos, polished_val = gradient_polish(
            gbest_pos, Phi, h_d, use_practical, n_steps=25, lr_init=0.06)
        if polished_val > gbest_val:
            gbest_pos = polished_pos
            gbest_val = polished_val

    # Return the best solution found (quantized if discrete)
    if discrete_set is not None:
        gbest_pos = quantize_angles(gbest_pos[np.newaxis, :], discrete_set)[0]

    return gbest_pos, gbest_val
