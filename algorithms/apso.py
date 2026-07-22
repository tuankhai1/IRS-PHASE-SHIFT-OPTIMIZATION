"""
Adaptive PSO (APSO) for IRS phase-shift optimization.

Each particle represents a candidate phase-shift vector theta in
[-pi, pi]^N. The swarm searches for the phase shifts that maximize

    F(v) = ||v^H Phi + h_d^H||^2.

Identical to standard PSO except for the inertia weight schedule:
    w(t) = w_max - (w_max - w_min) × t / max_iter

This linearly decreases inertia from 0.9 (exploration) to 0.4
(exploitation) over the course of optimization, providing better
exploration-exploitation balance than fixed inertia.

Reference:
    Y. Shi, R. Eberhart, "A modified particle swarm optimizer,"
    Proc. IEEE International Conference on Evolutionary Computation,
    pp. 69-73, 1998.
"""

import numpy as np

from config import (
    APSO_POP_SIZE,
    APSO_MAX_ITER,
    APSO_W_MAX,
    APSO_W_MIN,
    APSO_C1,
    APSO_C2,
    PSO_V_MAX,
)
from objective import compute_channel_gain, compute_rate
from phase_shift_model import quantize_angles, reflection_vector, wrap_angle


def apso_optimize(
    Phi,
    h_d,
    N,
    use_practical=True,
    discrete_set=None,
    pop_size=APSO_POP_SIZE,
    max_iter=APSO_MAX_ITER,
    w_max=APSO_W_MAX,
    w_min=APSO_W_MIN,
    c1=APSO_C1,
    c2=APSO_C2,
    v_max=PSO_V_MAX,
    rng=None,
):
    """
    Adaptive PSO with linearly decreasing inertia for IRS phase-shift
    optimization.

    w(t) = w_max - (w_max - w_min) × t / max_iter
         = 0.9  → 0.4 over iterations

    Early iterations: high inertia → large velocity → exploration.
    Late iterations:  low inertia → small velocity → exploitation.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
        Combined channel matrix Φ = diag(h_r^H) G.
    h_d : ndarray, shape (M,)
        AP-to-User direct channel.
    N : int
        Number of IRS reflecting elements.
    use_practical : bool
        Whether to use the practical phase shift model.
    discrete_set : ndarray, optional
        Discrete phase values for quantized search.
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
    v_max : float
        Velocity clamp (radians).
    rng : np.random.Generator, optional

    Returns
    -------
    gbest_pos : ndarray, shape (N,)
        Best phase shift vector found.
    gbest_val : float
        Best channel gain achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize particles uniformly in [-π, π]
    positions = rng.uniform(-np.pi, np.pi, size=(pop_size, N))
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1, size=(pop_size, N))

    eval_pos = (
        quantize_angles(positions, discrete_set)
        if discrete_set is not None else positions
    )
    fitness = compute_channel_gain(eval_pos, Phi, h_d, use_practical)

    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmax(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = float(fitness[gbest_idx])

    for t in range(max_iter):
        # Adaptive inertia: linearly decrease from w_max to w_min
        w = w_max - (w_max - w_min) * t / max_iter

        r1 = rng.random((pop_size, N))
        r2 = rng.random((pop_size, N))

        # Use wrap_angle for angular differences (respects periodicity)
        cognitive = c1 * r1 * wrap_angle(pbest_pos - positions)
        social = c2 * r2 * wrap_angle(gbest_pos[np.newaxis, :] - positions)
        velocities = w * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = wrap_angle(positions + velocities)

        eval_pos = (
            quantize_angles(positions, discrete_set)
            if discrete_set is not None else positions
        )
        fitness = compute_channel_gain(eval_pos, Phi, h_d, use_practical)

        improved = fitness > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        best_particle = int(np.argmax(pbest_val))
        if pbest_val[best_particle] > gbest_val:
            gbest_pos = pbest_pos[best_particle].copy()
            gbest_val = float(pbest_val[best_particle])

    if discrete_set is not None:
        gbest_pos = quantize_angles(gbest_pos[np.newaxis, :], discrete_set)[0]

    return gbest_pos, gbest_val
