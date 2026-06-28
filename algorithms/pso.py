"""
Particle Swarm Optimization (PSO) for IRS phase-shift optimization.

Each particle represents a candidate phase-shift vector theta in
[-pi, pi]^N. The swarm searches for the phase shifts that maximize

    F(v) = ||v^H Phi + h_d^H||^2.

Because the achievable rate is strictly increasing in F(v), maximizing
this channel gain also maximizes the rate objective used in the paper.
"""

import numpy as np

from config import (
    PSO_C1,
    PSO_C2,
    PSO_INERTIA,
    PSO_MAX_ITER,
    PSO_POP_SIZE,
    PSO_V_MAX,
)
from objective import compute_channel_gain, compute_rate
from phase_shift_model import quantize_angles, reflection_vector, wrap_angle


def _validate_objective_inputs(Phi, h_d, N):
    """Validate dimensions for F(v) = ||v^H Phi + h_d^H||^2."""
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2D with shape (N, M); got {Phi.shape}.")
    if h_d.ndim != 1:
        raise ValueError(f"h_d must be 1D with shape (M,); got {h_d.shape}.")
    if Phi.shape != (N, h_d.shape[0]):
        raise ValueError(
            "Objective dimensions are inconsistent: expected "
            f"Phi.shape == ({N}, {h_d.shape[0]}), got {Phi.shape}."
        )


def _objective_diagnostics(theta, Phi, h_d, use_practical):
    """
    Break F(v) into terms for debugging and explanation.

    This helper mirrors compute_channel_gain:
        reflected = v^H Phi
        direct    = h_d^H
        effective = reflected + direct
        gain      = ||effective||^2
    """
    v = reflection_vector(theta, use_practical)
    reflected = v.conj() @ Phi
    direct = h_d.conj()
    effective = reflected + direct
    gain = float(np.vdot(effective, effective).real)

    return {
        "reflection_amplitude_min": float(np.min(np.abs(v))),
        "reflection_amplitude_max": float(np.max(np.abs(v))),
        "reflected_norm": float(np.linalg.norm(reflected)),
        "direct_norm": float(np.linalg.norm(direct)),
        "effective_norm": float(np.linalg.norm(effective)),
        "gain": gain,
        "rate": float(compute_rate(gain)),
    }


def _print_objective_diagnostics(label, theta, Phi, h_d, use_practical):
    """Print a compact explanation of the objective for one candidate."""
    info = _objective_diagnostics(theta, Phi, h_d, use_practical)
    print(
        f"[PSO] {label}: "
        f"|v_n|=[{info['reflection_amplitude_min']:.4f}, "
        f"{info['reflection_amplitude_max']:.4f}], "
        f"||v^H Phi||={info['reflected_norm']:.6e}, "
        f"||h_d^H||={info['direct_norm']:.6e}, "
        f"||v^H Phi + h_d^H||={info['effective_norm']:.6e}, "
        f"F(v)={info['gain']:.6e}, "
        f"rate={info['rate']:.6f} bits/s/Hz"
    )


def pso_optimize(
    Phi,
    h_d,
    N,
    use_practical=True,
    discrete_set=None,
    pop_size=PSO_POP_SIZE,
    max_iter=PSO_MAX_ITER,
    inertia=PSO_INERTIA,
    c1=PSO_C1,
    c2=PSO_C2,
    v_max=PSO_V_MAX,
    rng=None,
    verbose=False,
    log_interval=10,
):
    """
    Standard global-best PSO for IRS phase-shift optimization.

    This is the single PSO implementation used by the simulation pipeline.
    It uses uniform-random initialization, global-best topology, inertia,
    cognitive/social attraction, velocity clipping, and angle wrapping.
    """
    _validate_objective_inputs(Phi, h_d, N)
    if pop_size < 2:
        raise ValueError("pop_size must be at least 2.")
    if log_interval < 1:
        raise ValueError("log_interval must be at least 1.")
    if rng is None:
        rng = np.random.default_rng()

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

    if verbose:
        model_name = "practical amplitude-phase" if use_practical else "ideal unit-amplitude"
        print(
            f"[PSO] Maximizing F(v) = ||v^H Phi + h_d^H||^2 "
            f"with the {model_name} model."
        )
        print(
            f"[PSO] swarm={pop_size}, dimensions={N}, iterations={max_iter}, "
            f"inertia={inertia:.6f}, c1={c1:.6f}, c2={c2:.6f}"
        )
        _print_objective_diagnostics(
            "best initialization", eval_pos[gbest_idx], Phi, h_d, use_practical
        )

    for t in range(max_iter):
        r1 = rng.random((pop_size, N))
        r2 = rng.random((pop_size, N))

        cognitive = c1 * r1 * wrap_angle(pbest_pos - positions)
        social = c2 * r2 * wrap_angle(gbest_pos[np.newaxis, :] - positions)
        velocities = inertia * velocities + cognitive + social
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

        if verbose and (
            t == 0 or (t + 1) % log_interval == 0 or t + 1 == max_iter
        ):
            print(
                f"[PSO] iteration {t + 1:>4}/{max_iter}: "
                f"best F(v)={gbest_val:.6e}, "
                f"rate={float(compute_rate(gbest_val)):.6f} bits/s/Hz"
            )

    if discrete_set is not None:
        gbest_pos = quantize_angles(gbest_pos[np.newaxis, :], discrete_set)[0]

    if verbose:
        _print_objective_diagnostics(
            "final solution", gbest_pos, Phi, h_d, use_practical
        )

    return gbest_pos, gbest_val
