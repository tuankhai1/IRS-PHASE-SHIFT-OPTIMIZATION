"""
Fixed-component ablation study for IRS optimization.

Investigates the impact of individual circuit components by fixing subsets
of {L1, L2, C, R} at their midpoint values and optimizing over the rest.

Ablation scenarios:
    1. Full optimization — optimize all 4 components (baseline)
    2. Fix R — fix resistance, optimize L1, L2, C
    3. Fix C, R — fix capacitance and resistance, optimize L1, L2
    4. Fix L2, C, R — fix all but coupling inductance, optimize L1 only

This reveals which components contribute most to the achievable rate,
and whether a simpler (fewer-variable) optimization can approximate
the full result.

Uses PSO component-level for all scenarios to ensure fair comparison.
"""

import numpy as np
import time
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    N_DEFAULT, NUM_REALIZATIONS, SEED,
    COMP_PSO_POP_SIZE, COMP_PSO_MAX_ITER,
    PSO_INERTIA, PSO_C1, PSO_C2,
    OMEGA as CIRCUIT_OMEGA, Z0,
    L1_BOUNDS, L2_BOUNDS, C_BOUNDS, R_BOUNDS,
)
from channel_model import generate_channels
from objective import compute_rate, compute_lower_bound_rate
from circuit_model import get_component_bounds
from objective import compute_rate_from_components

# Number of parallel workers
_N_WORKERS = min(max((os.cpu_count() or 2) // 2, 2), 12)

# Midpoint values for fixed components
L1_MID = (L1_BOUNDS[0] + L1_BOUNDS[1]) / 2
L2_MID = (L2_BOUNDS[0] + L2_BOUNDS[1]) / 2
C_MID  = (C_BOUNDS[0]  + C_BOUNDS[1])  / 2
R_MID  = (R_BOUNDS[0]  + R_BOUNDS[1])  / 2


# ============================================================
# Fixed-mask definitions
# ============================================================
# Each scenario specifies which of the 4 per-element parameters are fixed.
# Format: (name, mask, fixed_values)
#   mask[i] = True  → parameter i is FIXED
#   mask[i] = False → parameter i is OPTIMIZED
# Parameter order: [L1, L2, C, R]

ABLATION_SCENARIOS = {
    'full': {
        'label': 'Full (L₁, L₂, C, R)',
        'mask': [False, False, False, False],
        'fixed_vals': [None, None, None, None],
    },
    'fix_R': {
        'label': 'Fix R',
        'mask': [False, False, False, True],
        'fixed_vals': [None, None, None, R_MID],
    },
    'fix_CR': {
        'label': 'Fix C, R',
        'mask': [False, False, True, True],
        'fixed_vals': [None, None, C_MID, R_MID],
    },
    'fix_L2CR': {
        'label': 'Fix L₂, C, R',
        'mask': [False, True, True, True],
        'fixed_vals': [None, L2_MID, C_MID, R_MID],
    },
}


def _stable_seed(base_seed, *labels):
    """Derive a deterministic seed that does not depend on scheme order."""
    text = '|'.join([str(int(base_seed)), *(str(label) for label in labels)])
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little')


def _rng_for(base_seed, *labels):
    return np.random.default_rng(_stable_seed(base_seed, *labels))


def _pso_fixed_component_optimize(
    Phi, h_d, N, fix_mask, fixed_vals,
    pop_size=COMP_PSO_POP_SIZE,
    max_iter=COMP_PSO_MAX_ITER,
    inertia=PSO_INERTIA,
    c1=PSO_C1, c2=PSO_C2,
    omega=CIRCUIT_OMEGA, Z0_val=Z0,
    rng=None,
):
    """
    PSO component-level with some parameters fixed.

    Parameters
    ----------
    Phi : ndarray, shape (N, M)
    h_d : ndarray, shape (M,)
    N : int
    fix_mask : list of 4 bools
        Which of [L1, L2, C, R] are fixed (per element).
    fixed_vals : list of 4 floats or Nones
        Fixed values for the corresponding parameters.
    ...

    Returns
    -------
    best_rate : float
        Best R_SE achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    full_dim = 4 * N
    lower_full, upper_full = get_component_bounds(N)

    # Determine which dimensions are free vs fixed
    free_dims = []
    fixed_dims = {}
    for n in range(N):
        for p in range(4):
            idx = 4 * n + p
            if fix_mask[p]:
                fixed_dims[idx] = fixed_vals[p]
            else:
                free_dims.append(idx)

    free_dims = np.array(free_dims)
    n_free = len(free_dims)

    if n_free == 0:
        # All fixed — just evaluate
        x = np.zeros(full_dim)
        for idx, val in fixed_dims.items():
            x[idx] = val
        return float(compute_rate_from_components(
            x, N, Phi, h_d, omega, Z0_val
        ))

    lower_free = lower_full[free_dims]
    upper_free = upper_full[free_dims]
    v_max = 0.2 * (upper_free - lower_free)

    # Initialize particles (free dimensions only)
    positions = rng.uniform(lower_free, upper_free, size=(pop_size, n_free))
    velocities = rng.uniform(-v_max * 0.1, v_max * 0.1,
                             size=(pop_size, n_free))

    def _build_full(pos_free):
        """Expand free-dim positions to full 4*N vectors."""
        if pos_free.ndim == 1:
            x = np.zeros(full_dim)
            for idx, val in fixed_dims.items():
                x[idx] = val
            x[free_dims] = pos_free
            return x
        else:
            pop = pos_free.shape[0]
            x = np.zeros((pop, full_dim))
            for idx, val in fixed_dims.items():
                x[:, idx] = val
            x[:, free_dims] = pos_free
            return x

    # Evaluate initial fitness
    x_full = _build_full(positions)
    fitness = compute_rate_from_components(x_full, N, Phi, h_d, omega, Z0_val)

    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmax(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = float(fitness[gbest_idx])

    for t in range(max_iter):
        r1 = rng.random((pop_size, n_free))
        r2 = rng.random((pop_size, n_free))

        cognitive = c1 * r1 * (pbest_pos - positions)
        social = c2 * r2 * (gbest_pos[np.newaxis, :] - positions)
        velocities = inertia * velocities + cognitive + social
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lower_free, upper_free)

        x_full = _build_full(positions)
        fitness = compute_rate_from_components(
            x_full, N, Phi, h_d, omega, Z0_val
        )

        improved = fitness > pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        best_particle = int(np.argmax(pbest_val))
        if pbest_val[best_particle] > gbest_val:
            gbest_pos = pbest_pos[best_particle].copy()
            gbest_val = float(pbest_val[best_particle])

    return gbest_val


def _fixed_component_worker(args):
    """Worker: run all ablation scenarios for one channel realization."""
    N, d_horizontal, scenarios, seed = args

    h_d, Phi = generate_channels(
        N, d_horizontal, _rng_for(seed, 'channel')
    )

    results = {}

    # Lower bound (no IRS)
    results['lower_bound'] = float(compute_lower_bound_rate(h_d))

    for name, scenario in scenarios.items():
        s_rng = _rng_for(seed, name)
        rate = _pso_fixed_component_optimize(
            Phi, h_d, N,
            fix_mask=scenario['mask'],
            fixed_vals=scenario['fixed_vals'],
            rng=s_rng,
        )
        results[name] = rate

    return results


def run_simulation_fig12(num_realizations=NUM_REALIZATIONS, save_path=None,
                         seed=SEED):
    """
    Fig. 12: Fixed-component ablation study.

    Compares achievable rate when different subsets of circuit components
    are fixed at midpoint values, across AP-user distance.

    Scenarios:
        - Full: optimize all (L1, L2, C, R)
        - Fix R: optimize (L1, L2, C), R fixed at midpoint
        - Fix C, R: optimize (L1, L2), C and R fixed
        - Fix L2, C, R: optimize (L1 only), rest fixed

    Parameters
    ----------
    num_realizations : int
    save_path : str, optional
    seed : int

    Returns
    -------
    dict with 'd_values' and per-scenario average rates.
    """
    N = N_DEFAULT
    d_values = np.arange(480, 501, 2)  # Same range as Fig 5/8

    master_rng = np.random.default_rng(seed + 7)

    # Build task list
    all_tasks = []
    all_indices = []
    for pi, d in enumerate(d_values):
        for r in range(num_realizations):
            task_seed = int(master_rng.integers(0, 2**31))
            all_tasks.append((N, float(d), ABLATION_SCENARIOS, task_seed))
            all_indices.append((pi, r))

    total_tasks = len(all_tasks)
    n_workers = min(_N_WORKERS, total_tasks)
    start_time = time.time()

    scheme_names = list(ABLATION_SCENARIOS.keys()) + ['lower_bound']

    print(f"\n{'='*60}")
    print(f"  Fig. 12: Fixed-Component Ablation (N={N})")
    print(f"  ({num_realizations} realizations, {n_workers} workers)")
    print(f"{'='*60}")

    rates_all = {s: np.zeros((len(d_values), num_realizations))
                 for s in scheme_names}

    if n_workers > 1 and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {}
            for i, task in enumerate(all_tasks):
                future = executor.submit(_fixed_component_worker, task)
                future_to_idx[future] = all_indices[i]

            completed = 0
            for future in as_completed(future_to_idx):
                pi, r = future_to_idx[future]
                res = future.result()
                for s in scheme_names:
                    rates_all[s][pi, r] = res[s]
                completed += 1
                if completed % max(1, total_tasks // 10) == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed * (total_tasks - completed)
                           if completed > 0 else 0)
                    print(f"  Progress: {completed:>5}/{total_tasks} "
                          f"| elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    else:
        for i, task in enumerate(all_tasks):
            pi, r = all_indices[i]
            res = _fixed_component_worker(task)
            for s in scheme_names:
                rates_all[s][pi, r] = res[s]
            if (i + 1) % max(1, total_tasks // 10) == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1) * (total_tasks - i - 1)
                       if (i + 1) > 0 else 0)
                print(f"  Progress: {i+1:>5}/{total_tasks} "
                      f"| elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    wall_seconds = time.time() - start_time
    print(f"  Completed in {wall_seconds:.1f}s\n")

    output = {
        'd_values': d_values,
        'seed': np.array(seed),
    }
    for s in scheme_names:
        output[s] = np.mean(rates_all[s], axis=1)

    # Include fixed values metadata
    output['fixed_L1_mid'] = np.array(L1_MID)
    output['fixed_L2_mid'] = np.array(L2_MID)
    output['fixed_C_mid'] = np.array(C_MID)
    output['fixed_R_mid'] = np.array(R_MID)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output
