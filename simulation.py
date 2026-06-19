"""
Simulation runner for reproducing the paper's results.

Three scenarios matching the paper's figures:
    - Fig. 5: Achievable rate vs. AP-user horizontal distance (N=40)
    - Fig. 6: Achievable rate vs. number of reflecting elements (d=498m)
    - Fig. 7: Achievable rate vs. distance with discrete phase shifts (N=40)

Each scenario compares multiple schemes averaged over many channel realizations.

Performance: Uses multiprocessing to parallelize independent channel
realizations across all available CPU cores.
"""

import numpy as np
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import N_DEFAULT, NUM_REALIZATIONS, SEED
from channel_model import generate_channels
from objective import compute_channel_gain, compute_rate, compute_lower_bound_rate
from algorithms.ao import ao_optimize
from algorithms.pso import pso_optimize
from algorithms.cmaes import cmaes_optimize

# Number of parallel workers.
# Cap at half the CPU count (max 12) to avoid memory exhaustion —
# each worker process loads numpy/numba/scipy which consumes ~200-400MB.
_N_WORKERS = min(max((os.cpu_count() or 2) // 2, 2), 12)


def _run_single_realization(N, d_horizontal, schemes, rng):
    """
    Run all schemes for a single channel realization.

    Parameters
    ----------
    N : int
        Number of IRS elements.
    d_horizontal : float
        AP-user horizontal distance.
    schemes : list of str
        Which schemes to evaluate.
    rng : np.random.Generator
    Returns
    -------
    dict
        Mapping scheme name -> achievable rate.
    """
    # Generate channels using the base rng
    h_d, Phi = generate_channels(N, d_horizontal, rng)
    results = {}

    # Create independent rng for each scheme to guarantee that adding/removing
    # a scheme doesn't alter the random numbers seen by other schemes
    scheme_rngs = {s: np.random.default_rng(rng.integers(0, 2**31)) for s in schemes}

    # Cache ideal AO result if needed by multiple schemes
    # (avoids running the same ideal-model optimization twice)
    _ideal_theta, _ideal_gain = None, None
    if 'upper_bound' in schemes or 'ideal_design_practical_eval' in schemes:
        _ideal_rng = np.random.default_rng(rng.integers(0, 2**31))
        _ideal_theta, _ideal_gain = ao_optimize(
            Phi, h_d, N, method='prop1', use_practical=False, rng=_ideal_rng)

    for scheme in schemes:
        s_rng = scheme_rngs[scheme]
        
        if scheme == 'upper_bound':
            # Use cached ideal AO result
            results[scheme] = compute_rate(_ideal_gain)

        elif scheme == 'ao_practical_prop1':
            # AO with practical model, Proposition 1
            _, gain = ao_optimize(Phi, h_d, N, method='prop1',
                                  use_practical=True, rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme == 'ao_practical_1d':
            # AO with practical model, 1D search
            _, gain = ao_optimize(Phi, h_d, N, method='1d_search',
                                  use_practical=True, rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme == 'ideal_design_practical_eval':
            # Use cached ideal AO result, evaluate with practical model
            gain = compute_channel_gain(_ideal_theta, Phi, h_d, use_practical=True)
            results[scheme] = compute_rate(gain)

        elif scheme == 'lower_bound':
            # No IRS
            results[scheme] = compute_lower_bound_rate(h_d)

        elif scheme == 'pso_practical':
            _, gain = pso_optimize(Phi, h_d, N, use_practical=True,
                                   rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme == 'cmaes_practical':
            _, gain = cmaes_optimize(Phi, h_d, N, use_practical=True,
                                     rng=s_rng)
            results[scheme] = compute_rate(gain)

        # --- Discrete phase shift schemes (for Fig. 7) ---
        elif scheme.startswith('ao_practical_discrete_'):
            b = int(scheme.split('_')[-1])
            K = 2 ** b
            d_set = np.linspace(-np.pi, np.pi, K, endpoint=False)
            _, gain = ao_optimize(Phi, h_d, N, method='1d_search',
                                  use_practical=True,
                                  discrete_set=d_set, rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme.startswith('ao_ideal_discrete_'):
            b = int(scheme.split('_')[-1])
            K = 2 ** b
            d_set = np.linspace(-np.pi, np.pi, K, endpoint=False)
            _, gain = ao_optimize(Phi, h_d, N, method='1d_search',
                                  use_practical=False,
                                  discrete_set=d_set, rng=s_rng)
            # For ideal discrete, evaluate with ideal model
            results[scheme] = compute_rate(gain)

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    return results


def _realization_worker(args):
    """
    Worker function for parallel realization execution.

    Creates a fresh RNG from the given seed and runs one realization.
    This function is at module level for pickling (multiprocessing on Windows).
    """
    N, d_horizontal, schemes, seed = args
    rng = np.random.default_rng(seed)
    return _run_single_realization(N, d_horizontal, list(schemes), rng)


def _run_parallel(param_values, param_name, schemes, num_realizations,
                  master_rng, fig_name, fixed_N=None, fixed_d=None):
    """
    Generic parallelized sweep runner.

    Distributes all (parameter_value, realization) pairs across CPU cores
    using ProcessPoolExecutor for near-linear speedup.

    Parameters
    ----------
    param_values : ndarray
        Array of parameter values to sweep.
    param_name : str
        'N' or 'd' — which parameter is being swept.
    schemes : list of str
    num_realizations : int
    master_rng : np.random.Generator
    fig_name : str
    fixed_N : int, optional
        Fixed N when sweeping d.
    fixed_d : float, optional
        Fixed d when sweeping N.

    Returns
    -------
    results : dict
        Mapping scheme name -> ndarray of average rates.
    """
    # Pre-generate all tasks with deterministic seeds
    all_tasks = []
    all_indices = []
    for pi, pval in enumerate(param_values):
        N = int(pval) if param_name == 'N' else fixed_N
        d = float(pval) if param_name == 'd' else fixed_d
        for r in range(num_realizations):
            seed = int(master_rng.integers(0, 2**31))
            all_tasks.append((N, d, tuple(schemes), seed))
            all_indices.append((pi, r))

    total_tasks = len(all_tasks)
    start_time = time.time()
    n_workers = min(_N_WORKERS, total_tasks)

    print(f"\n{'='*60}")
    print(f"  {fig_name}  ({num_realizations} realizations, {n_workers} workers)")
    print(f"{'='*60}")

    rates_all = {s: np.zeros((len(param_values), num_realizations)) for s in schemes}

    if n_workers > 1 and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {}
            for i, task in enumerate(all_tasks):
                future = executor.submit(_realization_worker, task)
                future_to_idx[future] = all_indices[i]

            completed = 0
            for future in as_completed(future_to_idx):
                pi, r = future_to_idx[future]
                res = future.result()
                for s in schemes:
                    rates_all[s][pi, r] = res[s]
                completed += 1
                if completed % max(1, total_tasks // 10) == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed * (total_tasks - completed)
                           if completed > 0 else 0)
                    print(f"  Progress: {completed:>5}/{total_tasks} "
                          f"| elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    else:
        # Single-worker fallback (useful for debugging)
        for i, task in enumerate(all_tasks):
            pi, r = all_indices[i]
            res = _realization_worker(task)
            for s in schemes:
                rates_all[s][pi, r] = res[s]
            if (i + 1) % max(1, total_tasks // 10) == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1) * (total_tasks - i - 1)
                       if (i + 1) > 0 else 0)
                print(f"  Progress: {i+1:>5}/{total_tasks} "
                      f"| elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    print(f"  Completed in {time.time() - start_time:.1f}s\n")
    return {s: np.mean(rates_all[s], axis=1) for s in schemes}


def run_simulation_fig5(num_realizations=NUM_REALIZATIONS, save_path=None):
    """
    Fig. 5: Achievable rate vs. AP-user horizontal distance (N=40).

    Compares 7 schemes:
        1. Upper bound (ideal model)
        2. AO + practical model (Proposition 1)
        3. AO + practical model (1D search)
        4. Ideal design, practical evaluation
        5. Lower bound (no IRS)
        6. PSO + practical model
        7. CMA-ES + practical model

    Returns
    -------
    dict with 'd_values' and per-scheme average rates.
    """
    N = N_DEFAULT
    d_values = np.arange(480, 501, 2)  # 480, 482, ..., 500 (paper range)
    schemes = [
        'upper_bound',
        'ao_practical_prop1',
        'ao_practical_1d',
        'ideal_design_practical_eval',
        'lower_bound',
        'pso_practical',
        'cmaes_practical'
    ]

    master_rng = np.random.default_rng(SEED)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 5: Rate vs. Distance (N={N})",
        fixed_N=N
    )

    output = {'d_values': d_values}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig6(num_realizations=NUM_REALIZATIONS, save_path=None):
    """
    Fig. 6: Achievable rate vs. number of reflecting elements (d=498m).

    Returns
    -------
    dict with 'N_values' and per-scheme average rates.
    """
    d_horizontal = 498
    N_values = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    schemes = [
        'upper_bound',
        'ao_practical_prop1',
        'ao_practical_1d',
        'ideal_design_practical_eval',
        'lower_bound',
        'pso_practical',
        'cmaes_practical'
    ]

    master_rng = np.random.default_rng(SEED + 1)

    results = _run_parallel(
        N_values, 'N', schemes, num_realizations, master_rng,
        f"Fig. 6: Rate vs. N (d={d_horizontal}m)",
        fixed_d=d_horizontal
    )

    output = {'N_values': N_values}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig7(num_realizations=NUM_REALIZATIONS, save_path=None):
    """
    Fig. 7: Achievable rate vs. distance with discrete phase shifts (N=40).

    Compares b = 1, 2, 3 bits for both practical and ideal models,
    plus continuous baselines.

    Returns
    -------
    dict with 'd_values' and per-scheme average rates.
    """
    N = N_DEFAULT
    d_values = np.array([400, 420, 440, 460, 480, 498])
    bits_list = [1, 2, 3]

    schemes = ['upper_bound', 'lower_bound']
    for b in bits_list:
        schemes.append(f'ao_practical_discrete_{b}')
        schemes.append(f'ao_ideal_discrete_{b}')

    master_rng = np.random.default_rng(SEED + 2)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 7: Rate vs. Distance (Discrete, N={N})",
        fixed_N=N
    )

    output = {'d_values': d_values}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output
