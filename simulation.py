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
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import N_DEFAULT, NUM_REALIZATIONS, SEED
from channel_model import generate_channels
from objective import compute_channel_gain, compute_rate, compute_lower_bound_rate
from algorithms.ao import ao_optimize
from algorithms.pso import pso_optimize
from algorithms.apso import apso_optimize
from algorithms.gwo import gwo_optimize, gwo_component_optimize
from algorithms.pso_component import pso_component_optimize
from algorithms.apso_component import apso_component_optimize
from algorithms.hybrid import (
    hybrid_phase_component_optimize,
    hybrid_pso_pso_component_optimize,
    hybrid_pso_gwo_component_optimize,
)

# Number of parallel workers.
# Cap at half the CPU count (max 12) to avoid memory exhaustion —
# each worker process loads numpy/numba/scipy which consumes ~200-400MB.
_N_WORKERS = min(max((os.cpu_count() or 2) // 2, 2), 12)


PAPER_CONTINUOUS_SCHEMES = [
    'upper_bound',
    'ao_practical_prop1',
    'ao_practical_1d',
    'ideal_design_practical_eval',
    'lower_bound',
]

METAHEURISTIC_COMPARISON_SCHEMES = [
    'pso',
    'apso',
    'gwo',
]

CONTINUOUS_COMPARISON_SCHEMES = (
    PAPER_CONTINUOUS_SCHEMES + METAHEURISTIC_COMPARISON_SCHEMES
)

COMPONENT_COMPARISON_SCHEMES = [
    'pso_component',
    'apso_component',
    'gwo_component',
]


def _stable_seed(base_seed, *labels):
    """Derive a deterministic seed that does not depend on scheme order."""
    text = '|'.join([str(int(base_seed)), *(str(label) for label in labels)])
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little')


def _rng_for(base_seed, *labels):
    return np.random.default_rng(_stable_seed(base_seed, *labels))


def _run_single_realization(N, d_horizontal, schemes, seed):
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
    seed : int
        Task seed for this channel realization.
    Returns
    -------
    tuple of dict
        Per-scheme achievable rates and per-scheme runtimes.
    """
    # Keep channel randomness separate from optimizer randomness.
    h_d, Phi = generate_channels(N, d_horizontal, _rng_for(seed, 'channel'))
    results = {}
    runtimes = {}

    def run_ideal_design():
        """Run the ideal-model AO design used by two paper schemes."""
        return ao_optimize(
            Phi, h_d, N, method='prop1', use_practical=False,
            rng=_rng_for(seed, 'ideal_design'))

    for scheme in schemes:
        s_rng = _rng_for(seed, scheme)
        scheme_start = time.perf_counter()

        if scheme == 'upper_bound':
            _, gain = run_ideal_design()
            results[scheme] = compute_rate(gain)

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
            theta_ideal, _ = run_ideal_design()
            gain = compute_channel_gain(theta_ideal, Phi, h_d, use_practical=True)
            results[scheme] = compute_rate(gain)

        elif scheme == 'lower_bound':
            # No IRS
            results[scheme] = compute_lower_bound_rate(h_d)

        elif scheme == 'pso':
            _, gain = pso_optimize(Phi, h_d, N, use_practical=True,
                                   rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme == 'apso':
            _, gain = apso_optimize(Phi, h_d, N, use_practical=True,
                                    rng=s_rng)
            results[scheme] = compute_rate(gain)

        elif scheme == 'gwo':
            _, gain = gwo_optimize(Phi, h_d, N, use_practical=True,
                                   rng=s_rng)
            results[scheme] = compute_rate(gain)

        # --- Component-level schemes ---
        elif scheme == 'pso_component':
            _, rate = pso_component_optimize(Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

        elif scheme == 'apso_component':
            _, rate = apso_component_optimize(Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

        elif scheme == 'gwo_component':
            _, rate = gwo_component_optimize(Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

        elif scheme == 'hybrid_component':
            _, rate = hybrid_phase_component_optimize(
                Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

        elif scheme == 'hybrid_pso_pso_component':
            _, rate = hybrid_pso_pso_component_optimize(
                Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

        elif scheme == 'hybrid_pso_gwo_component':
            _, rate = hybrid_pso_gwo_component_optimize(
                Phi, h_d, N, rng=s_rng)
            results[scheme] = rate

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

        runtimes[scheme] = time.perf_counter() - scheme_start

    return results, runtimes


def _realization_worker(args):
    """
    Worker function for parallel realization execution.

    Runs one realization from a deterministic task seed.
    This function is at module level for pickling (multiprocessing on Windows).
    """
    N, d_horizontal, schemes, seed = args
    return _run_single_realization(N, d_horizontal, list(schemes), seed)


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
    times_all = {s: np.zeros((len(param_values), num_realizations)) for s in schemes}

    if n_workers > 1 and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {}
            for i, task in enumerate(all_tasks):
                future = executor.submit(_realization_worker, task)
                future_to_idx[future] = all_indices[i]

            completed = 0
            for future in as_completed(future_to_idx):
                pi, r = future_to_idx[future]
                res, runtimes = future.result()
                for s in schemes:
                    rates_all[s][pi, r] = res[s]
                    times_all[s][pi, r] = runtimes[s]
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
            res, runtimes = _realization_worker(task)
            for s in schemes:
                rates_all[s][pi, r] = res[s]
                times_all[s][pi, r] = runtimes[s]
            if (i + 1) % max(1, total_tasks // 10) == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1) * (total_tasks - i - 1)
                       if (i + 1) > 0 else 0)
                print(f"  Progress: {i+1:>5}/{total_tasks} "
                      f"| elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    wall_seconds = time.time() - start_time
    print(f"  Completed in {wall_seconds:.1f}s\n")

    results = {s: np.mean(rates_all[s], axis=1) for s in schemes}
    results.update({
        'runtime_scheme_names': np.array(schemes),
        'runtime_mean_seconds': np.vstack([
            np.mean(times_all[s], axis=1) for s in schemes
        ]),
        'runtime_total_seconds': np.vstack([
            np.sum(times_all[s], axis=1) for s in schemes
        ]),
        'runtime_overall_mean_seconds': np.array([
            np.mean(times_all[s]) for s in schemes
        ]),
        'runtime_overall_total_seconds': np.array([
            np.sum(times_all[s]) for s in schemes
        ]),
        'runtime_num_samples': np.array(total_tasks),
        'runtime_wall_seconds': np.array(wall_seconds),
    })
    return results


# ============================================================
# Component-Level Simulation Functions (Figs. 8-10)
# ============================================================

def _convergence_worker(args):
    """
    Worker function for parallel convergence realization.

    Runs all component-level algorithms with convergence history
    for a single channel realization.
    """
    N, d_horizontal, seed = args
    h_d, Phi = generate_channels(N, d_horizontal, _rng_for(seed, 'channel'))

    histories = {}

    _, _, hist = pso_component_optimize(
        Phi, h_d, N, rng=_rng_for(seed, 'pso_comp'), return_history=True)
    histories['pso_component'] = hist

    _, _, hist = apso_component_optimize(
        Phi, h_d, N, rng=_rng_for(seed, 'apso_comp'), return_history=True)
    histories['apso_component'] = hist

    _, _, hist = gwo_component_optimize(
        Phi, h_d, N, rng=_rng_for(seed, 'gwo_comp'), return_history=True)
    histories['gwo_component'] = hist

    return histories


def run_simulation_fig5(num_realizations=NUM_REALIZATIONS, save_path=None,
                        seed=SEED):
    """
    Fig. 5: Achievable rate vs. AP-user horizontal distance (N=40).

    Compares the paper's continuous-phase schemes plus PSO/GWO baselines:
        1. Upper bound (ideal model)
        2. AO + practical model (Proposition 1)
        3. AO + practical model (1D search)
        4. Ideal design, practical evaluation
        5. Lower bound (no IRS)
        6. PSO + practical model
        7. GWO + practical model

    Returns
    -------
    dict with 'd_values' and per-scheme average rates.
    """
    N = N_DEFAULT
    d_values = np.arange(480, 501, 2)  # 480, 482, ..., 500 (paper range)
    schemes = CONTINUOUS_COMPARISON_SCHEMES.copy()

    master_rng = np.random.default_rng(seed)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 5: Rate vs. Distance (N={N})",
        fixed_N=N
    )

    output = {
        'd_values': d_values,
        'seed': np.array(seed),
    }
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig6(num_realizations=NUM_REALIZATIONS, save_path=None,
                        seed=SEED):
    """
    Fig. 6: Achievable rate vs. number of reflecting elements (d=498m).

    Returns
    -------
    dict with 'N_values' and per-scheme average rates.
    """
    d_horizontal = 498
    N_values = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    schemes = CONTINUOUS_COMPARISON_SCHEMES.copy()

    master_rng = np.random.default_rng(seed + 1)

    results = _run_parallel(
        N_values, 'N', schemes, num_realizations, master_rng,
        f"Fig. 6: Rate vs. N (d={d_horizontal}m)",
        fixed_d=d_horizontal
    )

    output = {
        'N_values': N_values,
        'seed': np.array(seed),
    }
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig7(num_realizations=NUM_REALIZATIONS, save_path=None,
                        seed=SEED):
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

    master_rng = np.random.default_rng(seed + 2)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 7: Rate vs. Distance (Discrete, N={N})",
        fixed_N=N
    )

    output = {
        'd_values': d_values,
        'seed': np.array(seed),
    }
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig8(num_realizations=NUM_REALIZATIONS, save_path=None,
                        seed=SEED):
    """
    Fig. 8: Component-level rate vs. AP-user horizontal distance (N=40).

    Compares component-level PSO, APSO, GWO with phase-level AO baseline.

    Returns
    -------
    dict with 'd_values' and per-scheme average rates.
    """
    N = N_DEFAULT
    d_values = np.arange(480, 501, 2)
    schemes = [
        'upper_bound', 'ao_practical_prop1',
        'pso_component', 'apso_component', 'gwo_component',
        'lower_bound',
    ]

    master_rng = np.random.default_rng(seed + 3)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 8: Component-Level Rate vs. Distance (N={N})",
        fixed_N=N
    )

    output = {'d_values': d_values, 'seed': np.array(seed)}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig9(num_realizations=NUM_REALIZATIONS, save_path=None,
                        seed=SEED):
    """
    Fig. 9: Component-level rate vs. number of reflecting elements (d=498m).

    Returns
    -------
    dict with 'N_values' and per-scheme average rates.
    """
    d_horizontal = 498
    N_values = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    schemes = [
        'upper_bound', 'ao_practical_prop1',
        'pso_component', 'apso_component', 'gwo_component',
        'lower_bound',
    ]

    master_rng = np.random.default_rng(seed + 4)

    results = _run_parallel(
        N_values, 'N', schemes, num_realizations, master_rng,
        f"Fig. 9: Component-Level Rate vs. N (d={d_horizontal}m)",
        fixed_d=d_horizontal
    )

    output = {'N_values': N_values, 'seed': np.array(seed)}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig10(num_realizations=20, save_path=None, seed=SEED):
    """
    Fig. 10: Convergence comparison of component-level algorithms.

    Shows average R_SE vs. iteration for PSO, APSO, GWO component-level
    at d=498m, N=40, averaged over num_realizations channel realizations.

    Returns
    -------
    dict with 'iterations' and per-scheme average convergence curves.
    """
    N = N_DEFAULT
    d_horizontal = 498
    schemes = ['pso_component', 'apso_component', 'gwo_component']

    master_rng = np.random.default_rng(seed + 5)

    tasks = []
    for r in range(num_realizations):
        task_seed = int(master_rng.integers(0, 2**31))
        tasks.append((N, d_horizontal, task_seed))

    all_histories = {s: [] for s in schemes}
    start_time = time.time()
    n_workers = min(_N_WORKERS, num_realizations)

    print(f"\n{'='*60}")
    print(f"  Fig. 10: Convergence (N={N}, d={d_horizontal}m, "
          f"{num_realizations} realizations, {n_workers} workers)")
    print(f"{'='*60}")

    if n_workers > 1 and num_realizations > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {}
            for i, task in enumerate(tasks):
                future = executor.submit(_convergence_worker, task)
                future_to_idx[future] = i

            completed = 0
            for future in as_completed(future_to_idx):
                histories = future.result()
                for s in schemes:
                    all_histories[s].append(histories[s])
                completed += 1
                if completed % max(1, num_realizations // 5) == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {completed}/{num_realizations} "
                          f"| elapsed: {elapsed:.0f}s")
    else:
        for i, task in enumerate(tasks):
            histories = _convergence_worker(task)
            for s in schemes:
                all_histories[s].append(histories[s])

    wall_seconds = time.time() - start_time
    print(f"  Completed in {wall_seconds:.1f}s\n")

    # Average convergence over realizations
    output = {
        'iterations': np.arange(len(all_histories[schemes[0]][0])),
        'seed': np.array(seed),
    }
    for s in schemes:
        output[s] = np.mean(all_histories[s], axis=0)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output


def run_simulation_fig11(num_realizations=NUM_REALIZATIONS, save_path=None,
                         seed=SEED):
    """
    Fig. 11: Phase-level + component-level + hybrid comparison
    across AP-user horizontal distance.

    Focused comparison showing key schemes:
        - Upper bound (ideal IRS)
        - AO practical (Prop. 1)
        - Best phase-level metaheuristic (PSO or APSO)
        - Best component-level (PSO or APSO) + GWO component
        - Hybrid (AO → warm-started component PSO)
        - Lower bound (no IRS)

    Uses wider distance range d ∈ [400, 500].

    Returns
    -------
    dict with 'd_values' and per-scheme average rates.
    """
    N = N_DEFAULT
    d_values = np.arange(480, 501, 2)
    schemes = [
        'upper_bound',
        'pso_component', 'gwo_component',
        'hybrid_component',
        'hybrid_pso_pso_component',
        'hybrid_pso_gwo_component',
        'lower_bound',
    ]

    master_rng = np.random.default_rng(seed + 6)

    results = _run_parallel(
        d_values, 'd', schemes, num_realizations, master_rng,
        f"Fig. 11: Phase + Component + Hybrid (N={N})",
        fixed_N=N
    )

    output = {'d_values': d_values, 'seed': np.array(seed)}
    output.update(results)

    if save_path:
        np.savez(save_path, **output)
        print(f"  Results saved to {save_path}")

    return output
